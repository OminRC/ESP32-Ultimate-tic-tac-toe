"""Multi-process self-play + GPU training for Kaggle.

Why: the model is tiny, so a single batched NN forward call finishes almost
instantly -- the real bottleneck is the pure-Python MCTS tree traversal,
which is single-threaded (GIL) and only ever pins one CPU core. This script
fixes that by running self-play across `--workers` separate OS processes
(each pinning its own core, each doing CPU-only inference locally, no IPC
per NN call), while the main process does the actual training step on GPU
and re-saves weights each iteration for workers to pick up next round.

Usage:
    python train_kaggle_mp.py --iterations 150 --workers 4 \
        --parallel-per-worker 64 --sims 150 --out /kaggle/working/policy_params.npz
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from game import UltimateTTT
import net_torch
from train_kaggle import self_play_batch, train_step


def _cpu_batch_fn(model):
    model.eval()

    @torch.no_grad()
    def fn_batch(states):
        x = torch.from_numpy(states.astype(np.float32))
        policy, value = model(x)
        return policy.numpy(), value.numpy()

    return fn_batch


def self_play_worker(task):
    """Runs entirely on CPU in a worker process: loads current weights from
    `ckpt_path`, plays `n_games` self-play games in lockstep, returns the
    training data. No CUDA/GPU touched here on purpose (see module docstring).
    """
    ckpt_path, n_games, sims, temp_moves, seed = task
    torch.set_num_threads(1)  # avoid oversubscription: each worker owns 1 core

    model = net_torch.PolicyValueNet()
    net_torch.load_npz_into(model, ckpt_path)
    fn_batch = _cpu_batch_fn(model)

    rng = np.random.default_rng(seed)
    return self_play_batch(fn_batch, n_games, sims, temp_moves, rng)


def _greedy_move(model, game):
    """No-search move: single forward pass, argmax over legal actions.
    Used only for the cheap periodic eval-vs-random check, not self-play."""
    device = next(model.parameters()).device
    x = torch.from_numpy(game.encode().astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        policy, _ = model(x)
    policy = policy.squeeze(0).cpu().numpy() * game.legal_action_mask()
    a = int(np.argmax(policy))
    return divmod(a, 9)


def _random_move(game, rng):
    moves = game.legal_moves()
    return moves[rng.integers(len(moves))]


def quick_eval_vs_random(model, n_games, rng):
    """Plays `model` (greedy) against random moves, alternating sides.
    Returns a score in [0, 1]: 1.0 = always wins, 0.5 = as good as random."""
    model.eval()
    wins = draws = losses = 0
    for i in range(n_games):
        trained_is_x = (i % 2 == 0)
        game = UltimateTTT()
        while not game.done:
            if (game.player == 1) == trained_is_x:
                sub, cell = _greedy_move(model, game)
            else:
                sub, cell = _random_move(game, rng)
            game.play(sub, cell)
        if game.winner == 2:
            draws += 1
        elif (game.winner == 1) == trained_is_x:
            wins += 1
        else:
            losses += 1
    return (wins + 0.5 * draws) / n_games, wins, losses, draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=150)
    ap.add_argument("--workers", type=int, default=None,
                     help="default: os.cpu_count()")
    ap.add_argument("--parallel-per-worker", type=int, default=64,
                     help="self-play games each worker runs simultaneously")
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--temp-moves", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-replay", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="policy_params.npz")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--eval-every", type=int, default=10,
                     help="run a quick vs-random eval every N iterations (0 disables)")
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5,
                     help="stop if `eval-every`-spaced checks show no improvement "
                          "this many times in a row (0 disables early stopping)")
    ap.add_argument("--min-delta", type=float, default=0.01,
                     help="minimum score increase to count as an improvement")
    args = ap.parse_args()

    workers = args.workers or os.cpu_count() or 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"train device: {device}   self-play workers: {workers} (CPU, 1 core each)")
    print(f"games/iteration = {workers} workers x {args.parallel_per_worker} = "
          f"{workers * args.parallel_per_worker}")

    torch.manual_seed(args.seed)

    model = net_torch.PolicyValueNet().to(device)
    if args.resume:
        net_torch.load_npz_into(model, args.resume)
        print(f"resumed from {args.resume}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # workers need a checkpoint file to load; write the initial (possibly
    # random) weights before the first round.
    net_torch.save_npz(model, args.out)

    replay_states, replay_policies, replay_values = [], [], []
    best_score = -1.0
    patience_counter = 0
    best_path = args.out.rsplit(".", 1)[0] + "_best.npz"
    eval_rng = np.random.default_rng(args.seed + 999)

    # IMPORTANT: force 'spawn'. The main process has already initialized a
    # CUDA context (model.to(device) above); on Linux the default start
    # method is 'fork', and forking a process that already holds a CUDA
    # context is unsafe -- workers can hang forever (looks exactly like
    # "stuck, no progress"). 'spawn' starts each worker as a fresh
    # interpreter, which is slower to launch but actually works.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for it in range(1, args.iterations + 1):
            t0 = time.time()
            tasks = [
                (args.out, args.parallel_per_worker, args.sims, args.temp_moves,
                 args.seed * 100000 + it * 1000 + w)
                for w in range(workers)
            ]
            results = pool.map(self_play_worker, tasks)

            wins = {1: 0, -1: 0, 2: 0}
            for worker_data in results:
                for states, policies, values, winner in worker_data:
                    replay_states.extend(states)
                    replay_policies.extend(policies)
                    replay_values.extend(values)
                    wins[winner] += 1

            replay_states = replay_states[-args.max_replay:]
            replay_policies = replay_policies[-args.max_replay:]
            replay_values = replay_values[-args.max_replay:]

            X = np.stack(replay_states).astype(np.float32)
            P = np.stack(replay_policies).astype(np.float32)
            Z = np.array(replay_values, dtype=np.float32)

            avg_loss = train_step(model, optimizer, device, X, P, Z,
                                   args.batch_size, args.epochs)

            net_torch.save_npz(model, args.out)

            dt = time.time() - t0
            n_games = workers * args.parallel_per_worker
            print(f"iter {it:4d}/{args.iterations}  games={n_games}  "
                  f"replay={X.shape[0]:6d}  loss={avg_loss[0]:.4f} "
                  f"(pi={avg_loss[1]:.4f} v={avg_loss[2]:.4f})  "
                  f"wins X={wins[1]} O={wins[-1]} draw={wins[2]}  [{dt:.1f}s]")

            if args.eval_every > 0 and it % args.eval_every == 0:
                score, w, l, d = quick_eval_vs_random(model, args.eval_games, eval_rng)
                print(f"  eval vs random: score={score:.3f}  "
                      f"(win={w} loss={l} draw={d} / {args.eval_games})")

                if score > best_score + args.min_delta:
                    best_score = score
                    patience_counter = 0
                    net_torch.save_npz(model, best_path)
                    print(f"  new best ({score:.3f}) -> saved {best_path}")
                else:
                    patience_counter += 1
                    print(f"  no improvement over best ({best_score:.3f}): "
                          f"{patience_counter}/{args.patience}")
                    if args.patience > 0 and patience_counter >= args.patience:
                        print(f"early stopping at iteration {it}: "
                              f"no improvement for {args.patience} evals in a row")
                        break

    print(f"saved trained params -> {args.out}")
    if args.eval_every > 0:
        print(f"best checkpoint (score={best_score:.3f}) -> {best_path}")


if __name__ == "__main__":
    main()

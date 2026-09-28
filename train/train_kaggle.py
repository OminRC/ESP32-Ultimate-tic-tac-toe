"""GPU-accelerated self-play + training for Kaggle notebooks.

Runs many self-play games in lockstep (`--parallel`) so every MCTS leaf
evaluation is batched into a single NN forward call -- this is what actually
uses the GPU; running one game at a time (like train.py) leaves it idle.

Usage (inside a Kaggle notebook cell, GPU accelerator enabled):
    !python train_kaggle.py --iterations 100 --parallel 128 --sims 150 \
        --out /kaggle/working/policy_params.npz

Checkpoints (.npz, same format as train.py) are saved after every iteration,
so you can stop the notebook / hit the free-GPU quota and still keep the
latest weights. Download policy_params.npz and run export_weights.py
locally (or in the same notebook) to produce the C header.
"""
import argparse
import time
import numpy as np
import torch

from game import UltimateTTT
from mcts import run_mcts_batch, action_probs
import net_torch


def self_play_batch(fn_batch, n_parallel, sims, temp_moves, rng):
    games = [UltimateTTT() for _ in range(n_parallel)]
    trajectories = [[] for _ in range(n_parallel)]
    move_no = [0] * n_parallel
    finished = []  # (states, policies, values, winner)
    active = list(range(n_parallel))

    while active:
        active_games = [games[i] for i in active]
        roots = run_mcts_batch(active_games, fn_batch, n_simulations=sims,
                                dirichlet_eps=0.25)
        next_active = []
        for pos, i in enumerate(active):
            root = roots[pos]
            temperature = 1.0 if move_no[i] < temp_moves else 0.0
            pi = action_probs(root, temperature=temperature)
            trajectories[i].append((games[i].encode(), pi, games[i].player))

            if temperature > 0:
                action = int(rng.choice(81, p=pi))
            else:
                action = int(np.argmax(pi))
            sub, cell = divmod(action, 9)
            games[i].play(sub, cell)
            move_no[i] += 1

            if games[i].done:
                states = [s for s, _, _ in trajectories[i]]
                policies = [p for _, p, _ in trajectories[i]]
                players = [pl for _, _, pl in trajectories[i]]
                z = [games[i].result_for(pl) for pl in players]
                finished.append((states, policies, z, games[i].winner))
            else:
                next_active.append(i)
        active = next_active

    return finished


def train_step(model, optimizer, device, X, P, Z, batch_size, epochs, l2=1e-4):
    n = X.shape[0]
    rng = np.random.default_rng()
    losses = []
    Xt_full = torch.from_numpy(X).to(device)
    Pt_full = torch.from_numpy(P).to(device)
    Zt_full = torch.from_numpy(Z).to(device)

    model.train()
    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            bidx = idx[start:start + batch_size]
            b = torch.from_numpy(bidx).to(device)
            xb = Xt_full[b]
            pb = Pt_full[b]
            zb = Zt_full[b]

            policy, value = model(xb)
            eps = 1e-8
            policy_loss = -(pb * torch.log(policy + eps)).sum(dim=1).mean()
            value_loss = torch.mean((zb - value) ** 2)
            l2_loss = sum((p ** 2).sum() for p in model.parameters()) * l2
            loss = policy_loss + value_loss + l2_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append((loss.item(), policy_loss.item(), value_loss.item()))

    return np.mean(losses, axis=0) if losses else (0.0, 0.0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--parallel", type=int, default=128,
                     help="self-play games run simultaneously per iteration")
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--temp-moves", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-replay", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="policy_params.npz")
    ap.add_argument("--resume", type=str, default=None,
                     help="path to an existing .npz checkpoint to continue from")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = net_torch.PolicyValueNet().to(device)
    if args.resume:
        net_torch.load_npz_into(model, args.resume)
        print(f"resumed from {args.resume}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    replay_states, replay_policies, replay_values = [], [], []

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        fn_batch = net_torch.make_batch_fn(model, device)

        data = self_play_batch(fn_batch, args.parallel, args.sims,
                                args.temp_moves, rng)
        wins = {1: 0, -1: 0, 2: 0}
        for states, policies, values, winner in data:
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

        dt = time.time() - t0
        print(f"iter {it:4d}/{args.iterations}  games={args.parallel}  "
              f"replay={X.shape[0]:6d}  loss={avg_loss[0]:.4f} "
              f"(pi={avg_loss[1]:.4f} v={avg_loss[2]:.4f})  "
              f"wins X={wins[1]} O={wins[-1]} draw={wins[2]}  [{dt:.1f}s]")

        net_torch.save_npz(model, args.out)

    print(f"saved trained params -> {args.out}")


if __name__ == "__main__":
    main()

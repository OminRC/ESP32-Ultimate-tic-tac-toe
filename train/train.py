"""Self-play + training loop (AlphaZero-lite) for Ultimate Tic-Tac-Toe.

Usage:
    python train.py --iterations 30 --games-per-iter 40 --sims 60
"""
import argparse
import time
import numpy as np

from game import UltimateTTT
from mcts import run_mcts, action_probs
import net


def make_policy_value_fn(params):
    def fn(game):
        x = game.encode()
        policy, value = net.predict_single(params, x)
        return policy, value
    return fn


def self_play_game(params, n_sims, temp_moves=10):
    game = UltimateTTT()
    states, policies, players = [], [], []

    fn = make_policy_value_fn(params)
    move_no = 0
    while not game.done:
        root = run_mcts(game, fn, n_simulations=n_sims,
                         dirichlet_eps=0.25)
        temperature = 1.0 if move_no < temp_moves else 0.0
        pi = action_probs(root, temperature=temperature)

        states.append(game.encode())
        policies.append(pi)
        players.append(game.player)

        action = int(np.random.choice(81, p=pi)) if temperature > 0 else int(np.argmax(pi))
        sub, cell = divmod(action, 9)
        game.play(sub, cell)
        move_no += 1

    z = [game.result_for(p) for p in players]
    return states, policies, z, game.winner


def train(iterations, games_per_iter, sims, batch_size=128, epochs=1, lr=1e-3,
          seed=0, out_path="policy_params.npz"):
    rng = np.random.default_rng(seed)
    params = net.init_params(seed=seed)
    opt = net.Adam(params, lr=lr)

    replay_states, replay_policies, replay_values = [], [], []
    max_replay = 20000

    for it in range(1, iterations + 1):
        t0 = time.time()
        wins = {1: 0, -1: 0, 2: 0}
        for g in range(games_per_iter):
            states, policies, values, winner = self_play_game(params, sims)
            replay_states.extend(states)
            replay_policies.extend(policies)
            replay_values.extend(values)
            wins[winner] += 1

        replay_states = replay_states[-max_replay:]
        replay_policies = replay_policies[-max_replay:]
        replay_values = replay_values[-max_replay:]

        X = np.stack(replay_states)
        P = np.stack(replay_policies)
        Z = np.array(replay_values, dtype=np.float32)

        n = X.shape[0]
        losses = []
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                bidx = idx[start:start + batch_size]
                loss, ploss, vloss, grads = net.loss_and_grads(params, X[bidx], P[bidx], Z[bidx])
                opt.step(params, grads)
                losses.append((loss, ploss, vloss))

        avg = np.mean(losses, axis=0) if losses else (0, 0, 0)
        dt = time.time() - t0
        print(f"iter {it:3d}/{iterations}  games={games_per_iter}  "
              f"replay={n:5d}  loss={avg[0]:.4f} (pi={avg[1]:.4f} v={avg[2]:.4f})  "
              f"wins X={wins[1]} O={wins[-1]} draw={wins[2]}  [{dt:.1f}s]")

        net.save_params(params, out_path)

    print(f"saved trained params -> {out_path}")
    return params


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--games-per-iter", type=int, default=40)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="policy_params.npz")
    args = ap.parse_args()

    train(args.iterations, args.games_per_iter, args.sims,
          batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
          seed=args.seed, out_path=args.out)

"""Sanity check: trained policy (greedy, no search) vs a random-move player.

If training worked, the trained side should win the vast majority of games.
"""
import argparse
import numpy as np

from game import UltimateTTT
import net


def greedy_move(params, game):
    x = game.encode()
    policy, _ = net.predict_single(params, x)
    mask = game.legal_action_mask()
    policy = policy * mask
    a = int(np.argmax(policy))
    return divmod(a, 9)


def random_move(game, rng):
    moves = game.legal_moves()
    return moves[rng.integers(len(moves))]


def play_one(params, trained_is_x, rng):
    game = UltimateTTT()
    while not game.done:
        if (game.player == 1) == trained_is_x:
            sub, cell = greedy_move(params, game)
        else:
            sub, cell = random_move(game, rng)
        game.play(sub, cell)
    return game.winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("--games", type=int, default=100)
    args = ap.parse_args()

    params = net.load_params(args.npz_path)
    rng = np.random.default_rng(42)

    trained_wins = 0
    random_wins = 0
    draws = 0
    for i in range(args.games):
        trained_is_x = (i % 2 == 0)
        winner = play_one(params, trained_is_x, rng)
        if winner == 2:
            draws += 1
        elif (winner == 1) == trained_is_x:
            trained_wins += 1
        else:
            random_wins += 1

    print(f"trained: {trained_wins}  random: {random_wins}  draws: {draws}  "
          f"(out of {args.games} games)")


if __name__ == "__main__":
    main()

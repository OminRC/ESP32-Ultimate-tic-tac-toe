import numpy as np

from game import UltimateTTT
import mcts
import net


def uniform_policy_value_fn(game):
    policy = np.full(81, 1.0 / 81.0, dtype=np.float32)
    return policy, 0.0


def test_run_mcts_expands_only_legal_moves():
    g = UltimateTTT()
    root = mcts.run_mcts(g, uniform_policy_value_fn, n_simulations=20, dirichlet_eps=0)
    assert set(root.children.keys()) == {s * 9 + c for s, c in g.legal_moves()}


def test_action_probs_sum_to_one_and_match_legal_mask():
    g = UltimateTTT()
    root = mcts.run_mcts(g, uniform_policy_value_fn, n_simulations=30, dirichlet_eps=0)
    probs = mcts.action_probs(root, temperature=1.0)
    assert probs.shape == (81,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)

    mask = g.legal_action_mask()
    # probability mass must only sit on legal actions
    assert np.all(probs[mask == 0] == 0)


def test_action_probs_temperature_zero_is_argmax_of_visits():
    g = UltimateTTT()
    root = mcts.run_mcts(g, uniform_policy_value_fn, n_simulations=30, dirichlet_eps=0)
    probs = mcts.action_probs(root, temperature=0.0)
    assert probs.sum() == 1.0
    assert (probs == 1.0).sum() == 1  # one-hot


def test_run_mcts_with_trained_net_does_not_crash_near_terminal_state():
    params = net.init_params(seed=0)

    def fn(game):
        return net.predict_single(params, game.encode())

    g = UltimateTTT()
    # play down to a near-terminal state (white-box) to exercise the
    # "leaf is already done" branch in run_mcts/backpropagate
    g.sub_result[:] = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    g.board[2, 0] = 1
    g.board[2, 1] = 1
    root = mcts.run_mcts(g, fn, n_simulations=10, dirichlet_eps=0)
    probs = mcts.action_probs(root)
    assert probs.shape == (81,)

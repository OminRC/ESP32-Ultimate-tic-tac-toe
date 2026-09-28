import random

import numpy as np
import pytest

from game import UltimateTTT, check_winner


def test_initial_state_has_81_legal_moves():
    g = UltimateTTT()
    assert len(g.legal_moves()) == 81
    assert len(g.legal_subboards()) == 9


def test_encode_shape_and_mask_consistency():
    g = UltimateTTT()
    state = g.encode()
    assert state.shape == (126,)

    mask = g.legal_action_mask()
    assert mask.shape == (81,)
    assert mask.sum() == len(g.legal_moves())


@pytest.mark.parametrize("cells,expected", [
    ([1, 1, 1, 0, 0, 0, 0, 0, 0], 1),
    ([-1, -1, -1, 0, 0, 0, 0, 0, 0], -1),
    ([1, 0, 0, 1, 0, 0, 1, 0, 0], 1),   # left column
    ([1, 0, 0, 0, 1, 0, 0, 0, 1], 1),   # diagonal
    ([1, -1, 1, -1, 1, -1, -1, 1, -1], 0),  # full board, no winner
    ([0] * 9, 0),
])
def test_check_winner(cells, expected):
    assert check_winner(np.array(cells)) == expected


def test_winning_a_subboard_sets_sub_result():
    g = UltimateTTT()
    # White-box: pre-place two X marks directly, then complete the row via
    # the public API to check win detection fires and updates sub_result.
    g.board[0, 0] = 1
    g.board[0, 1] = 1
    done = g.play(0, 2)  # X completes top row of sub-board 0
    assert g.sub_result[0] == 1
    assert not done  # winning one sub-board doesn't end the whole game


def test_winning_the_big_board_ends_the_game():
    g = UltimateTTT()
    g.sub_result[0] = 1
    g.sub_result[1] = 1
    g.board[2, 0] = 1
    g.board[2, 1] = 1
    done = g.play(2, 2)  # X completes the top row of the big board
    assert done
    assert g.winner == 1


def test_active_sub_forces_correct_board():
    g = UltimateTTT()
    g.play(4, 4)  # played cell index 4 -> opponent forced into sub-board 4
    assert g.legal_subboards() == [4]
    for sub, cell in g.legal_moves():
        assert sub == 4


def test_active_sub_frees_up_when_target_subboard_decided():
    g = UltimateTTT()
    # Force sub-board 0 to be already decided (drawn), then verify playing a
    # move that points to sub 0 opens up all boards instead.
    # Manufacture a draw in sub 0 directly (white-box, mirrors the C test).
    fill = [1, -1, 1, -1, 1, 1, -1, 1, -1]
    for i in range(9):
        g.board[0, i] = fill[i]
    g.sub_result[0] = 2  # drawn

    g.play(1, 0)  # play cell 0 -> would normally force sub 0, but it's decided
    assert g.legal_subboards() == [s for s in range(9) if g.sub_result[s] == 0]


def test_full_random_game_terminates_with_valid_result():
    rng = random.Random(1234)
    for trial in range(20):
        g = UltimateTTT()
        moves = 0
        while not g.done and moves < 1000:
            sub, cell = rng.choice(g.legal_moves())
            g.play(sub, cell)
            moves += 1
        assert g.done, "game did not terminate within move cap"
        assert g.winner in (1, -1, 2)

        for p in (1, -1):
            r = g.result_for(p)
            if g.winner == 2:
                assert r == 0.0
            elif g.winner == p:
                assert r == 1.0
            else:
                assert r == -1.0


def test_clone_is_independent():
    g = UltimateTTT()
    g.play(0, 0)
    g2 = g.clone()
    g2.play(0, 1)
    assert g.board[0, 1] == 0
    assert g2.board[0, 1] != 0

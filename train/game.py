"""Ultimate Tic-Tac-Toe game engine.

Board layout: 9 sub-boards of 3x3, indexed 0-8 (row-major, big board).
Each sub-board has 9 cells, indexed 0-8 (row-major).
Cell value: 0 = empty, 1 = player 1 (X), -1 = player 2 (O).

State representation for the NN:
  - 81 cells (current player's marks = +1, opponent's marks = -1, empty = 0)
  - 9 sub-board results (0 = undecided, +1 = current player won it,
    -1 = opponent won it, 2 = drawn/full) -> encoded as one-hot(4) each = 36
  - 9 legal-sub-board flags (which big cell you're allowed to play in) = 9
Total input length = 81 + 36 + 9 = 126
"""
import numpy as np

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def check_winner(cells9):
    """cells9: length-9 array of -1/0/1. Returns 1, -1, 0 (no winner)."""
    for a, b, c in WIN_LINES:
        s = cells9[a] + cells9[b] + cells9[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


class UltimateTTT:
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=np.int8)
        self.sub_result = np.zeros(9, dtype=np.int8)  # 0 undecided, 1/-1 winner, 2 drawn
        self.player = 1
        self.active_sub = -1  # -1 means "any non-finished sub-board"
        self.winner = 0  # 0 ongoing, 1/-1 winner, 2 draw
        self.done = False

    def clone(self):
        g = UltimateTTT.__new__(UltimateTTT)
        g.board = self.board.copy()
        g.sub_result = self.sub_result.copy()
        g.player = self.player
        g.active_sub = self.active_sub
        g.winner = self.winner
        g.done = self.done
        return g

    def legal_subboards(self):
        if self.active_sub != -1 and self.sub_result[self.active_sub] == 0:
            return [self.active_sub]
        return [s for s in range(9) if self.sub_result[s] == 0]

    def legal_moves(self):
        """Returns list of (sub, cell) legal moves."""
        moves = []
        for s in self.legal_subboards():
            for c in range(9):
                if self.board[s, c] == 0:
                    moves.append((s, c))
        return moves

    def play(self, sub, cell):
        assert not self.done
        assert self.sub_result[sub] == 0
        assert self.board[sub, cell] == 0
        subs_allowed = self.legal_subboards()
        assert sub in subs_allowed

        self.board[sub, cell] = self.player

        w = check_winner(self.board[sub])
        if w != 0:
            self.sub_result[sub] = w
        elif np.all(self.board[sub] != 0):
            self.sub_result[sub] = 2  # drawn sub-board

        big_w = check_winner(self.sub_result.astype(np.int8))
        # treat drawn sub-boards (2) as neutral for big-board win check
        big_cells = np.where(self.sub_result == 2, 0, self.sub_result)
        big_w = check_winner(big_cells)
        if big_w != 0:
            self.winner = big_w
            self.done = True
        elif np.all(self.sub_result != 0):
            self.winner = 2
            self.done = True
        else:
            # next active sub-board = `cell` index, unless that board is finished
            self.active_sub = cell if self.sub_result[cell] == 0 else -1

        self.player *= -1
        return self.done

    def encode(self):
        """Encode state from perspective of the player about to move."""
        p = self.player
        cells = (self.board.flatten() * p).astype(np.float32)  # 81

        sub_onehot = np.zeros((9, 4), dtype=np.float32)
        for s in range(9):
            r = self.sub_result[s]
            if r == 0:
                sub_onehot[s, 0] = 1
            elif r == p:
                sub_onehot[s, 1] = 1
            elif r == -p:
                sub_onehot[s, 2] = 1
            else:
                sub_onehot[s, 3] = 1
        sub_onehot = sub_onehot.flatten()  # 36

        legal_sub_flag = np.zeros(9, dtype=np.float32)
        for s in self.legal_subboards():
            legal_sub_flag[s] = 1

        return np.concatenate([cells, sub_onehot, legal_sub_flag])  # 126

    def legal_action_mask(self):
        """81-length 0/1 mask over (sub*9+cell) actions."""
        mask = np.zeros(81, dtype=np.float32)
        for s, c in self.legal_moves():
            mask[s * 9 + c] = 1
        return mask

    def result_for(self, player):
        """Terminal value from `player`'s perspective: 1 win, -1 loss, 0 draw."""
        if self.winner == 2:
            return 0.0
        return 1.0 if self.winner == player else -1.0

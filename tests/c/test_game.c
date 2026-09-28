// Native (host-compiled, no ESP-IDF needed) unit tests for main/game.c.
// Build+run via tests/run_c_tests.sh, or manually:
//   gcc -I../../main -o test_game test_game.c ../../main/game.c && ./test_game
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "game.h"

static void test_initial_state(void) {
    Game g;
    game_init(&g);

    int subs[9], n;
    game_legal_subboards(&g, subs, &n);
    assert(n == 9);

    float mask[81];
    game_legal_mask81(&g, mask);
    int count = 0;
    for (int i = 0; i < 81; i++) if (mask[i] > 0) count++;
    assert(count == 81);

    assert(g.player == 1);
    assert(!g.done);
    printf("  ok: test_initial_state\n");
}

static void test_encode_shape_and_perspective(void) {
    Game g;
    game_init(&g);
    g.board[0][0] = 1; // X mark

    float state[126];
    game_encode(&g, state); // encoded from X's perspective (X is about to move)
    assert(state[0] == 1.0f); // X's own mark shows as +1

    g.player = -1; // now encode from O's perspective without changing the board
    game_encode(&g, state);
    assert(state[0] == -1.0f); // same physical mark now shows as -1 (opponent's)
    printf("  ok: test_encode_shape_and_perspective\n");
}

static void test_winning_a_subboard(void) {
    Game g;
    game_init(&g);
    g.board[0][0] = 1;
    g.board[0][1] = 1;
    bool done = game_play(&g, 0, 2); // X completes top row of sub-board 0
    assert(g.sub_result[0] == 1);
    assert(!done);
    printf("  ok: test_winning_a_subboard\n");
}

static void test_winning_the_big_board(void) {
    Game g;
    game_init(&g);
    g.sub_result[0] = 1;
    g.sub_result[1] = 1;
    g.board[2][0] = 1;
    g.board[2][1] = 1;
    bool done = game_play(&g, 2, 2); // X completes top row of the big board
    assert(done);
    assert(g.winner == 1);
    printf("  ok: test_winning_the_big_board\n");
}

static void test_drawn_subboard_counts_as_neutral(void) {
    Game g;
    game_init(&g);
    // Classic no-winner full board:
    //   X O X
    //   X O O
    //   O X X
    // game_play() writes the *current player's* mark (g.player, still X/+1
    // here since we haven't called play() yet) into the final cell, so
    // index 8 below must already be +1 to match what game_play will write.
    int8_t fill[9] = {1, -1, 1, 1, -1, -1, -1, 1, 1};
    for (int c = 0; c < 8; c++) g.board[0][c] = fill[c];
    bool done = game_play(&g, 0, 8);
    assert(g.sub_result[0] == 2); // drawn
    assert(!done);
    printf("  ok: test_drawn_subboard_counts_as_neutral\n");
}

static void test_active_sub_forces_correct_board(void) {
    Game g;
    game_init(&g);
    game_play(&g, 4, 4); // playing cell 4 sends opponent to sub-board 4
    int subs[9], n;
    game_legal_subboards(&g, subs, &n);
    assert(n == 1 && subs[0] == 4);
    for (int s = 0; s < 9; s++) {
        for (int c = 0; c < 9; c++) {
            if (g.board[s][c] == 0 && game_is_legal_move(&g, s, c)) {
                assert(s == 4);
            }
        }
    }
    printf("  ok: test_active_sub_forces_correct_board\n");
}

static void test_active_sub_frees_up_when_target_decided(void) {
    Game g;
    game_init(&g);
    int8_t fill[9] = {1, -1, 1, -1, 1, 1, -1, 1, -1};
    for (int c = 0; c < 9; c++) g.board[0][c] = fill[c];
    g.sub_result[0] = 2; // pre-mark sub-board 0 as drawn/decided

    game_play(&g, 1, 0); // played cell 0 -> would normally force sub 0, but it's finished
    int subs[9], n;
    game_legal_subboards(&g, subs, &n);
    assert(n == 8); // all boards except the decided sub 0
    for (int i = 0; i < n; i++) assert(subs[i] != 0);
    printf("  ok: test_active_sub_frees_up_when_target_decided\n");
}

static void test_result_for(void) {
    Game g;
    game_init(&g);
    g.sub_result[0] = 1;
    g.sub_result[1] = 1;
    g.board[2][0] = 1;
    g.board[2][1] = 1;
    game_play(&g, 2, 2);
    assert(g.winner == 1);
    assert(game_result_for(&g, 1) == 1.0f);
    assert(game_result_for(&g, -1) == -1.0f);
    printf("  ok: test_result_for\n");
}

static void test_full_random_game_terminates(void) {
    for (int trial = 0; trial < 20; trial++) {
        Game g;
        game_init(&g);
        int moves = 0;
        while (!g.done && moves < 1000) {
            int subs[9], n;
            game_legal_subboards(&g, subs, &n);
            int sub = subs[moves % n]; // deterministic pseudo-random pick
            int cell = -1;
            for (int c = 0; c < 9; c++) {
                if (g.board[sub][c] == 0) { cell = c; break; }
            }
            assert(cell != -1);
            game_play(&g, sub, cell);
            moves++;
        }
        assert(g.done);
        assert(g.winner == 1 || g.winner == -1 || g.winner == 2);
    }
    printf("  ok: test_full_random_game_terminates\n");
}

int main(void) {
    printf("test_game.c\n");
    test_initial_state();
    test_encode_shape_and_perspective();
    test_winning_a_subboard();
    test_winning_the_big_board();
    test_drawn_subboard_counts_as_neutral();
    test_active_sub_forces_correct_board();
    test_active_sub_frees_up_when_target_decided();
    test_result_for();
    test_full_random_game_terminates();
    printf("ALL PASSED\n");
    return 0;
}

// Native smoke test for main/net_infer.c against the real, trained
// main/model_weights.h -- catches C-side inference bugs (shape mismatches,
// bad indexing) without needing ESP32 hardware.
// Build+run via tests/run_c_tests.sh, or manually:
//   gcc -I../../main -o test_net_infer test_net_infer.c ../../main/net_infer.c ../../main/game.c -lm
//   ./test_net_infer
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "game.h"
#include "net_infer.h"

static void test_policy_is_a_valid_probability_distribution(void) {
    Game g;
    game_init(&g);

    float state[126];
    float policy[81];
    game_encode(&g, state);
    float value = net_infer(state, policy);

    float sum = 0;
    for (int i = 0; i < 81; i++) {
        assert(policy[i] >= 0.0f);
        sum += policy[i];
    }
    assert(fabsf(sum - 1.0f) < 1e-3f);
    assert(value >= -1.0f && value <= 1.0f);
    printf("  ok: test_policy_is_a_valid_probability_distribution (value=%.4f)\n", value);
}

static void test_inference_runs_on_a_near_terminal_state(void) {
    Game g;
    game_init(&g);
    g.sub_result[0] = 1;
    g.sub_result[1] = -1;
    g.board[4][0] = 1;
    g.board[4][4] = -1;

    float state[126];
    float policy[81];
    game_encode(&g, state);
    float value = net_infer(state, policy);

    float sum = 0;
    for (int i = 0; i < 81; i++) sum += policy[i];
    assert(fabsf(sum - 1.0f) < 1e-3f);
    assert(value >= -1.0f && value <= 1.0f);
    printf("  ok: test_inference_runs_on_a_near_terminal_state\n");
}

int main(void) {
    printf("test_net_infer.c\n");
    test_policy_is_a_valid_probability_distribution();
    test_inference_runs_on_a_near_terminal_state();
    printf("ALL PASSED\n");
    return 0;
}

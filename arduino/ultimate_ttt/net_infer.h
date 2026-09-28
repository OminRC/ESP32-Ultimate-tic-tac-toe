#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Runs the tiny policy+value MLP on a 126-dim encoded game state.
// out_policy: 81 raw softmax probabilities (mask + renormalize before use).
// returns value estimate in [-1, 1] from the current player's perspective.
float net_infer(const float state[126], float out_policy[81]);

#ifdef __cplusplus
}
#endif

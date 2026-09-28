#include "net_infer.h"
#include "model_weights.h"
#include <math.h>
#include <string.h>

static void dense(const float *in, int in_dim, const float *W, const float *b,
                   int out_dim, float *out) {
    for (int o = 0; o < out_dim; o++) {
        float acc = b[o];
        for (int i = 0; i < in_dim; i++) {
            acc += in[i] * W[i * out_dim + o];
        }
        out[o] = acc;
    }
}

static void relu_inplace(float *v, int n) {
    for (int i = 0; i < n; i++) if (v[i] < 0) v[i] = 0;
}

float net_infer(const float state[126], float out_policy[81]) {
    float a1[H1_DIM];
    float a2[H2_DIM];
    float logits[POLICY_DIM];
    float v_raw[1];

    dense(state, INPUT_DIM, g_W1, g_b1, H1_DIM, a1);
    relu_inplace(a1, H1_DIM);

    dense(a1, H1_DIM, g_W2, g_b2, H2_DIM, a2);
    relu_inplace(a2, H2_DIM);

    dense(a2, H2_DIM, g_Wp, g_bp, POLICY_DIM, logits);
    dense(a2, H2_DIM, g_Wv, g_bv, 1, v_raw);

    float maxl = logits[0];
    for (int i = 1; i < POLICY_DIM; i++) if (logits[i] > maxl) maxl = logits[i];
    float sum = 0;
    for (int i = 0; i < POLICY_DIM; i++) {
        out_policy[i] = expf(logits[i] - maxl);
        sum += out_policy[i];
    }
    for (int i = 0; i < POLICY_DIM; i++) out_policy[i] /= sum;

    return tanhf(v_raw[0]);
}

"""Tiny policy+value MLP implemented with plain NumPy (no TF dependency).

Architecture:
  input (126) -> Dense(96, relu) -> Dense(64, relu) -> {
      policy head: Dense(81) -> softmax (masked externally)
      value head:  Dense(1)  -> tanh
  }

Trained with manual forward/backward pass + Adam optimizer, on
(state, mcts_policy, game_result) triples from self-play.
"""
import numpy as np

INPUT_DIM = 126
H1 = 96
H2 = 64
POLICY_DIM = 81


def init_params(seed=0):
    rng = np.random.default_rng(seed)

    def layer(fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

    return {
        "W1": layer(INPUT_DIM, H1), "b1": np.zeros(H1, dtype=np.float32),
        "W2": layer(H1, H2), "b2": np.zeros(H2, dtype=np.float32),
        "Wp": layer(H2, POLICY_DIM), "bp": np.zeros(POLICY_DIM, dtype=np.float32),
        "Wv": layer(H2, 1), "bv": np.zeros(1, dtype=np.float32),
    }


def relu(x):
    return np.maximum(0, x)


def forward(params, x_batch):
    """x_batch: (N, INPUT_DIM). Returns cache dict with all intermediates."""
    z1 = x_batch @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)

    logits = a2 @ params["Wp"] + params["bp"]
    logits_shift = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shift)
    policy = exp / exp.sum(axis=1, keepdims=True)

    v_raw = a2 @ params["Wv"] + params["bv"]
    value = np.tanh(v_raw)

    return {
        "x": x_batch, "z1": z1, "a1": a1, "z2": z2, "a2": a2,
        "logits": logits, "policy": policy, "v_raw": v_raw, "value": value,
    }


def predict_single(params, x):
    cache = forward(params, x[None, :])
    return cache["policy"][0], float(cache["value"][0, 0])


def loss_and_grads(params, x_batch, pi_target, z_target, l2=1e-4):
    N = x_batch.shape[0]
    cache = forward(params, x_batch)
    policy = cache["policy"]
    value = cache["value"]

    eps = 1e-8
    policy_loss = -np.sum(pi_target * np.log(policy + eps)) / N
    value_loss = np.mean((z_target.reshape(-1, 1) - value) ** 2)
    l2_loss = l2 * sum(np.sum(params[k] ** 2) for k in ("W1", "W2", "Wp", "Wv"))
    total_loss = policy_loss + value_loss + l2_loss

    dlogits = (policy - pi_target) / N  # softmax + CE grad
    dv_raw = (2.0 / N) * (value - z_target.reshape(-1, 1)) * (1 - value ** 2)

    a2 = cache["a2"]
    grads = {}
    grads["Wp"] = a2.T @ dlogits + 2 * l2 * params["Wp"]
    grads["bp"] = dlogits.sum(axis=0)
    grads["Wv"] = a2.T @ dv_raw + 2 * l2 * params["Wv"]
    grads["bv"] = dv_raw.sum(axis=0)

    da2 = dlogits @ params["Wp"].T + dv_raw @ params["Wv"].T
    dz2 = da2 * (cache["z2"] > 0)
    grads["W2"] = cache["a1"].T @ dz2 + 2 * l2 * params["W2"]
    grads["b2"] = dz2.sum(axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    grads["W1"] = cache["x"].T @ dz1 + 2 * l2 * params["W1"]
    grads["b1"] = dz1.sum(axis=0)

    return total_loss, policy_loss, value_loss, grads


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def save_params(params, path):
    np.savez(path, **params)


def load_params(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}

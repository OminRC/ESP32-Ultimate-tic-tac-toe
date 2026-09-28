"""PyTorch version of the tiny policy+value MLP, for GPU-accelerated
self-play + training on Kaggle. Same architecture as net.py, and saves
checkpoints in the same flat-array .npz format so export_weights.py
(-> C header) works unchanged.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 126
H1 = 96
H2 = 64
POLICY_DIM = 81


class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.policy_head = nn.Linear(H2, POLICY_DIM)
        self.value_head = nn.Linear(H2, 1)

    def forward(self, x):
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        logits = self.policy_head(a2)
        policy = F.softmax(logits, dim=-1)
        value = torch.tanh(self.value_head(a2))
        return policy, value.squeeze(-1)


def make_batch_fn(model, device):
    """Returns fn_batch(states: np.ndarray[N,126]) -> (policy[N,81], value[N])
    for use by mcts.run_mcts_batch. Runs under no_grad on `device`.
    """
    model.eval()

    @torch.no_grad()
    def fn_batch(states):
        x = torch.from_numpy(states.astype(np.float32)).to(device)
        policy, value = model(x)
        return policy.cpu().numpy(), value.cpu().numpy()

    return fn_batch


def save_npz(model, path):
    sd = model.state_dict()
    params = {
        "W1": sd["fc1.weight"].detach().cpu().numpy().T.astype(np.float32),
        "b1": sd["fc1.bias"].detach().cpu().numpy().astype(np.float32),
        "W2": sd["fc2.weight"].detach().cpu().numpy().T.astype(np.float32),
        "b2": sd["fc2.bias"].detach().cpu().numpy().astype(np.float32),
        "Wp": sd["policy_head.weight"].detach().cpu().numpy().T.astype(np.float32),
        "bp": sd["policy_head.bias"].detach().cpu().numpy().astype(np.float32),
        "Wv": sd["value_head.weight"].detach().cpu().numpy().T.astype(np.float32),
        "bv": sd["value_head.bias"].detach().cpu().numpy().astype(np.float32),
    }
    np.savez(path, **params)


def load_npz_into(model, path):
    data = np.load(path)
    sd = model.state_dict()
    sd["fc1.weight"].copy_(torch.from_numpy(data["W1"].T))
    sd["fc1.bias"].copy_(torch.from_numpy(data["b1"]))
    sd["fc2.weight"].copy_(torch.from_numpy(data["W2"].T))
    sd["fc2.bias"].copy_(torch.from_numpy(data["b2"]))
    sd["policy_head.weight"].copy_(torch.from_numpy(data["Wp"].T))
    sd["policy_head.bias"].copy_(torch.from_numpy(data["bp"]))
    sd["value_head.weight"].copy_(torch.from_numpy(data["Wv"].T))
    sd["value_head.bias"].copy_(torch.from_numpy(data["bv"]))
    return model

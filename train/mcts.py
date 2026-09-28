"""Simple PUCT-style MCTS guided by a policy+value function (AlphaZero-lite).

The policy/value function is a callable: fn(state) -> (policy[81], value[float]).
Works with either a random/uniform prior (for bootstrap self-play) or a
trained NN.
"""
import math
import numpy as np


class Node:
    __slots__ = ("game", "parent", "prior", "children", "visit_count",
                 "value_sum", "expanded")

    def __init__(self, game, parent=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.children = {}  # action -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


C_PUCT = 1.5


def puct_score(parent, child, action_prior):
    q = -child.value()  # from parent's perspective (opponent moved)
    u = C_PUCT * action_prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return q + u


def run_mcts(root_game, policy_value_fn, n_simulations=100, dirichlet_eps=0.25):
    root = Node(root_game.clone())
    expand(root, policy_value_fn, add_noise=dirichlet_eps > 0, eps=dirichlet_eps)

    for _ in range(n_simulations):
        node = root
        path = [node]

        while node.expanded and not node.game.done:
            action, node = select_child(node)
            path.append(node)

        if node.game.done:
            leaf_value = node.game.result_for(node.game.player * -1)
            # result_for is from perspective of `player`; node.game.player already
            # flipped after last move, so the mover's perspective is -player
        else:
            leaf_value = expand(node, policy_value_fn, add_noise=False)

        backpropagate(path, leaf_value)

    return root


def select_child(node):
    total_prior_actions = node.children.items()
    best_score = -1e9
    best_action = None
    best_child = None
    for action, child in total_prior_actions:
        score = puct_score(node, child, child.prior)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def build_children(node, policy, value, add_noise=False, eps=0.25):
    """Shared expansion logic: given a raw policy/value for `node` (already
    computed, single or batched), mask+renormalize and create child nodes.
    """
    game = node.game
    if game.done:
        node.expanded = True
        return game.result_for(game.player * -1)

    mask = game.legal_action_mask()
    policy = policy * mask
    s = policy.sum()
    if s > 1e-8:
        policy = policy / s
    else:
        policy = mask / mask.sum()

    if add_noise:
        legal_idx = np.where(mask > 0)[0]
        noise = np.random.dirichlet([0.3] * len(legal_idx))
        for i, idx in enumerate(legal_idx):
            policy[idx] = 0.75 * policy[idx] + 0.25 * noise[i]

    for sub, cell in game.legal_moves():
        a = sub * 9 + cell
        child_game = game.clone()
        child_game.play(sub, cell)
        node.children[a] = Node(child_game, parent=node, prior=policy[a])

    node.expanded = True
    return value


def expand(node, policy_value_fn, add_noise=False, eps=0.25):
    game = node.game
    if game.done:
        node.expanded = True
        return game.result_for(game.player * -1)
    policy, value = policy_value_fn(game)
    return build_children(node, policy, value, add_noise=add_noise, eps=eps)


def backpropagate(path, leaf_value):
    # leaf_value is from the perspective of the player about to move at the leaf.
    # Alternate sign as we go up (each ancestor is the opponent's turn).
    v = leaf_value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += v
        v = -v


def run_mcts_batch(root_games, policy_value_fn_batch, n_simulations=100,
                    dirichlet_eps=0.25):
    """Runs MCTS on many independent games in lockstep, batching every NN
    evaluation across all of them (one forward call per 'wave' instead of
    one per game) -- this is what actually lets a GPU help, since a single
    tiny MLP call on a batch of 1 leaves the GPU almost idle.

    policy_value_fn_batch: fn(states: np.ndarray[N, 126]) -> (policy[N,81], value[N])
    Returns: list of root Nodes, one per input game.
    """
    roots = [Node(g.clone()) for g in root_games]

    root_states = np.stack([r.game.encode() for r in roots])
    root_policies, root_values = policy_value_fn_batch(root_states)
    for r, p, v in zip(roots, root_policies, root_values):
        build_children(r, p, v, add_noise=dirichlet_eps > 0, eps=dirichlet_eps)

    for _ in range(n_simulations):
        leaves = []  # (root_idx, path, leaf_node)
        for i, root in enumerate(roots):
            node = root
            path = [node]
            while node.expanded and not node.game.done:
                _, node = select_child(node)
                path.append(node)
            leaves.append((i, path, node))

        pending_idx = [k for k, (_, _, node) in enumerate(leaves) if not node.game.done]
        if pending_idx:
            states = np.stack([leaves[k][2].game.encode() for k in pending_idx])
            policies, values = policy_value_fn_batch(states)

        ptr = 0
        for k, (i, path, node) in enumerate(leaves):
            if node.game.done:
                leaf_value = node.game.result_for(node.game.player * -1)
            else:
                leaf_value = build_children(node, policies[ptr], values[ptr])
                ptr += 1
            backpropagate(path, leaf_value)

    return roots


def action_probs(root, temperature=1.0):
    counts = np.zeros(81, dtype=np.float32)
    for action, child in root.children.items():
        counts[action] = child.visit_count
    if temperature == 0:
        probs = np.zeros(81, dtype=np.float32)
        probs[np.argmax(counts)] = 1.0
        return probs
    counts = counts ** (1.0 / temperature)
    total = counts.sum()
    if total <= 0:
        mask = root.game.legal_action_mask()
        return mask / mask.sum()
    return counts / total

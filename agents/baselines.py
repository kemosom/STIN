import heapq
import numpy as np

class OSPFAgent:
    """Shortest-path baseline over the instantaneous orbital distance graph."""

    def __init__(self, agent_id, env):
        self.agent_id = agent_id
        self.env = env
        self.name = "shortest_path"
        self.q_table = {}

    def get_action(self, obs, epsilon=0.0):
        idx = int(self.agent_id.split("_")[1])
        target_idx = self.env._head_destination(idx) if hasattr(self.env, "_head_destination") else self.env.targets[idx]
        neighbors = self.env._get_neighbors(idx)
        best_action = 0
        min_dist = np.inf
        for i, n_idx in enumerate(neighbors):
            if n_idx == idx:
                continue
            dist = self.env.dist_matrix[n_idx][target_idx]
            if dist < min_dist:
                min_dist = dist
                best_action = i + 1
        return best_action

    def get_shared_q(self, obs):
        return np.zeros(5)

    def update(self, *args, **kwargs):
        return 0.0

class CGRAgent:
    """Contact-aware greedy routing with a conservative congestion guard."""

    def __init__(self, agent_id, env):
        self.agent_id = agent_id
        self.env = env
        self.name = "standard_cgr"
        self.q_table = {}

    def get_action(self, obs, epsilon=0.0):
        idx = int(self.agent_id.split("_")[1])
        target_idx = self.env._head_destination(idx) if hasattr(self.env, "_head_destination") else self.env.targets[idx]
        neighbors = self.env._get_neighbors(idx)
        best_action = 0
        best_cost = np.inf
        for i, n_idx in enumerate(neighbors):
            if n_idx == idx:
                continue
            dist_cost = self.env.dist_matrix[n_idx][target_idx]
            queue_cost = self.env.node_queues.get(n_idx, 0.0) * self.env.max_isl_range_km
            contact_penalty = 0.0 if self.env.dist_matrix[idx][n_idx] <= self.env.max_isl_range_km else self.env.max_isl_range_km
            cost = dist_cost + 0.35 * queue_cost + contact_penalty
            if cost < best_cost:
                best_cost = cost
                best_action = i + 1
        if best_action == 0:
            return 0
        next_hop = neighbors[best_action - 1]
        if self.env.node_queues.get(next_hop, 0.0) > 0.90:
            return 0
        return best_action

    def get_shared_q(self, obs):
        return np.zeros(5)

    def update(self, *args, **kwargs):
        return 0.0

class BackpressureAgent:
    """Queue-aware MaxWeight/backpressure baseline.

    The decision score combines local-to-neighbor queue differential with a
    distance-progress term so it remains destination-aware in sparse contacts.
    """

    def __init__(self, agent_id, env, progress_weight=0.25):
        self.agent_id = agent_id
        self.env = env
        self.name = "backpressure"
        self.q_table = {}
        self.progress_weight = progress_weight

    def get_action(self, obs, epsilon=0.0):
        idx = int(self.agent_id.split("_")[1])
        if hasattr(self.env, "queues") and not self.env.queues[idx]:
            return 0
        target_idx = self.env._head_destination(idx) if hasattr(self.env, "_head_destination") else self.env.targets[idx]
        local_q = self.env.node_queues.get(idx, 0.0)
        old_dist = self.env.dist_matrix[idx][target_idx]
        neighbors = self.env._get_neighbors(idx)
        best_action = 0
        best_score = -np.inf
        for i, n_idx in enumerate(neighbors):
            if n_idx == idx:
                continue
            neighbor_q = self.env.node_queues.get(n_idx, 0.0)
            pressure = local_q - neighbor_q
            progress = (old_dist - self.env.dist_matrix[n_idx][target_idx]) / max(old_dist, 1e-9)
            contact_ok = self.env.dist_matrix[idx][n_idx] <= self.env.max_isl_range_km
            contact_score = 0.0 if contact_ok else -1.0
            score = pressure + self.progress_weight * progress + contact_score
            if score > best_score:
                best_score = score
                best_action = i + 1
        if best_score <= -0.05:
            return 0
        return best_action

    def get_shared_q(self, obs):
        return np.zeros(5)

    def update(self, *args, **kwargs):
        return 0.0

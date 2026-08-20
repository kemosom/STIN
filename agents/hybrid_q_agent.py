"""
agents/hybrid_q_agent.py
========================
Tabular multi-agent routing agents for event-level STIN experiments.

HybridQAgent is intentionally conservative: it uses local contact geometry for
candidate ranking and uses the exchanged 1-hop Q-vector only as a compact
neighbor value summary. Neighbor values are not used in the Bellman target.
"""

import numpy as np

def _state_key(obs, decimals=1):
    return tuple(np.round(obs, decimals=decimals))

def _initial_q_values(num_actions):
    vals = np.zeros(num_actions, dtype=float)
    if num_actions > 0:
        vals[0] = -0.10  
    return vals

class HybridQAgent:
    """Contact-aware tabular Q agent with optional 1-hop Q-vector exchange."""

    def __init__(
        self,
        agent_id,
        num_actions,
        obs_dim,
        shared_q_table: dict = None,
        learning_rate=0.5,
        gamma=0.95,
        hybrid_weight=0.65,
        progress_weight=1.05,
        learned_weight=0.25,
        delay_weight=0.15,
        queue_relief_weight=1.25,
        variant='base',
        queue_guard_threshold=0.82,
        queue_guard_penalty=2.50,
        backpressure_weight=1.85,
        exchange_mode='full',
    ):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.hybrid_weight = hybrid_weight
        self.progress_weight = progress_weight
        self.learned_weight = learned_weight
        self.delay_weight = delay_weight
        self.queue_relief_weight = queue_relief_weight
        self.variant = variant
        self.queue_guard_threshold = queue_guard_threshold
        self.queue_guard_penalty = queue_guard_penalty
        self.backpressure_weight = backpressure_weight
        self.exchange_mode = exchange_mode
        self.q_table = shared_q_table if shared_q_table is not None else {}

    def _get_state_key(self, obs):
        return _state_key(obs)

    def _init_state(self, key):
        if key not in self.q_table:
            self.q_table[key] = _initial_q_values(self.num_actions)

    def _coerce_q_vector(self, value):
        if value is None:
            return np.zeros(self.num_actions, dtype=float)
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size < self.num_actions:
            padded = np.zeros(self.num_actions, dtype=float)
            padded[: arr.size] = arr
            return padded
        return arr[: self.num_actions]

    def _neighbor_summary(self, vec):
        arr = self._coerce_q_vector(vec).copy()
        if self.variant in ('normalized', 'adaptive_normalized', 'queue_guard', 'backpressure_hybrid', 'pressure_guard'):
            scale = float(np.std(arr))
            if scale > 1e-9:
                arr = (arr - float(np.mean(arr))) / scale
        downstream_forward = float(np.max(arr[1:])) if arr.size > 1 else float(np.max(arr))
        downstream_hold = float(arr[0]) if arr.size else 0.0
        downstream_advantage = downstream_forward - downstream_hold
        estimated_congestion = float(np.clip(-downstream_hold, 0.0, 1.0))
        uncertainty = float(np.std(arr)) if arr.size else 0.0
        return downstream_advantage, estimated_congestion, uncertainty

    def get_action(self, obs, epsilon=0.1, neighbor_qs=None, env=None):
        key = self._get_state_key(obs)
        self._init_state(key)

        learned = self.q_table[key].copy()
        scores = self.learned_weight * learned
        local_congestion = float(np.clip(obs[0], 0.0, 1.0))
        scores[0] -= 0.15 + 0.45 * local_congestion

        valid_forward_actions = []
        if env is not None and self.variant == 'pressure_guard':
            return self._pressure_guard_action(obs, epsilon, neighbor_qs, env, key)

        if env is not None:
            idx = self.agent_id if isinstance(self.agent_id, int) else int(str(self.agent_id).split('_')[-1])
            target = env._head_destination(idx)
            old_dist = env.dist_matrix[idx][target]
            neighbors = env._get_neighbors(idx)
            if not env.queues[idx]:
                return 0
            for action_idx in range(1, min(5, self.num_actions)):
                nb_node = neighbors[action_idx - 1]
                if nb_node == idx:
                    scores[action_idx] -= 4.0
                    continue

                link_distance = env.dist_matrix[idx][nb_node]
                link_ok = link_distance <= env.max_isl_range_km
                if link_ok:
                    valid_forward_actions.append(action_idx)
                else:
                    scores[action_idx] -= 2.5

                new_dist = env.dist_matrix[nb_node][target]
                progress = (old_dist - new_dist) / max(old_dist, 1e-9)
                progress = float(np.clip(progress, -1.0, 1.0))
                nb_queue = float(env._queue_util(nb_node))

                scores[action_idx] += self.progress_weight * progress
                scores[action_idx] -= self.delay_weight * float(link_distance / max(env.max_isl_range_km, 1e-9))
                if nb_node == target:
                    scores[action_idx] += 0.65

                if self.variant in ('queue_guard', 'backpressure_hybrid', 'pressure_guard'):
                    if nb_queue >= self.queue_guard_threshold:
                        scores[action_idx] -= self.queue_guard_penalty * nb_queue
                    if nb_queue >= 0.98:
                        scores[action_idx] -= 5.0

                if self.variant in ('backpressure_hybrid', 'pressure_guard'):
                    pressure = local_congestion - nb_queue
                    if self.variant == 'pressure_guard':

                        scores[action_idx] += 2.75 * pressure + 1.35 * progress
                        if pressure <= -0.15 and nb_node != target:
                            scores[action_idx] -= 1.2
                    else:
                        scores[action_idx] += self.backpressure_weight * pressure

                if self.exchange_mode == 'full' and neighbor_qs and self.hybrid_weight != 0.0:
                    
                    m_j_to_i = float(neighbor_qs.get(f'agent_{nb_node}', 0.0))
                    scores[action_idx] += self.hybrid_weight * m_j_to_i
                elif self.exchange_mode == 'queue':
                    
                    queue_relief = local_congestion - nb_queue
                    scores[action_idx] += self.queue_relief_weight * queue_relief
                elif self.exchange_mode == 'none':
                    
                    pass

            if self.variant == 'pressure_guard':

                scores[0] -= 0.55 * local_congestion

        if np.random.rand() < epsilon:
            if self.variant == 'pressure_guard' and valid_forward_actions:
                return int(np.random.choice(valid_forward_actions + [0]))
            return int(np.random.randint(self.num_actions))

        return int(np.argmax(scores))

    def _pressure_guard_action(self, obs, epsilon, neighbor_qs, env, key):
        idx = self.agent_id if isinstance(self.agent_id, int) else int(str(self.agent_id).split('_')[-1])
        if not env.queues[idx]:
            return 0
        target = env._head_destination(idx)
        old_dist = env.dist_matrix[idx][target]
        neighbors = env._get_neighbors(idx)
        local_q = float(env._queue_util(idx))
        learned = self.q_table[key].copy()
        scores = np.full(self.num_actions, -1e9, dtype=float)
        scores[0] = -0.20 - 0.55 * local_q + 0.02 * learned[0]
        valid = [0]

        for action_idx in range(1, min(5, self.num_actions)):
            nb_node = neighbors[action_idx - 1]
            if nb_node == idx:
                continue
            if env.dist_matrix[idx][nb_node] > env.max_isl_range_km:
                continue
            nb_q = float(env._queue_util(nb_node))
            if nb_q >= 0.995 and nb_node != target:
                continue
            new_dist = env.dist_matrix[nb_node][target]
            progress = float(np.clip((old_dist - new_dist) / max(old_dist, 1e-9), -1.0, 1.0))
            pressure = local_q - nb_q
            score = 1.15 * pressure + 1.20 * progress + 0.05 * learned[action_idx]
            if nb_node == target:
                score += 1.50
            if nb_q >= 0.94 and nb_node != target:
                score -= 1.35 * nb_q
            if pressure <= -0.20 and progress <= -0.05 and nb_node != target:
                score -= 0.55
            if neighbor_qs and self.hybrid_weight != 0.0:
                adv, est_congestion, uncertainty = self._neighbor_summary(neighbor_qs.get(f'agent_{nb_node}'))
                omega = np.clip(self.hybrid_weight * (0.15 + 0.65 * local_q + 0.10 * uncertainty), 0.01, 0.30)
                score += omega * adv + 0.45 * (local_q - est_congestion)
            if local_q >= 0.65 and progress > -0.15 and nb_q < 0.94:
                score += 0.45 * local_q
            scores[action_idx] = score
            valid.append(action_idx)

        if np.random.rand() < epsilon:
            forward = [a for a in valid if a != 0 and scores[a] >= scores[0] - 0.15]
            pool = forward if forward else valid
            return int(np.random.choice(pool))
        return int(np.argmax(scores))

    def get_shared_q(self, obs):
        """Return a scalar Q-value summary m_{j->i} = max_a Q(o_j, a) if mode='full', else 0.0"""
        if self.exchange_mode != 'full':
            return 0.0
        key = self._get_state_key(obs)
        self._init_state(key)
        vec = self.q_table[key]
        return float(np.max(vec))

    def update(self, obs, action, reward, next_obs, neighbor_qs: dict = None, **kwargs):
        key = self._get_state_key(obs)
        next_key = self._get_state_key(next_obs)
        self._init_state(key)
        self._init_state(next_key)
        target = reward + self.gamma * np.max(self.q_table[next_key])
        td_error = target - self.q_table[key][action]
        self.q_table[key][action] = np.clip(self.q_table[key][action] + self.lr * td_error, -50.0, 50.0)
        return abs(float(td_error))

class CentralizedAgent:
    """Shared-table centralized/parameter-sharing tabular baseline."""

    def __init__(self, agent_id, num_actions, obs_dim, shared_q_table: dict, learning_rate=0.5, gamma=0.95):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.q_table = shared_q_table

    def _get_state_key(self, obs):
        obs_copy = obs.copy()
        obs_copy[1] = 0.0
        obs_copy[2] = 0.0
        return _state_key(obs_copy)

    def _init_state(self, key):
        if key not in self.q_table:
            self.q_table[key] = _initial_q_values(self.num_actions)

    def get_action(self, obs, epsilon=0.1):
        key = self._get_state_key(obs)
        self._init_state(key)
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.q_table[key]))

    def _pressure_guard_action(self, obs, epsilon, neighbor_qs, env, key):
        idx = self.agent_id if isinstance(self.agent_id, int) else int(str(self.agent_id).split('_')[-1])
        if not env.queues[idx]:
            return 0
        target = env._head_destination(idx)
        old_dist = env.dist_matrix[idx][target]
        neighbors = env._get_neighbors(idx)
        local_q = float(env._queue_util(idx))
        learned = self.q_table[key].copy()
        scores = np.full(self.num_actions, -1e9, dtype=float)
        scores[0] = -0.20 - 0.55 * local_q + 0.02 * learned[0]
        valid = [0]

        for action_idx in range(1, min(5, self.num_actions)):
            nb_node = neighbors[action_idx - 1]
            if nb_node == idx:
                continue
            if env.dist_matrix[idx][nb_node] > env.max_isl_range_km:
                continue
            nb_q = float(env._queue_util(nb_node))
            if nb_q >= 0.995 and nb_node != target:
                continue
            new_dist = env.dist_matrix[nb_node][target]
            progress = float(np.clip((old_dist - new_dist) / max(old_dist, 1e-9), -1.0, 1.0))
            pressure = local_q - nb_q
            score = 1.15 * pressure + 1.20 * progress + 0.05 * learned[action_idx]
            if nb_node == target:
                score += 1.50
            if nb_q >= 0.94 and nb_node != target:
                score -= 1.35 * nb_q
            if pressure <= -0.20 and progress <= -0.05 and nb_node != target:
                score -= 0.55
            if neighbor_qs and self.hybrid_weight != 0.0:
                adv, est_congestion, uncertainty = self._neighbor_summary(neighbor_qs.get(f'agent_{nb_node}'))
                omega = np.clip(self.hybrid_weight * (0.15 + 0.65 * local_q + 0.10 * uncertainty), 0.01, 0.30)
                score += omega * adv + 0.45 * (local_q - est_congestion)
            if local_q >= 0.65 and progress > -0.15 and nb_q < 0.94:
                score += 0.45 * local_q
            scores[action_idx] = score
            valid.append(action_idx)

        if np.random.rand() < epsilon:
            forward = [a for a in valid if a != 0 and scores[a] >= scores[0] - 0.15]
            pool = forward if forward else valid
            return int(np.random.choice(pool))
        return int(np.argmax(scores))

    def get_shared_q(self, obs):
        key = self._get_state_key(obs)
        self._init_state(key)
        return self.q_table[key].copy()

    def update(self, obs, action, reward, next_obs, neighbor_qs: dict = None, **kwargs):
        key = self._get_state_key(obs)
        next_key = self._get_state_key(next_obs)
        self._init_state(key)
        self._init_state(next_key)
        target = reward + self.gamma * np.max(self.q_table[next_key])
        td_error = target - self.q_table[key][action]
        self.q_table[key][action] += self.lr * td_error
        return abs(float(td_error))

class DistributedAgent:
    """Independent tabular Q-learning baseline."""

    def __init__(self, agent_id, num_actions, obs_dim, learning_rate=0.5, gamma=0.95):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.q_table = {}

    def _get_state_key(self, obs):
        return _state_key(obs)

    def _init_state(self, key):
        if key not in self.q_table:
            self.q_table[key] = _initial_q_values(self.num_actions)

    def get_action(self, obs, epsilon=0.1):
        key = self._get_state_key(obs)
        self._init_state(key)
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.q_table[key]))

    def _pressure_guard_action(self, obs, epsilon, neighbor_qs, env, key):
        idx = self.agent_id if isinstance(self.agent_id, int) else int(str(self.agent_id).split('_')[-1])
        if not env.queues[idx]:
            return 0
        target = env._head_destination(idx)
        old_dist = env.dist_matrix[idx][target]
        neighbors = env._get_neighbors(idx)
        local_q = float(env._queue_util(idx))
        learned = self.q_table[key].copy()
        scores = np.full(self.num_actions, -1e9, dtype=float)
        scores[0] = -0.20 - 0.55 * local_q + 0.02 * learned[0]
        valid = [0]

        for action_idx in range(1, min(5, self.num_actions)):
            nb_node = neighbors[action_idx - 1]
            if nb_node == idx:
                continue
            if env.dist_matrix[idx][nb_node] > env.max_isl_range_km:
                continue
            nb_q = float(env._queue_util(nb_node))
            if nb_q >= 0.995 and nb_node != target:
                continue
            new_dist = env.dist_matrix[nb_node][target]
            progress = float(np.clip((old_dist - new_dist) / max(old_dist, 1e-9), -1.0, 1.0))
            pressure = local_q - nb_q
            score = 1.15 * pressure + 1.20 * progress + 0.05 * learned[action_idx]
            if nb_node == target:
                score += 1.50
            if nb_q >= 0.94 and nb_node != target:
                score -= 1.35 * nb_q
            if pressure <= -0.20 and progress <= -0.05 and nb_node != target:
                score -= 0.55
            if neighbor_qs and self.hybrid_weight != 0.0:
                adv, est_congestion, uncertainty = self._neighbor_summary(neighbor_qs.get(f'agent_{nb_node}'))
                omega = np.clip(self.hybrid_weight * (0.15 + 0.65 * local_q + 0.10 * uncertainty), 0.01, 0.30)
                score += omega * adv + 0.45 * (local_q - est_congestion)
            if local_q >= 0.65 and progress > -0.15 and nb_q < 0.94:
                score += 0.45 * local_q
            scores[action_idx] = score
            valid.append(action_idx)

        if np.random.rand() < epsilon:
            forward = [a for a in valid if a != 0 and scores[a] >= scores[0] - 0.15]
            pool = forward if forward else valid
            return int(np.random.choice(pool))
        return int(np.argmax(scores))

    def get_shared_q(self, obs):
        return np.zeros(self.num_actions, dtype=float)

    def update(self, obs, action, reward, next_obs, neighbor_qs: dict = None, **kwargs):
        key = self._get_state_key(obs)
        next_key = self._get_state_key(next_obs)
        self._init_state(key)
        self._init_state(next_key)
        target = reward + self.gamma * np.max(self.q_table[next_key])
        td_error = target - self.q_table[key][action]
        self.q_table[key][action] += self.lr * td_error
        return abs(float(td_error))


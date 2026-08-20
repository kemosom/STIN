import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

try:
    from skyfield.api import load
except Exception:  
    load = None

SPEED_OF_LIGHT_KM_S = 299_792.458
EARTH_RADIUS_KM = 6371.0

@dataclass
class Bundle:
    bundle_id: int
    source: int
    destination: int
    generation_step: int
    generation_time_ms: float
    enqueue_step: int
    enqueue_time_ms: float
    current_node: int
    hop_count: int = 0
    queue_wait_ms: float = 0.0
    contact_wait_ms: float = 0.0

class STINEnv(ParallelEnv):
    """Event-level STIN/DTN routing environment.

    The environment keeps the small action space used by the original paper
    (hold or forward to one of the four nearest orbital neighbors), but all
    network metrics are now measured from generated bundles. End-to-end delay,
    drops, hop count, waiting time, throughput, and delivery ratio are derived
    from bundle events rather than from QoE proxy equations.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        num_nodes: int = 25,
        max_steps: int = 50,
        mobility_speed: float = 10.0,
        traffic_load: float = 500.0,
        tle_file: str = "gp.php",
        observation_mode: str = "neighbor_queue",
        step_duration_s: float = 1.0,
        bundle_size_mbit: float = 250.0,
        queue_capacity_bundles: int = 40,
        isl_capacity_mbps: float = 2000.0,
        max_isl_range_km: float = 5000.0,
        base_isl_outage_prob: float = 0.02,
        bundle_ttl_steps: int = 80,
        seed: Optional[int] = None,
    ):
        self.num_nodes = int(num_nodes)
        self.max_steps = int(max_steps)
        self.current_step = 0
        self.mobility_speed = float(mobility_speed)
        self.traffic_load = float(traffic_load)
        self.tle_file = tle_file
        self.observation_mode = observation_mode
        self.step_duration_s = float(step_duration_s)
        self.step_duration_ms = self.step_duration_s * 1000.0
        self.bundle_size_mbit = float(bundle_size_mbit)
        self.queue_capacity_bundles = int(queue_capacity_bundles)
        self.isl_capacity_mbps = float(isl_capacity_mbps)
        self.max_isl_range_km = float(max_isl_range_km)
        self.base_isl_outage_prob = float(base_isl_outage_prob)
        self.bundle_ttl_steps = int(bundle_ttl_steps)
        self.max_neighbors = min(4, max(0, self.num_nodes - 1))
        self.rng = np.random.default_rng(seed)

        self.possible_agents = [f"agent_{i}" for i in range(self.num_nodes)]
        self.agents = self.possible_agents.copy()
        self.action_spaces = {a: spaces.Discrete(5) for a in self.possible_agents}
        self.observation_spaces = {
            a: spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
            for a in self.possible_agents
        }

        self.ts = None
        self.t_start = None
        self.sats = []
        self.synthetic_orbits = True
        self._load_orbits()

        self.sim_time_ms = 0.0
        self.t_current = self.t_start
        self.traffic_generation_enabled = True
        self.pos_dict: Dict[int, Tuple[float, float, float]] = {}
        self.dist_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=float)
        self.targets = {i: (i + self.num_nodes // 2) % self.num_nodes for i in range(self.num_nodes)}
        self.queues: Dict[int, List[Bundle]] = {i: [] for i in range(self.num_nodes)}
        self.node_queues = {i: 0.0 for i in range(self.num_nodes)}
        self.next_bundle_id = 0
        self.traffic_generation_enabled = True
        self._reset_metrics()
        self._update_orbital_state()

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def _load_orbits(self) -> None:
        if load is None:
            return
        local_candidates = [self.tle_file, "gp.php", "stations.txt"]
        for path in local_candidates:
            if path and os.path.exists(path):
                try:
                    self.ts = load.timescale()
                    self.t_start = self.ts.utc(2026, 4, 1, 0, 0, 0)
                    sats = load.tle_file(path)
                    if len(sats) >= self.num_nodes:
                        self.sats = sats[: self.num_nodes]
                        self.synthetic_orbits = False
                        return
                except Exception:
                    continue
        if self.ts is None:
            self.ts = load.timescale()
            self.t_start = self.ts.utc(2026, 4, 1, 0, 0, 0)

    def _reset_metrics(self) -> None:
        self.generated_count = 0
        self.delivered_count = 0
        self.dropped_count = 0
        self.drop_reasons = {
            "source_queue_overflow": 0,
            "next_hop_queue_overflow": 0,
            "ttl_expired": 0,
            "contact_unavailable": 0,
        }
        self.delivery_records = []
        self.drop_records = []
        self.queue_history = []
        self.step_history = []
        self.tx_attempts = 0
        self.tx_successes = 0
        self.contact_failures = 0

    def _synthetic_positions_km(self) -> np.ndarray:
        positions = np.zeros((self.num_nodes, 3), dtype=float)
        altitude = 550.0
        radius = EARTH_RADIUS_KM + altitude
        inclination = np.deg2rad(53.0)
        phase_shift = 2.0 * np.pi * (self.current_step * self.step_duration_s) / (95.0 * 60.0)
        planes = max(1, int(np.sqrt(self.num_nodes)))
        for i in range(self.num_nodes):
            plane = i % planes
            slot = i // planes
            raan = 2.0 * np.pi * plane / planes
            anomaly = 2.0 * np.pi * slot / max(1, int(np.ceil(self.num_nodes / planes))) + phase_shift
            x_orb = radius * np.cos(anomaly)
            y_orb = radius * np.sin(anomaly)
            z_orb = 0.0
            x_inc = x_orb
            y_inc = y_orb * np.cos(inclination) - z_orb * np.sin(inclination)
            z_inc = y_orb * np.sin(inclination) + z_orb * np.cos(inclination)
            x = x_inc * np.cos(raan) - y_inc * np.sin(raan)
            y = x_inc * np.sin(raan) + y_inc * np.cos(raan)
            positions[i] = [x, y, z_inc]
        return positions

    def _positions_to_lat_lon_alt(self, positions: np.ndarray) -> Dict[int, Tuple[float, float, float]]:
        out = {}
        for i, pos in enumerate(positions):
            r = float(np.linalg.norm(pos))
            lat = np.degrees(np.arcsin(np.clip(pos[2] / max(r, 1e-9), -1.0, 1.0)))
            lon = np.degrees(np.arctan2(pos[1], pos[0]))
            alt = r - EARTH_RADIUS_KM
            out[i] = (lat, lon, alt)
        return out

    def _get_position_vectors(self):
        if not self.synthetic_orbits and self.sats:
            try:
                return np.array([sat.at(self.t_current).position.km for sat in self.sats], dtype=float)
            except Exception:
                pass
        return self._synthetic_positions_km()

    def _update_orbital_state(self) -> None:
        if self.t_start is not None:
            self.t_current = self.t_start + timedelta(seconds=self.current_step * self.step_duration_s)
        positions = self._get_position_vectors()
        self.pos_dict = self._positions_to_lat_lon_alt(positions)
        diff = positions[:, None, :] - positions[None, :, :]
        self.dist_matrix = np.linalg.norm(diff, axis=2)

    def _get_neighbors(self, agent_idx, dist_matrix=None):
        if dist_matrix is None:
            dist_matrix = self.dist_matrix
        dists = dist_matrix[int(agent_idx)].copy()
        dists[int(agent_idx)] = np.inf
        closest = np.argsort(dists)[: self.max_neighbors].tolist()
        while len(closest) < 4:
            closest.append(int(agent_idx))
        return closest[:4]

    def _queue_util(self, node: int) -> float:
        return min(1.0, len(self.queues[node]) / max(1, self.queue_capacity_bundles))

    def _sync_node_queue_view(self) -> None:
        self.node_queues = {i: self._queue_util(i) for i in range(self.num_nodes)}

    def _head_destination(self, node: int) -> int:
        if self.queues[node]:
            return self.queues[node][0].destination
        return self.targets[node]

    def _get_obs(self, agent_idx):
        agent_idx = int(agent_idx)
        lat, lon, _alt = self.pos_dict[agent_idx]
        target_idx = self._head_destination(agent_idx)
        target_lat, target_lon, _ = self.pos_dict[target_idx]
        neighbors = self._get_neighbors(agent_idx, self.dist_matrix)

        obs = np.zeros(9, dtype=np.float32)
        obs[0] = self._queue_util(agent_idx)
        obs[1] = np.clip(lat / 90.0, -1.0, 1.0)
        obs[2] = np.clip(lon / 180.0, -1.0, 1.0)
        obs[3] = np.clip(target_lat / 90.0, -1.0, 1.0)
        obs[4] = np.clip(target_lon / 180.0, -1.0, 1.0)
        if self.observation_mode == "neighbor_queue":
            obs[5:9] = [self._queue_util(n) if n != agent_idx else 0.0 for n in neighbors]
        return obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.sim_time_ms = 0.0
        self.targets = {i: (i + self.num_nodes // 2) % self.num_nodes for i in range(self.num_nodes)}
        self.queues = {i: [] for i in range(self.num_nodes)}
        self.next_bundle_id = 0
        self._reset_metrics()
        self._update_orbital_state()
        self._generate_arrivals(initial=True)
        self._sync_node_queue_view()
        obs = {a: self._get_obs(int(a.split("_")[1])) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def set_traffic_generation(self, enabled: bool) -> None:
        self.traffic_generation_enabled = bool(enabled)

    def in_flight_count(self) -> int:
        return int(sum(len(q) for q in self.queues.values()))

    def _traffic_hotspot_multiplier(self, node: int) -> float:
        hotspot_count = max(1, self.num_nodes // 5)
        return 1.75 if node < hotspot_count else 0.85

    def _sample_destination(self, source: int) -> int:
        dest = int(self.rng.integers(0, self.num_nodes - 1))
        if dest >= source:
            dest += 1
        return dest

    def _record_drop(self, bundle: Bundle, reason: str) -> None:
        self.dropped_count += 1
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1
        self.drop_records.append(
            {
                "bundle_id": bundle.bundle_id,
                "source": bundle.source,
                "destination": bundle.destination,
                "node": bundle.current_node,
                "reason": reason,
                "time_ms": self.sim_time_ms,
                "age_steps": self.current_step - bundle.generation_step,
                "hop_count": bundle.hop_count,
            }
        )

    def _generate_arrivals(self, initial: bool = False) -> int:
        generated_this_step = 0
        total_mean = self.traffic_load * self.step_duration_s / max(self.bundle_size_mbit, 1e-9)
        base_mean = total_mean / max(1, self.num_nodes)
        if initial:
            base_mean = max(base_mean, 0.25)
        for node in range(self.num_nodes):
            lam = base_mean * self._traffic_hotspot_multiplier(node)
            arrivals = int(self.rng.poisson(lam))
            for _ in range(arrivals):
                bundle = Bundle(
                    bundle_id=self.next_bundle_id,
                    source=node,
                    destination=self._sample_destination(node),
                    generation_step=self.current_step,
                    generation_time_ms=self.sim_time_ms,
                    enqueue_step=self.current_step,
                    enqueue_time_ms=self.sim_time_ms,
                    current_node=node,
                )
                self.next_bundle_id += 1
                self.generated_count += 1
                generated_this_step += 1
                if len(self.queues[node]) >= self.queue_capacity_bundles:
                    self._record_drop(bundle, "source_queue_overflow")
                else:
                    self.queues[node].append(bundle)
        return generated_this_step

    def _link_available(self, src: int, dst: int) -> bool:
        if src == dst:
            return False
        if self.dist_matrix[src, dst] > self.max_isl_range_km:
            return False
        return bool(self.rng.random() >= self.base_isl_outage_prob)

    def _transmission_delay_ms(self, src: int, dst: int) -> float:
        prop_ms = (self.dist_matrix[src, dst] / SPEED_OF_LIGHT_KM_S) * 1000.0
        tx_ms = (self.bundle_size_mbit / max(self.isl_capacity_mbps, 1e-9)) * 1000.0
        return float(prop_ms + tx_ms)

    def _delivery_reward(self, delay_ms: float, hop_count: int) -> float:
        delay_penalty = min(0.5, delay_ms / 20_000.0)
        hop_penalty = min(0.2, hop_count * 0.02)
        return 1.2 - delay_penalty - hop_penalty

    def _expire_old_bundles(self) -> int:
        expired = 0
        for node in range(self.num_nodes):
            kept = []
            for bundle in self.queues[node]:
                if self.current_step - bundle.generation_step > self.bundle_ttl_steps:
                    self._record_drop(bundle, "ttl_expired")
                    expired += 1
                else:
                    kept.append(bundle)
            self.queues[node] = kept
        return expired

    def step(self, actions):
        active_agents = self.agents.copy()
        self.current_step += 1
        self._update_orbital_state()

        rewards = {a: -0.01 for a in active_agents}
        infos = {
            a: {
                "generated": 0,
                "delivered": 0,
                "dropped": 0,
                "tx_attempted": 0,
                "tx_success": 0,
                "step_latency": 0.0,
                "queue_occupancy": 0.0,
            }
            for a in active_agents
        }

        pending_enqueues: Dict[int, List[Bundle]] = {i: [] for i in range(self.num_nodes)}
        delivered_delays = []
        step_tx_attempts = 0
        step_tx_successes = 0
        step_contact_failures = 0
        step_drops = 0

        for agent_id in active_agents:
            idx = int(agent_id.split("_")[1])
            action = int(actions.get(agent_id, 0))
            queue = self.queues[idx]
            infos[agent_id]["queue_occupancy"] = self._queue_util(idx)

            if not queue:
                rewards[agent_id] = -0.01
                continue

            bundle = queue[0]
            neighbors = self._get_neighbors(idx, self.dist_matrix)

            if action <= 0:
                bundle.contact_wait_ms += self.step_duration_ms
                rewards[agent_id] = -0.04 - 0.20 * np.exp(1.0 * self._queue_util(idx))
                continue

            dst = neighbors[min(action - 1, 3)]
            self.tx_attempts += 1
            step_tx_attempts += 1
            infos[agent_id]["tx_attempted"] = 1

            if not self._link_available(idx, dst):
                bundle.contact_wait_ms += self.step_duration_ms
                self.contact_failures += 1
                step_contact_failures += 1
                rewards[agent_id] = -0.25 - 0.10 * np.exp(1.0 * self._queue_util(idx))
                continue

            bundle = queue.pop(0)
            old_dist = self.dist_matrix[idx, bundle.destination]
            new_dist = self.dist_matrix[dst, bundle.destination]
            tx_delay_ms = self._transmission_delay_ms(idx, dst)
            finish_time_ms = self.sim_time_ms + tx_delay_ms
            bundle.hop_count += 1
            bundle.current_node = dst
            bundle.enqueue_step = self.current_step
            bundle.enqueue_time_ms = finish_time_ms

            self.tx_successes += 1
            step_tx_successes += 1
            infos[agent_id]["tx_success"] = 1

            if dst == bundle.destination:
                delay_ms = finish_time_ms - bundle.generation_time_ms
                self.delivered_count += 1
                delivered_delays.append(delay_ms)
                self.delivery_records.append(
                    {
                        "bundle_id": bundle.bundle_id,
                        "source": bundle.source,
                        "destination": bundle.destination,
                        "delay_ms": delay_ms,
                        "hop_count": bundle.hop_count,
                        "queue_wait_ms": bundle.queue_wait_ms,
                        "contact_wait_ms": bundle.contact_wait_ms,
                    }
                )
                infos[agent_id]["delivered"] = 1
                infos[agent_id]["step_latency"] = delay_ms
                rewards[agent_id] = self._delivery_reward(delay_ms, bundle.hop_count)
            else:
                projected_len = len(self.queues[dst]) + len(pending_enqueues[dst])
                if projected_len >= self.queue_capacity_bundles:
                    self._record_drop(bundle, "next_hop_queue_overflow")
                    step_drops += 1
                    infos[agent_id]["dropped"] = 1
                    rewards[agent_id] = -1.0
                else:
                    pending_enqueues[dst].append(bundle)
                    progress = 1.0 if new_dist < old_dist else -1.0
                    congestion_penalty = self._queue_util(dst)
                    delay_penalty = min(0.2, tx_delay_ms / 1000.0)
                    rewards[agent_id] = 0.15 * progress - 0.45 * np.exp(1.0 * congestion_penalty) - delay_penalty
                    infos[agent_id]["step_latency"] = tx_delay_ms

        for dst, bundles in pending_enqueues.items():
            self.queues[dst].extend(bundles)

        for node, queue in self.queues.items():
            for pos, bundle in enumerate(queue):
                bundle.queue_wait_ms += self.step_duration_ms
                if pos > 0:
                    bundle.contact_wait_ms += 0.25 * self.step_duration_ms

        expired = self._expire_old_bundles()
        step_drops += expired
        generated = self._generate_arrivals(initial=False) if self.traffic_generation_enabled else 0
        self.sim_time_ms += self.step_duration_ms
        self._sync_node_queue_view()

        queue_utils = [self._queue_util(i) for i in range(self.num_nodes)]
        self.queue_history.append(float(np.mean(queue_utils)))
        self.step_history.append(
            {
                "step": self.current_step,
                "generated": generated,
                "delivered": len(delivered_delays),
                "dropped": step_drops,
                "tx_attempts": step_tx_attempts,
                "tx_successes": step_tx_successes,
                "contact_failures": step_contact_failures,
                "mean_delivery_delay_ms": float(np.mean(delivered_delays)) if delivered_delays else 0.0,
                "mean_queue_utilization": float(np.mean(queue_utils)),
            }
        )

        for agent_id in active_agents:
            infos[agent_id]["generated"] = generated / max(1, len(active_agents))
            if infos[agent_id]["step_latency"] == 0.0 and delivered_delays:
                infos[agent_id]["step_latency"] = float(np.mean(delivered_delays))

        done = self.current_step >= self.max_steps
        terminations = {a: done for a in active_agents}
        truncations = {a: False for a in active_agents}
        observations = {a: self._get_obs(int(a.split("_")[1])) for a in active_agents}
        return observations, rewards, terminations, truncations, infos

    def get_metrics(self) -> Dict[str, float]:
        delays = np.array([r['delay_ms'] for r in self.delivery_records], dtype=float)
        hops = np.array([r['hop_count'] for r in self.delivery_records], dtype=float)
        queue_wait = np.array([r['queue_wait_ms'] for r in self.delivery_records], dtype=float)
        contact_wait = np.array([r['contact_wait_ms'] for r in self.delivery_records], dtype=float)
        sim_seconds = max(self.sim_time_ms / 1000.0, 1e-9)
        generated = max(self.generated_count, 1)
        in_flight = self.in_flight_count()
        source_overflow = int(self.drop_reasons.get('source_queue_overflow', 0))
        next_hop_overflow = int(self.drop_reasons.get('next_hop_queue_overflow', 0))
        ttl_expired = int(self.drop_reasons.get('ttl_expired', 0))
        contact_unavailable = int(self.drop_reasons.get('contact_unavailable', 0))
        accounted = self.delivered_count + source_overflow + next_hop_overflow + ttl_expired + contact_unavailable + in_flight
        sum_check_error = self.generated_count - accounted
        return {
            'generated': float(self.generated_count),
            'delivered': float(self.delivered_count),
            'dropped': float(self.dropped_count),
            'in_flight_at_end': float(in_flight),
            'delivery_ratio': float(self.delivered_count / generated),
            'drop_rate': float(self.dropped_count / generated),
            'in_flight_ratio': float(in_flight / generated),
            'source_queue_overflow_ratio': float(source_overflow / generated),
            'next_hop_queue_overflow_ratio': float(next_hop_overflow / generated),
            'ttl_expired_ratio': float(ttl_expired / generated),
            'contact_unavailable_ratio': float(contact_unavailable / generated),
            'sum_check_error': float(sum_check_error),
            'sum_check_error_ratio': float(sum_check_error / generated),
            'mean_delay_ms': float(np.mean(delays)) if delays.size else 0.0,
            'p95_delay_ms': float(np.percentile(delays, 95)) if delays.size else 0.0,
            'mean_hop_count': float(np.mean(hops)) if hops.size else 0.0,
            'mean_queue_wait_ms': float(np.mean(queue_wait)) if queue_wait.size else 0.0,
            'mean_contact_wait_ms': float(np.mean(contact_wait)) if contact_wait.size else 0.0,
            'mean_queue_utilization': float(np.mean(self.queue_history)) if self.queue_history else 0.0,
            'throughput_mbps': float(self.delivered_count * self.bundle_size_mbit / sim_seconds),
            'tx_attempts': float(self.tx_attempts),
            'tx_successes': float(self.tx_successes),
            'contact_failures': float(self.contact_failures),
            'source_queue_overflow': float(source_overflow),
            'next_hop_queue_overflow': float(next_hop_overflow),
            'ttl_expired': float(ttl_expired),
            'contact_unavailable_drops': float(contact_unavailable),
        }

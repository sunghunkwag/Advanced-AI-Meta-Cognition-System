from dataclasses import dataclass
from typing import List, Tuple
import torch


@dataclass
class PlanResult:
    action: dict
    objective: float
    score_components: dict
    trace: List[float]


class System2Planner:
    def __init__(self, world_model, action_encoder, objective_weights=None, depth: int = 2, candidates: int = 6):
        self.world_model = world_model
        self.action_encoder = action_encoder
        self.depth = depth
        self.candidates = candidates
        self.objective_weights = objective_weights or {"energy": 1.0, "consistency": 1.0, "cortisol": 0.0}

    def rollout(self, state: torch.Tensor, candidate_actions: List[dict], cortisol: float = 0.0) -> PlanResult:
        best_obj = float("inf")
        best_action = candidate_actions[0]
        best_trace: List[float] = []

        for action in candidate_actions:
            encoded = self.action_encoder(action)
            obj, trace = self._simulate_depth(state, encoded, cortisol, current_depth=0)
            if obj < best_obj:
                best_obj = obj
                best_action = action
                best_trace = trace
        return PlanResult(action=best_action, objective=best_obj, score_components={}, trace=best_trace)

    def _simulate_depth(self, state: torch.Tensor, action_vec: torch.Tensor, cortisol: float, current_depth: int) -> Tuple[float, List[float]]:
        next_state, energy_pred, consistency_pred = self.world_model.simulate(state, action_vec)
        energy = energy_pred.item()
        consistency = consistency_pred.item()
        panic = self.objective_weights.get("cortisol", 0.0) * cortisol
        objective = self.objective_weights.get("energy", 1.0) * energy - self.objective_weights.get("consistency", 1.0) * consistency + panic

        trace = [objective]
        if current_depth + 1 >= self.depth:
            return objective, trace

        # simple heuristic: reuse same action_vec for deeper rollout to keep compute bounded
        deeper_obj, deeper_trace = self._simulate_depth(next_state, action_vec, cortisol, current_depth + 1)
        return objective + 0.9 * deeper_obj, trace + deeper_trace


class PlanningArbiter:
    """Arbitrates when to escalate control to the planner.

    This replaces the previous "PrefrontalCortex" naming to avoid confusion
    with the willpower/episodic-memory PFC implemented in ``cortex.py``.
    ``PrefrontalCortex`` is kept as an alias for backward compatibility.
    """

    def __init__(self, planner: System2Planner, cortisol_threshold: float, consistency_threshold: float, failure_streak: int, enabled: bool = True):
        self.planner = planner
        self.cortisol_threshold = cortisol_threshold
        self.consistency_threshold = consistency_threshold
        self.failure_streak = failure_streak
        self.enabled = enabled
        self.recent_failures: List[float] = []
        self.interventions = 0

    def should_intervene(self, cortisol: float, consistency: float, energy_delta: float) -> bool:
        self.recent_failures.append(energy_delta)
        if len(self.recent_failures) > self.failure_streak:
            self.recent_failures.pop(0)
        failure_loop = all(delta >= 0 for delta in self.recent_failures) and len(self.recent_failures) == self.failure_streak
        if cortisol > self.cortisol_threshold:
            return True
        if consistency < self.consistency_threshold:
            return True
        if failure_loop:
            return True
        return False

    def plan_if_needed(self, state: torch.Tensor, candidate_actions: List[dict], cortisol: float, consistency: float, energy_delta: float) -> Tuple[dict, PlanResult]:
        if self.enabled and self.should_intervene(cortisol, consistency, energy_delta):
            result = self.planner.rollout(state, candidate_actions, cortisol)
            self.interventions += 1
            return result.action, result
        return candidate_actions[0], None


# Backward compatibility with legacy naming used by older entrypoints/tests
PrefrontalCortex = PlanningArbiter

import argparse
import csv
import os
import time
from typing import List

import numpy as np
import torch

from action_decoder import ActionDecoder
from automata import IntrinsicAutomata
from config import DEFAULT_CONFIG, ExperimentConfig, set_global_seed
from energy import NeuroChemicalEngine
from manifold import GraphAttentionManifold
from meta_learner import MetaLearner
from planner import PlanningArbiter, System2Planner
from soul import get_soul_vectors
from vision import VisionSystem
from world import World
from world_model import WorldModel, WorldModelTrainer


def candidate_actions_from_logits(body: ActionDecoder, action_logits: torch.Tensor, params: torch.Tensor, k: int) -> List[dict]:
    base = body.decode_action(action_logits, params)
    actions = [base]
    for _ in range(k - 1):
        noisy_logits = action_logits + torch.randn_like(action_logits) * 0.5
        noisy_params = params + torch.randn_like(params) * 0.1
        actions.append(body.decode_action(noisy_logits, noisy_params))
    return actions


def run(cfg: ExperimentConfig):
    set_global_seed(cfg.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)

    world = World(size=cfg.grid_size)
    vision = VisionSystem()
    LATENT_DIM = 8
    v_id, v_truth, v_rej = get_soul_vectors(dim=LATENT_DIM)
    mind = GraphAttentionManifold(
        nfeat=3,
        nhid=16,
        nclass=LATENT_DIM,
        truth_vector=v_truth,
        reject_vector=v_rej,
    )
    body = ActionDecoder(latent_dim=LATENT_DIM)
    heart = NeuroChemicalEngine()
    soul = IntrinsicAutomata(mind)

    optimizer = torch.optim.Adam(list(mind.parameters()) + list(body.parameters()), lr=0.005)

    action_dim = body.num_actions + 4
    world_model = WorldModel(grid_size=cfg.grid_size, action_dim=action_dim)
    world_model_trainer = WorldModelTrainer(world_model)

    planner = System2Planner(
        world_model, action_encoder=body.encode_action, depth=cfg.planner.depth, candidates=cfg.planner.candidates
    )
    prefrontal = PlanningArbiter(
        planner,
        cfg.planner.cortisol_override,
        cfg.planner.consistency_override,
        cfg.planner.failure_streak,
        enabled=cfg.planner.enabled,
    )
    meta = MetaLearner(cfg.meta)

    exploration_ref = {"epsilon": 0.1}
    ewc_lambda_ref = {"lambda": soul.ewc_lambda}

    log_path = os.path.join(cfg.log_dir, f"run_seed{cfg.seed}_t{int(time.time())}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "energy",
                "consistency",
                "dopamine",
                "serotonin",
                "cortisol",
                "planner_objective",
                "interventions",
                "wm_loss",
            ],
        )
        writer.writeheader()

        prev_energy = None
        energy_history: List[float] = []
        consistency_history: List[float] = []
        action_history: List[str] = []
        hormone_history: List[Tuple[float, float]] = []

        for step in range(1, cfg.steps + 1):
            state = torch.tensor(world.get_state(), dtype=torch.float32)
            nodes, adj = vision.perceive(world.get_state())
            z = mind(nodes, adj)
            consistency = mind.check_consistency(z)
            energy = world.calculate_energy()

            energy_history.append(energy)
            consistency_history.append(consistency.item())

            density = np.sum(world.grid > 0.1) / world.grid.size
            h_sym = 1.0 - np.abs(world.grid - np.fliplr(world.grid)).mean()
            v_sym = 1.0 - np.abs(world.grid - np.flipud(world.grid)).mean()
            symmetry = (h_sym + v_sym) / 2.0
            prediction_error = 1.0 - consistency.item()

            heart.update(energy, consistency.item(), density, symmetry, prediction_error)
            hormones = heart.get_hormones()
            dopamine = hormones["dopamine"]
            serotonin = hormones["serotonin"]
            cortisol = hormones["cortisol"]

            hormone_history.append((dopamine, serotonin))
            soul.update_state(
                (dopamine, serotonin),
                nodes,
                adj,
                energy_history=energy_history,
                consistency_history=consistency_history,
                hormone_history=hormone_history,
            )
            soul.ewc_lambda = ewc_lambda_ref["lambda"]
            ewc_loss = soul.ewc_loss(mind)

            action_logits, params = body(z)
            # epsilon-greedy sample for baseline
            if np.random.rand() < exploration_ref["epsilon"]:
                action_logits = torch.randn_like(action_logits)
            candidates = candidate_actions_from_logits(body, action_logits, params, cfg.planner.candidates)

            energy_delta = 0.0 if prev_energy is None else energy - prev_energy
            chosen_action, plan_result = prefrontal.plan_if_needed(state, candidates, cortisol, consistency.item(), energy_delta)
            if plan_result is None:
                chosen_action = candidates[0]
            action_history.append(chosen_action["type"])

            prev_grid = torch.tensor(world.grid.copy(), dtype=torch.float32)
            world.apply_action(chosen_action)
            next_grid = torch.tensor(world.grid.copy(), dtype=torch.float32)

            world_model_trainer.add_transition(prev_grid, body.encode_action(chosen_action), next_grid, energy, consistency.item())
            wm_loss = world_model_trainer.train_step(batch_size=8)

            # compute task loss
            rejection_penalty = mind.check_rejection(z)
            loss = (
                torch.tensor(energy, dtype=torch.float32)
                + (1.0 - consistency)
                + (rejection_penalty * 0.5)
                + ewc_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if len(energy_history) > 20:
                energy_history.pop(0)
            if len(consistency_history) > 20:
                consistency_history.pop(0)
            if len(hormone_history) > 50:
                hormone_history.pop(0)

            energy_trend = 0.0
            if len(energy_history) >= 5:
                energy_trend = np.mean(np.diff(energy_history[-5:]))
            consistency_vol = float(np.std(consistency_history[-5:])) if len(consistency_history) >= 5 else 0.0
            action_div = len(set(action_history[-10:])) if len(action_history) >= 3 else 1.0
            meta.update(
                step,
                metrics={
                    "energy_trend": energy_trend,
                    "consistency_vol": consistency_vol,
                    "action_diversity": action_div,
                    "prediction_error": prediction_error,
                },
                optimizer=optimizer,
                exploration_ref=exploration_ref,
                ewc_lambda_ref=ewc_lambda_ref,
                heart=heart,
            )

            writer.writerow(
                {
                    "step": step,
                    "energy": energy,
                    "consistency": consistency.item(),
                    "dopamine": dopamine,
                    "serotonin": serotonin,
                    "cortisol": cortisol,
                    "planner_objective": plan_result.objective if plan_result else 0.0,
                    "interventions": prefrontal.interventions,
                    "wm_loss": wm_loss["total"] if wm_loss else 0.0,
                }
            )
            prev_energy = energy
    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_CONFIG.steps)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--planner", choices=["on", "off"], default="on")
    parser.add_argument("--meta", choices=["on", "off"], default="on")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    cfg.steps = args.steps
    cfg.seed = args.seed
    cfg.planner.enabled = args.planner == "on"
    cfg.meta.enabled = args.meta == "on"
    path = run(cfg)
    print(f"[RESULT] log saved to {path}")

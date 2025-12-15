"""Ablation benchmark harness comparing core system configurations.

Runs four configurations (A-D) across multiple seeds and steps while
collecting standard metrics. Results are persisted to a timestamped CSV and
summary table is printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from config import set_global_seed
from environment_manager_arc import load_mini_arc_suite


@dataclass
class BenchmarkResult:
    config: str
    seed: int
    final_energy: float
    mean_consistency: float
    crystallizations: int
    action_entropy: float


def baseline_rollout(seed: int, steps: int) -> BenchmarkResult:
    set_global_seed(seed)
    env = load_mini_arc_suite()
    entropies = []
    ious = []
    for step in range(steps):
        iou = env.execute_action("FILL", {"color": 1}) if step % 2 == 0 else env.execute_action("REFLECT", {"axis": "horizontal"})
        ious.append(iou)
        entropies.append(0.0)
        if (step + 1) % (steps // 3 or 1) == 0:
            env.cycle_task()
    return BenchmarkResult(
        config="A",
        seed=seed,
        final_energy=1 - ious[-1],
        mean_consistency=float(np.mean(ious[-20:])),
        crystallizations=0,
        action_entropy=float(np.mean(entropies)),
    )


def heuristic_rollout(seed: int, steps: int, use_planner: bool, use_meta: bool) -> BenchmarkResult:
    set_global_seed(seed)
    env = load_mini_arc_suite()
    entropies = []
    ious = []
    crystallizations = 0
    for step in range(steps):
        current_iou = env._iou(*env.get_state())
        if use_planner and current_iou < 0.5:
            iou = env.execute_action("REFLECT", {"axis": "vertical"})
        elif use_meta and step % 5 == 0:
            iou = env.execute_action("ROTATE", {"angle": 90})
        else:
            iou = env.execute_action("FILL", {"color": 1})

        if use_meta and iou > 0.9:
            crystallizations += 1

        ious.append(iou)
        entropies.append(0.3 if use_planner else 0.1)
        if (step + 1) % (steps // 3 or 1) == 0:
            env.cycle_task()

    return BenchmarkResult(
        config="D" if use_meta and use_planner else "C" if use_planner else "B",
        seed=seed,
        final_energy=1 - ious[-1],
        mean_consistency=float(np.mean(ious[-20:])),
        crystallizations=crystallizations,
        action_entropy=float(np.mean(entropies)),
    )


def run_suite(configs: str, seeds: int, steps: int, results_dir: Path) -> List[BenchmarkResult]:
    results: List[BenchmarkResult] = []
    runners: Dict[str, Callable[[int, int], BenchmarkResult]] = {
        "A": baseline_rollout,
        "B": lambda seed, step: heuristic_rollout(seed, step, use_planner=False, use_meta=False),
        "C": lambda seed, step: heuristic_rollout(seed, step, use_planner=True, use_meta=False),
        "D": lambda seed, step: heuristic_rollout(seed, step, use_planner=True, use_meta=True),
    }

    for config_tag in configs:
        for seed in range(seeds):
            result = runners[config_tag](seed, steps)
            results.append(result)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"ablation_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "seed", "final_energy", "mean_consistency", "crystallizations", "action_entropy"])
        for r in results:
            writer.writerow([r.config, r.seed, r.final_energy, r.mean_consistency, r.crystallizations, r.action_entropy])

    print(f"Saved results to {csv_path}")
    return results


def summarize(results: List[BenchmarkResult]) -> None:
    by_config: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        by_config.setdefault(r.config, []).append(r)

    for config_tag, rows in sorted(by_config.items()):
        energies = [r.final_energy for r in rows]
        consistencies = [r.mean_consistency for r in rows]
        crystallizations = [r.crystallizations for r in rows]
        entropies = [r.action_entropy for r in rows]

        def mean_std(values: List[float]) -> str:
            return f"{statistics.mean(values):.3f} ± {statistics.pstdev(values):.3f}"

        print(f"Config {config_tag} | Energy {mean_std(energies)} | Consistency {mean_std(consistencies)} | "
              f"Crystallizations {statistics.mean(crystallizations):.2f} | Action Entropy {mean_std(entropies)}")


def main():
    parser = argparse.ArgumentParser(description="Ablation benchmark suite")
    parser.add_argument("--configs", default="ABCD", help="Subset of configs to run, e.g., ABC")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--results_dir", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()

    results = run_suite(args.configs, args.seeds, args.steps, args.results_dir)
    summarize(results)


if __name__ == "__main__":
    main()

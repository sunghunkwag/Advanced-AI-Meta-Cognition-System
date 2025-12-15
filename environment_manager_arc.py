"""Minimal ARC-style environment manager with a few curated tasks.

This module provides a lightweight approximation of ARC puzzles so that
benchmark sweeps can stress-test the agent on simple pattern-completion
objectives without introducing heavy external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ARCTask:
    name: str
    input_grid: np.ndarray
    target_grid: np.ndarray


class ARCEnvironment:
    def __init__(self, tasks: List[ARCTask]):
        self.tasks = tasks
        self.current_index = 0
        self.current_grid = np.copy(self.tasks[self.current_index].input_grid)

    @property
    def target_grid(self) -> np.ndarray:
        return self.tasks[self.current_index].target_grid

    def cycle_task(self) -> None:
        self.current_index = (self.current_index + 1) % len(self.tasks)
        self.current_grid = np.copy(self.tasks[self.current_index].input_grid)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.copy(self.current_grid), np.copy(self.target_grid)

    def execute_action(self, action_type: str, params: Dict) -> float:
        if action_type == "FILL":
            color = params.get("color", 1)
            region = params.get("region", (slice(None), slice(None)))
            self.current_grid[region] = color
        elif action_type == "COPY":
            src = params.get("source_region", (slice(None), slice(None)))
            dst = params.get("dest_region", (slice(None), slice(None)))
            self.current_grid[dst] = self.current_grid[src]
        elif action_type == "ROTATE":
            region = params.get("region", (slice(None), slice(None)))
            angle = params.get("angle", 90)
            k = (angle // 90) % 4
            self.current_grid[region] = np.rot90(self.current_grid[region], k)
        elif action_type == "REFLECT":
            axis = params.get("axis", "horizontal")
            region = params.get("region", (slice(None), slice(None)))
            if axis == "horizontal":
                self.current_grid[region] = np.fliplr(self.current_grid[region])
            else:
                self.current_grid[region] = np.flipud(self.current_grid[region])

        return self._iou(self.current_grid, self.target_grid)

    @staticmethod
    def _iou(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
        overlap = np.logical_and(grid_a == grid_b, grid_a > 0)
        union = np.logical_or(grid_a > 0, grid_b > 0)
        if union.sum() == 0:
            return 1.0
        return overlap.sum() / union.sum()


def load_mini_arc_suite() -> ARCEnvironment:
    tasks = [
        ARCTask(
            name="cross_fill",
            input_grid=np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ]
            ),
            target_grid=np.array(
                [
                    [0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0],
                ]
            ),
        ),
        ARCTask(
            name="mirror_pair",
            input_grid=np.array(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ]
            ),
            target_grid=np.array(
                [
                    [1, 0, 1],
                    [0, 0, 0],
                    [1, 0, 1],
                ]
            ),
        ),
        ARCTask(
            name="rotate_square",
            input_grid=np.array(
                [
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                ]
            ),
            target_grid=np.array(
                [
                    [0, 0, 2],
                    [0, 2, 0],
                    [2, 0, 0],
                ]
            ),
        ),
    ]

    return ARCEnvironment(tasks)

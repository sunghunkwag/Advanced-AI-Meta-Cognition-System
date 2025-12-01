import numpy as np
from typing import Dict, List, Tuple, Optional
from world import World

class EnvironmentManager:
    """
    Manages multiple environment configurations and adaptive strategy selection.
    Enables the AI to work across different grid sizes, obstacle patterns, and dynamics.
    """
    
    def __init__(self):
        self.environment_catalog = {
            "tiny": {
                "size": 8,
                "difficulty": "easy",
                "target_density": 0.15,
                "description": "Small 8x8 grid for quick learning"
            },
            "small": {
                "size": 12,
                "difficulty": "easy",
                "target_density": 0.12,
                "description": "12x12 grid"
            },
            "medium": {
                "size": 16,
                "difficulty": "normal",
                "target_density": 0.10,
                "description": "Standard 16x16 grid"
            },
            "large": {
                "size": 24,
                "difficulty": "hard",
                "target_density": 0.08,
                "description": "Large 24x24 grid requiring planning"
            },
            "huge": {
                "size": 32,
                "difficulty": "very_hard",
                "target_density": 0.06,
                "description": "Massive 32x32 grid"
            }
        }
        
        self.current_environment = None
        self.environment_history = []
        
    def create_environment(self, env_type: str, **kwargs) -> World:
        """
        Create a world instance based on environment type.
        
        Args:
            env_type: One of ["tiny", "small", "medium", "large", "huge"]
            **kwargs: Additional configuration options
            
        Returns:
            Configured World instance
        """
        if env_type not in self.environment_catalog:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        config = self.environment_catalog[env_type].copy()
        config.update(kwargs)
        
        world = World(size=config['size'])
        self.current_environment = {
            "type": env_type,
            "config": config,
            "world": world
        }
        self.environment_history.append(env_type)
        
        return world
    
    def analyze_environment(self, world: World) -> Dict:
        """
        Analyze environment characteristics to inform strategy selection.
        
        Returns:
            Dictionary with environment features
        """
        grid = world.get_state()
        
        analysis = {
            "size": world.size,
            "total_cells": world.size ** 2,
            "current_density": np.mean(grid > 0.1),
            "occupied_cells": np.sum(grid > 0.1),
            "empty_cells": np.sum(grid <= 0.1),
            "max_value": np.max(grid),
            "min_value": np.min(grid),
            "complexity": self._estimate_complexity(grid),
            "symmetry_horizontal": self._measure_symmetry(grid, axis='horizontal'),
            "symmetry_vertical": self._measure_symmetry(grid, axis='vertical'),
            "center_bias": self._measure_center_bias(grid),
            "fragmentation": self._measure_fragmentation(grid)
        }
        
        # Classify difficulty
        if world.size <= 12:
            analysis['difficulty'] = "easy"
        elif world.size <= 20:
            analysis['difficulty'] = "medium"
        else:
            analysis['difficulty'] = "hard"
        
        return analysis
    
    def _estimate_complexity(self, grid: np.ndarray) -> float:
        """Estimate pattern complexity (0.0 = simple, 1.0 = complex)"""
        # Use variance and edge density as complexity proxy
        variance = np.var(grid)
        
        # Edge detection (rough approximation)
        h_edges = np.abs(np.diff(grid, axis=0)).sum()
        v_edges = np.abs(np.diff(grid, axis=1)).sum()
        edge_density = (h_edges + v_edges) / (2 * grid.size)
        
        complexity = (variance * 2 + edge_density) / 2
        return min(complexity, 1.0)
    
    def _measure_symmetry(self, grid: np.ndarray, axis: str = 'horizontal') -> float:
        """Measure symmetry (1.0 = perfect symmetry, 0.0 = no symmetry)"""
        if axis == 'horizontal':
            flipped = np.fliplr(grid)
        else:
            flipped = np.flipud(grid)
        
        difference = np.abs(grid - flipped).mean()
        symmetry = max(0, 1.0 - difference * 5)
        return symmetry
    
    def _measure_center_bias(self, grid: np.ndarray) -> float:
        """Measure how concentrated values are in the center"""
        size = grid.shape[0]
        center_start = size // 4
        center_end = 3 * size // 4
        
        center_mass = grid[center_start:center_end, center_start:center_end].sum()
        total_mass = grid.sum()
        
        if total_mass == 0:
            return 0.0
        
        return center_mass / total_mass
    
    def _measure_fragmentation(self, grid: np.ndarray) -> float:
        """Measure how fragmented (scattered) the pattern is"""
        # Simple fragmentation: number of isolated regions
        # High value = scattered, low value = consolidated
        occupied = (grid > 0.1).astype(int)
        
        # Count transitions (rough measure)
        h_transitions = np.abs(np.diff(occupied, axis=0)).sum()
        v_transitions = np.abs(np.diff(occupied, axis=1)).sum()
        
        total_transitions = h_transitions + v_transitions
        max_possible = 2 * grid.size
        
        fragmentation = total_transitions / max_possible if max_possible > 0 else 0
        return fragmentation


class AdaptiveStrategy:
    """
    Selects appropriate strategies based on environment analysis.
    Maps environment characteristics to optimal approaches.
    """
    
    def __init__(self):
        self.strategy_library = {
            "aggressive_filling": {
                "description": "Fast, dense filling for small spaces",
                "best_for": "tiny/small grids",
               "action_bias": {"DRAW": 0.7, "NOISE": 0.2, "SYMMETRIZE": 0.05, "CLEAR": 0.05}
            },
            "phased_expansion": {
                "description": "Center-out expansion for large spaces",
                "best_for": "large/huge grids",
                "action_bias": {"DRAW": 0.5, "SYMMETRIZE": 0.3, "NOISE": 0.1, "CLEAR": 0.1}
            },
            "balanced_growth": {
                "description": "Balanced approach for medium grids",
                "best_for": "medium grids",
                "action_bias": {"DRAW": 0.4, "SYMMETRIZE": 0.3, "NOISE": 0.2, "CLEAR": 0.1}
            },
            "refinement_focused": {
                "description": "Clean up and symmetrize existing patterns",
                "best_for": "dense/messy grids",
                "action_bias": {"SYMMETRIZE": 0.5, "CLEAR": 0.3, "DRAW": 0.15, "NOISE": 0.05}
            }
        }
        
        self.current_strategy = None
        
    def select_strategy(self, env_analysis: Dict) -> str:
        """
        Choose optimal strategy based on environment characteristics.
        
        Args:
            env_analysis: Output from EnvironmentManager.analyze_environment()
            
        Returns:
            Strategy name
        """
        size = env_analysis.get('size', 16)
        density = env_analysis.get('current_density', 0.0)
        complexity = env_analysis.get('complexity', 0.0)
        
        # Decision tree for strategy selection
        if size <=  12:
            strategy = "aggressive_filling"
            
        elif size >= 24:
            strategy = "phased_expansion"
            
        elif density > 0.7 or complexity > 0.6:
            strategy = "refinement_focused"
            
        else:
            strategy = "balanced_growth"
        
        self.current_strategy = strategy
        return strategy
    
    def get_strategy_config(self, strategy_name: str) -> Dict:
        """Get full configuration for a strategy"""
        return self.strategy_library.get(strategy_name, self.strategy_library["balanced_growth"])
    
    def recommend_meta_goal(self, env_analysis: Dict, strategy: str) -> str:
        """Recommend a meta-goal based on environment and strategy"""
        density = env_analysis.get('current_density', 0.0)
        symmetry = max(
            env_analysis.get('symmetry_horizontal', 0.0),
            env_analysis.get('symmetry_vertical', 0.0)
        )
        
        if density < 0.05:
            return "maximize_coverage"
        elif symmetry < 0.5:
            return "achieve_harmony"
        elif env_analysis.get('complexity', 0.0) < 0.3:
            return "create_complexity"
        else:
            return "minimize_chaos"


class CurriculumManager:
    """
    Manages progressive training across environments.
    Implements curriculum learning: easy → medium → hard.
    """
    
    def __init__(self):
        self.curriculum_sequence = [
            "tiny",      # Stage 1: Learn basics
            "small",     # Stage 2: Refine
            "medium",    # Stage 3: Standard challenge
            "large",     # Stage 4: Advanced
            "huge"       # Stage 5: Master level
        ]
        
        self.current_stage = 0
        self.stage_performance = {}
        self.promotion_threshold = 0.7  # Success rate to advance
        
    def get_next_environment(self) -> str:
        """Get the next environment in the curriculum"""
        if self.current_stage < len(self.curriculum_sequence):
            return self.curriculum_sequence[self.current_stage]
        else:
            return self.curriculum_sequence[-1]  # Stay at hardest
    
    def evaluate_performance(self, final_energy: float, steps: int, 
                            crystallized: bool) -> float:
        """
        Evaluate performance on current stage.
        
        Returns:
            Performance score 0.0 to 1.0
        """
        score = 0.0
        
        # Energy component (lower is better)
        if final_energy < 0.1:
            score += 0.4
        elif final_energy < 0.5:
            score += 0.2
        
        # Speed component
        if steps < 50:
            score += 0.3
        elif steps < 100:
            score += 0.2
        
        # Crystallization bonus
        if crystallized:
            score += 0.3
        
        return min(score, 1.0)
    
    def should_advance(self, performance_history: List[float]) -> bool:
        """Determine if agent should advance to next stage"""
        if len(performance_history) < 3:
            return False  # Need multiple attempts
        
        recent_performance = np.mean(performance_history[-3:])
        return recent_performance >= self.promotion_threshold
    
    def advance_stage(self):
        """Move to next curriculum stage"""
        if self.current_stage < len(self.curriculum_sequence) - 1:
            self.current_stage += 1
            print(f"[CURRICULUM] Advanced to stage {self.current_stage + 1}: "
                  f"{self.curriculum_sequence[self.current_stage]}")
    
    def get_current_stage_name(self) -> str:
        """Get name of current curriculum stage"""
        return self.curriculum_sequence[self.current_stage]
    
    def get_progress(self) -> Dict:
        """Get curriculum progress summary"""
        return {
            "current_stage": self.current_stage + 1,
            "total_stages": len(self.curriculum_sequence),
            "environment": self.get_current_stage_name(),
            "progress_percent": (self.current_stage / len(self.curriculum_sequence)) * 100
        }

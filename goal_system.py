import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

class GoalSystem:
    """
    Hierarchical Goal Generation System
    Enables AI to set its own goals and decompose them into actionable strategies.
    
    3-Level Hierarchy:
    1. Meta-Goals: Abstract, philosophical objectives
    2. Strategic Goals: Concrete sub-objectives
    3. Tactical Actions: Immediate executable actions
    """
    
    def __init__(self):
        # Level 1: Meta-Goals (What to achieve overall)
        self.meta_goals = [
            "achieve_harmony",      # Balance + symmetry
            "maximize_coverage",    # Fill space efficiently
            "create_complexity",    # Rich, interesting patterns
            "minimize_chaos"        # Clean, organized structure
        ]
        
        # Level 2: Strategic Goals (How to achieve meta-goals)
        self.goal_tree = {
            "achieve_harmony": [
                "make_symmetric_horizontal",
                "make_symmetric_vertical",
                "balance_density_distribution",
                "create_radial_symmetry"
            ],
            "maximize_coverage": [
                "fill_center_first",
                "expand_to_corners",
                "avoid_sparse_regions",
                "uniform_distribution"
            ],
            "create_complexity": [
                "add_fractal_patterns",
                "create_nested_structures",
                "introduce_variation",
                "layer_multiple_patterns"
            ],
            "minimize_chaos": [
                "remove_noise",
                "clean_boundaries",
                "enforce_consistency",
                "simplify_structure"
            ]
        }
        
        # Level 3: Tactical Action Mapping
        self.strategy_to_actions = {
            "fill_center_first": ["DRAW"],
            "make_symmetric_horizontal": ["SYMMETRIZE", "DRAW"],
            "create_nested_structures": ["DRAW", "NOISE"],
            "remove_noise": ["CLEAR", "SYMMETRIZE"],
            # ... more mappings
        }
        
        self.current_meta_goal = None
        self.current_strategic_goals = []
        self.goal_history = []
        
    def select_meta_goal(self, world_state: np.ndarray, emotions: Dict[str, float], 
                        context: Dict) -> str:
        """
        Select high-level meta-goal based on world state and emotional state.
        
        Args:
            world_state: Current grid state
            emotions: Dict with 'dopamine', 'serotonin' levels
            context: Additional context (step, energy, etc.)
            
        Returns:
            Selected meta-goal name
        """
        grid_density = np.mean(world_state)
        grid_sum = np.sum(world_state)
        
        # Decision logic based on state + emotions
        if grid_sum < 1.0:  # Empty or nearly empty
            return "maximize_coverage"
            
        elif emotions.get('dopamine', 0.5) > 0.7:  # High drive/curiosity
            return "create_complexity"
            
        elif self._is_messy(world_state):  # High asymmetry
            return "achieve_harmony"
            
        elif grid_density > 0.8:  # Too dense/noisy
            return "minimize_chaos"
            
        else:
            # Default: maintain current or switch based on progress
            if self.current_meta_goal:
                return self.current_meta_goal
            return "achieve_harmony"
    
    def decompose_to_strategy(self, meta_goal: str, world_state: np.ndarray, 
                             context: Dict) -> List[str]:
        """
        Break down meta-goal into strategic sub-goals.
        
        Args:
            meta_goal: High-level goal
            world_state: Current grid state
            context: Current context
            
        Returns:
            Prioritized list of strategic goals
        """
        available_strategies = self.goal_tree.get(meta_goal, [])
        
        # Prioritize based on current state
        prioritized = self._prioritize_strategies(
            available_strategies, 
            world_state, 
            context
        )
        
        return prioritized
    
    def _prioritize_strategies(self, strategies: List[str], world_state: np.ndarray,
                               context: Dict) -> List[str]:
        """Rank strategies by relevance to current situation"""
        scores = {}
        
        for strategy in strategies:
            score = 0.0
            
            # Strategy-specific scoring
            if strategy == "fill_center_first":
                center_density = self._get_center_density(world_state)
                score = 1.0 - center_density  # Higher score if center is empty
                
            elif strategy == "make_symmetric_horizontal":
                asymmetry = self._measure_horizontal_asymmetry(world_state)
                score = asymmetry  # Higher score if asymmetric
                
            elif strategy == "remove_noise":
                noise_level = self._estimate_noise(world_state)
                score = noise_level
                
            elif strategy == "expand_to_corners":
                corner_density = self._get_corner_density(world_state)
                score = 1.0 - corner_density
                
            else:
                score = 0.5  # Default relevance
                
            scores[strategy] = score
        
        # Sort by score (descending)
        sorted_strategies = sorted(strategies, key=lambda s: scores.get(s, 0), reverse=True)
        return sorted_strategies
    
    def plan_tactics(self, strategic_goal: str, world_state: np.ndarray) -> List[str]:
        """
        Convert strategic goal to sequence of tactical actions.
        
        Args:
            strategic_goal: Mid-level strategy
            world_state: Current grid
            
        Returns:
            Action sequence (e.g., ["DRAW", "DRAW", "SYMMETRIZE"])
        """
        base_actions = self.strategy_to_actions.get(
            strategic_goal, 
            ["DRAW"]  # Default fallback
        )
        
        # Generate action plan (simplified for now)
        # In full implementation, this would create multi-step plans
        return base_actions
    
    def get_action_bias(self, strategic_goal: str) -> Dict[str, float]:
        """
        Get STRONG action probability biases based on current strategic goal.
        Returns a dict mapping action types to bias weights.
        Uses 10x stronger biasing to ensure goals actually guide actions.
        """
        biases = {
            "DRAW": 0.1,
            "CLEAR": 0.1,
            "NOISE": 0.1,
            "SYMMETRIZE": 0.1
        }
        
        # MUCH STRONGER biases based on strategy
        if "symmetric" in strategic_goal:
            biases["SYMMETRIZE"] = 0.7  # Was 0.5
            biases["DRAW"] = 0.2  # Was 0.3
            biases["NOISE"] = 0.05
            biases["CLEAR"] = 0.05
            
        elif "fill" in strategic_goal or "coverage" in strategic_goal:
            biases["DRAW"] = 0.75  # Much stronger
            biases["NOISE"] = 0.15
            biases["SYMMETRIZE"] = 0.05
            biases["CLEAR"] = 0.05
            
        elif "remove" in strategic_goal or "clean" in strategic_goal:
            biases["CLEAR"] = 0.6  # Much stronger
            biases["SYMMETRIZE"] = 0.25
            biases["DRAW"] = 0.1
            biases["NOISE"] = 0.05
            
        elif "complexity" in strategic_goal or "variation" in strategic_goal:
            biases["NOISE"] = 0.4
            biases["DRAW"] = 0.4
            biases["SYMMETRIZE"] = 0.1
            biases["CLEAR"] = 0.1
        
        return biases
    
    # Helper methods for state analysis
    def _is_messy(self, world_state: np.ndarray) -> bool:
        """Check if grid is asymmetric/messy"""
        h_asym = self._measure_horizontal_asymmetry(world_state)
        v_asym = self._measure_vertical_asymmetry(world_state)
        return (h_asym + v_asym) / 2 > 0.3
    
    def _measure_horizontal_asymmetry(self, grid: np.ndarray) -> float:
        """Measure horizontal symmetry deviation"""
        flipped = np.fliplr(grid)
        return np.abs(grid - flipped).mean()
    
    def _measure_vertical_asymmetry(self, grid: np.ndarray) -> float:
        """Measure vertical symmetry deviation"""
        flipped = np.flipud(grid)
        return np.abs(grid - flipped).mean()
    
    def _get_center_density(self, grid: np.ndarray) -> float:
        """Calculate density in center region"""
        size = grid.shape[0]
        center_start = size // 4
        center_end = 3 * size // 4
        center_region = grid[center_start:center_end, center_start:center_end]
        return np.mean(center_region)
    
    def _get_corner_density(self, grid: np.ndarray) -> float:
        """Calculate average density in four corners"""
        size = grid.shape[0]
        corner_size = size // 4
        
        tl = grid[:corner_size, :corner_size]
        tr = grid[:corner_size, -corner_size:]
        bl = grid[-corner_size:, :corner_size]
        br = grid[-corner_size:, -corner_size:]
        
        return np.mean([tl.mean(), tr.mean(), bl.mean(), br.mean()])
    
    def _estimate_noise(self, grid: np.ndarray) -> float:
        """Estimate noise level (high-frequency variation)"""
        # Simple noise estimate: standard deviation of differences
        h_diff = np.abs(np.diff(grid, axis=0)).mean()
        v_diff = np.abs(np.diff(grid, axis=1)).mean()
        return (h_diff + v_diff) / 2


class GoalTracker:
    """
    Tracks goal progress and manages goal transitions.
    """
    
    def __init__(self):
        self.active_goals = []
        self.completed_goals = []
        self.failed_goals = []
        self.progress = {}
        self.stuck_counter = {}
        
    def set_active_goal(self, goal: str):
        """Set a new active goal"""
        if goal not in self.active_goals:
            self.active_goals.append(goal)
            self.progress[goal] = 0.0
            self.stuck_counter[goal] = 0
    
    def evaluate_progress(self, goal: str, world_state: np.ndarray, 
                         prev_state: Optional[np.ndarray] = None) -> float:
        """
        Measure progress toward a specific goal.
        
        Returns:
            Progress value 0.0 to 1.0
        """
        if goal == "fill_center_first":
            size = world_state.shape[0]
            center_start = size // 4
            center_end = 3 * size // 4
            center = world_state[center_start:center_end, center_start:center_end]
            target_density = 0.7
            current_density = np.mean(center > 0.1)
            progress = min(current_density / target_density, 1.0)
            
        elif "symmetric" in goal:
            if "horizontal" in goal:
                asym = np.abs(world_state - np.fliplr(world_state)).mean()
            else:
                asym = np.abs(world_state - np.flipud(world_state)).mean()
            progress = max(0, 1.0 - asym * 5)  # Scale asymmetry
            
        elif "coverage" in goal or "fill" in goal:
            occupied_ratio = np.sum(world_state > 0.1) / world_state.size
            target = 0.5
            progress = min(occupied_ratio / target, 1.0)
            
        else:
            # Default: check for any improvement
            if prev_state is not None:
                improvement = np.abs(world_state - prev_state).mean()
                progress = min(improvement * 10, 1.0)
            else:
                progress = 0.5
        
        self.progress[goal] = progress
        return progress
    
    def should_switch_goal(self, goal: str, steps_on_goal: int) -> bool:
        """
        Determine if it's time to switch to a different goal.
        NOW WITH BETTER PERSISTENCE - don't switch too early!
        """
        progress = self.progress.get(goal, 0.0)
        
        # CRITICAL: Minimum commitment period - don't switch too early
        if steps_on_goal < 30:
            return False
        
        # Switch if goal is nearly complete
        if progress > 0.85:
            return True
        
        # Switch if truly stuck (no progress AND enough time given)
        if steps_on_goal > 100 and progress < 0.2:
            return True
        
        # Switch if making very slow progress for extended period
        if steps_on_goal > 80 and progress < 0.4:
            return True
        
        return False
    
    def mark_completed(self, goal: str):
        """Mark goal as successfully completed"""
        if goal in self.active_goals:
            self.active_goals.remove(goal)
        self.completed_goals.append(goal)
    
    def mark_failed(self, goal: str):
        """Mark goal as failed (abandoned)"""
        if goal in self.active_goals:
            self.active_goals.remove(goal)
        self.failed_goals.append(goal)
    
    def get_completion_rate(self) -> float:
        """Calculate overall goal completion rate"""
        total = len(self.completed_goals) + len(self.failed_goals)
        if total == 0:
            return 0.0
        return len(self.completed_goals) / total

import numpy as np
import torch
from typing import Dict, Tuple, Optional

class CuriosityModule:
    """
    Intrinsic Curiosity Module
    Provides intrinsic motivation through prediction error.
    Rewards the agent for encountering surprising/unexpected outcomes.
    """
    
    def __init__(self):
        self.prediction_error_history = []
        self.surprise_threshold = 0.1
        
    def predict_next_state(self, current_state: np.ndarray, 
                          planned_action: str) -> np.ndarray:
        """
        Predict what the next state will be after taking an action.
        Simple heuristic-based prediction (can be learned with a network later).
        
        Args:
            current_state: Current grid state
            planned_action: Action about to be taken
            
        Returns:
            Predicted next state
        """
        predicted = current_state.copy()
        
        if planned_action == 'DRAW':
            # Expect small increase in density
            predicted += np.random.rand(*current_state.shape) * 0.05
            
        elif planned_action == 'CLEAR':
            # Expect significant decrease
            predicted *= 0.3
            
        elif planned_action == 'NOISE':
            # Expect some random additions
            predicted += np.random.rand(*current_state.shape) * 0.1
            
        elif planned_action == 'SYMMETRIZE':
            # Expect symmetry increase (rough approximation)
            predicted = (predicted + np.fliplr(predicted)) / 2
        
        # Keep in valid range
        predicted = np.clip(predicted, 0, 1)
        return predicted
    
    def calculate_curiosity_bonus(self, predicted_state: np.ndarray, 
                                   actual_state: np.ndarray) -> float:
        """
        Calculate intrinsic reward based on prediction error.
        Higher surprise = higher curiosity bonus.
        
        Args:
            predicted_state: What we thought would happen
            actual_state: What actually happened
            
        Returns:
            Curiosity bonus (0.0 to ~1.0)
        """
        # Prediction error = surprise
        prediction_error = np.abs(predicted_state - actual_state).mean()
        
        # Scale surprise to bonus
        # High surprise early = good (exploration)
        # High surprise late = may indicate instability
        curiosity_bonus = min(prediction_error * 3.0, 1.0)
        
        self.prediction_error_history.append(prediction_error)
        
        return curiosity_bonus
    
    def get_exploration_drive(self) -> float:
        """
        Get current exploration drive based on recent prediction accuracy.
        If predictions are too accurate, encourage more exploration.
        
        Returns:
            Exploration drive 0.0 to 1.0
        """
        if len(self.prediction_error_history) < 10:
            return 0.5  # Default moderate exploration
        
        recent_errors = self.prediction_error_history[-20:]
        avg_error = np.mean(recent_errors)
        
        if avg_error < 0.05:
            # Too predictable, boost exploration
            return 0.8
        elif avg_error > 0.3:
            # Too chaotic, reduce exploration
            return 0.2
        else:
            return 0.5
    
    def should_try_novel_action(self, action_history: list) -> bool:
        """
        Encourage trying actions that haven't been used recently.
        
        Args:
            action_history: Recent action history
            
        Returns:
            True if should try a novel action
        """
        if len(action_history) < 20:
            return False
        
        recent_actions = action_history[-20:]
        unique_count = len(set(recent_actions))
        
        # If using < 3 different actions, try something new
        return unique_count < 3


class AdaptiveGoalDifficulty:
    """
    Dynamically adjusts goal difficulty based on agent performance.
    Like a video game's dynamic difficulty adjustment.
    """
    
    def __init__(self):
        self.current_difficulty = 0.5  # Start at medium
        self.success_history = []
        self.adjustment_interval = 5  # Adjust every N goal attempts
        
    def record_goal_attempt(self, success: bool, progress: float):
        """Record outcome of a goal attempt"""
        self.success_history.append({
            'success': success,
            'progress': progress
        })
    
    def adjust_difficulty(self) -> float:
        """
        Auto-tune difficulty based on recent performance.
        
        Returns:
            New difficulty level (0.1 to 1.0)
        """
        if len(self.success_history) < self.adjustment_interval:
            return self.current_difficulty
        
        recent = self.success_history[-self.adjustment_interval:]
        success_rate = sum(1 for r in recent if r['success']) / len(recent)
        avg_progress = np.mean([r['progress'] for r in recent])
        
        # Adjust based on success rate
        if success_rate > 0.8 and avg_progress > 0.7:
            # Too easy, increase difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
            print(f"[CHALLENGE↑] Difficulty increased to {self.current_difficulty:.1f}")
            
        elif success_rate < 0.3 or avg_progress < 0.3:
            # Too hard, decrease difficulty
            self.current_difficulty = max(0.2, self.current_difficulty - 0.1)
            print(f"[CHALLENGE↓] Difficulty decreased to {self.current_difficulty:.1f}")
        
        return self.current_difficulty
    
    def get_scaled_targets(self) -> Dict[str, float]:
        """
        Get goal targets scaled by current difficulty.
        
        Returns:
            Dictionary of target thresholds
        """
        base_targets = {
            'fill_center': 0.7,
            'make_symmetric': 0.8,
            'maximize_coverage': 0.6,
            'balance_density': 0.5
        }
        
        # Scale targets by difficulty
        scaled = {}
        for goal, base_target in base_targets.items():
            # Higher difficulty = higher targets
            scaled[goal] = base_target * (0.5 + 0.5 * self.current_difficulty)
        
        return scaled
    
    def get_difficulty_bonus(self) -> float:
        """
        Reward for completing harder goals.
        
        Returns:
            Bonus multiplier (1.0 to 2.0)
        """
        return 1.0 + self.current_difficulty


class HierarchicalLossCalculator:
    """
    Calculates multi-level loss that aligns with hierarchical goals.
    """
    
    def __init__(self):
        self.loss_weights = {
            'physical': 0.25,      # World energy
            'meta_goal': 0.25,     # High-level objective
            'strategic': 0.30,     # Mid-level tactics
            'truth': 0.20          # Consistency with axioms
        }
    
    def calculate_meta_goal_loss(self, world_state: np.ndarray, 
                                  meta_goal: str) -> float:
        """Calculate how well we're satisfying the meta-goal"""
        
        if meta_goal == "achieve_harmony":
            # Measure symmetry
            h_asym = np.abs(world_state - np.fliplr(world_state)).mean()
            v_asym = np.abs(world_state - np.flipud(world_state)).mean()
            loss = (h_asym + v_asym) / 2
            
        elif meta_goal == "maximize_coverage":
            # Measure coverage
            target_density = 0.5
            actual_density = np.sum(world_state > 0.1) / world_state.size
            loss = abs(target_density - actual_density)
            
        elif meta_goal == "create_complexity":
            # Measure pattern complexity (want high variance)
            complexity = np.var(world_state)
            loss = max(0, 0.1 - complexity)  # Penalize if too simple
            
        elif meta_goal == "minimize_chaos":
            # Measure cleanliness (want low variance in differences)
            h_diff = np.abs(np.diff(world_state, axis=0)).mean()
            v_diff = np.abs(np.diff(world_state, axis=1)).mean()
            loss = (h_diff + v_diff) / 2
            
        else:
            loss = 0.0
        
        return loss
    
    def calculate_strategic_loss(self, world_state: np.ndarray, 
                                 strategic_goal: str,
                                 progress: float) -> float:
        """Calculate strategic goal progress loss"""
        
        # Target is 80% progress
        target_progress = 0.8
        progress_error = max(0, target_progress - progress)
        
        # Scale by how important this goal is
        strategic_loss = progress_error * 1.5
        
        return strategic_loss
    
    def calculate_total_loss(self, world_energy: float, 
                            consistency: float,
                            meta_goal: str,
                            strategic_goal: str,
                            strategic_progress: float,
                            world_state: np.ndarray) -> Tuple[float, Dict]:
        """
        Calculate hierarchical loss combining all levels.
        
        Returns:
            (total_loss, breakdown_dict)
        """
        # Level 1: Physical world
        physical_loss = world_energy * self.loss_weights['physical']
        
        # Level 2: Meta-goal
        meta_loss = 0.0
        if meta_goal:
            meta_loss = self.calculate_meta_goal_loss(world_state, meta_goal)
            meta_loss *= self.loss_weights['meta_goal']
        
        # Level 3: Strategic
        strategic_loss = 0.0
        if strategic_goal:
            strategic_loss = self.calculate_strategic_loss(
                world_state, strategic_goal, strategic_progress
            )
            strategic_loss *= self.loss_weights['strategic']
        
        # Level 4: Truth/Consistency
        truth_loss = (1.0 - consistency) * self.loss_weights['truth']
        
        total = physical_loss + meta_loss + strategic_loss + truth_loss
        
        breakdown = {
            'physical': physical_loss,
            'meta': meta_loss,
            'strategic': strategic_loss,
            'truth': truth_loss,
            'total': total
        }
        
        return total, breakdown

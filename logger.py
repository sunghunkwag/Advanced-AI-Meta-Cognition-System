"""Logging utilities for Advanced AI Meta-Cognition System.

Provides structured logging with metrics tracking and visualization support.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class StepMetrics:
    """Metrics for a single step."""
    step: int
    timestamp: float
    
    # State
    mode: str
    dopamine: float
    serotonin: float
    
    # Performance
    world_energy: float
    consistency: float
    loss: float
    ewc_loss: float
    
    # Action
    action_type: str
    action_params: Optional[Dict] = None
    
    # World state
    grid_sum: float = 0.0
    grid_density: float = 0.0
    
    # Crystallization
    is_crystallized: bool = False


class SystemLogger:
    """Logger for tracking system evolution."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.metrics_history: List[StepMetrics] = []
        self.start_time = time.time()
        
    def log_init(self, config_summary: str = ""):
        """Log system initialization."""
        if self.verbose:
            print("="*60)
            print("[INIT] Advanced AI Meta-Cognition System")
            print("="*60)
            if config_summary:
                print(config_summary)
            print("="*60)
    
    def log_step(self, metrics: StepMetrics):
        """Log a single step."""
        self.metrics_history.append(metrics)
        
        if self.verbose:
            print(f"Step {metrics.step:03d} | "
                  f"{metrics.mode:5s} | "
                  f"D:{metrics.dopamine:.2f} S:{metrics.serotonin:.2f} | "
                  f"E:{metrics.world_energy:.4f} C:{metrics.consistency:.4f} | "
                  f"L:{metrics.loss:.4f} | "
                  f"{metrics.action_type:10s} | "
                  f"Grid:{metrics.grid_sum:.1f}/{metrics.grid_density:.2%}")
    
    def log_crystallization(self, step: int):
        """Log crystallization event."""
        if self.verbose:
            print("\n" + "="*60)
            print(f"[NIRVANA] Mind crystallized at step {step}")
            print("="*60)
    
    def log_completion(self, final_metrics: StepMetrics):
        """Log final statistics."""
        elapsed = time.time() - self.start_time
        
        if self.verbose:
            print("\n" + "="*60)
            print(f"[DONE] Completed {final_metrics.step} steps in {elapsed:.2f}s")
            print(f"Final Energy: {final_metrics.world_energy:.4f}")
            print(f"Final Consistency: {final_metrics.consistency:.4f}")
            print(f"Final Grid Density: {final_metrics.grid_density:.2%}")
            print(f"Crystallized: {final_metrics.is_crystallized}")
            print("="*60)
    
    def get_summary_statistics(self) -> Dict:
        """Calculate summary statistics from metrics history."""
        if not self.metrics_history:
            return {}
        
        energies = [m.world_energy for m in self.metrics_history]
        consistencies = [m.consistency for m in self.metrics_history]
        dopamines = [m.dopamine for m in self.metrics_history]
        serotonins = [m.serotonin for m in self.metrics_history]
        
        return {
            "total_steps": len(self.metrics_history),
            "final_energy": energies[-1],
            "energy_reduction": energies[0] - energies[-1],
            "energy_reduction_pct": (energies[0] - energies[-1]) / energies[0] * 100,
            "avg_consistency": sum(consistencies) / len(consistencies),
            "final_consistency": consistencies[-1],
            "avg_dopamine": sum(dopamines) / len(dopamines),
            "avg_serotonin": sum(serotonins) / len(serotonins),
            "crystallized": self.metrics_history[-1].is_crystallized,
            "elapsed_time": time.time() - self.start_time
        }
    
    def save_metrics(self, filepath: str):
        """Save metrics history to JSON file."""
        data = {
            "summary": self.get_summary_statistics(),
            "steps": [
                {
                    "step": m.step,
                    "timestamp": m.timestamp,
                    "mode": m.mode,
                    "dopamine": m.dopamine,
                    "serotonin": m.serotonin,
                    "energy": m.world_energy,
                    "consistency": m.consistency,
                    "loss": m.loss,
                    "action_type": m.action_type,
                    "grid_sum": m.grid_sum,
                    "grid_density": m.grid_density,
                    "crystallized": m.is_crystallized
                }
                for m in self.metrics_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

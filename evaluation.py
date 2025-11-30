"""Evaluation utilities for analyzing system behavior.

Provides tools for analyzing metrics, comparing configurations,
and generating insights about system evolution.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class MetricsAnalyzer:
    """Analyzer for system metrics."""
    
    def __init__(self, metrics_path: str):
        with open(metrics_path, 'r') as f:
            self.data = json.load(f)
        
        self.steps = self.data['steps']
        self.summary = self.data['summary']
    
    def get_convergence_step(self, threshold: float = 0.3) -> int:
        """Find step where energy first drops below threshold."""
        for step_data in self.steps:
            if step_data['energy'] < threshold:
                return step_data['step']
        return -1
    
    def get_phase_transitions(self) -> List[Tuple[int, str]]:
        """Identify mode transitions (CHAOS <-> ORDER)."""
        transitions = []
        prev_mode = None
        
        for step_data in self.steps:
            mode = step_data['mode']
            if prev_mode is not None and mode != prev_mode:
                transitions.append((step_data['step'], f"{prev_mode}->{mode}"))
            prev_mode = mode
        
        return transitions
    
    def get_hormone_correlation(self) -> float:
        """Calculate correlation between dopamine and serotonin."""
        dopamines = [s['dopamine'] for s in self.steps]
        serotonins = [s['serotonin'] for s in self.steps]
        
        return np.corrcoef(dopamines, serotonins)[0, 1]
    
    def get_action_distribution(self) -> Dict[str, int]:
        """Count frequency of each action type."""
        distribution = {}
        for step_data in self.steps:
            action = step_data['action_type']
            distribution[action] = distribution.get(action, 0) + 1
        return distribution
    
    def get_learning_curve(self, window: int = 10) -> List[float]:
        """Compute smoothed learning curve."""
        energies = [s['energy'] for s in self.steps]
        smoothed = []
        
        for i in range(len(energies)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(energies[start:i+1]))
        
        return smoothed
    
    def analyze_stability(self) -> Dict:
        """Analyze system stability characteristics."""
        energies = [s['energy'] for s in self.steps]
        consistencies = [s['consistency'] for s in self.steps]
        
        # Split into early and late phases
        mid_point = len(energies) // 2
        early_energy_var = np.var(energies[:mid_point])
        late_energy_var = np.var(energies[mid_point:])
        
        return {
            "early_energy_variance": early_energy_var,
            "late_energy_variance": late_energy_var,
            "stability_improvement": (early_energy_var - late_energy_var) / early_energy_var,
            "mean_consistency_early": np.mean(consistencies[:mid_point]),
            "mean_consistency_late": np.mean(consistencies[mid_point:]),
            "consistency_improvement": (np.mean(consistencies[mid_point:]) - 
                                       np.mean(consistencies[:mid_point]))
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("="*60)
        report.append("SYSTEM ANALYSIS REPORT")
        report.append("="*60)
        
        # Summary statistics
        report.append("\n[Summary Statistics]")
        for key, value in self.summary.items():
            report.append(f"  {key}: {value}")
        
        # Convergence
        conv_step = self.get_convergence_step()
        report.append(f"\n[Convergence]")
        report.append(f"  First reached E<0.3 at step: {conv_step}")
        
        # Phase transitions
        transitions = self.get_phase_transitions()
        report.append(f"\n[Phase Transitions]")
        report.append(f"  Total transitions: {len(transitions)}")
        for step, transition in transitions[:5]:  # Show first 5
            report.append(f"    Step {step}: {transition}")
        
        # Hormone correlation
        correlation = self.get_hormone_correlation()
        report.append(f"\n[Hormone Dynamics]")
        report.append(f"  Dopamine-Serotonin correlation: {correlation:.3f}")
        
        # Action distribution
        actions = self.get_action_distribution()
        report.append(f"\n[Action Distribution]")
        for action, count in sorted(actions.items(), key=lambda x: -x[1]):
            pct = count / len(self.steps) * 100
            report.append(f"  {action}: {count} ({pct:.1f}%)")
        
        # Stability analysis
        stability = self.analyze_stability()
        report.append(f"\n[Stability Analysis]")
        for key, value in stability.items():
            report.append(f"  {key}: {value:.4f}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def compare_runs(paths: List[str]) -> str:
    """Compare multiple experimental runs.
    
    Args:
        paths: List of paths to metrics JSON files
        
    Returns:
        Comparison report as string
    """
    analyzers = [MetricsAnalyzer(path) for path in paths]
    
    report = []
    report.append("="*60)
    report.append("MULTI-RUN COMPARISON")
    report.append("="*60)
    
    # Compare key metrics
    metrics = ['final_energy', 'energy_reduction_pct', 'final_consistency', 
               'total_steps', 'crystallized']
    
    for metric in metrics:
        report.append(f"\n[{metric}]")
        for i, analyzer in enumerate(analyzers):
            value = analyzer.summary.get(metric, 'N/A')
            report.append(f"  Run {i+1}: {value}")
    
    # Compare convergence speeds
    report.append(f"\n[Convergence Speed (E<0.3)]")
    for i, analyzer in enumerate(analyzers):
        conv_step = analyzer.get_convergence_step()
        report.append(f"  Run {i+1}: {conv_step}")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py <metrics.json> [metrics2.json ...]")
        sys.exit(1)
    
    if len(sys.argv) == 2:
        # Single run analysis
        analyzer = MetricsAnalyzer(sys.argv[1])
        print(analyzer.generate_report())
    else:
        # Multi-run comparison
        print(compare_runs(sys.argv[1:]))

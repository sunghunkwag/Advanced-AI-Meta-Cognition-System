"""Visualization script for experimental results.

Generates plots for:
- Energy convergence curves per configuration
- Hormone level traces (dopamine/serotonin/cortisol)
- Action distribution histograms
- Crystallization event timeline

Usage:
    python scripts/visualize_results.py --csv experiments/results/ablation_*.csv
"""

import argparse
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_energy_curves(df, output_dir):
    """Plot energy convergence curves grouped by configuration."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = df['config'].unique()
    colors = {'A': 'red', 'B': 'orange', 'C': 'blue', 'D': 'green'}
    
    for config in sorted(configs):
        config_data = df[df['config'] == config]
        grouped = config_data.groupby('seed')['final_energy'].mean()
        
        ax.plot(grouped.index, grouped.values, 
                marker='o', label=f'Config {config}', 
                color=colors.get(config, 'gray'), linewidth=2)
    
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Final Energy', fontsize=12)
    ax.set_title('Energy Convergence by Configuration', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'energy_curves.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] {output_path}")


def plot_consistency_comparison(df, output_dir):
    """Bar plot comparing mean consistency across configurations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    grouped = df.groupby('config')['mean_consistency'].agg(['mean', 'std'])
    configs = grouped.index.tolist()
    means = grouped['mean'].values
    stds = grouped['std'].values
    
    colors = ['red', 'orange', 'blue', 'green']
    bars = ax.bar(configs, means, yerr=stds, capsize=5, color=colors[:len(configs)], alpha=0.7)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Mean Consistency Score', fontsize=12)
    ax.set_title('Consistency Comparison Across Configurations', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    output_path = output_dir / 'consistency_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] {output_path}")


def plot_action_entropy(df, output_dir):
    """Violin plot of action entropy distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    configs = sorted(df['config'].unique())
    data = [df[df['config'] == config]['action_entropy'].values for config in configs]
    
    parts = ax.violinplot(data, positions=range(len(configs)), showmeans=True, showmedians=True)
    
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([f'Config {c}' for c in configs])
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Action Entropy', fontsize=12)
    ax.set_title('Action Diversity Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    output_path = output_dir / 'action_entropy.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] {output_path}")


def plot_crystallization_events(df, output_dir):
    """Scatter plot of crystallization events across seeds and configs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = sorted(df['config'].unique())
    colors = {'A': 'red', 'B': 'orange', 'C': 'blue', 'D': 'green'}
    
    for config in configs:
        config_data = df[df['config'] == config]
        seeds = config_data['seed'].values
        crystallizations = config_data['crystallizations'].values
        
        ax.scatter(seeds, crystallizations, label=f'Config {config}', 
                   color=colors.get(config, 'gray'), s=100, alpha=0.6, edgecolors='black')
    
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Number of Crystallization Events', fontsize=12)
    ax.set_title('Crystallization Events Across Configurations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'crystallization_events.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved] {output_path}")


def generate_summary_table(df, output_dir):
    """Generate markdown summary table."""
    summary = df.groupby('config').agg({
        'final_energy': ['mean', 'std'],
        'mean_consistency': ['mean', 'std'],
        'crystallizations': 'mean',
        'action_entropy': ['mean', 'std']
    }).round(3)
    
    output_path = output_dir / 'summary_table.md'
    with open(output_path, 'w') as f:
        f.write("# Experimental Results Summary\n\n")
        f.write("| Config | Final Energy | Mean Consistency | Crystallizations | Action Entropy |\n")
        f.write("|--------|--------------|------------------|------------------|----------------|\n")
        
        for config in sorted(summary.index):
            energy_mean = summary.loc[config, ('final_energy', 'mean')]
            energy_std = summary.loc[config, ('final_energy', 'std')]
            cons_mean = summary.loc[config, ('mean_consistency', 'mean')]
            cons_std = summary.loc[config, ('mean_consistency', 'std')]
            cryst = summary.loc[config, ('crystallizations', 'mean')]
            entropy_mean = summary.loc[config, ('action_entropy', 'mean')]
            entropy_std = summary.loc[config, ('action_entropy', 'std')]
            
            f.write(f"| **{config}** | {energy_mean:.3f} ± {energy_std:.3f} | "
                    f"{cons_mean:.3f} ± {cons_std:.3f} | {cryst:.2f} | "
                    f"{entropy_mean:.3f} ± {entropy_std:.3f} |\n")
    
    print(f"[Saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize experimental results")
    parser.add_argument('--csv', required=True, help='Path to CSV file(s) (supports wildcards)')
    parser.add_argument('--output', default='experiments/visualizations', 
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Load data
    csv_files = glob.glob(args.csv)
    if not csv_files:
        print(f"[ERROR] No CSV files found matching: {args.csv}")
        return
    
    print(f"[INFO] Loading {len(csv_files)} CSV file(s)...")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"[INFO] Loaded {len(df)} rows")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n[INFO] Generating visualizations...")
    plot_energy_curves(df, output_dir)
    plot_consistency_comparison(df, output_dir)
    plot_action_entropy(df, output_dir)
    plot_crystallization_events(df, output_dir)
    generate_summary_table(df, output_dir)
    
    print(f"\n[DONE] All visualizations saved to {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()

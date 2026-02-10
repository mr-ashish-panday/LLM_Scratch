"""
NeurIPS Figure Generator
========================
Creates publication-quality figures for the ESH paper:

Figure 1: Intelligence Evolution (PPL vs α variance across training)
Figure 2: Self-Entropy Correlation (prediction uncertainty vs routing)
Figure 3: ESH vs Random Router Comparison
Figure 4: Routing Distribution Histogram

Usage:
    python plot_figures.py                          # Uses default data files
    python plot_figures.py --evolution evolution_data.json --entropy entropy_data.json
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# NeurIPS style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = {
    'esh': '#2196F3',       # Blue
    'random': '#F44336',    # Red
    'ssm': '#4CAF50',       # Green
    'attention': '#FF9800', # Orange
    'accent': '#9C27B0',    # Purple
}


def plot_intelligence_evolution(data, output_path="fig1_evolution.png"):
    """
    Figure 1: As PPL decreases, α variance increases.
    Shows the model becomes more "decisive" as it gets smarter.
    """
    steps = [d["step"] for d in data]
    ppls = [d["ppl"] for d in data]
    alpha_vars = [d["alpha_var"] for d in data]
    alpha_stds = [d["alpha_std"] for d in data]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # PPL (left axis)
    color1 = COLORS['esh']
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Perplexity (PPL)', color=color1)
    line1 = ax1.plot(steps, ppls, 'o-', color=color1, linewidth=2,
                     markersize=8, label='Perplexity', zorder=5)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)

    # α variance (right axis)
    ax2 = ax1.twinx()
    color2 = COLORS['accent']
    ax2.set_ylabel('α Variance (Routing Decisiveness)', color=color2)
    line2 = ax2.plot(steps, alpha_vars, 's--', color=color2, linewidth=2,
                     markersize=8, label='α Variance', zorder=5)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', framealpha=0.9)

    ax1.set_title('Intelligence Evolution: Smarter Models Route More Decisively')
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_entropy_correlation(data, output_path="fig2_entropy.png"):
    """
    Figure 2: Self-Entropy vs α correlation.
    Shows that uncertain tokens get more attention.
    """
    per_text = data["per_text"]
    entropies = [t["avg_entropy"] for t in per_text]
    alphas = [t["avg_alpha"] for t in per_text]
    correlation = data["correlation"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Color by complexity: simple=green, complex=orange
    colors = []
    labels_text = []
    for t in per_text:
        text = t["text"].lower()
        if any(w in text for w in ["question", "calculate", "mathematical", "algorithm", "derivative"]):
            colors.append(COLORS['attention'])
            labels_text.append('Complex')
        elif any(w in text for w in ["scientist", "ancient", "teacher", "furthermore"]):
            colors.append('#FFC107')
            labels_text.append('Medium')
        else:
            colors.append(COLORS['ssm'])
            labels_text.append('Simple')

    scatter = ax.scatter(entropies, alphas, c=colors, s=120, edgecolors='black',
                        linewidth=0.5, zorder=5, alpha=0.8)

    # Trend line
    if len(entropies) > 1:
        z = np.polyfit(entropies, alphas, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(entropies), max(entropies), 100)
        ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Token Prediction Entropy (bits)')
    ax.set_ylabel('Router α (Attention Allocation)')
    ax.set_title(f'Self-Entropy Correlation (r = {correlation:.3f})')

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['ssm'],
               markersize=10, label='Simple Text'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFC107',
               markersize=10, label='Medium Text'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['attention'],
               markersize=10, label='Complex Text'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_esh_vs_random(output_path="fig3_ablation.png"):
    """
    Figure 3: ESH vs Random Router comparison bar chart.
    Uses the empirical data from our experiments.
    """
    # Empirical results from our runs
    categories = ['Math α', 'Story α', 'Final PPL', 'α Gap (%)']
    esh_values = [0.293, 0.171, 1.27, 71.0]
    random_values = [0.499, 0.500, 1.76, 0.2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: α comparison
    ax = axes[0]
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, [0.293, 0.171], width, label='ESH (Learned)',
                   color=COLORS['esh'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [0.499, 0.500], width, label='Random Router',
                   color=COLORS['random'], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Router α (Attention Allocation)')
    ax.set_title('Routing by Prompt Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(['Math Prompt', 'Story Prompt'])
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 0.65)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    # Right: PPL and Gap comparison
    ax = axes[1]
    x = np.arange(2)
    bars1 = ax.bar(x - width/2, [1.27, 71.0], width, label='ESH (Learned)',
                   color=COLORS['esh'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [1.76, 0.2], width, label='Random Router',
                   color=COLORS['random'], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Value')
    ax.set_title('Performance & Decisiveness')
    ax.set_xticks(x)
    ax.set_xticklabels(['Final PPL ↓', 'α Gap (%) ↑'])
    ax.legend(framealpha=0.9)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('ESH vs Random Router: Ablation Study', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_routing_trajectory(output_path="fig4_trajectory.png"):
    """
    Figure 4: Routing trajectory across training phases.
    Shows how Attn% changes from Phase 1 to Phase 2.
    """
    # Phase 1 data (from training logs)
    phase1_steps = [0, 2000, 5000, 10000, 15000, 20000]
    phase1_attn = [56.8, 50.2, 48.5, 47.9, 47.9, 47.9]

    # Phase 2 data (from training logs)
    phase2_steps = [0, 2000, 5000, 10000, 15000, 20000, 25000]
    phase2_attn = [48.0, 38.0, 36.1, 36.0, 36.5, 36.7, 36.4]

    # Random baseline
    random_steps = [0, 2000, 5000, 10000, 15000, 20000, 25000]
    random_attn = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(phase1_steps, phase1_attn, 'o-', color=COLORS['ssm'],
            linewidth=2, markersize=6, label='Phase 1: TinyStories Only')
    ax.plot(phase2_steps, phase2_attn, 's-', color=COLORS['esh'],
            linewidth=2, markersize=6, label='Phase 2: Mixed Complexity')
    ax.plot(random_steps, random_attn, '^--', color=COLORS['random'],
            linewidth=2, markersize=6, label='Random Router', alpha=0.7)

    # Annotations
    ax.axhline(y=50.0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=30.0, color=COLORS['esh'], linestyle=':', alpha=0.3)
    ax.text(26000, 30.5, '30% Complex Data', fontsize=9, color=COLORS['esh'], alpha=0.7)
    ax.text(26000, 50.5, '50% Baseline', fontsize=9, color='gray', alpha=0.7)

    ax.fill_between([0, 25000], 30, 37, alpha=0.1, color=COLORS['esh'],
                    label='_nolegend_')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Attention Routing (%)')
    ax.set_title('Routing Trajectory: ESH Learns to Match Data Complexity')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(25, 60)
    ax.set_xlim(-500, 27000)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS Figures")
    parser.add_argument("--evolution", type=str, default="evolution_data.json")
    parser.add_argument("--entropy", type=str, default="entropy_data.json")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating NeurIPS Figures")
    print("=" * 60)

    # Figure 1: Intelligence Evolution
    if Path(args.evolution).exists():
        print("\n[Figure 1] Intelligence Evolution")
        with open(args.evolution) as f:
            evolution_data = json.load(f)
        plot_intelligence_evolution(evolution_data,
                                   str(output_dir / "fig1_evolution.png"))
    else:
        print(f"\n[Figure 1] Skipped (run analyze_evolution.py first)")

    # Figure 2: Entropy Correlation
    if Path(args.entropy).exists():
        print("\n[Figure 2] Self-Entropy Correlation")
        with open(args.entropy) as f:
            entropy_data = json.load(f)
        plot_entropy_correlation(entropy_data,
                                str(output_dir / "fig2_entropy.png"))
    else:
        print(f"\n[Figure 2] Skipped (run analyze_entropy.py first)")

    # Figure 3: ESH vs Random (hardcoded from experiments)
    print("\n[Figure 3] ESH vs Random Router Ablation")
    plot_esh_vs_random(str(output_dir / "fig3_ablation.png"))

    # Figure 4: Routing Trajectory (hardcoded from logs)
    print("\n[Figure 4] Routing Trajectory")
    plot_routing_trajectory(str(output_dir / "fig4_trajectory.png"))

    print(f"\n{'=' * 60}")
    print(f"All figures saved to {output_dir}/")
    print("Ready for NeurIPS submission!")


if __name__ == "__main__":
    main()

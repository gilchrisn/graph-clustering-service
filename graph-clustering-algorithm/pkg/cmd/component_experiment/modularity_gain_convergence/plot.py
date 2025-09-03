#!/usr/bin/env python3
"""
Gain Distribution Experiment Visualizer

Reads distance_table_*.json files and creates plots showing:
- Distance metric evolution over moves
- SCAR convergence to Louvain as k increases
- Distribution similarity analysis
- Aggregated metrics comparison

Requirements: pip install matplotlib numpy seaborn
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set1")  # More contrasting colors than viridis

# Simple high-contrast colors
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3', '#ffff33', '#a65628', '#f781bf']

def load_experiment_data():
    """Load experiment data from fixed filename."""
    try:
        with open('gain_distribution_experiment.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: gain_distribution_experiment.json not found!")
        print("Make sure you ran the Go gain distribution experiment first.")
        exit(1)

def load_distance_tables():
    """Load distance tables for different metrics."""
    tables = {}
    metrics = ['kl_divergence', 'jensen_shannon', 'wasserstein_p1', 'cosine_similarity']
    
    for metric in metrics:
        filename = f'distance_table_{metric}.json'
        try:
            with open(filename, 'r') as f:
                tables[metric] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping {metric}")
    
    return tables

def plot_distance_evolution(tables):
    """Plot distance metric evolution over moves for all k values."""
    print("Creating distance evolution plots...")
    
    # Create subplot for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metric_names = ['kl_divergence', 'jensen_shannon', 'wasserstein_p1', 'cosine_similarity']
    metric_labels = ['KL Divergence', 'Jensen-Shannon', 'Wasserstein P1', 'Cosine Similarity']
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        
        if metric not in tables:
            ax.text(0.5, 0.5, f'No data for {label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue
        
        table = tables[metric]
        k_values = [info['k_value'] for info in table['k_value_info']]
        sample_moves = table['sample_moves']
        distance_data = table['distance_data']
        
        # Plot each k value
        for k_idx, k in enumerate(k_values):
            values = distance_data[k_idx]
            
            # Filter out invalid values
            valid_moves = []
            valid_values = []
            for move, value in zip(sample_moves, values):
                if not (np.isnan(value) or np.isinf(value)):
                    valid_moves.append(move)
                    valid_values.append(value)
            
            if len(valid_values) > 0:
                color = COLORS[k_idx % len(COLORS)]
                ax.plot(valid_moves, valid_values, '--', linewidth=2, 
                       label=f'k={k}', alpha=0.8, color=color)
        
        ax.set_xlabel('Move Number')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Evolution Over Moves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Special handling for cosine similarity (higher is better)
        if metric == 'cosine_similarity':
            ax.set_ylabel(f'{label} (↑ better)')
        else:
            ax.set_ylabel(f'{label} (↓ better)')
    
    plt.tight_layout()
    plt.savefig('gain_distance_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence_analysis(data, tables):
    """Plot SCAR convergence analysis."""
    print("Creating convergence analysis...")
    
    aggregated = data['aggregated_metrics']
    k_values = sorted([int(k) for k in aggregated.keys()])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Extract aggregated metrics
    kl_divs = [aggregated[str(k)]['avg_kl_divergence'] for k in k_values]
    js_divs = [aggregated[str(k)]['avg_jensen_shannon'] for k in k_values]
    ws_divs = [aggregated[str(k)]['avg_wasserstein_p1'] for k in k_values]
    cos_sims = [aggregated[str(k)]['avg_cosine_similarity'] for k in k_values]
    
    # Plot 1: Final distances vs k
    ax1 = axes[0, 0]
    ax1.plot(k_values, kl_divs, 'o-', linewidth=2, markersize=6, 
             label='KL Divergence', color=COLORS[0])
    ax1.plot(k_values, js_divs, 's-', linewidth=2, markersize=6, 
             label='Jensen-Shannon', color=COLORS[1])
    ax1.plot(k_values, ws_divs, '^-', linewidth=2, markersize=6, 
             label='Wasserstein P1', color=COLORS[2])
    
    ax1.set_xlabel('k (SCAR parameter)')
    ax1.set_ylabel('Average Distance')
    ax1.set_title('Distance Metrics vs k (Lower = Better Convergence)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cosine similarity vs k
    ax2 = axes[0, 1]
    ax2.plot(k_values, cos_sims, 'o-', linewidth=2, markersize=6, color=COLORS[3])
    ax2.axhline(y=1.0, color=COLORS[0], linestyle='--', alpha=0.7, label='Perfect Similarity')
    
    ax2.set_xlabel('k (SCAR parameter)')
    ax2.set_ylabel('Average Cosine Similarity')
    ax2.set_title('Cosine Similarity vs k (Higher = Better Convergence)')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Standard deviations
    ax3 = axes[1, 0]
    kl_stds = [aggregated[str(k)]['std_kl_divergence'] for k in k_values]
    js_stds = [aggregated[str(k)]['std_jensen_shannon'] for k in k_values]
    ws_stds = [aggregated[str(k)]['std_wasserstein_p1'] for k in k_values]
    
    ax3.plot(k_values, kl_stds, 'o-', linewidth=2, markersize=6, 
             label='KL Divergence Std', color=COLORS[0])
    ax3.plot(k_values, js_stds, 's-', linewidth=2, markersize=6, 
             label='Jensen-Shannon Std', color=COLORS[1])
    ax3.plot(k_values, ws_stds, '^-', linewidth=2, markersize=6, 
             label='Wasserstein P1 Std', color=COLORS[2])
    
    ax3.set_xlabel('k (SCAR parameter)')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Distance Variability')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final distance comparison from tables 
    ax4 = axes[1, 1]
    if 'kl_divergence' in tables:
        table = tables['kl_divergence']
        final_distances = [info['final_distance'] for info in table['k_value_info']]
        improvements = [info['improvement'] for info in table['k_value_info']]
        
        ax4.scatter(k_values, final_distances, s=100, alpha=0.7, 
                   c=improvements, cmap='RdYlGn', edgecolors='black')
        ax4.set_xlabel('k (SCAR parameter)')
        ax4.set_ylabel('Final KL Divergence')
        ax4.set_title('Final Gain Distribution Quality')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(ax4.collections[0], ax=ax4, label='Improvement')
    
    plt.tight_layout()
    plt.savefig('gain_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_final_comparison(data):
    """Plot final distance comparison across k values."""
    print("Creating final comparison plot...")
    
    aggregated = data['aggregated_metrics']
    k_values = sorted([int(k) for k in aggregated.keys()])
    
    # Create bar plots for final distances
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    metrics = ['avg_kl_divergence', 'avg_jensen_shannon', 'avg_wasserstein_p1', 'avg_cosine_similarity']
    metric_labels = ['KL Divergence', 'Jensen-Shannon', 'Wasserstein P1', 'Cosine Similarity']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        values = [aggregated[str(k)][metric] for k in k_values]
        x_pos = np.arange(len(k_values))
        
        # Color bars by convergence quality
        if metric == 'avg_cosine_similarity':
            # Higher is better for cosine similarity
            colors = plt.cm.RdYlGn([v for v in values])
        else:
            # Lower is better for distance metrics
            max_val = max(values) if values else 1
            colors = plt.cm.RdYlGn_r([v/max_val for v in values])
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('k value')
        ax.set_ylabel(label)
        ax.set_title(f'Final {label} by k')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'k={k}' for k in k_values], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gain_final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_heatmap_analysis(tables):
    """Create heatmap showing distance patterns."""
    print("Creating heatmap analysis...")
    
    # Create heatmap for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    heatmap_metrics = ['kl_divergence', 'jensen_shannon', 'wasserstein_p1', 'cosine_similarity']
    titles = ['KL Divergence', 'Jensen-Shannon', 'Wasserstein P1', 'Cosine Similarity']
    
    for idx, (metric, title) in enumerate(zip(heatmap_metrics, titles)):
        ax = axes[idx]
        
        if metric not in tables:
            ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        table = tables[metric]
        k_values = [info['k_value'] for info in table['k_value_info']]
        sample_moves = table['sample_moves']
        distance_data = table['distance_data']
        
        # Prepare data matrix: k_values x samples
        matrix = np.array(distance_data)
        
        # Handle infinite/NaN values
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=np.nanmax(matrix[np.isfinite(matrix)]), 
                              neginf=np.nanmin(matrix[np.isfinite(matrix)]))
        
        # Create heatmap
        if metric == 'cosine_similarity':
            cmap = 'RdYlGn'  # Higher is better
        else:
            cmap = 'RdYlGn_r'  # Lower is better
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # Set labels
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('k value')
        ax.set_title(f'{title} Heatmap')
        
        # Set ticks
        sample_ticks = np.linspace(0, len(sample_moves)-1, min(10, len(sample_moves)), dtype=int)
        ax.set_xticks(sample_ticks)
        ax.set_xticklabels([f'{sample_moves[i]}' for i in sample_ticks])
        
        k_ticks = np.arange(len(k_values))
        ax.set_yticks(k_ticks)
        ax.set_yticklabels([f'{k}' for k in k_values])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('gain_heatmap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_distribution_examples(data):
    """Plot example gain distributions for specific moves."""
    print("Creating distribution examples...")
    
    samples = data.get('samples', [])
    if len(samples) == 0:
        print("No samples found for distribution examples")
        return
    
    # Select a few interesting samples (early, middle, late)
    sample_indices = [0, len(samples)//2, len(samples)-1] if len(samples) >= 3 else [0]
    
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(14, 4*len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]
    
    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = axes[plot_idx]
        sample = samples[sample_idx]
        
        louvain_gains = sample['louvain_gains']
        scar_gains = sample.get('scar_gains', {})
        
        # Plot Louvain distribution
        communities = list(range(len(louvain_gains)))
        ax.plot(communities, louvain_gains, 'k-', linewidth=3, label='Louvain', alpha=0.8)
        
        # Plot a few SCAR distributions
        k_to_plot = [2, 16, 128] if all(str(k) in scar_gains for k in [2, 16, 128]) else list(scar_gains.keys())[:3]
        
        for i, k in enumerate(k_to_plot):
            k_str = str(k)
            if k_str in scar_gains:
                gains = scar_gains[k_str]
                if len(gains) == len(louvain_gains):
                    color = COLORS[(i+1) % len(COLORS)]
                    ax.plot(communities, gains, '--', linewidth=2, 
                           label=f'SCAR k={k}', alpha=0.7, color=color)
        
        ax.set_xlabel('Community ID')
        ax.set_ylabel('Modularity Gain')
        ax.set_title(f'Gain Distributions - Move {sample["move_id"]} (Node {sample["node"]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gain_distribution_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_accuracy_vs_k(data, tables):
    """Plot accuracy metrics vs k to show convergence quality."""
    print("Creating accuracy vs k analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Distance metrics vs k (log scale)
    ax1 = axes[0, 0]
    aggregated = data['aggregated_metrics']
    k_values = sorted([int(k) for k in aggregated.keys()])
    
    kl_divs = [aggregated[str(k)]['avg_kl_divergence'] for k in k_values]
    js_divs = [aggregated[str(k)]['avg_jensen_shannon'] for k in k_values]
    
    ax1.loglog(k_values, kl_divs, 'o-', linewidth=2, markersize=6, 
               label='KL Divergence', color=COLORS[0])
    ax1.loglog(k_values, js_divs, 's-', linewidth=2, markersize=6, 
               label='Jensen-Shannon', color=COLORS[1])
    
    ax1.set_xlabel('k (SCAR parameter)')
    ax1.set_ylabel('Distance (log scale)')
    ax1.set_title('Gain Distribution Error vs k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Correlation analysis
    ax2 = axes[0, 1]
    correlations = [aggregated[str(k)]['avg_pearson_corr'] for k in k_values]
    
    ax2.semilogx(k_values, correlations, 'o-', linewidth=2, markersize=6, color=COLORS[2])
    ax2.axhline(y=1.0, color=COLORS[3], linestyle='--', alpha=0.7, label='Perfect Correlation')
    ax2.set_xlabel('k (SCAR parameter)')
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Gain Distribution Correlation vs k')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convergence rate analysis
    ax3 = axes[1, 0]
    if 'kl_divergence' in tables:
        table = tables['kl_divergence']
        improvements = [info['improvement'] for info in table['k_value_info']]
        avg_distances = [info['avg_distance'] for info in table['k_value_info']]
        
        ax3.scatter(k_values, improvements, s=100, alpha=0.7, 
                   c=avg_distances, cmap='RdYlBu_r', edgecolors='black')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('k (SCAR parameter)')
        ax3.set_ylabel('Distance Improvement')
        ax3.set_title('Gain Distribution Improvement')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        plt.colorbar(ax3.collections[0], ax=ax3, label='Avg Distance')
    
    # Plot 4: Efficiency frontier
    ax4 = axes[1, 1]
    efficiency_scores = []
    for k in k_values:
        # Efficiency = 1 / (k * avg_distance)
        avg_dist = aggregated[str(k)]['avg_kl_divergence']
        efficiency = 1.0 / (k * avg_dist) if avg_dist > 0 else 0
        efficiency_scores.append(efficiency)
    
    ax4.semilogx(k_values, efficiency_scores, 'o-', linewidth=2, markersize=6, color=COLORS[4])
    ax4.set_xlabel('k (SCAR parameter)')
    ax4.set_ylabel('Efficiency Score (1 / k·distance)')
    ax4.set_title('Gain Distribution Efficiency')
    ax4.grid(True, alpha=0.3)
    
    # Highlight the best efficiency point
    best_idx = np.argmax(efficiency_scores)
    ax4.scatter(k_values[best_idx], efficiency_scores[best_idx], 
               s=200, color=COLORS[5], marker='*', zorder=5, 
               label=f'Best k={k_values[best_idx]}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('gain_accuracy_vs_k.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("GAIN DISTRIBUTION EXPERIMENT SUMMARY")
    print("="*70)
    
    config = data['config']
    graph_info = data['graph_info']
    aggregated = data['aggregated_metrics']
    
    print(f"Graph: {graph_info['num_nodes']} nodes, {graph_info['total_weight']:.2f} total weight")
    print(f"Samples: {len(data['samples'])} (every {config['sample_interval']} moves)")
    
    k_values = sorted([int(k) for k in aggregated.keys()])
    print(f"K values tested: {k_values}")
    
    print("\nFINAL AVERAGE DISTANCES BY K:")
    print(f"{'K':>8s} {'KL-Div':>10s} {'Jensen-S':>10s} {'Wasserst':>10s} {'Cosine':>10s} {'Pearson':>10s}")
    print("-" * 75)
    
    for k in k_values:
        metrics = aggregated[str(k)]
        print(f"{k:>8d} {metrics['avg_kl_divergence']:>10.6f} {metrics['avg_jensen_shannon']:>10.6f} "
              f"{metrics['avg_wasserstein_p1']:>10.6f} {metrics['avg_cosine_similarity']:>10.6f} ")
    
    # Find best k values
    best_kl_k = min(k_values, key=lambda k: aggregated[str(k)]['avg_kl_divergence'])
    best_cos_k = max(k_values, key=lambda k: aggregated[str(k)]['avg_cosine_similarity'])
    
    print(f"\nBEST PERFORMING K VALUES:")
    print(f"  Lowest KL divergence: k={best_kl_k} (distance: {aggregated[str(best_kl_k)]['avg_kl_divergence']:.6f})")
    print(f"  Highest cosine similarity: k={best_cos_k} (similarity: {aggregated[str(best_cos_k)]['avg_cosine_similarity']:.6f})")
    
    print("\n💡 INTERPRETATION:")
    print("   • Lower KL/Jensen-Shannon/Wasserstein = SCAR gain distributions closer to Louvain")
    print("   • Higher Cosine/Pearson = SCAR gain distributions more similar to Louvain")
    print("   • Expected: distances → 0, similarities → 1 as k → ∞")
    print("   • If distances don't decrease with k, investigate sketch approximation quality")
    print("="*70)

def main():
    print("Loading gain distribution experiment results...")
    data = load_experiment_data()
    tables = load_distance_tables()
    
    print_summary(data)
    
    # Create all plots
    plot_distance_evolution(tables)
    # plot_convergence_analysis(data, tables)
    # plot_final_comparison(data)
    # plot_heatmap_analysis(tables)
    # plot_distribution_examples(data)
    # plot_accuracy_vs_k(data, tables)
    
    print("\nDone! Check these files:")
    print("  - gain_distance_evolution.png")
    print("  - gain_convergence_analysis.png") 
    print("  - gain_final_comparison.png")
    print("  - gain_heatmap_analysis.png")
    print("  - gain_distribution_examples.png")
    print("  - gain_accuracy_vs_k.png")

if __name__ == "__main__":
    main()
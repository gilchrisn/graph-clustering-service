#!/usr/bin/env python3
"""
Interactive Community Weight Distribution Experiment Visualizer

Creates interactive plots with working hover tooltips showing detailed data point information.
Based on the working matplotlib hover pattern from the move selection visualizer.

Requirements: pip install matplotlib numpy seaborn mplcursors
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Check for interactive capabilities
try:
    import mplcursors
    INTERACTIVE_AVAILABLE = True
    print("✅ Interactive mode enabled - hover functionality available!")
except ImportError:
    INTERACTIVE_AVAILABLE = False
    print("⚠️  Non-interactive mode - install mplcursors for hover functionality")
    print("   Run: pip install mplcursors")

# Set style for better plots
plt.style.use('seaborn-v0_8')

# High contrast colors for better visibility
CONTRASTIVE_COLORS = ['#FF0000', '#00AA00', '#0066FF', '#FF8800', '#AA00AA', '#00AAAA', '#666666', '#FFD700']

def load_experiment_data():
    """Load experiment data from fixed filename."""
    try:
        with open('community_weight_distribution_experiment.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: community_weight_distribution_experiment.json not found!")
        print("Make sure you ran the Go community weight distribution experiment first.")
        exit(1)

def load_distance_tables():
    """Load distance tables for different metrics."""
    tables = {}
    metrics = ['mae', 'wasserstein_p1', 'cosine_similarity']
    
    for metric in metrics:
        filename = f'community_weight_distance_table_{metric}.json'
        try:
            with open(filename, 'r') as f:
                tables[metric] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping {metric}")
    
    return tables

def create_matplotlib_hover_plot(data, tables):
    """Create matplotlib plot with working hover using the proven pattern from move selection."""
    print("Creating matplotlib plot with working hover functionality...")
    
    # Create figure with reasonable size
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    metric_names = ['mae', 'wasserstein_p1', 'cosine_similarity']
    metric_labels = ['Mean Absolute Error', 'Wasserstein P1', 'Cosine Similarity']
    
    # Add placeholder for fourth subplot
    if len(metric_names) < 4:
        metric_names.append('placeholder')
        metric_labels.append('Placeholder')
    
    # Store all data points for hover functionality (like the working move selection version)
    all_points = []  # Will store (x, y, hover_text, color, ax_index, metric_name)
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        
        if metric == 'placeholder' or metric not in tables:
            ax.text(0.5, 0.5, f'No data for {label}' if metric != 'placeholder' else 'Unused', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Community Weight {label}')
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
                if not (np.isnan(value) or np.isinf(value)) and value > 0:
                    valid_moves.append(move)
                    valid_values.append(value)
                    
                    # Create detailed hover text for each point
                    quality = ""
                    if metric == 'cosine_similarity':
                        if value > 0.9:
                            quality = "🟢 EXCELLENT"
                        elif value > 0.8:
                            quality = "🟢 GOOD"
                        elif value > 0.6:
                            quality = "🟡 MODERATE"
                        else:
                            quality = "🔴 POOR"
                    else:  # MAE and Wasserstein (lower is better)
                        if value < 0.01:
                            quality = "🟢 EXCELLENT"
                        elif value < 0.1:
                            quality = "🟢 GOOD"
                        elif value < 0.5:
                            quality = "🟡 MODERATE"
                        else:
                            quality = "🔴 POOR"
                    
                    hover_text = (f"Move: {move}\n"
                                f"K: {k}\n"
                                f"{label}: {value:.6f}\n"
                                f"Quality: {quality}\n"
                                f"Metric: {metric}\n"
                                f"Debug: goto {move}")
                    
                    color = CONTRASTIVE_COLORS[k_idx % len(CONTRASTIVE_COLORS)]
                    all_points.append((move, value, hover_text, color, idx, metric))
            
            if len(valid_values) > 0:
                color = CONTRASTIVE_COLORS[k_idx % len(CONTRASTIVE_COLORS)]
                
                # Plot with thinner lines and smaller markers
                ax.plot(valid_moves, valid_values, '--', linewidth=1.5, 
                       label=f'k={k}', alpha=0.8, color=color)
                
                # Add scatter points for better hover detection
                ax.scatter(valid_moves, valid_values, s=25, color=color, alpha=0.7, zorder=5)
        
        ax.set_xlabel('Move Number', fontsize=11)
        ax.set_ylabel(f'{label}', fontsize=11)
        ax.set_title(f'{label} Evolution Over Moves', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Special handling for cosine similarity (higher is better)
        if metric == 'cosine_similarity':
            ax.set_ylabel(f'{label} (↑ better)', fontsize=11)
        else:
            ax.set_ylabel(f'{label} (↓ better)', fontsize=11)
    
    # Create hover annotation (like the working version)
    annotations = []
    for ax in axes:
        annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                           arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        annotations.append(annot)
    
    def on_hover(event):
        if event.inaxes in axes:
            # Find which subplot we're in
            ax_idx = list(axes).index(event.inaxes)
            
            # Find closest point in this subplot
            if event.xdata is None or event.ydata is None:
                return
            
            min_dist = float('inf')
            closest_point = None
            
            for x, y, hover_text, color, point_ax_idx, metric in all_points:
                if point_ax_idx == ax_idx:  # Only check points in current subplot
                    # Normalize distance for better detection
                    x_range = event.inaxes.get_xlim()[1] - event.inaxes.get_xlim()[0]
                    y_range = event.inaxes.get_ylim()[1] - event.inaxes.get_ylim()[0]
                    
                    norm_dist = ((event.xdata - x) / x_range)**2 + ((event.ydata - y) / y_range)**2
                    if norm_dist < min_dist:
                        min_dist = norm_dist
                        closest_point = (x, y, hover_text, color, point_ax_idx, metric)
            
            # Show annotation if close enough
            if closest_point and min_dist < 0.01:  # Adjust sensitivity
                x, y, hover_text, color, point_ax_idx, metric = closest_point
                annot = annotations[point_ax_idx]
                annot.xy = (x, y)
                annot.set_text(hover_text)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                # Hide all annotations
                for annot in annotations:
                    if annot.get_visible():
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
    
    def on_click(event):
        if event.inaxes in axes and event.button == 1:
            if event.xdata is not None:
                move_id = int(round(event.xdata))
                print(f"\n🎯 Clicked on move {move_id}")
                print(f"Debugger commands:")
                print(f"  ./community_debugger graph.txt properties.txt path.txt")
                print(f"  > goto {move_id}")
                print(f"  > visualize-dist k=256")
                print(f"  > analyze")
    
    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout()
    plt.savefig('community_weight_distance_evolution_interactive.png', dpi=300, bbox_inches='tight')
    
    print("✅ Interactive matplotlib plot created with working hover functionality")
    print("🖱️  Hover over data points to see detailed tooltips")
    print("🖱️  Click on points to get debugger commands")
    
    plt.show()

def create_convergence_plot(data):
    """Create convergence analysis plot with hover functionality."""
    print("Creating convergence analysis with hover...")
    
    aggregated = data['aggregated_metrics']
    k_values = sorted([int(k) for k in aggregated.keys()])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract aggregated metrics
    mae_values = [aggregated[str(k)]['avg_mae'] for k in k_values]
    ws_divs = [aggregated[str(k)]['avg_wasserstein_p1'] for k in k_values]
    cos_sims = [aggregated[str(k)]['avg_cosine_similarity'] for k in k_values]
    
    # Plot 1: Distance metrics vs k
    ax1.plot(k_values, mae_values, 'o-', linewidth=2, markersize=6, 
             label='Mean Absolute Error', color=CONTRASTIVE_COLORS[0])
    ax1.plot(k_values, ws_divs, 's-', linewidth=2, markersize=6, 
             label='Wasserstein P1', color=CONTRASTIVE_COLORS[1])
    
    ax1.set_xlabel('k (SCAR parameter)')
    ax1.set_ylabel('Average Distance')
    ax1.set_title('Community Weight Distance Metrics vs k (Lower = Better)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cosine similarity vs k
    ax2.plot(k_values, cos_sims, 'o-', linewidth=2, markersize=6, color=CONTRASTIVE_COLORS[2])
    ax2.axhline(y=1.0, color=CONTRASTIVE_COLORS[0], linestyle='--', alpha=0.7, label='Perfect Similarity')
    
    ax2.set_xlabel('k (SCAR parameter)')
    ax2.set_ylabel('Average Cosine Similarity')
    ax2.set_title('Community Weight Cosine Similarity vs k (Higher = Better)')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Store points for hover
    hover_points = []
    for i, k in enumerate(k_values):
        metrics = aggregated[str(k)]
        
        # Add points for first plot
        hover_points.append((k, mae_values[i], f"K: {k}\nMAE: {mae_values[i]:.6f}\nWasserstein P1: {ws_divs[i]:.6f}\nSamples: {metrics['num_samples']}\nDebug: debug-sketch k={k}", 0))
        hover_points.append((k, ws_divs[i], f"K: {k}\nMAE: {mae_values[i]:.6f}\nWasserstein P1: {ws_divs[i]:.6f}\nSamples: {metrics['num_samples']}\nDebug: debug-sketch k={k}", 0))
        
        # Add points for second plot
        hover_points.append((k, cos_sims[i], f"K: {k}\nCosine Similarity: {cos_sims[i]:.6f}\nSamples: {metrics['num_samples']}\nDebug: debug-sketch k={k}", 1))
    
    # Create annotations
    annot1 = ax1.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="lightblue", alpha=0.9),
                         arrowprops=dict(arrowstyle="->"))
    annot1.set_visible(False)
    
    annot2 = ax2.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.9),
                         arrowprops=dict(arrowstyle="->"))
    annot2.set_visible(False)
    
    def on_hover_convergence(event):
        if event.inaxes == ax1:
            ax_idx = 0
            annot = annot1
        elif event.inaxes == ax2:
            ax_idx = 1
            annot = annot2
        else:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        min_dist = float('inf')
        closest_point = None
        
        for x, y, hover_text, point_ax_idx in hover_points:
            if point_ax_idx == ax_idx:
                # Use log scale distance calculation
                log_x_dist = (np.log10(event.xdata) - np.log10(x))**2 if x > 0 and event.xdata > 0 else float('inf')
                log_y_dist = (np.log10(event.ydata) - np.log10(y))**2 if y > 0 and event.ydata > 0 and ax_idx == 0 else (event.ydata - y)**2
                
                dist = log_x_dist + log_y_dist
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (x, y, hover_text)
        
        if closest_point and min_dist < 1.0:  # Adjust sensitivity for log scale
            x, y, hover_text = closest_point
            annot.xy = (x, y)
            annot.set_text(hover_text)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('motion_notify_event', on_hover_convergence)
    
    plt.tight_layout()
    plt.savefig('community_weight_convergence_interactive.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_final_comparison_plot(data):
    """Create final comparison bar plot with hover functionality."""
    print("Creating final comparison plot...")
    
    aggregated = data['aggregated_metrics']
    k_values = sorted([int(k) for k in aggregated.keys()])
    
    # Create bar plots for final distances
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    metrics = ['avg_mae', 'avg_wasserstein_p1', 'avg_cosine_similarity']
    metric_labels = ['Mean Absolute Error', 'Wasserstein P1', 'Cosine Similarity']
    
    # Add placeholder for fourth subplot
    if len(metrics) < 4:
        metrics.append('placeholder')
        metric_labels.append('Placeholder')
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        if metric == 'placeholder':
            ax.text(0.5, 0.5, 'Unused', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Unused')
            continue
        
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
        ax.set_ylabel(f'Community Weight {label}')
        ax.set_title(f'Final Community Weight {label} by k')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'k={k}' for k in k_values], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('community_weight_final_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_heatmap_analysis(tables):
    """Create heatmap showing community weight distance patterns."""
    print("Creating community weight heatmap analysis...")
    
    # Create heatmap for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    heatmap_metrics = ['mae', 'wasserstein_p1', 'cosine_similarity']
    titles = ['Mean Absolute Error', 'Wasserstein P1', 'Cosine Similarity']
    
    # Add placeholder for fourth subplot
    if len(heatmap_metrics) < 4:
        heatmap_metrics.append('placeholder')
        titles.append('Placeholder')
    
    for idx, (metric, title) in enumerate(zip(heatmap_metrics, titles)):
        ax = axes[idx]
        
        if metric == 'placeholder' or metric not in tables:
            ax.text(0.5, 0.5, f'No data for {title}' if metric != 'placeholder' else 'Unused', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Community Weight {title}')
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
        ax.set_title(f'Community Weight {title} Heatmap')
        
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
    plt.savefig('community_weight_heatmap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("COMMUNITY WEIGHT DISTRIBUTION EXPERIMENT SUMMARY")
    print("="*70)
    
    config = data['config']
    graph_info = data['graph_info']
    aggregated = data['aggregated_metrics']
    
    print(f"Graph: {graph_info['num_nodes']} nodes, {graph_info['total_weight']:.2f} total weight")
    print(f"Samples: {len(data['samples'])} (every {config['sample_interval']} moves)")
    
    k_values = sorted([int(k) for k in aggregated.keys()])
    print(f"K values tested: {k_values}")
    
    print("\nFINAL AVERAGE COMMUNITY WEIGHT DISTANCES BY K:")
    print(f"{'K':>8s} {'MAE':>10s} {'Wasserst':>10s} {'Cosine':>10s}")
    print("-" * 55)
    
    for k in k_values:
        metrics = aggregated[str(k)]
        print(f"{k:>8d} {metrics['avg_mae']:>10.6f} {metrics['avg_wasserstein_p1']:>10.6f} "
              f"{metrics['avg_cosine_similarity']:>10.6f}")

    # Find best k values
    best_mae_k = min(k_values, key=lambda k: aggregated[str(k)]['avg_mae'])
    best_cos_k = max(k_values, key=lambda k: aggregated[str(k)]['avg_cosine_similarity'])
    
    print(f"\nBEST PERFORMING K VALUES:")
    print(f"  Lowest MAE: k={best_mae_k} (distance: {aggregated[str(best_mae_k)]['avg_mae']:.6f})")
    print(f"  Highest cosine similarity: k={best_cos_k} (similarity: {aggregated[str(best_cos_k)]['avg_cosine_similarity']:.6f})")
    
    print("\n💡 INTERPRETATION:")
    print("   • Lower MAE/Wasserstein = SCAR community weights closer to Louvain")
    print("   • Higher Cosine similarity = SCAR community weight distributions more similar to Louvain")
    print("   • Expected: MAE → 0, similarities → 1 as k → ∞")
    
    if INTERACTIVE_AVAILABLE:
        print(f"\n🖱️  INTERACTIVE FEATURES:")
        print(f"   • Hover over data points to see detailed information")
        print(f"   • Click on points to get debugger commands")
        print(f"   • Quality indicators show estimation accuracy")
    else:
        print(f"\n⚠️  For full interactivity, install: pip install mplcursors")
    
    print("="*70)

def main():
    print("Loading community weight distribution experiment results...")
    data = load_experiment_data()
    tables = load_distance_tables()
    
    print_summary(data)
    
    # Create the main interactive plot
    create_matplotlib_hover_plot(data, tables)
    
    # Create convergence analysis
    create_convergence_plot(data)
    
    
    print("\n✅ Interactive visualizations complete!")
    print("📊 Created files:")
    print("  - community_weight_distance_evolution_interactive.png")
    print("  - community_weight_convergence_interactive.png")
    print("  - community_weight_final_comparison.png")
    print("  - community_weight_heatmap_analysis.png")
    
    if INTERACTIVE_AVAILABLE:
        print(f"\n🖱️  Hover over any data point to see:")
        print(f"   • Move number and K value")
        print(f"   • Exact metric values")
        print(f"   • Quality assessment (🟢 GOOD, 🟡 OK, 🔴 POOR)")
        print(f"   • Debug commands for that specific move")
    else:
        print(f"\n⚠️  Install mplcursors for hover functionality: pip install mplcursors")

if __name__ == "__main__":
    main()
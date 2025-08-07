#!/usr/bin/env python3
"""
Interactive Move Selection Quality Experiment Visualizer

Creates truly interactive plots with reliable hover tooltips using matplotlib's built-in functionality.
Also includes a Plotly version that's guaranteed to work.

Requirements: pip install matplotlib numpy plotly
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Use high contrast colors
CONTRASTIVE_COLORS = ['#FF0000', '#00AA00', '#0066FF', '#FF8800', '#AA00AA', '#00AAAA', '#666666', '#FFD700']

def load_experiment_data():
    """Load aggregated experiment data from fixed filename."""
    try:
        with open('move_selection_aggregated_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: move_selection_aggregated_results.json not found!")
        print("Make sure you ran the Go move selection quality experiment first.")
        exit(1)

def create_plotly_interactive_plot(data):
    """Create a fully interactive plot using Plotly (guaranteed to work)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.offline as pyo
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")
        return
    
    print("Creating Plotly interactive visualization...")
    
    all_samples = data.get('all_samples', [])
    config = data['config']
    k_values = config['KValues']
    
    if not all_samples:
        print("No sample data available")
        return
    
    # Group samples by move ID
    move_samples = {}
    for sample in all_samples:
        move_id = sample['move_id']
        if move_id not in move_samples:
            move_samples[move_id] = sample
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Target Community Ranking Over Moves', 'Aggregated Quality Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # Plot 1: Ranking evolution
    sorted_moves = sorted(move_samples.keys())
    if len(sorted_moves) > 300:  # Limit for performance
        step = len(sorted_moves) // 300
        sorted_moves = sorted_moves[::step]
    
    colors_plotly = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'gray', 'gold']
    
    for k_idx, k in enumerate(k_values):
        moves = []
        ranks = []
        hover_texts = []
        perfect_moves = []
        perfect_hover = []
        
        for move_id in sorted_moves:
            sample = move_samples[move_id]
            scar_rankings = sample.get('scar_rankings', {})
            
            if str(k) in scar_rankings:
                ranking = scar_rankings[str(k)]
                if ranking and ranking.get('target_rank', 0) > 0:
                    moves.append(move_id)
                    rank = ranking['target_rank']
                    ranks.append(rank)
                    
                    # Create detailed hover text
                    hover_text = (f"<b>Move {move_id}</b><br>"
                                f"Node: {sample.get('node', 'N/A')}<br>"
                                f"Target Community: {sample.get('louvain_target_community', 'N/A')}<br>"
                                f"Louvain Gain: {sample.get('louvain_gain', 0):.6f}<br>"
                                f"K: {k}<br>"
                                f"SCAR Rank: {rank}/{len(ranking.get('communities', []))}<br>"
                                f"Target Gain: {ranking.get('target_gain', 0):.6f}<br>"
                                f"Best Gain: {ranking.get('best_gain', 0):.6f}<br>"
                                f"<i>Debug: goto {move_id}</i>")
                    hover_texts.append(hover_text)
                    
                    if rank == 1:
                        perfect_moves.append(move_id)
                        perfect_hover.append(hover_text)
        
        if moves and ranks:
            color = colors_plotly[k_idx % len(colors_plotly)]
            
            # Add main line
            fig.add_trace(
                go.Scatter(
                    x=moves, y=ranks,
                    mode='lines+markers',
                    name=f'k={k}',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6, color=color),
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_texts
                ),
                row=1, col=1
            )
            
            # Add perfect matches as stars
            if perfect_moves:
                fig.add_trace(
                    go.Scatter(
                        x=perfect_moves, y=[1]*len(perfect_moves),
                        mode='markers',
                        name=f'k={k} Perfect',
                        marker=dict(size=15, color=color, symbol='star', 
                                  line=dict(color='black', width=1)),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=perfect_hover,
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    # Add perfect reference line
    fig.add_hline(y=1, line=dict(color='red', width=3, dash='solid'), 
                  annotation_text="Perfect Match", row=1, col=1)
    
    # Plot 2: Aggregated metrics
    aggregated = data['aggregated_metrics']
    k_vals = []
    avg_ranks = []
    rank_errors = []
    perfect_rates = []
    perfect_errors = []
    bar_hover_texts = []
    
    for k in k_values:
        k_str = str(k)
        if k_str in aggregated:
            metric = aggregated[k_str]
            k_vals.append(k)
            avg_ranks.append(metric['avg_target_rank'])
            rank_errors.append(metric['std_target_rank'])
            perfect_rates.append(metric['avg_perfect_match_rate'] * 100)
            perfect_errors.append(metric['std_perfect_match_rate'] * 100)
            
            bar_hover_text = (f"<b>K = {k}</b><br>"
                            f"Avg Rank: {metric['avg_target_rank']:.3f} ± {metric['std_target_rank']:.3f}<br>"
                            f"Perfect Rate: {metric['avg_perfect_match_rate']*100:.1f}% ± {metric['std_perfect_match_rate']*100:.1f}%<br>"
                            f"Top-3 Rate: {metric['avg_top3_rate']*100:.1f}%<br>"
                            f"Consistency: {metric['consistency_score']:.3f}<br>"
                            f"Total Samples: {metric['total_samples']}")
            bar_hover_texts.append(bar_hover_text)
    
    if k_vals:
        # Add rank bars
        fig.add_trace(
            go.Bar(
                x=[f'k={k}' for k in k_vals], y=avg_ranks,
                error_y=dict(type='data', array=rank_errors),
                name='Avg Target Rank',
                marker_color='rgba(255, 68, 68, 0.8)',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=bar_hover_texts,
                yaxis='y3'
            ),
            row=1, col=2
        )
        
        # Add perfect rate bars (secondary y-axis)
        fig.add_trace(
            go.Bar(
                x=[f'k={k}' for k in k_vals], y=perfect_rates,
                error_y=dict(type='data', array=perfect_errors),
                name='Perfect Match %',
                marker_color='rgba(0, 170, 0, 0.8)',
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=bar_hover_texts,
                yaxis='y4'
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Interactive SCAR Move Selection Quality Analysis<br><i>Hover over points for detailed information</i>",
        title_x=0.5,
        width=1400,
        height=700,
        hovermode='closest'
    )
    
    # Update x and y axes
    fig.update_xaxes(title_text="Move Number", row=1, col=1)
    fig.update_yaxes(title_text="Target Community Rank (1=perfect)", row=1, col=1)
    fig.update_xaxes(title_text="K Value", row=1, col=2)
    fig.update_yaxes(title_text="Average Target Rank", row=1, col=2)
    
    # Add secondary y-axis for perfect match rate
    fig.update_layout(yaxis4=dict(title="Perfect Match Rate %", overlaying='y3', side='right'))
    
    # Save and show
    pyo.plot(fig, filename='interactive_move_analysis.html', auto_open=True)
    print("✅ Interactive Plotly visualization saved as 'interactive_move_analysis.html'")
    print("🖱️  Open the HTML file in your browser to see full interactivity!")

def create_matplotlib_hover_plot(data):
    """Create matplotlib plot with working hover using built-in functionality."""
    print("Creating matplotlib plot with hover functionality...")
    
    all_samples = data.get('all_samples', [])
    config = data['config']
    k_values = config['KValues']
    
    if not all_samples:
        print("No sample data available")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Prepare data for hover
    move_samples = {}
    for sample in all_samples:
        move_id = sample['move_id']
        if move_id not in move_samples:
            move_samples[move_id] = sample
    
    # Plot 1: Store all data points for hover
    all_points = []  # Will store (x, y, hover_text, color, marker_type)
    
    sorted_moves = sorted(move_samples.keys())
    if len(sorted_moves) > 200:
        step = len(sorted_moves) // 200
        sorted_moves = sorted_moves[::step]
    
    for k_idx, k in enumerate(k_values):
        moves = []
        ranks = []
        color = CONTRASTIVE_COLORS[k_idx % len(CONTRASTIVE_COLORS)]
        
        for move_id in sorted_moves:
            sample = move_samples[move_id]
            scar_rankings = sample.get('scar_rankings', {})
            
            if str(k) in scar_rankings:
                ranking = scar_rankings[str(k)]
                if ranking and ranking.get('target_rank', 0) > 0:
                    moves.append(move_id)
                    rank = ranking['target_rank']
                    ranks.append(rank)
                    
                    # Create hover text
                    hover_text = (f"Move: {move_id}\n"
                                f"Node: {sample.get('node', 'N/A')}\n"
                                f"Target Community: {sample.get('louvain_target_community', 'N/A')}\n"
                                f"Louvain Gain: {sample.get('louvain_gain', 0):.6f}\n"
                                f"K: {k}\n"
                                f"SCAR Rank: {rank}/{len(ranking.get('communities', []))}\n"
                                f"Target Gain: {ranking.get('target_gain', 0):.6f}\n"
                                f"Best Gain: {ranking.get('best_gain', 0):.6f}\n"
                                f"Debug: goto {move_id}")
                    
                    marker_type = 'star' if rank == 1 else 'circle'
                    all_points.append((move_id, rank, hover_text, color, marker_type))
        
        if moves and ranks:
            # Plot line
            ax1.plot(moves, ranks, '--', linewidth=2, color=color, alpha=0.7, label=f'k={k}')
            
            # Plot markers separately for hover
            regular_x, regular_y = [], []
            star_x, star_y = [], []
            
            for x, y, _, _, marker_type in [(x, y, ht, c, mt) for x, y, ht, c, mt in all_points if c == color]:
                if marker_type == 'star':
                    star_x.append(x)
                    star_y.append(y)
                else:
                    regular_x.append(x)
                    regular_y.append(y)
            
            if regular_x:
                ax1.scatter(regular_x, regular_y, color=color, s=40, alpha=0.8, zorder=5)
            if star_x:
                ax1.scatter(star_x, star_y, color=color, s=120, marker='*', 
                          edgecolors='black', linewidth=1, zorder=6)
    
    # Add perfect reference line
    ax1.axhline(y=1, color='red', linestyle='-', linewidth=3, alpha=0.9, label='Perfect Match')
    
    # Setup plot 1
    ax1.set_xlabel('Move Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Target Community Rank (1=perfect)', fontsize=12, fontweight='bold')
    ax1.set_title('Target Community Ranking Over Moves\n* = Perfect Matches\n(Hover for details)', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.4)
    
    # Create hover annotation
    annot = ax1.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="yellow", alpha=0.8),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def on_hover(event):
        if event.inaxes == ax1:
            # Find closest point
            if event.xdata is None or event.ydata is None:
                return
            
            min_dist = float('inf')
            closest_point = None
            
            for x, y, hover_text, color, marker_type in all_points:
                dist = ((event.xdata - x)**2 + (event.ydata - y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_point = (x, y, hover_text, color, marker_type)
            
            # Show annotation if close enough
            if closest_point and min_dist < 20:  # Adjust sensitivity
                x, y, hover_text, color, marker_type = closest_point
                annot.xy = (x, y)
                annot.set_text(hover_text)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    
    def on_click(event):
        if event.inaxes == ax1 and event.button == 1:
            if event.xdata is not None:
                move_id = int(round(event.xdata))
                print(f"\n🎯 Clicked on move {move_id}")
                print(f"Debugger commands:")
                print(f"  ./move_debugger graph.txt properties.txt path.txt")
                print(f"  > goto {move_id}")
                print(f"  > analyze")
    
    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    # Plot 2: Aggregated summary (simplified)
    plot_aggregated_summary_simple(data, ax2)
    
    plt.tight_layout()
    plt.savefig('move_selection_quality_analysis.png', dpi=300, bbox_inches='tight')
    
    print("✅ Matplotlib plot created with hover functionality")
    print("🖱️  Hover over data points to see tooltips")
    print("🖱️  Click on points to get debugger commands")
    
    plt.show()

def plot_aggregated_summary_simple(data, ax):
    """Plot aggregated summary metrics (non-interactive version for matplotlib)."""
    aggregated = data['aggregated_metrics']
    config = data['config']
    k_values = config['KValues']
    
    # Extract data
    k_vals = []
    avg_ranks = []
    rank_errors = []
    perfect_rates = []
    perfect_errors = []
    
    for k in k_values:
        k_str = str(k)
        if k_str in aggregated:
            metric = aggregated[k_str]
            k_vals.append(k)
            avg_ranks.append(metric['avg_target_rank'])
            rank_errors.append(metric['std_target_rank'])
            perfect_rates.append(metric['avg_perfect_match_rate'] * 100)
            perfect_errors.append(metric['std_perfect_match_rate'] * 100)
    
    if not k_vals:
        ax.text(0.5, 0.5, 'No aggregated data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create dual axis
    ax2 = ax.twinx()
    x_pos = np.arange(len(k_vals))
    
    # Plot bars
    bars1 = ax.bar(x_pos - 0.2, avg_ranks, width=0.4, yerr=rank_errors, 
                   capsize=5, color='#FF4444', alpha=0.8, label='Avg Target Rank')
    bars2 = ax2.bar(x_pos + 0.2, perfect_rates, width=0.4, yerr=perfect_errors,
                    capsize=5, color='#00AA00', alpha=0.8, label='Perfect Match %')
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2, height1 + rank_errors[i] + 0.05,
               f'{avg_ranks[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        height2 = bar2.get_height()
        ax2.text(bar2.get_x() + bar2.get_width()/2, height2 + perfect_errors[i] + 2,
                f'{perfect_rates[i]:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Formatting
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('K Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Target Rank (±std)', fontsize=12, fontweight='bold', color='#FF4444')
    ax2.set_ylabel('Perfect Match Rate % (±std)', fontsize=12, fontweight='bold', color='#00AA00')
    ax.set_title(f'Aggregated Quality Metrics\n({config["NumRuns"]} runs each)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k}' for k in k_vals])
    ax.grid(True, alpha=0.4, axis='y')
    
    ax.tick_params(axis='y', labelcolor='#FF4444')
    ax2.tick_params(axis='y', labelcolor='#00AA00')
    
    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

def main():
    print("Loading move selection quality experiment results...")
    data = load_experiment_data()
    
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZATION OPTIONS")
    print("="*60)
    print("1. Plotly version (HTML, guaranteed hover functionality)")
    print("2. Matplotlib version (hover with mouse movement)")
    print("3. Both versions")
    
    choice = input("\nWhich would you like to create? (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        create_plotly_interactive_plot(data)
    
    if choice in ['2', '3']:
        create_matplotlib_hover_plot(data)
    
    if choice not in ['1', '2', '3']:
        print("Invalid choice, creating Plotly version...")
        create_plotly_interactive_plot(data)

if __name__ == "__main__":
    main()

# Use high contrast colors instead of subtle palettes
CONTRASTIVE_COLORS = ['#FF0000', '#00AA00', '#0066FF', '#FF8800', '#AA00AA', '#00AAAA', '#666666', '#FFD700']

def load_experiment_data():
    """Load aggregated experiment data from fixed filename."""
    try:
        with open('move_selection_aggregated_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: move_selection_aggregated_results.json not found!")
        print("Make sure you ran the Go move selection quality experiment first.")
        exit(1)

def plot_key_visualizations(data):
    """Create the two most important plots with interactivity."""
    print("Creating interactive move selection quality visualizations...")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot 1: Ranking Evolution Over Moves (Sample from all runs)
    plot_ranking_evolution_interactive(data, ax1)
    
    # Plot 2: Aggregated Summary Metrics with Error Bars
    plot_aggregated_summary_interactive(data, ax2)
    
    plt.tight_layout()
    
    # Add instructions
    if INTERACTIVE_AVAILABLE:
        fig.suptitle('Interactive Plots: Hover over points for details, click to print debugging info\n', 
                    fontsize=14, y=0.98)
    
    plt.savefig('move_selection_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_ranking_evolution_interactive(data, ax):
    """Plot target community ranking evolution with interactive hover tooltips."""
    all_samples = data.get('all_samples', [])
    config = data['config']
    k_values = config['KValues']
    
    if not all_samples:
        ax.text(0.5, 0.5, 'No sample data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Target Community Ranking Evolution')
        return
    
    # Group samples by move ID and take one sample per move (from any run)
    move_samples = {}
    for sample in all_samples:
        move_id = sample['move_id']
        if move_id not in move_samples:
            move_samples[move_id] = sample
    
    # Sort by move ID and take a reasonable sample for visualization
    sorted_moves = sorted(move_samples.keys())
    if len(sorted_moves) > 200:  # Increase limit for more detail
        step = len(sorted_moves) // 200
        sorted_moves = sorted_moves[::step]
    
    # Store plot elements and data for hover functionality
    plot_elements = []
    hover_data = []
    
    # Plot each k value with high contrast colors
    for k_idx, k in enumerate(k_values):
        moves = []
        ranks = []
        perfect_moves = []
        sample_data = []  # Store sample data for hover
        
        for move_id in sorted_moves:
            sample = move_samples[move_id]
            scar_rankings = sample.get('scar_rankings', {})
            
            if str(k) in scar_rankings:
                ranking = scar_rankings[str(k)]
                if ranking and ranking.get('target_rank', 0) > 0:
                    moves.append(move_id)
                    rank = ranking['target_rank']
                    ranks.append(rank)
                    
                    # Store detailed sample data for hover
                    sample_data.append({
                        'move_id': move_id,
                        'node': sample.get('node', 'N/A'),
                        'target_community': sample.get('louvain_target_community', 'N/A'),
                        'louvain_gain': sample.get('louvain_gain', 0),
                        'k': k,
                        'rank': rank,
                        'total_communities': len(ranking.get('communities', [])),
                        'target_gain': ranking.get('target_gain', 0),
                        'best_gain': ranking.get('best_gain', 0),
                        'worst_gain': ranking.get('worst_gain', 0)
                    })
                    
                    # Track perfect matches for special marking
                    if rank == 1:
                        perfect_moves.append(move_id)
        
        if moves and ranks:
            # Use high contrast colors
            color = CONTRASTIVE_COLORS[k_idx % len(CONTRASTIVE_COLORS)]
            
            # Plot line with markers - store the plot objects
            line, = ax.plot(moves, ranks, '--', linewidth=2.5, marker='o', markersize=6,
                          label=f'k={k}', color=color, alpha=0.8)
            plot_elements.append(line)
            hover_data.extend(sample_data)
            
            # Highlight perfect matches with stars
            if perfect_moves:
                perfect_ranks = [1] * len(perfect_moves)
                stars = ax.scatter(perfect_moves, perfect_ranks, s=150, marker='*', 
                          color=color, edgecolors='black', linewidth=1.5, zorder=5)
                plot_elements.append(stars)
    
    # Perfect match reference line
    ref_line = ax.axhline(y=1, color='red', linestyle='-', linewidth=3, alpha=0.9, label='Perfect Match (Rank 1)')
    
    ax.set_xlabel('Move Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Community Rank (1=perfect)', fontsize=12, fontweight='bold')
    ax.set_title('Target Community Ranking Over Moves\n* = Perfect Matches', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.4)
    
    # Set reasonable y-axis limits
    if all_samples:
        max_rank = 1
        for sample in all_samples:
            for ranking in sample.get('scar_rankings', {}).values():
                if ranking and ranking.get('target_rank', 0) > 0:
                    max_rank = max(max_rank, ranking['target_rank'])
        ax.set_ylim(0.5, min(max_rank + 0.5, 15.5))
    
    # Add interactive hover functionality
    if INTERACTIVE_AVAILABLE and hover_data:
        def hover_formatter(sel):
            # Find the corresponding data point
            if hasattr(sel.target, 'get_offsets'):  # Scatter plot
                x, y = sel.target.get_offsets()[sel.target_index]
            else:  # Line plot
                x, y = sel.target.get_xydata()[sel.target_index]
            
            # Find matching data point
            for data_point in hover_data:
                if abs(data_point['move_id'] - x) < 0.5 and abs(data_point['rank'] - y) < 0.5:
                    return (f"Move: {data_point['move_id']}\n"
                           f"Node: {data_point['node']}\n"
                           f"Target Community: {data_point['target_community']}\n"
                           f"Louvain Gain: {data_point['louvain_gain']:.6f}\n"
                           f"K: {data_point['k']}\n"
                           f"SCAR Rank: {data_point['rank']}/{data_point['total_communities']}\n"
                           f"Target Gain: {data_point['target_gain']:.6f}\n"
                           f"Best Gain: {data_point['best_gain']:.6f}\n"
                           f"Debug: goto {data_point['move_id']}")
            
            return f"Move: {int(x)}, Rank: {int(y)}"
        
        # Add cursors to all plot elements
        cursors = []
        for element in plot_elements:
            if element is not None:
                cursor = mplcursors.cursor(element, hover=True)
                cursor.connect("add", lambda sel: sel.annotation.set_text(hover_formatter(sel)))
                cursor.connect("add", lambda sel: sel.annotation.get_bbox_patch().set(boxstyle="round,pad=0.5", 
                                                                                     facecolor="yellow", alpha=0.8))
                cursors.append(cursor)
        
        # Add click functionality to print debugging commands
        def on_click(event):
            if event.inaxes == ax and event.button == 1:  # Left click
                x_click = event.xdata
                if x_click is not None:
                    move_id = int(round(x_click))
                    print(f"\n🎯 Clicked on move {move_id}")
                    print(f"Debugger commands:")
                    print(f"  ./move_debugger graph.txt properties.txt path.txt")
                    print(f"  > goto {move_id}")
                    print(f"  > analyze")
                    print(f"  > communities")
        
        ax.figure.canvas.mpl_connect('button_press_event', on_click)

def plot_aggregated_summary_interactive(data, ax):
    """Plot aggregated summary metrics with error bars and hover tooltips."""
    aggregated = data['aggregated_metrics']
    config = data['config']
    k_values = config['KValues']
    
    # Extract data with error bars
    k_vals = []
    avg_ranks = []
    rank_errors = []
    perfect_rates = []
    perfect_errors = []
    summary_data = []
    
    for k in k_values:
        k_str = str(k)
        if k_str in aggregated:
            metric = aggregated[k_str]
            k_vals.append(k)
            avg_ranks.append(metric['avg_target_rank'])
            rank_errors.append(metric['std_target_rank'])
            perfect_rates.append(metric['avg_perfect_match_rate'] * 100)
            perfect_errors.append(metric['std_perfect_match_rate'] * 100)
            
            # Store data for hover
            summary_data.append({
                'k': k,
                'avg_rank': metric['avg_target_rank'],
                'std_rank': metric['std_target_rank'],
                'perfect_rate': metric['avg_perfect_match_rate'] * 100,
                'std_perfect': metric['std_perfect_match_rate'] * 100,
                'top3_rate': metric['avg_top3_rate'] * 100,
                'consistency': metric['consistency_score'],
                'total_samples': metric['total_samples']
            })
    
    if not k_vals:
        ax.text(0.5, 0.5, 'No aggregated data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Aggregated Summary Metrics')
        return
    
    # Create dual y-axis plot
    ax2 = ax.twinx()
    
    # Plot 1: Average target rank (lower is better) - LEFT Y-AXIS
    x_pos = np.arange(len(k_vals))
    bars1 = ax.bar(x_pos - 0.2, avg_ranks, width=0.4, yerr=rank_errors, 
                   capsize=5, color='#FF4444', alpha=0.8, label='Avg Target Rank')
    
    # Plot 2: Perfect match rate (higher is better) - RIGHT Y-AXIS  
    bars2 = ax2.bar(x_pos + 0.2, perfect_rates, width=0.4, yerr=perfect_errors,
                    capsize=5, color='#00AA00', alpha=0.8, label='Perfect Match %')
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Rank labels (on red bars)
        height1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2, height1 + rank_errors[i] + 0.1,
               f'{avg_ranks[i]:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Perfect rate labels (on green bars)  
        height2 = bar2.get_height()
        ax2.text(bar2.get_x() + bar2.get_width()/2, height2 + perfect_errors[i] + 2,
                f'{perfect_rates[i]:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Perfect match reference line
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Rank')
    
    # Formatting
    ax.set_xlabel('k value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Target Rank (±std)', fontsize=12, fontweight='bold', color='#FF4444')
    ax2.set_ylabel('Perfect Match Rate % (±std)', fontsize=12, fontweight='bold', color='#00AA00')
    ax.set_title(f'Aggregated Quality Metrics\n({config["NumRuns"]} runs each)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k}' for k in k_vals], rotation=45)
    ax.grid(True, alpha=0.4, axis='y')
    
    # Color the y-axis labels to match the bars
    ax.tick_params(axis='y', labelcolor='#FF4444')
    ax2.tick_params(axis='y', labelcolor='#00AA00')
    
    # Set reasonable y-axis limits
    ax.set_ylim(0, max(avg_ranks) + max(rank_errors) + 1)
    ax2.set_ylim(0, 105)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Add interactive hover functionality for bars
    if INTERACTIVE_AVAILABLE and summary_data:
        def hover_formatter_bars(sel):
            # Get the bar index
            if hasattr(sel.target, 'get_x'):  # Bar chart
                bar_x = sel.target.get_x()
                # Find which bar was hovered over
                for i, x_position in enumerate(x_pos):
                    if abs(bar_x - (x_position - 0.2)) < 0.1 or abs(bar_x - (x_position + 0.2)) < 0.1:
                        data = summary_data[i]
                        return (f"K = {data['k']}\n"
                               f"Avg Rank: {data['avg_rank']:.3f} ± {data['std_rank']:.3f}\n"
                               f"Perfect Rate: {data['perfect_rate']:.1f}% ± {data['std_perfect']:.1f}%\n"
                               f"Top-3 Rate: {data['top3_rate']:.1f}%\n"
                               f"Consistency: {data['consistency']:.3f}\n"
                               f"Total Samples: {data['total_samples']}")
            return "Summary data"
        
        # Add cursors to bars
        cursor1 = mplcursors.cursor(bars1, hover=True)
        cursor1.connect("add", lambda sel: sel.annotation.set_text(hover_formatter_bars(sel)))
        cursor1.connect("add", lambda sel: sel.annotation.get_bbox_patch().set(boxstyle="round,pad=0.5", 
                                                                               facecolor="lightcoral", alpha=0.8))
        
        cursor2 = mplcursors.cursor(bars2, hover=True)
        cursor2.connect("add", lambda sel: sel.annotation.set_text(hover_formatter_bars(sel)))
        cursor2.connect("add", lambda sel: sel.annotation.get_bbox_patch().set(boxstyle="round,pad=0.5", 
                                                                               facecolor="lightgreen", alpha=0.8))

def find_transition_points(data):
    """Analyze and print information about transition points."""
    print("\n🔍 TRANSITION POINT ANALYSIS")
    print("="*50)
    
    all_samples = data.get('all_samples', [])
    if not all_samples:
        print("No sample data available for transition analysis.")
        return
    
    # Group by move_id and analyze degradation
    move_data = {}
    for sample in all_samples:
        move_id = sample['move_id']
        if move_id not in move_data:
            move_data[move_id] = {'perfect': 0, 'total': 0, 'bad_ranks': []}
        
        for k, ranking in sample.get('scar_rankings', {}).items():
            if ranking and ranking.get('target_rank', 0) > 0:
                move_data[move_id]['total'] += 1
                rank = ranking['target_rank']
                if rank == 1:
                    move_data[move_id]['perfect'] += 1
                else:
                    move_data[move_id]['bad_ranks'].append(rank)
    
    # Find moves with significant degradation
    problem_moves = []
    for move_id, data_point in move_data.items():
        if data_point['total'] > 0:
            perfect_rate = data_point['perfect'] / data_point['total']
            avg_bad_rank = np.mean(data_point['bad_ranks']) if data_point['bad_ranks'] else 1
            
            if perfect_rate < 0.5 and avg_bad_rank > 2:  # Less than 50% perfect, avg bad rank > 2
                problem_moves.append((move_id, perfect_rate, avg_bad_rank))
    
    # Sort by move_id and find clusters
    problem_moves.sort()
    
    if problem_moves:
        print(f"Found {len(problem_moves)} problematic moves:")
        first_problem = problem_moves[0][0]
        last_good = max(0, first_problem - 50)  # Look 50 moves before first problem
        
        print(f"  🎯 First degradation around move: {first_problem}")
        print(f"  📊 Transition window: moves {last_good}-{first_problem}")
        print(f"  🔧 Debugging suggestion:")
        print(f"     ./move_debugger graph.txt properties.txt path.txt")
        print(f"     > goto {last_good}")
        print(f"     > step 10  # Step through transition")
        print(f"     > analyze  # When you reach problems")
        
        print(f"\n  Top 5 most problematic moves:")
        for i, (move_id, perfect_rate, avg_bad_rank) in enumerate(problem_moves[:5]):
            print(f"    Move {move_id}: {perfect_rate*100:.1f}% perfect, avg bad rank: {avg_bad_rank:.1f}")
    else:
        print("No clear transition points detected.")

def print_detailed_summary(data):
    """Print comprehensive summary with statistical details."""
    print("\n" + "="*80)
    print("AGGREGATED MOVE SELECTION QUALITY EXPERIMENT SUMMARY")
    print("="*80)
    
    config = data['config']
    graph_info = data['graph_info']
    aggregated = data['aggregated_metrics']
    analysis = data['best_k_analysis']
    
    print(f"Experiment Configuration:")
    print(f"  • Runs: {config['NumRuns']} independent experiments")
    print(f"  • K values tested: {config['KValues']}")
    print(f"  • Sampling: every {config['SampleInterval']} moves")
    print(f"  • Max samples per run: {config['MaxSamples']}")
    print(f"  • Total samples: {len(data['all_samples'])}")
    
    print(f"\nGraph Properties:")
    print(f"  • Nodes: {graph_info['num_nodes']}")
    print(f"  • Total weight: {graph_info['total_weight']:.2f}")
    print(f"  • Average degree: {graph_info['avg_degree']:.2f}")
    
    print(f"\nAGGREGATED RESULTS (MEAN ± STANDARD DEVIATION):")
    print(f"{'K':>6s} {'Avg Rank':>12s} {'Perfect%':>12s} {'Top3%':>12s} {'Consistency':>12s} {'Samples':>8s}")
    print("-" * 80)
    
    k_values = config['KValues']
    for k in k_values:
        k_str = str(k)
        if k_str in aggregated:
            metric = aggregated[k_str]
            print(f"{k:>6d} "
                  f"{metric['avg_target_rank']:>6.2f}±{metric['std_target_rank']:>4.2f} "
                  f"{metric['avg_perfect_match_rate']*100:>6.1f}±{metric['std_perfect_match_rate']*100:>4.1f} "
                  f"{metric['avg_top3_rate']*100:>6.1f}±{metric['std_top3_rate']*100:>4.1f} "
                  f"{metric['consistency_score']:>11.3f} "
                  f"{metric['total_samples']:>7d}")
    
    print(f"\n🏆 BEST K VALUE ANALYSIS:")
    print(f"  • Best average rank: k={analysis['best_k_by_avg_rank']} "
          f"(avg: {aggregated[str(analysis['best_k_by_avg_rank'])]['avg_target_rank']:.2f})")
    print(f"  • Best perfect rate: k={analysis['best_k_by_perfect_rate']} "
          f"({aggregated[str(analysis['best_k_by_perfect_rate'])]['avg_perfect_match_rate']*100:.1f}%)")
    print(f"  • Most consistent: k={analysis['best_k_by_consistency']} "
          f"(score: {aggregated[str(analysis['best_k_by_consistency'])]['consistency_score']:.3f})")
    print(f"  🎯 RECOMMENDED: k={analysis['recommended_k']} "
          f"(efficiency score: {analysis['recommended_k_score']:.3f})")
    
    # Show efficiency ranking
    print(f"\nK VALUE EFFICIENCY RANKING:")
    print(f"{'Rank':>4s} {'K':>6s} {'Efficiency':>10s} {'Avg Rank':>9s} {'Perfect%':>9s} {'Consistency':>11s}")
    print("-" * 65)
    
    for i, k_eff in enumerate(analysis['k_efficiency_ranking'][:5]):  # Top 5
        print(f"{i+1:>4d} {k_eff['k']:>6d} {k_eff['efficiency_score']:>9.3f} "
              f"{k_eff['avg_rank']:>8.2f} {k_eff['perfect_rate']*100:>8.1f} "
              f"{k_eff['consistency_score']:>10.3f}")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   • Lower average rank = SCAR's choice closer to Louvain's optimal choice")
    print(f"   • Higher perfect% = more often SCAR makes exact same decision as Louvain")
    print(f"   • Higher consistency = more reliable results across different random runs")
    print(f"   • Error bars show variability across {config['NumRuns']} independent runs")
    print(f"   • Recommended K balances accuracy, computational cost, and reliability")
    
    if INTERACTIVE_AVAILABLE:
        print(f"   • 🖱️  HOVER over plot points for detailed information")
        print(f"   • 🖱️  CLICK on ranking plot points for debugger commands")
    
    print("="*80)

def main():
    print("Loading aggregated move selection quality experiment results...")
    data = load_experiment_data()
    
    # Check if interactive mode is available
    if INTERACTIVE_AVAILABLE:
        print("✅ Interactive mode enabled - hover over points for details!")
    else:
        print("⚠️  Non-interactive mode - install mplcursors for hover functionality")
        print("   Run: pip install mplcursors")
    
    # Print detailed summary first
    print_detailed_summary(data)
    
    # Analyze transition points
    find_transition_points(data)
    
    # Create focused visualizations (interactive if available)
    plot_key_visualizations(data)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Key visualization saved: move_selection_quality_analysis.png")
    print(f"📈 Left plot: Target ranking evolution over moves (* = perfect matches)")
    print(f"📊 Right plot: Aggregated metrics with error bars across all runs")
    
    if INTERACTIVE_AVAILABLE:
        print(f"🖱️  Interactive features:")
        print(f"   • Hover over data points to see detailed information")
        print(f"   • Click on ranking plot points to get debugger commands")
        print(f"   • Yellow tooltips show move details, gains, and debug commands")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
SCAR Community Evolution Plotter

Creates interactive plots showing community count, modularity, and sketch health over moves.
Designed to work with data exported from the SCAR debugger.

Requirements: pip install matplotlib numpy seaborn mplcursors
"""

import json
import matplotlib.pyplot as plt
import numpy as np
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

# Colors for different metrics
COLORS = {
    'communities': '#2E86AB',
    'modularity': '#A23B72', 
    'sketch_health': '#F18F01',
    'trend': '#C73E1D'
}

def load_evolution_data():
    """Load community evolution data from the debugger export."""
    filename = 'scar_community_evolution.json'
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        print("Make sure you ran the SCAR debugger and used the 'export' command.")
        exit(1)

def create_community_evolution_plot(data):
    """Create the main community evolution plot with interactive features."""
    print("Creating community evolution plot...")
    
    # Extract data
    moves = np.array(data['moves'])
    num_communities = np.array(data['num_communities'])
    modularity = np.array(data['modularity'])
    sketch_health = np.array(data['sketch_health'])
    config = data['config']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'SCAR Community Evolution (K={config["k"]}, NK={config["nk"]})', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Number of Communities over Moves
    ax1 = axes[0, 0]
    line1 = ax1.plot(moves, num_communities, 'o-', color=COLORS['communities'], 
                     linewidth=2, markersize=4, alpha=0.8, label='Communities')[0]
    ax1.set_xlabel('Move Number')
    ax1.set_ylabel('Number of Communities')
    ax1.set_title('Community Count Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line for communities
    if len(moves) > 1:
        z = np.polyfit(moves, num_communities, 1)
        p = np.poly1d(z)
        ax1.plot(moves, p(moves), '--', color=COLORS['trend'], alpha=0.7, 
                 label=f'Trend (slope: {z[0]:.3f})')
        ax1.legend()
    
    # Plot 2: Modularity over Moves
    ax2 = axes[0, 1]
    line2 = ax2.plot(moves, modularity, 's-', color=COLORS['modularity'], 
                     linewidth=2, markersize=4, alpha=0.8, label='Modularity')[0]
    ax2.set_xlabel('Move Number')
    ax2.set_ylabel('Modularity (Estimated)')
    ax2.set_title('Modularity Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sketch Health over Moves
    ax3 = axes[1, 0]
    line3 = ax3.plot(moves, sketch_health * 100, '^-', color=COLORS['sketch_health'], 
                     linewidth=2, markersize=4, alpha=0.8, label='Sketch Health')[0]
    ax3.set_xlabel('Move Number')
    ax3.set_ylabel('Sketch Health (%)')
    ax3.set_title('Sketch Utilization Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)
    
    # Add health zones
    ax3.axhspan(80, 100, alpha=0.2, color='green', label='Excellent (80-100%)')
    ax3.axhspan(60, 80, alpha=0.2, color='yellow', label='Good (60-80%)')
    ax3.axhspan(0, 60, alpha=0.2, color='red', label='Poor (0-60%)')
    ax3.legend()
    
    # Plot 4: Combined Overview (dual y-axis)
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    # Communities on left axis
    line4a = ax4.plot(moves, num_communities, 'o-', color=COLORS['communities'], 
                      linewidth=2, markersize=3, alpha=0.8, label='Communities')[0]
    ax4.set_xlabel('Move Number')
    ax4.set_ylabel('Number of Communities', color=COLORS['communities'])
    ax4.tick_params(axis='y', labelcolor=COLORS['communities'])
    
    # Modularity on right axis
    line4b = ax4_twin.plot(moves, modularity, 's-', color=COLORS['modularity'], 
                           linewidth=2, markersize=3, alpha=0.8, label='Modularity')[0]
    ax4_twin.set_ylabel('Modularity', color=COLORS['modularity'])
    ax4_twin.tick_params(axis='y', labelcolor=COLORS['modularity'])
    
    ax4.set_title('Communities vs Modularity')
    ax4.grid(True, alpha=0.3)
    
    # Store all data points for hover functionality
    hover_data = []
    
    # Prepare hover data for each plot
    for i, move in enumerate(moves):
        # Plot 1 data
        hover_data.append({
            'x': move, 'y': num_communities[i], 'ax': ax1,
            'text': f"Move: {move}\nCommunities: {num_communities[i]}\nModularity: {modularity[i]:.6f}\nSketch Health: {sketch_health[i]*100:.1f}%\nDebug: goto {move}",
            'line': line1
        })
        
        # Plot 2 data
        hover_data.append({
            'x': move, 'y': modularity[i], 'ax': ax2,
            'text': f"Move: {move}\nModularity: {modularity[i]:.6f}\nCommunities: {num_communities[i]}\nSketch Health: {sketch_health[i]*100:.1f}%\nDebug: goto {move}",
            'line': line2
        })
        
        # Plot 3 data
        hover_data.append({
            'x': move, 'y': sketch_health[i]*100, 'ax': ax3,
            'text': f"Move: {move}\nSketch Health: {sketch_health[i]*100:.1f}%\nCommunities: {num_communities[i]}\nModularity: {modularity[i]:.6f}\nDebug: goto {move}",
            'line': line3
        })
        
        # Plot 4a data (communities)
        hover_data.append({
            'x': move, 'y': num_communities[i], 'ax': ax4,
            'text': f"Move: {move}\nCommunities: {num_communities[i]}\nModularity: {modularity[i]:.6f}\nSketch Health: {sketch_health[i]*100:.1f}%\nDebug: goto {move}",
            'line': line4a
        })
        
        # Plot 4b data (modularity) - Note: using ax4_twin
        hover_data.append({
            'x': move, 'y': modularity[i], 'ax': ax4_twin,
            'text': f"Move: {move}\nModularity: {modularity[i]:.6f}\nCommunities: {num_communities[i]}\nSketch Health: {sketch_health[i]*100:.1f}%\nDebug: goto {move}",
            'line': line4b
        })
    
    # Create hover annotations
    all_axes = [ax1, ax2, ax3, ax4, ax4_twin]
    annotations = {}
    for ax in all_axes:
        annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9),
                           arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)
        annotations[ax] = annot
    
    def on_hover(event):
        if event.inaxes not in all_axes:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        # Find closest point in the current axes
        min_dist = float('inf')
        closest_point = None
        
        for point in hover_data:
            if point['ax'] == event.inaxes:
                # Normalize distance calculation
                x_range = event.inaxes.get_xlim()[1] - event.inaxes.get_xlim()[0]
                y_range = event.inaxes.get_ylim()[1] - event.inaxes.get_ylim()[0]
                
                if x_range > 0 and y_range > 0:
                    norm_dist = ((event.xdata - point['x']) / x_range)**2 + ((event.ydata - point['y']) / y_range)**2
                    if norm_dist < min_dist:
                        min_dist = norm_dist
                        closest_point = point
        
        # Show annotation if close enough
        if closest_point and min_dist < 0.01:  # Adjust sensitivity
            annot = annotations[closest_point['ax']]
            annot.xy = (closest_point['x'], closest_point['y'])
            annot.set_text(closest_point['text'])
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            # Hide all annotations
            for annot in annotations.values():
                if annot.get_visible():
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    
    def on_click(event):
        if event.inaxes in all_axes and event.button == 1:
            if event.xdata is not None:
                move_id = int(round(event.xdata))
                print(f"\n🎯 Clicked on move {move_id}")
                print(f"Debugger commands:")
                print(f"  ./scar_debugger debug {config['graph_file']} properties.txt path.txt")
                print(f"  > goto {move_id}")
                print(f"  > status")
                print(f"  > communities")
    
    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout()
    plt.savefig('scar_community_evolution.png', dpi=300, bbox_inches='tight')
    
    return fig

def create_summary_plot(data):
    """Create a summary plot showing key statistics."""
    print("Creating summary statistics plot...")
    
    moves = np.array(data['moves'])
    num_communities = np.array(data['num_communities'])
    modularity = np.array(data['modularity'])
    sketch_health = np.array(data['sketch_health'])
    config = data['config']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'SCAR Algorithm Summary (K={config["k"]})', fontsize=14, fontweight='bold')
    
    # Plot 1: Distribution of community counts
    ax1.hist(num_communities, bins=min(20, len(np.unique(num_communities))), 
             alpha=0.7, color=COLORS['communities'], edgecolor='black')
    ax1.set_xlabel('Number of Communities')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Community Counts')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_comms = np.mean(num_communities)
    std_comms = np.std(num_communities)
    ax1.axvline(mean_comms, color='red', linestyle='--', 
                label=f'Mean: {mean_comms:.1f} ± {std_comms:.1f}')
    ax1.legend()
    
    # Plot 2: Convergence analysis
    ax2.plot(moves, num_communities, 'o-', color=COLORS['communities'], 
             linewidth=2, markersize=3, alpha=0.8, label='Communities')
    
    # Add rolling average
    if len(moves) > 10:
        window = min(20, len(moves) // 5)
        rolling_avg = np.convolve(num_communities, np.ones(window)/window, mode='valid')
        rolling_moves = moves[window-1:]
        ax2.plot(rolling_moves, rolling_avg, '--', color=COLORS['trend'], 
                 linewidth=3, alpha=0.8, label=f'Rolling Avg (window={window})')
    
    ax2.set_xlabel('Move Number')
    ax2.set_ylabel('Number of Communities')
    ax2.set_title('Community Count Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('scar_summary_analysis.png', dpi=300, bbox_inches='tight')
    
    return fig

def print_analysis_summary(data):
    """Print detailed analysis of the evolution data."""
    moves = np.array(data['moves'])
    num_communities = np.array(data['num_communities'])
    modularity = np.array(data['modularity'])
    sketch_health = np.array(data['sketch_health'])
    config = data['config']
    
    print("\n" + "="*60)
    print("SCAR COMMUNITY EVOLUTION ANALYSIS")
    print("="*60)
    
    print(f"Configuration:")
    print(f"  Graph: {config['graph_file']}")
    print(f"  K: {config['k']}")
    print(f"  NK: {config['nk']}")
    print(f"  Total Moves: {config['total_moves']}")
    print(f"  Analyzed Moves: {len(moves)}")
    
    print(f"\nCommunity Statistics:")
    print(f"  Initial Communities: {num_communities[0] if len(num_communities) > 0 else 'N/A'}")
    print(f"  Final Communities: {num_communities[-1] if len(num_communities) > 0 else 'N/A'}")
    print(f"  Min Communities: {np.min(num_communities)}")
    print(f"  Max Communities: {np.max(num_communities)}")
    print(f"  Mean Communities: {np.mean(num_communities):.1f} ± {np.std(num_communities):.1f}")
    
    print(f"\nModularity Statistics:")
    # print(f"  Initial Modularity: {modularity[0]:.6f if len(modularity) > 0 else 'N/A'}")
    # print(f"  Final Modularity: {modularity[-1]:.6f if len(modularity) > 0 else 'N/A'}")
    # print(f"  Max Modularity: {np.max(modularity):.6f}")
    # print(f"  Mean Modularity: {np.mean(modularity):.6f} ± {np.std(modularity):.6f}")
    
    print(f"\nSketch Health Statistics:")
    print(f"  Initial Health: {sketch_health[0]*100:.1f}% if len(sketch_health) > 0 else 'N/A'")
    print(f"  Final Health: {sketch_health[-1]*100:.1f}% if len(sketch_health) > 0 else 'N/A'")
    print(f"  Mean Health: {np.mean(sketch_health)*100:.1f}% ± {np.std(sketch_health)*100:.1f}%")
    
    # Convergence analysis
    if len(num_communities) > 10:
        # Check if communities are stabilizing (last 10% of moves)
        last_portion = num_communities[int(len(num_communities)*0.9):]
        stability = np.std(last_portion)
        print(f"\nConvergence Analysis:")
        print(f"  Final 10% Stability (std): {stability:.2f}")
        if stability < 1.0:
            print(f"  Status: ✅ CONVERGED - communities are stable")
        elif stability < 5.0:
            print(f"  Status: ⚠️ CONVERGING - communities are stabilizing")
        else:
            print(f"  Status: 🔄 EVOLVING - communities still changing significantly")
    
    # Algorithm performance insights
    print(f"\nAlgorithm Performance:")
    if np.mean(sketch_health) > 0.8:
        print(f"  Sketch Efficiency: ✅ EXCELLENT (>{80:.0f}% avg health)")
    elif np.mean(sketch_health) > 0.6:
        print(f"  Sketch Efficiency: ⚠️ GOOD ({np.mean(sketch_health)*100:.0f}% avg health)")
    else:
        print(f"  Sketch Efficiency: 🔴 POOR ({np.mean(sketch_health)*100:.0f}% avg health)")
    
    community_reduction = ((num_communities[0] - num_communities[-1]) / num_communities[0] * 100) if len(num_communities) > 0 and num_communities[0] > 0 else 0
    print(f"  Community Reduction: {community_reduction:.1f}%")
    
    if INTERACTIVE_AVAILABLE:
        print(f"\n🖱️  Interactive Features Available:")
        print(f"   • Hover over data points for detailed information")
        print(f"   • Click on points to get debugger commands")
        print(f"   • Quality indicators and trend analysis")
    
    print("="*60)

def main():
    print("SCAR Community Evolution Plotter")
    print("================================")
    
    # Load data
    data = load_evolution_data()
    
    # Print summary
    print_analysis_summary(data)
    
    # Create plots
    fig1 = create_community_evolution_plot(data)
    fig2 = create_summary_plot(data)
    
    print("\n✅ Plots created successfully!")
    print("📊 Generated files:")
    print("  - scar_community_evolution.png")
    print("  - scar_summary_analysis.png")
    
    if INTERACTIVE_AVAILABLE:
        print(f"\n🖱️  Interactive features enabled:")
        print(f"   • Hover over data points to see move details")
        print(f"   • Click on points to get debugger commands")
    else:
        print(f"\n⚠️  For interactive features, install: pip install mplcursors")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
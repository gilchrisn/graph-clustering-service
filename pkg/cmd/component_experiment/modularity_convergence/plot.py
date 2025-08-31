#!/usr/bin/env python3
"""
Modularity Table Visualizer

Reads modularity_table.json and creates plots showing:
- Modularity evolution over moves
- Final modularity comparison
- SCAR convergence analysis

Requirements: pip install matplotlib numpy
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_table():
    """Load modularity table from fixed filename."""
    try:
        with open('modularity_table.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: modularity_table.json not found!")
        print("Make sure you ran the Go program first.")
        exit(1)

def plot_modularity_evolution(data):
    """Plot modularity evolution over moves for all algorithms."""
    print("Creating modularity evolution plot...")
    
    algorithms = data['algorithms']
    modularity_data = data['modularity_data']
    max_moves = data['max_moves']
    
    plt.figure(figsize=(14, 8))
    
    moves = list(range(len(modularity_data)))
    
    # Plot each algorithm
    for i, alg in enumerate(algorithms):
        modularity_values = [row[i] for row in modularity_data]
        
        if alg['algorithm'] == 'louvain':
            plt.plot(moves, modularity_values, 'k-', linewidth=3, 
                    label=alg['name'], alpha=0.8)
        else:
            # Color SCAR variants by k value
            color_intensity = min(1.0, alg['k'] / 512.0)  # Normalize by max k
            plt.plot(moves, modularity_values, '--', linewidth=2, 
                    label=f"{alg['name']} (k={alg['k']})", 
                    alpha=0.7, color=plt.cm.viridis(color_intensity))
    
    plt.xlabel('Move Number', fontsize=12)
    plt.ylabel('Modularity', fontsize=12)
    plt.title('Modularity Evolution: Louvain vs SCAR(k)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('modularity_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_final_modularity_comparison(data):
    """Plot final modularity values for all algorithms."""
    print("Creating final modularity comparison...")
    
    algorithms = data['algorithms']
    modularity_data = data['modularity_data']
    
    # Get final modularity values
    final_modularities = modularity_data[-1]  # Last row
    names = [alg['name'] for alg in algorithms]
    
    plt.figure(figsize=(12, 6))
    
    # Color bars: Louvain in black, SCAR variants in gradient
    colors = []
    for alg in algorithms:
        if alg['algorithm'] == 'louvain':
            colors.append('black')
        else:
            color_intensity = min(1.0, alg['k'] / 512.0)
            colors.append(plt.cm.viridis(color_intensity))
    
    bars = plt.bar(names, final_modularities, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_modularities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{value:.5f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Final Modularity', fontsize=12)
    plt.title('Final Modularity Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('final_modularity.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_scar_convergence(data):
    """Plot SCAR convergence analysis."""
    print("Creating SCAR convergence analysis...")
    
    algorithms = data['algorithms']
    modularity_data = data['modularity_data']
    
    # Extract SCAR algorithms and sort by k
    scar_algorithms = [(i, alg) for i, alg in enumerate(algorithms) if alg['algorithm'] == 'scar']
    scar_algorithms.sort(key=lambda x: x[1]['k'])
    
    if len(scar_algorithms) < 2:
        print("Not enough SCAR variants for convergence analysis")
        return
    
    # Find Louvain for reference
    louvain_idx = None
    for i, alg in enumerate(algorithms):
        if alg['algorithm'] == 'louvain':
            louvain_idx = i
            break
    
    if louvain_idx is None:
        print("Louvain not found for reference")
        return
    
    louvain_final = modularity_data[-1][louvain_idx]
    
    # Extract k values and final modularities
    k_values = [alg[1]['k'] for alg in scar_algorithms]
    scar_finals = [modularity_data[-1][alg[0]] for alg in scar_algorithms]
    
    # Calculate gaps from Louvain
    gaps = [mod - louvain_final for mod in scar_finals]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Final modularity vs k
    ax1.plot(k_values, scar_finals, 'bo-', linewidth=2, markersize=8, label='SCAR(k)')
    ax1.axhline(y=louvain_final, color='red', linestyle='--', linewidth=2, label='Louvain')
    ax1.set_xlabel('k (SCAR parameter)', fontsize=12)
    ax1.set_ylabel('Final Modularity', fontsize=12)
    ax1.set_title('SCAR Convergence to Louvain', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Gap from Louvain
    colors = ['red' if gap < -0.001 else 'orange' if gap < 0 else 'green' for gap in gaps]
    bars = ax2.bar(range(len(k_values)), gaps, color=colors, alpha=0.7)
    ax2.set_xlabel('SCAR Variant', fontsize=12)
    ax2.set_ylabel('Modularity Gap from Louvain', fontsize=12)
    ax2.set_title('Gap from Louvain Optimum', fontsize=14)
    ax2.set_xticks(range(len(k_values)))
    ax2.set_xticklabels([f'k={k}' for k in k_values], rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, gap in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.0001 if gap >= 0 else -0.0001),
                f'{gap:+.5f}', ha='center', 
                va='bottom' if gap >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('scar_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_move_by_move_detail(data, max_moves_to_show=100):
    """Plot detailed move-by-move for first N moves."""
    print(f"Creating detailed move-by-move plot (first {max_moves_to_show} moves)...")
    
    algorithms = data['algorithms']
    modularity_data = data['modularity_data']
    
    # Limit to first N moves for readability
    moves_to_plot = min(max_moves_to_show, len(modularity_data))
    moves = list(range(moves_to_plot))
    
    plt.figure(figsize=(14, 8))
    
    # Plot each algorithm
    for i, alg in enumerate(algorithms):
        modularity_values = [modularity_data[move][i] for move in range(moves_to_plot)]
        
        if alg['algorithm'] == 'louvain':
            plt.plot(moves, modularity_values, 'k-', linewidth=2, 
                    label=alg['name'], alpha=0.9)
        else:
            color_intensity = min(1.0, alg['k'] / 512.0)
            plt.plot(moves, modularity_values, '--', linewidth=1.5, 
                    label=f"k={alg['k']}", alpha=0.7,
                    color=plt.cm.viridis(color_intensity))
    
    plt.xlabel('Move Number', fontsize=12)
    plt.ylabel('Modularity', fontsize=12)
    plt.title(f'Detailed Modularity Evolution (First {moves_to_plot} Moves)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('modularity_detail.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(data):
    """Print summary statistics."""
    algorithms = data['algorithms']
    modularity_data = data['modularity_data']
    
    print("\n" + "="*60)
    print("MODULARITY COMPARISON SUMMARY")
    print("="*60)
    
    # Find Louvain reference
    louvain_final = None
    for i, alg in enumerate(algorithms):
        if alg['algorithm'] == 'louvain':
            louvain_final = modularity_data[-1][i]
            print(f"Louvain final modularity: {louvain_final:.6f}")
            break
    
    print(f"Total moves tracked: {len(modularity_data) - 1}")
    print(f"Algorithms compared: {len(algorithms)}")
    
    print("\nFinal modularity by algorithm:")
    for i, alg in enumerate(algorithms):
        final_mod = modularity_data[-1][i]
        if louvain_final is not None and alg['algorithm'] == 'scar':
            gap = final_mod - louvain_final
            print(f"  {alg['name']:12s}: {final_mod:.6f} (gap: {gap:+.6f})")
        else:
            print(f"  {alg['name']:12s}: {final_mod:.6f}")
    
    # SCAR convergence analysis
    scar_algorithms = [(i, alg) for i, alg in enumerate(algorithms) if alg['algorithm'] == 'scar']
    if len(scar_algorithms) >= 2:
        scar_algorithms.sort(key=lambda x: x[1]['k'])
        best_scar = max(scar_algorithms, key=lambda x: modularity_data[-1][x[0]])
        worst_scar = min(scar_algorithms, key=lambda x: modularity_data[-1][x[0]])
        
        best_mod = modularity_data[-1][best_scar[0]]
        worst_mod = modularity_data[-1][worst_scar[0]]
        
        print(f"\nSCAR performance range:")
        print(f"  Best SCAR (k={best_scar[1]['k']}): {best_mod:.6f}")
        print(f"  Worst SCAR (k={worst_scar[1]['k']}): {worst_mod:.6f}")
        print(f"  SCAR range: {best_mod - worst_mod:.6f}")
    
    print("="*60)

def main():
    print("Loading modularity table...")
    data = load_table()
    
    print_summary(data)
    
    # Create all plots
    plot_modularity_evolution(data)
    # plot_final_modularity_comparison(data)
    # plot_scar_convergence(data)
    # plot_move_by_move_detail(data, max_moves_to_show=50)
    
    print("\nDone! Check these files:")
    print("  - modularity_evolution.png")
    print("  - final_modularity.png") 
    print("  - scar_convergence.png")
    print("  - modularity_detail.png")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def load_results():
    if not os.path.exists('degree_estimation_results.json'):
        print("Results file not found. Run the Go experiment first.")
        sys.exit(1)
    
    with open('degree_estimation_results.json', 'r') as f:
        return json.load(f)

def analyze_results(data):
    results = []
    
    for exp in data['results']:
        config = exp['config']
        node_results = exp['node_results']
        
        # Calculate metrics
        true_degrees = [nr['true_degree'] for nr in node_results]
        sketch_estimates = [nr['sketch_estimate'] for nr in node_results]
        edge_estimates = [nr['edge_by_edge_estimate'] for nr in node_results]
        
        # Filter finite values
        finite_mask = [not (np.isinf(s) or np.isnan(s) or np.isinf(e) or np.isnan(e)) 
                      for s, e in zip(sketch_estimates, edge_estimates)]
        
        if not any(finite_mask):
            continue
            
        true_finite = [t for t, m in zip(true_degrees, finite_mask) if m]
        sketch_finite = [s for s, m in zip(sketch_estimates, finite_mask) if m]
        edge_finite = [e for e, m in zip(edge_estimates, finite_mask) if m]
        
        sketch_mae = np.mean([abs(t - s) for t, s in zip(true_finite, sketch_finite)])
        edge_mae = np.mean([abs(t - e) for t, e in zip(true_finite, edge_finite)])
        
        results.append({
            'n_nodes': config['num_nodes'],
            'edge_prob': config['edge_prob'],
            'k': config['k'],
            'nk': config['nk'],
            'sketch_mae': sketch_mae,
            'edge_mae': edge_mae,
            'avg_degree': exp['graph_info']['avg_degree'],
            'density': exp['graph_info']['density'],
            'run_id': config.get('run_id', 0),
            'runtime_ms': exp['runtime_ms']
        })
    
    return pd.DataFrame(results)

def create_plots(df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. MAE vs Graph Size
    df_grouped = df.groupby(['n_nodes', 'edge_prob']).mean().reset_index()
    
    for prob in sorted(df['edge_prob'].unique()):
        prob_data = df_grouped[df_grouped['edge_prob'] == prob]
        ax1.plot(prob_data['n_nodes'], prob_data['sketch_mae'], 
                'o-', label=f'Sketch p={prob}', linewidth=2)
        ax1.plot(prob_data['n_nodes'], prob_data['edge_mae'], 
                's--', label=f'Edge p={prob}', linewidth=2)
    
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Performance vs Graph Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE vs K values
    k_grouped = df.groupby('k').mean().reset_index()
    
    ax2.plot(k_grouped['k'], k_grouped['sketch_mae'], 'o-', 
            label='Sketch-based', linewidth=2, markersize=8)
    ax2.plot(k_grouped['k'], k_grouped['edge_mae'], 's-', 
            label='Edge-by-edge', linewidth=2, markersize=8)
    
    ax2.set_xlabel('K Value')
    ax2.set_ylabel('Average MAE')
    ax2.set_title('Performance vs Sketch Size (K)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # 3. Performance vs Density
    density_grouped = df.groupby('edge_prob').mean().reset_index()
    
    ax3.plot(density_grouped['edge_prob'], density_grouped['sketch_mae'], 
            'o-', label='Sketch-based', linewidth=2, markersize=8)
    ax3.plot(density_grouped['edge_prob'], density_grouped['edge_mae'], 
            's-', label='Edge-by-edge', linewidth=2, markersize=8)
    
    ax3.set_xlabel('Edge Probability')
    ax3.set_ylabel('Average MAE')
    ax3.set_title('Performance vs Graph Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Relative Performance
    # Handle cases where edge_mae is 0 (perfect accuracy)
    df['relative_performance'] = df.apply(lambda row: 
        row['sketch_mae'] / row['edge_mae'] if row['edge_mae'] > 0 else 
        (0 if row['sketch_mae'] == 0 else float('inf')), axis=1)
    
    # Filter out infinite values for plotting
    finite_perf = df[np.isfinite(df['relative_performance'])]
    
    if len(finite_perf) > 0:
        configs = [f"N={row['n_nodes']}, p={row['edge_prob']:.1f}, K={row['k']}" 
                  for _, row in finite_perf.iterrows()]
        
        colors = ['green' if r < 1 else 'red' for r in finite_perf['relative_performance']]
        bars = ax4.bar(range(len(finite_perf)), finite_perf['relative_performance'], 
                      color=colors, alpha=0.7)
        
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Sketch MAE / Edge MAE')
        ax4.set_title('Relative Performance\n(< 1 means Sketch is better)')
        ax4.set_xticks(range(len(configs)))
        ax4.set_xticklabels(configs, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No finite relative\nperformance data', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig('degree_estimation_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to: degree_estimation_results.png")

def print_summary(df):
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"Total experiments: {len(df)}")
    print(f"Graph sizes: {sorted(df['n_nodes'].unique())}")
    print(f"Edge probabilities: {sorted(df['edge_prob'].unique())}")
    print(f"K values: {sorted(df['k'].unique())}")
    print(f"NK values: {sorted(df['nk'].unique())}")
    
    # Count wins, handling ties when both MAEs are 0
    sketch_wins = sum((df['sketch_mae'] < df['edge_mae']) & (df['edge_mae'] > 0))
    edge_wins = sum((df['edge_mae'] < df['sketch_mae']) & (df['sketch_mae'] > 0))
    ties = sum((df['sketch_mae'] == df['edge_mae']))
    total = len(df)
    
    sketch_win_rate = sketch_wins / total * 100 if total > 0 else 0
    edge_win_rate = edge_wins / total * 100 if total > 0 else 0
    tie_rate = ties / total * 100 if total > 0 else 0
    
    print(f"\nMethod Performance:")
    print(f"Sketch method wins: {sketch_wins}/{total} ({sketch_win_rate:.1f}%)")
    print(f"Edge method wins: {edge_wins}/{total} ({edge_win_rate:.1f}%)")
    print(f"Ties (both perfect): {ties}/{total} ({tie_rate:.1f}%)")
    
    print(f"\nAverage Performance:")
    print(f"Average sketch MAE: {df['sketch_mae'].mean():.4f}")
    print(f"Average edge MAE: {df['edge_mae'].mean():.4f}")
    
    # Find best performance for each method
    if not df.empty:
        best_sketch = df.loc[df['sketch_mae'].idxmin()]
        best_edge = df.loc[df['edge_mae'].idxmin()]
        
        print(f"\nBest Sketch Performance:")
        print(f"  Config: N={best_sketch['n_nodes']}, p={best_sketch['edge_prob']:.1f}, "
              f"K={best_sketch['k']}, MAE={best_sketch['sketch_mae']:.4f}")
        
        print(f"\nBest Edge Performance:")
        print(f"  Config: N={best_edge['n_nodes']}, p={best_edge['edge_prob']:.1f}, "
              f"K={best_edge['k']}, MAE={best_edge['edge_mae']:.4f}")
        
        # Analysis of when estimation is actually happening
        estimation_cases = df[df['sketch_mae'] > 0]
        if not estimation_cases.empty:
            print(f"\nEstimation Cases (MAE > 0): {len(estimation_cases)}/{len(df)}")
            print(f"  Avg sketch MAE: {estimation_cases['sketch_mae'].mean():.4f}")
            print(f"  Avg edge MAE: {estimation_cases['edge_mae'].mean():.4f}")
        else:
            print(f"\nNo estimation cases found (all sketches using exact calculation)")

def create_results_table(df):
    print("\n" + "="*100)
    print("DEGREE ESTIMATION RESULTS TABLE")
    print("="*100)
    
    # Sort by edge_prob, then by K for better readability
    df_sorted = df.sort_values(['edge_prob', 'k', 'run_id'])
    
    print(f"{'Nodes':<6} {'EdgeProb':<9} {'K':<6} {'NK':<3} {'Run':<3} {'AvgDeg':<7} {'SketchMAE':<10} {'EdgeMAE':<10} {'Winner':<10}")
    print("-" * 100)
    
    for _, row in df_sorted.iterrows():
        # Determine winner
        if row['sketch_mae'] < row['edge_mae']:
            winner = "Sketch"
        elif row['edge_mae'] < row['sketch_mae']:
            winner = "Edge"
        else:
            winner = "Tie"
        
        print(f"{row['n_nodes']:<6} {row['edge_prob']:<9.1f} {row['k']:<6} {row['nk']:<3} {row['run_id']:<3} "
              f"{row['avg_degree']:<7.1f} {row['sketch_mae']:<10.4f} {row['edge_mae']:<10.4f} {winner:<10}")

def create_summary_table(df):
    print("\n" + "="*80)
    print("SUMMARY BY CONFIGURATION")
    print("="*80)
    
    # Group by configuration (excluding run_id) and calculate averages
    summary = df.groupby(['n_nodes', 'edge_prob', 'k', 'nk']).agg({
        'sketch_mae': ['mean', 'std'],
        'edge_mae': ['mean', 'std'],
        'avg_degree': 'mean'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['sketch_mae_mean', 'sketch_mae_std', 'edge_mae_mean', 'edge_mae_std', 'avg_degree']
    summary = summary.reset_index()
    
    print(f"{'Nodes':<6} {'EdgeProb':<9} {'K':<6} {'AvgDeg':<7} {'Sketch_MAE':<12} {'Edge_MAE':<12} {'Winner':<10}")
    print("-" * 80)
    
    for _, row in summary.iterrows():
        # Determine winner
        if row['sketch_mae_mean'] < row['edge_mae_mean']:
            winner = "Sketch"
        elif row['edge_mae_mean'] < row['sketch_mae_mean']:
            winner = "Edge"
        else:
            winner = "Tie"
        
        sketch_display = f"{row['sketch_mae_mean']:.4f}"
        if row['sketch_mae_std'] > 0:
            sketch_display += f"±{row['sketch_mae_std']:.3f}"
            
        edge_display = f"{row['edge_mae_mean']:.4f}"
        if row['edge_mae_std'] > 0:
            edge_display += f"±{row['edge_mae_std']:.3f}"
        
        print(f"{row['n_nodes']:<6} {row['edge_prob']:<9.1f} {row['k']:<6} {row['avg_degree']:<7.1f} "
              f"{sketch_display:<12} {edge_display:<12} {winner:<10}")

def highlight_interesting_cases(df):
    print("\n" + "="*80)
    print("INTERESTING CASES (WHERE ESTIMATION ACTUALLY HAPPENS)")
    print("="*80)
    
    # Filter cases where either method has non-zero error
    interesting = df[(df['sketch_mae'] > 0.001) | (df['edge_mae'] > 0.001)]
    
    if interesting.empty:
        print("No cases found where sketches were used for estimation (all exact calculations)")
        return
    
    print(f"Found {len(interesting)} cases with actual estimation:")
    print()
    print(f"{'EdgeProb':<9} {'K':<6} {'AvgDeg':<7} {'SketchMAE':<10} {'EdgeMAE':<10} {'Improvement':<12}")
    print("-" * 70)
    
    for _, row in interesting.iterrows():
        improvement = "N/A"
        if row['edge_mae'] > 0:
            improvement_ratio = (row['edge_mae'] - row['sketch_mae']) / row['edge_mae'] * 100
            improvement = f"{improvement_ratio:+.1f}%"
        
        print(f"{row['edge_prob']:<9.1f} {row['k']:<6} {row['avg_degree']:<7.1f} "
              f"{row['sketch_mae']:<10.4f} {row['edge_mae']:<10.4f} {improvement:<12}")

def main():
    print("Loading SCAR degree estimation experiment results...")
    data = load_results()
    df = analyze_results(data)
    
    if df.empty:
        print("No valid results to analyze!")
        return
    
    # Show detailed results table
    create_results_table(df)
    
    # Show summary by configuration
    create_summary_table(df)
    
    # Highlight interesting cases
    highlight_interesting_cases(df)
    
    # Show overall summary
    print_summary(df)
    
    print("\nTable analysis complete!")

if __name__ == "__main__":
    main()
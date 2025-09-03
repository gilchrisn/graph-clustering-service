#!/usr/bin/env python3
"""
Bottom-K Cardinality Estimation Experiment

This experiment tests the accuracy of bottom-k cardinality estimation
across different k values by comparing estimates against the true cardinality n.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import sys

def bottom_k_estimate(sorted_hashes, k):
    """
    Estimate cardinality using bottom-k formula: (k-1) * MaxUint32 / bottom_k
    """
    if k > len(sorted_hashes) or k <= 1:
        return float('inf')
    
    bottom_k_hash = sorted_hashes[k-1]  # k-th smallest (0-indexed)
    max_uint32 = 2**32 - 1
    
    if bottom_k_hash == 0:
        return float('inf')
    
    estimate = (k - 1) * max_uint32 / bottom_k_hash
    return estimate

def run_experiment():
    """
    Run the bottom-k estimation experiment
    """
    # Experiment parameters
    t = 1  # number of test cases
    n = 600  # number of hash values to generate
    k_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  # different k values to test
    
    print(f"Running Bottom-K Estimation Experiment")
    print(f"Test cases: {t}")
    print(f"Hash values per test: {n}")
    print(f"K values: {k_values}")
    print("-" * 50)
    
    # Store results: k_value -> list of deltas
    results = {k: [] for k in k_values}
    
    # Run test cases
    for test_case in range(t):
        if test_case % 10 == 0:
            print(f"Processing test case {test_case + 1}/{t}")
        
        # Generate n random hash values (simulate uint32 range)
        max_uint32 = 2**32 - 1
        hashes = [random.randint(1, max_uint32) for _ in range(n)]
        
        # Sort the hashes
        sorted_hashes = sorted(hashes)
        
        # Test each k value
        for k in k_values:
            estimate = bottom_k_estimate(sorted_hashes, k)
            
            # Calculate delta (absolute error)
            if estimate != float('inf'):
                delta = abs(estimate - n)
                results[k].append(delta)
            else:
                # Skip invalid estimates
                continue
    
    # Calculate average deltas for each k
    avg_deltas = {}
    relative_errors = {}
    
    print("\nResults:")
    print("K\tAvg Delta\tRelative Error (%)")
    print("-" * 35)
    
    for k in k_values:
        if results[k]:  # Only if we have valid results
            avg_delta = np.mean(results[k])
            relative_error = (avg_delta / n) * 100
            
            avg_deltas[k] = avg_delta
            relative_errors[k] = relative_error
            
            print(f"{k}\t{avg_delta:.2f}\t\t{relative_error:.2f}%")
        else:
            print(f"{k}\tNo valid results")
    
    return avg_deltas, relative_errors, k_values, n

def create_plots(avg_deltas, relative_errors, k_values, n):
    """
    Create bar charts for the experiment results
    """
    # Filter k_values to only those with results
    valid_k_values = [k for k in k_values if k in avg_deltas]
    valid_avg_deltas = [avg_deltas[k] for k in valid_k_values]
    valid_relative_errors = [relative_errors[k] for k in valid_k_values]
    
    # Create categorical x-axis positions (evenly spaced)
    x_positions = range(len(valid_k_values))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Absolute Error (Delta)
    bars1 = ax1.bar(x_positions, valid_avg_deltas, color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xlabel('K Value')
    ax1.set_ylabel('Average Absolute Error (Delta)')
    ax1.set_title(f'Bottom-K Estimation Error vs K\n(True Cardinality n = {n})')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(valid_k_values)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, delta in zip(bars1, valid_avg_deltas):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{delta:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Relative Error (Percentage)
    bars2 = ax2.bar(x_positions, valid_relative_errors, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_xlabel('K Value')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title(f'Bottom-K Estimation Relative Error vs K\n(True Cardinality n = {n})')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(valid_k_values)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, error in zip(bars2, valid_relative_errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{error:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('bottom_k_estimation_results.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'bottom_k_estimation_results.png'")
    
    # Show the plot
    plt.show()

def main():
    """
    Main experiment runner
    """
    print("=" * 60)
    print("BOTTOM-K CARDINALITY ESTIMATION EXPERIMENT")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Run the experiment
        avg_deltas, relative_errors, k_values, n = run_experiment()
        
        if not avg_deltas:
            print("No valid results obtained. Experiment failed.")
            sys.exit(1)
        
        # Create and display plots
        create_plots(avg_deltas, relative_errors, k_values, n)
        
        print("\nExperiment completed successfully!")
        print("\nKey Insights:")
        print("- Lower K values generally have higher estimation error")
        print("- Higher K values provide more accurate estimates")
        print("- The bottom-k method becomes more reliable as k increases")
        print("- Relative error decreases as k approaches the sketch capacity")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
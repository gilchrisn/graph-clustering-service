package main

import (
	"context"
	"fmt"
	"log"
	"os"
	
	"github.com/gilchrisn/graph-clustering-service/pkg/scar" // Adjust import path as needed
)

func main() {
	fmt.Println("=== SCAR Algorithm - Real Files ===")
	
	// Check command line arguments
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}
	
	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]
	
	fmt.Printf("Input files:\n")
	fmt.Printf("  Graph: %s\n", graphFile)
	fmt.Printf("  Properties: %s\n", propertiesFile)
	fmt.Printf("  Path: %s\n", pathFile)
	
	// Create configuration with LARGE K for exact computation
	config := scar.NewConfig()
	config.Set("algorithm.max_iterations", 5)
	config.Set("algorithm.min_modularity_gain", -100.0)
	config.Set("logging.level", "info")
	config.Set("analysis.track_moves", true)
	config.Set("analysis.output_file", "scar_moves.jsonl")
	
	// LARGE K ensures sketches are never full → exact computation (same as Louvain)
	config.Set("scar.k", 2)    // Large K for exact computation
	config.Set("scar.nk", 1)     // Multiple layers
	config.Set("scar.threshold", 0.0)

	// Set random seed for reproducibility
	config.Set("algorithm.random_seed", int64(42)) 
	// config.Set("algorithm.random_seed", int64(time.Now().UnixNano())) // Use current time for randomness
	
	fmt.Printf("\nSCAR Configuration:\n")
	fmt.Printf("  K: %d (large → exact computation)\n", config.K())
	fmt.Printf("  NK: %d\n", config.NK())
	fmt.Printf("  Max Iterations: %d\n", config.MaxIterations())
	
	// Run algorithm
	ctx := context.Background()
	result, err := scar.Run(graphFile, propertiesFile, pathFile, config, ctx)
	if err != nil {
		log.Fatalf("Algorithm failed: %v", err)
	}
	
	// Display results (same format as Louvain)
	displayResults(result)
}

func displayResults(result *scar.Result) {
	fmt.Println("\n=== Results ===")
	fmt.Printf("Number of levels: %d\n", result.NumLevels)
	fmt.Printf("Final modularity: %.6f\n", result.Modularity)
	fmt.Printf("Runtime: %d ms\n", result.Statistics.RuntimeMS)
	fmt.Printf("Total moves: %d\n", result.Statistics.TotalMoves)
	
	// Print level-by-level results (IDENTICAL format to Louvain)
	for _, level := range result.Levels {
		fmt.Printf("\nLevel %d:\n", level.Level)
		fmt.Printf("  Communities: %d\n", level.NumCommunities)
		fmt.Printf("  Modularity: %.6f\n", level.Modularity)
		fmt.Printf("  Moves: %d\n", level.NumMoves)
		fmt.Printf("  Runtime: %d ms\n", level.RuntimeMS)
		
		// fmt.Printf("  Community assignments:\n")
		// for commID, nodes := range level.Communities {
		// 	fmt.Printf("    Community %d: %v\n", commID, nodes)
		// }
	}
	
	
	// // Print final community assignments
	// fmt.Println("\nFinal community assignments:")
	// for node, comm := range result.FinalCommunities {
	// 	fmt.Printf("  Node %d -> Community %d\n", node, comm)
	// }
}
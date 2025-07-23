
package main

import (
	"context"
	"fmt"
	"log"
	
	"graph-clustering-service/pkg2/louvain" // Adjust import path as needed
)

func main() {
	fmt.Println("=== Louvain Algorithm - Clean Rewrite ===")
	
	// Create test graph
	graph := createTestGraph()
	
	// Create configuration
	config := louvain.NewConfig()
	config.Set("algorithm.max_iterations", 50)
	config.Set("algorithm.min_modularity_gain", 1e-6)
	config.Set("logging.level", "info")
	
	// Run algorithm
	ctx := context.Background()
	result, err := louvain.Run(graph, config, ctx)
	if err != nil {
		log.Fatalf("Algorithm failed: %v", err)
	}
	
	// Display results
	displayResults(result)
}

func createTestGraph() *louvain.Graph {
	// Create same test graph as before
	graph := louvain.NewGraph(6)
	
	edges := [][3]float64{
		{0, 1, 1.0}, {1, 2, 1.0}, {0, 3, 1.0},
		{1, 4, 1.0}, {2, 5, 1.0}, {3, 4, 1.0},
	}
	
	fmt.Println("Creating graph with edges:")
	for _, edge := range edges {
		u, v, w := int(edge[0]), int(edge[1]), edge[2]
		if err := graph.AddEdge(u, v, w); err != nil {
			log.Printf("Failed to add edge %d-%d: %v", u, v, err)
			continue
		}
		fmt.Printf("  %d -- %d (weight: %.1f)\n", u, v, w)
	}
	
	fmt.Printf("\nGraph: %d nodes, total weight: %.1f\n", graph.NumNodes, graph.TotalWeight)
	return graph
}

func displayResults(result *louvain.Result) {
	fmt.Println("\n=== Results ===")
	fmt.Printf("Number of levels: %d\n", result.NumLevels)
	fmt.Printf("Final modularity: %.6f\n", result.Modularity)
	fmt.Printf("Runtime: %d ms\n", result.Statistics.RuntimeMS)
	fmt.Printf("Total moves: %d\n", result.Statistics.TotalMoves)
	
	// Print level-by-level results
	for _, level := range result.Levels {
		fmt.Printf("\nLevel %d:\n", level.Level)
		fmt.Printf("  Communities: %d\n", level.NumCommunities)
		fmt.Printf("  Modularity: %.6f\n", level.Modularity)
		fmt.Printf("  Moves: %d\n", level.NumMoves)
		fmt.Printf("  Runtime: %d ms\n", level.RuntimeMS)
		
		fmt.Printf("  Community assignments:\n")
		for commID, nodes := range level.Communities {
			fmt.Printf("    Community %d: %v\n", commID, nodes)
		}
	}
	
	// Print final community assignments
	fmt.Println("\nFinal community assignments:")
	for node := 0; node < 6; node++ { // Assuming 6 nodes
		if comm, exists := result.FinalCommunities[node]; exists {
			fmt.Printf("  Node %d -> Community %d\n", node, comm)
		}
	}
}
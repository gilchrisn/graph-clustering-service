package main

import (
	"fmt"
	"log"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
)

func main() {
	fmt.Println("=== Louvain Algorithm Test ===")
	
	// Create a simple 6-node graph manually
	graph := louvain.NewNormalizedGraph(6)
	
	// Add edges with weight 1.0
	edges := [][2]int{
		{0, 1}, {1, 2}, {0, 3}, {1, 4}, {2, 5}, {3, 4},
	}
	
	fmt.Println("Creating graph with edges:")
	for _, edge := range edges {
		graph.AddEdge(edge[0], edge[1], 1.0)
		fmt.Printf("  %d -- %d (weight: 1.0)\n", edge[0], edge[1])
	}
	
	fmt.Printf("\nGraph created with %d nodes, total weight: %.1f\n", 
		graph.NumNodes, graph.TotalWeight)
	
	// Print node degrees
	fmt.Println("\nNode degrees:")
	for i := 0; i < graph.NumNodes; i++ {
		fmt.Printf("  Node %d: degree %.1f\n", i, graph.GetNodeDegree(i))
	}
	
	// Configure Louvain
	config := louvain.DefaultLouvainConfig()
	config.MaxIterations = 10
	config.MinModularity = 0.001
	config.Verbose = true
	
	// Add progress callback for detailed output
	config.ProgressCallback = func(level, iteration int, message string) {
		fmt.Printf("[PROGRESS] %s\n", message)
	}
	
	fmt.Println("\n=== Running Louvain Algorithm ===")
	
	// Run Louvain
	result, err := louvain.RunLouvain(graph, config)
	if err != nil {
		log.Fatalf("Louvain failed: %v", err)
	}
	
	fmt.Println("\n=== Results ===")
	fmt.Printf("Number of levels: %d\n", result.NumLevels)
	fmt.Printf("Final modularity: %.6f\n", result.Modularity)
	fmt.Printf("Runtime: %d ms\n", result.Statistics.RuntimeMS)
	
	// Print level-by-level results
	for i, level := range result.Levels {
		fmt.Printf("\nLevel %d:\n", i)
		fmt.Printf("  Communities: %d\n", level.NumCommunities)
		fmt.Printf("  Modularity: %.6f\n", level.Modularity)
		fmt.Printf("  Moves: %d\n", level.NumMoves)
		
		fmt.Printf("  Community assignments:\n")
		for commID, nodes := range level.Communities {
			fmt.Printf("    Community %d: %v\n", commID, nodes)
		}
	}
	
	// Print final community assignments
	fmt.Println("\nFinal community assignments:")
	for node := 0; node < graph.NumNodes; node++ {
		if comm, exists := result.FinalCommunities[node]; exists {
			fmt.Printf("  Node %d -> Community %d\n", node, comm)
		}
	}
	
	fmt.Printf("\nAlgorithm Statistics:\n")
	fmt.Printf("  Total iterations: %d\n", result.Statistics.TotalIterations)
	fmt.Printf("  Total moves: %d\n", result.Statistics.TotalMoves)
	fmt.Printf("  Memory peak: %d MB\n", result.Statistics.MemoryPeakMB)
	
	for i, levelStats := range result.Statistics.LevelStats {
		fmt.Printf("  Level %d: %d iterations, %d moves, %.2f -> %.6f modularity\n",
			i, levelStats.Iterations, levelStats.Moves, 
			levelStats.InitialModularity, levelStats.FinalModularity)
	}
	
	fmt.Println("\n=== Test Complete ===")
}
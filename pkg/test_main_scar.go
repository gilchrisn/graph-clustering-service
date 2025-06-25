package main

import (
	"fmt"
	"log"
	"time"
	
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

func main() {
	fmt.Println("=== Direct SCAR Test ===")
	
	// Create the heterogeneous graph that materializes to your target
	graph := createTestHeterogeneousGraph()
	
	// Print graph statistics
	fmt.Println("\n=== Heterogeneous Graph Statistics ===")
	graph.PrintStatistics()
	
	// Validate the graph
	if err := graph.ValidateConsistency(); err != nil {
		log.Fatalf("Graph validation failed: %v", err)
	}
	fmt.Println("✓ Heterogeneous graph validation passed")
	
	// Create meta-path for Author-Paper-Author
	metaPath := &scar.MetaPath{
		ID:          "author_paper_author",
		NodeTypes:   []string{"Author", "Paper", "Author"},
		EdgeTypes:   []string{"writes", "writes"},
		Description: "Authors connected through co-authored papers",
	}
	
	fmt.Printf("\n=== Meta-Path: %s ===\n", metaPath.String())
	fmt.Printf("Valid: %v, Symmetric: %v\n", metaPath.IsValid(), metaPath.IsSymmetric())
	
	// Configure SCAR
	config := scar.ScarConfig{
		K:             64,  // Smaller for testing
		NK:            4,
		MetaPath:      metaPath,
		MaxIterations: 20,
		MinModularity: 1e-6,
		RandomSeed:    42,
		Verbose:       true,
		Parallel: scar.ParallelConfig{
			NumWorkers:   2,
			BatchSize:    100,
			UpdateBuffer: 1000,
			Enabled:      true,
		},
	}
	
	fmt.Printf("\n=== Expected Materialized Graph ===\n")
	printExpectedMaterialization()
	
	// Run SCAR
	fmt.Printf("\n=== Running SCAR ===\n")
	startTime := time.Now()
	
	result, err := scar.RunScar(graph, config)
	if err != nil {
		log.Fatalf("SCAR failed: %v", err)
	}
	
	duration := time.Since(startTime)
	fmt.Printf("✓ SCAR completed in %v\n", duration)
	
	// Print results
	fmt.Printf("\n=== Results ===\n")
	fmt.Printf("Final modularity: %.6f\n", result.Modularity)
	fmt.Printf("Number of levels: %d\n", result.NumLevels)
	fmt.Printf("Total iterations: %d\n", result.Statistics.TotalIterations)
	
	fmt.Printf("\n=== Community Assignments ===\n")
	for originalID, community := range result.FinalCommunities {
		fmt.Printf("Author %s -> Community %d\n", originalID, community)
	}
	
	// Debug: Print sketch information  
	fmt.Printf("\n=== Debug Information ===\n")
	printBasicDebugInfo(graph, config)
}

// Creates a heterogeneous graph that materializes to the target normalized graph
func createTestHeterogeneousGraph() *scar.NormalizedGraph {
	/*
	Target materialized graph (Author-Author connections):
	{0,1}, {1,2}, {0,3}, {1,4}, {2,5}, {3,4}
	
	This means we need a bipartite Author-Paper graph where:
	- Authors 0,1 share a paper (creates edge 0-1)
	- Authors 1,2 share a paper (creates edge 1-2) 
	- Authors 0,3 share a paper (creates edge 0-3)
	- Authors 1,4 share a paper (creates edge 1-4)
	- Authors 2,5 share a paper (creates edge 2-5)
	- Authors 3,4 share a paper (creates edge 3-4)
	
	Bipartite structure:
	Paper p0: written by Authors a0, a1
	Paper p1: written by Authors a1, a2  
	Paper p2: written by Authors a0, a3
	Paper p3: written by Authors a1, a4
	Paper p4: written by Authors a2, a5
	Paper p5: written by Authors a3, a4
	*/
	
	ng := &scar.NormalizedGraph{
		NodesByType:     make(map[string]int),
		NodeToIndex:     make(map[string]map[string]int),
		IndexToNode:     make(map[string][]string),
		EdgeTypeToIndex: make(map[string]int),
		IndexToEdgeType: []string{"writes"},
		OutEdges:        make(map[string][][][]scar.EdgeTarget),
		InEdges:         make(map[string][][][]scar.EdgeTarget),
		NormalizedNodes: make([]scar.NormalizedNode, 0),
		NormalizedEdges: make([]scar.NormalizedEdge, 0),
	}
	
	// Setup node types
	ng.NodesByType["Author"] = 6  // a0, a1, a2, a3, a4, a5
	ng.NodesByType["Paper"] = 6   // p0, p1, p2, p3, p4, p5
	
	// Setup edge types
	ng.EdgeTypeToIndex["writes"] = 0
	
	// Initialize node mappings
	ng.NodeToIndex["Author"] = make(map[string]int)
	ng.NodeToIndex["Paper"] = make(map[string]int)
	ng.IndexToNode["Author"] = make([]string, 6)
	ng.IndexToNode["Paper"] = make([]string, 6)
	
	// Create Author nodes
	for i := 0; i < 6; i++ {
		authorID := fmt.Sprintf("a%d", i)
		ng.NodeToIndex["Author"][authorID] = i
		ng.IndexToNode["Author"][i] = authorID
		
		ng.NormalizedNodes = append(ng.NormalizedNodes, scar.NormalizedNode{
			OriginalID: authorID,
			Type:       "Author",
			Index:      i,
			Properties: make(map[string]interface{}),
		})
	}
	
	// Create Paper nodes  
	for i := 0; i < 6; i++ {
		paperID := fmt.Sprintf("p%d", i)
		ng.NodeToIndex["Paper"][paperID] = i
		ng.IndexToNode["Paper"][i] = paperID
		
		ng.NormalizedNodes = append(ng.NormalizedNodes, scar.NormalizedNode{
			OriginalID: paperID,
			Type:       "Paper", 
			Index:      i,
			Properties: make(map[string]interface{}),
		})
	}
	
	// Initialize adjacency lists
	ng.OutEdges["Author"] = make([][][]scar.EdgeTarget, 6)
	ng.InEdges["Author"] = make([][][]scar.EdgeTarget, 6)
	ng.OutEdges["Paper"] = make([][][]scar.EdgeTarget, 6)
	ng.InEdges["Paper"] = make([][][]scar.EdgeTarget, 6)
	
	for i := 0; i < 6; i++ {
		ng.OutEdges["Author"][i] = make([][]scar.EdgeTarget, 1)
		ng.InEdges["Author"][i] = make([][]scar.EdgeTarget, 1)
		ng.OutEdges["Paper"][i] = make([][]scar.EdgeTarget, 1)
		ng.InEdges["Paper"][i] = make([][]scar.EdgeTarget, 1)
		
		ng.OutEdges["Author"][i][0] = make([]scar.EdgeTarget, 0)
		ng.InEdges["Author"][i][0] = make([]scar.EdgeTarget, 0)
		ng.OutEdges["Paper"][i][0] = make([]scar.EdgeTarget, 0)
		ng.InEdges["Paper"][i][0] = make([]scar.EdgeTarget, 0)
	}
	
	// Create the bipartite Author-Paper edges
	authorPaperEdges := [][2]int{
		// Paper p0: Authors a0, a1 (creates Author edge 0-1)
		{0, 0}, {1, 0},
		// Paper p1: Authors a1, a2 (creates Author edge 1-2)
		{1, 1}, {2, 1},
		// Paper p2: Authors a0, a3 (creates Author edge 0-3)
		{0, 2}, {3, 2},
		// Paper p3: Authors a1, a4 (creates Author edge 1-4)
		{1, 3}, {4, 3},
		// Paper p4: Authors a2, a5 (creates Author edge 2-5)
		{2, 4}, {5, 4},
		// Paper p5: Authors a3, a4 (creates Author edge 3-4)
		{3, 5}, {4, 5},
	}
	
	// Add edges to adjacency lists and normalized edges
	for _, edge := range authorPaperEdges {
		authorIdx := edge[0]
		paperIdx := edge[1]
		
		// Author -> Paper edge
		target := scar.EdgeTarget{
			TargetType:  "Paper",
			TargetIndex: paperIdx,
		}
		ng.OutEdges["Author"][authorIdx][0] = append(ng.OutEdges["Author"][authorIdx][0], target)
		
		// Paper <- Author edge (incoming)
		source := scar.EdgeTarget{
			TargetType:  "Author", 
			TargetIndex: authorIdx,
		}
		ng.InEdges["Paper"][paperIdx][0] = append(ng.InEdges["Paper"][paperIdx][0], source)
		
		// Add to normalized edges
		ng.NormalizedEdges = append(ng.NormalizedEdges, scar.NormalizedEdge{
			FromType:     "Author",
			ToType:       "Paper",
			FromIndex:    authorIdx,
			ToIndex:      paperIdx,
			Type:         "writes",
			Weight:       1.0,
			OriginalFrom: fmt.Sprintf("a%d", authorIdx),
			OriginalTo:   fmt.Sprintf("p%d", paperIdx),
		})
	}
	
	return ng
}

func printExpectedMaterialization() {
	fmt.Println("When materialized via Author-Paper-Author meta-path:")
	expectedEdges := [][2]int{
		{0, 1}, {1, 2}, {0, 3}, {1, 4}, {2, 5}, {3, 4},
	}
	
	fmt.Println("Expected Author-Author edges:")
	for _, edge := range expectedEdges {
		fmt.Printf("  a%d -- a%d\n", edge[0], edge[1])
	}
	
	fmt.Println("\nReasoning:")
	fmt.Println("  p0: a0,a1 -> edge a0-a1")
	fmt.Println("  p1: a1,a2 -> edge a1-a2") 
	fmt.Println("  p2: a0,a3 -> edge a0-a3")
	fmt.Println("  p3: a1,a4 -> edge a1-a4")
	fmt.Println("  p4: a2,a5 -> edge a2-a5")
	fmt.Println("  p5: a3,a4 -> edge a3-a4")
}

func printBasicDebugInfo(graph *scar.NormalizedGraph, config scar.ScarConfig) {
	fmt.Printf("Graph structure:\n")
	fmt.Printf("  Authors: %d\n", graph.NodesByType["Author"])
	fmt.Printf("  Papers: %d\n", graph.NodesByType["Paper"]) 
	fmt.Printf("  Total edges: %d\n", len(graph.NormalizedEdges))
	
	fmt.Printf("\nActual Author-Paper connections:\n")
	for i := 0; i < graph.NodesByType["Author"]; i++ {
		papers := graph.GetNeighborsFast("Author", i, "writes")
		fmt.Printf("  Author a%d writes papers: ", i)
		for _, paper := range papers {
			fmt.Printf("p%d ", paper.TargetIndex)
		}
		fmt.Printf("\n")
	}
	
	fmt.Printf("\nExpected materialization verification:\n")
	// Manually check which authors should be connected
	expectedPairs := [][2]int{{0,1}, {1,2}, {0,3}, {1,4}, {2,5}, {3,4}}
	for _, pair := range expectedPairs {
		author1, author2 := pair[0], pair[1]
		
		// Find shared papers
		papers1 := graph.GetNeighborsFast("Author", author1, "writes")
		papers2 := graph.GetNeighborsFast("Author", author2, "writes")
		
		sharedPapers := []int{}
		for _, p1 := range papers1 {
			for _, p2 := range papers2 {
				if p1.TargetIndex == p2.TargetIndex {
					sharedPapers = append(sharedPapers, p1.TargetIndex)
				}
			}
		}
		
		fmt.Printf("  a%d-a%d: shared papers %v\n", author1, author2, sharedPapers)
	}
}

func printSketchDebugInfo(state *scar.ScarState) {
	// This function is no longer used since we removed the unexported method call
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
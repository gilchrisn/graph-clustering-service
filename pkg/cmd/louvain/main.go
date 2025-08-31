package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	
	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain" // Adjust import path as needed
)

func main() {
	fmt.Println("=== Materialization + Louvain Pipeline ===")
	
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
	
	// Step 1: Parse SCAR input for materialization
	fmt.Println("\nStep 1: Parsing input files...")
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to parse input: %v", err)
	}
	fmt.Printf("  Loaded graph with %d nodes\n", len(graph.Nodes))
	
	// Step 2: Run materialization
	fmt.Println("\nStep 2: Running materialization...")
	config := materialization.DefaultMaterializationConfig()
	config.Aggregation.Strategy = materialization.Average
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
	materializationResult, err := engine.Materialize()
	if err != nil {
		log.Fatalf("Materialization failed: %v", err)
	}
	
	materializedGraph := materializationResult.HomogeneousGraph
	fmt.Printf("  Materialized graph: %d nodes, %d edges\n", 
		len(materializedGraph.Nodes), len(materializedGraph.Edges))

	// Step 3: Convert to Louvain graph format
	fmt.Println("\nStep 3: Converting to Louvain format...")
	louvainGraph := convertToLouvainGraph(materializedGraph)
	fmt.Printf("  Louvain graph: %d nodes, total weight: %.1f\n", 
		louvainGraph.NumNodes, louvainGraph.TotalWeight)
	
	// Step 4: Run Louvain
	fmt.Println("\nStep 4: Running Louvain clustering...")
	louvainConfig := louvain.NewConfig()
	louvainConfig.Set("algorithm.max_iterations", 5)
	louvainConfig.Set("algorithm.min_modularity_gain", -100.0)
	louvainConfig.Set("logging.level", "info")
	louvainConfig.Set("algorithm.random_seed", int64(42))
	
	ctx := context.Background()
	result, err := louvain.Run(louvainGraph, louvainConfig, ctx)
	if err != nil {
		log.Fatalf("Louvain failed: %v", err)
	}
	
	// Display results
	displayResults(result)
}

func convertToLouvainGraph(hgraph *materialization.HomogeneousGraph) *louvain.Graph {
	if len(hgraph.Nodes) == 0 {
		log.Fatal("Empty homogeneous graph")
	}
	
	// Create ordered list of node IDs with intelligent sorting
	nodeList := make([]string, 0, len(hgraph.Nodes))
	for nodeID := range hgraph.Nodes {
		nodeList = append(nodeList, nodeID)
	}
	
	// Sort nodes intelligently (numeric if all are integers, lexicographic otherwise)
	allIntegers := true
	for _, nodeID := range nodeList {
		if _, err := strconv.Atoi(nodeID); err != nil {
			allIntegers = false
			break
		}
	}
	
	if allIntegers {
		sort.Slice(nodeList, func(i, j int) bool {
			a, _ := strconv.Atoi(nodeList[i])
			b, _ := strconv.Atoi(nodeList[j])
			return a < b
		})
	} else {
		sort.Strings(nodeList)
	}
	
	// Create mapping from original IDs to normalized indices
	originalToNormalized := make(map[string]int)
	for i, originalID := range nodeList {
		originalToNormalized[originalID] = i
	}
	
	// Create Louvain graph
	graph := louvain.NewGraph(len(nodeList))
	
	// Add edges with deduplication
	processedEdges := make(map[string]bool)
	for edgeKey, weight := range hgraph.Edges {
		fromNormalized, fromExists := originalToNormalized[edgeKey.From]
		toNormalized, toExists := originalToNormalized[edgeKey.To]
		
		if !fromExists || !toExists {
			log.Printf("Warning: edge references unknown nodes: %s -> %s", edgeKey.From, edgeKey.To)
			continue
		}
		
		// Create canonical edge ID to avoid duplicates
		var canonicalID string
		if fromNormalized <= toNormalized {
			canonicalID = fmt.Sprintf("%d-%d", fromNormalized, toNormalized)
		} else {
			canonicalID = fmt.Sprintf("%d-%d", toNormalized, fromNormalized)
		}
		
		// Only process each undirected edge once
		if !processedEdges[canonicalID] {
			if err := graph.AddEdge(fromNormalized, toNormalized, weight); err != nil {
				log.Printf("Failed to add edge %d-%d: %v", fromNormalized, toNormalized, err)
				continue
			}
			processedEdges[canonicalID] = true
		}
	}
	
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
	for node, comm := range result.FinalCommunities {
		fmt.Printf("  Node %d -> Community %d\n", node, comm)
	}
}

func debugMaterializedGraph(materializedGraph *materialization.HomogeneousGraph) {
	fmt.Printf("\n=== MATERIALIZED GRAPH DEBUG ===\n")
	fmt.Printf("Nodes: %d\n", len(materializedGraph.Nodes))
	fmt.Printf("Edges: %d\n", len(materializedGraph.Edges))
	// fmt.Printf("Total weight from graph field: %.4f\n", materializedGraph.TotalWeight)
	
	// Analyze edge weights
	weightCounts := make(map[float64]int)
	totalWeightSum := 0.0
	selfLoops := 0
	bidirectionalPairs := 0
	unidirectionalEdges := 0
	
	// Create a map to track bidirectional edges
	edgeMap := make(map[string]float64) // "from-to" -> weight
	reverseEdgeMap := make(map[string]bool) // track if reverse exists
	
	for edgeKey, weight := range materializedGraph.Edges {
		// Track weight distribution
		weightCounts[weight]++
		totalWeightSum += weight
		
		// Track self-loops
		if edgeKey.From == edgeKey.To {
			selfLoops++
		}
		
		// Track directionality
		edgeID := fmt.Sprintf("%s-%s", edgeKey.From, edgeKey.To)
		reverseID := fmt.Sprintf("%s-%s", edgeKey.To, edgeKey.From)
		
		edgeMap[edgeID] = weight
		
		// Check if reverse exists
		if _, exists := edgeMap[reverseID]; exists {
			if !reverseEdgeMap[edgeID] && !reverseEdgeMap[reverseID] {
				bidirectionalPairs++
				reverseEdgeMap[edgeID] = true
				reverseEdgeMap[reverseID] = true
			}
		}
	}
	
	// Count unidirectional edges
	for edgeID := range edgeMap {
		if !reverseEdgeMap[edgeID] {
			unidirectionalEdges++
		}
	}
	
	fmt.Printf("\nEdge weight distribution:\n")
	for weight, count := range weightCounts {
		fmt.Printf("  Weight %.4f: %d edges (%.2f%%)\n", 
			weight, count, float64(count)/float64(len(materializedGraph.Edges))*100)
	}
	
	fmt.Printf("\nEdge directionality:\n")
	fmt.Printf("  Self-loops: %d\n", selfLoops)
	fmt.Printf("  Bidirectional pairs: %d (= %d total edges)\n", bidirectionalPairs, bidirectionalPairs*2)
	fmt.Printf("  Unidirectional edges: %d\n", unidirectionalEdges)
	fmt.Printf("  Total edges: %d\n", bidirectionalPairs*2 + unidirectionalEdges + selfLoops)
	
	fmt.Printf("\nWeight analysis:\n")
	fmt.Printf("  Sum of all edge weights: %.4f\n", totalWeightSum)
	fmt.Printf("  Average edge weight: %.4f\n", totalWeightSum/float64(len(materializedGraph.Edges)))
	fmt.Printf("  Expected if all weights = 1.0: %.4f\n", float64(len(materializedGraph.Edges)))
	
	// Sample some edges
	fmt.Printf("\nSample edges (first 10):\n")
	count := 0
	for edgeKey, weight := range materializedGraph.Edges {
		fmt.Printf("  %s -> %s: weight %.4f\n", edgeKey.From, edgeKey.To, weight)
		count++
		if count >= 10 {
			break
		}
	}
	
	fmt.Printf("=================================\n")
}
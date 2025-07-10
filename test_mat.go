// edge_comparison_tool.go - Standalone tool to compare materialization vs SCAR edges

package main

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

func main() {
	if len(os.Args) != 4 {
		fmt.Println("Usage: go run edge_comparison_tool.go <graph_file> <properties_file> <path_file>")
		os.Exit(1)
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("=== Edge Comparison Tool ===")
	fmt.Printf("Graph file: %s\n", graphFile)
	fmt.Printf("Properties file: %s\n", propertiesFile)
	fmt.Printf("Path file: %s\n", pathFile)
	fmt.Println()

	// Step 1: Run materialization to get ground truth edges
	fmt.Println("Step 1: Running materialization...")
	matStart := time.Now()
	materializationEdges, err := runMaterialization(graphFile, propertiesFile, pathFile)
	if err != nil {
		fmt.Printf("Materialization failed: %v\n", err)
		os.Exit(1)
	}
	matTime := time.Since(matStart)
	fmt.Printf("Materialization completed in %v\n", matTime)
	fmt.Printf("Found %d materialization edges\n", len(materializationEdges))
	fmt.Println()

	// Step 2: Run SCAR sketch to get sketch-based edges
	fmt.Println("Step 2: Running SCAR sketch propagation...")
	scarStart := time.Now()
	sketchEdges, err := runSCARSketch(graphFile, propertiesFile, pathFile)
	if err != nil {
		fmt.Printf("SCAR sketch failed: %v\n", err)
		os.Exit(1)
	}
	scarTime := time.Since(scarStart)
	fmt.Printf("SCAR sketch completed in %v\n", scarTime)
	fmt.Printf("Found %d sketch edges\n", len(sketchEdges))
	fmt.Println()

	// Step 3: Compare edge sets
	fmt.Println("Step 3: Comparing edge sets...")
	compareEdges(materializationEdges, sketchEdges)
	
	fmt.Println("\n=== Summary ===")
	fmt.Printf("Materialization time: %v\n", matTime)
	fmt.Printf("SCAR sketch time: %v\n", scarTime)
	fmt.Printf("Speedup: %.2fx\n", float64(matTime.Nanoseconds())/float64(scarTime.Nanoseconds()))
}

// runMaterialization runs the materialization pipeline and extracts edges
func runMaterialization(graphFile, propertiesFile, pathFile string) (map[string]bool, error) {
	// Parse SCAR input files
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
	if err != nil {
		return nil, fmt.Errorf("failed to parse input: %w", err)
	}

	// Run materialization with default config
	config := materialization.DefaultMaterializationConfig()
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
	
	result, err := engine.Materialize()
	if err != nil {
		return nil, fmt.Errorf("materialization failed: %w", err)
	}

	// Extract edges from HomogeneousGraph
	edges := make(map[string]bool)
	for edgeKey := range result.HomogeneousGraph.Edges {
		// Create canonical edge key (lexicographically sorted)
		edgeStr := makeCanonicalEdgeKey(edgeKey.From, edgeKey.To)
		edges[edgeStr] = true
	}

	return edges, nil
}

// runSCARSketch runs SCAR sketch propagation and extracts edges
func runSCARSketch(graphFile, propertiesFile, pathFile string) (map[string]bool, error) {
	// Configure SCAR
	config := scar.SCARConfig{
		GraphFile:    graphFile,
		PropertyFile: propertiesFile,
		PathFile:     pathFile,
		K:            1024,    // Default sketch size
		NK:           1,     // Default number of layers
		Threshold:    0.5,   // Default threshold
		UseLouvain:   false, // We only want sketch computation
		NumWorkers:   1,     // Single threaded for consistency
	}

	// Create SCAR engine
	engine := scar.NewSketchLouvainEngine(config)
	
	// Initialize graph and sketches (this does the sketch propagation)
	err := engine.InitializeGraphAndSketches()
	if err != nil {
		return nil, fmt.Errorf("SCAR initialization failed: %w", err)
	}

	// Extract edges from sketch adjacency list
	edges := make(map[string]bool)
	sketchState := engine.GetSketchLouvainState()
	
	for nodeId, neighbors := range sketchState.GetSketchAdjacencyList() {
		nodeIdStr := strconv.FormatInt(nodeId, 10)
		for _, edge := range neighbors {
			neighborStr := strconv.FormatInt(edge.GetNeighbor(), 10)
			// Create canonical edge key
			edgeStr := makeCanonicalEdgeKey(nodeIdStr, neighborStr)
			edges[edgeStr] = true
		}
	}

	return edges, nil
}

// makeCanonicalEdgeKey creates a canonical edge key with nodes in sorted order
func makeCanonicalEdgeKey(nodeA, nodeB string) string {
	if nodeA <= nodeB {
		return nodeA + "-" + nodeB
	}
	return nodeB + "-" + nodeA
}

// compareEdges compares two edge sets and prints detailed comparison
func compareEdges(materializationEdges, sketchEdges map[string]bool) {
	// Find common edges
	commonEdges := make(map[string]bool)
	for edge := range materializationEdges {
		if sketchEdges[edge] {
			commonEdges[edge] = true
		}
	}

	// Find materialization-only edges (false negatives for sketch)
	matOnlyEdges := make(map[string]bool)
	for edge := range materializationEdges {
		if !sketchEdges[edge] {
			matOnlyEdges[edge] = true
		}
	}

	// Find sketch-only edges (false positives for sketch)
	sketchOnlyEdges := make(map[string]bool)
	for edge := range sketchEdges {
		if !materializationEdges[edge] {
			sketchOnlyEdges[edge] = true
		}
	}

	// Calculate metrics
	totalUnionEdges := len(materializationEdges) + len(sketchEdges) - len(commonEdges)
	jaccard := float64(len(commonEdges)) / float64(totalUnionEdges)
	
	precision := float64(len(commonEdges)) / float64(len(sketchEdges))
	recall := float64(len(commonEdges)) / float64(len(materializationEdges))
	f1 := 2 * precision * recall / (precision + recall)

	// Print results
	fmt.Printf("Edge Comparison Results:\n")
	fmt.Printf("  Materialization edges: %d\n", len(materializationEdges))
	fmt.Printf("  Sketch edges: %d\n", len(sketchEdges))
	fmt.Printf("  Common edges: %d\n", len(commonEdges))
	fmt.Printf("  Materialization-only edges: %d\n", len(matOnlyEdges))
	fmt.Printf("  Sketch-only edges: %d\n", len(sketchOnlyEdges))
	fmt.Printf("\n")
	fmt.Printf("Metrics:\n")
	fmt.Printf("  Jaccard similarity: %.4f\n", jaccard)
	fmt.Printf("  Precision: %.4f (common / sketch_total)\n", precision)
	fmt.Printf("  Recall: %.4f (common / materialization_total)\n", recall)
	fmt.Printf("  F1-score: %.4f\n", f1)
	fmt.Printf("\n")

	// Show sample edges if the lists are not too large
	if len(matOnlyEdges) > 0 && len(matOnlyEdges) <= 20 {
		fmt.Printf("Sample materialization-only edges:\n")
		printSampleEdges(matOnlyEdges, 20)
	}
	
	if len(sketchOnlyEdges) > 0 && len(sketchOnlyEdges) <= 20 {
		fmt.Printf("Sample sketch-only edges:\n")
		printSampleEdges(sketchOnlyEdges, 20)
	}
}

// printSampleEdges prints a sample of edges
func printSampleEdges(edges map[string]bool, limit int) {
	edgeList := make([]string, 0, len(edges))
	for edge := range edges {
		edgeList = append(edgeList, edge)
	}
	sort.Strings(edgeList)
	
	count := 0
	for _, edge := range edgeList {
		if count >= limit {
			fmt.Printf("  ... and %d more\n", len(edgeList)-limit)
			break
		}
		fmt.Printf("  %s\n", edge)
		count++
	}
	fmt.Printf("\n")
}
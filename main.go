package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
	"github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

func main() {
	fmt.Println("SCAR-based Heterogeneous Graph Clustering Service")
	fmt.Println("=================================================")

	// Example usage of the validation system
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run main.go <graph_file> <metapath_file>")
		fmt.Println("Example: go run main.go data/graph_input.json data/meta_path.json")
		os.Exit(1)
	}

	graphFile := os.Args[1]
	metaPathFile := os.Args[2]

	// Test the validation pipeline
	testValidation(graphFile, metaPathFile)
}

func testValidation(graphFile, metaPathFile string) {
	fmt.Printf("Loading graph from: %s\n", graphFile)
	fmt.Printf("Loading meta path from: %s\n", metaPathFile)

	// Load and validate graph
	graph, err := validation.LoadAndValidateGraph(graphFile)
	if err != nil {
		log.Fatalf("Graph validation failed: %v", err)
	}

	fmt.Printf("✓ Graph loaded successfully with %d nodes and %d edges\n", 
		len(graph.Nodes), len(graph.Edges))

	// Load and validate meta path
	metaPath, err := validation.LoadAndValidateMetaPath(metaPathFile)
	if err != nil {
		log.Fatalf("Meta path validation failed: %v", err)
	}

	fmt.Printf("✓ Meta path loaded: %s\n", metaPath.Description)

	// Validate meta path against graph
	if err := validation.ValidateMetaPathAgainstGraph(metaPath, graph); err != nil {
		log.Fatalf("Meta path incompatible with graph: %v", err)
	}

	fmt.Println("✓ Meta path is compatible with graph structure")

	// Print validation summary
	printValidationSummary(graph, metaPath)
}

func printValidationSummary(graph *models.HeterogeneousGraph, metaPath *models.MetaPath) {
	fmt.Println("\n📊 Validation Summary:")
	fmt.Println("======================")
	
	// Graph statistics
	nodeTypeCounts := make(map[string]int)
	edgeTypeCounts := make(map[string]int)
	
	for _, node := range graph.Nodes {
		nodeTypeCounts[node.Type]++
	}
	
	for _, edge := range graph.Edges {
		edgeTypeCounts[edge.Type]++
	}
	
	fmt.Println("Graph Statistics:")
	fmt.Printf("  - Total Nodes: %d\n", len(graph.Nodes))
	fmt.Printf("  - Total Edges: %d\n", len(graph.Edges))
	
	fmt.Println("  - Node Types:")
	for nodeType, count := range nodeTypeCounts {
		fmt.Printf("    • %s: %d nodes\n", nodeType, count)
	}
	
	fmt.Println("  - Edge Types:")
	for edgeType, count := range edgeTypeCounts {
		fmt.Printf("    • %s: %d edges\n", edgeType, count)
	}
	
	// Meta path info
	fmt.Println("\nMeta Path Information:")
	fmt.Printf("  - ID: %s\n", metaPath.ID)
	fmt.Printf("  - Length: %d\n", len(metaPath.NodeSequence))
	fmt.Printf("  - Pattern: %v\n", metaPath.NodeSequence)
	fmt.Printf("  - Edge Pattern: %v\n", metaPath.EdgeSequence)
}
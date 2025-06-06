package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/models"
	"github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

func main() {
	fmt.Println("SCAR-based Heterogeneous Graph Clustering Service")
	fmt.Println("=================================================")

	// Check command line arguments
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run main.go <mode> <graph_file> <metapath_file>")
		fmt.Println("Modes:")
		fmt.Println("  validate    - Run validation only")
		fmt.Println("  materialize - Run validation + materialization")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  go run main.go validate data/graph_input.json data/meta_path.json")
		fmt.Println("  go run main.go materialize data/graph_input.json data/meta_path.json")
		os.Exit(1)
	}

	mode := os.Args[1]
	
	if len(os.Args) < 4 {
		fmt.Printf("Error: Mode '%s' requires graph and meta path files\n", mode)
		os.Exit(1)
	}
	
	graphFile := os.Args[2]
	metaPathFile := os.Args[3]

	switch mode {
	case "validate":
		// Test the validation pipeline
		testValidation(graphFile, metaPathFile)
	case "materialize":
		// Test validation + materialization
		testMaterialization(graphFile, metaPathFile)
	default:
		fmt.Printf("Unknown mode: %s\n", mode)
		fmt.Println("Available modes: validate, materialize")
		os.Exit(1)
	}
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

func testMaterialization(graphFile, metaPathFile string) {
	fmt.Printf("Running validation + materialization pipeline\n")
	fmt.Printf("Graph file: %s\n", graphFile)
	fmt.Printf("Meta path file: %s\n", metaPathFile)
	fmt.Println()

	// Step 1: Load and validate (same as validation-only mode)
	fmt.Println("🔍 Step 1: Validation")
	graph, err := validation.LoadAndValidateGraph(graphFile)
	if err != nil {
		log.Fatalf("Graph validation failed: %v", err)
	}

	metaPath, err := validation.LoadAndValidateMetaPath(metaPathFile)
	if err != nil {
		log.Fatalf("Meta path validation failed: %v", err)
	}

	if err := validation.ValidateMetaPathAgainstGraph(metaPath, graph); err != nil {
		log.Fatalf("Meta path incompatible with graph: %v", err)
	}

	fmt.Printf("✅ Validation successful!\n")
	fmt.Printf("   Graph: %d nodes, %d edges\n", len(graph.Nodes), len(graph.Edges))
	fmt.Printf("   Meta path: %s\n", metaPath.String())

	// Step 2: Test BOTH interpretations
	fmt.Println("\n⚙️  Step 2: Testing Both Interpretations")
	
	// Test 1: Direct Traversal (original implementation)
	fmt.Println("\n--- Testing Direct Traversal ---")
	testWithInterpretation(graph, metaPath, materialization.DirectTraversal)
	
	// Test 2: Meeting-Based (new implementation)
	fmt.Println("\n--- Testing Meeting-Based ---")
	testWithInterpretation(graph, metaPath, materialization.MeetingBased)
}

func testWithInterpretation(graph *models.HeterogeneousGraph, metaPath *models.MetaPath, 
	interpretation materialization.MetaPathInterpretation) {
	
	// Configure materialization
	config := materialization.DefaultMaterializationConfig()
	config.Aggregation.Strategy = materialization.Count
	config.Aggregation.Interpretation = interpretation
	config.Aggregation.Symmetric = true
	config.Aggregation.MinWeight = 1.0
	
	// Progress callback
	progressCallback := func(current, total int, message string) {
		if total > 0 {
			percentage := float64(current) / float64(total) * 100
			fmt.Printf("\r   Progress: %.1f%% (%d/%d) - %s", percentage, current, total, message)
		} else {
			fmt.Printf("\r   %s", message)
		}
	}
	
	// Create engine and check feasibility
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, progressCallback)
	
	canMaterialize, reason, err := engine.CanMaterialize(1000) // 1GB limit
	if err != nil {
		log.Fatalf("Failed to check feasibility: %v", err)
	}
	
	fmt.Printf("Feasibility check: %v (%s)\n", canMaterialize, reason)
	
	if !canMaterialize {
		fmt.Println("❌ Materialization not feasible with current settings")
		return
	}
	
	// Perform materialization
	fmt.Printf("Starting materialization with interpretation: %v\n", interpretation)
	result, err := engine.Materialize()
	if err != nil {
		log.Fatalf("Materialization failed: %v", err)
	}
	
	fmt.Println() // New line after progress
	
	if !result.Success {
		log.Fatalf("Materialization unsuccessful: %s", result.Error)
	}

	// Save results
	err = materialization.SaveHomogeneousGraph(result.HomogeneousGraph, "output/graph.edgelist")
	
	// Print results
	printMaterializationResults(result)
}

func printMaterializationResults(result *materialization.MaterializationResult) {
	homogGraph := result.HomogeneousGraph
	stats := result.Statistics
	
	fmt.Printf("✅ Materialization completed successfully!\n\n")
	
	fmt.Printf("📈 Homogeneous Graph:\n")
	fmt.Printf("   Node type: %s\n", homogGraph.NodeType)
	fmt.Printf("   Nodes: %d\n", len(homogGraph.Nodes))
	fmt.Printf("   Edges: %d\n", len(homogGraph.Edges))
	fmt.Printf("   Density: %.4f\n", homogGraph.Statistics.Density)
	fmt.Printf("   Average weight: %.2f\n", homogGraph.Statistics.AverageWeight)
	fmt.Printf("   Weight range: %.2f - %.2f\n", homogGraph.Statistics.MinWeight, homogGraph.Statistics.MaxWeight)
	
	fmt.Printf("\n⏱️  Performance:\n")
	fmt.Printf("   Runtime: %d ms\n", stats.RuntimeMS)
	fmt.Printf("   Peak memory: %d MB\n", stats.MemoryPeakMB)
	fmt.Printf("   Instances generated: %d\n", stats.InstancesGenerated)
	fmt.Printf("   Instances filtered: %d\n", stats.InstancesFiltered)
	
	fmt.Printf("\n🔗 Sample Connections:\n")
	count := 0
	maxSamples := 5
	for edgeKey, weight := range homogGraph.Edges {
		if count >= maxSamples {
			break
		}
		fmt.Printf("   %s ↔ %s (weight: %.2f)\n", edgeKey.From, edgeKey.To, weight)
		count++
	}
	
	if len(homogGraph.Edges) > maxSamples {
		fmt.Printf("   ... and %d more connections\n", len(homogGraph.Edges)-maxSamples)
	}
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
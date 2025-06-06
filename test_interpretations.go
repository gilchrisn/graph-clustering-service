package main

import (
	"fmt"
	"log"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/models"
	"github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

func main() {
	fmt.Println("Testing Both Meta Path Interpretations")
	fmt.Println("=====================================")

	// Test 1: Author → Paper → Venue (clear difference expected)
	fmt.Println("\n🧪 Test 1: Author → Paper → Venue")
	testAuthorVenuePattern()

	// Test 2: Author → Paper → Author (might be similar results)
	fmt.Println("\n🧪 Test 2: Author → Paper → Author")
	testAuthorCoauthorshipPattern()
}

func testAuthorVenuePattern() {
	// Load the author-venue test data
	graph, err := validation.LoadAndValidateGraph("data/test_meeting_based.json")
	if err != nil {
		log.Fatalf("Failed to load graph: %v", err)
	}

	metaPath, err := validation.LoadAndValidateMetaPath("data/author_venue_metapath.json")
	if err != nil {
		log.Fatalf("Failed to load meta path: %v", err)
	}

	fmt.Printf("Graph: %d nodes, %d edges\n", len(graph.Nodes), len(graph.Edges))
	fmt.Printf("Meta path: %s\n", metaPath.String())

	// Print raw instances first
	printAllInstances(graph, metaPath)

	// Test Direct Traversal
	fmt.Println("\n--- Direct Traversal Result ---")
	result1 := runMaterialization(graph, metaPath, materialization.DirectTraversal)
	printResult(result1, "DirectTraversal")

	// Test Meeting-Based
	fmt.Println("\n--- Meeting-Based Result ---")
	result2 := runMaterialization(graph, metaPath, materialization.MeetingBased)
	printResult(result2, "MeetingBased")

	fmt.Println("\n📊 Expected Difference:")
	fmt.Println("  Direct: Alice→ICML, Bob→ICML, Carol→NeurIPS (no author connections)")
	fmt.Println("  Meeting: Alice↔Bob (both at ICML), Carol isolated")
}

func testAuthorCoauthorshipPattern() {
	// Load the original author-paper data
	graph, err := validation.LoadAndValidateGraph("data/graph_input.json")
	if err != nil {
		log.Fatalf("Failed to load graph: %v", err)
	}

	metaPath, err := validation.LoadAndValidateMetaPath("data/meta_path.json")
	if err != nil {
		log.Fatalf("Failed to load meta path: %v", err)
	}

	fmt.Printf("Graph: %d nodes, %d edges\n", len(graph.Nodes), len(graph.Edges))
	fmt.Printf("Meta path: %s\n", metaPath.String())

	// Test both interpretations
	fmt.Println("\n--- Direct Traversal Result ---")
	result1 := runMaterialization(graph, metaPath, materialization.DirectTraversal)
	printResult(result1, "DirectTraversal")

	fmt.Println("\n--- Meeting-Based Result ---")
	result2 := runMaterialization(graph, metaPath, materialization.MeetingBased)
	printResult(result2, "MeetingBased")

	fmt.Println("\n📊 Expected:")
	fmt.Println("  Both should give similar results for Author→Paper→Author")
}

func printAllInstances(graph *models.HeterogeneousGraph, metaPath *models.MetaPath) {
	fmt.Println("\n🔍 All Possible Meta Path Instances:")

	config := materialization.DefaultTraversalConfig()
	generator := materialization.NewInstanceGenerator(graph, metaPath, config)

	instances, _, err := generator.FindAllInstances(nil)
	if err != nil {
		fmt.Printf("Error generating instances: %v\n", err)
		return
	}

	for i, instance := range instances {
		fmt.Printf("  Instance %d: %s (start: %s, end: %s)\n",
			i+1, instance.String(), instance.GetStartNode(), instance.GetEndNode())
	}
}

func runMaterialization(graph *models.HeterogeneousGraph, metaPath *models.MetaPath,
	interpretation materialization.MetaPathInterpretation) *materialization.MaterializationResult {

	config := materialization.DefaultMaterializationConfig()
	config.Aggregation.Strategy = materialization.Count
	config.Aggregation.Interpretation = interpretation
	config.Aggregation.Symmetric = true
	config.Progress.EnableProgress = false // No progress for testing

	engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()
	if err != nil {
		log.Fatalf("Materialization failed: %v", err)
	}

	return result
}

func printResult(result *materialization.MaterializationResult, label string) {
	if !result.Success {
		fmt.Printf("❌ %s failed: %s\n", label, result.Error)
		return
	}

	homogGraph := result.HomogeneousGraph
	fmt.Printf("✅ %s: %d nodes, %d edges\n", label, len(homogGraph.Nodes), len(homogGraph.Edges))

	// Print nodes
	fmt.Printf("  Nodes: ")
	for nodeID := range homogGraph.Nodes {
		fmt.Printf("%s ", nodeID)
	}
	fmt.Println()

	// Print edges
	fmt.Printf("  Edges:\n")
	for edgeKey, weight := range homogGraph.Edges {
		fmt.Printf("    %s ↔ %s (weight: %.2f)\n", edgeKey.From, edgeKey.To, weight)
	}

	if len(homogGraph.Edges) == 0 {
		fmt.Printf("    (no edges)\n")
	}
}
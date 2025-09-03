package main

import (
	"fmt"
	"log"
	"os"
	
	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
)

func main() {
	fmt.Println("🔄 Testing Graph Materialization...")
	
	// Test files
	graphFile := "test_graph.txt"
	propsFile := "properties.txt"
	pathFile := "path.txt"
	outputFile := "output_edges.txt"
	
	// Check if test files exist
	files := []string{graphFile, propsFile, pathFile}
	for _, file := range files {
		if _, err := os.Stat(file); os.IsNotExist(err) {
			log.Fatalf("❌ Test file missing: %s", file)
		}
	}
	
	fmt.Println("✅ Test files found")
	
	// Method 1: Simple conversion
	fmt.Println("\n🚀 Method 1: Simple SCARToMaterialization")
	err := materialization.SCARToMaterialization(graphFile, propsFile, pathFile, outputFile)
	if err != nil {
		log.Fatalf("❌ Simple conversion failed: %v", err)
	}
	fmt.Printf("✅ Created: %s\n", outputFile)
	
	// Method 2: Detailed conversion with custom config
	fmt.Println("\n🚀 Method 2: Detailed conversion")
	
	// Parse input
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propsFile, pathFile)
	if err != nil {
		log.Fatalf("❌ Failed to parse input: %v", err)
	}
	
	fmt.Printf("📊 Parsed graph: %d nodes, %d edges\n", len(graph.Nodes), len(graph.Edges))
	fmt.Printf("📋 Meta-path: %v\n", metaPath.NodeSequence)
	
	// Configure materialization
	config := materialization.DefaultMaterializationConfig()
	config.Aggregation.Strategy = materialization.Count
	config.Aggregation.Symmetric = true
	config.Traversal.MaxInstances = 10000
	
	// Add progress tracking
	progressCb := func(current, total int, message string) {
		if total > 0 {
			percentage := float64(current) / float64(total) * 100
			fmt.Printf("\r   Progress: %.1f%% (%d/%d) - %s", percentage, current, total, message)
		} else {
			fmt.Printf("\r   %s", message)
		}
	}
	
	// Run materialization
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, progressCb)
	result, err := engine.Materialize()
	if err != nil {
		log.Fatalf("❌ Materialization failed: %v", err)
	}
	
	fmt.Println() // New line after progress
	
	// Print results
	fmt.Printf("✅ Materialization complete!\n")
	fmt.Printf("   Homogeneous nodes: %d\n", len(result.HomogeneousGraph.Nodes))
	fmt.Printf("   Homogeneous edges: %d\n", len(result.HomogeneousGraph.Edges))
	fmt.Printf("   Runtime: %d ms\n", result.Statistics.RuntimeMS)
	fmt.Printf("   Instances generated: %d\n", result.Statistics.InstancesGenerated)
	
	// Save different output formats
	fmt.Println("\n💾 Saving outputs...")
	
	// Simple edge list (for community detection tools)
	edgeListFile := "detailed_edges.txt"
	err = materialization.SaveAsSimpleEdgeList(result.HomogeneousGraph, edgeListFile)
	if err != nil {
		log.Printf("⚠️  Failed to save edge list: %v", err)
	} else {
		fmt.Printf("✅ Edge list: %s\n", edgeListFile)
	}
	
	// CSV format
	csvFile := "edges.csv"
	err = materialization.SaveAsCSV(result.HomogeneousGraph, csvFile)
	if err != nil {
		log.Printf("⚠️  Failed to save CSV: %v", err)
	} else {
		fmt.Printf("✅ CSV: %s\n", csvFile)
	}
	
	// JSON format with full details
	jsonFile := "result.json"
	err = materialization.SaveMaterializationResult(result, jsonFile)
	if err != nil {
		log.Printf("⚠️  Failed to save JSON: %v", err)
	} else {
		fmt.Printf("✅ Full result: %s\n", jsonFile)
	}
	
	// Show edge list content
	fmt.Println("\n📋 Generated edges (Author collaborations):")
	for edgeKey, weight := range result.HomogeneousGraph.Edges {
		fmt.Printf("   %s ↔ %s (weight: %.1f)\n", edgeKey.From, edgeKey.To, weight)
	}
	
	// Verify results
	fmt.Println("\n🔍 Verification:")
	verifier := materialization.NewGraphVerifier()
	verifier.LoadFromObjects(graph, metaPath)
	
	verificationResult, err := verifier.VerifyMaterialization(config)
	if err != nil {
		log.Printf("⚠️  Verification failed: %v", err)
	} else {
		fmt.Printf("   Tests: %d total, %d passed, %d failed\n", 
			verificationResult.TotalTests, 
			verificationResult.PassedTests, 
			verificationResult.FailedTests)
		
		if verificationResult.Passed {
			fmt.Println("   ✅ All critical tests passed!")
		} else {
			fmt.Println("   ⚠️  Some tests failed - check verification details")
		}
	}
	
	// Show pipeline command example
	fmt.Println("\n🔗 Pipeline Integration:")
	fmt.Printf("   To run Louvain: ./louvain %s\n", edgeListFile)
	fmt.Printf("   To run SCAR: ./scar.exe \"%s\" -louvain\n", edgeListFile)
	
	fmt.Println("\n🎯 Test complete! Check the output files.")
}

// Helper function to create test files if they don't exist
func createTestFiles() {
	// Create graph.txt
	graphContent := `alice paper1
bob paper1
charlie paper2
alice paper2
diana paper3
alice paper3
bob paper4
charlie paper4`
	
	err := os.WriteFile("graph.txt", []byte(graphContent), 0644)
	if err != nil {
		log.Printf("Could not create graph.txt: %v", err)
	}
	
	// Create properties.txt
	propsContent := `alice 0
bob 0
charlie 0
diana 0
paper1 1
paper2 1
paper3 1
paper4 1`
	
	err = os.WriteFile("properties.txt", []byte(propsContent), 0644)
	if err != nil {
		log.Printf("Could not create properties.txt: %v", err)
	}
	
	// Create path.txt
	pathContent := `0
1
0`
	
	err = os.WriteFile("path.txt", []byte(pathContent), 0644)
	if err != nil {
		log.Printf("Could not create path.txt: %v", err)
	}
	
	fmt.Println("✅ Created test files: graph.txt, properties.txt, path.txt")
}
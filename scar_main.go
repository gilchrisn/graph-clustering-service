package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

func main() {
	fmt.Println("SCAR - Sketch-based Community Detection")
	fmt.Println("======================================")

	// Check command line arguments
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run scar_main.go <graph_file> <meta_path_file> [output_dir]")
		fmt.Println("Example: go run scar_main.go data/graph.json data/meta_path.json output/")
		os.Exit(1)
	}

	graphFile := os.Args[1]
	metaPathFile := os.Args[2]
	outputDir := "output/scar"
	
	if len(os.Args) >= 4 {
		outputDir = os.Args[3]
	}

	// Step 1: Load and validate input data
	fmt.Println("\n🔍 Step 1: Loading and Validating Data")
	fmt.Println("=====================================")

	graph, metaPath, err := loadAndValidateData(graphFile, metaPathFile)
	if err != nil {
		log.Fatalf("❌ Validation failed: %v", err)
	}

	fmt.Printf("✅ Validation successful!\n")
	fmt.Printf("   📊 Graph: %d nodes, %d edges\n", len(graph.Nodes), len(graph.Edges))
	fmt.Printf("   🛤️  Meta path: %s\n", metaPath.String())
	fmt.Printf("   🔗 Meta path length: %d\n", len(metaPath.EdgeSequence))

	// Step 2: Configure SCAR
	fmt.Println("\n⚙️  Step 2: Configuring SCAR")
	fmt.Println("===========================")

	config := scar.DefaultScarConfig()
	config.MetaPath = convertToScarMetaPath(metaPath)
	config.K = 64
	config.NK = 8
	config.MaxIterations = 50
	config.RandomSeed = 42
	config.Verbose = true


	config.Parallel.Enabled = true // Enable parallel processing
	config.Parallel.NumWorkers = 4 // Set number of workers for parallel processing
	config.Parallel.BatchSize = 100 // Set max queue size for parallel tasks

	fmt.Printf("   Parallel processing enabled with %d workers\n", config.Parallel.NumWorkers)

	// Progress callback
	config.ProgressCallback = func(level int, iteration int, modularity float64, nodes int) {
		fmt.Printf("\r   Level %d, Iteration %d: modularity=%.6f, nodes=%d", 
			level, iteration, modularity, nodes)
	}

	fmt.Printf("📋 Configuration:\n")
	fmt.Printf("   K (sketch size): %d\n", config.K)
	fmt.Printf("   NK (hash functions): %d\n", config.NK)
	fmt.Printf("   Max iterations: %d\n", config.MaxIterations)
	fmt.Printf("   Random seed: %d\n", config.RandomSeed)
	fmt.Printf("   Meta path: %s\n", config.MetaPath.String())

	// Step 3: Run SCAR
	fmt.Println("\n🚀 Step 3: Running SCAR Algorithm")
	fmt.Println("=================================")

	startTime := time.Now()
	result, err := scar.RunScar(convertToScarGraph(graph), config)
	runtime := time.Since(startTime)

	if err != nil {
		log.Fatalf("❌ SCAR failed: %v", err)
	}

	fmt.Println() // New line after progress

	// Step 4: Display Results
	fmt.Println("\n📊 Step 4: Results")
	fmt.Println("==================")

	fmt.Printf("✅ SCAR completed successfully!\n")
	fmt.Printf("   ⏱️  Runtime: %v\n", runtime)
	fmt.Printf("   🎯 Final modularity: %.6f\n", result.Modularity)
	fmt.Printf("   👥 Communities found: %d\n", len(getUniqueCommunities(result.FinalCommunities)))
	fmt.Printf("   📊 Levels processed: %d\n", result.NumLevels)
	fmt.Printf("   🔄 Total iterations: %d\n", result.Statistics.TotalIterations)

	// Display level breakdown
	fmt.Printf("\n📈 Level Breakdown:\n")
	for _, level := range result.Levels {
		fmt.Printf("   Level %d: %d nodes → %d communities (modularity: %.6f, %d iterations)\n",
			level.Level, level.Nodes, level.Communities, level.Modularity, level.Iterations)
	}

	// Step 5: Save Results
	fmt.Println("\n💾 Step 5: Saving Results")
	fmt.Println("========================")

	if err := saveResults(result, convertToScarGraph(graph), outputDir); err != nil {
		fmt.Printf("⚠️  Warning: Failed to save some outputs: %v\n", err)
	} else {
		fmt.Printf("✅ Results saved to: %s\n", outputDir)
	}

	// Step 6: Analysis
	fmt.Println("\n🔍 Step 6: Analysis")
	fmt.Println("==================")

	analyzeResults(result, graph)

	fmt.Println("\n🎉 SCAR execution completed!")
}

func loadAndValidateData(graphFile, metaPathFile string) (*models.HeterogeneousGraph, *models.MetaPath, error) {
	// Load and validate graph
	graph, err := validation.LoadAndValidateGraph(graphFile)
	if err != nil {
		return nil, nil, fmt.Errorf("graph validation failed: %w", err)
	}

	// Load and validate meta path
	metaPath, err := validation.LoadAndValidateMetaPath(metaPathFile)
	if err != nil {
		return nil, nil, fmt.Errorf("meta path validation failed: %w", err)
	}

	// Check compatibility
	if err := validation.ValidateMetaPathAgainstGraph(metaPath, graph); err != nil {
		return nil, nil, fmt.Errorf("meta path incompatible with graph: %w", err)
	}

	return graph, metaPath, nil
}

func convertToScarGraph(graph *models.HeterogeneousGraph) *scar.HeterogeneousGraph {
	scarGraph := scar.NewHeterogeneousGraph()

	// Add nodes
	for nodeID, node := range graph.Nodes {
		scarNode := scar.HeteroNode{
			ID:         nodeID,
			Type:       node.Type,
			Properties: node.Properties,
		}
		scarGraph.AddNode(scarNode)
	}

	// Add edges
	for _, edge := range graph.Edges {
		scarEdge := scar.HeteroEdge{
			From:   edge.From,
			To:     edge.To,
			Type:   edge.Type,
			Weight: edge.Weight,
		}
		scarGraph.AddEdge(scarEdge)
	}

	return scarGraph
}

func convertToScarMetaPath(metaPath *models.MetaPath) scar.MetaPath {
	return scar.MetaPath{
		NodeTypes: metaPath.NodeSequence,
		EdgeTypes: metaPath.EdgeSequence,
	}
}

func saveResults(result *scar.ScarResult, graph *scar.HeterogeneousGraph, outputDir string) error {
	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Try to save using SCAR's output functions
	if err := scar.WriteAll(result, graph, outputDir, "scar"); err != nil {
		return fmt.Errorf("failed to write SCAR outputs: %w", err)
	}

	return nil
}

func analyzeResults(result *scar.ScarResult, originalGraph *models.HeterogeneousGraph) {
	uniqueComms := getUniqueCommunities(result.FinalCommunities)
	
	fmt.Printf("🔹 Original graph: %d nodes, %d edges\n", 
		len(originalGraph.Nodes), len(originalGraph.Edges))
	fmt.Printf("🔹 Communities found: %d\n", len(uniqueComms))
	fmt.Printf("🔹 Compression ratio: %.2fx\n", 
		float64(len(originalGraph.Nodes))/float64(len(uniqueComms)))

	// Check for issues
	if result.Modularity > 1.0 {
		fmt.Printf("⚠️  WARNING: Modularity > 1.0 (%.6f) indicates a calculation error\n", result.Modularity)
	}

	if len(uniqueComms) == len(originalGraph.Nodes) {
		fmt.Printf("⚠️  WARNING: No community detection occurred (each node in its own community)\n")
	}

	if result.Modularity < 0 {
		fmt.Printf("⚠️  WARNING: Negative modularity (%.6f) suggests poor community structure\n", result.Modularity)
	}

	// Community size distribution
	commSizes := make(map[int]int)
	for _, comm := range result.FinalCommunities {
		commSizes[comm]++
	}

	fmt.Printf("🔹 Community size distribution:\n")
	sizeDistribution := make(map[int]int) // size -> count
	for _, size := range commSizes {
		sizeDistribution[size]++
	}

	for size := 1; size <= 10; size++ {
		if count := sizeDistribution[size]; count > 0 {
			fmt.Printf("   Size %d: %d communities\n", size, count)
		}
	}

	largerCommunities := 0
	for size, count := range sizeDistribution {
		if size > 10 {
			largerCommunities += count
		}
	}
	if largerCommunities > 0 {
		fmt.Printf("   Size >10: %d communities\n", largerCommunities)
	}
}

func getUniqueCommunities(finalCommunities map[string]int) map[int]bool {
	unique := make(map[int]bool)
	for _, comm := range finalCommunities {
		unique[comm] = true
	}
	return unique
}
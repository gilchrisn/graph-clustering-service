package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/models"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

func main() {
	fmt.Println("Heterogeneous Graph Clustering Service")
	fmt.Println("=====================================")
	fmt.Println("Two Approaches: (1) Materialization + Louvain  (2) SCAR")
	fmt.Println()

	// Check command line arguments
	if len(os.Args) < 4 {
		fmt.Println("Usage: go run clustering_main.go <approach> <graph_file> <meta_path_file> [output_dir]")
		fmt.Println()
		fmt.Println("Approaches:")
		fmt.Println("  materialization  - Materialize heterogeneous graph to homogeneous, then apply Louvain")
		fmt.Println("  scar            - Sketch-based Community detection with Approximated Resistance")
		fmt.Println("  both            - Run both approaches for comparison")
		fmt.Println()
		fmt.Println("Examples:")
		fmt.Println("  go run clustering_main.go materialization data/graph.json data/meta_path.json")
		fmt.Println("  go run clustering_main.go scar data/graph.json data/meta_path.json")
		fmt.Println("  go run clustering_main.go both data/graph.json data/meta_path.json output/")
		os.Exit(1)
	}

	approach := os.Args[1]
	graphFile := os.Args[2]
	metaPathFile := os.Args[3]
	outputDir := "output/"
	
	if len(os.Args) >= 5 {
		outputDir = os.Args[4]
	}

	// Step 1: Load and validate input data
	fmt.Println("🔍 Step 1: Loading and Validating Input Data")
	fmt.Println("===========================================")

	graph, metaPath, err := loadAndValidateData(graphFile, metaPathFile)
	if err != nil {
		log.Fatalf("❌ Validation failed: %v", err)
	}

	fmt.Printf("✅ Validation successful!\n")
	fmt.Printf("   📊 Graph: %d nodes, %d edges\n", len(graph.Nodes), len(graph.Edges))
	fmt.Printf("   🔗 Meta path: %s\n", metaPath.String())
	fmt.Printf("   📏 Meta path length: %d\n", len(metaPath.EdgeSequence))
	fmt.Printf("   🏷️  Node types: %v\n", getNodeTypes(graph))
	fmt.Printf("   🔀 Edge types: %v\n", getEdgeTypes(graph))
	fmt.Println()

	// Step 2: Execute chosen approach(es)
	switch approach {
	case "materialization":
		fmt.Println("🔄 Executing Approach 1: Materialization + Louvain")
		fmt.Println("================================================")
		runMaterializationApproach(graph, metaPath, outputDir+"/materialization")

	case "scar":
		fmt.Println("⚡ Executing Approach 2: SCAR")
		fmt.Println("===========================")
		runScarApproach(graph, metaPath, outputDir+"/scar")

	case "both":
		fmt.Println("🔄 Executing Approach 1: Materialization + Louvain")
		fmt.Println("================================================")
		matResult := runMaterializationApproach(graph, metaPath, outputDir+"/materialization")
		
		fmt.Println()
		fmt.Println("⚡ Executing Approach 2: SCAR")
		fmt.Println("===========================")
		scarResult := runScarApproach(graph, metaPath, outputDir+"/scar")
		
		fmt.Println()
		fmt.Println("📊 Comparing Results")
		fmt.Println("==================")
		compareResults(matResult, scarResult)

	default:
		log.Fatalf("❌ Unknown approach: %s. Use 'materialization', 'scar', or 'both'", approach)
	}

	fmt.Println("\n🎉 Clustering completed successfully!")
}

// ApproachResult holds results from either approach for comparison
type ApproachResult struct {
	Approach         string
	Communities      map[string]int
	Modularity       float64
	NumCommunities   int
	Runtime          time.Duration
	MemoryPeakMB     int64
	Success          bool
	Error            string
}

func runMaterializationApproach(graph *models.HeterogeneousGraph, metaPath *models.MetaPath, outputDir string) *ApproachResult {
	startTime := time.Now()
	result := &ApproachResult{
		Approach: "Materialization + Louvain",
	}

	fmt.Println("\n📋 Configuration - Materialization + Louvain:")
	
	// Configure materialization
	matConfig := materialization.DefaultMaterializationConfig()
	matConfig.Aggregation.Strategy = materialization.Count
	matConfig.Aggregation.Interpretation = materialization.MeetingBased
	matConfig.Aggregation.Symmetric = true
	matConfig.Aggregation.MinWeight = 1.0
	matConfig.Traversal.MaxInstances = 1000000
	matConfig.Traversal.TimeoutSeconds = 300
	matConfig.Traversal.Parallelism = 4

	fmt.Printf("   🏗️  Materialization Strategy: %s\n", getStrategyName(matConfig.Aggregation.Strategy))
	fmt.Printf("   🎯 Interpretation: %s\n", getInterpretationName(matConfig.Aggregation.Interpretation))
	fmt.Printf("   ⚖️  Symmetric: %v\n", matConfig.Aggregation.Symmetric)
	fmt.Printf("   📊 Min Weight: %.1f\n", matConfig.Aggregation.MinWeight)
	fmt.Printf("   🔢 Max Instances: %d\n", matConfig.Traversal.MaxInstances)
	fmt.Printf("   ⏱️  Timeout: %d seconds\n", matConfig.Traversal.TimeoutSeconds)
	fmt.Printf("   🔄 Parallelism: %d workers\n", matConfig.Traversal.Parallelism)

	// Configure Louvain
	louvainConfig := louvain.DefaultLouvainConfig()
	louvainConfig.MaxIterations = 1
	louvainConfig.MinModularity = 0.001
	louvainConfig.RandomSeed = 42
	louvainConfig.Verbose = false

	fmt.Printf("   🔁 Louvain Max Iterations: %d\n", louvainConfig.MaxIterations)
	fmt.Printf("   📈 Min Modularity Improvement: %.6f\n", louvainConfig.MinModularity)
	fmt.Printf("   🎲 Random Seed: %d\n", louvainConfig.RandomSeed)

	fmt.Println("\n🔄 Phase 1: Materializing Heterogeneous Graph")
	fmt.Println("--------------------------------------------")

	// Progress callback for materialization
	progressCallback := func(current, total int, message string) {
		if total > 0 {
			percentage := float64(current) / float64(total) * 100
			fmt.Printf("\r   Progress: %.1f%% (%d/%d) - %s", percentage, current, total, message)
		} else {
			fmt.Printf("\r   %s", message)
		}
	}

	// Create materialization engine
	engine := materialization.NewMaterializationEngine(graph, metaPath, matConfig, progressCallback)

	// Check feasibility
	canMaterialize, reason, err := engine.CanMaterialize(2000) // 2GB limit
	if err != nil {
		result.Error = fmt.Sprintf("feasibility check failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result
	}

	if !canMaterialize {
		result.Error = fmt.Sprintf("materialization not feasible: %s", reason)
		result.Runtime = time.Since(startTime)
		return result
	}

	fmt.Printf("✅ Feasibility check passed: %s\n", reason)

	// Perform materialization
	matResult, err := engine.Materialize()
	if err != nil {
		result.Error = fmt.Sprintf("materialization failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result
	}

	fmt.Println() // New line after progress

	if !matResult.Success {
		result.Error = fmt.Sprintf("materialization unsuccessful: %s", matResult.Error)
		result.Runtime = time.Since(startTime)
		return result
	}

	fmt.Printf("✅ Materialization completed!\n")
	fmt.Printf("   📊 Generated: %d nodes, %d edges\n", 
		len(matResult.HomogeneousGraph.Nodes), len(matResult.HomogeneousGraph.Edges))
	fmt.Printf("   ⚡ Instances processed: %d\n", matResult.Statistics.InstancesGenerated)
	fmt.Printf("   💾 Memory peak: %d MB\n", matResult.Statistics.MemoryPeakMB)

	fmt.Println("\n🔁 Phase 2: Running Louvain Algorithm")
	fmt.Println("-----------------------------------")

	// Convert to Louvain format and run clustering
	louvainGraph := convertToLouvainGraph(matResult.HomogeneousGraph)

	louvainResult, err := louvain.RunLouvain(louvainGraph, louvainConfig)
	if err != nil {
		result.Error = fmt.Sprintf("louvain failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result
	}

	fmt.Printf("✅ Louvain completed!\n")
	fmt.Printf("   📈 Modularity: %.6f\n", louvainResult.Modularity)
	fmt.Printf("   🏘️  Communities: %d\n", len(louvainResult.FinalCommunities))
	fmt.Printf("   📊 Levels: %d\n", louvainResult.NumLevels)
	fmt.Printf("   🔄 Total iterations: %d\n", louvainResult.Statistics.TotalIterations)

	// Save outputs
	fmt.Println("\n💾 Phase 3: Saving Output Files")
	fmt.Println("------------------------------")
	
	if err := saveLouvainOutputs(louvainResult, louvainGraph, outputDir); err != nil {
		fmt.Printf("⚠️  Warning: Failed to save some outputs: %v\n", err)
	} else {
		fmt.Printf("✅ Outputs saved to: %s\n", outputDir)
		fmt.Printf("   📁 Files generated:\n")
		fmt.Printf("      - communities.mapping (community assignments)\n")
		fmt.Printf("      - communities.hierarchy (hierarchical structure)\n")
		fmt.Printf("      - communities.root (top-level communities)\n")
		fmt.Printf("      - communities.edges (inter-community edges)\n")
	}

	// Update result
	result.Communities = louvainResult.FinalCommunities
	result.Modularity = louvainResult.Modularity
	result.NumCommunities = len(getUniqueCommunities(louvainResult.FinalCommunities))
	result.Runtime = time.Since(startTime)
	result.MemoryPeakMB = matResult.Statistics.MemoryPeakMB
	result.Success = true

	fmt.Printf("\n📊 Materialization + Louvain Results:")
	fmt.Printf("   ⏱️  Total runtime: %v\n", result.Runtime)
	fmt.Printf("   📈 Final modularity: %.6f\n", result.Modularity)
	fmt.Printf("   🏘️  Communities found: %d\n", result.NumCommunities)
	fmt.Printf("   💾 Memory usage: %d MB\n", result.MemoryPeakMB)

	return result
}

func runScarApproach(graph *models.HeterogeneousGraph, metaPath *models.MetaPath, outputDir string) *ApproachResult {
	startTime := time.Now()
	result := &ApproachResult{
		Approach: "SCAR",
	}

	fmt.Println("\n📋 Configuration - SCAR:")
	
	// Configure SCAR
	config := scar.DefaultScarConfig()
	config.MetaPath = convertToScarMetaPath(metaPath)
	config.K = 64
	config.NK = 8
	config.MaxIterations = 50
	config.RandomSeed = 42
	config.Verbose = true

	// Parallel configuration
	config.Parallel.Enabled = true
	config.Parallel.NumWorkers = 4
	config.Parallel.BatchSize = 100

	fmt.Printf("   📏 K (sketch size): %d\n", config.K)
	fmt.Printf("   🔢 NK (hash functions): %d\n", config.NK)
	fmt.Printf("   🔁 Max iterations: %d\n", config.MaxIterations)
	fmt.Printf("   🎲 Random seed: %d\n", config.RandomSeed)
	fmt.Printf("   ⚡ Parallel processing: %v (%d workers)\n", config.Parallel.Enabled, config.Parallel.NumWorkers)
	fmt.Printf("   📦 Batch size: %d\n", config.Parallel.BatchSize)
	fmt.Printf("   🔗 Meta path: %s\n", config.MetaPath.String())

	// Progress callback
	config.ProgressCallback = func(level int, iteration int, modularity float64, nodes int) {
		fmt.Printf("\r   Level %d, Iteration %d: modularity=%.6f, nodes=%d", 
			level, iteration, modularity, nodes)
	}

	fmt.Println("\n⚡ Running SCAR Algorithm")
	fmt.Println("-----------------------")

	// Convert to SCAR format and run
	scarGraph := convertToScarGraph(graph)
	scarResult, err := scar.RunScar(scarGraph, config)

	if err != nil {
		result.Error = fmt.Sprintf("SCAR failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result
	}

	fmt.Println() // New line after progress

	fmt.Printf("✅ SCAR completed!\n")
	fmt.Printf("   📈 Modularity: %.6f\n", scarResult.Modularity)
	fmt.Printf("   🏘️  Communities: %d\n", len(getUniqueCommunities(scarResult.FinalCommunities)))
	fmt.Printf("   📊 Levels: %d\n", scarResult.NumLevels)
	fmt.Printf("   🔄 Total iterations: %d\n", scarResult.Statistics.TotalIterations)
	fmt.Printf("   ⏱️  Total duration: %v\n", scarResult.Statistics.TotalDuration)

	// Save outputs
	fmt.Println("\n💾 Saving Output Files")
	fmt.Println("--------------------")
	
	if err := saveScarOutputs(scarResult, scarGraph, outputDir); err != nil {
		fmt.Printf("⚠️  Warning: Failed to save some outputs: %v\n", err)
	} else {
		fmt.Printf("✅ Outputs saved to: %s\n", outputDir)
		fmt.Printf("   📁 Files generated:\n")
		fmt.Printf("      - scar.root (top-level structure)\n")
		fmt.Printf("      - hierarchy-output/ (hierarchical structure per level)\n")
		fmt.Printf("      - mapping-output/ (community mappings per level)\n")
		fmt.Printf("      - edges-output/ (inter-community edges per level)\n")
	}

	// Update result
	result.Communities = scarResult.FinalCommunities
	result.Modularity = scarResult.Modularity
	result.NumCommunities = len(getUniqueCommunities(scarResult.FinalCommunities))
	result.Runtime = time.Since(startTime)
	result.MemoryPeakMB = 0 // SCAR doesn't track this the same way
	result.Success = true

	fmt.Printf("\n📊 SCAR Results:")
	fmt.Printf("   ⏱️  Total runtime: %v\n", result.Runtime)
	fmt.Printf("   📈 Final modularity: %.6f\n", result.Modularity)
	fmt.Printf("   🏘️  Communities found: %d\n", result.NumCommunities)

	return result
}

func compareResults(matResult, scarResult *ApproachResult) {
	fmt.Println("┌────────────────────────┬──────────────────────┬──────────────────────┐")
	fmt.Println("│ Metric                 │ Materialization      │ SCAR                 │")
	fmt.Println("├────────────────────────┼──────────────────────┼──────────────────────┤")
	
	// Success status
	matSuccess := "❌"
	scarSuccess := "❌"
	if matResult.Success {
		matSuccess = "✅"
	}
	if scarResult.Success {
		scarSuccess = "✅"
	}
	
	fmt.Printf("│ Success                │ %-20s │ %-20s │\n", matSuccess, scarSuccess)
	
	if matResult.Success && scarResult.Success {
		// Runtime comparison
		fmt.Printf("│ Runtime                │ %-20s │ %-20s │\n", 
			matResult.Runtime.String(), scarResult.Runtime.String())
		
		// Memory comparison
		memMat := fmt.Sprintf("%d MB", matResult.MemoryPeakMB)
		memScar := "~0 MB (sketch-based)"
		fmt.Printf("│ Peak Memory            │ %-20s │ %-20s │\n", memMat, memScar)
		
		// Modularity comparison
		fmt.Printf("│ Modularity             │ %-20.6f │ %-20.6f │\n", 
			matResult.Modularity, scarResult.Modularity)
		
		// Community count comparison
		fmt.Printf("│ Communities Found      │ %-20d │ %-20d │\n", 
			matResult.NumCommunities, scarResult.NumCommunities)
		
	} else {
		// Show errors if either failed
		if !matResult.Success {
			fmt.Printf("│ Error                  │ %-20s │ %-20s │\n", 
				truncateString(matResult.Error, 20), "-")
		}
		if !scarResult.Success {
			fmt.Printf("│ Error                  │ %-20s │ %-20s │\n", 
				"-", truncateString(scarResult.Error, 20))
		}
	}
	
	fmt.Println("└────────────────────────┴──────────────────────┴──────────────────────┘")
	
	// Analysis
	fmt.Println("\n📊 Analysis:")
	
	if matResult.Success && scarResult.Success {
		// Compare runtime
		speedup := float64(matResult.Runtime.Nanoseconds()) / float64(scarResult.Runtime.Nanoseconds())
		if speedup > 1.1 {
			fmt.Printf("⚡ SCAR is %.1fx faster than materialization approach\n", speedup)
		} else if speedup < 0.9 {
			fmt.Printf("⚡ Materialization approach is %.1fx faster than SCAR\n", 1.0/speedup)
		} else {
			fmt.Println("⚡ Both approaches have similar runtime performance")
		}
		
		// Compare modularity
		modularityDiff := scarResult.Modularity - matResult.Modularity
		if abs(modularityDiff) < 0.01 {
			fmt.Printf("📈 Both approaches achieve similar modularity (difference: %.6f)\n", modularityDiff)
		} else if modularityDiff > 0 {
			fmt.Printf("📈 SCAR achieves higher modularity (+%.6f)\n", modularityDiff)
		} else {
			fmt.Printf("📈 Materialization achieves higher modularity (+%.6f)\n", -modularityDiff)
		}
		
		// Memory analysis
		if matResult.MemoryPeakMB > 100 {
			fmt.Printf("💾 Materialization uses significant memory (%d MB), SCAR is more memory-efficient\n", 
				matResult.MemoryPeakMB)
		}
	}
	
	// Recommendations
	fmt.Println("\n💡 Recommendations:")
	
	if matResult.Success && scarResult.Success {
		if matResult.MemoryPeakMB > 1000 {
			fmt.Println("   🏗️  For large graphs, prefer SCAR due to memory efficiency")
		}
		if scarResult.Runtime < matResult.Runtime && abs(scarResult.Modularity - matResult.Modularity) < 0.05 {
			fmt.Println("   ⚡ SCAR provides good speed-accuracy tradeoff for this graph")
		}
		if matResult.Modularity > scarResult.Modularity + 0.05 {
			fmt.Println("   🎯 If accuracy is critical, materialization approach may be preferred")
		}
	} else if matResult.Success && !scarResult.Success {
		fmt.Println("   🏗️  For this graph, materialization approach is more reliable")
	} else if !matResult.Success && scarResult.Success {
		fmt.Println("   ⚡ SCAR handles this graph better, likely due to memory constraints")
	} else {
		fmt.Println("   ❌ Both approaches failed - check input data and configuration")
	}
}

// Helper functions

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

func convertToLouvainGraph(homogGraph *materialization.HomogeneousGraph) *louvain.HomogeneousGraph {
	louvainGraph := louvain.NewHomogeneousGraph()

	// Add nodes
	for nodeID, _ := range homogGraph.Nodes {
		louvainGraph.AddNode(nodeID, 1.0) // Default weight
	}

	// Add edges
	for edgeKey, weight := range homogGraph.Edges {
		louvainGraph.AddEdge(edgeKey.From, edgeKey.To, weight)
	}

	return louvainGraph
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

func saveLouvainOutputs(result *louvain.LouvainResult, graph *louvain.HomogeneousGraph, outputDir string) error {
	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Use Louvain's file writer
	writer := louvain.NewFileWriter()
	return writer.WriteAll(result, graph, outputDir, "communities")
}

func saveScarOutputs(result *scar.ScarResult, graph *scar.HeterogeneousGraph, outputDir string) error {
	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Use SCAR's output writer
	return scar.WriteAll(result, graph, outputDir, "scar")
}

func getUniqueCommunities(communities map[string]int) map[int]bool {
	unique := make(map[int]bool)
	for _, comm := range communities {
		unique[comm] = true
	}
	return unique
}

func getNodeTypes(graph *models.HeterogeneousGraph) []string {
	types := make(map[string]bool)
	for _, node := range graph.Nodes {
		types[node.Type] = true
	}
	
	result := make([]string, 0, len(types))
	for nodeType := range types {
		result = append(result, nodeType)
	}
	return result
}

func getEdgeTypes(graph *models.HeterogeneousGraph) []string {
	types := make(map[string]bool)
	for _, edge := range graph.Edges {
		types[edge.Type] = true
	}
	
	result := make([]string, 0, len(types))
	for edgeType := range types {
		result = append(result, edgeType)
	}
	return result
}

func getStrategyName(strategy materialization.AggregationStrategy) string {
	switch strategy {
	case materialization.Count:
		return "Count"
	case materialization.Sum:
		return "Sum"
	case materialization.Average:
		return "Average"
	case materialization.Maximum:
		return "Maximum"
	case materialization.Minimum:
		return "Minimum"
	default:
		return "Unknown"
	}
}

func getInterpretationName(interpretation materialization.MetaPathInterpretation) string {
	switch interpretation {
	case materialization.DirectTraversal:
		return "Direct Traversal"
	case materialization.MeetingBased:
		return "Meeting Based"
	default:
		return "Unknown"
	}
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
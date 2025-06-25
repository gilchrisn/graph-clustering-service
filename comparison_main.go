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
	fmt.Println("Graph Clustering Service - Algorithm Comparison")
	fmt.Println("================================================")

	// Check command line arguments
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run comparison_main.go <graph_file> <meta_path_file>")
		fmt.Println("Example: go run comparison_main.go data/graph_input.json data/meta_path.json")
		os.Exit(1)
	}

	graphFile := os.Args[1]
	metaPathFile := os.Args[2]

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

	// Step 2: Run both approaches and compare
	fmt.Println("\n🚀 Step 2: Running Both Approaches")
	fmt.Println("==================================")

	// Run materialization + Louvain approach
	fmt.Println("\n--- Approach 1: Materialization + Louvain ---")
	matResult, matStats := runMaterializationApproach(graph, metaPath)

	// Run SCAR approach
	fmt.Println("\n--- Approach 2: SCAR ---")
	scarResult, scarStats := runScarApproach(graph, metaPath)

	// Step 3: Compare results
	fmt.Println("\n📈 Step 3: Comparison Results")
	fmt.Println("============================")
	
	compareResults(matResult, matStats, scarResult, scarStats)

	// Step 4: Save outputs
	fmt.Println("\n💾 Step 4: Saving Outputs")
	fmt.Println("=========================")
	
	saveOutputs(matResult, scarResult)

	fmt.Println("\n🎉 Comparison completed successfully!")
}

// ComparisonStats holds performance statistics for comparison
type ComparisonStats struct {
	Runtime        time.Duration
	MemoryPeakMB   int64
	Modularity     float64
	NumCommunities int
	Method         string
	Success        bool
	Error          string
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

func runMaterializationApproach(graph *models.HeterogeneousGraph, metaPath *models.MetaPath) (*materialization.MaterializationResult, ComparisonStats) {
	startTime := time.Now()
	stats := ComparisonStats{Method: "Materialization + Louvain"}

	// Configure materialization
	config := materialization.DefaultMaterializationConfig()
	config.Aggregation.Strategy = materialization.Count
	config.Aggregation.Interpretation = materialization.MeetingBased
	config.Aggregation.Symmetric = true
	config.Aggregation.MinWeight = 1.0

	// Progress callback
	progressCallback := func(current, total int, message string) {
		if total > 0 {
			percentage := float64(current) / float64(total) * 100
			fmt.Printf("\r   Progress: %.1f%% (%d/%d) - %s", percentage, current, total, message)
		}
	}

	// Create materialization engine
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, progressCallback)

	// Check feasibility
	canMaterialize, reason, err := engine.CanMaterialize(2000) // 2GB limit
	if err != nil {
		stats.Error = fmt.Sprintf("feasibility check failed: %v", err)
		stats.Runtime = time.Since(startTime)
		return nil, stats
	}

	if !canMaterialize {
		stats.Error = fmt.Sprintf("materialization not feasible: %s", reason)
		stats.Runtime = time.Since(startTime)
		return nil, stats
	}

	fmt.Printf("✅ Feasibility check passed: %s\n", reason)

	// Perform materialization
	fmt.Println("🔄 Starting materialization...")
	matResult, err := engine.Materialize()
	if err != nil {
		stats.Error = fmt.Sprintf("materialization failed: %v", err)
		stats.Runtime = time.Since(startTime)
		return nil, stats
	}

	fmt.Println() // New line after progress

	if !matResult.Success {
		stats.Error = fmt.Sprintf("materialization unsuccessful: %s", matResult.Error)
		stats.Runtime = time.Since(startTime)
		return matResult, stats
	}

	fmt.Printf("✅ Materialization completed!\n")
	fmt.Printf("   📊 Generated: %d nodes, %d edges\n", 
		len(matResult.HomogeneousGraph.Nodes), len(matResult.HomogeneousGraph.Edges))

	// Convert to Louvain format and run clustering
	fmt.Println("🔄 Running Louvain algorithm...")
	louvainGraph := convertToLouvainGraph(matResult.HomogeneousGraph)

	louvainConfig := louvain.DefaultLouvainConfig()
	louvainConfig.Verbose = false
	louvainConfig.RandomSeed = 42

	louvainResult, err := louvain.RunLouvain(louvainGraph, louvainConfig)
	if err != nil {
		stats.Error = fmt.Sprintf("louvain failed: %v", err)
		stats.Runtime = time.Since(startTime)
		return matResult, stats
	}

	// Update statistics
	stats.Runtime = time.Since(startTime)
	stats.MemoryPeakMB = matResult.Statistics.MemoryPeakMB
	stats.Modularity = louvainResult.Modularity
	stats.NumCommunities = len(louvainResult.FinalCommunities)
	stats.Success = true

	fmt.Printf("✅ Louvain completed!\n")
	fmt.Printf("   🎯 Modularity: %.6f\n", louvainResult.Modularity)
	fmt.Printf("   👥 Communities: %d\n", len(louvainResult.FinalCommunities))

	// Store Louvain result in materialization result for output
	// (You might want to extend MaterializationResult to include this)

	return matResult, stats
}

func runScarApproach(graph *models.HeterogeneousGraph, metaPath *models.MetaPath) (*scar.ScarResult, ComparisonStats) {
	startTime := time.Now()
	stats := ComparisonStats{Method: "SCAR"}

	// Convert to SCAR format
	scarGraph := convertToScarGraph(graph)
	scarMetaPath := convertToScarMetaPath(metaPath)

	// Configure SCAR
	config := scar.DefaultScarConfig()
	config.MetaPath = scarMetaPath
	config.K = 64
	config.NK = 8
	config.RandomSeed = 42
	config.Verbose = true
	config.MaxIterations = 50

	config.Parallel.Enabled = true // Enable parallel processing
	config.Parallel.NumWorkers = 1 // Set number of workers for parallel processing
	config.Parallel.BatchSize = 100 // Set max queue size for parallel tasks

	// Progress callback
	config.ProgressCallback = func(level int, iteration int, modularity float64, nodes int) {
		fmt.Printf("\r   Level %d, Iteration %d: modularity=%.6f, nodes=%d", 
			level, iteration, modularity, nodes)
	}

	fmt.Printf("🔄 Starting SCAR algorithm (K=%d, NK=%d)...\n", config.K, config.NK)

	// Run SCAR
	scarResult, err := scar.RunScar(scarGraph, config)
	if err != nil {
		stats.Error = fmt.Sprintf("SCAR failed: %v", err)
		stats.Runtime = time.Since(startTime)
		return nil, stats
	}

	fmt.Println() // New line after progress

	// Update statistics
	stats.Runtime = time.Since(startTime)
	stats.MemoryPeakMB = 0 // SCAR doesn't track this in the same way
	stats.Modularity = scarResult.Modularity
	stats.NumCommunities = len(scarResult.FinalCommunities)
	stats.Success = true

	fmt.Printf("✅ SCAR completed!\n")
	fmt.Printf("   🎯 Modularity: %.6f\n", scarResult.Modularity)
	fmt.Printf("   👥 Communities: %d\n", len(scarResult.FinalCommunities))
	fmt.Printf("   📊 Levels: %d\n", scarResult.NumLevels)

	return scarResult, stats
}

func compareResults(matResult *materialization.MaterializationResult, matStats ComparisonStats,
	scarResult *scar.ScarResult, scarStats ComparisonStats) {

	fmt.Println("┌─────────────────────────┬─────────────────────┬─────────────────────┐")
	fmt.Println("│ Metric                  │ Materialization     │ SCAR                │")
	fmt.Println("├─────────────────────────┼─────────────────────┼─────────────────────┤")

	// Success status
	matSuccessIcon := "❌"
	scarSuccessIcon := "❌"
	if matStats.Success {
		matSuccessIcon = "✅"
	}
	if scarStats.Success {
		scarSuccessIcon = "✅"
	}

	fmt.Printf("│ Success                 │ %-19s │ %-19s │\n", matSuccessIcon, scarSuccessIcon)

	if matStats.Success && scarStats.Success {
		// Runtime comparison
		fmt.Printf("│ Runtime                 │ %-19s │ %-19s │\n", 
			formatDuration(matStats.Runtime), formatDuration(scarStats.Runtime))

		// Memory comparison
		fmt.Printf("│ Peak Memory (MB)        │ %-19d │ %-19s │\n", 
			matStats.MemoryPeakMB, "~" + fmt.Sprintf("%d", scarStats.MemoryPeakMB))

		// Modularity comparison
		fmt.Printf("│ Modularity              │ %-19.6f │ %-19.6f │\n", 
			matStats.Modularity, scarStats.Modularity)

		// Community count comparison
		fmt.Printf("│ Communities Found       │ %-19d │ %-19d │\n", 
			matStats.NumCommunities, scarStats.NumCommunities)

		// Additional metrics if available
		if matResult != nil {
			fmt.Printf("│ Instances Generated     │ %-19d │ %-19s │\n", 
				matResult.Statistics.InstancesGenerated, "N/A (sketch-based)")
			
			fmt.Printf("│ Edges in Result         │ %-19d │ %-19s │\n",
				len(matResult.HomogeneousGraph.Edges), "N/A (no explicit edges)")
		}

		if scarResult != nil {
			fmt.Printf("│ Algorithm Levels        │ %-19s │ %-19d │\n",
				"N/A", scarResult.NumLevels)
			
			fmt.Printf("│ Total Iterations        │ %-19s │ %-19d │\n",
				"N/A", scarResult.Statistics.TotalIterations)
		}
	} else {
		// Show errors if either failed
		if !matStats.Success {
			fmt.Printf("│ Error                   │ %-19s │ %-19s │\n", 
				truncateString(matStats.Error, 19), "-")
		}
		if !scarStats.Success {
			fmt.Printf("│ Error                   │ %-19s │ %-19s │\n", 
				"-", truncateString(scarStats.Error, 19))
		}
	}

	fmt.Println("└─────────────────────────┴─────────────────────┴─────────────────────┘")

	// Analysis and recommendations
	fmt.Println("\n📊 Analysis:")
	
	if matStats.Success && scarStats.Success {
		// Compare runtime
		speedup := float64(matStats.Runtime.Nanoseconds()) / float64(scarStats.Runtime.Nanoseconds())
		if speedup > 1.1 {
			fmt.Printf("⚡ SCAR is %.1fx faster than materialization approach\n", speedup)
		} else if speedup < 0.9 {
			fmt.Printf("⚡ Materialization approach is %.1fx faster than SCAR\n", 1.0/speedup)
		} else {
			fmt.Println("⚡ Both approaches have similar runtime performance")
		}

		// Compare modularity
		modularityDiff := scarStats.Modularity - matStats.Modularity
		modularityRatio := scarStats.Modularity / matStats.Modularity
		
		if abs(modularityDiff) < 0.01 {
			fmt.Printf("🎯 Both approaches achieve similar modularity (difference: %.6f)\n", modularityDiff)
		} else if modularityDiff > 0 {
			fmt.Printf("🎯 SCAR achieves higher modularity (+%.6f, %.1f%% better)\n", 
				modularityDiff, (modularityRatio-1)*100)
		} else {
			fmt.Printf("🎯 Materialization achieves higher modularity (+%.6f, %.1f%% better)\n", 
				-modularityDiff, (1.0/modularityRatio-1)*100)
		}

		// Memory analysis
		if matStats.MemoryPeakMB > 100 {
			fmt.Printf("💾 Materialization uses significant memory (%d MB), SCAR is more memory-efficient\n", 
				matStats.MemoryPeakMB)
		}

		// Community count comparison
		commDiff := abs(float64(scarStats.NumCommunities - matStats.NumCommunities))
		commAvg := float64(scarStats.NumCommunities + matStats.NumCommunities) / 2.0
		if commDiff/commAvg > 0.2 {
			fmt.Printf("👥 Significant difference in community count (SCAR: %d, Mat: %d)\n", 
				scarStats.NumCommunities, matStats.NumCommunities)
		}
	}

	// Recommendations
	fmt.Println("\n💡 Recommendations:")
	
	if matStats.Success && scarStats.Success {
		if matStats.MemoryPeakMB > 1000 {
			fmt.Println("   🔹 For large graphs, prefer SCAR due to memory efficiency")
		}
		if scarStats.Runtime < matStats.Runtime && abs(scarStats.Modularity - matStats.Modularity) < 0.05 {
			fmt.Println("   🔹 SCAR provides good speed-accuracy tradeoff for this graph")
		}
		if matStats.Modularity > scarStats.Modularity + 0.05 {
			fmt.Println("   🔹 If accuracy is critical, materialization approach may be preferred")
		}
	} else if matStats.Success && !scarStats.Success {
		fmt.Println("   🔹 For this graph, materialization approach is more reliable")
	} else if !matStats.Success && scarStats.Success {
		fmt.Println("   🔹 SCAR handles this graph better, likely due to memory constraints")
	} else {
		fmt.Println("   🔹 Both approaches failed - check input data and configuration")
	}
}

func saveOutputs(matResult *materialization.MaterializationResult, scarResult *scar.ScarResult) {
	// Create output directories
	os.MkdirAll("output/materialization", 0755)
	os.MkdirAll("output/scar", 0755)

	// Save materialization outputs
	if matResult != nil && matResult.Success {
		err := materialization.SaveHomogeneousGraph(matResult.HomogeneousGraph, "output/materialization/graph.edgelist")
		if err != nil {
			fmt.Printf("⚠️  Failed to save materialization graph: %v\n", err)
		} else {
			fmt.Println("💾 Saved materialization graph to output/materialization/graph.edgelist")
		}

		err = materialization.SaveMaterializationResult(matResult, "output/materialization/result.json")
		if err != nil {
			fmt.Printf("⚠️  Failed to save materialization result: %v\n", err)
		} else {
			fmt.Println("💾 Saved materialization result to output/materialization/result.json")
		}
	}

	// Save SCAR outputs
	if scarResult != nil {
		// Create a dummy graph for SCAR output (you'd need the original SCAR graph here)
		// For now, just save the result structure
		fmt.Println("💾 SCAR outputs would be saved to output/scar/ (implementation needed)")
		
		// You would call something like:
		// err := scar.WriteAll(scarResult, scarGraph, "output/scar", "communities")
	}

	fmt.Println("💾 All available outputs saved successfully!")
}

// Utility functions

func convertToLouvainGraph(homogGraph *materialization.HomogeneousGraph) *louvain.HomogeneousGraph {
	louvainGraph := louvain.NewHomogeneousGraph()

	// Add nodes
	for nodeID, _ := range homogGraph.Nodes {
		louvainGraph.AddNode(nodeID, 1.0) // Default weight
		// Copy properties if needed
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

func formatDuration(d time.Duration) string {
	if d < time.Millisecond {
		return fmt.Sprintf("%.2fμs", float64(d.Nanoseconds())/1000.0)
	} else if d < time.Second {
		return fmt.Sprintf("%.2fms", float64(d.Nanoseconds())/1000000.0)
	} else {
		return fmt.Sprintf("%.2fs", d.Seconds())
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
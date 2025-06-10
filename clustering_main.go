// Example usage of the clustering package
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/gilchrisn/graph-clustering-service/pkg/clustering"
	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
)

func main() {
	fmt.Println("🔬 Clustering Package Example Usage")
	fmt.Println("===================================")

	// Example 1: Run Materialization + Louvain approach
	runMaterializationExample()

	fmt.Println()

	// Example 2: Run SCAR approach
	runScarExample()

	fmt.Println()

	// Example 3: Compare both approaches
	runComparisonExample()
}

func runMaterializationExample() {
	fmt.Println("🔄 Example 1: Materialization + Louvain Approach")
	fmt.Println("===============================================")

	// Create configuration with custom settings
	config := clustering.DefaultMaterializationConfig()
	config.GraphFile = "data/imdb_graph.json"
	config.MetaPathFile = "data/imdb_graph_meta1.json"
	config.OutputDir = "output/materialization_example"
	config.AggregationStrategy = materialization.Count
	config.MetaPathInterpretation = materialization.MeetingBased
	config.Symmetric = true
	config.MinWeight = 1.0
	config.MaxInstances = 500000
	config.TimeoutSeconds = 180
	config.TraversalParallelism = 4
	config.LouvainMaxIterations = 1
	config.LouvainMinModularity = 0.00001
	config.RandomSeed = 42
	config.OutputPrefix = "communities"
	config.Verbose = true

	fmt.Printf("📋 Configuration:\n")
	fmt.Printf("   🏗️  Strategy: %v\n", config.AggregationStrategy)
	fmt.Printf("   🎯 Interpretation: %v\n", config.MetaPathInterpretation)
	fmt.Printf("   📊 Max Instances: %d\n", config.MaxInstances)
	fmt.Printf("   🔄 Parallelism: %d\n", config.TraversalParallelism)

	// Run clustering
	result, err := clustering.RunMaterializationClustering(config)
	if err != nil {
		log.Printf("❌ Materialization failed: %v", err)
		return
	}

	// Display results
	displayResults("Materialization + Louvain", result)

	// Verify outputs
	if err := clustering.VerifyClusteringResult(result); err != nil {
		log.Printf("⚠️  Verification warning: %v", err)
	} else {
		fmt.Println("✅ Output verification passed!")
	}

	// Save result to JSON for later analysis
	saveResultToJSON(result, "output/materialization_result.json")
}

func runScarExample() {
	fmt.Println("⚡ Example 2: SCAR Approach")
	fmt.Println("=========================")

	// Create configuration with custom settings
	config := clustering.DefaultScarConfig()
	config.GraphFile = "data/imdb_graph.json"
	config.MetaPathFile = "data/imdb_graph_meta1.json"
	config.OutputDir = "output/scar_example"
	config.K = 64
	config.NK = 8
	config.MaxIterations = 50
	config.MinModularity = 1e-6
	config.RandomSeed = 42
	config.ParallelEnabled = true
	config.NumWorkers = 4
	config.BatchSize = 100
	config.UpdateBuffer = 10000
	config.OutputPrefix = "scar"
	config.Verbose = true

	fmt.Printf("📋 Configuration:\n")
	fmt.Printf("   📏 K (sketch size): %d\n", config.K)
	fmt.Printf("   🔢 NK (hash functions): %d\n", config.NK)
	fmt.Printf("   🔁 Max iterations: %d\n", config.MaxIterations)
	fmt.Printf("   ⚡ Parallel: %v (%d workers)\n", config.ParallelEnabled, config.NumWorkers)

	// Run clustering
	result, err := clustering.RunScarClustering(config)
	if err != nil {
		log.Printf("❌ SCAR failed: %v", err)
		return
	}

	// Display results
	displayResults("SCAR", result)

	// Verify outputs
	if err := clustering.VerifyClusteringResult(result); err != nil {
		log.Printf("⚠️  Verification warning: %v", err)
	} else {
		fmt.Println("✅ Output verification passed!")
	}

	// Save result to JSON for later analysis
	saveResultToJSON(result, "output/scar_result.json")
}

func runComparisonExample() {
	fmt.Println("📊 Example 3: Comparing Both Approaches")
	fmt.Println("======================================")

	// Configure both approaches with identical base settings
	baseGraphFile := "data/example_graph.json"
	baseMetaPathFile := "data/example_meta_path.json"
	baseSeed := int64(42)

	// Materialization config
	matConfig := clustering.DefaultMaterializationConfig()
	matConfig.GraphFile = baseGraphFile
	matConfig.MetaPathFile = baseMetaPathFile
	matConfig.OutputDir = "output/comparison/materialization"
	matConfig.RandomSeed = baseSeed
	matConfig.Verbose = false // Less verbose for comparison

	// SCAR config
	scarConfig := clustering.DefaultScarConfig()
	scarConfig.GraphFile = baseGraphFile
	scarConfig.MetaPathFile = baseMetaPathFile
	scarConfig.OutputDir = "output/comparison/scar"
	scarConfig.RandomSeed = baseSeed
	scarConfig.Verbose = false // Less verbose for comparison

	fmt.Println("🔄 Running Materialization + Louvain...")
	matResult, matErr := clustering.RunMaterializationClustering(matConfig)

	fmt.Println("⚡ Running SCAR...")
	scarResult, scarErr := clustering.RunScarClustering(scarConfig)

	// Compare results
	fmt.Println("\n📊 Comparison Results:")
	fmt.Println("┌─────────────────────────┬────────────────────┬────────────────────┐")
	fmt.Println("│ Metric                  │ Materialization    │ SCAR               │")
	fmt.Println("├─────────────────────────┼────────────────────┼────────────────────┤")

	// Success status
	matStatus := "❌"
	scarStatus := "❌"
	if matErr == nil && matResult.Success {
		matStatus = "✅"
	}
	if scarErr == nil && scarResult.Success {
		scarStatus = "✅"
	}

	fmt.Printf("│ Success                 │ %-18s │ %-18s │\n", matStatus, scarStatus)

	if matErr == nil && scarErr == nil && matResult.Success && scarResult.Success {
		// Performance comparison
		fmt.Printf("│ Runtime                 │ %-18s │ %-18s │\n", 
			matResult.Runtime.String(), scarResult.Runtime.String())
		fmt.Printf("│ Memory Peak (MB)        │ %-18d │ %-18s │\n", 
			matResult.MemoryPeakMB, "~0 (sketch)")
		fmt.Printf("│ Modularity              │ %-18.6f │ %-18.6f │\n", 
			matResult.Modularity, scarResult.Modularity)
		fmt.Printf("│ Communities             │ %-18d │ %-18d │\n", 
			matResult.NumCommunities, scarResult.NumCommunities)
		fmt.Printf("│ Levels                  │ %-18d │ %-18d │\n", 
			matResult.NumLevels, scarResult.NumLevels)
		fmt.Printf("│ Total Iterations        │ %-18d │ %-18d │\n", 
			matResult.TotalIterations, scarResult.TotalIterations)

		// Analysis
		fmt.Println("└─────────────────────────┴────────────────────┴────────────────────┘")
		fmt.Println("\n🔍 Analysis:")

		// Speed comparison
		speedRatio := float64(matResult.Runtime.Nanoseconds()) / float64(scarResult.Runtime.Nanoseconds())
		if speedRatio > 1.2 {
			fmt.Printf("⚡ SCAR is %.1fx faster than Materialization\n", speedRatio)
		} else if speedRatio < 0.8 {
			fmt.Printf("⚡ Materialization is %.1fx faster than SCAR\n", 1.0/speedRatio)
		} else {
			fmt.Println("⚡ Both approaches have similar performance")
		}

		// Quality comparison
		modDiff := scarResult.Modularity - matResult.Modularity
		if abs(modDiff) < 0.01 {
			fmt.Printf("📈 Similar modularity quality (diff: %.6f)\n", modDiff)
		} else if modDiff > 0 {
			fmt.Printf("📈 SCAR achieves higher modularity (+%.6f)\n", modDiff)
		} else {
			fmt.Printf("📈 Materialization achieves higher modularity (+%.6f)\n", -modDiff)
		}

		// Memory efficiency
		if matResult.MemoryPeakMB > 500 {
			fmt.Printf("💾 SCAR is significantly more memory-efficient (%d MB vs ~0 MB)\n", 
				matResult.MemoryPeakMB)
		}

		// Save comparison report
		saveComparisonReport(matResult, scarResult, "output/comparison_report.json")

	} else {
		fmt.Println("└─────────────────────────┴────────────────────┴────────────────────┘")
		if matErr != nil {
			fmt.Printf("❌ Materialization error: %v\n", matErr)
		}
		if scarErr != nil {
			fmt.Printf("❌ SCAR error: %v\n", scarErr)
		}
	}
}

// Helper functions

func displayResults(approach string, result *clustering.ClusteringResult) {
	fmt.Printf("\n📊 %s Results:\n", approach)
	if result.Success {
		fmt.Printf("   ✅ Success: %v\n", result.Success)
		fmt.Printf("   ⏱️  Runtime: %v\n", result.Runtime)
		fmt.Printf("   📈 Modularity: %.6f\n", result.Modularity)
		fmt.Printf("   🏘️  Communities: %d\n", result.NumCommunities)
		fmt.Printf("   📊 Levels: %d\n", result.NumLevels)
		fmt.Printf("   🔄 Iterations: %d\n", result.TotalIterations)
		fmt.Printf("   💾 Memory: %d MB\n", result.MemoryPeakMB)
		fmt.Printf("   📁 Output files:\n")
		if result.OutputFiles.MappingFile != "" {
			fmt.Printf("      - %s\n", result.OutputFiles.MappingFile)
		}
		if result.OutputFiles.HierarchyFile != "" {
			fmt.Printf("      - %s\n", result.OutputFiles.HierarchyFile)
		}
		if result.OutputFiles.RootFile != "" {
			fmt.Printf("      - %s\n", result.OutputFiles.RootFile)
		}
		if result.OutputFiles.EdgesFile != "" {
			fmt.Printf("      - %s\n", result.OutputFiles.EdgesFile)
		}
		if result.OutputFiles.HierarchyDir != "" {
			fmt.Printf("      - %s/ (directory)\n", result.OutputFiles.HierarchyDir)
		}
		if result.OutputFiles.MappingDir != "" {
			fmt.Printf("      - %s/ (directory)\n", result.OutputFiles.MappingDir)
		}
		if result.OutputFiles.EdgesDir != "" {
			fmt.Printf("      - %s/ (directory)\n", result.OutputFiles.EdgesDir)
		}
	} else {
		fmt.Printf("   ❌ Failed: %s\n", result.Error)
	}
}

func saveResultToJSON(result *clustering.ClusteringResult, filename string) {
	// Create output directory
	os.MkdirAll("output", 0755)

	// Save to JSON
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		log.Printf("⚠️  Failed to marshal result: %v", err)
		return
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		log.Printf("⚠️  Failed to save result: %v", err)
		return
	}

	fmt.Printf("💾 Result saved to: %s\n", filename)
}

func saveComparisonReport(matResult, scarResult *clustering.ClusteringResult, filename string) {
	report := map[string]interface{}{
		"comparison_timestamp": fmt.Sprintf("%v", matResult.Runtime),
		"materialization": map[string]interface{}{
			"success":        matResult.Success,
			"runtime_ms":     matResult.Runtime.Milliseconds(),
			"memory_mb":      matResult.MemoryPeakMB,
			"modularity":     matResult.Modularity,
			"communities":    matResult.NumCommunities,
			"levels":         matResult.NumLevels,
			"iterations":     matResult.TotalIterations,
		},
		"scar": map[string]interface{}{
			"success":        scarResult.Success,
			"runtime_ms":     scarResult.Runtime.Milliseconds(),
			"memory_mb":      scarResult.MemoryPeakMB,
			"modularity":     scarResult.Modularity,
			"communities":    scarResult.NumCommunities,
			"levels":         scarResult.NumLevels,
			"iterations":     scarResult.TotalIterations,
		},
		"analysis": map[string]interface{}{
			"speed_ratio":      float64(matResult.Runtime.Nanoseconds()) / float64(scarResult.Runtime.Nanoseconds()),
			"modularity_diff":  scarResult.Modularity - matResult.Modularity,
			"memory_savings":   matResult.MemoryPeakMB, // SCAR uses ~0
		},
	}

	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		log.Printf("⚠️  Failed to create comparison report: %v", err)
		return
	}

	os.MkdirAll("output", 0755)
	if err := os.WriteFile(filename, data, 0644); err != nil {
		log.Printf("⚠️  Failed to save comparison report: %v", err)
		return
	}

	fmt.Printf("📊 Comparison report saved to: %s\n", filename)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// Example of how to use the package programmatically in other projects
func ExampleProgrammaticUsage() {
	// This shows how you would import and use the clustering package in your own code

	// 1. Simple materialization clustering
	matConfig := clustering.DefaultMaterializationConfig()
	matConfig.GraphFile = "my_graph.json"
	matConfig.MetaPathFile = "my_meta_path.json"
	matConfig.OutputDir = "my_output/"
	matConfig.Verbose = false

	result1, err := clustering.RunMaterializationClustering(matConfig)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Found %d communities with modularity %.3f\n", 
		result1.NumCommunities, result1.Modularity)

	// 2. Simple SCAR clustering
	scarConfig := clustering.DefaultScarConfig()
	scarConfig.GraphFile = "my_graph.json"
	scarConfig.MetaPathFile = "my_meta_path.json"
	scarConfig.OutputDir = "my_output_scar/"
	scarConfig.Verbose = false

	result2, err := clustering.RunScarClustering(scarConfig)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Found %d communities with modularity %.3f\n", 
		result2.NumCommunities, result2.Modularity)

	// 3. Verify results
	if err := clustering.VerifyClusteringResult(result1); err != nil {
		log.Printf("Verification failed: %v", err)
	}

	if err := clustering.VerifyClusteringResult(result2); err != nil {
		log.Printf("Verification failed: %v", err)
	}
}
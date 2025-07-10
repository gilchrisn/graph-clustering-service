package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	// "sort"

	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

// PipelineType defines which pipeline to run
type PipelineType int

const (
	HighPrecisionSCAR PipelineType = iota
	OptimizedSCAR
	SCARComparison
	MultiKComparison
)

// PipelineConfig holds configuration for both SCAR pipeline types
type PipelineConfig struct {
	// Common options
	Verbose      bool
	OutputDir    string
	OutputPrefix string
	
	// High-precision SCAR config (approximates materialization+louvain)
	HighPrecisionSCARConfig scar.SCARConfig
	
	// Optimized SCAR config (from command line)
	OptimizedSCARConfig scar.SCARConfig
}

// PipelineResult contains results from either pipeline
type PipelineResult struct {
	PipelineType    PipelineType
	TotalRuntimeMS  int64
	
	// SCAR results (basic info - actual files written to disk)
	SCARSuccess bool
	SCARConfig  *scar.SCARConfig
}

// ComparisonResult contains results from comparing both SCAR pipelines
type ComparisonResult struct {
	HighPrecisionResult *PipelineResult
	OptimizedResult     *PipelineResult
	
	// Hierarchical clustering metrics by level
	NMIScores           map[int]float64 // Normalized Mutual Information
	ARIScores           map[int]float64 // Adjusted Rand Index
	VIScores            map[int]float64 // Variation of Information
	HierarchicalF1Scores map[int]float64 // Hierarchical F1
	
	// Structural hierarchy metrics
	TreeEditDistance    float64         // Tree structure similarity
	HierarchyDepthDiff  int            // Difference in hierarchy depths
	
	// Partition data
	HighPrecisionPartition *HierarchicalPartition
	OptimizedPartition     *HierarchicalPartition
	
	// Timing and basic info
	NumNodes            int
	HighPrecisionTime   int64
	OptimizedTime       int64
	TotalComparisonTime int64
	CommonLevels        []int  // Levels present in both partitions
	
	// Community structure comparison
	CommunityCountDiff  map[int]int     // Difference in community counts by level
	AvgCommunitySizeDiff map[int]float64 // Difference in average community sizes by level
}

// MultiKComparisonResult contains results from comparing across multiple k values
type MultiKComparisonResult struct {
	HighPrecisionResult *PipelineResult
	KResults           map[int64]*PipelineResult  // Results for each k value
	KPartitions        map[int64]*HierarchicalPartition // Partitions for each k value
	
	// Metrics by k value and level: [k][level] -> metric_value
	KNMIScores           map[int64]map[int]float64
	KARIScores           map[int64]map[int]float64
	KVIScores            map[int64]map[int]float64
	KHierarchicalF1Scores map[int64]map[int]float64
	
	// Aggregate metrics by k value
	KAvgNMI              map[int64]float64
	KAvgARI              map[int64]float64
	KAvgVI               map[int64]float64
	KAvgHF1              map[int64]float64
	KTreeEditDistance    map[int64]float64
	KHierarchyDepthDiff  map[int64]int
	
	// Timing by k value
	KExecutionTime       map[int64]int64
	
	// Test configuration
	TestedKValues        []int64
	HighPrecisionPartition *HierarchicalPartition
	NumNodes             int
	TotalComparisonTime  int64
}


// NewPipelineConfig creates default configuration for both SCAR pipelines
func NewPipelineConfig() *PipelineConfig {
	return &PipelineConfig{
		Verbose:      true,
		OutputDir:    "scar_comparison_output",
		OutputPrefix: "communities",
		
		// High-precision SCAR config (approximates materialization+louvain)
		HighPrecisionSCARConfig: scar.SCARConfig{
			K:           1024,  // Large k for high precision
			NK:          1,    // Single layer
			Threshold:   0.0,  // No threshold
			UseLouvain:  true,
			SketchOutput: true, // For hierarchy output compatible with PPRViz
			WriteSketchGraph: true, // Write sketch graph files
			SketchGraphWeights: false, // Use weights in sketch graph files
		},
		
		// Default optimized SCAR config (will be overridden by command line)
		OptimizedSCARConfig: scar.SCARConfig{
			K:           64,
			NK:          4,
			Threshold:   0.5,
			UseLouvain:  true,
			SketchOutput: true, // For hierarchy output compatible with PPRViz
			WriteSketchGraph: true, // Write sketch graph files
			SketchGraphWeights: false, // Use weights in sketch graph files
		},
	}
}

// RunHighPrecisionSCAR executes the high-precision SCAR pipeline (approximates materialization+louvain)
func RunHighPrecisionSCAR(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*PipelineResult, error) {
	startTime := time.Now()
	
	if config.Verbose {
		fmt.Println("=== Running High-Precision SCAR Pipeline (k=512, nk=1, th=0) ===")
	}
	
	// Step 1: Configure SCAR with input files
	if config.Verbose {
		fmt.Println("Step 1: Configuring high-precision SCAR engine...")
	}
	
	// Create a copy of SCAR config and set file paths
	scarConfig := config.HighPrecisionSCARConfig
	scarConfig.GraphFile = graphFile
	scarConfig.PropertyFile = propertiesFile
	scarConfig.PathFile = pathFile
	scarConfig.Prefix = filepath.Join(config.OutputDir, config.OutputPrefix)
	scarConfig.NumWorkers = 1
	if config.Verbose {
		fmt.Printf("  Graph file: %s\n", graphFile)
		fmt.Printf("  Properties file: %s\n", propertiesFile)
		fmt.Printf("  Path file: %s\n", pathFile)
		fmt.Printf("  High-precision SCAR parameters: k=%d, nk=%d, threshold=%.3f\n", 
			scarConfig.K, scarConfig.NK, scarConfig.Threshold)
		fmt.Printf("  Sketch output: %t\n", scarConfig.SketchOutput)
	}
	
	// Step 2: Create output directory
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Step 3: Run SCAR engine
	if config.Verbose {
		fmt.Println("Step 2: Running high-precision SCAR...")
	}
	
	scarStart := time.Now()
	
	engine := scar.NewSketchLouvainEngine(scarConfig)
	err := engine.RunLouvain()
	if err != nil {
		return nil, fmt.Errorf("high-precision SCAR failed: %w", err)
	}
	
	scarTime := time.Since(scarStart)
	totalTime := time.Since(startTime)
	
	if config.Verbose {
		fmt.Printf("  High-precision SCAR completed in %v\n", scarTime)
		fmt.Println("Step 3: Writing high-precision SCAR summary...")
	}
	
	// Step 4: Write pipeline summary (SCAR writes its own output files)
	if err := writeSCARSummary(&scarConfig, config, totalTime, "high_precision"); err != nil {
		return nil, fmt.Errorf("failed to write high-precision SCAR summary: %w", err)
	}
	
	// Create final result
	result := &PipelineResult{
		PipelineType:   HighPrecisionSCAR,
		TotalRuntimeMS: totalTime.Milliseconds(),
		SCARSuccess:    true,
		SCARConfig:     &scarConfig,
	}
	
	if config.Verbose {
		fmt.Println("=== High-Precision SCAR Pipeline Complete ===")
		fmt.Printf("Total runtime: %v\n", totalTime)
		fmt.Printf("SCAR execution: %v\n", scarTime)
		if scarConfig.SketchOutput {
			fmt.Println("Generated SCAR hierarchy files for PPRViz integration")
		}
	}
	
	return result, nil
}

// RunMultiKComparison executes SCAR across multiple k values and compares against high-precision baseline
func RunMultiKComparison(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*MultiKComparisonResult, error) {
	startTime := time.Now()
	
	// Define k values to test: powers of 2 from 1 to 512
	kValues := []int64{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
	
	if config.Verbose {
		fmt.Println("=== Running Multi-K SCAR Comparison ===")
		fmt.Printf("Testing k values: %v\n", kValues)
		fmt.Println("Comparing against high-precision baseline (k=512)")
	}
	
	// Step 1: Run high-precision SCAR baseline (k=512)
	if config.Verbose {
		fmt.Println("\nStep 1: Running high-precision SCAR baseline...")
	}
	
	baselineOutputDir := filepath.Join(config.OutputDir, "baseline_k512")
	baselineConfig := *config
	baselineConfig.OutputDir = baselineOutputDir
	baselineConfig.OutputPrefix = "baseline_communities"
	
	baselineResult, err := RunHighPrecisionSCAR(graphFile, propertiesFile, pathFile, &baselineConfig)
	if err != nil {
		return nil, fmt.Errorf("baseline high-precision SCAR failed: %w", err)
	}
	
	baselinePartition, err := parseSCAROutput(baselineOutputDir, "baseline_communities")
	if err != nil {
		return nil, fmt.Errorf("failed to parse baseline output: %w", err)
	}
	
	if config.Verbose {
		fmt.Printf("Baseline completed: %d levels, max level %d\n", 
			len(baselinePartition.Levels), baselinePartition.MaxLevel)
	}
	
	// Step 2: Run SCAR for each k value
	kResults := make(map[int64]*PipelineResult)
	kPartitions := make(map[int64]*HierarchicalPartition)
	kExecutionTime := make(map[int64]int64)
	
	for _, k := range kValues {
		if k == 512 {
			// Skip k=512 since it's the baseline
			kResults[k] = baselineResult
			kPartitions[k] = baselinePartition
			kExecutionTime[k] = baselineResult.TotalRuntimeMS
			continue
		}
		
		if config.Verbose {
			fmt.Printf("\nStep 2.%d: Running SCAR with k=%d...\n", k, k)
		}
		
		// Configure for this k value
		kOutputDir := filepath.Join(config.OutputDir, fmt.Sprintf("k_%d", k))
		kConfig := *config
		kConfig.OutputDir = kOutputDir
		kConfig.OutputPrefix = fmt.Sprintf("k%d_communities", k)
		kConfig.OptimizedSCARConfig.K = k
		
		kStart := time.Now()
		kResult, err := RunOptimizedSCAR(graphFile, propertiesFile, pathFile, &kConfig)
		if err != nil {
			return nil, fmt.Errorf("SCAR with k=%d failed: %w", k, err)
		}
		kTime := time.Since(kStart)
		
		kPartition, err := parseSCAROutput(kOutputDir, fmt.Sprintf("k%d_communities", k))
		if err != nil {
			return nil, fmt.Errorf("failed to parse k=%d output: %w", k, err)
		}
		
		kResults[k] = kResult
		kPartitions[k] = kPartition
		kExecutionTime[k] = kTime.Milliseconds()
		
		if config.Verbose {
			fmt.Printf("  k=%d completed in %v: %d levels, max level %d\n", 
				k, kTime, len(kPartition.Levels), kPartition.MaxLevel)
		}
	}
	
	// Step 3: Calculate metrics for each k value against baseline
	if config.Verbose {
		fmt.Println("\nStep 3: Calculating metrics for each k value...")
	}
	
	kNMIScores := make(map[int64]map[int]float64)
	kARIScores := make(map[int64]map[int]float64)
	kVIScores := make(map[int64]map[int]float64)
	kHierarchicalF1Scores := make(map[int64]map[int]float64)
	kAvgNMI := make(map[int64]float64)
	kAvgARI := make(map[int64]float64)
	kAvgVI := make(map[int64]float64)
	kAvgHF1 := make(map[int64]float64)
	kTreeEditDistance := make(map[int64]float64)
	kHierarchyDepthDiff := make(map[int64]int)
	
	for _, k := range kValues {
		kPartition := kPartitions[k]
		
		// Calculate all metrics against baseline
		nmiScores, err := calculateHierarchicalNMIWithMissingLevels(baselinePartition, kPartition)
		if err != nil {
			return nil, fmt.Errorf("NMI calculation failed for k=%d: %w", k, err)
		}
		
		ariScores, err := calculateHierarchicalARIWithMissingLevels(baselinePartition, kPartition)
		if err != nil {
			return nil, fmt.Errorf("ARI calculation failed for k=%d: %w", k, err)
		}
		
		viScores, err := calculateHierarchicalVIWithMissingLevels(baselinePartition, kPartition)
		if err != nil {
			return nil, fmt.Errorf("VI calculation failed for k=%d: %w", k, err)
		}
		
		hf1Scores, err := calculateHierarchicalF1WithMissingLevels(baselinePartition, kPartition)
		if err != nil {
			return nil, fmt.Errorf("HF1 calculation failed for k=%d: %w", k, err)
		}
		
		// Calculate structural metrics
		treeEditDist := calculateTreeEditDistance(baselinePartition, kPartition)
		hierarchyDepthDiff := abs(baselinePartition.MaxLevel - kPartition.MaxLevel)
		
		// Store level-wise metrics
		kNMIScores[k] = nmiScores
		kARIScores[k] = ariScores
		kVIScores[k] = viScores
		kHierarchicalF1Scores[k] = hf1Scores
		kTreeEditDistance[k] = treeEditDist
		kHierarchyDepthDiff[k] = hierarchyDepthDiff
		
		// Calculate averages
		if len(nmiScores) > 0 {
			totalNMI, totalARI, totalVI, totalHF1 := 0.0, 0.0, 0.0, 0.0
			for level := range nmiScores {
				totalNMI += nmiScores[level]
				totalARI += ariScores[level]
				totalVI += viScores[level]
				totalHF1 += hf1Scores[level]
			}
			numLevels := float64(len(nmiScores))
			kAvgNMI[k] = totalNMI / numLevels
			kAvgARI[k] = totalARI / numLevels
			kAvgVI[k] = totalVI / numLevels
			kAvgHF1[k] = totalHF1 / numLevels
		}
		
		if config.Verbose {
			fmt.Printf("  k=%d: AvgNMI=%.4f, AvgARI=%.4f, AvgVI=%.4f, AvgHF1=%.4f, TreeEdit=%.4f\n",
				k, kAvgNMI[k], kAvgARI[k], kAvgVI[k], kAvgHF1[k], treeEditDist)
		}
	}
	
	// Step 4: Generate outputs
	if config.Verbose {
		fmt.Println("\nStep 4: Generating multi-k comparison outputs...")
	}
	
	totalTime := time.Since(startTime)
	
	// Count total nodes
	totalNodes := 0
	if level0, exists := baselinePartition.Levels[0]; exists {
		for _, nodes := range level0 {
			totalNodes += len(nodes)
		}
	}
	
	result := &MultiKComparisonResult{
		HighPrecisionResult:   baselineResult,
		KResults:             kResults,
		KPartitions:          kPartitions,
		KNMIScores:           kNMIScores,
		KARIScores:           kARIScores,
		KVIScores:            kVIScores,
		KHierarchicalF1Scores: kHierarchicalF1Scores,
		KAvgNMI:              kAvgNMI,
		KAvgARI:              kAvgARI,
		KAvgVI:               kAvgVI,
		KAvgHF1:              kAvgHF1,
		KTreeEditDistance:    kTreeEditDistance,
		KHierarchyDepthDiff:  kHierarchyDepthDiff,
		KExecutionTime:       kExecutionTime,
		TestedKValues:        kValues,
		HighPrecisionPartition: baselinePartition,
		NumNodes:             totalNodes,
		TotalComparisonTime:  totalTime.Milliseconds(),
	}
	
	// Write outputs
	if err := writeMultiKComparisonReport(result, config); err != nil {
		return nil, fmt.Errorf("failed to write multi-k comparison report: %w", err)
	}
	
	if err := writeMultiKCSVForPlotting(result, config); err != nil {
		return nil, fmt.Errorf("failed to write multi-k CSV for plotting: %w", err)
	}
	
	if config.Verbose {
		fmt.Println("\n=== Multi-K SCAR Comparison Complete ===")
		fmt.Printf("Total comparison time: %v\n", totalTime)
		fmt.Printf("Tested k values: %v\n", kValues)
		fmt.Printf("Results saved for Jupyter notebook plotting\n")
	}
	
	return result, nil
}

// RunOptimizedSCAR executes the optimized SCAR pipeline
func RunOptimizedSCAR(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*PipelineResult, error) {
	startTime := time.Now()
	
	if config.Verbose {
		fmt.Println("=== Running Optimized SCAR Pipeline ===")
	}
	
	// Step 1: Configure SCAR with input files
	if config.Verbose {
		fmt.Println("Step 1: Configuring optimized SCAR engine...")
	}
	
	// Create a copy of SCAR config and set file paths
	scarConfig := config.OptimizedSCARConfig
	scarConfig.GraphFile = graphFile
	scarConfig.PropertyFile = propertiesFile
	scarConfig.PathFile = pathFile
	scarConfig.Prefix = filepath.Join(config.OutputDir, config.OutputPrefix)
	scarConfig.NumWorkers = 1
	if config.Verbose {
		fmt.Printf("  Graph file: %s\n", graphFile)
		fmt.Printf("  Properties file: %s\n", propertiesFile)
		fmt.Printf("  Path file: %s\n", pathFile)
		fmt.Printf("  Optimized SCAR parameters: k=%d, nk=%d, threshold=%.3f\n", 
			scarConfig.K, scarConfig.NK, scarConfig.Threshold)
		fmt.Printf("  Sketch output: %t\n", scarConfig.SketchOutput)
	}
	
	// Step 2: Create output directory
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Step 3: Run SCAR engine
	if config.Verbose {
		fmt.Println("Step 2: Running optimized SCAR...")
	}
	
	scarStart := time.Now()
	
	engine := scar.NewSketchLouvainEngine(scarConfig)
	err := engine.RunLouvain()
	if err != nil {
		return nil, fmt.Errorf("optimized SCAR failed: %w", err)
	}
	
	scarTime := time.Since(scarStart)
	totalTime := time.Since(startTime)
	
	if config.Verbose {
		fmt.Printf("  Optimized SCAR completed in %v\n", scarTime)
		fmt.Println("Step 3: Writing optimized SCAR summary...")
	}
	
	// Step 4: Write pipeline summary (SCAR writes its own output files)
	if err := writeSCARSummary(&scarConfig, config, totalTime, "optimized"); err != nil {
		return nil, fmt.Errorf("failed to write optimized SCAR summary: %w", err)
	}
	
	// Create final result
	result := &PipelineResult{
		PipelineType:   OptimizedSCAR,
		TotalRuntimeMS: totalTime.Milliseconds(),
		SCARSuccess:    true,
		SCARConfig:     &scarConfig,
	}
	
	if config.Verbose {
		fmt.Println("=== Optimized SCAR Pipeline Complete ===")
		fmt.Printf("Total runtime: %v\n", totalTime)
		fmt.Printf("SCAR execution: %v\n", scarTime)
		if scarConfig.SketchOutput {
			fmt.Println("Generated SCAR hierarchy files for PPRViz integration")
		}
	}
	
	return result, nil
}

// RunSCARComparison executes both SCAR pipelines and compares their results using NMI
func RunSCARComparison(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*ComparisonResult, error) {
	startTime := time.Now()
	
	if config.Verbose {
		fmt.Println("=== Running SCAR vs SCAR Comparison ===")
		fmt.Println("Will run both High-Precision SCAR and Optimized SCAR pipelines and compare results")
	}
	
	// Create separate output directories for each pipeline
	highPrecisionOutputDir := filepath.Join(config.OutputDir, "high_precision_scar")
	optimizedOutputDir := filepath.Join(config.OutputDir, "optimized_scar")
	
	// Step 1: Run High-Precision SCAR pipeline
	if config.Verbose {
		fmt.Println("\nStep 1: Running High-Precision SCAR pipeline...")
	}
	
	highPrecisionConfig := *config // Copy config
	highPrecisionConfig.OutputDir = highPrecisionOutputDir
	highPrecisionConfig.OutputPrefix = "hp_communities"
	
	hpStart := time.Now()
	hpResult, err := RunHighPrecisionSCAR(graphFile, propertiesFile, pathFile, &highPrecisionConfig)
	if err != nil {
		return nil, fmt.Errorf("high-precision SCAR pipeline failed: %w", err)
	}
	hpTime := time.Since(hpStart)
	
	// Step 2: Run Optimized SCAR pipeline  
	if config.Verbose {
		fmt.Println("\nStep 2: Running Optimized SCAR pipeline...")
	}
	
	optimizedConfig := *config // Copy config
	optimizedConfig.OutputDir = optimizedOutputDir
	optimizedConfig.OutputPrefix = "opt_communities"
	
	optStart := time.Now()
	optResult, err := RunOptimizedSCAR(graphFile, propertiesFile, pathFile, &optimizedConfig)
	if err != nil {
		return nil, fmt.Errorf("optimized SCAR pipeline failed: %w", err)
	}
	optTime := time.Since(optStart)
	
	// Step 3: Parse community assignments from both pipelines
	if config.Verbose {
		fmt.Println("\nStep 3: Parsing hierarchical community assignments...")
	}
	
	hpPartition, err := parseSCAROutput(highPrecisionOutputDir, "hp_communities")
	if err != nil {
		return nil, fmt.Errorf("failed to parse high-precision SCAR output: %w", err)
	}
	
	optPartition, err := parseSCAROutput(optimizedOutputDir, "opt_communities")
	if err != nil {
		return nil, fmt.Errorf("failed to parse optimized SCAR output: %w", err)
	}
	
	if config.Verbose {
		fmt.Printf("  High-Precision SCAR: %d levels (max level %d)\n", len(hpPartition.Levels), hpPartition.MaxLevel)
		fmt.Printf("  Optimized SCAR: %d levels (max level %d)\n", len(optPartition.Levels), optPartition.MaxLevel)
		
		// Debug: Show level details
		fmt.Println("  High-Precision SCAR level details:")
		for level := 0; level <= hpPartition.MaxLevel; level++ {
			if levelData, exists := hpPartition.Levels[level]; exists {
				totalNodes := 0
				for _, nodes := range levelData {
					totalNodes += len(nodes)
				}
				fmt.Printf("    Level %d: %d communities, %d nodes\n", level, len(levelData), totalNodes)
			} else {
				fmt.Printf("    Level %d: NOT FOUND\n", level)
			}
		}
		
		fmt.Println("  Optimized SCAR level details:")
		for level := 0; level <= optPartition.MaxLevel; level++ {
			if levelData, exists := optPartition.Levels[level]; exists {
				totalNodes := 0
				for _, nodes := range levelData {
					totalNodes += len(nodes)
				}
				fmt.Printf("    Level %d: %d communities, %d nodes\n", level, len(levelData), totalNodes)
			} else {
				fmt.Printf("    Level %d: NOT FOUND\n", level)
			}
		}
		
		// Count total nodes from level 0
		totalNodes := 0
		if level0, exists := hpPartition.Levels[0]; exists {
			for _, nodes := range level0 {
				totalNodes += len(nodes)
			}
		}
		fmt.Printf("  Total nodes in analysis: %d\n", totalNodes)
	}
	
	// Step 4: Calculate hierarchical clustering metrics
	if config.Verbose {
		fmt.Println("\nStep 4: Calculating hierarchical clustering metrics...")
	}
	
	// Calculate all metrics
	nmiScores, err := calculateHierarchicalNMI(hpPartition, optPartition)
	if err != nil {
		return nil, fmt.Errorf("hierarchical NMI calculation failed: %w", err)
	}
	
	ariScores, err := calculateHierarchicalARI(hpPartition, optPartition)
	if err != nil {
		return nil, fmt.Errorf("hierarchical ARI calculation failed: %w", err)
	}
	
	viScores, err := calculateHierarchicalVI(hpPartition, optPartition)
	if err != nil {
		return nil, fmt.Errorf("hierarchical VI calculation failed: %w", err)
	}
	
	hierarchicalF1Scores, err := calculateHierarchicalF1(hpPartition, optPartition)
	if err != nil {
		return nil, fmt.Errorf("hierarchical F1 calculation failed: %w", err)
	}
	
	// Calculate structural metrics
	treeEditDistance := calculateTreeEditDistance(hpPartition, optPartition)
	hierarchyDepthDiff := abs(hpPartition.MaxLevel - optPartition.MaxLevel)
	communityCountDiff := calculateCommunityCountDifferences(hpPartition, optPartition)
	avgCommunitySizeDiff := calculateAvgCommunitySizeDifferences(hpPartition, optPartition)
	
	// Find common levels
	commonLevels := make([]int, 0, len(nmiScores))
	for level := range nmiScores {
		commonLevels = append(commonLevels, level)
	}
	
	if config.Verbose {
		fmt.Printf("  Calculated metrics for %d common levels\n", len(commonLevels))
		fmt.Printf("  Tree edit distance: %.2f\n", treeEditDistance)
		fmt.Printf("  Hierarchy depth difference: %d\n", hierarchyDepthDiff)
		
		for _, level := range commonLevels {
			if nmi, exists := nmiScores[level]; exists {
				ari := ariScores[level]
				vi := viScores[level]
				hf1 := hierarchicalF1Scores[level]
				
				// Get community counts for this level
				hpComms := 0
				optComms := 0
				hpNodes := 0
				optNodes := 0
				
				if hpLevel, exists := hpPartition.Levels[level]; exists {
					hpComms = len(hpLevel)
					for _, nodes := range hpLevel {
						hpNodes += len(nodes)
					}
				}
				if optLevel, exists := optPartition.Levels[level]; exists {
					optComms = len(optLevel)
					for _, nodes := range optLevel {
						optNodes += len(nodes)
					}
				}
				
				fmt.Printf("    Level %d: NMI=%.4f, ARI=%.4f, VI=%.4f, HF1=%.4f (HP: %d comms/%d nodes, Opt: %d comms/%d nodes)\n", 
					level, nmi, ari, vi, hf1, hpComms, hpNodes, optComms, optNodes)
					
				// Debug: Show detailed info for problematic levels
				if nmi == 0.0 {
					fmt.Printf("      WARNING: NMI = 0 at level %d - investigating...\n", level)
					flat1 := convertPartitionToFlat(hpPartition, level)
					flat2 := convertPartitionToFlat(optPartition, level)
					fmt.Printf("      Flat partition sizes: HP=%d nodes, Opt=%d nodes\n", len(flat1), len(flat2))
					
					// Count common nodes
					commonNodes := 0
					for nodeID := range flat1 {
						if _, exists := flat2[nodeID]; exists {
							commonNodes++
						}
					}
					fmt.Printf("      Common nodes between partitions: %d\n", commonNodes)
				}
			}
		}
	}
	
	// Step 5: Generate comparison report
	if config.Verbose {
		fmt.Println("\nStep 5: Generating SCAR comparison report...")
	}
	
	totalTime := time.Since(startTime)
	
	// Count total nodes from finest level (level 0)
	totalNodes := 0
	if level0, exists := hpPartition.Levels[0]; exists {
		for _, nodes := range level0 {
			totalNodes += len(nodes)
		}
	}
	
	result := &ComparisonResult{
		HighPrecisionResult:    hpResult,
		OptimizedResult:        optResult,
		NMIScores:              nmiScores,
		ARIScores:              ariScores,
		VIScores:               viScores,
		HierarchicalF1Scores:   hierarchicalF1Scores,
		TreeEditDistance:       treeEditDistance,
		HierarchyDepthDiff:     hierarchyDepthDiff,
		HighPrecisionPartition: hpPartition,
		OptimizedPartition:     optPartition,
		NumNodes:               totalNodes,
		HighPrecisionTime:      hpTime.Milliseconds(),
		OptimizedTime:          optTime.Milliseconds(),
		TotalComparisonTime:    totalTime.Milliseconds(),
		CommonLevels:           commonLevels,
		CommunityCountDiff:     communityCountDiff,
		AvgCommunitySizeDiff:   avgCommunitySizeDiff,
	}
	
	if err := writeSCARComparisonReport(result, config); err != nil {
		return nil, fmt.Errorf("failed to write comparison report: %w", err)
	}
	
	if config.Verbose {
		fmt.Println("\n=== SCAR vs SCAR Comparison Complete ===")
		fmt.Printf("Total comparison time: %v\n", totalTime)
		fmt.Printf("High-Precision SCAR time: %v\n", hpTime)
		fmt.Printf("Optimized SCAR time: %v\n", optTime)
		if len(nmiScores) > 0 {
			// Report average metrics and best levels
			totalNMI := 0.0
			totalARI := 0.0
			totalVI := 0.0
			totalHF1 := 0.0
			bestNMI := 0.0
			bestLevel := -1
			
			for level, nmi := range nmiScores {
				totalNMI += nmi
				totalARI += ariScores[level]
				totalVI += viScores[level]
				totalHF1 += hierarchicalF1Scores[level]
				if nmi > bestNMI {
					bestNMI = nmi
					bestLevel = level
				}
			}
			
			numLevels := float64(len(nmiScores))
			avgNMI := totalNMI / numLevels
			avgARI := totalARI / numLevels
			avgVI := totalVI / numLevels
			avgHF1 := totalHF1 / numLevels
			
			fmt.Printf("Average metrics across levels:\n")
			fmt.Printf("  NMI: %.6f, ARI: %.6f, VI: %.6f, HF1: %.6f\n", avgNMI, avgARI, avgVI, avgHF1)
			fmt.Printf("Best NMI: %.6f (level %d)\n", bestNMI, bestLevel)
			fmt.Printf("Tree edit distance: %.2f\n", treeEditDistance)
		}
		fmt.Printf("Speedup: %.2fx\n", float64(hpTime.Milliseconds())/float64(optTime.Milliseconds()))
	}
	
	return result, nil
}

// HierarchicalPartition represents partitions at multiple levels
type HierarchicalPartition struct {
	Levels map[int]map[int][]string // [level][community_id][node_list]
	MaxLevel int
}

// parseSCAROutput parses SCAR output files to extract hierarchical partitions
func parseSCAROutput(outputDir, prefix string) (*HierarchicalPartition, error) {
	mappingFile := filepath.Join(outputDir, prefix+"_mapping.dat")
	return parseSCARMappingFile(mappingFile)
}

// parseSCARMappingFile parses SCAR's _mapping.dat file format
func parseSCARMappingFile(filePath string) (*HierarchicalPartition, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open SCAR mapping file %s: %w", filePath, err)
	}
	defer file.Close()
	
	partition := &HierarchicalPartition{
		Levels: make(map[int]map[int][]string),
		MaxLevel: 0,
	}
	
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		
		// Check if this is a community header (c0_l{level}_{community_id})
		if strings.HasPrefix(line, "c0_l") {
			level, communityID, err := parseCommunityID(line)
			if err != nil {
				continue // Skip malformed community IDs
			}
			
			// Update max level
			if level > partition.MaxLevel {
				partition.MaxLevel = level
			}
			
			// Initialize level if needed
			if partition.Levels[level] == nil {
				partition.Levels[level] = make(map[int][]string)
			}
			
			// Read node count
			if !scanner.Scan() {
				break
			}
			countLine := strings.TrimSpace(scanner.Text())
			nodeCount, err := strconv.Atoi(countLine)
			if err != nil {
				continue // Skip if count is invalid
			}
			
			// Read nodes for this community
			nodes := make([]string, 0, nodeCount)
			for i := 0; i < nodeCount && scanner.Scan(); i++ {
				nodeID := strings.TrimSpace(scanner.Text())
				if nodeID != "" {
					nodes = append(nodes, nodeID)
				}
			}
			
			partition.Levels[level][communityID] = nodes
		}
	}
	
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading SCAR mapping file: %w", err)
	}
	
	return partition, nil
}

// parseCommunityID extracts level and community ID from strings like "c0_l2_5"
func parseCommunityID(communityStr string) (level int, communityID int, err error) {
	// Expected format: c0_l{level}_{community_id}
	if !strings.HasPrefix(communityStr, "c0_l") {
		return 0, 0, fmt.Errorf("invalid community ID format: %s", communityStr)
	}
	
	// Remove "c0_l" prefix
	remaining := communityStr[4:]
	
	// Find the last underscore to separate level from community ID
	lastUnderscore := strings.LastIndex(remaining, "_")
	if lastUnderscore == -1 {
		return 0, 0, fmt.Errorf("invalid community ID format: %s", communityStr)
	}
	
	levelStr := remaining[:lastUnderscore]
	communityIDStr := remaining[lastUnderscore+1:]
	
	level, err = strconv.Atoi(levelStr)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid level in community ID %s: %w", communityStr, err)
	}
	
	communityID, err = strconv.Atoi(communityIDStr)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid community ID in %s: %w", communityStr, err)
	}
	
	return level, communityID, nil
}

// convertPartitionToFlat converts hierarchical partition at specific level to flat partition
func convertPartitionToFlat(partition *HierarchicalPartition, level int) map[string]int {
	flatPartition := make(map[string]int)
	
	if levelData, exists := partition.Levels[level]; exists {
		for communityID, nodes := range levelData {
			for _, nodeID := range nodes {
				flatPartition[nodeID] = communityID
			}
		}
	}
	
	return flatPartition
}

// calculateHierarchicalNMI calculates NMI for each level between two hierarchical partitions
func calculateHierarchicalNMI(partition1, partition2 *HierarchicalPartition) (map[int]float64, error) {
	nmiScores := make(map[int]float64)
	
	// Find common levels
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		// Check if both partitions have this level
		if _, exists1 := partition1.Levels[level]; !exists1 {
			continue
		}
		if _, exists2 := partition2.Levels[level]; !exists2 {
			continue
		}
		
		// Convert to flat partitions for NMI calculation
		flat1 := convertPartitionToFlat(partition1, level)
		flat2 := convertPartitionToFlat(partition2, level)
		
		if len(flat1) == 0 || len(flat2) == 0 {
			continue
		}
		
		nmi, err := calculateNMI(flat1, flat2)
		if err != nil {
			return nil, fmt.Errorf("NMI calculation failed for level %d: %w", level, err)
		}
		
		nmiScores[level] = nmi
	}
	
	return nmiScores, nil
}

// calculateNMI computes the Normalized Mutual Information between two partitions
func calculateNMI(partition1, partition2 map[string]int) (float64, error) {
	// Find common nodes
	commonNodes := make([]string, 0)
	for nodeID := range partition1 {
		if _, exists := partition2[nodeID]; exists {
			commonNodes = append(commonNodes, nodeID)
		}
	}
	
	if len(commonNodes) == 0 {
		return 0.0, fmt.Errorf("no common nodes between partitions (p1: %d nodes, p2: %d nodes)", 
			len(partition1), len(partition2))
	}
	
	n := float64(len(commonNodes))
	
	// Build contingency table
	contingencyTable := make(map[int]map[int]int)
	comm1Counts := make(map[int]int)
	comm2Counts := make(map[int]int)
	
	for _, nodeID := range commonNodes {
		c1 := partition1[nodeID]
		c2 := partition2[nodeID]
		
		if contingencyTable[c1] == nil {
			contingencyTable[c1] = make(map[int]int)
		}
		contingencyTable[c1][c2]++
		comm1Counts[c1]++
		comm2Counts[c2]++
	}
	
	// Debug info for troubleshooting
	if len(comm1Counts) == 1 && len(comm2Counts) == 1 {
		// Both partitions have only one community
		return 1.0, nil
	}
	
	// Calculate mutual information
	mutualInfo := 0.0
	for c1, row := range contingencyTable {
		for c2, count := range row {
			if count > 0 {
				pij := float64(count) / n
				pi := float64(comm1Counts[c1]) / n
				pj := float64(comm2Counts[c2]) / n
				if pij > 0 && pi > 0 && pj > 0 {
					mutualInfo += pij * math.Log2(pij/(pi*pj))
				}
			}
		}
	}
	
	// Calculate entropies
	entropy1 := 0.0
	for _, count := range comm1Counts {
		if count > 0 {
			p := float64(count) / n
			entropy1 -= p * math.Log2(p)
		}
	}
	
	entropy2 := 0.0
	for _, count := range comm2Counts {
		if count > 0 {
			p := float64(count) / n
			entropy2 -= p * math.Log2(p)
		}
	}
	
	// Calculate NMI
	if entropy1 == 0 && entropy2 == 0 {
		return 1.0, nil // Both partitions have single community
	}
	if entropy1 == 0 || entropy2 == 0 {
		return 0.0, nil // One partition has single community, other doesn't
	}
	
	nmi := 2.0 * mutualInfo / (entropy1 + entropy2)
	
	// Clamp to [0,1] range due to potential floating point errors
	if nmi < 0 {
		nmi = 0
	}
	if nmi > 1 {
		nmi = 1
	}
	
	return nmi, nil
}

// calculateARI computes the Adjusted Rand Index between two partitions
func calculateARI(partition1, partition2 map[string]int) (float64, error) {
	// Find common nodes
	commonNodes := make([]string, 0)
	for nodeID := range partition1 {
		if _, exists := partition2[nodeID]; exists {
			commonNodes = append(commonNodes, nodeID)
		}
	}
	
	if len(commonNodes) == 0 {
		return 0.0, fmt.Errorf("no common nodes between partitions")
	}
	
	n := len(commonNodes)
	
	// Build contingency table
	contingencyTable := make(map[int]map[int]int)
	comm1Counts := make(map[int]int)
	comm2Counts := make(map[int]int)
	
	for _, nodeID := range commonNodes {
		c1 := partition1[nodeID]
		c2 := partition2[nodeID]
		
		if contingencyTable[c1] == nil {
			contingencyTable[c1] = make(map[int]int)
		}
		contingencyTable[c1][c2]++
		comm1Counts[c1]++
		comm2Counts[c2]++
	}
	
	// Calculate sum of combinations
	sumCombinationsNij := 0.0
	for _, row := range contingencyTable {
		for _, count := range row {
			if count >= 2 {
				sumCombinationsNij += float64(count * (count - 1)) / 2.0
			}
		}
	}
	
	sumCombinationsAi := 0.0
	for _, count := range comm1Counts {
		if count >= 2 {
			sumCombinationsAi += float64(count * (count - 1)) / 2.0
		}
	}
	
	sumCombinationsBj := 0.0
	for _, count := range comm2Counts {
		if count >= 2 {
			sumCombinationsBj += float64(count * (count - 1)) / 2.0
		}
	}
	
	totalCombinations := float64(n * (n - 1)) / 2.0
	expectedIndex := (sumCombinationsAi * sumCombinationsBj) / totalCombinations
	maxIndex := (sumCombinationsAi + sumCombinationsBj) / 2.0
	
	if maxIndex == expectedIndex {
		return 1.0, nil
	}
	
	ari := (sumCombinationsNij - expectedIndex) / (maxIndex - expectedIndex)
	
	// Clamp to [-1,1] range
	if ari < -1 {
		ari = -1
	}
	if ari > 1 {
		ari = 1
	}
	
	return ari, nil
}

// calculateVI computes the Variation of Information between two partitions
func calculateVI(partition1, partition2 map[string]int) (float64, error) {
	// Find common nodes
	commonNodes := make([]string, 0)
	for nodeID := range partition1 {
		if _, exists := partition2[nodeID]; exists {
			commonNodes = append(commonNodes, nodeID)
		}
	}
	
	if len(commonNodes) == 0 {
		return 0.0, fmt.Errorf("no common nodes between partitions")
	}
	
	n := float64(len(commonNodes))
	
	// Build contingency table
	contingencyTable := make(map[int]map[int]int)
	comm1Counts := make(map[int]int)
	comm2Counts := make(map[int]int)
	
	for _, nodeID := range commonNodes {
		c1 := partition1[nodeID]
		c2 := partition2[nodeID]
		
		if contingencyTable[c1] == nil {
			contingencyTable[c1] = make(map[int]int)
		}
		contingencyTable[c1][c2]++
		comm1Counts[c1]++
		comm2Counts[c2]++
	}
	
	// Calculate mutual information
	mutualInfo := 0.0
	for c1, row := range contingencyTable {
		for c2, count := range row {
			if count > 0 {
				pij := float64(count) / n
				pi := float64(comm1Counts[c1]) / n
				pj := float64(comm2Counts[c2]) / n
				if pij > 0 && pi > 0 && pj > 0 {
					mutualInfo += pij * math.Log2(pij/(pi*pj))
				}
			}
		}
	}
	
	// Calculate entropies
	entropy1 := 0.0
	for _, count := range comm1Counts {
		if count > 0 {
			p := float64(count) / n
			entropy1 -= p * math.Log2(p)
		}
	}
	
	entropy2 := 0.0
	for _, count := range comm2Counts {
		if count > 0 {
			p := float64(count) / n
			entropy2 -= p * math.Log2(p)
		}
	}
	
	// VI = H(X) + H(Y) - 2*I(X,Y)
	vi := entropy1 + entropy2 - 2*mutualInfo
	
	// VI should be non-negative
	if vi < 0 {
		vi = 0
	}
	
	return vi, nil
}

// calculateHierarchicalF1 computes hierarchical F1 score based on community membership overlap
func calculateHierarchicalF1(partition1, partition2 *HierarchicalPartition) (map[int]float64, error) {
	f1Scores := make(map[int]float64)
	
	// Find common levels
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		// Check if both partitions have this level
		if _, exists1 := partition1.Levels[level]; !exists1 {
			continue
		}
		if _, exists2 := partition2.Levels[level]; !exists2 {
			continue
		}
		
		f1, err := calculateF1AtLevel(partition1, partition2, level)
		if err != nil {
			return nil, fmt.Errorf("F1 calculation failed for level %d: %w", level, err)
		}
		
		f1Scores[level] = f1
	}
	
	return f1Scores, nil
}

// calculateF1AtLevel computes F1 score at a specific level
func calculateF1AtLevel(partition1, partition2 *HierarchicalPartition, level int) (float64, error) {
	level1 := partition1.Levels[level]
	level2 := partition2.Levels[level]
	
	if len(level1) == 0 || len(level2) == 0 {
		return 0.0, nil
	}
	
	// Convert to sets for easier computation
	comm1Sets := make([]map[string]bool, 0, len(level1))
	for _, nodes := range level1 {
		nodeSet := make(map[string]bool)
		for _, node := range nodes {
			nodeSet[node] = true
		}
		comm1Sets = append(comm1Sets, nodeSet)
	}
	
	comm2Sets := make([]map[string]bool, 0, len(level2))
	for _, nodes := range level2 {
		nodeSet := make(map[string]bool)
		for _, node := range nodes {
			nodeSet[node] = true
		}
		comm2Sets = append(comm2Sets, nodeSet)
	}
	
	// Calculate best F1 for each community in partition1
	totalF1 := 0.0
	totalWeight := 0.0
	
	for _, comm1 := range comm1Sets {
		bestF1 := 0.0
		
		for _, comm2 := range comm2Sets {
			// Calculate intersection
			intersection := 0
			for node := range comm1 {
				if comm2[node] {
					intersection++
				}
			}
			
			if intersection == 0 {
				continue
			}
			
			precision := float64(intersection) / float64(len(comm2))
			recall := float64(intersection) / float64(len(comm1))
			
			if precision+recall > 0 {
				f1 := 2 * precision * recall / (precision + recall)
				if f1 > bestF1 {
					bestF1 = f1
				}
			}
		}
		
		weight := float64(len(comm1))
		totalF1 += bestF1 * weight
		totalWeight += weight
	}
	
	if totalWeight == 0 {
		return 0.0, nil
	}
	
	return totalF1 / totalWeight, nil
}

// calculateTreeEditDistance computes a simplified tree edit distance between hierarchies
func calculateTreeEditDistance(partition1, partition2 *HierarchicalPartition) float64 {
	// Simple metric: normalized difference in number of communities at each level
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	if maxLevel == 0 {
		return 0.0
	}
	
	totalDiff := 0.0
	totalLevels := 0.0
	
	for level := 0; level <= maxLevel; level++ {
		count1 := 0
		count2 := 0
		
		if level1, exists := partition1.Levels[level]; exists {
			count1 = len(level1)
		}
		if level2, exists := partition2.Levels[level]; exists {
			count2 = len(level2)
		}
		
		// Skip levels where both partitions have no communities
		if count1 == 0 && count2 == 0 {
			continue
		}
		
		// Normalized difference
		maxCount := max(count1, count2)
		if maxCount > 0 {
			diff := float64(abs(count1-count2)) / float64(maxCount)
			totalDiff += diff
			totalLevels++
		}
	}
	
	if totalLevels == 0 {
		return 0.0
	}
	
	return totalDiff / totalLevels
}

// Helper functions for structural metrics
func calculateCommunityCountDifferences(partition1, partition2 *HierarchicalPartition) map[int]int {
	diffs := make(map[int]int)
	
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		count1 := 0
		count2 := 0
		
		if level1, exists := partition1.Levels[level]; exists {
			count1 = len(level1)
		}
		if level2, exists := partition2.Levels[level]; exists {
			count2 = len(level2)
		}
		
		diffs[level] = count1 - count2
	}
	
	return diffs
}

func calculateAvgCommunitySizeDifferences(partition1, partition2 *HierarchicalPartition) map[int]float64 {
	diffs := make(map[int]float64)
	
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		avg1 := 0.0
		avg2 := 0.0
		
		if level1, exists := partition1.Levels[level]; exists && len(level1) > 0 {
			totalNodes := 0
			for _, nodes := range level1 {
				totalNodes += len(nodes)
			}
			avg1 = float64(totalNodes) / float64(len(level1))
		}
		
		if level2, exists := partition2.Levels[level]; exists && len(level2) > 0 {
			totalNodes := 0
			for _, nodes := range level2 {
				totalNodes += len(nodes)
			}
			avg2 = float64(totalNodes) / float64(len(level2))
		}
		
		diffs[level] = avg1 - avg2
	}
	
	return diffs
}

// calculateHierarchicalARI calculates ARI for each level between two hierarchical partitions
func calculateHierarchicalARI(partition1, partition2 *HierarchicalPartition) (map[int]float64, error) {
	ariScores := make(map[int]float64)
	
	// Find common levels
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		// Check if both partitions have this level
		if _, exists1 := partition1.Levels[level]; !exists1 {
			continue
		}
		if _, exists2 := partition2.Levels[level]; !exists2 {
			continue
		}
		
		// Convert to flat partitions for ARI calculation
		flat1 := convertPartitionToFlat(partition1, level)
		flat2 := convertPartitionToFlat(partition2, level)
		
		if len(flat1) == 0 || len(flat2) == 0 {
			continue
		}
		
		ari, err := calculateARI(flat1, flat2)
		if err != nil {
			return nil, fmt.Errorf("ARI calculation failed for level %d: %w", level, err)
		}
		
		ariScores[level] = ari
	}
	
	return ariScores, nil
}

// calculateHierarchicalVI calculates VI for each level between two hierarchical partitions
func calculateHierarchicalVI(partition1, partition2 *HierarchicalPartition) (map[int]float64, error) {
	viScores := make(map[int]float64)
	
	// Find common levels
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		// Check if both partitions have this level
		if _, exists1 := partition1.Levels[level]; !exists1 {
			continue
		}
		if _, exists2 := partition2.Levels[level]; !exists2 {
			continue
		}
		
		// Convert to flat partitions for VI calculation
		flat1 := convertPartitionToFlat(partition1, level)
		flat2 := convertPartitionToFlat(partition2, level)
		
		if len(flat1) == 0 || len(flat2) == 0 {
			continue
		}
		
		vi, err := calculateVI(flat1, flat2)
		if err != nil {
			return nil, fmt.Errorf("VI calculation failed for level %d: %w", level, err)
		}
		
		viScores[level] = vi
	}
	
	return viScores, nil
}

// calculateHierarchicalNMIWithMissingLevels calculates NMI with missing levels assumed as 0 similarity
func calculateHierarchicalNMIWithMissingLevels(partition1, partition2 *HierarchicalPartition) (map[int]float64, error) {
	nmiScores := make(map[int]float64)
	
	// Find maximum level across both partitions
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		level1Exists := false
		level2Exists := false
		
		if _, exists := partition1.Levels[level]; exists {
			level1Exists = true
		}
		if _, exists := partition2.Levels[level]; exists {
			level2Exists = true
		}
		
		// If either level is missing, assume 0 similarity
		if !level1Exists || !level2Exists {
			nmiScores[level] = 0.0
			continue
		}
		
		// Convert to flat partitions for NMI calculation
		flat1 := convertPartitionToFlat(partition1, level)
		flat2 := convertPartitionToFlat(partition2, level)
		
		if len(flat1) == 0 || len(flat2) == 0 {
			nmiScores[level] = 0.0
			continue
		}
		
		nmi, err := calculateNMI(flat1, flat2)
		if err != nil {
			return nil, fmt.Errorf("NMI calculation failed for level %d: %w", level, err)
		}
		
		nmiScores[level] = nmi
	}
	
	return nmiScores, nil
}

// calculateHierarchicalARIWithMissingLevels calculates ARI with missing levels assumed as 0 similarity
func calculateHierarchicalARIWithMissingLevels(partition1, partition2 *HierarchicalPartition) (map[int]float64, error) {
	ariScores := make(map[int]float64)
	
	// Find maximum level across both partitions
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		level1Exists := false
		level2Exists := false
		
		if _, exists := partition1.Levels[level]; exists {
			level1Exists = true
		}
		if _, exists := partition2.Levels[level]; exists {
			level2Exists = true
		}
		
		// If either level is missing, assume 0 similarity
		if !level1Exists || !level2Exists {
			ariScores[level] = 0.0
			continue
		}
		
		// Convert to flat partitions for ARI calculation
		flat1 := convertPartitionToFlat(partition1, level)
		flat2 := convertPartitionToFlat(partition2, level)
		
		if len(flat1) == 0 || len(flat2) == 0 {
			ariScores[level] = 0.0
			continue
		}
		
		ari, err := calculateARI(flat1, flat2)
		if err != nil {
			return nil, fmt.Errorf("ARI calculation failed for level %d: %w", level, err)
		}
		
		ariScores[level] = ari
	}
	
	return ariScores, nil
}

// calculateHierarchicalVIWithMissingLevels calculates VI with missing levels assumed as maximum VI
func calculateHierarchicalVIWithMissingLevels(partition1, partition2 *HierarchicalPartition) (map[int]float64, error) {
	viScores := make(map[int]float64)
	
	// Find maximum level across both partitions
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		level1Exists := false
		level2Exists := false
		
		if _, exists := partition1.Levels[level]; exists {
			level1Exists = true
		}
		if _, exists := partition2.Levels[level]; exists {
			level2Exists = true
		}
		
		// If either level is missing, assume maximum VI (worst case)
		if !level1Exists || !level2Exists {
			viScores[level] = 4.0 // Reasonable upper bound for VI
			continue
		}
		
		// Convert to flat partitions for VI calculation
		flat1 := convertPartitionToFlat(partition1, level)
		flat2 := convertPartitionToFlat(partition2, level)
		
		if len(flat1) == 0 || len(flat2) == 0 {
			viScores[level] = 4.0
			continue
		}
		
		vi, err := calculateVI(flat1, flat2)
		if err != nil {
			return nil, fmt.Errorf("VI calculation failed for level %d: %w", level, err)
		}
		
		viScores[level] = vi
	}
	
	return viScores, nil
}

// calculateHierarchicalF1WithMissingLevels calculates HF1 with missing levels assumed as 0 similarity
func calculateHierarchicalF1WithMissingLevels(partition1, partition2 *HierarchicalPartition) (map[int]float64, error) {
	f1Scores := make(map[int]float64)
	
	// Find maximum level across both partitions
	maxLevel := partition1.MaxLevel
	if partition2.MaxLevel > maxLevel {
		maxLevel = partition2.MaxLevel
	}
	
	for level := 0; level <= maxLevel; level++ {
		level1Exists := false
		level2Exists := false
		
		if _, exists := partition1.Levels[level]; exists {
			level1Exists = true
		}
		if _, exists := partition2.Levels[level]; exists {
			level2Exists = true
		}
		
		// If either level is missing, assume 0 similarity
		if !level1Exists || !level2Exists {
			f1Scores[level] = 0.0
			continue
		}
		
		f1, err := calculateF1AtLevel(partition1, partition2, level)
		if err != nil {
			return nil, fmt.Errorf("F1 calculation failed for level %d: %w", level, err)
		}
		
		f1Scores[level] = f1
	}
	
	return f1Scores, nil
}

// abs returns the absolute value of an integer
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// writeMultiKComparisonReport generates a comprehensive multi-k comparison report
func writeMultiKComparisonReport(result *MultiKComparisonResult, config *PipelineConfig) error {
	reportPath := filepath.Join(config.OutputDir, "multi_k_comparison_report.txt")
	
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	file, err := os.Create(reportPath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "=== Multi-K SCAR Comparison Report ===\n\n")
	
	// Overall summary
	fmt.Fprintf(file, "Comparison Summary:\n")
	fmt.Fprintf(file, "  Number of Nodes: %d\n", result.NumNodes)
	fmt.Fprintf(file, "  Tested K Values: %v\n", result.TestedKValues)
	fmt.Fprintf(file, "  Baseline: High-precision SCAR (k=512)\n")
	fmt.Fprintf(file, "  Total Comparison Time: %d ms\n", result.TotalComparisonTime)
	fmt.Fprintf(file, "\n")
	
	// Performance summary table
	fmt.Fprintf(file, "Performance Summary:\n")
	fmt.Fprintf(file, "  K Value | Execution Time (ms) | Speedup vs k=512\n")
	fmt.Fprintf(file, "  --------|--------------------|-----------------\n")
	
	baselineTime := result.KExecutionTime[512]
	for _, k := range result.TestedKValues {
		execTime := result.KExecutionTime[k]
		speedup := float64(baselineTime) / float64(execTime)
		fmt.Fprintf(file, "  %-7d | %-18d | %.2fx\n", k, execTime, speedup)
	}
	fmt.Fprintf(file, "\n")
	
	// Quality metrics summary table
	fmt.Fprintf(file, "Quality Metrics Summary:\n")
	fmt.Fprintf(file, "  K Value | Avg NMI | Avg ARI | Avg VI  | Avg HF1 | Tree Edit | Depth Diff\n")
	fmt.Fprintf(file, "  --------|---------|---------|---------|---------|-----------|------------\n")
	
	for _, k := range result.TestedKValues {
		avgNMI := result.KAvgNMI[k]
		avgARI := result.KAvgARI[k]
		avgVI := result.KAvgVI[k]
		avgHF1 := result.KAvgHF1[k]
		treeEdit := result.KTreeEditDistance[k]
		depthDiff := result.KHierarchyDepthDiff[k]
		
		fmt.Fprintf(file, "  %-7d | %-7.4f | %-7.4f | %-7.4f | %-7.4f | %-9.4f | %-10d\n",
			k, avgNMI, avgARI, avgVI, avgHF1, treeEdit, depthDiff)
	}
	fmt.Fprintf(file, "\n")
	
	// Level-by-level analysis for each k
	fmt.Fprintf(file, "Level-by-Level Analysis:\n")
	
	// Find maximum level across all partitions
	maxLevel := result.HighPrecisionPartition.MaxLevel
	for _, partition := range result.KPartitions {
		if partition.MaxLevel > maxLevel {
			maxLevel = partition.MaxLevel
		}
	}
	
	for level := 0; level <= min(maxLevel, 5); level++ { // Limit to first 6 levels for readability
		fmt.Fprintf(file, "\n  Level %d:\n", level)
		fmt.Fprintf(file, "    K Value | NMI    | ARI    | VI     | HF1    | Communities\n")
		fmt.Fprintf(file, "    --------|--------|--------|--------|--------|-----------\n")
		
		for _, k := range result.TestedKValues {
			nmi := 0.0
			ari := 0.0
			vi := 0.0
			hf1 := 0.0
			communities := 0
			
			if nmiMap, exists := result.KNMIScores[k]; exists {
				if val, levelExists := nmiMap[level]; levelExists {
					nmi = val
				}
			}
			if ariMap, exists := result.KARIScores[k]; exists {
				if val, levelExists := ariMap[level]; levelExists {
					ari = val
				}
			}
			if viMap, exists := result.KVIScores[k]; exists {
				if val, levelExists := viMap[level]; levelExists {
					vi = val
				}
			}
			if hf1Map, exists := result.KHierarchicalF1Scores[k]; exists {
				if val, levelExists := hf1Map[level]; levelExists {
					hf1 = val
				}
			}
			
			// Count communities at this level
			if partition, exists := result.KPartitions[k]; exists {
				if levelData, levelExists := partition.Levels[level]; levelExists {
					communities = len(levelData)
				}
			}
			
			fmt.Fprintf(file, "    %-7d | %-6.4f | %-6.4f | %-6.4f | %-6.4f | %-10d\n",
				k, nmi, ari, vi, hf1, communities)
		}
	}
	
	// Analysis and recommendations
	fmt.Fprintf(file, "\nAnalysis and Recommendations:\n")
	
	// Find best trade-off point
	bestK := int64(1)
	bestScore := 0.0
	
	for _, k := range result.TestedKValues {
		if k == 512 {
			continue // Skip baseline
		}
		
		avgNMI := result.KAvgNMI[k]
		speedup := float64(result.KExecutionTime[512]) / float64(result.KExecutionTime[k])
		
		// Simple scoring: balance quality and speed
		score := avgNMI * math.Log2(speedup+1) // Logarithmic speedup reward
		
		if score > bestScore {
			bestScore = score
			bestK = k
		}
	}
	
	fmt.Fprintf(file, "  Recommended K Value: %d\n", bestK)
	fmt.Fprintf(file, "    Average NMI: %.4f\n", result.KAvgNMI[bestK])
	fmt.Fprintf(file, "    Average ARI: %.4f\n", result.KAvgARI[bestK])
	fmt.Fprintf(file, "    Speedup: %.2fx\n", float64(result.KExecutionTime[512])/float64(result.KExecutionTime[bestK]))
	
	// Quality vs speed analysis
	fmt.Fprintf(file, "\n  Quality vs Speed Analysis:\n")
	for _, k := range result.TestedKValues {
		if k == 512 {
			continue
		}
		
		avgNMI := result.KAvgNMI[k]
		speedup := float64(result.KExecutionTime[512]) / float64(result.KExecutionTime[k])
		qualityLoss := result.KAvgNMI[512] - avgNMI
		
		fmt.Fprintf(file, "    k=%d: %.1fx speedup, %.4f quality loss (%.1f%% degradation)\n",
			k, speedup, qualityLoss, qualityLoss*100/result.KAvgNMI[512])
	}
	
	return nil
}

// writeMultiKCSVForPlotting writes CSV data for Jupyter notebook plotting
func writeMultiKCSVForPlotting(result *MultiKComparisonResult, config *PipelineConfig) error {
	csvPath := filepath.Join(config.OutputDir, "multi_k_comparison_data.csv")
	
	file, err := os.Create(csvPath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Write header
	fmt.Fprintf(file, "k_value,execution_time_ms,speedup,avg_nmi,avg_ari,avg_vi,avg_hf1,tree_edit_distance,hierarchy_depth_diff")
	
	// Add level-specific columns for first few levels
	maxDisplayLevel := 5
	for level := 0; level <= maxDisplayLevel; level++ {
		fmt.Fprintf(file, ",nmi_level_%d,ari_level_%d,vi_level_%d,hf1_level_%d,communities_level_%d",
			level, level, level, level, level)
	}
	fmt.Fprintf(file, "\n")
	
	// Write data for each k value
	baselineTime := result.KExecutionTime[512]
	
	for _, k := range result.TestedKValues {
		execTime := result.KExecutionTime[k]
		speedup := float64(baselineTime) / float64(execTime)
		avgNMI := result.KAvgNMI[k]
		avgARI := result.KAvgARI[k]
		avgVI := result.KAvgVI[k]
		avgHF1 := result.KAvgHF1[k]
		treeEdit := result.KTreeEditDistance[k]
		depthDiff := result.KHierarchyDepthDiff[k]
		
		fmt.Fprintf(file, "%d,%d,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%d",
			k, execTime, speedup, avgNMI, avgARI, avgVI, avgHF1, treeEdit, depthDiff)
		
		// Add level-specific data
		for level := 0; level <= maxDisplayLevel; level++ {
			nmi := 0.0
			ari := 0.0
			vi := 0.0
			hf1 := 0.0
			communities := 0
			
			if nmiMap, exists := result.KNMIScores[k]; exists {
				if val, levelExists := nmiMap[level]; levelExists {
					nmi = val
				}
			}
			if ariMap, exists := result.KARIScores[k]; exists {
				if val, levelExists := ariMap[level]; levelExists {
					ari = val
				}
			}
			if viMap, exists := result.KVIScores[k]; exists {
				if val, levelExists := viMap[level]; levelExists {
					vi = val
				}
			}
			if hf1Map, exists := result.KHierarchicalF1Scores[k]; exists {
				if val, levelExists := hf1Map[level]; levelExists {
					hf1 = val
				}
			}
			
			// Count communities at this level
			if partition, exists := result.KPartitions[k]; exists {
				if levelData, levelExists := partition.Levels[level]; levelExists {
					communities = len(levelData)
				}
			}
			
			fmt.Fprintf(file, ",%.6f,%.6f,%.6f,%.6f,%d", nmi, ari, vi, hf1, communities)
		}
		
		fmt.Fprintf(file, "\n")
	}
	
	return nil
}

// writeSCARComparisonReport generates a comprehensive SCAR comparison report
func writeSCARComparisonReport(result *ComparisonResult, config *PipelineConfig) error {
	reportPath := filepath.Join(config.OutputDir, "scar_comparison_report.txt")
	
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	file, err := os.Create(reportPath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "=== SCAR vs SCAR Comparison Report ===\n\n")
	
	// Overall comparison
	fmt.Fprintf(file, "Comparison Summary:\n")
	fmt.Fprintf(file, "  Number of Nodes: %d\n", result.NumNodes)
	fmt.Fprintf(file, "  Common Levels: %d\n", len(result.CommonLevels))
	fmt.Fprintf(file, "  Total Comparison Time: %d ms\n", result.TotalComparisonTime)
	fmt.Fprintf(file, "\n")
	
	// Performance comparison
	fmt.Fprintf(file, "Performance Comparison:\n")
	fmt.Fprintf(file, "  High-Precision SCAR Time: %d ms\n", result.HighPrecisionTime)
	fmt.Fprintf(file, "  Optimized SCAR Time: %d ms\n", result.OptimizedTime)
	if result.OptimizedTime > 0 {
		speedup := float64(result.HighPrecisionTime) / float64(result.OptimizedTime)
		fmt.Fprintf(file, "  Optimized SCAR Speedup: %.2fx\n", speedup)
	}
	fmt.Fprintf(file, "\n")
	
	// Configuration comparison
	fmt.Fprintf(file, "Configuration Comparison:\n")
	if result.HighPrecisionResult.SCARConfig != nil {
		hp := result.HighPrecisionResult.SCARConfig
		fmt.Fprintf(file, "  High-Precision SCAR: k=%d, nk=%d, threshold=%.3f\n", hp.K, hp.NK, hp.Threshold)
	}
	if result.OptimizedResult.SCARConfig != nil {
		opt := result.OptimizedResult.SCARConfig
		fmt.Fprintf(file, "  Optimized SCAR: k=%d, nk=%d, threshold=%.3f\n", opt.K, opt.NK, opt.Threshold)
	}
	fmt.Fprintf(file, "\n")
	
	// Hierarchical structure comparison
	fmt.Fprintf(file, "Hierarchical Structure:\n")
	fmt.Fprintf(file, "  High-Precision SCAR Max Level: %d\n", result.HighPrecisionPartition.MaxLevel)
	fmt.Fprintf(file, "  Optimized SCAR Max Level: %d\n", result.OptimizedPartition.MaxLevel)
	fmt.Fprintf(file, "  High-Precision SCAR Levels: %d\n", len(result.HighPrecisionPartition.Levels))
	fmt.Fprintf(file, "  Optimized SCAR Levels: %d\n", len(result.OptimizedPartition.Levels))
	fmt.Fprintf(file, "\n")
	
	// Level-by-level clustering metric analysis
	fmt.Fprintf(file, "Level-by-Level Clustering Metrics Analysis:\n")
	if len(result.NMIScores) == 0 {
		fmt.Fprintf(file, "  No common levels found for comparison\n")
	} else {
		totalNMI := 0.0
		totalARI := 0.0
		totalVI := 0.0
		totalHF1 := 0.0
		bestNMI := 0.0
		bestLevel := -1
		worstNMI := 1.0
		worstLevel := -1
		
		// Print header
		fmt.Fprintf(file, "  Level | NMI      | ARI      | VI       | HF1      | HP Comms | Opt Comms\n")
		fmt.Fprintf(file, "  ------|----------|----------|----------|----------|----------|----------\n")
		
		// Print each level
		for level := 0; level <= max(result.HighPrecisionPartition.MaxLevel, result.OptimizedPartition.MaxLevel); level++ {
			nmi, hasNMI := result.NMIScores[level]
			ari, hasARI := result.ARIScores[level]
			vi, hasVI := result.VIScores[level]
			hf1, hasHF1 := result.HierarchicalF1Scores[level]
			
			hpComms := 0
			optComms := 0
			
			if hpLevel, exists := result.HighPrecisionPartition.Levels[level]; exists {
				hpComms = len(hpLevel)
			}
			if optLevel, exists := result.OptimizedPartition.Levels[level]; exists {
				optComms = len(optLevel)
			}
			
			if hasNMI && hasARI && hasVI && hasHF1 {
				fmt.Fprintf(file, "  %-5d | %-8.4f | %-8.4f | %-8.4f | %-8.4f | %-8d | %-9d\n", 
					level, nmi, ari, vi, hf1, hpComms, optComms)
				
				totalNMI += nmi
				totalARI += ari
				totalVI += vi
				totalHF1 += hf1
				
				if nmi > bestNMI {
					bestNMI = nmi
					bestLevel = level
				}
				if nmi < worstNMI {
					worstNMI = nmi
					worstLevel = level
				}
			} else if hpComms > 0 || optComms > 0 {
				fmt.Fprintf(file, "  %-5d | %-8s | %-8s | %-8s | %-8s | %-8d | %-9d\n", 
					level, "N/A", "N/A", "N/A", "N/A", hpComms, optComms)
			}
		}
		
		// Summary statistics
		numLevels := float64(len(result.NMIScores))
		avgNMI := totalNMI / numLevels
		avgARI := totalARI / numLevels
		avgVI := totalVI / numLevels
		avgHF1 := totalHF1 / numLevels
		
		fmt.Fprintf(file, "\n")
		fmt.Fprintf(file, "  Summary Statistics:\n")
		fmt.Fprintf(file, "    Average NMI: %.6f\n", avgNMI)
		fmt.Fprintf(file, "    Average ARI: %.6f\n", avgARI)
		fmt.Fprintf(file, "    Average VI:  %.6f\n", avgVI)
		fmt.Fprintf(file, "    Average HF1: %.6f\n", avgHF1)
		fmt.Fprintf(file, "    Best NMI: %.6f (level %d)\n", bestNMI, bestLevel)
		fmt.Fprintf(file, "    Worst NMI: %.6f (level %d)\n", worstNMI, worstLevel)
		fmt.Fprintf(file, "    Tree Edit Distance: %.4f\n", result.TreeEditDistance)
		fmt.Fprintf(file, "    Hierarchy Depth Difference: %d\n", result.HierarchyDepthDiff)
	}
	fmt.Fprintf(file, "\n")
	
	// Detailed level analysis
	fmt.Fprintf(file, "Detailed Level Analysis:\n")
	for level := 0; level <= max(result.HighPrecisionPartition.MaxLevel, result.OptimizedPartition.MaxLevel); level++ {
		hpLevel, hpExists := result.HighPrecisionPartition.Levels[level]
		optLevel, optExists := result.OptimizedPartition.Levels[level]
		
		if !hpExists && !optExists {
			continue
		}
		
		fmt.Fprintf(file, "  Level %d:\n", level)
		
		if hpExists {
			hpNodes := 0
			for _, nodes := range hpLevel {
				hpNodes += len(nodes)
			}
			avgSize := float64(hpNodes) / float64(len(hpLevel))
			fmt.Fprintf(file, "    High-Precision SCAR: %d communities, %d nodes, avg size %.1f\n", 
				len(hpLevel), hpNodes, avgSize)
		} else {
			fmt.Fprintf(file, "    High-Precision SCAR: No data\n")
		}
		
		if optExists {
			optNodes := 0
			for _, nodes := range optLevel {
				optNodes += len(nodes)
			}
			avgSize := float64(optNodes) / float64(len(optLevel))
			fmt.Fprintf(file, "    Optimized SCAR: %d communities, %d nodes, avg size %.1f\n", 
				len(optLevel), optNodes, avgSize)
		} else {
			fmt.Fprintf(file, "    Optimized SCAR: No data\n")
		}
		
		// Show all metrics for this level
		if nmi, hasNMI := result.NMIScores[level]; hasNMI {
			ari := result.ARIScores[level]
			vi := result.VIScores[level]
			hf1 := result.HierarchicalF1Scores[level]
			fmt.Fprintf(file, "    Metrics: NMI=%.4f, ARI=%.4f, VI=%.4f, HF1=%.4f\n", nmi, ari, vi, hf1)
		} else {
			fmt.Fprintf(file, "    Metrics: Not available\n")
		}
		
		// Show structural differences
		if commDiff, exists := result.CommunityCountDiff[level]; exists {
			avgSizeDiff := result.AvgCommunitySizeDiff[level]
			fmt.Fprintf(file, "    Differences: Community count %+d, Avg size %+.1f\n", commDiff, avgSizeDiff)
		}
		
		fmt.Fprintf(file, "\n")
	}
	
	// Structural comparison summary
	fmt.Fprintf(file, "Structural Comparison Summary:\n")
	fmt.Fprintf(file, "  Tree Edit Distance: %.4f\n", result.TreeEditDistance)
	fmt.Fprintf(file, "  Hierarchy Depth Difference: %d levels\n", result.HierarchyDepthDiff)
	
	// Show level-by-level structural differences
	fmt.Fprintf(file, "  Level-by-Level Structural Differences:\n")
	fmt.Fprintf(file, "    Level | Comm Count Diff | Avg Size Diff\n")
	fmt.Fprintf(file, "    ------|-----------------|---------------\n")
	for level := 0; level <= max(result.HighPrecisionPartition.MaxLevel, result.OptimizedPartition.MaxLevel); level++ {
		if commDiff, exists := result.CommunityCountDiff[level]; exists {
			avgSizeDiff := result.AvgCommunitySizeDiff[level]
			fmt.Fprintf(file, "    %-5d | %+15d | %+13.1f\n", level, commDiff, avgSizeDiff)
		}
	}
	fmt.Fprintf(file, "\n")
	
	// Overall interpretation
	fmt.Fprintf(file, "Overall Interpretation:\n")
	if len(result.NMIScores) == 0 {
		fmt.Fprintf(file, "  No common levels found - SCAR configurations produced different hierarchical structures\n")
	} else {
		totalNMI := 0.0
		totalARI := 0.0
		totalVI := 0.0
		totalHF1 := 0.0
		
		for level := range result.NMIScores {
			totalNMI += result.NMIScores[level]
			totalARI += result.ARIScores[level]
			totalVI += result.VIScores[level]
			totalHF1 += result.HierarchicalF1Scores[level]
		}
		
		numLevels := float64(len(result.NMIScores))
		avgNMI := totalNMI / numLevels
		avgARI := totalARI / numLevels
		avgVI := totalVI / numLevels
		avgHF1 := totalHF1 / numLevels
		
		// Overall agreement assessment
		if avgNMI >= 0.8 && avgARI >= 0.8 && avgHF1 >= 0.8 {
			fmt.Fprintf(file, "  EXCELLENT AGREEMENT: Both SCAR configurations found very similar hierarchical structures\n")
		} else if avgNMI >= 0.6 && avgARI >= 0.6 && avgHF1 >= 0.6 {
			fmt.Fprintf(file, "  GOOD AGREEMENT: Both SCAR configurations found similar hierarchical structures\n")
		} else if avgNMI >= 0.4 && avgARI >= 0.4 && avgHF1 >= 0.4 {
			fmt.Fprintf(file, "  MODERATE AGREEMENT: SCAR configurations found somewhat different hierarchical structures\n")
		} else {
			fmt.Fprintf(file, "  POOR AGREEMENT: SCAR configurations found very different hierarchical structures\n")
		}
		
		// Detailed metric interpretation
		fmt.Fprintf(file, "\n  Metric-specific Analysis:\n")
		fmt.Fprintf(file, "    Clustering Similarity (NMI): %.4f - ", avgNMI)
		if avgNMI >= 0.7 {
			fmt.Fprintf(file, "High similarity in community assignments\n")
		} else if avgNMI >= 0.5 {
			fmt.Fprintf(file, "Moderate similarity in community assignments\n")
		} else {
			fmt.Fprintf(file, "Low similarity in community assignments\n")
		}
		
		fmt.Fprintf(file, "    Pair Agreement (ARI): %.4f - ", avgARI)
		if avgARI >= 0.7 {
			fmt.Fprintf(file, "High agreement on node pair assignments\n")
		} else if avgARI >= 0.5 {
			fmt.Fprintf(file, "Moderate agreement on node pair assignments\n")
		} else {
			fmt.Fprintf(file, "Low agreement on node pair assignments\n")
		}
		
		fmt.Fprintf(file, "    Information Distance (VI): %.4f - ", avgVI)
		if avgVI <= 1.0 {
			fmt.Fprintf(file, "Low information distance (similar structures)\n")
		} else if avgVI <= 2.0 {
			fmt.Fprintf(file, "Moderate information distance\n")
		} else {
			fmt.Fprintf(file, "High information distance (different structures)\n")
		}
		
		fmt.Fprintf(file, "    Hierarchical Overlap (HF1): %.4f - ", avgHF1)
		if avgHF1 >= 0.7 {
			fmt.Fprintf(file, "High community overlap\n")
		} else if avgHF1 >= 0.5 {
			fmt.Fprintf(file, "Moderate community overlap\n")
		} else {
			fmt.Fprintf(file, "Low community overlap\n")
		}
		
		fmt.Fprintf(file, "    Structural Similarity: %.4f - ", 1.0-result.TreeEditDistance)
		if result.TreeEditDistance <= 0.2 {
			fmt.Fprintf(file, "Very similar hierarchical structures\n")
		} else if result.TreeEditDistance <= 0.4 {
			fmt.Fprintf(file, "Moderately similar hierarchical structures\n")
		} else {
			fmt.Fprintf(file, "Different hierarchical structures\n")
		}
		
		// Parameter analysis
		if result.HighPrecisionResult.SCARConfig != nil && result.OptimizedResult.SCARConfig != nil {
			hp := result.HighPrecisionResult.SCARConfig
			opt := result.OptimizedResult.SCARConfig
			
			fmt.Fprintf(file, "\n  Parameter Impact Analysis:\n")
			reductionRatio := float64(hp.K) / float64(opt.K)
			fmt.Fprintf(file, "    Sketch size reduction: %d -> %d (%.1fx smaller)\n", 
				hp.K, opt.K, reductionRatio)
			fmt.Fprintf(file, "    Layer count change: %d -> %d\n", hp.NK, opt.NK)
			fmt.Fprintf(file, "    Threshold change: %.3f -> %.3f\n", hp.Threshold, opt.Threshold)
			
			// Quality vs efficiency trade-off analysis
			if avgNMI >= 0.8 && reductionRatio >= 4.0 {
				fmt.Fprintf(file, "    EXCELLENT TRADE-OFF: Significant parameter reduction with minimal quality loss\n")
			} else if avgNMI >= 0.6 && reductionRatio >= 2.0 {
				fmt.Fprintf(file, "    GOOD TRADE-OFF: Parameter reduction with acceptable quality loss\n")
			} else if avgNMI >= 0.4 {
				fmt.Fprintf(file, "    FAIR TRADE-OFF: Parameter reduction causes moderate quality loss\n")
			} else {
				fmt.Fprintf(file, "    POOR TRADE-OFF: Parameter reduction causes significant quality loss\n")
			}
		}
	}
	
	if result.OptimizedTime > 0 && result.HighPrecisionTime > 0 {
		speedup := float64(result.HighPrecisionTime) / float64(result.OptimizedTime)
		if speedup > 2.0 {
			fmt.Fprintf(file, "  PERFORMANCE: Optimized SCAR shows significant speedup (%.1fx)\n", speedup)
		} else if speedup > 1.2 {
			fmt.Fprintf(file, "  PERFORMANCE: Optimized SCAR shows moderate speedup (%.1fx)\n", speedup)
		} else {
			fmt.Fprintf(file, "  PERFORMANCE: Similar execution times\n")
		}
	}
	
	return nil
}

// max returns the maximum of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// writeSCARSummary creates a summary file for SCAR results
func writeSCARSummary(scarConfig *scar.SCARConfig, config *PipelineConfig, totalTime time.Duration, variant string) error {
	summaryPath := filepath.Join(config.OutputDir, config.OutputPrefix+"_"+variant+"_scar_summary.txt")
	
	file, err := os.Create(summaryPath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "=== %s SCAR Pipeline Summary ===\n\n", strings.Title(variant))
	fmt.Fprintf(file, "SCAR Configuration:\n")
	fmt.Fprintf(file, "  Sketch Size (k): %d\n", scarConfig.K)
	fmt.Fprintf(file, "  Sketch Layers (nk): %d\n", scarConfig.NK)
	fmt.Fprintf(file, "  Threshold: %.3f\n", scarConfig.Threshold)
	fmt.Fprintf(file, "  Use Louvain: %t\n", scarConfig.UseLouvain)
	fmt.Fprintf(file, "  Sketch Output: %t\n", scarConfig.SketchOutput)
	fmt.Fprintf(file, "\nInput Files:\n")
	fmt.Fprintf(file, "  Graph: %s\n", scarConfig.GraphFile)
	fmt.Fprintf(file, "  Properties: %s\n", scarConfig.PropertyFile)
	fmt.Fprintf(file, "  Path: %s\n", scarConfig.PathFile)
	fmt.Fprintf(file, "\nExecution:\n")
	fmt.Fprintf(file, "  Total Runtime: %v\n", totalTime)
	fmt.Fprintf(file, "  Success: %t\n", true)
	
	fmt.Fprintf(file, "\nOutput Files Generated:\n")
	if scarConfig.SketchOutput {
		fmt.Fprintf(file, "  %s_mapping.dat - Community hierarchy mapping\n", config.OutputPrefix)
		fmt.Fprintf(file, "  %s_hierarchy.dat - Parent-child relationships\n", config.OutputPrefix)
		fmt.Fprintf(file, "  %s_root.dat - Top-level communities\n", config.OutputPrefix)
		fmt.Fprintf(file, "  %s.sketch - Node sketches with levels\n", config.OutputPrefix)
	} else {
		fmt.Fprintf(file, "  output.txt - Simple node-community pairs\n")
	}
	
	return nil
}

// allNodesAreIntegers checks if all node IDs are integers
func allNodesAreIntegers(nodes []string) bool {
	for _, node := range nodes {
		if _, err := strconv.Atoi(node); err != nil {
			return false
		}
	}
	return true
}

// Example usage and main function
func main() {
	// Define command line flags
	var (
		// Optimized SCAR flags (the second run)
		scarK         = flag.Int64("scar-k", 64, "Optimized SCAR sketch size (default: 64)")
		scarNK        = flag.Int64("scar-nk", 4, "Optimized SCAR number of sketch layers (default: 4)")
		scarThreshold = flag.Float64("scar-threshold", 0.5, "Optimized SCAR threshold (default: 0.5)")
		scarSketch    = flag.Bool("scar-sketch-output", true, "SCAR generate hierarchy files for PPRViz (default: true)")
		
		// General flags
		verbose = flag.Bool("verbose", true, "Verbose output (default: true)")
		prefix  = flag.String("prefix", "communities", "Output file prefix (default: communities)")
	)
	
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <pipeline_type> <graph_file> <properties_file> <path_file> [output_dir]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Pipeline Types:\n")
		fmt.Fprintf(os.Stderr, "  high_precision - High-precision SCAR pipeline (k=512, nk=1, th=0)\n")
		fmt.Fprintf(os.Stderr, "  optimized      - Optimized SCAR pipeline (uses command line params)\n")
		fmt.Fprintf(os.Stderr, "  compare        - Run both SCAR pipelines and compare results using NMI\n")
		fmt.Fprintf(os.Stderr, "  multi_k        - Compare SCAR across multiple k values (1,2,4,8,16,32,64,128,256,512)\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  %s high_precision graph.txt properties.txt path.txt results/\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s optimized graph.txt properties.txt path.txt results/\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s compare graph.txt properties.txt path.txt results/\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s multi_k graph.txt properties.txt path.txt results/\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -scar-k=128 -scar-nk=6 optimized graph.txt props.txt path.txt results/\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -scar-k=32 -scar-nk=2 -scar-threshold=0.3 compare graph.txt props.txt path.txt results/\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Notes:\n")
		fmt.Fprintf(os.Stderr, "  - High-precision SCAR uses fixed parameters: k=512, nk=1, threshold=0\n")
		fmt.Fprintf(os.Stderr, "  - This approximates the behavior of materialization+louvain\n")
		fmt.Fprintf(os.Stderr, "  - Optimized SCAR uses the command line parameters for efficiency\n")
		fmt.Fprintf(os.Stderr, "  - Multi-k mode tests k values: 1,2,4,8,16,32,64,128,256,512 against k=512 baseline\n")
		fmt.Fprintf(os.Stderr, "  - Multi-k generates CSV files for Jupyter notebook plotting\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
	
	flag.Parse()
	
	// Check remaining arguments after flag parsing
	args := flag.Args()
	if len(args) < 4 {
		flag.Usage()
		os.Exit(1)
	}
	
	pipelineType := args[0]
	graphFile := args[1]
	propertiesFile := args[2]
	pathFile := args[3]
	
	outputDir := "scar_comparison_output"
	if len(args) > 4 {
		outputDir = args[4]
	}
	
	// Create configuration
	config := NewPipelineConfig()
	config.OutputDir = outputDir
	config.OutputPrefix = *prefix
	config.Verbose = *verbose
	
	// Configure optimized SCAR with command line flags
	config.OptimizedSCARConfig.K = *scarK
	config.OptimizedSCARConfig.NK = *scarNK
	config.OptimizedSCARConfig.Threshold = *scarThreshold
	config.OptimizedSCARConfig.SketchOutput = *scarSketch
	
	var result *PipelineResult
	var compResult *ComparisonResult
	var multiKResult *MultiKComparisonResult
	var err error
	
	switch pipelineType {
	case "high_precision":
		if config.Verbose {
			fmt.Printf("High-precision SCAR config: k=%d, nk=%d, threshold=%.3f\n", 
				config.HighPrecisionSCARConfig.K, config.HighPrecisionSCARConfig.NK, config.HighPrecisionSCARConfig.Threshold)
		}
		
		result, err = RunHighPrecisionSCAR(graphFile, propertiesFile, pathFile, config)
		
	case "optimized":
		if config.Verbose {
			fmt.Printf("Optimized SCAR config: k=%d, nk=%d, threshold=%.3f, sketch_output=%t\n", 
				*scarK, *scarNK, *scarThreshold, *scarSketch)
		}
		
		result, err = RunOptimizedSCAR(graphFile, propertiesFile, pathFile, config)
		
	case "compare":
		if config.Verbose {
			fmt.Printf("Comparison mode: Running both SCAR configurations\n")
			fmt.Printf("High-precision SCAR config: k=%d, nk=%d, threshold=%.3f\n", 
				config.HighPrecisionSCARConfig.K, config.HighPrecisionSCARConfig.NK, config.HighPrecisionSCARConfig.Threshold)
			fmt.Printf("Optimized SCAR config: k=%d, nk=%d, threshold=%.3f, sketch_output=%t\n", 
				*scarK, *scarNK, *scarThreshold, *scarSketch)
		}
		
		compResult, err = RunSCARComparison(graphFile, propertiesFile, pathFile, config)
		
	case "multi_k":
		if config.Verbose {
			fmt.Printf("Multi-k comparison mode: Testing k values 1,2,4,8,16,32,64,128,256,512\n")
			fmt.Printf("Baseline: k=512, nk=1, threshold=0\n")
			fmt.Printf("Other parameters: nk=%d, threshold=%.3f, sketch_output=%t\n", 
				*scarNK, *scarThreshold, *scarSketch)
		}
		
		multiKResult, err = RunMultiKComparison(graphFile, propertiesFile, pathFile, config)
		
	default:
		log.Fatalf("Unknown pipeline type: %s. Use 'high_precision', 'optimized', 'compare', or 'multi_k'", pipelineType)
	}
	
	if err != nil {
		log.Fatalf("Pipeline failed: %v", err)
	}
	
	// Print final summary
	fmt.Println("\n=== Final Results ===")
	
	if multiKResult != nil {
		// Multi-k comparison results
		fmt.Printf("Multi-k SCAR comparison completed successfully!\n")
		fmt.Printf("Total comparison time: %d ms\n", multiKResult.TotalComparisonTime)
		fmt.Printf("Tested k values: %v\n", multiKResult.TestedKValues)
		
		// Find best performing k
		bestK := int64(1)
		bestNMI := 0.0
		for _, k := range multiKResult.TestedKValues {
			if k == 512 {
				continue // Skip baseline
			}
			if avgNMI := multiKResult.KAvgNMI[k]; avgNMI > bestNMI {
				bestNMI = avgNMI
				bestK = k
			}
		}
		
		fmt.Printf("\nPerformance Summary:\n")
		baselineTime := multiKResult.KExecutionTime[512]
		for _, k := range multiKResult.TestedKValues {
			execTime := multiKResult.KExecutionTime[k]
			speedup := float64(baselineTime) / float64(execTime)
			avgNMI := multiKResult.KAvgNMI[k]
			avgARI := multiKResult.KAvgARI[k]
			avgVI := multiKResult.KAvgVI[k]
			avgHF1 := multiKResult.KAvgHF1[k]
			
			fmt.Printf("  k=%d: %.1fx speedup, NMI=%.4f, ARI=%.4f, VI=%.4f, HF1=%.4f\n",
				k, speedup, avgNMI, avgARI, avgVI, avgHF1)
		}
		
		fmt.Printf("\nRecommended k value: %d (best balance of quality and speed)\n", bestK)
		fmt.Printf("Quality at recommended k: NMI=%.4f, ARI=%.4f\n", 
			multiKResult.KAvgNMI[bestK], multiKResult.KAvgARI[bestK])
		fmt.Printf("Speedup at recommended k: %.1fx\n", 
			float64(baselineTime)/float64(multiKResult.KExecutionTime[bestK]))
		
		fmt.Printf("\nOutput files:\n")
		fmt.Printf("  Report: %s/multi_k_comparison_report.txt\n", config.OutputDir)
		fmt.Printf("  CSV for plotting: %s/multi_k_comparison_data.csv\n", config.OutputDir)
		fmt.Printf("  Ready for Jupyter notebook analysis!\n")
		
	} else if compResult != nil {
		// Comparison results
		fmt.Printf("SCAR vs SCAR comparison completed successfully!\n")
		fmt.Printf("Total comparison time: %d ms\n", compResult.TotalComparisonTime)
		fmt.Printf("High-precision SCAR time: %d ms\n", compResult.HighPrecisionTime)
		fmt.Printf("Optimized SCAR time: %d ms\n", compResult.OptimizedTime)
		
		if compResult.OptimizedTime > 0 {
			speedup := float64(compResult.HighPrecisionTime) / float64(compResult.OptimizedTime)
			fmt.Printf("Optimized SCAR Speedup: %.2fx\n", speedup)
		}
		
		fmt.Printf("Common levels analyzed: %d\n", len(compResult.CommonLevels))
		
		if len(compResult.NMIScores) > 0 {
			// Calculate and display metric statistics
			totalNMI := 0.0
			totalARI := 0.0
			totalVI := 0.0
			totalHF1 := 0.0
			bestNMI := 0.0
			bestLevel := -1
			
			for level, nmi := range compResult.NMIScores {
				totalNMI += nmi
				totalARI += compResult.ARIScores[level]
				totalVI += compResult.VIScores[level]
				totalHF1 += compResult.HierarchicalF1Scores[level]
				if nmi > bestNMI {
					bestNMI = nmi
					bestLevel = level
				}
			}
			
			numLevels := float64(len(compResult.NMIScores))
			avgNMI := totalNMI / numLevels
			avgARI := totalARI / numLevels
			avgVI := totalVI / numLevels
			avgHF1 := totalHF1 / numLevels
			
			fmt.Printf("Average Clustering Metrics:\n")
			fmt.Printf("  NMI: %.4f, ARI: %.4f, VI: %.4f, HF1: %.4f\n", avgNMI, avgARI, avgVI, avgHF1)
			fmt.Printf("Best NMI Score: %.6f (level %d)\n", bestNMI, bestLevel)
			fmt.Printf("Tree Edit Distance: %.4f\n", compResult.TreeEditDistance)
			fmt.Printf("Hierarchy Depth Difference: %d\n", compResult.HierarchyDepthDiff)
			
			// Display level-by-level metrics for key levels
			fmt.Printf("Key Level Metrics:\n")
			for level := 0; level <= min(3, max(compResult.HighPrecisionPartition.MaxLevel, compResult.OptimizedPartition.MaxLevel)); level++ {
				if nmi, exists := compResult.NMIScores[level]; exists {
					ari := compResult.ARIScores[level]
					vi := compResult.VIScores[level]
					hf1 := compResult.HierarchicalF1Scores[level]
					fmt.Printf("  Level %d: NMI=%.4f, ARI=%.4f, VI=%.4f, HF1=%.4f\n", level, nmi, ari, vi, hf1)
				}
			}
			
			// Overall interpretation
			if avgNMI >= 0.8 && avgARI >= 0.8 && avgHF1 >= 0.8 {
				fmt.Printf("Result: EXCELLENT AGREEMENT between SCAR configurations\n")
			} else if avgNMI >= 0.6 && avgARI >= 0.6 && avgHF1 >= 0.6 {
				fmt.Printf("Result: GOOD AGREEMENT between SCAR configurations\n")
			} else if avgNMI >= 0.4 && avgARI >= 0.4 && avgHF1 >= 0.4 {
				fmt.Printf("Result: MODERATE AGREEMENT between SCAR configurations\n")
			} else {
				fmt.Printf("Result: POOR AGREEMENT between SCAR configurations\n")
			}
		} else {
			fmt.Printf("No common levels found for comparison\n")
		}
		
		fmt.Printf("Detailed SCAR comparison report: %s/scar_comparison_report.txt\n", config.OutputDir)
		
	} else {
		// Single pipeline results
		fmt.Printf("SCAR pipeline completed successfully!\n")
		if result.PipelineType == HighPrecisionSCAR {
			fmt.Printf("Pipeline type: High-precision SCAR (k=512, nk=1, th=0)\n")
		} else {
			fmt.Printf("Pipeline type: Optimized SCAR\n")
		}
		fmt.Printf("Total runtime: %d ms\n", result.TotalRuntimeMS)
		fmt.Printf("Generated hierarchy files for PPRViz integration\n")
		fmt.Printf("Output files written to: %s/\n", config.OutputDir)
	}
}
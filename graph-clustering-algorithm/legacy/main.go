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
	"sort"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	
	"gonum.org/v1/gonum/stat"
)

// PipelineType defines which pipeline to run
type PipelineType int

const (
	MaterializationLouvain PipelineType = iota
	SketchLouvain
	Comparison
)

// PipelineConfig holds configuration for both pipeline types
type PipelineConfig struct {
	// Common options
	Verbose      bool
	OutputDir    string
	OutputPrefix string
	
	// Materialization + Louvain config
	MaterializationConfig materialization.MaterializationConfig
	LouvainConfig        louvain.LouvainConfig
	
	// SCAR config
	SCARConfig scar.SCARConfig
}

// PipelineResult contains results from either pipeline
type PipelineResult struct {
	PipelineType    PipelineType
	TotalRuntimeMS  int64
	
	// Materialization + Louvain results (nil if SketchLouvain was used)
	MaterializedGraph *materialization.HomogeneousGraph
	LouvainResult     *louvain.LouvainResult
	
	// SCAR results (basic info - actual files written to disk)
	SCARSuccess bool
	SCARConfig  *scar.SCARConfig
}

// ComparisonResult contains results from comparing both pipelines
type ComparisonResult struct {
	MaterializationResult *PipelineResult
	SCARResult           *PipelineResult
	NMIScores            map[int]float64 // NMI scores by level
	MaterializationPartition *HierarchicalPartition
	SCARPartition            *HierarchicalPartition
	NumNodes             int
	MaterializationTime  int64
	SCARTime            int64
	TotalComparisonTime int64
	CommonLevels        []int  // Levels present in both partitions
}

// NewPipelineConfig creates default configuration for both pipelines
func NewPipelineConfig() *PipelineConfig {
	return &PipelineConfig{
		Verbose:      true,
		OutputDir:    "pipeline_output",
		OutputPrefix: "communities",
		
		// Default materialization + Louvain
		MaterializationConfig: materialization.DefaultMaterializationConfig(),
		LouvainConfig:        louvain.DefaultLouvainConfig(),
		
		// Default SCAR config
		SCARConfig: scar.SCARConfig{
			K:           1024,
			NK:          4,
			Threshold:   0.5,
			UseLouvain:  true,
			SketchOutput: true, // For hierarchy output compatible with PPRViz
			WriteSketchGraph: true, // Write sketch graph files
			SketchGraphWeights: false, // Use weights in sketch graph files
		},
	}
}

// RunMaterializationLouvain executes the materialization + Louvain pipeline
func RunMaterializationLouvain(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*PipelineResult, error) {
	startTime := time.Now()
	
	if config.Verbose {
		fmt.Println("=== Running Materialization + Louvain Pipeline ===")
	}
	
	// Step 1: Parse SCAR input for materialization
	if config.Verbose {
		fmt.Println("Step 1: Parsing input files for materialization...")
	}
	
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)

	// Print full graph structure for debugging
	if err != nil {
		return nil, fmt.Errorf("failed to parse SCAR input: %w", err)
	}
	
	if config.Verbose {
		fmt.Printf("  Loaded graph with %d nodes\n", len(graph.Nodes))
	}
	
	// Step 2: Run materialization
	if config.Verbose {
		fmt.Println("Step 2: Running graph materialization...")
	}
	
	materializationStart := time.Now()
	
	// Setup progress callback for materialization
	var materializationProgressCb func(int, int, string)
	if config.Verbose {
		materializationProgressCb = func(current, total int, message string) {
			fmt.Printf("  Materialization progress: %d/%d - %s\n", current, total, message)
		}
	}
	
	engine := materialization.NewMaterializationEngine(graph, metaPath, config.MaterializationConfig, materializationProgressCb)
	materializationResult, err := engine.Materialize()
	if err != nil {
		return nil, fmt.Errorf("materialization failed: %w", err)
	}
	
	materializationTime := time.Since(materializationStart)
	
	if config.Verbose {
		fmt.Printf("  Materialization completed in %v\n", materializationTime)
		fmt.Printf("  Materialized graph has %d nodes and %d edges\n", 
			len(materializationResult.HomogeneousGraph.Nodes),
			len(materializationResult.HomogeneousGraph.Edges))
	}
	
	// Step 3: Convert HomogeneousGraph to NormalizedGraph for Louvain
	if config.Verbose {
		fmt.Println("Step 3: Converting graph format for Louvain...")
	}
	
	normalizedGraph, graphParser, err := convertHomogeneousToNormalized(materializationResult.HomogeneousGraph)
	if err != nil {
		return nil, fmt.Errorf("graph conversion failed: %w", err)
	}
	
	if config.Verbose {
		fmt.Printf("  Converted to normalized graph with %d nodes\n", normalizedGraph.NumNodes)
		fmt.Printf("  Total edge weight: %.2f\n", normalizedGraph.TotalWeight)
	}
	
	// Step 4: Run Louvain clustering
	if config.Verbose {
		fmt.Println("Step 4: Running Louvain community detection...")
	}
	
	louvainStart := time.Now()
	
	// Setup progress callback for Louvain
	if config.Verbose {
		config.LouvainConfig.Verbose = true
		config.LouvainConfig.ProgressCallback = func(level, iteration int, message string) {
			fmt.Printf("  Louvain [L%d I%d]: %s\n", level, iteration, message)
		}
	}
	
	louvainResult, err := louvain.RunLouvain(normalizedGraph, config.LouvainConfig)
	if err != nil {
		return nil, fmt.Errorf("Louvain clustering failed: %w", err)
	}
	
	louvainTime := time.Since(louvainStart)
	louvainResult.Parser = graphParser // Attach parser for output generation
	
	if config.Verbose {
		fmt.Printf("  Louvain completed in %v\n", louvainTime)
		fmt.Printf("  Final modularity: %.6f\n", louvainResult.Modularity)
		finalLevel := louvainResult.Levels[len(louvainResult.Levels)-1]
		fmt.Printf("  Number of communities: %d\n", finalLevel.NumCommunities)
		fmt.Printf("  Hierarchy levels: %d\n", louvainResult.NumLevels)
	}
	
	// Step 5: Generate output files
	if config.Verbose {
		fmt.Println("Step 5: Writing output files...")
	}
	
	if err := writeLouvainOutputs(louvainResult, graphParser, materializationResult.HomogeneousGraph, config); err != nil {
		return nil, fmt.Errorf("output generation failed: %w", err)
	}
	
	totalTime := time.Since(startTime)
	
	// Create final result
	result := &PipelineResult{
		PipelineType:      MaterializationLouvain,
		MaterializedGraph: materializationResult.HomogeneousGraph,
		LouvainResult:     louvainResult,
		TotalRuntimeMS:    totalTime.Milliseconds(),
	}
	
	if config.Verbose {
		fmt.Println("=== Materialization + Louvain Pipeline Complete ===")
		fmt.Printf("Total runtime: %v\n", totalTime)
		fmt.Printf("Materialization: %v, Louvain: %v\n", materializationTime, louvainTime)
		fmt.Printf("Final modularity: %.6f\n", result.LouvainResult.Modularity)
	}
	
	return result, nil
}

// RunComparison executes both pipelines and compares their results using NMI
func RunComparison(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*ComparisonResult, error) {
    startTime := time.Now()
    if config.Verbose {
        fmt.Println("=== Running Pipeline Comparison ===")
        fmt.Println("Will run both Materialization+Louvain and SCAR pipelines and compare results")
    }

    // Prepare output dirs
    matOutputDir := filepath.Join(config.OutputDir, "materialization_louvain")
    scarOutputDir := filepath.Join(config.OutputDir, "scar")

    // Step 1: Materialization + Louvain
    if config.Verbose {
        fmt.Println("\nStep 1: Running Materialization + Louvain pipeline...")
    }
    matConfig := *config
    matConfig.OutputDir = matOutputDir
    matConfig.OutputPrefix = "mat_communities"
    matStart := time.Now()
    matResult, err := RunMaterializationLouvain(graphFile, propertiesFile, pathFile, &matConfig)
    if err != nil {
        return nil, fmt.Errorf("materialization+Louvain pipeline failed: %w", err)
    }
    matTime := time.Since(matStart)

    // Step 2: SCAR
    if config.Verbose {
        fmt.Println("\nStep 2: Running SCAR sketch-based Louvain pipeline...")
    }
    scarConfig := *config
    scarConfig.OutputDir = scarOutputDir
    scarConfig.OutputPrefix = "scar_communities"
    scarConfig.SCARConfig.WriteSketchGraph = true
    scarConfig.SCARConfig.SketchGraphWeights = false
    scarStart := time.Now()
    scarResult, err := RunSketchLouvain(graphFile, propertiesFile, pathFile, &scarConfig)
    if err != nil {
        return nil, fmt.Errorf("SCAR pipeline failed: %w", err)
    }
    scarTime := time.Since(scarStart)

    // Step 3: Parse partitions
    if config.Verbose {
        fmt.Println("\nStep 3: Parsing hierarchical community assignments...")
    }
    matPartition, err := parseLouvainOutput(matOutputDir, "mat_communities")
    if err != nil {
        return nil, fmt.Errorf("failed to parse materialization+Louvain output: %w", err)
    }
    scarPartition, err := parseSCAROutput(scarOutputDir, "scar_communities")
    if err != nil {
        return nil, fmt.Errorf("failed to parse SCAR output: %w", err)
    }

    // Step 4: Calculate per-level NMI
    if config.Verbose {
        fmt.Println("\nStep 4: Calculating hierarchical NMI scores...")
    }
    nmiScores, err := calculateHierarchicalNMI(matPartition, scarPartition)
    if err != nil {
        return nil, fmt.Errorf("hierarchical NMI calculation failed: %w", err)
    }
    commonLevels := make([]int, 0, len(nmiScores))
    for lvl := range nmiScores {
        commonLevels = append(commonLevels, lvl)
    }
    if config.Verbose {
        fmt.Printf("  Calculated NMI for %d common levels\n", len(commonLevels))
        for _, lvl := range commonLevels {
            fmt.Printf("    Level %d: NMI = %.6f\n", lvl, nmiScores[lvl])
        }
    }

    // --- New Metrics ---
    // 4a. Hierarchical Mutual Information (average NMI)
    hmi := hierarchicalMutualInformation(nmiScores)
    if config.Verbose {
        fmt.Printf("Hierarchical Mutual Information (avg NMI): %.6f\n", hmi)
    }

    // 4b. Cophenetic Correlation
    cc, err := CopheneticCorrelation(matPartition, scarPartition)
    if err != nil {
        fmt.Printf("Warning: could not compute cophenetic correlation: %v\n", err)
    } else if config.Verbose {
        fmt.Printf("Cophenetic Correlation: %.6f\n", cc)
    }

    // Step 5: Report and write
    if config.Verbose {
        fmt.Println("\nStep 5: Generating hierarchical comparison report...")
    }
    totalTime := time.Since(startTime)

    totalNodes := 0
    if lvl0, ok := matPartition.Levels[0]; ok {
        for _, nodes := range lvl0 {
            totalNodes += len(nodes)
        }
    }

    result := &ComparisonResult{
        MaterializationResult:    matResult,
        SCARResult:               scarResult,
        NMIScores:                nmiScores,
        MaterializationPartition: matPartition,
        SCARPartition:            scarPartition,
        NumNodes:                 totalNodes,
        MaterializationTime:      matTime.Milliseconds(),
        SCARTime:                 scarTime.Milliseconds(),
        TotalComparisonTime:      totalTime.Milliseconds(),
        CommonLevels:             commonLevels,
    }

    if err := writeHierarchicalComparisonReport(result, config); err != nil {
        return nil, fmt.Errorf("failed to write comparison report: %w", err)
    }

    if config.Verbose {
        fmt.Println("\n=== Hierarchical Pipeline Comparison Complete ===")
        fmt.Printf("Total comparison time: %v\n", totalTime)
        fmt.Printf("Materialization+Louvain time: %v\n", matTime)
        fmt.Printf("SCAR time: %v\n", scarTime)
        avgNMI := hierarchicalMutualInformation(nmiScores)
        fmt.Printf("Average NMI: %.6f, HMI: %.6f, CopheneticCorr: %.6f\n", avgNMI, hmi, cc)
    }

    return result, nil
}


// HierarchicalPartition represents partitions at multiple levels
type HierarchicalPartition struct {
	Levels map[int]map[int][]string // [level][community_id][node_list]
	MaxLevel int
}

// parseLouvainOutput parses the Louvain hierarchy files (.mapping, .hierarchy, .root)
func parseLouvainOutput(outputDir, prefix string) (*HierarchicalPartition, error) {
	mappingFile := filepath.Join(outputDir, prefix+".mapping")
	return parseLouvainMappingFile(mappingFile)
}

// parseLouvainMappingFile parses the .mapping file to extract hierarchical partitions
func parseLouvainMappingFile(filePath string) (*HierarchicalPartition, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open Louvain mapping file %s: %w", filePath, err)
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
		return nil, fmt.Errorf("error reading Louvain mapping file: %w", err)
	}
	
	return partition, nil
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

// writeHierarchicalComparisonReport generates a comprehensive hierarchical comparison report
func writeHierarchicalComparisonReport(result *ComparisonResult, config *PipelineConfig) error {
	reportPath := filepath.Join(config.OutputDir, "hierarchical_comparison_report.txt")
	
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	file, err := os.Create(reportPath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "=== Hierarchical Pipeline Comparison Report ===\n\n")
	
	// Overall comparison
	fmt.Fprintf(file, "Comparison Summary:\n")
	fmt.Fprintf(file, "  Number of Nodes: %d\n", result.NumNodes)
	fmt.Fprintf(file, "  Common Levels: %d\n", len(result.CommonLevels))
	fmt.Fprintf(file, "  Total Comparison Time: %d ms\n", result.TotalComparisonTime)
	fmt.Fprintf(file, "\n")
	
	// Performance comparison
	fmt.Fprintf(file, "Performance Comparison:\n")
	fmt.Fprintf(file, "  Materialization+Louvain Time: %d ms\n", result.MaterializationTime)
	fmt.Fprintf(file, "  SCAR Time: %d ms\n", result.SCARTime)
	if result.SCARTime > 0 {
		speedup := float64(result.MaterializationTime) / float64(result.SCARTime)
		fmt.Fprintf(file, "  SCAR Speedup: %.2fx\n", speedup)
	}
	fmt.Fprintf(file, "\n")
	
	// Hierarchical structure comparison
	fmt.Fprintf(file, "Hierarchical Structure:\n")
	fmt.Fprintf(file, "  Materialization+Louvain Max Level: %d\n", result.MaterializationPartition.MaxLevel)
	fmt.Fprintf(file, "  SCAR Max Level: %d\n", result.SCARPartition.MaxLevel)
	fmt.Fprintf(file, "  Materialization+Louvain Levels: %d\n", len(result.MaterializationPartition.Levels))
	fmt.Fprintf(file, "  SCAR Levels: %d\n", len(result.SCARPartition.Levels))
	fmt.Fprintf(file, "\n")
	
	// Level-by-level NMI analysis
	fmt.Fprintf(file, "Level-by-Level NMI Analysis:\n")
	if len(result.NMIScores) == 0 {
		fmt.Fprintf(file, "  No common levels found for comparison\n")
	} else {
		totalNMI := 0.0
		bestNMI := 0.0
		bestLevel := -1
		worstNMI := 1.0
		worstLevel := -1
		
		// Print each level
		for level := 0; level <= max(result.MaterializationPartition.MaxLevel, result.SCARPartition.MaxLevel); level++ {
			nmi, hasNMI := result.NMIScores[level]
			matComms := 0
			scarComms := 0
			
			if matLevel, exists := result.MaterializationPartition.Levels[level]; exists {
				matComms = len(matLevel)
			}
			if scarLevel, exists := result.SCARPartition.Levels[level]; exists {
				scarComms = len(scarLevel)
			}
			
			if hasNMI {
				fmt.Fprintf(file, "  Level %d: NMI=%.6f, Mat=%d comms, SCAR=%d comms\n", 
					level, nmi, matComms, scarComms)
				totalNMI += nmi
				if nmi > bestNMI {
					bestNMI = nmi
					bestLevel = level
				}
				if nmi < worstNMI {
					worstNMI = nmi
					worstLevel = level
				}
			} else if matComms > 0 || scarComms > 0 {
				fmt.Fprintf(file, "  Level %d: No comparison (Mat=%d comms, SCAR=%d comms)\n", 
					level, matComms, scarComms)
			}
		}
		
		// Summary statistics
		avgNMI := totalNMI / float64(len(result.NMIScores))
		fmt.Fprintf(file, "\n")
		fmt.Fprintf(file, "  Summary Statistics:\n")
		fmt.Fprintf(file, "    Average NMI: %.6f\n", avgNMI)
		fmt.Fprintf(file, "    Best NMI: %.6f (level %d)\n", bestNMI, bestLevel)
		fmt.Fprintf(file, "    Worst NMI: %.6f (level %d)\n", worstNMI, worstLevel)
	}
	fmt.Fprintf(file, "\n")
	
	// Quality metrics
	if result.MaterializationResult.LouvainResult != nil {
		fmt.Fprintf(file, "Quality Metrics:\n")
		fmt.Fprintf(file, "  Materialization+Louvain Modularity: %.6f\n", result.MaterializationResult.LouvainResult.Modularity)
		if len(result.NMIScores) > 0 {
			totalNMI := 0.0
			for _, nmi := range result.NMIScores {
				totalNMI += nmi
			}
			avgNMI := totalNMI / float64(len(result.NMIScores))
			fmt.Fprintf(file, "  Average NMI Agreement: %.6f\n", avgNMI)
		}
		fmt.Fprintf(file, "\n")
	}
	
	// Detailed level analysis
	fmt.Fprintf(file, "Detailed Level Analysis:\n")
	for level := 0; level <= max(result.MaterializationPartition.MaxLevel, result.SCARPartition.MaxLevel); level++ {
		matLevel, matExists := result.MaterializationPartition.Levels[level]
		scarLevel, scarExists := result.SCARPartition.Levels[level]
		
		if !matExists && !scarExists {
			continue
		}
		
		fmt.Fprintf(file, "  Level %d:\n", level)
		
		if matExists {
			matNodes := 0
			for _, nodes := range matLevel {
				matNodes += len(nodes)
			}
			fmt.Fprintf(file, "    Materialization+Louvain: %d communities, %d nodes\n", len(matLevel), matNodes)
		} else {
			fmt.Fprintf(file, "    Materialization+Louvain: No data\n")
		}
		
		if scarExists {
			scarNodes := 0
			for _, nodes := range scarLevel {
				scarNodes += len(nodes)
			}
			fmt.Fprintf(file, "    SCAR: %d communities, %d nodes\n", len(scarLevel), scarNodes)
		} else {
			fmt.Fprintf(file, "    SCAR: No data\n")
		}
		
		if nmi, hasNMI := result.NMIScores[level]; hasNMI {
			fmt.Fprintf(file, "    NMI Score: %.6f\n", nmi)
		} else {
			fmt.Fprintf(file, "    NMI Score: Not available\n")
		}
		fmt.Fprintf(file, "\n")
	}
	
	// Overall interpretation
	fmt.Fprintf(file, "Overall Interpretation:\n")
	if len(result.NMIScores) == 0 {
		fmt.Fprintf(file, "  No common levels found - algorithms produced different hierarchical structures\n")
	} else {
		totalNMI := 0.0
		for _, nmi := range result.NMIScores {
			totalNMI += nmi
		}
		avgNMI := totalNMI / float64(len(result.NMIScores))
		
		if avgNMI >= 0.8 {
			fmt.Fprintf(file, "  HIGH AGREEMENT: Both methods found very similar hierarchical structures\n")
		} else if avgNMI >= 0.6 {
			fmt.Fprintf(file, "  MODERATE AGREEMENT: Both methods found similar hierarchical structures\n")
		} else if avgNMI >= 0.4 {
			fmt.Fprintf(file, "  LOW AGREEMENT: Methods found different hierarchical structures\n")
		} else {
			fmt.Fprintf(file, "  POOR AGREEMENT: Methods found very different hierarchical structures\n")
		}
	}
	
	if result.SCARTime > 0 && result.MaterializationTime > 0 {
		speedup := float64(result.MaterializationTime) / float64(result.SCARTime)
		if speedup > 2.0 {
			fmt.Fprintf(file, "  PERFORMANCE: SCAR shows significant speedup (%.1fx)\n", speedup)
		} else if speedup > 1.2 {
			fmt.Fprintf(file, "  PERFORMANCE: SCAR shows moderate speedup (%.1fx)\n", speedup)
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

// RunSketchLouvain executes the SCAR sketch-based Louvain pipeline
func RunSketchLouvain(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*PipelineResult, error) {
	startTime := time.Now()
	
	if config.Verbose {
		fmt.Println("=== Running SCAR Sketch-based Louvain Pipeline ===")
	}
	
	// Step 1: Configure SCAR with input files
	if config.Verbose {
		fmt.Println("Step 1: Configuring SCAR engine...")
	}
	
	// Create a copy of SCAR config and set file paths
	scarConfig := config.SCARConfig
	scarConfig.GraphFile = graphFile
	scarConfig.PropertyFile = propertiesFile
	scarConfig.PathFile = pathFile
	scarConfig.Prefix = filepath.Join(config.OutputDir, config.OutputPrefix)
	scarConfig.NumWorkers = 4
	if config.Verbose {
		fmt.Printf("  Graph file: %s\n", graphFile)
		fmt.Printf("  Properties file: %s\n", propertiesFile)
		fmt.Printf("  Path file: %s\n", pathFile)
		fmt.Printf("  SCAR parameters: k=%d, nk=%d, threshold=%.3f\n", 
			scarConfig.K, scarConfig.NK, scarConfig.Threshold)
		fmt.Printf("  Sketch output: %t\n", scarConfig.SketchOutput)
	}
	
	// Step 2: Create output directory
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Step 3: Run SCAR engine
	if config.Verbose {
		fmt.Println("Step 2: Running SCAR sketch-based Louvain...")
	}
	
	scarStart := time.Now()
	
	engine := scar.NewSketchLouvainEngine(scarConfig)
	err := engine.RunLouvain()
	if err != nil {
		return nil, fmt.Errorf("SCAR sketch Louvain failed: %w", err)
	}
	
	scarTime := time.Since(scarStart)
	totalTime := time.Since(startTime)
	
	if config.Verbose {
		fmt.Printf("  SCAR completed in %v\n", scarTime)
		fmt.Println("Step 3: Writing SCAR summary...")
	}
	
	// Step 4: Write pipeline summary (SCAR writes its own output files)
	if err := writeSCARSummary(&scarConfig, config, totalTime); err != nil {
		return nil, fmt.Errorf("failed to write SCAR summary: %w", err)
	}
	
	// Create final result
	result := &PipelineResult{
		PipelineType:   SketchLouvain,
		TotalRuntimeMS: totalTime.Milliseconds(),
		SCARSuccess:    true,
		SCARConfig:     &scarConfig,
	}
	
	if config.Verbose {
		fmt.Println("=== SCAR Sketch Louvain Pipeline Complete ===")
		fmt.Printf("Total runtime: %v\n", totalTime)
		fmt.Printf("SCAR execution: %v\n", scarTime)
		if scarConfig.SketchOutput {
			fmt.Println("Generated SCAR hierarchy files for PPRViz integration")
		}
	}
	
	return result, nil
}

// convertHomogeneousToNormalized converts materialization output to Louvain input format
func convertHomogeneousToNormalized(hgraph *materialization.HomogeneousGraph) (*louvain.NormalizedGraph, *louvain.GraphParser, error) {
	if len(hgraph.Nodes) == 0 {
		return nil, nil, fmt.Errorf("empty homogeneous graph")
	}
	
	// // ========== BEFORE: INPUT HOMOGENEOUS GRAPH ==========
	// fmt.Println("\n=== BEFORE CONVERSION: HOMOGENEOUS GRAPH ===")
	// fmt.Printf("Nodes: %d\n", len(hgraph.Nodes))
	// fmt.Printf("Edges: %d\n", len(hgraph.Edges))
	
	// // Show all nodes
	// fmt.Println("Nodes:")
	// for nodeID, node := range hgraph.Nodes {
	// 	fmt.Printf("  %s: degree=%d\n", nodeID, node.Degree)
	// }
	
	// // Show all edges
	// fmt.Println("Edges:")
	// totalWeight := 0.0
	// for edgeKey, weight := range hgraph.Edges {
	// 	fmt.Printf("  %s -> %s: %.2f\n", edgeKey.From, edgeKey.To, weight)
	// 	totalWeight += weight
	// }
	// fmt.Printf("Total edge weight: %.2f\n", totalWeight)
	
	// ========== CONVERSION PROCESS ==========
	parser := louvain.NewGraphParser()
	
	// Create ordered list of node IDs
	nodeList := make([]string, 0, len(hgraph.Nodes))
	for nodeID := range hgraph.Nodes {
		nodeList = append(nodeList, nodeID)
	}

	// ✅ Use intelligent sorting (same as parser.go)
	allIntegers := allNodesAreIntegers(nodeList)
	if allIntegers {
		// Sort numerically: 1 < 2 < 5 < 10
		sort.Slice(nodeList, func(i, j int) bool {
			a, _ := strconv.Atoi(nodeList[i])
			b, _ := strconv.Atoi(nodeList[j])
			return a < b
		})
	} else {
		// Sort lexicographically: "1" < "10" < "2" < "5"
		sort.Strings(nodeList)
	}
	// fmt.Println("\n=== CONVERSION PROCESS ===")
	// fmt.Println("Node ID mapping:")
	// for i, originalID := range nodeList {
	// 	fmt.Printf("  %s -> %d\n", originalID, i)
	// }
	
	// Create normalized graph
	normalizedGraph := louvain.NewNormalizedGraph(len(nodeList))
	
	// Build node ID mappings and set weights
	for i, originalID := range nodeList {
		parser.OriginalToNormalized[originalID] = i
		parser.NormalizedToOriginal[i] = originalID
		
		// Set node weights (using default for now)
		if _, exists := hgraph.Nodes[originalID]; exists {
			normalizedGraph.Weights[i] = 1.0 // Default weight
		}
	}
	parser.NumNodes = len(nodeList)
	
	// ✅ Convert edges with deduplication to prevent double counting
	// fmt.Println("\nEdge conversion:")
	processedEdges := make(map[string]bool)
	edgeCount := 0
	
	for edgeKey, weight := range hgraph.Edges {
		fromNormalized, fromExists := parser.OriginalToNormalized[edgeKey.From]
		toNormalized, toExists := parser.OriginalToNormalized[edgeKey.To]
		
		if !fromExists || !toExists {
			return nil, nil, fmt.Errorf("edge references unknown nodes: %s -> %s", edgeKey.From, edgeKey.To)
		}
		
		// ✅ Create canonical edge ID (smaller index first) to avoid duplicates
		var canonicalID string
		if fromNormalized <= toNormalized {
			canonicalID = fmt.Sprintf("%d-%d", fromNormalized, toNormalized)
		} else {
			canonicalID = fmt.Sprintf("%d-%d", toNormalized, fromNormalized)
		}
		
		// ✅ Only process each undirected edge once
		if !processedEdges[canonicalID] {
			// fmt.Printf("  %s->%s (%.2f) becomes %d->%d (%.2f) [ADDED]\n", 
			// 	edgeKey.From, edgeKey.To, weight, fromNormalized, toNormalized, weight)
			
			normalizedGraph.AddEdge(fromNormalized, toNormalized, weight)
			processedEdges[canonicalID] = true
			edgeCount++
		} else {
			// fmt.Printf("  %s->%s (%.2f) becomes %d->%d (%.2f) [SKIPPED - duplicate]\n", 
			// 	edgeKey.From, edgeKey.To, weight, fromNormalized, toNormalized, weight)
		}
	}
	
	// fmt.Printf("Processed %d unique edges out of %d total edges\n", edgeCount, len(hgraph.Edges))
	
	// // ========== AFTER: NORMALIZED GRAPH ==========
	// fmt.Println("\n=== AFTER CONVERSION: NORMALIZED GRAPH ===")
	// fmt.Printf("NumNodes: %d\n", normalizedGraph.NumNodes)
	// fmt.Printf("TotalWeight: %.2f\n", normalizedGraph.TotalWeight)
	
	// // Show node degrees and weights
	// fmt.Println("Node degrees and weights:")
	// for i := 0; i < normalizedGraph.NumNodes; i++ {
	// 	originalID := parser.NormalizedToOriginal[i]
	// 	fmt.Printf("  Node %d (%s): degree=%.2f, weight=%.2f\n", 
	// 		i, originalID, normalizedGraph.Degrees[i], normalizedGraph.Weights[i])
	// }
	
	// // Show adjacency lists
	// fmt.Println("Adjacency lists:")
	// for i := 0; i < normalizedGraph.NumNodes; i++ {
	// 	originalID := parser.NormalizedToOriginal[i]
	// 	neighbors := normalizedGraph.Adjacency[i]
	// 	weights := normalizedGraph.EdgeWeights[i]
		
	// 	fmt.Printf("  Node %d (%s): ", i, originalID)
	// 	for j, neighbor := range neighbors {
	// 		neighborOriginalID := parser.NormalizedToOriginal[neighbor]
	// 		fmt.Printf("(%d/%s, %.2f) ", neighbor, neighborOriginalID, weights[j])
	// 	}
	// 	fmt.Println()
		
	// 	// Check for duplicates in adjacency list
	// 	seen := make(map[int]int)
	// 	for _, neighbor := range neighbors {
	// 		seen[neighbor]++
	// 	}
	// 	for neighbor, count := range seen {
	// 		if count > 1 {
	// 			neighborOriginalID := parser.NormalizedToOriginal[neighbor]
	// 			fmt.Printf("    ❌ DUPLICATE: neighbor %d (%s) appears %d times\n", 
	// 				neighbor, neighborOriginalID, count)
	// 		}
	// 	}
	// }
	
	// // Show all edges in normalized format
	// fmt.Println("All edges (reconstructed from adjacency):")
	// processedPairs := make(map[string]bool)
	// for i := 0; i < normalizedGraph.NumNodes; i++ {
	// 	for j, neighbor := range normalizedGraph.Adjacency[i] {
	// 		// Avoid showing each edge twice
	// 		var pairKey string
	// 		if i <= neighbor {
	// 			pairKey = fmt.Sprintf("%d-%d", i, neighbor)
	// 		} else {
	// 			pairKey = fmt.Sprintf("%d-%d", neighbor, i)
	// 		}
			
	// 		if !processedPairs[pairKey] {
	// 			weight := normalizedGraph.EdgeWeights[i][j]
	// 			iOriginal := parser.NormalizedToOriginal[i]
	// 			neighborOriginal := parser.NormalizedToOriginal[neighbor]
	// 			fmt.Printf("  %d (%s) <-> %d (%s): %.2f\n", 
	// 				i, iOriginal, neighbor, neighborOriginal, weight)
	// 			processedPairs[pairKey] = true
	// 		}
	// 	}
	// }
	
	// Validate the converted graph
	if err := normalizedGraph.Validate(); err != nil {
		fmt.Printf("❌ VALIDATION FAILED: %v\n", err)
		return nil, nil, fmt.Errorf("converted graph validation failed: %w", err)
	} else {
		fmt.Println("✅ VALIDATION PASSED")
	}

	fmt.Println("=== CONVERSION COMPLETE ===\n")
	return normalizedGraph, parser, nil
}

// writeLouvainOutputs generates Louvain output files
func writeLouvainOutputs(result *louvain.LouvainResult, parser *louvain.GraphParser, materializedGraph *materialization.HomogeneousGraph, config *PipelineConfig) error {
	// Create output directory
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// ✅ NEW: Write materialized graph files FIRST
	if materializedGraph != nil {
		if config.Verbose {
			fmt.Println("  Writing materialized graph files...")
		}
		
		// Write edge list (simple format for other tools)
		edgeListPath := filepath.Join(config.OutputDir, config.OutputPrefix+"_materialized.edgelist")
		if err := materialization.SaveAsSimpleEdgeList(materializedGraph, edgeListPath); err != nil {
			return fmt.Errorf("failed to write materialized edgelist: %w", err)
		}
		
		if config.Verbose {
			fmt.Printf("    Edgelist: %s\n", edgeListPath)
		}
	}
	
	// Write Louvain results (existing functionality)
	if config.Verbose {
		fmt.Println("  Writing Louvain clustering results...")
	}
	writer := louvain.NewFileWriter()
	if err := writer.WriteAll(result, parser, config.OutputDir, config.OutputPrefix); err != nil {
		return fmt.Errorf("failed to write Louvain results: %w", err)
	}
	
	// Write pipeline summary
	summaryPath := filepath.Join(config.OutputDir, config.OutputPrefix+"_summary.txt")
	if err := writeLouvainSummary(result, summaryPath); err != nil {
		return fmt.Errorf("failed to write summary: %w", err)
	}
	
	return nil
}

// writeLouvainSummary creates a summary file for Louvain results
func writeLouvainSummary(result *louvain.LouvainResult, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "=== Materialization + Louvain Pipeline Summary ===\n\n")
	fmt.Fprintf(file, "Louvain Community Detection Results:\n")
	fmt.Fprintf(file, "  Final Modularity: %.6f\n", result.Modularity)
	fmt.Fprintf(file, "  Number of Hierarchy Levels: %d\n", result.NumLevels)
	
	if result.Statistics.RuntimeMS > 0 {
		fmt.Fprintf(file, "  Runtime: %d ms\n", result.Statistics.RuntimeMS)
	}
	if result.Statistics.TotalIterations > 0 {
		fmt.Fprintf(file, "  Total Iterations: %d\n", result.Statistics.TotalIterations)
	}
	if result.Statistics.TotalMoves > 0 {
		fmt.Fprintf(file, "  Total Node Moves: %d\n", result.Statistics.TotalMoves)
	}
	
	if len(result.Levels) > 0 {
		finalLevel := result.Levels[len(result.Levels)-1]
		fmt.Fprintf(file, "\nCommunity Structure:\n")
		fmt.Fprintf(file, "  Number of Communities: %d\n", finalLevel.NumCommunities)
		
		if len(finalLevel.Communities) > 0 {
			sizes := make([]int, 0, len(finalLevel.Communities))
			for _, nodes := range finalLevel.Communities {
				sizes = append(sizes, len(nodes))
			}
			fmt.Fprintf(file, "  Community Sizes: %v\n", sizes)
		}
		
		fmt.Fprintf(file, "\nHierarchy Breakdown:\n")
		for i, level := range result.Levels {
			fmt.Fprintf(file, "  Level %d: %d communities, modularity %.6f\n", 
				i, level.NumCommunities, level.Modularity)
		}
	}
	
	return nil
}

// writeSCARSummary creates a summary file for SCAR results
func writeSCARSummary(scarConfig *scar.SCARConfig, config *PipelineConfig, totalTime time.Duration) error {
	summaryPath := filepath.Join(config.OutputDir, config.OutputPrefix+"_scar_summary.txt")
	
	file, err := os.Create(summaryPath)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "=== SCAR Sketch-based Louvain Pipeline Summary ===\n\n")
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

// Example usage and main function
func main() {
	// Define command line flags
	var (
		// SCAR-specific flags
		scarK         = flag.Int64("scar-k", 64, "SCAR sketch size (default: 64)")
		scarNK        = flag.Int64("scar-nk", 4, "SCAR number of sketch layers (default: 4)")
		scarThreshold = flag.Float64("scar-threshold", 0.5, "SCAR threshold (default: 0.5)")
		scarSketch    = flag.Bool("scar-sketch-output", true, "SCAR generate hierarchy files for PPRViz (default: true)")
		
		// Materialization flags
		matMaxInstances = flag.Int("mat-max-instances", 1000000, "Materialization max instances (default: 1000000)")
		matSymmetric    = flag.Bool("mat-symmetric", true, "Materialization symmetric edges (default: true)")
		
		// Louvain flags
		louvainMaxIter   = flag.Int("louvain-max-iter", 10, "Louvain max iterations (default: 10)")
		louvainMinMod    = flag.Float64("louvain-min-mod", 0.001, "Louvain min modularity improvement (default: 0.001)")
		louvainSeed      = flag.Int64("louvain-seed", 42, "Louvain random seed (default: 42)")
		
		// General flags
		verbose = flag.Bool("verbose", true, "Verbose output (default: true)")
		prefix  = flag.String("prefix", "communities", "Output file prefix (default: communities)")
	)
	
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <pipeline_type> <graph_file> <properties_file> <path_file> [output_dir]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Pipeline Types:\n")
		fmt.Fprintf(os.Stderr, "  mat_louvain    - Materialization + Louvain pipeline\n")
		fmt.Fprintf(os.Stderr, "  sketch_louvain - SCAR sketch-based Louvain pipeline\n")
		fmt.Fprintf(os.Stderr, "  compare        - Run both pipelines and compare results using NMI\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  %s mat_louvain graph.txt properties.txt path.txt results/\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s sketch_louvain graph.txt properties.txt path.txt results/\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s compare graph.txt properties.txt path.txt results/\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -scar-k=128 -scar-nk=6 sketch_louvain graph.txt props.txt path.txt results/\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -louvain-max-iter=20 -mat-max-instances=500000 mat_louvain graph.txt props.txt path.txt results/\n\n", os.Args[0])
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
	
	outputDir := "pipeline_output"
	if len(args) > 4 {
		outputDir = args[4]
	}
	
	// Create configuration
	config := NewPipelineConfig()
	config.OutputDir = outputDir
	config.OutputPrefix = *prefix
	config.Verbose = *verbose
	
	var result *PipelineResult
	var compResult *ComparisonResult
	var err error
	
	switch pipelineType {
	case "mat_louvain":
		// Configure materialization + Louvain with flags
		config.MaterializationConfig.Aggregation.Strategy = materialization.Average
		config.MaterializationConfig.Aggregation.Symmetric = *matSymmetric
		config.MaterializationConfig.Traversal.MaxInstances = *matMaxInstances
		
		config.LouvainConfig.MaxIterations = *louvainMaxIter
		config.LouvainConfig.MinModularity = *louvainMinMod
		config.LouvainConfig.RandomSeed = *louvainSeed
		
		if config.Verbose {
			fmt.Printf("Materialization config: max_instances=%d, symmetric=%t\n", *matMaxInstances, *matSymmetric)
			fmt.Printf("Louvain config: max_iter=%d, min_mod=%.6f, seed=%d\n", *louvainMaxIter, *louvainMinMod, *louvainSeed)
		}
		
		result, err = RunMaterializationLouvain(graphFile, propertiesFile, pathFile, config)
		
	case "sketch_louvain":
		// Configure SCAR with flags
		config.SCARConfig.K = *scarK
		config.SCARConfig.NK = *scarNK
		config.SCARConfig.Threshold = *scarThreshold
		config.SCARConfig.UseLouvain = true
		config.SCARConfig.SketchOutput = *scarSketch
		
		if config.Verbose {
			fmt.Printf("SCAR config: k=%d, nk=%d, threshold=%.3f, sketch_output=%t\n", 
				*scarK, *scarNK, *scarThreshold, *scarSketch)
		}
		
		result, err = RunSketchLouvain(graphFile, propertiesFile, pathFile, config)
		
	case "compare":
		// Configure both pipelines for comparison
		config.MaterializationConfig.Aggregation.Strategy = materialization.Average
		config.MaterializationConfig.Aggregation.Symmetric = *matSymmetric
		config.MaterializationConfig.Traversal.MaxInstances = *matMaxInstances
		
		config.LouvainConfig.MaxIterations = *louvainMaxIter
		config.LouvainConfig.MinModularity = *louvainMinMod
		config.LouvainConfig.RandomSeed = *louvainSeed
		
		config.SCARConfig.K = *scarK
		config.SCARConfig.NK = *scarNK
		config.SCARConfig.Threshold = *scarThreshold
		config.SCARConfig.UseLouvain = true
		config.SCARConfig.SketchOutput = *scarSketch
		
		if config.Verbose {
			fmt.Printf("Comparison mode: Running both pipelines with same input\n")
			fmt.Printf("Materialization config: max_instances=%d, symmetric=%t\n", *matMaxInstances, *matSymmetric)
			fmt.Printf("Louvain config: max_iter=%d, min_mod=%.6f, seed=%d\n", *louvainMaxIter, *louvainMinMod, *louvainSeed)
			fmt.Printf("SCAR config: k=%d, nk=%d, threshold=%.3f, sketch_output=%t\n", 
				*scarK, *scarNK, *scarThreshold, *scarSketch)
		}
		
		compResult, err = RunComparison(graphFile, propertiesFile, pathFile, config)
		
	default:
		log.Fatalf("Unknown pipeline type: %s. Use 'mat_louvain', 'sketch_louvain', or 'compare'", pipelineType)
	}
	
	if err != nil {
		log.Fatalf("Pipeline failed: %v", err)
	}
	
	// Print final summary
	// fmt.Println("\n=== Final Results ===")
	
	if compResult != nil {
		// // Comparison results
		// fmt.Printf("Hierarchical pipeline comparison completed successfully!\n")
		// fmt.Printf("Total comparison time: %d ms\n", compResult.TotalComparisonTime)
		// fmt.Printf("Materialization+Louvain time: %d ms\n", compResult.MaterializationTime)
		// fmt.Printf("SCAR time: %d ms\n", compResult.SCARTime)
		
		// if compResult.SCARTime > 0 {
		// 	speedup := float64(compResult.MaterializationTime) / float64(compResult.SCARTime)
		// 	fmt.Printf("SCAR Speedup: %.2fx\n", speedup)
		// }
		
		// fmt.Printf("Common levels analyzed: %d\n", len(compResult.CommonLevels))
		
		// if len(compResult.NMIScores) > 0 {
		// 	// Calculate and display NMI statistics
		// 	totalNMI := 0.0
		// 	bestNMI := 0.0
		// 	bestLevel := -1
		// 	for level, nmi := range compResult.NMIScores {
		// 		totalNMI += nmi
		// 		if nmi > bestNMI {
		// 			bestNMI = nmi
		// 			bestLevel = level
		// 		}
		// 	}
		// 	avgNMI := totalNMI / float64(len(compResult.NMIScores))
			
		// 	fmt.Printf("Average NMI Score: %.6f\n", avgNMI)
		// 	fmt.Printf("Best NMI Score: %.6f (level %d)\n", bestNMI, bestLevel)
			
		// 	// Display level-by-level NMI
		// 	fmt.Printf("Level-by-level NMI:\n")
		// 	for level := 0; level <= max(compResult.MaterializationPartition.MaxLevel, compResult.SCARPartition.MaxLevel); level++ {
		// 		if nmi, exists := compResult.NMIScores[level]; exists {
		// 			fmt.Printf("  Level %d: %.6f\n", level, nmi)
		// 		}
		// 	}
			
		// 	// Overall interpretation
		// 	if avgNMI >= 0.8 {
		// 		fmt.Printf("Result: HIGH AGREEMENT between methods\n")
		// 	} else if avgNMI >= 0.6 {
		// 		fmt.Printf("Result: MODERATE AGREEMENT between methods\n")
		// 	} else if avgNMI >= 0.4 {
		// 		fmt.Printf("Result: LOW AGREEMENT between methods\n")
		// 	} else {
		// 		fmt.Printf("Result: POOR AGREEMENT between methods\n")
		// 	}
		// } else {
		// 	fmt.Printf("No common levels found for comparison\n")
		// }
		
		// if compResult.MaterializationResult.LouvainResult != nil {
		// 	fmt.Printf("Materialization+Louvain modularity: %.6f\n", compResult.MaterializationResult.LouvainResult.Modularity)
		// }
		
		// fmt.Printf("Detailed hierarchical comparison report: %s/hierarchical_comparison_report.txt\n", config.OutputDir)
		
	} else {
		// Single pipeline results
		fmt.Printf("Pipeline completed successfully!\n")
		fmt.Printf("Pipeline type: %s\n", pipelineType)
		fmt.Printf("Total runtime: %d ms\n", result.TotalRuntimeMS)
		
		if result.PipelineType == MaterializationLouvain && result.LouvainResult != nil {
			fmt.Printf("Final modularity: %.6f\n", result.LouvainResult.Modularity)
			if len(result.LouvainResult.Levels) > 0 {
				finalLevel := result.LouvainResult.Levels[len(result.LouvainResult.Levels)-1]
				fmt.Printf("Communities found: %d\n", finalLevel.NumCommunities)
			}
		} else if result.PipelineType == SketchLouvain {
			fmt.Printf("SCAR sketch-based clustering completed successfully\n")
			fmt.Printf("Generated hierarchy files for PPRViz integration\n")
		}
		
		fmt.Printf("Output files written to: %s/\n", config.OutputDir)
	}
}

// hierarchicalMutualInformation averages the NMI scores across all common levels.
func hierarchicalMutualInformation(nmiScores map[int]float64) float64 {
    if len(nmiScores) == 0 {
        return 0
    }
    sum := 0.0
    for _, v := range nmiScores {
        sum += v
    }
    return sum / float64(len(nmiScores))
}

// CopheneticCorrelation builds, for each pair of nodes, the level at which
// they first coalesce in each hierarchy, then returns the Pearson correlation
// between those two “height” arrays.
func CopheneticCorrelation(p1, p2 *HierarchicalPartition) (float64, error) {
    // 1) Build a map from “i,j”→level for each partition
    build := func(p *HierarchicalPartition) map[string]float64 {
        m := make(map[string]float64)
        for lvl, comms := range p.Levels {
            for _, nodes := range comms {
                for i := 0; i < len(nodes); i++ {
                    for j := i + 1; j < len(nodes); j++ {
                        // canonical key
                        a, b := nodes[i], nodes[j]
                        key := a + "," + b
                        if _, seen := m[key]; !seen {
                            m[key] = float64(lvl)
                        }
                    }
                }
            }
        }
        return m
    }

    h1 := build(p1)
    h2 := build(p2)

    // 2) Collect matched heights
    x, y := make([]float64, 0, len(h1)), make([]float64, 0, len(h1))
    for key, v1 := range h1 {
        if v2, ok := h2[key]; ok {
            x = append(x, v1)
            y = append(y, v2)
        }
    }
    if len(x) == 0 {
        return 0, fmt.Errorf("no common node‐pairs found")
    }

    // 3) Compute Pearson correlation
    corr := stat.Correlation(x, y, nil)
    return corr, nil
}


// Add this function near the top of your main file
func allNodesAreIntegers(nodes []string) bool {
	for _, node := range nodes {
		if _, err := strconv.Atoi(node); err != nil {
			return false
		}
	}
	return true
}
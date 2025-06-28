package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
)

// IntegratedPipeline runs materialization followed by Louvain clustering
// without creating intermediary files
type IntegratedPipeline struct {
	// Materialization config
	MaterializationConfig materialization.MaterializationConfig
	
	// Louvain config
	LouvainConfig louvain.LouvainConfig
	
	// Pipeline options
	Verbose bool
	OutputDir string
	OutputPrefix string
}

// PipelineResult contains the complete pipeline output
type PipelineResult struct {
	// Materialization results
	MaterializedGraph *materialization.HomogeneousGraph
	
	// Louvain results
	LouvainResult *louvain.LouvainResult
	
	// Combined stats
	TotalRuntimeMS int64
}

// NewIntegratedPipeline creates a new pipeline with default configurations
func NewIntegratedPipeline() *IntegratedPipeline {
	return &IntegratedPipeline{
		MaterializationConfig: materialization.DefaultMaterializationConfig(),
		LouvainConfig: louvain.DefaultLouvainConfig(),
		Verbose: false,
		OutputDir: "output",
		OutputPrefix: "pipeline_result",
	}
}

// RunPipeline executes the complete materialization + Louvain pipeline
func (ip *IntegratedPipeline) RunPipeline(graphFile, propertiesFile, pathFile string) (*PipelineResult, error) {
	startTime := time.Now()
	
	if ip.Verbose {
		fmt.Println("=== Starting Integrated Materialization + Louvain Pipeline ===")
	}
	
	// Step 1: Parse SCAR input for materialization
	if ip.Verbose {
		fmt.Println("Step 1: Parsing input files for materialization...")
	}
	
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
	if err != nil {
		return nil, fmt.Errorf("failed to parse SCAR input: %w", err)
	}
	
	if ip.Verbose {
		fmt.Printf("  Loaded graph with %d nodes\n", len(graph.Nodes))
	}
	
	// Step 2: Run materialization
	if ip.Verbose {
		fmt.Println("Step 2: Running graph materialization...")
	}
	
	materializationStart := time.Now()
	
	// Setup progress callback for materialization if available
	var materializationProgressCb func(int, int, string)
	if ip.Verbose {
		materializationProgressCb = func(current, total int, message string) {
			fmt.Printf("  Materialization progress: %d/%d - %s\n", current, total, message)
		}
	}
	
	engine := materialization.NewMaterializationEngine(graph, metaPath, ip.MaterializationConfig, materializationProgressCb)
	materializationResult, err := engine.Materialize()
	if err != nil {
		return nil, fmt.Errorf("materialization failed: %w", err)
	}
	
	materializationTime := time.Since(materializationStart)
	
	if ip.Verbose {
		fmt.Printf("  Materialization completed in %v\n", materializationTime)
		fmt.Printf("  Materialized graph has %d nodes and %d edges\n", 
			len(materializationResult.HomogeneousGraph.Nodes),
			len(materializationResult.HomogeneousGraph.Edges))
	}
	
	// Step 3: Convert HomogeneousGraph to NormalizedGraph for Louvain
	if ip.Verbose {
		fmt.Println("Step 3: Converting graph format for Louvain...")
	}
	
	normalizedGraph, graphParser, err := ip.convertHomogeneousToNormalized(materializationResult.HomogeneousGraph)
	if err != nil {
		return nil, fmt.Errorf("graph conversion failed: %w", err)
	}
	
	if ip.Verbose {
		fmt.Printf("  Converted to normalized graph with %d nodes\n", normalizedGraph.NumNodes)
		fmt.Printf("  Total edge weight: %.2f\n", normalizedGraph.TotalWeight)
	}
	
	// Step 4: Run Louvain clustering
	if ip.Verbose {
		fmt.Println("Step 4: Running Louvain community detection...")
	}
	
	louvainStart := time.Now()
	
	// Setup progress callback for Louvain
	if ip.Verbose {
		ip.LouvainConfig.Verbose = true
		ip.LouvainConfig.ProgressCallback = func(level, iteration int, message string) {
			fmt.Printf("  Louvain [L%d I%d]: %s\n", level, iteration, message)
		}
	}
	
	louvainResult, err := louvain.RunLouvain(normalizedGraph, ip.LouvainConfig)
	if err != nil {
		return nil, fmt.Errorf("Louvain clustering failed: %w", err)
	}
	
	louvainTime := time.Since(louvainStart)
	louvainResult.Parser = graphParser // Attach parser for output generation
	
	if ip.Verbose {
		fmt.Printf("  Louvain completed in %v\n", louvainTime)
		fmt.Printf("  Final modularity: %.6f\n", louvainResult.Modularity)
		finalLevel := louvainResult.Levels[len(louvainResult.Levels)-1]
		fmt.Printf("  Number of communities: %d\n", finalLevel.NumCommunities)
		fmt.Printf("  Hierarchy levels: %d\n", louvainResult.NumLevels)
	}
	
	// Step 5: Generate output files
	if ip.Verbose {
		fmt.Println("Step 5: Writing output files...")
	}
	
	if err := ip.writeOutputs(louvainResult, graphParser); err != nil {
		return nil, fmt.Errorf("output generation failed: %w", err)
	}
	
	totalTime := time.Since(startTime)
	
	// Create final result
	result := &PipelineResult{
		MaterializedGraph: materializationResult.HomogeneousGraph,
		LouvainResult: louvainResult,
		TotalRuntimeMS: totalTime.Milliseconds(),
	}
	
	if ip.Verbose {
		fmt.Println("=== Pipeline Complete ===")
		fmt.Printf("Total runtime: %v\n", totalTime)
		fmt.Printf("Materialization: %v, Louvain: %v\n", materializationTime, louvainTime)
		fmt.Printf("Final modularity: %.6f\n", result.LouvainResult.Modularity)
	}
	
	return result, nil
}

// convertHomogeneousToNormalized converts materialization output to Louvain input format
// This is the critical bridge function between the two systems
func (ip *IntegratedPipeline) convertHomogeneousToNormalized(hgraph *materialization.HomogeneousGraph) (*louvain.NormalizedGraph, *louvain.GraphParser, error) {
	if len(hgraph.Nodes) == 0 {
		return nil, nil, fmt.Errorf("empty homogeneous graph")
	}
	
	// Create parser for ID mapping
	parser := louvain.NewGraphParser()
	
	// Create ordered list of node IDs from the graph
	nodeList := make([]string, 0, len(hgraph.Nodes))
	for nodeID := range hgraph.Nodes {
		nodeList = append(nodeList, nodeID)
	}
	
	// Create normalized graph
	normalizedGraph := louvain.NewNormalizedGraph(len(nodeList))
	
	// Build node ID mappings
	for i, originalID := range nodeList {
		parser.OriginalToNormalized[originalID] = i
		parser.NormalizedToOriginal[i] = originalID
	}
	parser.NumNodes = len(nodeList)
	
	// Convert edges - iterate through the edge map
	for edgeKey, weight := range hgraph.Edges {
		fromNormalized, fromExists := parser.OriginalToNormalized[edgeKey.From]
		toNormalized, toExists := parser.OriginalToNormalized[edgeKey.To]
		
		if !fromExists || !toExists {
			return nil, nil, fmt.Errorf("edge references unknown nodes: %s -> %s", edgeKey.From, edgeKey.To)
		}
		
		normalizedGraph.AddEdge(fromNormalized, toNormalized, weight)
	}
	
	// Validate the converted graph
	if err := normalizedGraph.Validate(); err != nil {
		return nil, nil, fmt.Errorf("converted graph validation failed: %w", err)
	}
	
	return normalizedGraph, parser, nil
}

// writeOutputs generates all output files
func (ip *IntegratedPipeline) writeOutputs(result *louvain.LouvainResult, parser *louvain.GraphParser) error {
	// Create output directory
	if err := os.MkdirAll(ip.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Write Louvain results using the documented API
	writer := louvain.NewFileWriter()
	if err := writer.WriteAll(result, parser, ip.OutputDir, ip.OutputPrefix); err != nil {
		return fmt.Errorf("failed to write Louvain results: %w", err)
	}
	
	// Write simple pipeline summary
	summaryPath := filepath.Join(ip.OutputDir, ip.OutputPrefix+"_summary.txt")
	if err := ip.writePipelineSummary(result, summaryPath); err != nil {
		return fmt.Errorf("failed to write summary: %w", err)
	}
	
	return nil
}

// writePipelineSummary creates a summary file with pipeline statistics
func (ip *IntegratedPipeline) writePipelineSummary(result *louvain.LouvainResult, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "=== Integrated Materialization + Louvain Pipeline Summary ===\n\n")
	
	// Louvain results - using only documented fields
	fmt.Fprintf(file, "Louvain Community Detection Results:\n")
	fmt.Fprintf(file, "  Final Modularity: %.6f\n", result.Modularity)
	fmt.Fprintf(file, "  Number of Hierarchy Levels: %d\n", result.NumLevels)
	
	// Check if Statistics field exists and has documented fields
	if result.Statistics.RuntimeMS > 0 {
		fmt.Fprintf(file, "  Runtime: %d ms\n", result.Statistics.RuntimeMS)
	}
	if result.Statistics.TotalIterations > 0 {
		fmt.Fprintf(file, "  Total Iterations: %d\n", result.Statistics.TotalIterations)
	}
	if result.Statistics.TotalMoves > 0 {
		fmt.Fprintf(file, "  Total Node Moves: %d\n", result.Statistics.TotalMoves)
	}
	
	// Community structure
	if len(result.Levels) > 0 {
		finalLevel := result.Levels[len(result.Levels)-1]
		fmt.Fprintf(file, "\nCommunity Structure:\n")
		fmt.Fprintf(file, "  Number of Communities: %d\n", finalLevel.NumCommunities)
		
		// Community size distribution
		if len(finalLevel.Communities) > 0 {
			sizes := make([]int, 0, len(finalLevel.Communities))
			for _, nodes := range finalLevel.Communities {
				sizes = append(sizes, len(nodes))
			}
			fmt.Fprintf(file, "  Community Sizes: %v\n", sizes)
		}
		
		// Level-by-level breakdown
		fmt.Fprintf(file, "\nHierarchy Breakdown:\n")
		for i, level := range result.Levels {
			fmt.Fprintf(file, "  Level %d: %d communities, modularity %.6f\n", 
				i, level.NumCommunities, level.Modularity)
		}
	}
	
	return nil
}

// Example usage and main function
func main() {
	// Check command line arguments
	if len(os.Args) < 4 {
		fmt.Fprintf(os.Stderr, "Usage: %s <graph_file> <properties_file> <path_file> [output_dir]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  %s graph.txt properties.txt path.txt results/\n", os.Args[0])
		os.Exit(1)
	}
	
	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]
	
	outputDir := "pipeline_output"
	if len(os.Args) > 4 {
		outputDir = os.Args[4]
	}
	
	// Create and configure pipeline
	pipeline := NewIntegratedPipeline()
	pipeline.Verbose = true
	pipeline.OutputDir = outputDir
	pipeline.OutputPrefix = "communities"
	
	// Configure materialization using documented API
	pipeline.MaterializationConfig.Aggregation.Strategy = materialization.Count
	pipeline.MaterializationConfig.Aggregation.Symmetric = true
	pipeline.MaterializationConfig.Traversal.MaxInstances = 1000000
	
	// Configure Louvain using documented API
	pipeline.LouvainConfig.MaxIterations = 10
	pipeline.LouvainConfig.MinModularity = 0.001
	pipeline.LouvainConfig.RandomSeed = 42 // For reproducible results
	
	// Run the integrated pipeline
	result, err := pipeline.RunPipeline(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Pipeline failed: %v", err)
	}
	
	// Print final summary
	fmt.Println("\n=== Final Results ===")
	fmt.Printf("Pipeline completed successfully!\n")
	fmt.Printf("Total runtime: %d ms\n", result.TotalRuntimeMS)
	fmt.Printf("Final modularity: %.6f\n", result.LouvainResult.Modularity)
	
	if len(result.LouvainResult.Levels) > 0 {
		finalLevel := result.LouvainResult.Levels[len(result.LouvainResult.Levels)-1]
		fmt.Printf("Communities found: %d\n", finalLevel.NumCommunities)
	}
	
	fmt.Printf("Output files written to: %s/\n", pipeline.OutputDir)
	
	// List output files
	fmt.Println("\nGenerated files:")
	fmt.Printf("  %s/%s.mapping - Community assignments\n", pipeline.OutputDir, pipeline.OutputPrefix)
	fmt.Printf("  %s/%s.hierarchy - Hierarchical structure\n", pipeline.OutputDir, pipeline.OutputPrefix)
	fmt.Printf("  %s/%s.root - Top-level communities\n", pipeline.OutputDir, pipeline.OutputPrefix)
	fmt.Printf("  %s/%s.edges - Inter-community edges\n", pipeline.OutputDir, pipeline.OutputPrefix)
	fmt.Printf("  %s/%s_summary.txt - Pipeline summary\n", pipeline.OutputDir, pipeline.OutputPrefix)
}

// Configuration helpers for different use cases - using only documented options

// ConfigureForLargeGraphs optimizes the pipeline for large graphs (>100K nodes)
func (ip *IntegratedPipeline) ConfigureForLargeGraphs() {
	// Materialization optimizations - using documented fields
	ip.MaterializationConfig.Traversal.MaxInstances = 500000
	// Only use fields that are documented in the API
	
	// Louvain optimizations - using documented fields
	ip.LouvainConfig.MaxIterations = 5
	ip.LouvainConfig.MinModularity = 0.0001
	ip.LouvainConfig.ChunkSize = 256
}

// ConfigureForHighQuality optimizes for best community detection quality
func (ip *IntegratedPipeline) ConfigureForHighQuality() {
	// Materialization for quality - using documented options
	ip.MaterializationConfig.Aggregation.Strategy = materialization.Average
	ip.MaterializationConfig.Traversal.AllowCycles = false
	
	// Louvain for quality
	ip.LouvainConfig.MaxIterations = 20
	ip.LouvainConfig.MinModularity = 0.00001
	ip.LouvainConfig.ChunkSize = 32
}

// ConfigureForSpeed optimizes for fastest execution
func (ip *IntegratedPipeline) ConfigureForSpeed() {
	// Materialization for speed
	ip.MaterializationConfig.Traversal.MaxInstances = 100000
	ip.MaterializationConfig.Aggregation.Strategy = materialization.Count
	
	// Louvain for speed
	ip.LouvainConfig.MaxIterations = 3
	ip.LouvainConfig.MinModularity = 0.01
	ip.LouvainConfig.Verbose = false
}
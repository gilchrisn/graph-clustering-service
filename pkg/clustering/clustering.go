// Package clustering provides two approaches for heterogeneous graph clustering:
// 1. Materialization + Louvain
// 2. SCAR (Sketch-based Community detection with Approximated Resistance)
package clustering

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/models"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

// ===== CONFIGURATION STRUCTS =====

// MaterializationClusteringConfig contains all configuration for the materialization + Louvain approach
type MaterializationClusteringConfig struct {
	// Input files
	GraphFile    string `json:"graph_file"`
	MetaPathFile string `json:"meta_path_file"`
	OutputDir    string `json:"output_dir"`

	// Materialization configuration
	AggregationStrategy     materialization.AggregationStrategy     `json:"aggregation_strategy"`     // Count, Sum, Average, Maximum, Minimum
	MetaPathInterpretation  materialization.MetaPathInterpretation  `json:"meta_path_interpretation"` // DirectTraversal, MeetingBased
	Symmetric              bool                                     `json:"symmetric"`                // Force symmetric edges
	MinWeight              float64                                  `json:"min_weight"`               // Filter weak edges
	MaxEdges               int                                      `json:"max_edges"`                // Keep only top-k edges (0 = no limit)
	
	// Traversal configuration
	MaxInstances           int                                      `json:"max_instances"`            // Memory safety limit
	TimeoutSeconds         int                                      `json:"timeout_seconds"`          // Processing timeout
	TraversalParallelism   int                                      `json:"traversal_parallelism"`    // Number of parallel workers
	
	// Louvain configuration
	LouvainMaxIterations   int                                      `json:"louvain_max_iterations"`   // Maximum Louvain iterations
	LouvainMinModularity   float64                                  `json:"louvain_min_modularity"`   // Minimum modularity improvement
	RandomSeed             int64                                    `json:"random_seed"`              // For reproducibility
	
	// Output configuration
	OutputPrefix           string                                   `json:"output_prefix"`            // Prefix for output files
	Verbose                bool                                     `json:"verbose"`                  // Enable verbose output
}

// ScarClusteringConfig contains all configuration for the SCAR approach
type ScarClusteringConfig struct {
	// Input files
	GraphFile    string `json:"graph_file"`
	MetaPathFile string `json:"meta_path_file"`
	OutputDir    string `json:"output_dir"`

	// SCAR algorithm configuration
	K                      int     `json:"k"`                        // Sketch size
	NK                     int     `json:"nk"`                       // Number of independent hash functions
	MaxIterations          int     `json:"max_iterations"`           // Maximum algorithm iterations
	MinModularity          float64 `json:"min_modularity"`           // Minimum modularity improvement
	RandomSeed             int64   `json:"random_seed"`              // For reproducibility
	
	// Parallel processing configuration
	ParallelEnabled        bool    `json:"parallel_enabled"`         // Enable parallel processing
	NumWorkers             int     `json:"num_workers"`              // Number of worker goroutines
	BatchSize              int     `json:"batch_size"`               // Nodes per batch
	UpdateBuffer           int     `json:"update_buffer"`            // Channel buffer size
	
	// Output configuration
	OutputPrefix           string  `json:"output_prefix"`            // Prefix for output files
	Verbose                bool    `json:"verbose"`                  // Enable verbose output
}

// ===== RESULT STRUCTS =====

// ClusteringResult contains the results from clustering operation
type ClusteringResult struct {
	Success            bool                    `json:"success"`
	Error              string                  `json:"error,omitempty"`
	Approach           string                  `json:"approach"`
	
	// Timing and performance
	Runtime            time.Duration           `json:"runtime"`
	MemoryPeakMB       int64                   `json:"memory_peak_mb"`
	
	// Clustering results
	Communities        map[string]int          `json:"communities"`        // node_id -> community_id
	Modularity         float64                 `json:"modularity"`
	NumCommunities     int                     `json:"num_communities"`
	NumLevels          int                     `json:"num_levels"`
	TotalIterations    int                     `json:"total_iterations"`
	
	// Output files generated
	OutputFiles        OutputFiles             `json:"output_files"`
	
	// Algorithm-specific details
	AlgorithmDetails   interface{}             `json:"algorithm_details,omitempty"`
}

// OutputFiles contains paths to all generated output files
type OutputFiles struct {
	MappingFile    string `json:"mapping_file"`     // Community assignments
	HierarchyFile  string `json:"hierarchy_file"`   // Hierarchical structure  
	RootFile       string `json:"root_file"`        // Top-level communities
	EdgesFile      string `json:"edges_file"`       // Inter-community edges
	StatsFile      string `json:"stats_file"`       // Algorithm statistics
	
	// Directory-based outputs (for SCAR)
	HierarchyDir   string `json:"hierarchy_dir,omitempty"`   
	MappingDir     string `json:"mapping_dir,omitempty"`     
	EdgesDir       string `json:"edges_dir,omitempty"`       
}

// MaterializationDetails contains materialization-specific details
type MaterializationDetails struct {
	InstancesGenerated     int                                `json:"instances_generated"`
	InstancesFiltered      int                                `json:"instances_filtered"`
	EdgesCreated           int                                `json:"edges_created"`
	NodesInResult          int                                `json:"nodes_in_result"`
	MaterializationTimeMS  int64                              `json:"materialization_time_ms"`
	LouvainLevels          int                                `json:"louvain_levels"`
}

// ScarDetails contains SCAR-specific details
type ScarDetails struct {
	SketchSize             int                                `json:"sketch_size"`
	NumHashFunctions       int                                `json:"num_hash_functions"`
	ParallelProcessing     bool                               `json:"parallel_processing"`
	NumWorkers             int                                `json:"num_workers"`
	TotalDuration          time.Duration                      `json:"total_duration"`
}

// ===== DEFAULT CONFIGURATIONS =====

// DefaultMaterializationConfig returns sensible defaults for materialization approach
func DefaultMaterializationConfig() MaterializationClusteringConfig {
	return MaterializationClusteringConfig{
		AggregationStrategy:    materialization.Count,
		MetaPathInterpretation: materialization.MeetingBased,
		Symmetric:              true,
		MinWeight:              1.0,
		MaxEdges:               0, // No limit
		MaxInstances:           1000000,
		TimeoutSeconds:         300,
		TraversalParallelism:   4,
		LouvainMaxIterations:   1,
		LouvainMinModularity:   0.001,
		RandomSeed:             42,
		OutputPrefix:           "communities",
		Verbose:                false,
	}
}

// DefaultScarConfig returns sensible defaults for SCAR approach
func DefaultScarConfig() ScarClusteringConfig {
	return ScarClusteringConfig{
		K:                      64,
		NK:                     8,
		MaxIterations:          50,
		MinModularity:          1e-6,
		RandomSeed:             42,
		ParallelEnabled:        true,
		NumWorkers:             4,
		BatchSize:              100,
		UpdateBuffer:           10000,
		OutputPrefix:           "scar",
		Verbose:                false,
	}
}

// ===== MAIN CLUSTERING FUNCTIONS =====

// RunMaterializationClustering executes the materialization + Louvain approach
func RunMaterializationClustering(config MaterializationClusteringConfig) (*ClusteringResult, error) {
	startTime := time.Now()
	result := &ClusteringResult{
		Approach: "Materialization + Louvain",
	}

	// Step 1: Load and validate input data
	if config.Verbose {
		fmt.Println("Loading and validating input data...")
	}
	
	graph, metaPath, err := loadAndValidateData(config.GraphFile, config.MetaPathFile)
	if err != nil {
		result.Error = fmt.Sprintf("validation failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result, err
	}

	// Step 2: Configure materialization
	matConfig := materialization.DefaultMaterializationConfig()
	matConfig.Aggregation.Strategy = config.AggregationStrategy
	matConfig.Aggregation.Interpretation = config.MetaPathInterpretation
	matConfig.Aggregation.Symmetric = config.Symmetric
	matConfig.Aggregation.MinWeight = config.MinWeight
	matConfig.Aggregation.MaxEdges = config.MaxEdges
	matConfig.Traversal.MaxInstances = config.MaxInstances
	matConfig.Traversal.TimeoutSeconds = config.TimeoutSeconds
	matConfig.Traversal.Parallelism = config.TraversalParallelism

	// Step 3: Perform materialization
	if config.Verbose {
		fmt.Println("Starting materialization...")
	}

	var progressCallback materialization.ProgressCallback
	if config.Verbose {
		progressCallback = func(current, total int, message string) {
			if total > 0 {
				percentage := float64(current) / float64(total) * 100
				fmt.Printf("\rProgress: %.1f%% (%d/%d) - %s", percentage, current, total, message)
			}
		}
	}

	engine := materialization.NewMaterializationEngine(graph, metaPath, matConfig, progressCallback)

	// Check feasibility
	canMaterialize, reason, err := engine.CanMaterialize(2000) // 2GB limit
	if err != nil {
		result.Error = fmt.Sprintf("feasibility check failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result, err
	}

	if !canMaterialize {
		result.Error = fmt.Sprintf("materialization not feasible: %s", reason)
		result.Runtime = time.Since(startTime)
		return result, fmt.Errorf("materialization not feasible: %s", reason)
	}

	matResult, err := engine.Materialize()
	if err != nil {
		result.Error = fmt.Sprintf("materialization failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result, err
	}

	if !matResult.Success {
		result.Error = fmt.Sprintf("materialization unsuccessful: %s", matResult.Error)
		result.Runtime = time.Since(startTime)
		return result, fmt.Errorf("materialization unsuccessful: %s", matResult.Error)
	}

	if config.Verbose {
		fmt.Printf("\nMaterialization completed: %d nodes, %d edges\n", 
			len(matResult.HomogeneousGraph.Nodes), len(matResult.HomogeneousGraph.Edges))
	}

	// Step 4: Configure and run Louvain
	if config.Verbose {
		fmt.Println("Running Louvain algorithm...")
	}

	louvainConfig := louvain.DefaultLouvainConfig()
	louvainConfig.MaxIterations = config.LouvainMaxIterations
	louvainConfig.MinModularity = config.LouvainMinModularity
	louvainConfig.RandomSeed = config.RandomSeed
	louvainConfig.Verbose = config.Verbose

	louvainGraph := convertToLouvainGraph(matResult.HomogeneousGraph)
	louvainResult, err := louvain.RunLouvain(louvainGraph, louvainConfig)
	if err != nil {
		result.Error = fmt.Sprintf("louvain failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result, err
	}

	if config.Verbose {
		fmt.Printf("Louvain completed: modularity=%.6f, communities=%d\n", 
			louvainResult.Modularity, len(getUniqueCommunities(louvainResult.FinalCommunities)))
	}

	// Step 5: Save outputs
	if config.Verbose {
		fmt.Println("Saving output files...")
	}

	outputFiles, err := saveMaterializationOutputs(louvainResult, louvainGraph, config.OutputDir, config.OutputPrefix)
	if err != nil {
		result.Error = fmt.Sprintf("failed to save outputs: %v", err)
		result.Runtime = time.Since(startTime)
		return result, err
	}

	// Step 6: Populate result
	result.Success = true
	result.Runtime = time.Since(startTime)
	result.MemoryPeakMB = matResult.Statistics.MemoryPeakMB
	result.Communities = louvainResult.FinalCommunities
	result.Modularity = louvainResult.Modularity
	result.NumCommunities = len(getUniqueCommunities(louvainResult.FinalCommunities))
	result.NumLevels = louvainResult.NumLevels
	result.TotalIterations = louvainResult.Statistics.TotalIterations
	result.OutputFiles = outputFiles
	result.AlgorithmDetails = MaterializationDetails{
		InstancesGenerated:    matResult.Statistics.InstancesGenerated,
		InstancesFiltered:     matResult.Statistics.InstancesFiltered,
		EdgesCreated:          matResult.Statistics.EdgesCreated,
		NodesInResult:         matResult.Statistics.NodesInResult,
		MaterializationTimeMS: matResult.Statistics.RuntimeMS,
		LouvainLevels:         louvainResult.NumLevels,
	}

	if config.Verbose {
		fmt.Printf("Materialization + Louvain completed successfully!\n")
		fmt.Printf("Runtime: %v, Modularity: %.6f, Communities: %d\n",
			result.Runtime, result.Modularity, result.NumCommunities)
	}

	return result, nil
}

// RunScarClustering executes the SCAR approach
func RunScarClustering(config ScarClusteringConfig) (*ClusteringResult, error) {
	startTime := time.Now()
	result := &ClusteringResult{
		Approach: "SCAR",
	}

	// Step 1: Load and validate input data
	if config.Verbose {
		fmt.Println("Loading and validating input data...")
	}
	
	graph, metaPath, err := loadAndValidateData(config.GraphFile, config.MetaPathFile)
	if err != nil {
		result.Error = fmt.Sprintf("validation failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result, err
	}

	// Step 2: Configure SCAR
	scarConfig := scar.DefaultScarConfig()
	scarConfig.MetaPath = convertToScarMetaPath(metaPath)
	scarConfig.K = config.K
	scarConfig.NK = config.NK
	scarConfig.MaxIterations = config.MaxIterations
	scarConfig.MinModularity = config.MinModularity
	scarConfig.RandomSeed = config.RandomSeed
	scarConfig.Verbose = config.Verbose

	// Parallel configuration
	scarConfig.Parallel.Enabled = config.ParallelEnabled
	scarConfig.Parallel.NumWorkers = config.NumWorkers
	scarConfig.Parallel.BatchSize = config.BatchSize
	scarConfig.Parallel.UpdateBuffer = config.UpdateBuffer

	// Progress callback
	if config.Verbose {
		scarConfig.ProgressCallback = func(level int, iteration int, modularity float64, nodes int) {
			fmt.Printf("\rLevel %d, Iteration %d: modularity=%.6f, nodes=%d", 
				level, iteration, modularity, nodes)
		}
	}

	// Step 3: Run SCAR
	if config.Verbose {
		fmt.Println("Running SCAR algorithm...")
	}

	scarGraph := convertToScarGraph(graph)
	scarResult, err := scar.RunScar(scarGraph, scarConfig)
	if err != nil {
		result.Error = fmt.Sprintf("SCAR failed: %v", err)
		result.Runtime = time.Since(startTime)
		return result, err
	}

	if config.Verbose {
		fmt.Printf("\nSCAR completed: modularity=%.6f, communities=%d\n", 
			scarResult.Modularity, len(getUniqueCommunities(scarResult.FinalCommunities)))
	}

	// Step 4: Save outputs
	if config.Verbose {
		fmt.Println("Saving output files...")
	}

	outputFiles, err := saveScarOutputs(scarResult, scarGraph, config.OutputDir, config.OutputPrefix)
	if err != nil {
		result.Error = fmt.Sprintf("failed to save outputs: %v", err)
		result.Runtime = time.Since(startTime)
		return result, err
	}

	// Step 5: Populate result
	result.Success = true
	result.Runtime = time.Since(startTime)
	result.MemoryPeakMB = 0 // SCAR doesn't track this the same way
	result.Communities = scarResult.FinalCommunities
	result.Modularity = scarResult.Modularity
	result.NumCommunities = len(getUniqueCommunities(scarResult.FinalCommunities))
	result.NumLevels = scarResult.NumLevels
	result.TotalIterations = scarResult.Statistics.TotalIterations
	result.OutputFiles = outputFiles
	result.AlgorithmDetails = ScarDetails{
		SketchSize:        config.K,
		NumHashFunctions:  config.NK,
		ParallelProcessing: config.ParallelEnabled,
		NumWorkers:       config.NumWorkers,
		TotalDuration:    scarResult.Statistics.TotalDuration,
	}

	if config.Verbose {
		fmt.Printf("SCAR completed successfully!\n")
		fmt.Printf("Runtime: %v, Modularity: %.6f, Communities: %d\n",
			result.Runtime, result.Modularity, result.NumCommunities)
	}

	return result, nil
}

// ===== VERIFICATION FUNCTIONS =====

// VerifyMaterializationOutput verifies that materialization output files are correctly generated
func VerifyMaterializationOutput(outputDir string, prefix string) error {
	requiredFiles := []string{
		fmt.Sprintf("%s.mapping", prefix),
		fmt.Sprintf("%s.hierarchy", prefix),
		fmt.Sprintf("%s.root", prefix),
		fmt.Sprintf("%s.edges", prefix),
	}

	for _, filename := range requiredFiles {
		filePath := filepath.Join(outputDir, filename)
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			return fmt.Errorf("required output file not found: %s", filePath)
		}
		
		// Check if file is not empty
		info, err := os.Stat(filePath)
		if err != nil {
			return fmt.Errorf("cannot stat file %s: %v", filePath, err)
		}
		if info.Size() == 0 {
			return fmt.Errorf("output file is empty: %s", filePath)
		}
	}

	return nil
}

// VerifyScarOutput verifies that SCAR output files are correctly generated
func VerifyScarOutput(outputDir string, prefix string) error {
	// Check root file
	rootFile := filepath.Join(outputDir, fmt.Sprintf("%s.root", prefix))
	if _, err := os.Stat(rootFile); os.IsNotExist(err) {
		return fmt.Errorf("required root file not found: %s", rootFile)
	}

	// Check required directories
	requiredDirs := []string{"hierarchy-output", "mapping-output", "edges-output"}
	
	for _, dir := range requiredDirs {
		dirPath := filepath.Join(outputDir, dir)
		if _, err := os.Stat(dirPath); os.IsNotExist(err) {
			return fmt.Errorf("required directory not found: %s", dirPath)
		}
		
		// Check if directory contains files
		files, err := filepath.Glob(filepath.Join(dirPath, fmt.Sprintf("%s_*.dat", prefix)))
		if err != nil {
			return fmt.Errorf("error checking files in %s: %v", dirPath, err)
		}
		if len(files) == 0 {
			return fmt.Errorf("no output files found in directory: %s", dirPath)
		}
	}

	return nil
}

// VerifyClusteringResult performs comprehensive verification of clustering results
func VerifyClusteringResult(result *ClusteringResult) error {
	if result == nil {
		return fmt.Errorf("result is nil")
	}

	if !result.Success {
		return fmt.Errorf("clustering was not successful: %s", result.Error)
	}

	// Check basic metrics
	if result.Modularity < -1.0 || result.Modularity > 1.0 {
		return fmt.Errorf("invalid modularity value: %.6f (should be between -1 and 1)", result.Modularity)
	}

	if result.NumCommunities <= 0 {
		return fmt.Errorf("invalid number of communities: %d", result.NumCommunities)
	}

	if len(result.Communities) == 0 {
		return fmt.Errorf("no community assignments found")
	}

	// Verify output files exist
	switch result.Approach {
	case "Materialization + Louvain":
		outputDir := filepath.Dir(result.OutputFiles.MappingFile)
		prefix := getFilenameWithoutExt(filepath.Base(result.OutputFiles.MappingFile))
		if err := VerifyMaterializationOutput(outputDir, prefix); err != nil {
			return fmt.Errorf("materialization output verification failed: %v", err)
		}
	case "SCAR":
		outputDir := filepath.Dir(result.OutputFiles.RootFile)
		prefix := getFilenameWithoutExt(filepath.Base(result.OutputFiles.RootFile))
		if err := VerifyScarOutput(outputDir, prefix); err != nil {
			return fmt.Errorf("SCAR output verification failed: %v", err)
		}
	default:
		return fmt.Errorf("unknown approach: %s", result.Approach)
	}

	return nil
}

// ===== HELPER FUNCTIONS =====

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

func saveMaterializationOutputs(result *louvain.LouvainResult, graph *louvain.HomogeneousGraph, outputDir, prefix string) (OutputFiles, error) {
	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return OutputFiles{}, fmt.Errorf("failed to create output directory: %w", err)
	}

	// Use Louvain's file writer
	writer := louvain.NewFileWriter()
	if err := writer.WriteAll(result, graph, outputDir, prefix); err != nil {
		return OutputFiles{}, err
	}

	// Return file paths
	return OutputFiles{
		MappingFile:   filepath.Join(outputDir, fmt.Sprintf("%s.mapping", prefix)),
		HierarchyFile: filepath.Join(outputDir, fmt.Sprintf("%s.hierarchy", prefix)),
		RootFile:      filepath.Join(outputDir, fmt.Sprintf("%s.root", prefix)),
		EdgesFile:     filepath.Join(outputDir, fmt.Sprintf("%s.edges", prefix)),
	}, nil
}

func saveScarOutputs(result *scar.ScarResult, graph *scar.HeterogeneousGraph, outputDir, prefix string) (OutputFiles, error) {
	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return OutputFiles{}, fmt.Errorf("failed to create output directory: %w", err)
	}

	// Use SCAR's output writer
	if err := scar.WriteAll(result, graph, outputDir, prefix); err != nil {
		return OutputFiles{}, err
	}

	// Return file paths
	return OutputFiles{
		RootFile:     filepath.Join(outputDir, fmt.Sprintf("%s.root", prefix)),
		HierarchyDir: filepath.Join(outputDir, "hierarchy-output"),
		MappingDir:   filepath.Join(outputDir, "mapping-output"),
		EdgesDir:     filepath.Join(outputDir, "edges-output"),
	}, nil
}

func getUniqueCommunities(communities map[string]int) map[int]bool {
	unique := make(map[int]bool)
	for _, comm := range communities {
		unique[comm] = true
	}
	return unique
}

func getFilenameWithoutExt(filename string) string {
	ext := filepath.Ext(filename)
	return filename[0 : len(filename)-len(ext)]
}
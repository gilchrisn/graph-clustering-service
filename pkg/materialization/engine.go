package materialization

import (
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// MaterializationEngine is the main component that orchestrates graph materialization
type MaterializationEngine struct {
	graph         *models.HeterogeneousGraph
	metaPath      *models.MetaPath
	config        MaterializationConfig
	progressCb    ProgressCallback
	
	// Internal components
	instanceGen   *InstanceGenerator
	homogBuilder  *HomogeneousBuilder
	weightCalc    *WeightCalculator
	
	// State tracking
	startTime     time.Time
	instanceCount int
	mu            sync.RWMutex
}

// NewMaterializationEngine creates a new materialization engine
func NewMaterializationEngine(graph *models.HeterogeneousGraph, metaPath *models.MetaPath, 
	config MaterializationConfig, progressCb ProgressCallback) *MaterializationEngine {
	
	engine := &MaterializationEngine{
		graph:      graph,
		metaPath:   metaPath,
		config:     config,
		progressCb: progressCb,
	}
	
	// Initialize components
	engine.instanceGen = NewInstanceGenerator(graph, metaPath, config.Traversal)
	engine.homogBuilder = NewHomogeneousBuilder(metaPath, config.Aggregation)
	engine.weightCalc = NewWeightCalculator(config.Aggregation)
	
	return engine
}

// Materialize performs the complete materialization process
func (me *MaterializationEngine) Materialize() (*MaterializationResult, error) {
	me.startTime = time.Now()
	
	// Validate inputs
	if err := me.validateInputs(); err != nil {
		return me.createErrorResult(err)
	}
	
	// Estimate instance count for progress tracking
	estimatedInstances, err := me.instanceGen.EstimateInstanceCount()
	if err != nil {
		return me.createErrorResult(fmt.Errorf("failed to estimate instance count: %w", err))
	}
	
	me.reportProgress(0, estimatedInstances, "Starting materialization...")
	
	// Generate all path instances using BFS
	instances, traversalStats, err := me.generateInstances(estimatedInstances)
	if err != nil {
		return me.createErrorResult(fmt.Errorf("failed to generate instances: %w", err))
	}

	// Print all instances for debugging
	for _, instance := range instances {
		fmt.Println(instance)
	}
	
	me.reportProgress(len(instances), estimatedInstances, "Instances generated, building homogeneous graph...")
	
	// Build homogeneous graph from instances
	homogeneousGraph, aggStats, err := me.buildHomogeneousGraph(instances)
	if err != nil {
		return me.createErrorResult(fmt.Errorf("failed to build homogeneous graph: %w", err))
	}

	// Print homogeneous graph for debugging
	fmt.Println("Homogeneous Graph:")
	for _, node := range homogeneousGraph.Nodes {
		fmt.Printf("Node: %s, Type: %s\n", node.ID, node.Type)
	}
	for edgeKey, weight := range homogeneousGraph.Edges {
		fmt.Printf("Edge: %s ↔ %s, Weight: %.2f\n", edgeKey.From, edgeKey.To, weight)
	}
	
	me.reportProgress(estimatedInstances, estimatedInstances, "Materialization complete!")
	
	// Create final result
	result := &MaterializationResult{
		HomogeneousGraph: homogeneousGraph,
		Statistics: ProcessingStatistics{
			RuntimeMS:             time.Since(me.startTime).Milliseconds(),
			MemoryPeakMB:          me.getMemoryUsage(),
			InstancesGenerated:    len(instances),
			InstancesFiltered:     me.instanceCount - len(instances),
			EdgesCreated:          len(homogeneousGraph.Edges),
			NodesInResult:         len(homogeneousGraph.Nodes),
			TraversalStatistics:   traversalStats,
			AggregationStatistics: aggStats,
		},
		Config:  me.config,
		Success: true,
	}
	
	return result, nil
}

// validateInputs performs basic validation of inputs
func (me *MaterializationEngine) validateInputs() error {
	if me.graph == nil {
		return MaterializationError{Component: "input", Message: "heterogeneous graph cannot be nil"}
	}
	
	if me.metaPath == nil {
		return MaterializationError{Component: "input", Message: "meta path cannot be nil"}
	}
	
	if len(me.graph.Nodes) == 0 {
		return MaterializationError{Component: "input", Message: "graph has no nodes"}
	}
	
	if len(me.graph.Edges) == 0 {
		return MaterializationError{Component: "input", Message: "graph has no edges"}
	}
	
	// Validate meta path against graph (this should have been done earlier, but double-check)
	startNodeType := me.metaPath.NodeSequence[0]
	hasStartNodes := false
	for _, node := range me.graph.Nodes {
		if node.Type == startNodeType {
			hasStartNodes = true
			break
		}
	}
	
	if !hasStartNodes {
		return MaterializationError{
			Component: "input", 
			Message:   fmt.Sprintf("no nodes of starting type '%s' found in graph", startNodeType),
		}
	}
	
	return nil
}

// generateInstances uses the InstanceGenerator to find all path instances
func (me *MaterializationEngine) generateInstances(estimatedCount int) ([]PathInstance, TraversalStats, error) {
	// Set up progress callback for instance generation
	instanceProgressCb := func(current int, message string) {
		me.mu.Lock()
		me.instanceCount = current
		me.mu.Unlock()
		me.reportProgress(current, estimatedCount, message)
	}
	
	return me.instanceGen.FindAllInstances(instanceProgressCb)
}

// buildHomogeneousGraph converts instances into a homogeneous graph
func (me *MaterializationEngine) buildHomogeneousGraph(instances []PathInstance) (*HomogeneousGraph, AggregationStats, error) {
	// Add instances to the builder
	for _, instance := range instances {
		me.homogBuilder.AddInstance(instance)
	}
	
	// Build the final homogeneous graph
	homogGraph, aggStats := me.homogBuilder.Build()
	
	// Apply weight calculation and normalization
	if err := me.weightCalc.ProcessGraph(homogGraph); err != nil {
		return nil, aggStats, fmt.Errorf("failed to process weights: %w", err)
	}
	
	// Calculate final statistics
	homogGraph.CalculateStatistics()
	
	return homogGraph, aggStats, nil
}

// reportProgress calls the progress callback if one is provided
func (me *MaterializationEngine) reportProgress(current, total int, message string) {
	if me.progressCb != nil && me.config.Progress.EnableProgress {
		me.progressCb(current, total, message)
	}
}

// getMemoryUsage returns current memory usage in MB
func (me *MaterializationEngine) getMemoryUsage() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc / 1024 / 1024)
}

// createErrorResult creates a MaterializationResult for errors
func (me *MaterializationEngine) createErrorResult(err error) (*MaterializationResult, error) {
	return &MaterializationResult{
		Statistics: ProcessingStatistics{
			RuntimeMS: time.Since(me.startTime).Milliseconds(),
		},
		Config:  me.config,
		Success: false,
		Error:   err.Error(),
	}, err
}

// EstimateComplexity estimates the computational complexity of materialization
func (me *MaterializationEngine) EstimateComplexity() (int, error) {
	return me.instanceGen.EstimateInstanceCount()
}

// GetMemoryEstimate estimates memory usage for materialization
func (me *MaterializationEngine) GetMemoryEstimate() (int64, error) {
	instanceCount, err := me.EstimateComplexity()
	if err != nil {
		return 0, err
	}
	
	// Rough estimate: each instance takes ~100 bytes, plus overhead
	estimatedMB := int64(instanceCount * 100 / 1024 / 1024)
	
	// Add overhead for graph structures
	overhead := int64(len(me.graph.Nodes) + len(me.graph.Edges)) * 50 / 1024 / 1024
	
	return estimatedMB + overhead + 100, nil // +100MB for safety buffer
}

// CanMaterialize checks if materialization is feasible given memory constraints
func (me *MaterializationEngine) CanMaterialize(maxMemoryMB int64) (bool, string, error) {
	estimated, err := me.GetMemoryEstimate()
	if err != nil {
		return false, "", err
	}
	
	if estimated > maxMemoryMB {
		return false, fmt.Sprintf("estimated memory usage (%d MB) exceeds limit (%d MB)", estimated, maxMemoryMB), nil
	}
	
	instanceCount, _ := me.EstimateComplexity()
	if instanceCount > me.config.Traversal.MaxInstances {
		return false, fmt.Sprintf("estimated instances (%d) exceeds limit (%d)", instanceCount, me.config.Traversal.MaxInstances), nil
	}
	
	return true, "materialization is feasible", nil
}

// MaterializeWithDefaults is a convenience function that uses default configuration
func MaterializeWithDefaults(graph *models.HeterogeneousGraph, metaPath *models.MetaPath, 
	progressCb ProgressCallback) (*MaterializationResult, error) {
	
	config := DefaultMaterializationConfig()
	engine := NewMaterializationEngine(graph, metaPath, config, progressCb)
	return engine.Materialize()
}

// MaterializeToFile performs materialization and saves result to file
func MaterializeToFile(graph *models.HeterogeneousGraph, metaPath *models.MetaPath, 
	outputPath string, config MaterializationConfig) error {
	
	// Create progress callback that prints to console
	progressCb := func(current, total int, message string) {
		if total > 0 {
			percentage := float64(current) / float64(total) * 100
			fmt.Printf("\rProgress: %.1f%% (%d/%d) - %s", percentage, current, total, message)
		} else {
			fmt.Printf("\r%s", message)
		}
	}
	
	// Perform materialization
	engine := NewMaterializationEngine(graph, metaPath, config, progressCb)
	result, err := engine.Materialize()
	if err != nil {
		return err
	}
	
	fmt.Println() // New line after progress
	
	// Save result to file (implementation would depend on desired format)
	// For now, we'll just return success
	fmt.Printf("Materialization complete! Generated %d edges from %d instances\n", 
		len(result.HomogeneousGraph.Edges), result.Statistics.InstancesGenerated)
	
	return nil
}

// BatchMaterialize performs materialization on multiple meta paths
func BatchMaterialize(graph *models.HeterogeneousGraph, metaPaths []*models.MetaPath, 
	config MaterializationConfig, progressCb ProgressCallback) ([]*MaterializationResult, error) {
	
	results := make([]*MaterializationResult, len(metaPaths))
	
	for i, metaPath := range metaPaths {
		if progressCb != nil {
			progressCb(i, len(metaPaths), fmt.Sprintf("Processing meta path %d/%d: %s", i+1, len(metaPaths), metaPath.ID))
		}
		
		engine := NewMaterializationEngine(graph, metaPath, config, nil) // No per-path progress
		result, err := engine.Materialize()
		if err != nil {
			// Continue with other meta paths, but record the error
			result = &MaterializationResult{
				Config:  config,
				Success: false,
				Error:   err.Error(),
			}
		}
		results[i] = result
	}
	
	return results, nil
}
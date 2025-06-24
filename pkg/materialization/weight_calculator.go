package materialization

import (
	"fmt"
	"math"
)

// WeightCalculator handles weight processing and normalization for homogeneous graphs
type WeightCalculator struct {
	config AggregationConfig
}

// NewWeightCalculator creates a new weight calculator
func NewWeightCalculator(config AggregationConfig) *WeightCalculator {
	return &WeightCalculator{
		config: config,
	}
}

// ProcessGraph applies all weight processing steps to the homogeneous graph
func (wc *WeightCalculator) ProcessGraph(graph *HomogeneousGraph) error {
	// Validate input
	if graph == nil {
		return MaterializationError{Component: "weight_calculation", Message: "graph cannot be nil"}
	}
	
	if len(graph.Edges) == 0 {
		fmt.Println("No edges to process, skipping weight calculation")
		return nil // Nothing to do
	}
	
	// Apply normalization if specified
	if wc.config.Normalization != NoNormalization {
		if err := wc.applyNormalization(graph); err != nil {
			return fmt.Errorf("failed to apply normalization: %w", err)
		}
	}
	
	// Apply final filtering based on normalized weights
	// wc.applyFinalFiltering(graph)
	
	return nil
}

// applyNormalization applies the specified normalization strategy
func (wc *WeightCalculator) applyNormalization(graph *HomogeneousGraph) error {
	switch wc.config.Normalization {
	case DegreeNorm:
		return wc.applyDegreeNormalization(graph)
	case MaxNorm:
		return wc.applyMaxNormalization(graph)
	case StandardNorm:
		return wc.applyStandardNormalization(graph)
	default:
		return nil // No normalization
	}
}

// applyDegreeNormalization normalizes edge weights by node degrees
func (wc *WeightCalculator) applyDegreeNormalization(graph *HomogeneousGraph) error {
	// Calculate node degrees if not already done
	nodeDegrees := make(map[string]float64)
	for nodeID := range graph.Nodes {
		nodeDegrees[nodeID] = 0.0
	}
	
	// Count degrees from edges
	for edgeKey, weight := range graph.Edges {
		nodeDegrees[edgeKey.From] += weight
		nodeDegrees[edgeKey.To] += weight
	}
	
	// Normalize edge weights by geometric mean of node degrees
	normalizedEdges := make(map[EdgeKey]float64)
	for edgeKey, weight := range graph.Edges {
		fromDegree := nodeDegrees[edgeKey.From]
		toDegree := nodeDegrees[edgeKey.To]
		
		if fromDegree > 0 && toDegree > 0 {
			// Geometric mean normalization
			normFactor := math.Sqrt(fromDegree * toDegree)
			normalizedWeight := weight / normFactor
			normalizedEdges[edgeKey] = normalizedWeight
		} else {
			normalizedEdges[edgeKey] = weight
		}
	}
	
	graph.Edges = normalizedEdges
	
	// Recalculate node degrees after normalization
	wc.recalculateNodeDegrees(graph)
	
	return nil
}

// applyMaxNormalization normalizes all weights to [0,1] range
func (wc *WeightCalculator) applyMaxNormalization(graph *HomogeneousGraph) error {
	if len(graph.Edges) == 0 {
		return nil
	}
	
	// Find min and max weights
	minWeight := math.Inf(1)
	maxWeight := math.Inf(-1)
	
	for _, weight := range graph.Edges {
		if weight < minWeight {
			minWeight = weight
		}
		if weight > maxWeight {
			maxWeight = weight
		}
	}
	
	// Handle edge case where all weights are the same
	if maxWeight == minWeight {
		for edgeKey := range graph.Edges {
			graph.Edges[edgeKey] = 1.0
		}
		return nil
	}
	
	// Normalize to [0,1]
	weightRange := maxWeight - minWeight
	for edgeKey, weight := range graph.Edges {
		normalizedWeight := (weight - minWeight) / weightRange
		graph.Edges[edgeKey] = normalizedWeight
	}
	
	// Recalculate node degrees
	wc.recalculateNodeDegrees(graph)
	
	return nil
}

// applyStandardNormalization applies z-score normalization (mean=0, std=1)
func (wc *WeightCalculator) applyStandardNormalization(graph *HomogeneousGraph) error {
	if len(graph.Edges) == 0 {
		return nil
	}
	
	// Calculate mean
	totalWeight := 0.0
	edgeCount := len(graph.Edges)
	
	for _, weight := range graph.Edges {
		totalWeight += weight
	}
	
	mean := totalWeight / float64(edgeCount)
	
	// Calculate standard deviation
	sumSquaredDiffs := 0.0
	for _, weight := range graph.Edges {
		diff := weight - mean
		sumSquaredDiffs += diff * diff
	}
	
	variance := sumSquaredDiffs / float64(edgeCount)
	stdDev := math.Sqrt(variance)
	
	// Handle edge case where standard deviation is 0
	if stdDev == 0 {
		for edgeKey := range graph.Edges {
			graph.Edges[edgeKey] = 0.0
		}
		return nil
	}
	
	// Apply z-score normalization
	for edgeKey, weight := range graph.Edges {
		normalizedWeight := (weight - mean) / stdDev
		graph.Edges[edgeKey] = normalizedWeight
	}
	
	// Recalculate node degrees
	wc.recalculateNodeDegrees(graph)
	
	return nil
}

// applyFinalFiltering removes edges below minimum weight threshold after normalization
func (wc *WeightCalculator) applyFinalFiltering(graph *HomogeneousGraph) {
	if wc.config.MinWeight <= 0 {
		return // No filtering needed
	}
	
	// Remove edges below threshold
	filteredEdges := make(map[EdgeKey]float64)
	for edgeKey, weight := range graph.Edges {
		if weight >= wc.config.MinWeight {
			filteredEdges[edgeKey] = weight
		}
	}
	
	graph.Edges = filteredEdges
	
	// Recalculate node degrees
	wc.recalculateNodeDegrees(graph)
	
	// Remove isolated nodes (nodes with no edges)
	wc.removeIsolatedNodes(graph)
}

// recalculateNodeDegrees recalculates node degrees after weight changes
func (wc *WeightCalculator) recalculateNodeDegrees(graph *HomogeneousGraph) {
	// Reset all degrees
	for nodeID, node := range graph.Nodes {
		node.Degree = 0
		graph.Nodes[nodeID] = node
	}
	
	// Count degrees from current edges
	for edgeKey := range graph.Edges {
		if fromNode, exists := graph.Nodes[edgeKey.From]; exists {
			fromNode.Degree++
			graph.Nodes[edgeKey.From] = fromNode
		}
		
		if toNode, exists := graph.Nodes[edgeKey.To]; exists {
			toNode.Degree++
			graph.Nodes[edgeKey.To] = toNode
		}
	}
}

// removeIsolatedNodes removes nodes that have no edges after filtering
func (wc *WeightCalculator) removeIsolatedNodes(graph *HomogeneousGraph) {
	// Find nodes that appear in edges
	connectedNodes := make(map[string]bool)
	for edgeKey := range graph.Edges {
		connectedNodes[edgeKey.From] = true
		connectedNodes[edgeKey.To] = true
	}
	
	// Remove isolated nodes
	filteredNodes := make(map[string]Node)
	for nodeID, node := range graph.Nodes {
		if connectedNodes[nodeID] {
			filteredNodes[nodeID] = node
		}
	}
	
	graph.Nodes = filteredNodes
}

// GetWeightStatistics calculates detailed statistics about edge weights
func (wc *WeightCalculator) GetWeightStatistics(graph *HomogeneousGraph) WeightStatistics {
	stats := WeightStatistics{}
	
	if len(graph.Edges) == 0 {
		return stats
	}
	
	// Basic statistics
	totalWeight := 0.0
	minWeight := math.Inf(1)
	maxWeight := math.Inf(-1)
	weights := make([]float64, 0, len(graph.Edges))
	
	for _, weight := range graph.Edges {
		weights = append(weights, weight)
		totalWeight += weight
		
		if weight < minWeight {
			minWeight = weight
		}
		if weight > maxWeight {
			maxWeight = weight
		}
	}
	
	stats.Count = len(graph.Edges)
	stats.Sum = totalWeight
	stats.Mean = totalWeight / float64(len(graph.Edges))
	stats.Min = minWeight
	stats.Max = maxWeight
	
	// Calculate variance and standard deviation
	sumSquaredDiffs := 0.0
	for _, weight := range weights {
		diff := weight - stats.Mean
		sumSquaredDiffs += diff * diff
	}
	
	stats.Variance = sumSquaredDiffs / float64(len(weights))
	stats.StandardDeviation = math.Sqrt(stats.Variance)
	
	// Calculate percentiles (simple implementation)
	stats.Percentiles = wc.calculatePercentiles(weights)
	
	return stats
}

// calculatePercentiles calculates common percentiles for weight distribution
func (wc *WeightCalculator) calculatePercentiles(weights []float64) map[string]float64 {
	if len(weights) == 0 {
		return make(map[string]float64)
	}
	
	// Simple sorting-based percentile calculation
	// Note: For large datasets, more efficient algorithms exist
	sorted := make([]float64, len(weights))
	copy(sorted, weights)
	
	// Simple bubble sort (fine for moderate sizes)
	for i := 0; i < len(sorted); i++ {
		for j := 0; j < len(sorted)-1-i; j++ {
			if sorted[j] > sorted[j+1] {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}
	
	percentiles := make(map[string]float64)
	n := len(sorted)
	
	percentiles["p25"] = sorted[n/4]
	percentiles["p50"] = sorted[n/2]
	percentiles["p75"] = sorted[3*n/4]
	percentiles["p90"] = sorted[9*n/10]
	percentiles["p95"] = sorted[95*n/100]
	percentiles["p99"] = sorted[99*n/100]
	
	return percentiles
}

// ValidateWeights checks if all weights in the graph are valid
func (wc *WeightCalculator) ValidateWeights(graph *HomogeneousGraph) error {
	for edgeKey, weight := range graph.Edges {
		if math.IsNaN(weight) {
			return MaterializationError{
				Component: "weight_calculation",
				Message:   "NaN weight detected",
				Details:   fmt.Sprintf("edge: %s", edgeKey.String()),
			}
		}
		
		if math.IsInf(weight, 0) {
			return MaterializationError{
				Component: "weight_calculation",
				Message:   "infinite weight detected",
				Details:   fmt.Sprintf("edge: %s, weight: %f", edgeKey.String(), weight),
			}
		}
		
		if weight < 0 {
			return MaterializationError{
				Component: "weight_calculation",
				Message:   "negative weight detected",
				Details:   fmt.Sprintf("edge: %s, weight: %f", edgeKey.String(), weight),
			}
		}
	}
	
	return nil
}

// WeightStatistics contains detailed statistics about edge weights
type WeightStatistics struct {
	Count             int                `json:"count"`
	Sum               float64            `json:"sum"`
	Mean              float64            `json:"mean"`
	Min               float64            `json:"min"`
	Max               float64            `json:"max"`
	Variance          float64            `json:"variance"`
	StandardDeviation float64            `json:"standard_deviation"`
	Percentiles       map[string]float64 `json:"percentiles"`
}

// ApplyCustomNormalization allows applying a custom normalization function
func (wc *WeightCalculator) ApplyCustomNormalization(graph *HomogeneousGraph, 
	normFunc func(EdgeKey, float64, *HomogeneousGraph) float64) error {
	
	normalizedEdges := make(map[EdgeKey]float64)
	
	for edgeKey, weight := range graph.Edges {
		normalizedWeight := normFunc(edgeKey, weight, graph)
		
		// Validate the result
		if math.IsNaN(normalizedWeight) || math.IsInf(normalizedWeight, 0) {
			return MaterializationError{
				Component: "weight_calculation",
				Message:   "custom normalization produced invalid weight",
				Details:   fmt.Sprintf("edge: %s, original: %f, normalized: %f", edgeKey.String(), weight, normalizedWeight),
			}
		}
		
		normalizedEdges[edgeKey] = normalizedWeight
	}
	
	graph.Edges = normalizedEdges
	wc.recalculateNodeDegrees(graph)
	
	return nil
}

// GetNormalizationSummary returns a summary of the normalization that would be applied
func (wc *WeightCalculator) GetNormalizationSummary(graph *HomogeneousGraph) string {
	switch wc.config.Normalization {
	case DegreeNorm:
		return "Degree normalization: weights divided by geometric mean of node degrees"
	case MaxNorm:
		return "Max normalization: weights scaled to [0,1] range"
	case StandardNorm:
		return "Standard normalization: weights standardized to mean=0, std=1"
	case NoNormalization:
		return "No normalization applied"
	default:
		return "Unknown normalization type"
	}
}
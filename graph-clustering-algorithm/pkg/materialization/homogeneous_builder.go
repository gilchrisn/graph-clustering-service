package materialization

import (
	"fmt"
	"sort"
	"math"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// HomogeneousBuilder converts path instances into a homogeneous graph
type HomogeneousBuilder struct {
	metaPath *models.MetaPath
	config   AggregationConfig
	
	// Instance storage grouped by edge endpoints (for direct traversal)
	instanceGroups map[EdgeKey][]PathInstance
	nodeSet        map[string]bool // Track unique nodes
	
	// Statistics
	stats AggregationStats
}

// NewHomogeneousBuilder creates a new homogeneous graph builder
func NewHomogeneousBuilder(metaPath *models.MetaPath, config AggregationConfig) *HomogeneousBuilder {
	return &HomogeneousBuilder{
		metaPath:       metaPath,
		config:         config,
		instanceGroups: make(map[EdgeKey][]PathInstance),
		nodeSet:        make(map[string]bool),
		stats: AggregationStats{
			WeightDistribution: make(map[string]int),
		},
	}
}

// AddInstance adds a path instance to the builder
func (hb *HomogeneousBuilder) AddInstance(instance PathInstance) {
	if !instance.IsValid() {
		fmt.Printf("Skipping invalid instance: %v\n", instance)
		return // Skip invalid instances
	}
	
	startNode := instance.GetStartNode()
	endNode := instance.GetEndNode()
	
	// Track unique nodes
	hb.nodeSet[startNode] = true
	hb.nodeSet[endNode] = true
	
	// Group instances by edge key
	edgeKey := EdgeKey{From: startNode, To: endNode}
	hb.instanceGroups[edgeKey] = append(hb.instanceGroups[edgeKey], instance)
	
	
	hb.stats.InstancesAggregated++
}

// Build constructs the final homogeneous graph from all added instances
func (hb *HomogeneousBuilder) Build() (*HomogeneousGraph, AggregationStats) {
	// Determine the node type for the homogeneous graph
	var nodeType string
	if hb.metaPath.IsSymmetric() {
		// For symmetric paths, use the start/end node type
		nodeType = hb.metaPath.NodeSequence[0]
	} else {
		// For non-symmetric paths, it's a mixed graph
		nodeType = "Mixed"
	}
	
	// Create the homogeneous graph
	homogGraph := &HomogeneousGraph{
		NodeType:   nodeType,
		Nodes:      make(map[string]Node),
		Edges:      make(map[EdgeKey]float64),
		MetaPath:   *hb.metaPath,
	}
	
	// Add nodes to the homogeneous graph
	hb.addNodesToGraph(homogGraph)
	
	// Process instance groups into edges based on interpretation
	if hb.config.Interpretation == MeetingBased {
		hb.processMeetingInstanceGroups(homogGraph)
	} else {
		hb.processDirectInstanceGroups(homogGraph)
	}
	
	// Apply filtering
	hb.applyFiltering(homogGraph)
	
	// Update statistics
	hb.stats.EdgeGroupsProcessed = len(hb.instanceGroups)
	hb.updateWeightDistribution(homogGraph)
	
	fmt.Printf("Homogeneous graph built with %d nodes and %d edges\n", len(homogGraph.Nodes), len(homogGraph.Edges))
	// Print the full graph structure
	
	return homogGraph, hb.stats
}

// addNodesToGraph adds all unique nodes to the homogeneous graph
func (hb *HomogeneousBuilder) addNodesToGraph(homogGraph *HomogeneousGraph) {
	for nodeID := range hb.nodeSet {
		// Create a node entry (properties would come from original graph if needed)
		homogGraph.Nodes[nodeID] = Node{
			ID:         nodeID,
			Type:       homogGraph.NodeType,
			Properties: make(map[string]interface{}),
			Degree:     0, // Will be updated when edges are added
		}
	}
}

// processDirectInstanceGroups converts instance groups into weighted edges (existing implementation)
func (hb *HomogeneousBuilder) processDirectInstanceGroups(homogGraph *HomogeneousGraph) {
	for edgeKey, instances := range hb.instanceGroups {
		if len(instances) == 0 {
			continue
		}
		
		// Calculate edge weight using the configured strategy
		weight := hb.calculateEdgeWeight(instances)
		
		// Apply minimum weight filter
		if weight < hb.config.MinWeight {
			hb.stats.EdgesFiltered++
			continue
		}
		
		// Add edge to graph
		hb.addEdgeToGraph(homogGraph, edgeKey, weight)
	}
}


// processMeetingInstanceGroups implements meeting-based meta path interpretation
func (hb *HomogeneousBuilder) processMeetingInstanceGroups(homogGraph *HomogeneousGraph) {
	// Group instances by meeting point (middle node of the path)
	meetingGroups := make(map[string][]PathInstance)
	
	for _, instances := range hb.instanceGroups {
		for _, instance := range instances {
			if !instance.IsValid() || len(instance.Nodes) < 2 {
				continue // Need at least 2 nodes for meeting-based (A -> B)
			}
			
			// Find the meeting point - typically the middle node
			meetingPoint := instance.GetEndNode()
			
			meetingGroups[meetingPoint] = append(meetingGroups[meetingPoint], instance)
		}
	}
	
	// For each meeting point, generate edges between all pairs of start nodes
	for _, instances := range meetingGroups {
		if len(instances) < 2 {
			continue // Need at least 2 instances to create edges
		}
		
		// Collect all start nodes for this meeting point
		startNodes := make([]string, 0, len(instances))
		instanceWeights := make(map[string][]float64) // startNode -> weights
		
		for _, instance := range instances {
			startNode := instance.GetStartNode()
			startNodes = append(startNodes, startNode)
			instanceWeights[startNode] = append(instanceWeights[startNode], instance.Weight)
		}
		
		// Generate edges between all pairs of start nodes
		for i := 0; i < len(startNodes); i++ {
			for j := i + 1; j < len(startNodes); j++ {
				nodeA := startNodes[i]
				nodeB := startNodes[j]
				
				// Skip self-loops unless explicitly allowed
				if nodeA == nodeB {
					continue
				}
				
				// Calculate edge weight based on the instances that share this meeting point
				// We use the combination of weights from both nodes
				weightsA := instanceWeights[nodeA]
				weightsB := instanceWeights[nodeB]
				
				edgeWeight := hb.calculateMeetingBasedWeight(weightsA, weightsB)
				
				// Apply minimum weight filter
				if edgeWeight < hb.config.MinWeight {
					hb.stats.EdgesFiltered++
					continue
				}
				
				// Add edge (and reverse edge if symmetric)
				edgeKey := EdgeKey{From: nodeA, To: nodeB}
				hb.addEdgeToGraph(homogGraph, edgeKey, edgeWeight)
				
				// Add reverse edge for symmetry if configured
				if hb.config.Symmetric {
					reverseKey := EdgeKey{From: nodeB, To: nodeA}
					hb.addEdgeToGraph(homogGraph, reverseKey, edgeWeight)
				}
			}
		}
	}
}


// calculateMeetingBasedWeight calculates the weight for a meeting-based edge
func (hb *HomogeneousBuilder) calculateMeetingBasedWeight(weightsA, weightsB []float64) float64 {
	switch hb.config.Strategy {
	case Count:
		// Count the number of co-occurrences
		return float64(len(weightsA) * len(weightsB))
		
	case Sum:
		// Sum all combinations
		total := 0.0
		for _, wA := range weightsA {
			for _, wB := range weightsB {
				total += wA * wB
			}
		}
		return total
		
	case Average:
		// Average of all combinations
		total := 0.0
		count := 0
		for _, wA := range weightsA {
			for _, wB := range weightsB {
				total += wA * wB
				count++
			}
		}
		if count == 0 {
			return 0.0
		}
		return total / float64(count)
		
	case Maximum:
		// Maximum combination
		maxWeight := 0.0
		for _, wA := range weightsA {
			for _, wB := range weightsB {
				weight := wA * wB
				if weight > maxWeight {
					maxWeight = weight
				}
			}
		}
		return maxWeight
		
	case Minimum:
		// Minimum combination
		minWeight := math.Inf(1)
		for _, wA := range weightsA {
			for _, wB := range weightsB {
				weight := wA * wB
				if weight < minWeight {
					minWeight = weight
				}
			}
		}
		if math.IsInf(minWeight, 1) {
			return 0.0
		}
		return minWeight
		
	default:
		// Default to count
		return float64(len(weightsA) * len(weightsB))
	}
}

// addEdgeToGraph safely adds an edge to the graph, updating node degrees
func (hb *HomogeneousBuilder) addEdgeToGraph(homogGraph *HomogeneousGraph, edgeKey EdgeKey, weight float64) {
	// Add or update edge weight (in case of multiple edges between same nodes)
	if existingWeight, exists := homogGraph.Edges[edgeKey]; exists {
		homogGraph.Edges[edgeKey] = existingWeight + weight
	} else {
		homogGraph.Edges[edgeKey] = weight
	}
	
	// Update node degrees
	if fromNode, exists := homogGraph.Nodes[edgeKey.From]; exists {
		fromNode.Degree++
		homogGraph.Nodes[edgeKey.From] = fromNode
	}
	
	if toNode, exists := homogGraph.Nodes[edgeKey.To]; exists {
		toNode.Degree++
		homogGraph.Nodes[edgeKey.To] = toNode
	}
}


// calculateEdgeWeight computes the weight for an edge based on its instances
func (hb *HomogeneousBuilder) calculateEdgeWeight(instances []PathInstance) float64 {
	if len(instances) == 0 {
		return 0.0
	}
	
	switch hb.config.Strategy {
	case Count:
		return float64(len(instances))
		
	case Sum:
		total := 0.0
		for _, instance := range instances {
			total += instance.Weight
		}
		return total
		
	case Average:
		total := 0.0
		for _, instance := range instances {
			total += instance.Weight
		}
		return total / float64(len(instances))
		
	case Maximum:
		max := instances[0].Weight
		for _, instance := range instances {
			if instance.Weight > max {
				max = instance.Weight
			}
		}
		return max
		
	case Minimum:
		min := instances[0].Weight
		for _, instance := range instances {
			if instance.Weight < min {
				min = instance.Weight
			}
		}
		return min
		
	default:
		// Default to count
		return float64(len(instances))
	}
}

// applyFiltering applies edge count and other filtering rules
func (hb *HomogeneousBuilder) applyFiltering(homogGraph *HomogeneousGraph) {
	// Apply maximum edge count filtering (keep only top-k edges)
	if hb.config.MaxEdges > 0 && len(homogGraph.Edges) > hb.config.MaxEdges {
		hb.filterTopKEdges(homogGraph)
	}
	
	// Remove self-loops if not desired (depends on configuration)
	if !hb.shouldKeepSelfLoops() {
		hb.removeSelfLoops(homogGraph)
	}
}

// filterTopKEdges keeps only the top-k highest weight edges
func (hb *HomogeneousBuilder) filterTopKEdges(homogGraph *HomogeneousGraph) {
	// Create slice of edge-weight pairs
	type EdgeWeight struct {
		Key    EdgeKey
		Weight float64
	}
	
	var edgeWeights []EdgeWeight
	for key, weight := range homogGraph.Edges {
		edgeWeights = append(edgeWeights, EdgeWeight{Key: key, Weight: weight})
	}
	
	// Sort by weight (descending)
	sort.Slice(edgeWeights, func(i, j int) bool {
		return edgeWeights[i].Weight > edgeWeights[j].Weight
	})
	
	// Keep only top-k edges
	newEdges := make(map[EdgeKey]float64)
	for i := 0; i < hb.config.MaxEdges && i < len(edgeWeights); i++ {
		ew := edgeWeights[i]
		newEdges[ew.Key] = ew.Weight
	}
	
	// Update graph and statistics
	filtered := len(homogGraph.Edges) - len(newEdges)
	hb.stats.EdgesFiltered += filtered
	homogGraph.Edges = newEdges
	
	// Recalculate node degrees
	hb.recalculateNodeDegrees(homogGraph)
}

// removeSelfLoops removes edges from a node to itself
func (hb *HomogeneousBuilder) removeSelfLoops(homogGraph *HomogeneousGraph) {
	for key := range homogGraph.Edges {
		if key.From == key.To {
			delete(homogGraph.Edges, key)
			hb.stats.EdgesFiltered++
		}
	}
	
	// Recalculate node degrees
	hb.recalculateNodeDegrees(homogGraph)
}

// recalculateNodeDegrees recalculates degree counts after edge filtering
func (hb *HomogeneousBuilder) recalculateNodeDegrees(homogGraph *HomogeneousGraph) {
	// Reset all degrees to 0
	for nodeID, node := range homogGraph.Nodes {
		node.Degree = 0
		homogGraph.Nodes[nodeID] = node
	}
	
	// Count degrees from remaining edges
	for edgeKey := range homogGraph.Edges {
		if fromNode, exists := homogGraph.Nodes[edgeKey.From]; exists {
			fromNode.Degree++
			homogGraph.Nodes[edgeKey.From] = fromNode
		}
		
		if toNode, exists := homogGraph.Nodes[edgeKey.To]; exists {
			toNode.Degree++
			homogGraph.Nodes[edgeKey.To] = toNode
		}
	}
}

// shouldKeepSelfLoops determines whether self-loops should be kept
func (hb *HomogeneousBuilder) shouldKeepSelfLoops() bool {
	// For most clustering algorithms, self-loops are not useful
	// But for some analyses (like node importance), they might be relevant
	return false // Default to removing self-loops
}

// updateWeightDistribution creates weight distribution statistics
func (hb *HomogeneousBuilder) updateWeightDistribution(homogGraph *HomogeneousGraph) {
	hb.stats.WeightDistribution = make(map[string]int)
	
	for _, weight := range homogGraph.Edges {
		// Create weight buckets for distribution
		bucket := hb.getWeightBucket(weight)
		hb.stats.WeightDistribution[bucket]++
	}
}

// getWeightBucket determines which bucket a weight falls into for distribution analysis
func (hb *HomogeneousBuilder) getWeightBucket(weight float64) string {
	if weight < 1.0 {
		return "0.0-1.0"
	} else if weight < 5.0 {
		return "1.0-5.0"
	} else if weight < 10.0 {
		return "5.0-10.0"
	} else if weight < 50.0 {
		return "10.0-50.0"
	} else if weight < 100.0 {
		return "50.0-100.0"
	} else {
		return "100.0+"
	}
}

// GetInstanceGroups returns the current instance groups (for debugging/analysis)
func (hb *HomogeneousBuilder) GetInstanceGroups() map[EdgeKey][]PathInstance {
	// Return a copy to prevent external modification
	result := make(map[EdgeKey][]PathInstance)
	for key, instances := range hb.instanceGroups {
		result[key] = make([]PathInstance, len(instances))
		copy(result[key], instances)
	}
	return result
}

// GetEdgeInstanceCount returns the number of instances for a specific edge
func (hb *HomogeneousBuilder) GetEdgeInstanceCount(from, to string) int {
	key := EdgeKey{From: from, To: to}
	return len(hb.instanceGroups[key])
}

// GetNodeCount returns the number of unique nodes
func (hb *HomogeneousBuilder) GetNodeCount() int {
	return len(hb.nodeSet)
}

// GetEdgeGroupCount returns the number of unique edge groups
func (hb *HomogeneousBuilder) GetEdgeGroupCount() int {
	return len(hb.instanceGroups)
}

// Reset clears all stored data (useful for reusing the builder)
func (hb *HomogeneousBuilder) Reset() {
	hb.instanceGroups = make(map[EdgeKey][]PathInstance)
	hb.nodeSet = make(map[string]bool)
	hb.stats = AggregationStats{
		WeightDistribution: make(map[string]int),
	}
}

// ValidateConfiguration checks if the aggregation configuration is valid
func (hb *HomogeneousBuilder) ValidateConfiguration() error {
	if hb.config.MinWeight < 0 {
		return MaterializationError{
			Component: "aggregation",
			Message:   "minimum weight cannot be negative",
			Details:   fmt.Sprintf("got: %.2f", hb.config.MinWeight),
		}
	}
	
	if hb.config.MaxEdges < 0 {
		return MaterializationError{
			Component: "aggregation",
			Message:   "maximum edges cannot be negative",
			Details:   fmt.Sprintf("got: %d", hb.config.MaxEdges),
		}
	}
	
	return nil
}

// GetAggregationStatistics returns current aggregation statistics
func (hb *HomogeneousBuilder) GetAggregationStatistics() AggregationStats {
	return hb.stats
}
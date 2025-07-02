package materialization

import (
	"fmt"
	"math"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// PathInstance represents a single instance of the meta path in the graph
type PathInstance struct {
	Nodes    []string               `json:"nodes"`    // Sequence of node IDs [a1, p1, a2]
	Edges    []string               `json:"edges"`    // Sequence of edge types used
	Weight   float64                `json:"weight"`   // Accumulated weight along path
	Metadata map[string]interface{} `json:"metadata"` // Additional information
}

// String returns a human-readable representation of the path instance
func (pi PathInstance) String() string {
	if len(pi.Nodes) == 0 {
		return "empty path"
	}
	
	result := pi.Nodes[0]
	for i, edge := range pi.Edges {
		if i+1 < len(pi.Nodes) {
			result += fmt.Sprintf(" -[%s]-> %s", edge, pi.Nodes[i+1])
		}
	}
	return fmt.Sprintf("%s (weight: %.2f)", result, pi.Weight)
}

// GetStartNode returns the first node in the path
func (pi PathInstance) GetStartNode() string {
	if len(pi.Nodes) == 0 {
		return ""
	}
	return pi.Nodes[0]
}

// GetEndNode returns the last node in the path
func (pi PathInstance) GetEndNode() string {
	if len(pi.Nodes) == 0 {
		return ""
	}
	return pi.Nodes[len(pi.Nodes)-1]
}

// IsValid checks if the path instance is structurally valid
func (pi PathInstance) IsValid() bool {
	return len(pi.Nodes) >= 2 && len(pi.Edges) == len(pi.Nodes)-1 && pi.Weight > 0
}

// HomogeneousGraph represents the materialized graph result
type HomogeneousGraph struct {
	NodeType   string                 `json:"node_type"`   // "Author" for symmetric paths
	Nodes      map[string]Node        `json:"nodes"`       // Nodes in result graph  
	Edges      map[EdgeKey]float64    `json:"edges"`       // (from,to) -> weight
	Statistics GraphStatistics        `json:"statistics"`  // Graph metrics
	MetaPath   models.MetaPath        `json:"meta_path"`   // Original meta path used
}

// Node represents a node in the homogeneous graph
type Node struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	Degree     int                    `json:"degree"`     // Number of connections
}

// EdgeKey represents a directed edge between two nodes
type EdgeKey struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// String returns a string representation of the edge key
func (ek EdgeKey) String() string {
	return fmt.Sprintf("%s -> %s", ek.From, ek.To)
}

// Reverse returns the reverse edge key
func (ek EdgeKey) Reverse() EdgeKey {
	return EdgeKey{From: ek.To, To: ek.From}
}

// GraphStatistics contains metrics about the homogeneous graph
type GraphStatistics struct {
	NodeCount          int                `json:"node_count"`
	EdgeCount          int                `json:"edge_count"`
	InstanceCount      int                `json:"instance_count"`      // Total meta path instances
	Density            float64            `json:"density"`             // Edge density
	AverageWeight      float64            `json:"average_weight"`      // Average edge weight
	MaxWeight          float64            `json:"max_weight"`          // Maximum edge weight
	MinWeight          float64            `json:"min_weight"`          // Minimum edge weight
	DegreeDistribution map[int]int        `json:"degree_distribution"` // degree -> count
	WeightDistribution map[string]int     `json:"weight_distribution"` // weight_bucket -> count
}

// TraversalStrategy defines how to traverse the meta path
type TraversalStrategy int

const (
	BFS TraversalStrategy = iota // Breadth-First Search
	DFS                          // Depth-First Search (for future extension)
)

// AggregationStrategy defines how to combine multiple path instances into edge weights
type AggregationStrategy int

const (
	Count   AggregationStrategy = iota // Count number of instances
	Sum                                // Sum of instance weights  
	Average                            // Average of instance weights
	Maximum                            // Maximum instance weight
	Minimum                            // Minimum instance weight
)

// NormalizationType defines edge weight normalization strategies
type NormalizationType int

const (
	NoNormalization NormalizationType = iota // No normalization
	DegreeNorm                               // Normalize by node degrees
	MaxNorm                                  // Normalize to [0,1] range
	StandardNorm                             // Z-score normalization
)

// TraversalConfig contains configuration for meta path traversal
type TraversalConfig struct {
	Strategy       TraversalStrategy `json:"strategy"`         // BFS, DFS, etc.
	MaxPathLength  int               `json:"max_path_length"`  // Prevent infinite recursion
	AllowCycles    bool              `json:"allow_cycles"`     // Whether to allow node revisits
	MaxInstances   int               `json:"max_instances"`    // Memory safety limit
	TimeoutSeconds int               `json:"timeout_seconds"`  // Processing timeout
	Parallelism    int               `json:"parallelism"`      // Number of parallel workers
}

// DefaultTraversalConfig returns sensible default configuration
func DefaultTraversalConfig() TraversalConfig {
	return TraversalConfig{
		Strategy:       BFS,
		MaxPathLength:  10,
		AllowCycles:    false,
		MaxInstances:   1000000, // 1M instances max
		TimeoutSeconds: 300,     // 5 minutes
		Parallelism:    4,       // 4 parallel workers
	}
}

// MetaPathInterpretation defines how the meta path is interpreted
type MetaPathInterpretation int

const (
	DirectTraversal  MetaPathInterpretation = iota // Alice → Paper → Bob = Alice ↔ Bob
	MeetingBased                                   // Alice → Venue, Bob → Venue = Alice ↔ Bob
)

// AggregationConfig contains configuration for edge weight calculation
type AggregationConfig struct {
	Strategy      AggregationStrategy `json:"strategy"`       // Count, Sum, Average, etc.
	Interpretation MetaPathInterpretation `json:"interpretation"` // DirectTraversal, MeetingBased
	Normalization NormalizationType   `json:"normalization"`  // None, Degree, Max, etc.
	MinWeight     float64             `json:"min_weight"`     // Filter weak edges
	MaxEdges      int                 `json:"max_edges"`      // Keep only top-k edges (0 = no limit)
	Symmetric     bool                `json:"symmetric"`      // Force symmetric edges
}

// DefaultAggregationConfig returns sensible default configuration
func DefaultAggregationConfig() AggregationConfig {
	return AggregationConfig{
		Strategy:      Count,
		Interpretation: DirectTraversal,
		Normalization: NoNormalization,
		MinWeight:     0.0,
		MaxEdges:      0,    // No limit
		Symmetric:     true, // Most clustering algorithms expect symmetric graphs
	}
}

// MaterializationConfig combines all configuration options
type MaterializationConfig struct {
	Traversal   TraversalConfig   `json:"traversal"`
	Aggregation AggregationConfig `json:"aggregation"`
	Progress    ProgressConfig    `json:"progress"`
}

// ProgressConfig contains configuration for progress reporting
type ProgressConfig struct {
	EnableProgress bool `json:"enable_progress"` // Whether to report progress
	ReportInterval int  `json:"report_interval"` // Report every N instances
}

// DefaultMaterializationConfig returns sensible defaults
func DefaultMaterializationConfig() MaterializationConfig {
	return MaterializationConfig{
		Traversal:   DefaultTraversalConfig(),
		Aggregation: DefaultAggregationConfig(),
		Progress: ProgressConfig{
			EnableProgress: true,
			ReportInterval: 10000,
		},
	}
}

// MaterializationResult contains the complete result of materialization
type MaterializationResult struct {
	HomogeneousGraph *HomogeneousGraph     `json:"homogeneous_graph"`
	Statistics       ProcessingStatistics  `json:"statistics"`
	Config           MaterializationConfig `json:"config"`
	Success          bool                  `json:"success"`
	Error            string                `json:"error,omitempty"`
}

// ProcessingStatistics contains detailed metrics about the materialization process
type ProcessingStatistics struct {
	RuntimeMS              int64                  `json:"runtime_ms"`
	MemoryPeakMB           int64                  `json:"memory_peak_mb"`
	InstancesGenerated     int                    `json:"instances_generated"`
	InstancesFiltered      int                    `json:"instances_filtered"`
	EdgesCreated           int                    `json:"edges_created"`
	NodesInResult          int                    `json:"nodes_in_result"`
	TraversalStatistics    TraversalStats         `json:"traversal_stats"`
	AggregationStatistics  AggregationStats       `json:"aggregation_stats"`
}

// TraversalStats contains statistics about the traversal process
type TraversalStats struct {
	StartingNodes      int                    `json:"starting_nodes"`
	NodesVisited       int                    `json:"nodes_visited"`
	EdgesTraversed     int                    `json:"edges_traversed"`
	PathsExplored      int                    `json:"paths_explored"`
	CyclesDetected     int                    `json:"cycles_detected"`
	TimeoutOccurred    bool                   `json:"timeout_occurred"`
	WorkerUtilization  map[int]int           `json:"worker_utilization"` // worker_id -> instances_processed
	RuntimeMS 		int64                 	 `json:"runtime_ms"`         // Total traversal time
}

// AggregationStats contains statistics about the aggregation process
type AggregationStats struct {
	EdgeGroupsProcessed    int                `json:"edge_groups_processed"`
	InstancesAggregated    int                `json:"instances_aggregated"`
	EdgesFiltered          int                `json:"edges_filtered"`
	WeightDistribution     map[string]int     `json:"weight_distribution"`
	NormalizationApplied   bool               `json:"normalization_applied"`
}

// ProgressCallback is a function type for progress reporting
type ProgressCallback func(current, total int, message string)

// ValidationError represents materialization-specific validation errors
type MaterializationError struct {
	Component string `json:"component"` // "traversal", "aggregation", "memory", etc.
	Message   string `json:"message"`
	Details   string `json:"details,omitempty"`
}

func (me MaterializationError) Error() string {
	if me.Details != "" {
		return fmt.Sprintf("materialization error in %s: %s (details: %s)", me.Component, me.Message, me.Details)
	}
	return fmt.Sprintf("materialization error in %s: %s", me.Component, me.Message)
}

// Helper methods for HomogeneousGraph

// AddNode adds a node to the homogeneous graph
func (hg *HomogeneousGraph) AddNode(nodeID string, originalNode models.Node) {
	if hg.Nodes == nil {
		hg.Nodes = make(map[string]Node)
	}
	
	hg.Nodes[nodeID] = Node{
		ID:         nodeID,
		Type:       originalNode.Type,
		Properties: originalNode.Properties,
		Degree:     0, // Will be calculated later
	}
}

// AddEdge adds an edge to the homogeneous graph
func (hg *HomogeneousGraph) AddEdge(from, to string, weight float64) {
	if hg.Edges == nil {
		hg.Edges = make(map[EdgeKey]float64)
	}
	
	key := EdgeKey{From: from, To: to}
	hg.Edges[key] = weight
	
	// Update node degrees
	if node, exists := hg.Nodes[from]; exists {
		node.Degree++
		hg.Nodes[from] = node
	}
	
	if node, exists := hg.Nodes[to]; exists {
		node.Degree++
		hg.Nodes[to] = node
	}
}

// GetWeight returns the weight of an edge, or 0 if it doesn't exist
func (hg *HomogeneousGraph) GetWeight(from, to string) float64 {
	key := EdgeKey{From: from, To: to}
	return hg.Edges[key]
}

// HasEdge checks if an edge exists
func (hg *HomogeneousGraph) HasEdge(from, to string) bool {
	key := EdgeKey{From: from, To: to}
	_, exists := hg.Edges[key]
	return exists
}

// GetNeighbors returns all neighbors of a node
func (hg *HomogeneousGraph) GetNeighbors(nodeID string) []string {
	var neighbors []string
	for edge := range hg.Edges {
		if edge.From == nodeID {
			neighbors = append(neighbors, edge.To)
		}
		if edge.To == nodeID {
			neighbors = append(neighbors, edge.From)
		}
	}
	return neighbors
}

// CalculateStatistics computes graph statistics
func (hg *HomogeneousGraph) CalculateStatistics() {
	hg.Statistics = GraphStatistics{
		NodeCount:          len(hg.Nodes),
		EdgeCount:          len(hg.Edges),
		DegreeDistribution: make(map[int]int),
		WeightDistribution: make(map[string]int),
	}
	
	// Calculate density
	n := float64(len(hg.Nodes))
	if n > 1 {
		hg.Statistics.Density = float64(len(hg.Edges)) / (n * (n - 1) / 2)
	}
	
	// Calculate weight statistics
	if len(hg.Edges) > 0 {
		totalWeight := 0.0
		hg.Statistics.MinWeight = math.Inf(1)
		hg.Statistics.MaxWeight = math.Inf(-1)
		
		for _, weight := range hg.Edges {
			totalWeight += weight
			if weight < hg.Statistics.MinWeight {
				hg.Statistics.MinWeight = weight
			}
			if weight > hg.Statistics.MaxWeight {
				hg.Statistics.MaxWeight = weight
			}
			
			// Weight distribution (bucket weights)
			bucket := fmt.Sprintf("%.1f", weight)
			hg.Statistics.WeightDistribution[bucket]++
		}
		
		hg.Statistics.AverageWeight = totalWeight / float64(len(hg.Edges))
	}
	
	// Calculate degree distribution
	for _, node := range hg.Nodes {
		hg.Statistics.DegreeDistribution[node.Degree]++
	}
}
package models

import (
	// "encoding/json"
	"fmt"
)

// HeterogeneousGraph represents the main graph structure
type HeterogeneousGraph struct {
	Nodes     map[string]Node    `json:"nodes"`      // node_id -> Node
	Edges     []Edge             `json:"edges"`      // All edges
	NodeTypes map[string]string  `json:"-"`          // node_id -> type (computed)
	EdgeTypes map[string]string  `json:"-"`          // edge_id -> type (computed)
}

// Node represents a single node in the heterogeneous graph
type Node struct {
	ID         string                 `json:"id,omitempty"`    // Optional in JSON if key is used
	Type       string                 `json:"type"`            // "Author", "Paper", "Venue"
	Properties map[string]interface{} `json:",inline"`         // Flatten other properties
}

// Edge represents a connection between two nodes
type Edge struct {
	From   string  `json:"from"`   // Source node ID
	To     string  `json:"to"`     // Target node ID
	Type   string  `json:"type"`   // "writes", "published_in", "cites"
	Weight float64 `json:"weight"` // Edge weight (default 1.0)
}

// MetaPath defines the pattern for traversing the heterogeneous graph
type MetaPath struct {
	ID           string   `json:"id"`              // Unique identifier
	NodeSequence []string `json:"node_sequence"`   // ["Author", "Paper", "Author"]
	EdgeSequence []string `json:"edge_sequence"`   // ["writes", "writes"]
	Description  string   `json:"description"`     // Human-readable description
}

// ClusteringRequest represents the API request structure
type ClusteringRequest struct {
	GraphFile    string                 `json:"graph_file"`
	MetaPathFile string                 `json:"meta_path_file"`
	Algorithm    string                 `json:"algorithm"`    // "materialized" or "sketch"
	Parameters   map[string]interface{} `json:"parameters"`
	OutputDir    string                 `json:"output_dir"`
}

// JobStatus represents the current status of a clustering job
type JobStatus struct {
	JobID              string  `json:"job_id"`
	Status             string  `json:"status"`              // "queued", "processing", "completed", "failed"
	Algorithm          string  `json:"algorithm"`
	Progress           int     `json:"progress"`            // 0-100
	EstimatedRemaining string  `json:"estimated_remaining"` // e.g., "15s"
	StartedAt          string  `json:"started_at"`
	Error              *string `json:"error,omitempty"`
}

// ClusteringResult represents the final clustering output
type ClusteringResult struct {
	JobID          string                 `json:"job_id"`
	Algorithm      string                 `json:"algorithm"`
	Communities    map[string]int         `json:"communities"`     // node_id -> community_id
	Modularity     float64                `json:"modularity"`
	NumCommunities int                    `json:"num_communities"`
	Statistics     ProcessingStats        `json:"statistics"`
}

// ProcessingStats contains detailed performance metrics
type ProcessingStats struct {
	RuntimeMS      int64                  `json:"runtime_ms"`
	MemoryPeakMB   int64                  `json:"memory_peak_mb"`
	Iterations     int                    `json:"iterations"`
	GraphStats     GraphStatistics        `json:"graph_stats"`
	AlgorithmStats interface{}            `json:"algorithm_stats"` // SketchStats or MaterializedStats
}

// GraphStatistics provides basic graph metrics
type GraphStatistics struct {
	TotalNodes          int            `json:"total_nodes"`
	TotalEdges          int            `json:"total_edges"`
	NodeTypes           map[string]int `json:"node_types"`
	EdgeTypes           map[string]int `json:"edge_types"`
	MetaPathInstances   int            `json:"meta_path_instances"`
	MetaPathLength      int            `json:"meta_path_length"`
	ConnectedComponents int            `json:"connected_components"`
}

// SketchStats contains SCAR algorithm specific metrics
type SketchStats struct {
	K                     int                `json:"k"`                       // Sketch size
	NK                    int                `json:"nk"`                      // Number of hash functions
	TotalSketchesComputed int                `json:"total_sketches_computed"`
	DegreeEstimates       map[string]float64 `json:"degree_estimates"`        // node_id -> estimated degree
	SketchIntersections   int                `json:"sketch_intersections"`    // Total intersections computed
	EstimationAccuracy    float64            `json:"estimation_accuracy"`     // Compared to materialized if available
}

// MaterializedStats contains traditional algorithm metrics
type MaterializedStats struct {
	PathInstancesGenerated int     `json:"path_instances_generated"`
	HomogeneousEdges      int     `json:"homogeneous_edges"`
	MaterializationTimeMS int64   `json:"materialization_time_ms"`
	LouvainIterations     int     `json:"louvain_iterations"`
	FinalModularity       float64 `json:"final_modularity"`
}

// ValidationError represents structured validation errors
type ValidationError struct {
	Field   string `json:"field"`
	Message string `json:"message"`
	Value   string `json:"value,omitempty"`
}

func (ve ValidationError) Error() string {
	if ve.Value != "" {
		return fmt.Sprintf("validation error in field '%s': %s (value: %s)", ve.Field, ve.Message, ve.Value)
	}
	return fmt.Sprintf("validation error in field '%s': %s", ve.Field, ve.Message)
}

// ValidationErrors is a collection of validation errors
type ValidationErrors []ValidationError

func (ve ValidationErrors) Error() string {
	if len(ve) == 0 {
		return "no validation errors"
	}
	if len(ve) == 1 {
		return ve[0].Error()
	}
	return fmt.Sprintf("%d validation errors: %s (and %d more)", len(ve), ve[0].Error(), len(ve)-1)
}

// Helper methods for HeterogeneousGraph

// PopulateTypeMaps fills in the NodeTypes and EdgeTypes maps from the actual data
func (hg *HeterogeneousGraph) PopulateTypeMaps() {
	hg.NodeTypes = make(map[string]string)
	hg.EdgeTypes = make(map[string]string)
	
	
	// Populate node types
	for nodeID, node := range hg.Nodes {
		hg.NodeTypes[nodeID] = node.Type
	}
	
	// Populate edge types (using from-to as key)
	for _, edge := range hg.Edges {
		edgeKey := fmt.Sprintf("%s-%s", edge.From, edge.To)
		hg.EdgeTypes[edgeKey] = edge.Type
	}
}

// GetNodesByType returns all nodes of a specific type
func (hg *HeterogeneousGraph) GetNodesByType(nodeType string) []Node {
	var nodes []Node
	for _, node := range hg.Nodes {
		if node.Type == nodeType {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// GetEdgesByType returns all edges of a specific type
func (hg *HeterogeneousGraph) GetEdgesByType(edgeType string) []Edge {
	var edges []Edge
	for _, edge := range hg.Edges {
		if edge.Type == edgeType {
			edges = append(edges, edge)
		}
	}
	return edges
}

// GetNeighbors returns all neighbors of a node with optional type filtering
func (hg *HeterogeneousGraph) GetNeighbors(nodeID string, edgeType string) []string {
	var neighbors []string
	for _, edge := range hg.Edges {
		if edge.From == nodeID && (edgeType == "" || edge.Type == edgeType) {
			neighbors = append(neighbors, edge.To)
		}
		// For undirected edges, also check reverse direction
		if edge.To == nodeID && (edgeType == "" || edge.Type == edgeType) {
			neighbors = append(neighbors, edge.From)
		}
	}
	return neighbors
}

// Helper methods for MetaPath

// Validate performs basic validation on the meta path structure
func (mp *MetaPath) Validate() error {
	if mp.ID == "" {
		return ValidationError{Field: "id", Message: "meta path ID cannot be empty"}
	}
	
	if len(mp.NodeSequence) < 2 {
		return ValidationError{Field: "node_sequence", Message: "meta path must have at least 2 nodes"}
	}
	
	if len(mp.EdgeSequence) != len(mp.NodeSequence)-1 {
		return ValidationError{
			Field:   "edge_sequence",
			Message: fmt.Sprintf("edge sequence length (%d) must be one less than node sequence length (%d)", 
				len(mp.EdgeSequence), len(mp.NodeSequence)),
		}
	}
	
	return nil
}

// IsSymmetric checks if the meta path is symmetric (starts and ends with same node type)
func (mp *MetaPath) IsSymmetric() bool {
	if len(mp.NodeSequence) < 2 {
		return false
	}
	return mp.NodeSequence[0] == mp.NodeSequence[len(mp.NodeSequence)-1]
}

// GetLength returns the length of the meta path (number of edges)
func (mp *MetaPath) GetLength() int {
	return len(mp.EdgeSequence)
}

// String returns a human-readable representation of the meta path
func (mp *MetaPath) String() string {
	if len(mp.NodeSequence) == 0 {
		return "empty meta path"
	}
	
	result := mp.NodeSequence[0]
	for i, edgeType := range mp.EdgeSequence {
		result += fmt.Sprintf(" -[%s]-> %s", edgeType, mp.NodeSequence[i+1])
	}
	return result
}
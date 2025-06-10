package louvain

import (
	"fmt"
	"math"
)

// HomogeneousGraph represents a weighted undirected graph
type HomogeneousGraph struct {
	Nodes       map[string]Node        `json:"nodes"`
	Edges       map[EdgeKey]float64    `json:"edges"`
	TotalWeight float64                `json:"total_weight"`
	NodeList    []string               `json:"-"` // Ordered list of node IDs for consistent iteration
}

// Node represents a node in the homogeneous graph
type Node struct {
	ID     string                 `json:"id"`
	Weight float64                `json:"weight"`
	Degree float64                `json:"degree"`
	Properties map[string]interface{} `json:"properties,omitempty"`
}

// EdgeKey represents a directed edge between two nodes
type EdgeKey struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// LouvainConfig contains configuration for the Louvain algorithm
type LouvainConfig struct {
	MaxCommunitySize  int              `json:"max_community_size"`
	MinModularity     float64          `json:"min_modularity"`
	MaxIterations     int              `json:"max_iterations"`
	NumWorkers        int              `json:"num_workers"`
	ChunkSize         int              `json:"chunk_size"`
	RandomSeed        int64            `json:"random_seed"`
	Verbose           bool             `json:"verbose"`
	ProgressCallback  ProgressCallback `json:"-"`
}

// LouvainState maintains the current state of community assignments
type LouvainState struct {
	Graph            *HomogeneousGraph
	Config           LouvainConfig
	N2C              map[string]int     // node -> community mapping
	C2N              map[int][]string   // community -> nodes mapping
	In               map[int]float64    // internal weight of each community.
	Tot              map[int]float64    // total weight of each community. 
	CommunityCounter int                // Counter for generating community IDs
	Iteration        int
}

// LouvainResult contains the complete result of Louvain clustering
type LouvainResult struct {
	Levels         []LevelInfo          `json:"levels"`
	FinalCommunities map[string]int     `json:"final_communities"`
	Modularity     float64              `json:"modularity"`
	NumLevels      int                  `json:"num_levels"`
	Statistics     LouvainStats         `json:"statistics"`
}

// LevelInfo contains information about one level in the hierarchy
type LevelInfo struct {
	Level          int                  `json:"level"`
	Communities    map[int][]string     `json:"communities"`     // community ID -> nodes
	CommunityMap   map[string]int       `json:"community_map"`   // node -> community
	Graph          *HomogeneousGraph    `json:"graph,omitempty"`
	Modularity     float64              `json:"modularity"`
	NumCommunities int                  `json:"num_communities"`
	NumMoves       int                  `json:"num_moves"`
	
}

// LouvainStats contains statistics about the algorithm execution
type LouvainStats struct {
	TotalIterations   int                `json:"total_iterations"`
	TotalMoves        int                `json:"total_moves"`
	RuntimeMS         int64              `json:"runtime_ms"`
	MemoryPeakMB      int64              `json:"memory_peak_mb"`
	LevelStats        []LevelStats       `json:"level_stats"`
}

// LevelStats contains statistics for a single level
type LevelStats struct {
	Level             int                `json:"level"`
	Iterations        int                `json:"iterations"`
	Moves             int                `json:"moves"`
	InitialModularity float64            `json:"initial_modularity"`
	FinalModularity   float64            `json:"final_modularity"`
	RuntimeMS         int64              `json:"runtime_ms"`
}

// ProgressCallback is a function type for progress reporting
type ProgressCallback func(level, iteration int, message string)

// NeighborWeight tracks weights to neighbor communities
type NeighborWeight struct {
	Community int
	Weight    float64
}

// DefaultLouvainConfig returns sensible default configuration
func DefaultLouvainConfig() LouvainConfig {
	return LouvainConfig{
		MaxCommunitySize: 0,        // No limit by default
		MinModularity:    0.001,   
		MaxIterations:    1,      
		NumWorkers:       4,
		ChunkSize:        32,
		RandomSeed:       -1,       // Use time-based seed
		Verbose:          false,
	}
}

// NewHomogeneousGraph creates a new empty homogeneous graph
func NewHomogeneousGraph() *HomogeneousGraph {
	return &HomogeneousGraph{
		Nodes:    make(map[string]Node),
		Edges:    make(map[EdgeKey]float64),
		NodeList: []string{},
	}
}

// AddNode adds a node to the graph
func (g *HomogeneousGraph) AddNode(nodeID string, weight float64) {
	if _, exists := g.Nodes[nodeID]; !exists {
		g.Nodes[nodeID] = Node{
			ID:     nodeID,
			Weight: weight,
			Degree: 0,
			Properties: make(map[string]interface{}),
		}
		g.NodeList = append(g.NodeList, nodeID)
	}
}

// AddEdge adds an undirected weighted edge to the graph
func (g *HomogeneousGraph) AddEdge(from, to string, weight float64) {
	// Add nodes if they don't exist
	if _, exists := g.Nodes[from]; !exists {
		fmt.Printf("Adding missing node: %s\n", from)
		g.AddNode(from, 1.0)
	}
	if _, exists := g.Nodes[to]; !exists {
		fmt.Printf("Adding missing node: %s\n", to)
		g.AddNode(to, 1.0)
	}
	
	// Add edge (undirected)
	g.Edges[EdgeKey{From: from, To: to}] = weight
	if from != to {
		g.Edges[EdgeKey{From: to, To: from}] = weight
	}
	
	// Update degrees
	fromNode := g.Nodes[from]
	fromNode.Degree += weight
	g.Nodes[from] = fromNode
	
	if from != to {
		toNode := g.Nodes[to]
		toNode.Degree += weight
		g.Nodes[to] = toNode
	} else {
		fromNode.Degree += weight // Self-loop case
		g.Nodes[from] = fromNode
	}	
	// Update total weight
	g.TotalWeight += weight
}

// GetNeighbors returns all neighbors of a node with their edge weights
func (g *HomogeneousGraph) GetNeighbors(nodeID string) map[string]float64 {
	neighbors := make(map[string]float64)
	
	// Check outgoing edges
	for edge, weight := range g.Edges {
		if edge.From == nodeID {
			neighbors[edge.To] = weight
		}
	}
	
	return neighbors
}

// GetNodeDegree returns the weighted degree of a node
func (g *HomogeneousGraph) GetNodeDegree(nodeID string) float64 {
	if node, exists := g.Nodes[nodeID]; exists {
		return node.Degree
	}
	return 0
}

// GetEdgeWeight returns the weight of an edge
func (g *HomogeneousGraph) GetEdgeWeight(from, to string) float64 {
	if weight, exists := g.Edges[EdgeKey{From: from, To: to}]; exists {
		return weight
	}
	return 0
}

// Clone creates a deep copy of the graph
func (g *HomogeneousGraph) Clone() *HomogeneousGraph {
	clone := NewHomogeneousGraph()
	clone.TotalWeight = g.TotalWeight
	
	// Copy nodes
	for id, node := range g.Nodes {
		cloneNode := Node{
			ID:     node.ID,
			Weight: node.Weight,
			Degree: node.Degree,
			Properties: make(map[string]interface{}),
		}
		for k, v := range node.Properties {
			cloneNode.Properties[k] = v
		}
		clone.Nodes[id] = cloneNode
	}
	
	// Copy node list
	clone.NodeList = make([]string, len(g.NodeList))
	copy(clone.NodeList, g.NodeList)
	
	// Copy edges
	for edge, weight := range g.Edges {
		clone.Edges[edge] = weight
	}
	
	return clone
}

// Validate checks if the graph is valid
func (g *HomogeneousGraph) Validate() error {
	// Check if graph is empty
	if len(g.Nodes) == 0 {
		return fmt.Errorf("graph has no nodes")
	}
	
	// Check edge references
	for edge := range g.Edges {
		if _, exists := g.Nodes[edge.From]; !exists {
			return fmt.Errorf("edge references non-existent node: %s", edge.From)
		}
		if _, exists := g.Nodes[edge.To]; !exists {
			return fmt.Errorf("edge references non-existent node: %s", edge.To)
		}
	}
	
	// Check symmetry (undirected graph)
	for edge, weight := range g.Edges {
		reverseEdge := EdgeKey{From: edge.To, To: edge.From}
		if edge.From != edge.To {
			if reverseWeight, exists := g.Edges[reverseEdge]; !exists || math.Abs(reverseWeight-weight) > 1e-9 {
				return fmt.Errorf("graph is not symmetric: edge %s->%s", edge.From, edge.To)
			}
		}
	}
	
	return nil
}

// GetModularity calculates the modularity of the current partition
func (s *LouvainState) GetModularity() float64 {
	if s.Graph.TotalWeight == 0 {
		return 0
	}
	
	
	q := 0.0
	m2 := 2 * s.Graph.TotalWeight  // Factor of 2 here, OR divide 'in' by 2
	
	for comm, tot := range s.Tot {
		if tot > 0 {
			in := s.In[comm]
			if in < 0 {
				fmt.Printf("WARNING: Negative internal weight for community %d: %f\n", comm, in)
				fmt.Printf("Community mapping: %v\n", s.C2N[comm])
				// print community mapping for all communities for debug and panic
				panic(fmt.Sprintf("Negative internal weight for community %d: %f", comm, in))
			}
			q += in/m2 - (tot/m2)*(tot/m2)
		}
	}

	return q
}

// String returns a string representation of an edge key
func (e EdgeKey) String() string {
	return fmt.Sprintf("%s->%s", e.From, e.To)
}
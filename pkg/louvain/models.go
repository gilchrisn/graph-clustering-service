package louvain

import (
	"fmt"
	"math"
)

// NormalizedGraph represents a weighted undirected graph with integer node indices
type NormalizedGraph struct {
	NumNodes     int         `json:"num_nodes"`
	Degrees      []float64   `json:"degrees"`      // Node degrees
	Weights      []float64   `json:"weights"`      // Node weights
	Adjacency    [][]int     `json:"-"`           // Adjacency list (neighbor indices)
	EdgeWeights  [][]float64 `json:"-"`           // Edge weights corresponding to adjacency
	TotalWeight  float64     `json:"total_weight"`
}

// HomogeneousGraph represents a weighted undirected graph (legacy interface)
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
	Graph            *NormalizedGraph
	Config           LouvainConfig
	N2C              []int              // node -> community mapping (array)
	C2N              map[int][]int      // community -> nodes mapping
	In               map[int]float64    // internal weight of each community
	Tot              map[int]float64    // total weight of each community
	CommunityCounter int                // Counter for generating community IDs
	Iteration        int
}

// LouvainResult contains the complete result of Louvain clustering
type LouvainResult struct {
	Levels         []LevelInfo          `json:"levels"`
	FinalCommunities map[int]int        `json:"final_communities"` // normalized node -> community
	Modularity     float64              `json:"modularity"`
	NumLevels      int                  `json:"num_levels"`
	Statistics     LouvainStats         `json:"statistics"`
	Parser         *GraphParser         `json:"-"` // For converting back to original IDs
}

// LevelInfo contains information about one level in the hierarchy
type LevelInfo struct {
	Level          int                  `json:"level"`
	Communities    map[int][]int        `json:"communities"`     // community ID -> normalized nodes
	CommunityMap   map[int]int          `json:"community_map"`   // normalized node -> community
	Graph          *NormalizedGraph     `json:"graph,omitempty"`
	Modularity     float64              `json:"modularity"`
	NumCommunities int                  `json:"num_communities"`
	NumMoves       int                  `json:"num_moves"`
	SuperNodeToCommMap map[int]int 		`json:"super_node_to_comm_map,omitempty"` // For hierarchical levels
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
		MaxIterations:    10,      
		NumWorkers:       4,
		ChunkSize:        32,
		RandomSeed:       -1,       // Use time-based seed
		Verbose:          false,
	}
}

// NewNormalizedGraph creates a new normalized graph with the given number of nodes
func NewNormalizedGraph(numNodes int) *NormalizedGraph {
	return &NormalizedGraph{
		NumNodes:    numNodes,
		Degrees:     make([]float64, numNodes),
		Weights:     make([]float64, numNodes),
		Adjacency:   make([][]int, numNodes),
		EdgeWeights: make([][]float64, numNodes),
		TotalWeight: 0,
	}
}

// AddEdge adds an undirected weighted edge to the normalized graph
func (g *NormalizedGraph) AddEdge(from, to int, weight float64) {
	if from < 0 || from >= g.NumNodes || to < 0 || to >= g.NumNodes {
		fmt.Printf("ERROR: Invalid node indices: %d, %d (numNodes: %d)\n", from, to, g.NumNodes)
		return
	}
	
	// Add to adjacency list

	g.Adjacency[from] = append(g.Adjacency[from], to)
	g.EdgeWeights[from] = append(g.EdgeWeights[from], weight)
	
	if from != to {
		g.Adjacency[to] = append(g.Adjacency[to], from)
		g.EdgeWeights[to] = append(g.EdgeWeights[to], weight)
	}
	
	// Update degrees
	g.Degrees[from] += weight
	if from != to {
		g.Degrees[to] += weight
	} else {
		g.Degrees[from] += weight // Self-loop case
	}
	
	// Update total weight
	g.TotalWeight += weight
	
	// Initialize node weights if not set
	if g.Weights[from] == 0 {
		g.Weights[from] = 1.0
	}
	if g.Weights[to] == 0 {
		g.Weights[to] = 1.0
	}
}

// GetNeighbors returns all neighbors of a node with their edge weights
func (g *NormalizedGraph) GetNeighbors(nodeID int) map[int]float64 {
	neighbors := make(map[int]float64)
	
	if nodeID < 0 || nodeID >= g.NumNodes {
		return neighbors
	}
	
	for i, neighbor := range g.Adjacency[nodeID] {
		neighbors[neighbor] = g.EdgeWeights[nodeID][i]
	}
	
	return neighbors
}

// GetNodeDegree returns the weighted degree of a node
func (g *NormalizedGraph) GetNodeDegree(nodeID int) float64 {
	if nodeID < 0 || nodeID >= g.NumNodes {
		return 0
	}
	return g.Degrees[nodeID]
}

// GetEdgeWeight returns the weight of an edge
func (g *NormalizedGraph) GetEdgeWeight(from, to int) float64 {
	if from < 0 || from >= g.NumNodes || to < 0 || to >= g.NumNodes {
		return 0
	}
	
	for i, neighbor := range g.Adjacency[from] {
		if neighbor == to {
			return g.EdgeWeights[from][i]
		}
	}
	return 0
}

// Clone creates a deep copy of the normalized graph
func (g *NormalizedGraph) Clone() *NormalizedGraph {
	clone := NewNormalizedGraph(g.NumNodes)
	clone.TotalWeight = g.TotalWeight
	
	// Copy arrays
	copy(clone.Degrees, g.Degrees)
	copy(clone.Weights, g.Weights)
	
	// Copy adjacency lists
	for i := 0; i < g.NumNodes; i++ {
		clone.Adjacency[i] = make([]int, len(g.Adjacency[i]))
		clone.EdgeWeights[i] = make([]float64, len(g.EdgeWeights[i]))
		copy(clone.Adjacency[i], g.Adjacency[i])
		copy(clone.EdgeWeights[i], g.EdgeWeights[i])
	}
	
	return clone
}

// Validate checks if the normalized graph is valid
func (g *NormalizedGraph) Validate() error {
	if g.NumNodes <= 0 {
		return fmt.Errorf("graph has no nodes")
	}
	
	// Check adjacency list consistency
	for i := 0; i < g.NumNodes; i++ {
		if len(g.Adjacency[i]) != len(g.EdgeWeights[i]) {
			return fmt.Errorf("adjacency and edge weights length mismatch for node %d", i)
		}
		
		for j, neighbor := range g.Adjacency[i] {
			if neighbor < 0 || neighbor >= g.NumNodes {
				return fmt.Errorf("invalid neighbor %d for node %d", neighbor, i)
			}
			
			weight := g.EdgeWeights[i][j]
			if weight < 0 {
				return fmt.Errorf("negative edge weight %f between nodes %d and %d", weight, i, neighbor)
			}
			
			// Check symmetry for undirected graph (except self-loops)
			if i != neighbor {
				found := false
				for k, reverseNeighbor := range g.Adjacency[neighbor] {
					if reverseNeighbor == i && math.Abs(g.EdgeWeights[neighbor][k]-weight) < 1e-9 {
						found = true
						break
					}
				}
				if !found {
					return fmt.Errorf("graph is not symmetric: edge %d->%d", i, neighbor)
				}
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
	m2 := 2 * s.Graph.TotalWeight
	
	for comm, tot := range s.Tot {
		if tot > 0 {
			in := s.In[comm]
			if in < 0 {
				fmt.Printf("WARNING: Negative internal weight for community %d: %f\n", comm, in)
				panic(fmt.Sprintf("Negative internal weight for community %d: %f", comm, in))
			}
			q += in/m2 - (tot/m2)*(tot/m2)
		}
	}

	return q
}

// Legacy HomogeneousGraph methods for backward compatibility

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
		g.AddNode(from, 1.0)
	}
	if _, exists := g.Nodes[to]; !exists {
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
		fromNode.Degree += weight
		g.Nodes[from] = fromNode
	}
	
	g.TotalWeight += weight
}

// GetNeighbors returns all neighbors of a node with their edge weights
func (g *HomogeneousGraph) GetNeighbors(nodeID string) map[string]float64 {
	neighbors := make(map[string]float64)
	
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
	
	clone.NodeList = make([]string, len(g.NodeList))
	copy(clone.NodeList, g.NodeList)
	
	for edge, weight := range g.Edges {
		clone.Edges[edge] = weight
	}
	
	return clone
}

// Validate checks if the graph is valid
func (g *HomogeneousGraph) Validate() error {
	if len(g.Nodes) == 0 {
		return fmt.Errorf("graph has no nodes")
	}
	
	for edge := range g.Edges {
		if _, exists := g.Nodes[edge.From]; !exists {
			return fmt.Errorf("edge references non-existent node: %s", edge.From)
		}
		if _, exists := g.Nodes[edge.To]; !exists {
			return fmt.Errorf("edge references non-existent node: %s", edge.To)
		}
	}
	
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

// String returns a string representation of an edge key
func (e EdgeKey) String() string {
	return fmt.Sprintf("%s->%s", e.From, e.To)
}
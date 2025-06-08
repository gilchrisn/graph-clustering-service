package scar

import (
	"fmt"
	// "math"
	"time"
)

// EdgeKey represents a unique edge identifier
type EdgeKey struct {
	From string
	To   string
}

// HeterogeneousGraph represents a heterogeneous information network
type HeterogeneousGraph struct {
	Nodes     map[string]HeteroNode
	Edges     map[EdgeKey]HeteroEdge
	NodeTypes map[string]string // nodeID -> node type
	EdgeTypes map[EdgeKey]string // edge -> edge type
	NodeList  []string          // for consistent iteration
}

// HeteroNode represents a node in heterogeneous graph
type HeteroNode struct {
	ID         string
	Type       string
	Properties map[string]interface{}
}

// HeteroEdge represents a typed edge
type HeteroEdge struct {
	From   string
	To     string
	Type   string
	Weight float64
}

// MetaPath represents a sequence of node and edge types
type MetaPath struct {
	NodeTypes []string // [Author, Paper, Author]
	EdgeTypes []string // [writes, writes]
}

// String returns a string representation of the meta-path
func (mp MetaPath) String() string {
	if len(mp.NodeTypes) == 0 {
		return ""
	}
	
	result := mp.NodeTypes[0]
	for i, edgeType := range mp.EdgeTypes {
		if i+1 < len(mp.NodeTypes) {
			result += fmt.Sprintf(" -[%s]-> %s", edgeType, mp.NodeTypes[i+1])
		}
	}
	return result
}

// IsValid checks if the meta-path is valid
func (mp MetaPath) IsValid() bool {
	return len(mp.NodeTypes) > 0 && len(mp.EdgeTypes) == len(mp.NodeTypes)-1
}

// Multi-sketch structure with nK independent hash functions
type VertexBottomKSketch struct {
	Sketches [][]uint64 // nK independent sketches, each of size K
	K        int        // sketch size
	NK       int        // number of independent hash functions
	PathPos  int        // position in meta-path (0 to path_length-1)
}

// Hash-to-node mapping for sketch-based adjacency discovery
type HashToNodeMap struct {
	Mapping map[uint64]string // hash value -> node ID
}

// Sophisticated degree estimation with saturated/undersaturated handling
type DegreeEstimate struct {
	Value      float64
	IsSaturated bool
	SketchSize int
}

// ProgressCallback allows monitoring of algorithm progress
type ProgressCallback func(level int, iteration int, modularity float64, nodes int)

// ScarConfig contains algorithm configuration
type ScarConfig struct {
	K                int              // sketch size (default: 512)
	NK               int              // number of independent hash functions (default: 8)
	MetaPath         MetaPath         // meta-path for community detection
	MaxIterations    int              // maximum iterations (default: 100)
	MinModularity    float64          // minimum modularity improvement (default: 1e-6)
	RandomSeed       int64            // for reproducibility
	Verbose          bool             // enable verbose output
	ProgressCallback ProgressCallback // optional progress callback
}

// DefaultScarConfig returns default configuration
func DefaultScarConfig() ScarConfig {
	return ScarConfig{
		K:             512,
		NK:            8,   // Added nK parameter
		MaxIterations: 100,
		MinModularity: 1e-6,
		RandomSeed:    42,
		Verbose:       false,
	}
}

// Enhanced state with hash-to-node mapping and multi-sketch support
type ScarState struct {
	Graph             *HeterogeneousGraph
	Config            ScarConfig
	N2C               map[string]int                      // node -> community
	C2N               map[int][]string                    // community -> nodes
	Sketches          map[string]*VertexBottomKSketch     // node -> multi-sketch
	CommunitySketches map[int]*VertexBottomKSketch        // community -> multi-sketch
	CommunityCounter  int
	Iteration         int
	HashToNodeMap     *HashToNodeMap                      // Added hash-to-node mapping
	
	// Enhanced degree tracking
	CommunityDegrees  map[int]*DegreeEstimate            // community -> degree estimate
	NodeDegrees       map[string]*DegreeEstimate         // node -> degree estimate
	
	// For tracking levels like Louvain
	CurrentLevel      int
	NodeToOriginal    map[string][]string                 // current node -> original nodes
	
	// Three-phase merging state
	MergePhase        int                                 // 0=initial, 1=quick, 2=sophisticated
}

// LevelInfo stores information about each level
type LevelInfo struct {
	Level         int
	Nodes         int
	Communities   int
	Modularity    float64
	Improvement   float64
	Iterations    int
	Duration      time.Duration
	N2C           map[string]int    // node -> community at this level
	C2N           map[int][]string  // community -> nodes at this level
	NodeMapping   map[string][]string // current node -> original nodes
}

// ScarStats contains algorithm statistics
type ScarStats struct {
	TotalLevels     int
	TotalIterations int
	TotalDuration   time.Duration
	FinalModularity float64
	FinalNodes      int
	FinalEdges      int
	InitialNodes    int
	InitialEdges    int
}

// ScarResult contains final results
type ScarResult struct {
	Levels           []LevelInfo
	FinalCommunities map[string]int // original node -> final community
	Modularity       float64
	NumLevels        int
	Statistics       ScarStats
	
	// For output compatibility with Louvain
	HierarchyLevels  []map[string][]string // level -> supernode -> children
	MappingLevels    []map[string][]string // level -> supernode -> leaf nodes
}

// CommunityInfo stores community statistics
type CommunityInfo struct {
	ID      int
	Size    int
	Degree  *DegreeEstimate // Enhanced degree estimation
	Members []string
}

// E-function calculation structure
type EFunctionResult struct {
	Value           float64
	C1Size          float64
	C2Size          float64
	IntersectK      float64
	ExpectedEdges   float64
	ActualEdges     float64
}

// GetNodesByType returns all nodes of a specific type
func (g *HeterogeneousGraph) GetNodesByType(nodeType string) []string {
	var nodes []string
	for nodeID, nType := range g.NodeTypes {
		if nType == nodeType {
			nodes = append(nodes, nodeID)
		}
	}
	return nodes
}

// GetNeighbors returns neighbors of a node with specific edge type
func (g *HeterogeneousGraph) GetNeighbors(nodeID string, edgeType string) []string {
	var neighbors []string
	
	// Check outgoing edges
	for edge, eType := range g.EdgeTypes {
		if edge.From == nodeID && eType == edgeType {
			neighbors = append(neighbors, edge.To)
		}
	}
	
	// Check incoming edges (for undirected behavior)
	for edge, eType := range g.EdgeTypes {
		if edge.To == nodeID && eType == edgeType {
			neighbors = append(neighbors, edge.From)
		}
	}
	
	return neighbors
}

// GetEdgeType returns the type of edge between two nodes
func (g *HeterogeneousGraph) GetEdgeType(from, to string) (string, bool) {
	key1 := EdgeKey{From: from, To: to}
	key2 := EdgeKey{From: to, To: from}
	
	if eType, exists := g.EdgeTypes[key1]; exists {
		return eType, true
	}
	if eType, exists := g.EdgeTypes[key2]; exists {
		return eType, true
	}
	
	return "", false
}

// HasEdge checks if an edge exists between two nodes
func (g *HeterogeneousGraph) HasEdge(from, to string) bool {
	key1 := EdgeKey{From: from, To: to}
	key2 := EdgeKey{From: to, To: from}
	
	_, exists1 := g.Edges[key1]
	_, exists2 := g.Edges[key2]
	
	return exists1 || exists2
}

// NewHeterogeneousGraph creates a new heterogeneous graph
func NewHeterogeneousGraph() *HeterogeneousGraph {
	return &HeterogeneousGraph{
		Nodes:     make(map[string]HeteroNode),
		Edges:     make(map[EdgeKey]HeteroEdge),
		NodeTypes: make(map[string]string),
		EdgeTypes: make(map[EdgeKey]string),
		NodeList:  make([]string, 0),
	}
}

// AddNode adds a node to the heterogeneous graph
func (g *HeterogeneousGraph) AddNode(node HeteroNode) {
	g.Nodes[node.ID] = node
	g.NodeTypes[node.ID] = node.Type
	g.NodeList = append(g.NodeList, node.ID)
}

// AddEdge adds an edge to the heterogeneous graph
func (g *HeterogeneousGraph) AddEdge(edge HeteroEdge) {
	key := EdgeKey{From: edge.From, To: edge.To}
	g.Edges[key] = edge
	g.EdgeTypes[key] = edge.Type
}

// NumNodes returns the number of nodes
func (g *HeterogeneousGraph) NumNodes() int {
	return len(g.Nodes)
}

// NumEdges returns the number of edges
func (g *HeterogeneousGraph) NumEdges() int {
	return len(g.Edges)
}

// Create new hash-to-node mapping
func NewHashToNodeMap() *HashToNodeMap {
	return &HashToNodeMap{
		Mapping: make(map[uint64]string),
	}
}

// Add hash value to node mapping
func (h *HashToNodeMap) AddMapping(hashValue uint64, nodeID string) {
	h.Mapping[hashValue] = nodeID
}

// Get node from hash value
func (h *HashToNodeMap) GetNode(hashValue uint64) (string, bool) {
	nodeID, exists := h.Mapping[hashValue]
	return nodeID, exists
}

// Check if hash exists
func (h *HashToNodeMap) HasHash(hashValue uint64) bool {
	_, exists := h.Mapping[hashValue]
	return exists
}
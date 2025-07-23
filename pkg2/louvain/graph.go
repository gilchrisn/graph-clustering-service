package louvain

import (
	"fmt"
	// "math"
)

// Graph represents a weighted undirected graph using simple arrays (NetworkX style)
type Graph struct {
	NumNodes    int         `json:"num_nodes"`
	Adjacency   [][]int     `json:"-"`              // adjacency[i] = list of neighbors of node i
	Weights     [][]float64 `json:"-"`              // weights[i][j] = weight of edge from node i to neighbor adjacency[i][j]
	Degrees     []float64   `json:"degrees"`        // degrees[i] = weighted degree of node i
	TotalWeight float64     `json:"total_weight"`   // sum of all edge weights
}

// NewGraph creates a new graph with n nodes
func NewGraph(numNodes int) *Graph {
	return &Graph{
		NumNodes:    numNodes,
		Adjacency:   make([][]int, numNodes),
		Weights:     make([][]float64, numNodes),
		Degrees:     make([]float64, numNodes),
		TotalWeight: 0.0,
	}
}

// AddEdge adds a weighted edge between two nodes
func (g *Graph) AddEdge(u, v int, weight float64) error {
	if u < 0 || u >= g.NumNodes || v < 0 || v >= g.NumNodes {
		return fmt.Errorf("node index out of range: u=%d, v=%d, numNodes=%d", u, v, g.NumNodes)
	}
	
	if weight <= 0 {
		return fmt.Errorf("edge weight must be positive: %f", weight)
	}
	
	// Add edge u -> v
	g.Adjacency[u] = append(g.Adjacency[u], v)
	g.Weights[u] = append(g.Weights[u], weight)
	g.Degrees[u] += weight
	
	// Add edge v -> u (undirected graph)
	if u != v {
		g.Adjacency[v] = append(g.Adjacency[v], u)
		g.Weights[v] = append(g.Weights[v], weight)
		g.Degrees[v] += weight
	} else {
		// Self-loop: count weight twice for degree
		g.Degrees[u] += weight
	}
	
	g.TotalWeight += weight
	return nil
}

// GetEdgeWeight returns the weight of edge between u and v
func (g *Graph) GetEdgeWeight(u, v int) float64 {
	if u < 0 || u >= g.NumNodes || v < 0 || v >= g.NumNodes {
		return 0.0
	}
	
	for i, neighbor := range g.Adjacency[u] {
		if neighbor == v {
			return g.Weights[u][i]
		}
	}
	return 0.0
}

// GetNeighbors returns neighbors and their edge weights for a node
func (g *Graph) GetNeighbors(node int) ([]int, []float64) {
	if node < 0 || node >= g.NumNodes {
		return nil, nil
	}
	return g.Adjacency[node], g.Weights[node]
}

// Clone creates a deep copy of the graph
func (g *Graph) Clone() *Graph {
	clone := NewGraph(g.NumNodes)
	clone.TotalWeight = g.TotalWeight
	copy(clone.Degrees, g.Degrees)
	
	for i := 0; i < g.NumNodes; i++ {
		clone.Adjacency[i] = make([]int, len(g.Adjacency[i]))
		clone.Weights[i] = make([]float64, len(g.Weights[i]))
		copy(clone.Adjacency[i], g.Adjacency[i])
		copy(clone.Weights[i], g.Weights[i])
	}
	
	return clone
}

// Validate checks graph consistency
func (g *Graph) Validate() error {
	if g.NumNodes <= 0 {
		return fmt.Errorf("graph must have positive number of nodes")
	}
	
	for i := 0; i < g.NumNodes; i++ {
		if len(g.Adjacency[i]) != len(g.Weights[i]) {
			return fmt.Errorf("adjacency and weights arrays inconsistent for node %d", i)
		}
		
		for j, neighbor := range g.Adjacency[i] {
			if neighbor < 0 || neighbor >= g.NumNodes {
				return fmt.Errorf("invalid neighbor %d for node %d", neighbor, i)
			}
			
			if g.Weights[i][j] <= 0 {
				return fmt.Errorf("non-positive weight %f for edge %d-%d", g.Weights[i][j], i, neighbor)
			}
		}
	}
	
	return nil
}
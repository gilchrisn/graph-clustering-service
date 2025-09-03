package coordinates

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/network"
	"gonum.org/v1/gonum/graph/simple"
)

// PageRankResult contains PageRank scores and derived metrics
type PageRankResult struct {
	Scores map[int64]float64 // gonum nodeID -> PageRank score
	MinScore float64
	MaxScore float64
}

// PageRankCalculator computes PageRank scores for graphs
type PageRankCalculator struct {
	dampingFactor float64
	tolerance     float64
}

// NewPageRankCalculator creates a new PageRank calculator
func NewPageRankCalculator() *PageRankCalculator {
	return &PageRankCalculator{
		dampingFactor: 0.85,  // Standard damping factor
		tolerance:     1e-6,  // Convergence tolerance
	}
}

// WithDampingFactor sets the damping factor (default: 0.85)
func (pr *PageRankCalculator) WithDampingFactor(factor float64) *PageRankCalculator {
	pr.dampingFactor = factor
	return pr
}

// WithTolerance sets the convergence tolerance (default: 1e-6)
func (pr *PageRankCalculator) WithTolerance(tolerance float64) *PageRankCalculator {
	pr.tolerance = tolerance
	return pr
}

// Calculate computes PageRank scores for the given graph
func (pr *PageRankCalculator) Calculate(g graph.Graph) (*PageRankResult, error) {
	if g.Nodes().Len() == 0 {
		return nil, fmt.Errorf("graph has no nodes")
	}

	// Convert to directed graph if needed for PageRank
	var directedGraph graph.Directed
	
	switch tg := g.(type) {
	case *simple.WeightedUndirectedGraph:
		// Convert undirected to directed by adding both directions
		directedGraph = pr.convertWeightedUndirectedToDirected(tg)
	case *simple.UndirectedGraph:
		// Convert undirected to directed by adding both directions
		directedGraph = pr.convertUndirectedToDirected(tg)
	case *simple.DirectedGraph:
		directedGraph = tg
	default:
		return nil, fmt.Errorf("unsupported graph type: %T", g)
	}

	// Compute PageRank using gonum's network package
	scores := network.PageRank(directedGraph, pr.dampingFactor, pr.tolerance)
	
	if len(scores) == 0 {
		return nil, fmt.Errorf("PageRank computation returned no scores")
	}

	// Find min and max scores for normalization
	var minScore, maxScore float64
	first := true
	
	for _, score := range scores {
		if first {
			minScore, maxScore = score, score
			first = false
		} else {
			if score < minScore {
				minScore = score
			}
			if score > maxScore {
				maxScore = score
			}
		}
	}

	return &PageRankResult{
		Scores:   scores,
		MinScore: minScore,
		MaxScore: maxScore,
	}, nil
}

func (pr *PageRankCalculator) convertWeightedUndirectedToDirected(weighted *simple.WeightedUndirectedGraph) *simple.WeightedDirectedGraph {
    directed := simple.NewWeightedDirectedGraph(0, math.Inf(1))
    
    // Add all nodes
    nodes := weighted.Nodes()
    for nodes.Next() {
        directed.AddNode(nodes.Node())
    }

    // Add edges in both directions with weights
    edges := weighted.WeightedEdges()
    for edges.Next() {
        edge := edges.WeightedEdge()
        from := edge.From()
        to := edge.To()
        weight := edge.Weight()
        
        // Add edge in both directions (undirected -> directed)
        directed.SetWeightedEdge(simple.WeightedEdge{F: from, T: to, W: weight})
        directed.SetWeightedEdge(simple.WeightedEdge{F: to, T: from, W: weight})
    }

    return directed
}

// convertUndirectedToDirected converts undirected graph to directed by adding both edge directions
func (pr *PageRankCalculator) convertUndirectedToDirected(undirected *simple.UndirectedGraph) *simple.DirectedGraph {
	directed := simple.NewDirectedGraph()
	
	// Add all nodes
	nodes := undirected.Nodes()
	for nodes.Next() {
		directed.AddNode(nodes.Node())
	}

	// Add edges in both directions
	edges := undirected.Edges()
	for edges.Next() {
		edge := edges.Edge()
		from := edge.From()
		to := edge.To()
		
		// Add edge in both directions (undirected -> directed)
		directed.SetEdge(simple.Edge{F: from, T: to})
		directed.SetEdge(simple.Edge{F: to, T: from})
	}

	return directed
}

// GetNormalizedScore returns a normalized PageRank score (0-1 range)
func (result *PageRankResult) GetNormalizedScore(nodeID int64) float64 {
	score, exists := result.Scores[nodeID]
	if !exists {
		return 0.0
	}

	if result.MaxScore == result.MinScore {
		return 1.0 // All nodes have same score
	}

	return (score - result.MinScore) / (result.MaxScore - result.MinScore)
}

// GetRadiusFromScore converts PageRank score to node radius for visualization
func (result *PageRankResult) GetRadiusFromScore(nodeID int64, minRadius, maxRadius float64) float64 {
	normalized := result.GetNormalizedScore(nodeID)
	return minRadius + normalized*(maxRadius-minRadius)
}
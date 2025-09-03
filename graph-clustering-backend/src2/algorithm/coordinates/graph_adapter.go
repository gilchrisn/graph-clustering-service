package coordinates

import (
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/graph/simple"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

// GraphAdapter converts different graph formats to gonum graph for coordinate generation
type GraphAdapter struct{}

// NewGraphAdapter creates a new graph adapter
func NewGraphAdapter() *GraphAdapter {
	return &GraphAdapter{}
}

// ConvertLouvainGraph converts louvain.Graph to gonum graph with node mapping
func (ga *GraphAdapter) ConvertLouvainGraph(louvainGraph *louvain.Graph, nodeIDs []int) (*simple.WeightedUndirectedGraph, map[int64]int, error) {
	if louvainGraph == nil {
		return nil, nil, fmt.Errorf("louvain graph is nil")
	}

	graph := simple.NewWeightedUndirectedGraph(0, math.Inf(1))
	nodeMapping := make(map[int64]int) // gonum nodeID -> original nodeID

	// Add all nodes first
	for i, originalID := range nodeIDs {
		gnodeID := int64(i)
		graph.AddNode(simple.Node(gnodeID))
		nodeMapping[gnodeID] = originalID
	}

	// Create reverse mapping for edge lookup
	originalToGonum := make(map[int]int64)
	for gnodeID, originalID := range nodeMapping {
		originalToGonum[originalID] = gnodeID
	}

	// Add edges using louvain graph adjacency
	for fromNode := 0; fromNode < louvainGraph.NumNodes; fromNode++ {
		if fromNode >= len(nodeIDs) {
			continue
		}

		neighbors, _ := louvainGraph.GetNeighbors(fromNode)
		for _, toNode := range neighbors {
			if toNode >= len(nodeIDs) {
				continue
			}

			fromGID, fromExists := originalToGonum[nodeIDs[fromNode]]
			toGID, toExists := originalToGonum[nodeIDs[toNode]]

			if fromExists && toExists && fromGID != toGID {
				// Only add if edge doesn't already exist
				if !graph.HasEdgeBetween(fromGID, toGID) {
					graph.SetWeightedEdge(simple.WeightedEdge{
						F: simple.Node(fromGID),
						T: simple.Node(toGID),
						W: 1.0,
					})
				}
			}
		}
	}

	return graph, nodeMapping, nil
}

// ConvertSCARGraph converts scar.SketchGraph to gonum graph with node mapping
func (ga *GraphAdapter) ConvertSCARGraph(sketchGraph *scar.SketchGraph, nodeIDs []int) (*simple.WeightedUndirectedGraph, map[int64]int, error) {
	if sketchGraph == nil {
		return nil, nil, fmt.Errorf("sketch graph is nil")
	}

	graph := simple.NewWeightedUndirectedGraph(0, math.Inf(1))
	nodeMapping := make(map[int64]int) // gonum nodeID -> original nodeID

	// Add all nodes first
	for i, originalID := range nodeIDs {
		gnodeID := int64(i)
		graph.AddNode(simple.Node(gnodeID))
		nodeMapping[gnodeID] = originalID
	}

	// Create reverse mapping for edge lookup
	originalToGonum := make(map[int]int64)
	for gnodeID, originalID := range nodeMapping {
		originalToGonum[originalID] = gnodeID
	}

	// Add edges using sketch graph operations
	for fromNode := 0; fromNode < len(nodeIDs); fromNode++ {
		neighbors, _ := sketchGraph.GetNeighbors(nodeIDs[fromNode])
		
		for _, toNodeOriginal := range neighbors {
			// Find gonum IDs
			fromGID, fromExists := originalToGonum[nodeIDs[fromNode]]
			toGID, toExists := originalToGonum[toNodeOriginal]

			if fromExists && toExists && fromGID != toGID {
				// Only add if edge doesn't already exist
				if !graph.HasEdgeBetween(fromGID, toGID) {
					graph.SetWeightedEdge(simple.WeightedEdge{
						F: simple.Node(fromGID),
						T: simple.Node(toGID),
						W: 1.0,
					})
				}
			}
		}
	}

	return graph, nodeMapping, nil
}

// BuildNodeList extracts node IDs from communities for a given level
func (ga *GraphAdapter) BuildNodeList(communities map[int][]int) []int {
	nodeSet := make(map[int]bool)
	
	// Collect all unique node IDs from communities
	for _, nodes := range communities {
		for _, nodeID := range nodes {
			nodeSet[nodeID] = true
		}
	}

	// Convert to slice with DETERMINISTIC ordering
	nodeList := make([]int, 0, len(nodeSet))
	for nodeID := range nodeSet {
		nodeList = append(nodeList, nodeID)
	}
	
	// SORT for deterministic ordering
	sort.Ints(nodeList)

	return nodeList
}


func (ga *GraphAdapter) ConvertSCARGraphWithInclusionExclusion(sketchGraph *scar.SketchGraph, nodeIDs []int, threshold float64) (*simple.WeightedUndirectedGraph, map[int64]int, error) {
    if sketchGraph == nil {
        return nil, nil, fmt.Errorf("sketch graph is nil")
    }

    graph := simple.NewWeightedUndirectedGraph(0, math.Inf(1))
    nodeMapping := make(map[int64]int)

    // Add all nodes first
    for i, originalID := range nodeIDs {
        gnodeID := int64(i)
        graph.AddNode(simple.Node(gnodeID))
        nodeMapping[gnodeID] = originalID
    }

	total := 0

    // For each node, get ALL neighbors using inclusion-exclusion
    for i, _ := range nodeIDs {
        // This will return ALL other nodes with inclusion-exclusion weights
        neighbors, weights := sketchGraph.GetNeighborsInclusionExclusion(i, threshold)


		total += len(neighbors)
        
        for j, neighborOriginalID := range neighbors {
            // Find the corresponding gonum node IDs
            fromGID := int64(i)
            
            // Find which index this neighbor corresponds to
            var toGID int64 = -1
            for k, nodeID := range nodeIDs {
                if nodeID == neighborOriginalID {
                    toGID = int64(k)
                    break
                }
            }
            
            if toGID != -1 && fromGID != toGID && !graph.HasEdgeBetween(fromGID, toGID) {
                graph.SetWeightedEdge(simple.WeightedEdge{
                    F: simple.Node(fromGID),
                    T: simple.Node(toGID),
                    W: weights[j],
                })
            }
        }
    }

	fmt.Printf("Total edges after inclusion-exclusion: %d\n", total)

    return graph, nodeMapping, nil
}
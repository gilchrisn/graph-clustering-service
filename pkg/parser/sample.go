package main

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/graph/network"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/mds"
)

// NodeVisual represents a node for visualization
type NodeVisual struct {
	ID       int64   `json:"id"`
	X        float64 `json:"x"`
	Y        float64 `json:"y"`
	Radius   float64 `json:"radius"`
	Label    string  `json:"label"`
	Level    int     `json:"level"`
	ClusterID int    `json:"cluster_id,omitempty"`
}

// HierarchicalGraph represents the graph at different clustering levels
type HierarchicalGraph struct {
	Levels       map[int]*simple.DirectedGraph // Level -> Graph
	ClusterMaps  map[int]map[int64]int          // Level -> (NodeID -> ClusterID)
	NodeMappings map[int]map[int][]int64        // Level -> (SupernodeID -> Original NodeIDs)
}

func main() {
	fmt.Println("🚀 Hierarchical PageRank + MDS Demo")
	fmt.Println("=" * 50)

	// 1. Create original graph
	originalGraph := createLargerDummyGraph()
	fmt.Printf("📈 Original graph: %d nodes, %d edges\n", 
		originalGraph.Nodes().Len(), originalGraph.Edges().Len())

	// 2. Create hierarchy with clustering
	hierarchy := createHierarchy(originalGraph)
	
	// 3. Run PageRank + MDS on each level
	visualizations := createHierarchicalVisualization(hierarchy)
	
	// 4. Display results
	displayResults(visualizations)
}

// createLargerDummyGraph creates a graph with clear cluster structure
func createLargerDummyGraph() *simple.DirectedGraph {
	g := simple.NewDirectedGraph()

	// Add 12 nodes
	for i := 0; i < 12; i++ {
		g.AddNode(simple.Node(i))
	}

	// Create 3 distinct clusters with some inter-cluster connections
	edges := []struct{ from, to int }{
		// Cluster 1 (nodes 0-3): Dense internal connections
		{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}, {1, 3},
		
		// Cluster 2 (nodes 4-7): Ring structure
		{4, 5}, {5, 6}, {6, 7}, {7, 4}, {4, 6}, // + diagonal
		
		// Cluster 3 (nodes 8-11): Star structure
		{8, 9}, {8, 10}, {8, 11}, {9, 8}, {10, 8},
		
		// Inter-cluster bridges (these create supernodes connections)
		{2, 4}, {3, 5},  // Cluster 1 -> Cluster 2
		{6, 8}, {7, 9},  // Cluster 2 -> Cluster 3
		{10, 1},         // Cluster 3 -> Cluster 1 (cycle)
	}

	for _, edge := range edges {
		g.SetEdge(simple.Edge{
			F: simple.Node(edge.from),
			T: simple.Node(edge.to),
		})
	}

	return g
}

// createHierarchy simulates Louvain clustering result
func createHierarchy(originalGraph *simple.DirectedGraph) HierarchicalGraph {
	hierarchy := HierarchicalGraph{
		Levels:       make(map[int]*simple.DirectedGraph),
		ClusterMaps:  make(map[int]map[int64]int),
		NodeMappings: make(map[int]map[int][]int64),
	}

	// Level 0: Original graph
	hierarchy.Levels[0] = originalGraph
	hierarchy.ClusterMaps[0] = make(map[int64]int)
	for i := int64(0); i < 12; i++ {
		hierarchy.ClusterMaps[0][i] = int(i) // Each node is its own cluster
	}

	// Level 1: Simulated Louvain clustering result
	// Cluster 0: nodes {0,1,2,3}
	// Cluster 1: nodes {4,5,6,7} 
	// Cluster 2: nodes {8,9,10,11}
	clustering := map[int64]int{
		0: 0, 1: 0, 2: 0, 3: 0,      // Cluster 0
		4: 1, 5: 1, 6: 1, 7: 1,      // Cluster 1  
		8: 2, 9: 2, 10: 2, 11: 2,    // Cluster 2
	}
	
	hierarchy.ClusterMaps[1] = clustering
	hierarchy.NodeMappings[1] = map[int][]int64{
		0: {0, 1, 2, 3},
		1: {4, 5, 6, 7},
		2: {8, 9, 10, 11},
	}

	// Create supernode graph for level 1
	hierarchy.Levels[1] = deriveSupernodeGraph(originalGraph, clustering)

	return hierarchy
}

// deriveSupernodeGraph creates a graph of supernodes based on clustering
func deriveSupernodeGraph(originalGraph *simple.DirectedGraph, clustering map[int64]int) *simple.DirectedGraph {
	supernodeGraph := simple.NewDirectedGraph()
	
	// Get unique cluster IDs
	clusters := make(map[int]bool)
	for _, clusterID := range clustering {
		clusters[clusterID] = true
	}
	
	// Add supernode for each cluster
	for clusterID := range clusters {
		supernodeGraph.AddNode(simple.Node(clusterID))
	}
	
	// Aggregate edges between clusters
	edgeWeights := make(map[struct{from, to int}]float64)
	
	edges := originalGraph.Edges()
	for edges.Next() {
		edge := edges.Edge()
		fromCluster := clustering[edge.From().ID()]
		toCluster := clustering[edge.To().ID()]
		
		// Only add edges between different clusters
		if fromCluster != toCluster {
			key := struct{from, to int}{fromCluster, toCluster}
			edgeWeights[key]++
		}
	}
	
	// Add aggregated edges to supernode graph
	for edgeKey, weight := range edgeWeights {
		if weight > 0 { // Only add if there are actual connections
			// For now, just add unweighted edges
			// In real implementation, you'd use weighted edges
			supernodeGraph.SetEdge(simple.Edge{
				F: simple.Node(edgeKey.from),
				T: simple.Node(edgeKey.to),
			})
		}
	}
	
	fmt.Printf("   Level 1 supernodes: %d nodes, %d edges\n",
		supernodeGraph.Nodes().Len(), supernodeGraph.Edges().Len())
	
	return supernodeGraph
}

// createHierarchicalVisualization runs PageRank + MDS on each level
func createHierarchicalVisualization(hierarchy HierarchicalGraph) map[int][]NodeVisual {
	visualizations := make(map[int][]NodeVisual)
	
	for level, graph := range hierarchy.Levels {
		fmt.Printf("\n🔍 Processing Level %d...\n", level)
		
		// Run PageRank
		pageRankScores := network.PageRank(graph, 0.85, 1e-6)
		fmt.Printf("   PageRank computed for %d nodes\n", len(pageRankScores))
		
		// Create distance matrix and apply MDS
		distMatrix := createDistanceMatrix(graph)
		coords := applyMDS(distMatrix)
		
		// Create visualization data
		visualizations[level] = createVisualizationData(pageRankScores, coords, level, hierarchy.NodeMappings[level])
		
		fmt.Printf("   Level %d: %d visual nodes created\n", level, len(visualizations[level]))
	}
	
	return visualizations
}

// Updated visualization data creation with level info
func createVisualizationData(pageRankScores map[int64]float64, coords *mat.Dense, level int, nodeMapping map[int][]int64) []NodeVisual {
	var nodes []NodeVisual
	
	// Find min/max PageRank for scaling
	minPR, maxPR := findMinMax(pageRankScores)
	if maxPR == minPR {
		maxPR = minPR + 1 // Avoid division by zero
	}

	for nodeID, score := range pageRankScores {
		// Scale radius based on PageRank
		normalizedScore := (score - minPR) / (maxPR - minPR)
		baseRadius := 5.0
		if level == 0 {
			baseRadius = 3.0 // Smaller for leaf nodes
		} else {
			baseRadius = 8.0 // Larger for supernodes
		}
		radius := baseRadius + normalizedScore*15.0

		// Create label based on level
		var label string
		if level == 0 {
			label = fmt.Sprintf("Node_%d", nodeID)
		} else {
			// For supernodes, show contained nodes
			if contained, exists := nodeMapping[int(nodeID)]; exists {
				label = fmt.Sprintf("Cluster_%d %v", nodeID, contained)
			} else {
				label = fmt.Sprintf("Cluster_%d", nodeID)
			}
		}

		node := NodeVisual{
			ID:        nodeID,
			X:         coords.At(int(nodeID), 0),
			Y:         coords.At(int(nodeID), 1),
			Radius:    radius,
			Label:     label,
			Level:     level,
			ClusterID: int(nodeID),
		}
		nodes = append(nodes, node)
	}

	return nodes
}

// Display results for each level
func displayResults(visualizations map[int][]NodeVisual) {
	fmt.Println("\n" + "=" * 60)
	fmt.Println("🎨 VISUALIZATION RESULTS")
	fmt.Println("=" * 60)
	
	for level := 0; level <= 1; level++ {
		if nodes, exists := visualizations[level]; exists {
			fmt.Printf("\n📊 LEVEL %d:\n", level)
			for _, node := range nodes {
				fmt.Printf("   %s: x=%6.2f, y=%6.2f, radius=%4.1f\n",
					node.Label, node.X, node.Y, node.Radius)
			}
		}
	}
	
	fmt.Println("\n🎯 SUCCESS!")
	fmt.Println("✅ Level 0: Individual nodes positioned and sized")
	fmt.Println("✅ Level 1: Clusters (supernodes) positioned and sized")
	fmt.Println("💡 Use this for multi-scale graph visualization!")
}

// Utility functions (same as before)

func createDistanceMatrix(g *simple.DirectedGraph) *mat.SymDense {
	nodes := g.Nodes()
	nodeList := make([]int64, 0)
	for nodes.Next() {
		nodeList = append(nodeList, nodes.Node().ID())
	}
	n := len(nodeList)

	distMatrix := mat.NewSymDense(n, nil)

	for i, nodeI := range nodeList {
		distances := bfsDistances(g, nodeI)
		for j, nodeJ := range nodeList {
			dist := distances[nodeJ]
			if dist == -1 {
				dist = float64(n) // Use graph diameter as max distance
			}
			distMatrix.SetSym(i, j, dist)
		}
	}

	return distMatrix
}

func bfsDistances(g *simple.DirectedGraph, source int64) map[int64]float64 {
	distances := make(map[int64]float64)
	visited := make(map[int64]bool)
	queue := []int64{source}
	
	distances[source] = 0
	visited[source] = true

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		neighbors := g.From(current)
		for neighbors.Next() {
			neighbor := neighbors.Node().ID()
			if !visited[neighbor] {
				visited[neighbor] = true
				distances[neighbor] = distances[current] + 1
				queue = append(queue, neighbor)
			}
		}
	}

	nodes := g.Nodes()
	for nodes.Next() {
		nodeID := nodes.Node().ID()
		if _, exists := distances[nodeID]; !exists {
			distances[nodeID] = -1
		}
	}

	return distances
}

func applyMDS(distMatrix *mat.SymDense) *mat.Dense {
	var coords mat.Dense
	var eigenvals []float64

	k, eig := mds.TorgersonScaling(&coords, eigenvals, distMatrix)
	
	fmt.Printf("   MDS: %d positive eigenvalues\n", k)

	return &coords
}

func findMinMax(scores map[int64]float64) (float64, float64) {
	var min, max float64
	first := true
	
	for _, score := range scores {
		if first {
			min, max = score, score
			first = false
		} else {
			if score < min {
				min = score
			}
			if score > max {
				max = score
			}
		}
	}
	
	return min, max
}
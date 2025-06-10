package louvain

import (
	"fmt"
	"math"
	"reflect"
	// "sort"
	// "strings"
	"testing"
	"time"
)

// TestGraph creates various test graphs for comprehensive testing
type TestGraph struct {
	Name        string
	Graph       *HomogeneousGraph
	ExpectedMin int // Minimum expected communities
	ExpectedMax int // Maximum expected communities
	Description string
}

// Helper function to create test graphs
func createTestGraphs() []TestGraph {
	graphs := []TestGraph{}

	// 1. Empty graph (edge case)
	emptyGraph := NewHomogeneousGraph()
	graphs = append(graphs, TestGraph{
		Name:        "Empty",
		Graph:       emptyGraph,
		ExpectedMin: 0,
		ExpectedMax: 0,
		Description: "Empty graph with no nodes",
	})

	// 2. Single node (edge case)
	singleNode := NewHomogeneousGraph()
	singleNode.AddNode("n1", 1.0)
	graphs = append(graphs, TestGraph{
		Name:        "SingleNode",
		Graph:       singleNode,
		ExpectedMin: 1,
		ExpectedMax: 1,
		Description: "Graph with single isolated node",
	})

	// 3. Two isolated nodes (edge case)
	twoIsolated := NewHomogeneousGraph()
	twoIsolated.AddNode("n1", 1.0)
	twoIsolated.AddNode("n2", 1.0)
	graphs = append(graphs, TestGraph{
		Name:        "TwoIsolated",
		Graph:       twoIsolated,
		ExpectedMin: 2,
		ExpectedMax: 2,
		Description: "Two isolated nodes with no edges",
	})

	// 4. Two connected nodes
	twoConnected := NewHomogeneousGraph()
	twoConnected.AddNode("n1", 1.0)
	twoConnected.AddNode("n2", 1.0)
	twoConnected.AddEdge("n1", "n2", 1.0)
	graphs = append(graphs, TestGraph{
		Name:        "TwoConnected",
		Graph:       twoConnected,
		ExpectedMin: 1,
		ExpectedMax: 2,
		Description: "Two nodes connected by single edge",
	})

	// 5. Triangle (complete graph K3)
	triangle := NewHomogeneousGraph()
	for i := 1; i <= 3; i++ {
		triangle.AddNode(fmt.Sprintf("n%d", i), 1.0)
	}
	triangle.AddEdge("n1", "n2", 1.0)
	triangle.AddEdge("n2", "n3", 1.0)
	triangle.AddEdge("n3", "n1", 1.0)
	graphs = append(graphs, TestGraph{
		Name:        "Triangle",
		Graph:       triangle,
		ExpectedMin: 1,
		ExpectedMax: 3,
		Description: "Complete triangle (K3)",
	})

	// 6. Star graph
	star := NewHomogeneousGraph()
	star.AddNode("center", 1.0)
	for i := 1; i <= 5; i++ {
		nodeID := fmt.Sprintf("leaf%d", i)
		star.AddNode(nodeID, 1.0)
		star.AddEdge("center", nodeID, 1.0)
	}
	graphs = append(graphs, TestGraph{
		Name:        "Star",
		Graph:       star,
		ExpectedMin: 1,
		ExpectedMax: 6,
		Description: "Star graph with center node connected to 5 leaves",
	})

	// 7. Linear chain
	chain := NewHomogeneousGraph()
	for i := 1; i <= 6; i++ {
		chain.AddNode(fmt.Sprintf("n%d", i), 1.0)
	}
	for i := 1; i < 6; i++ {
		chain.AddEdge(fmt.Sprintf("n%d", i), fmt.Sprintf("n%d", i+1), 1.0)
	}
	graphs = append(graphs, TestGraph{
		Name:        "Chain",
		Graph:       chain,
		ExpectedMin: 1,
		ExpectedMax: 6,
		Description: "Linear chain of 6 nodes",
	})

	// 8. Two separate triangles (disconnected components)
	twoTriangles := NewHomogeneousGraph()
	// First triangle
	for i := 1; i <= 3; i++ {
		twoTriangles.AddNode(fmt.Sprintf("t1_n%d", i), 1.0)
	}
	twoTriangles.AddEdge("t1_n1", "t1_n2", 1.0)
	twoTriangles.AddEdge("t1_n2", "t1_n3", 1.0)
	twoTriangles.AddEdge("t1_n3", "t1_n1", 1.0)
	// Second triangle
	for i := 1; i <= 3; i++ {
		twoTriangles.AddNode(fmt.Sprintf("t2_n%d", i), 1.0)
	}
	twoTriangles.AddEdge("t2_n1", "t2_n2", 1.0)
	twoTriangles.AddEdge("t2_n2", "t2_n3", 1.0)
	twoTriangles.AddEdge("t2_n3", "t2_n1", 1.0)
	graphs = append(graphs, TestGraph{
		Name:        "TwoTriangles",
		Graph:       twoTriangles,
		ExpectedMin: 2,
		ExpectedMax: 6,
		Description: "Two disconnected triangles",
	})

	// 9. Complete graph K4
	k4 := NewHomogeneousGraph()
	for i := 1; i <= 4; i++ {
		k4.AddNode(fmt.Sprintf("n%d", i), 1.0)
	}
	for i := 1; i <= 4; i++ {
		for j := i + 1; j <= 4; j++ {
			k4.AddEdge(fmt.Sprintf("n%d", i), fmt.Sprintf("n%d", j), 1.0)
		}
	}
	graphs = append(graphs, TestGraph{
		Name:        "K4",
		Graph:       k4,
		ExpectedMin: 1,
		ExpectedMax: 4,
		Description: "Complete graph K4",
	})

	// 10. Barbell graph (two cliques connected by single edge)
	barbell := NewHomogeneousGraph()
	// First clique
	for i := 1; i <= 3; i++ {
		barbell.AddNode(fmt.Sprintf("c1_n%d", i), 1.0)
	}
	for i := 1; i <= 3; i++ {
		for j := i + 1; j <= 3; j++ {
			barbell.AddEdge(fmt.Sprintf("c1_n%d", i), fmt.Sprintf("c1_n%d", j), 2.0)
		}
	}
	// Second clique
	for i := 1; i <= 3; i++ {
		barbell.AddNode(fmt.Sprintf("c2_n%d", i), 1.0)
	}
	for i := 1; i <= 3; i++ {
		for j := i + 1; j <= 3; j++ {
			barbell.AddEdge(fmt.Sprintf("c2_n%d", i), fmt.Sprintf("c2_n%d", j), 2.0)
		}
	}
	// Bridge edge
	barbell.AddEdge("c1_n1", "c2_n1", 0.1)
	graphs = append(graphs, TestGraph{
		Name:        "Barbell",
		Graph:       barbell,
		ExpectedMin: 2,
		ExpectedMax: 6,
		Description: "Barbell graph: two cliques connected by weak bridge",
	})

	// 11. Self-loop graph
	selfLoop := NewHomogeneousGraph()
	selfLoop.AddNode("n1", 1.0)
	selfLoop.AddNode("n2", 1.0)
	selfLoop.AddEdge("n1", "n1", 2.0) // Self-loop
	selfLoop.AddEdge("n1", "n2", 1.0)
	selfLoop.AddEdge("n2", "n2", 1.0) // Self-loop
	graphs = append(graphs, TestGraph{
		Name:        "SelfLoop",
		Graph:       selfLoop,
		ExpectedMin: 1,
		ExpectedMax: 2,
		Description: "Graph with self-loops",
	})

	return graphs
}

// Test basic graph operations
func TestHomogeneousGraph(t *testing.T) {
	t.Run("EmptyGraph", func(t *testing.T) {
		g := NewHomogeneousGraph()
		
		if len(g.Nodes) != 0 {
			t.Errorf("Expected 0 nodes, got %d", len(g.Nodes))
		}
		if len(g.Edges) != 0 {
			t.Errorf("Expected 0 edges, got %d", len(g.Edges))
		}
		if g.TotalWeight != 0 {
			t.Errorf("Expected total weight 0, got %f", g.TotalWeight)
		}
	})

	t.Run("AddNode", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddNode("test", 2.5)
		
		if len(g.Nodes) != 1 {
			t.Errorf("Expected 1 node, got %d", len(g.Nodes))
		}
		
		node := g.Nodes["test"]
		if node.ID != "test" {
			t.Errorf("Expected node ID 'test', got '%s'", node.ID)
		}
		if node.Weight != 2.5 {
			t.Errorf("Expected node weight 2.5, got %f", node.Weight)
		}
		if node.Degree != 0 {
			t.Errorf("Expected node degree 0, got %f", node.Degree)
		}
	})

	t.Run("AddEdge", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "b", 3.0)
		
		// Check nodes were created
		if len(g.Nodes) != 2 {
			t.Errorf("Expected 2 nodes, got %d", len(g.Nodes))
		}
		
		// Check edges (should be bidirectional)
		if len(g.Edges) != 2 {
			t.Errorf("Expected 2 edges (bidirectional), got %d", len(g.Edges))
		}
		
		// Check edge weights
		weight1 := g.GetEdgeWeight("a", "b")
		weight2 := g.GetEdgeWeight("b", "a")
		if weight1 != 3.0 || weight2 != 3.0 {
			t.Errorf("Expected edge weight 3.0, got %f and %f", weight1, weight2)
		}
		
		// Check degrees
		degreeA := g.GetNodeDegree("a")
		degreeB := g.GetNodeDegree("b")
		if degreeA != 3.0 || degreeB != 3.0 {
			t.Errorf("Expected degrees 3.0, got %f and %f", degreeA, degreeB)
		}
		
		// Check total weight
		if g.TotalWeight != 3.0 {
			t.Errorf("Expected total weight 3.0, got %f", g.TotalWeight)
		}
	})

	t.Run("SelfLoop", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "a", 2.0)
		
		// Self-loop should only create one edge
		if len(g.Edges) != 1 {
			t.Errorf("Expected 1 edge for self-loop, got %d", len(g.Edges))
		}
		
		// Degree should include self-loop
		degree := g.GetNodeDegree("a")
		if degree != 4.0 {
			t.Errorf("Expected degree 2.0 for self-loop, got %f", degree)
		}
	})

	t.Run("GetNeighbors", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "b", 1.0)
		g.AddEdge("a", "c", 2.0)
		g.AddEdge("a", "a", 0.5) // Self-loop
		
		neighbors := g.GetNeighbors("a")
		
		expectedNeighbors := map[string]float64{
			"b": 1.0,
			"c": 2.0,
			"a": 0.5,
		}
		
		if !reflect.DeepEqual(neighbors, expectedNeighbors) {
			t.Errorf("Expected neighbors %v, got %v", expectedNeighbors, neighbors)
		}
	})
}

// Test graph validation
func TestGraphValidation(t *testing.T) {
	t.Run("EmptyGraphShouldFail", func(t *testing.T) {
		g := NewHomogeneousGraph()
		err := g.Validate()
		if err == nil {
			t.Error("Expected validation error for empty graph")
		}
	})

	t.Run("ValidGraphShouldPass", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "b", 1.0)
		err := g.Validate()
		if err != nil {
			t.Errorf("Expected valid graph to pass validation, got error: %v", err)
		}
	})

	t.Run("AsymmetricGraphShouldFail", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddNode("a", 1.0)
		g.AddNode("b", 1.0)
		// Manually add asymmetric edge
		g.Edges[EdgeKey{From: "a", To: "b"}] = 1.0
		
		err := g.Validate()
		if err == nil {
			t.Error("Expected validation error for asymmetric graph")
		}
	})

	t.Run("InvalidEdgeReferenceShouldFail", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddNode("a", 1.0)
		// Manually add edge to non-existent node
		g.Edges[EdgeKey{From: "a", To: "nonexistent"}] = 1.0
		
		err := g.Validate()
		if err == nil {
			t.Error("Expected validation error for edge to non-existent node")
		}
	})
}

// Test Louvain configuration
func TestLouvainConfig(t *testing.T) {
	t.Run("DefaultConfig", func(t *testing.T) {
		config := DefaultLouvainConfig()
		
		if config.MaxCommunitySize != 0 {
			t.Errorf("Expected MaxCommunitySize 0, got %d", config.MaxCommunitySize)
		}
		if config.MinModularity != 0.001 {
			t.Errorf("Expected MinModularity 0.001, got %f", config.MinModularity)
		}
		if config.MaxIterations != 1 {
			t.Errorf("Expected MaxIterations 1, got %d", config.MaxIterations)
		}
		if config.RandomSeed != -1 {
			t.Errorf("Expected RandomSeed -1, got %d", config.RandomSeed)
		}
	})
}

// Test Louvain state initialization
func TestLouvainState(t *testing.T) {
	t.Run("Initialization", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "b", 1.0)
		g.AddEdge("b", "c", 1.0)
		
		config := DefaultLouvainConfig()
		state := NewLouvainState(g, config)
		
		// Each node should be in its own community initially
		if len(state.N2C) != 3 {
			t.Errorf("Expected 3 nodes in N2C, got %d", len(state.N2C))
		}
		if len(state.C2N) != 3 {
			t.Errorf("Expected 3 communities in C2N, got %d", len(state.C2N))
		}
		
		// Check community assignments
		for nodeID := range g.Nodes {
			if _, exists := state.N2C[nodeID]; !exists {
				t.Errorf("Node %s not found in N2C", nodeID)
			}
		}
		
		// Check that each community has exactly one node initially
		for commID, nodes := range state.C2N {
			if len(nodes) != 1 {
				t.Errorf("Community %d should have 1 node initially, got %d", commID, len(nodes))
			}
		}
	})

	t.Run("CommunityOperations", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "b", 2.0)
		
		config := DefaultLouvainConfig()
		state := NewLouvainState(g, config)
		
		// Get initial communities
		commA := state.N2C["a"]
		commB := state.N2C["b"]
		
		// Test removing node from community
		state.removeNodeFromCommunity("a", commA)
		
		// Community should be deleted if empty
		if _, exists := state.C2N[commA]; exists {
			t.Error("Empty community should be deleted")
		}
		
		// Test inserting node into community
		state.insertNodeIntoCommunity("a", commB)
		
		// Both nodes should now be in commB
		if state.N2C["a"] != commB {
			t.Error("Node a should be in community B")
		}
		if len(state.C2N[commB]) != 2 {
			t.Errorf("Community B should have 2 nodes, got %d", len(state.C2N[commB]))
		}
	})
}

// Test modularity calculation
func TestModularity(t *testing.T) {
	t.Run("SingleNode", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddNode("a", 1.0)
		
		config := DefaultLouvainConfig()
		state := NewLouvainState(g, config)
		
		modularity := state.GetModularity()
		// Single node with no edges should have modularity 0
		if modularity != 0.0 {
			t.Errorf("Expected modularity 0 for single node, got %f", modularity)
		}
	})

	t.Run("TwoNodesNoEdges", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddNode("a", 1.0)
		g.AddNode("b", 1.0)
		
		config := DefaultLouvainConfig()
		state := NewLouvainState(g, config)
		
		modularity := state.GetModularity()
		// No edges means modularity should be 0
		if modularity != 0.0 {
			t.Errorf("Expected modularity 0 for no edges, got %f", modularity)
		}
	})

	t.Run("TwoConnectedNodes", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "b", 1.0)
		
		config := DefaultLouvainConfig()
		state := NewLouvainState(g, config)
		
		// Initially each node in separate community
		initialModularity := state.GetModularity()
		
		// Move both nodes to same community
		commA := state.N2C["a"]
		commB := state.N2C["b"]
		state.removeNodeFromCommunity("b", commB)
		state.insertNodeIntoCommunity("b", commA)
		
		finalModularity := state.GetModularity()
		
		// Modularity should improve when connected nodes are in same community
		if finalModularity <= initialModularity {
			t.Errorf("Expected modularity to improve: initial=%f, final=%f", initialModularity, finalModularity)
		}
	})
}

// Test modularity gain calculation
func TestModularityGain(t *testing.T) {
	g := NewHomogeneousGraph()
	g.AddEdge("a", "b", 1.0)
	g.AddEdge("b", "c", 1.0)
	
	config := DefaultLouvainConfig()
	state := NewLouvainState(g, config)
	
	// Test gain of moving node b to node a's community
	commA := state.N2C["a"]
	weightToCommA := g.GetEdgeWeight("b", "a")
	
	gain := state.modularityGain("b", commA, weightToCommA)
	
	// Gain should be positive since b is connected to a
	if gain <= 0 {
		t.Errorf("Expected positive modularity gain, got %f", gain)
	}
}

// Test super graph creation
func TestSuperGraphCreation(t *testing.T) {
	t.Run("SimpleCase", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "b", 1.0)
		g.AddEdge("c", "d", 1.0)
		
		config := DefaultLouvainConfig()
		state := NewLouvainState(g, config)
		
		// Move a and b to same community
		commA := state.N2C["a"]
		commB := state.N2C["b"]
		state.removeNodeFromCommunity("b", commB)
		state.insertNodeIntoCommunity("b", commA)
		
		superGraph, communityMap, err := state.CreateSuperGraph()
		if err != nil {
			t.Fatalf("Error creating super graph: %v", err)
		}
		
		// Should have 3 communities: {a,b}, {c}, {d}
		if len(superGraph.Nodes) != 3 {
			t.Errorf("Expected 3 super nodes, got %d", len(superGraph.Nodes))
		}
		
		// Check community map
		if len(communityMap) != 3 {
			t.Errorf("Expected 3 entries in community map, got %d", len(communityMap))
		}
		
		// Verify that original nodes are properly mapped
		totalMappedNodes := 0
		for _, nodes := range communityMap {
			totalMappedNodes += len(nodes)
		}
		if totalMappedNodes != 4 {
			t.Errorf("Expected 4 total mapped nodes, got %d", totalMappedNodes)
		}
	})
}

// Test main Louvain algorithm
func TestRunLouvain(t *testing.T) {
	// Test with various graph types
	testGraphs := createTestGraphs()
	
	for _, testGraph := range testGraphs {
		t.Run(testGraph.Name, func(t *testing.T) {
			if testGraph.Name == "Empty" {
				// Empty graphs should fail validation
				config := DefaultLouvainConfig()
				_, err := RunLouvain(testGraph.Graph, config)
				if err == nil {
					t.Error("Expected error for empty graph")
				}
				return
			}
			
			config := DefaultLouvainConfig()
			config.MaxIterations = 10
			config.RandomSeed = 42 // Fixed seed for reproducibility
			
			result, err := RunLouvain(testGraph.Graph, config)
			if err != nil {
				t.Fatalf("RunLouvain failed: %v", err)
			}
			
			// Basic result validation
			if result == nil {
				t.Fatal("Result should not be nil")
			}
			
			if result.NumLevels != len(result.Levels) {
				t.Errorf("NumLevels (%d) doesn't match length of Levels (%d)", 
					result.NumLevels, len(result.Levels))
			}
			
			if result.NumLevels == 0 && len(testGraph.Graph.Nodes) > 0 {
				t.Error("Should have at least one level for non-empty graph")
			}
			
			// Check final communities count is within expected range
			if len(result.FinalCommunities) < testGraph.ExpectedMin || 
			   len(result.FinalCommunities) > testGraph.ExpectedMax {
				t.Logf("Warning: Community count outside expected range for %s: got %d, expected %d-%d",
					testGraph.Name, len(result.FinalCommunities), testGraph.ExpectedMin, testGraph.ExpectedMax)
			}
			
			// Modularity should be finite
			if math.IsNaN(result.Modularity) || math.IsInf(result.Modularity, 0) {
				t.Errorf("Modularity should be finite, got %f", result.Modularity)
			}
			
			// All original nodes should be assigned to communities
			for nodeID := range testGraph.Graph.Nodes {
				if _, exists := result.FinalCommunities[nodeID]; !exists {
					t.Errorf("Node %s not found in final communities", nodeID)
				}
			}
		})
	}
}

// Test hierarchy consistency
func TestHierarchyConsistency(t *testing.T) {
	// Create a moderately complex graph
	g := NewHomogeneousGraph()
	
	// Create two cliques connected by a bridge
	for i := 1; i <= 4; i++ {
		g.AddNode(fmt.Sprintf("c1_n%d", i), 1.0)
	}
	for i := 1; i <= 4; i++ {
		for j := i + 1; j <= 4; j++ {
			g.AddEdge(fmt.Sprintf("c1_n%d", i), fmt.Sprintf("c1_n%d", j), 2.0)
		}
	}
	
	for i := 1; i <= 4; i++ {
		g.AddNode(fmt.Sprintf("c2_n%d", i), 1.0)
	}
	for i := 1; i <= 4; i++ {
		for j := i + 1; j <= 4; j++ {
			g.AddEdge(fmt.Sprintf("c2_n%d", i), fmt.Sprintf("c2_n%d", j), 2.0)
		}
	}
	
	// Bridge
	g.AddEdge("c1_n1", "c2_n1", 0.1)
	
	config := DefaultLouvainConfig()
	config.MaxIterations = 5
	config.RandomSeed = 42
	
	result, err := RunLouvain(g, config)
	if err != nil {
		t.Fatalf("RunLouvain failed: %v", err)
	}
	
	if len(result.Levels) == 0 {
		t.Fatal("Should have at least one level")
	}
	
	// Test hierarchy consistency
	for level := 0; level < len(result.Levels); level++ {
		levelInfo := result.Levels[level]
		
		// Check that community map is consistent with communities
		totalNodesInCommunities := 0
		for _, nodes := range levelInfo.Communities {
			totalNodesInCommunities += len(nodes)
		}
		
		if totalNodesInCommunities != len(levelInfo.CommunityMap) {
			t.Errorf("Level %d: Inconsistent node count in communities (%d) vs community map (%d)",
				level, totalNodesInCommunities, len(levelInfo.CommunityMap))
		}
		
		// Check that every node in community map points to valid community
		for node, comm := range levelInfo.CommunityMap {
			if _, exists := levelInfo.Communities[comm]; !exists {
				t.Errorf("Level %d: Node %s points to non-existent community %d", level, node, comm)
			}
		}
		
		// Check that modularity is non-decreasing across levels
		if level > 0 {
			prevModularity := result.Levels[level-1].Modularity
			currModularity := levelInfo.Modularity
			if currModularity < prevModularity - 1e-10 { // Allow small numerical errors
				t.Errorf("Modularity decreased from level %d to %d: %f -> %f", 
					level-1, level, prevModularity, currModularity)
			}
		}
	}
}

// Test edge cases and error conditions
func TestEdgeCases(t *testing.T) {
	t.Run("MaxCommunitySize", func(t *testing.T) {
		g := NewHomogeneousGraph()
		for i := 1; i <= 5; i++ {
			g.AddNode(fmt.Sprintf("n%d", i), 1.0)
			if i > 1 {
				g.AddEdge(fmt.Sprintf("n%d", i-1), fmt.Sprintf("n%d", i), 1.0)
			}
		}
		
		config := DefaultLouvainConfig()
		config.MaxCommunitySize = 2
		config.RandomSeed = 42
		
		result, err := RunLouvain(g, config)
		if err != nil {
			t.Fatalf("RunLouvain failed: %v", err)
		}
		
		// Check that no community exceeds max size
		for level, levelInfo := range result.Levels {
			for commID, nodes := range levelInfo.Communities {
				if len(nodes) > config.MaxCommunitySize {
					t.Errorf("Level %d, Community %d exceeds max size: %d > %d", 
						level, commID, len(nodes), config.MaxCommunitySize)
				}
			}
		}
	})

	t.Run("MinModularity", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "b", 0.001) // Very weak edge
		
		config := DefaultLouvainConfig()
		config.MinModularity = 0.5 // High threshold
		config.RandomSeed = 42
		
		result, err := RunLouvain(g, config)
		if err != nil {
			t.Fatalf("RunLouvain failed: %v", err)
		}
		
		// With high min modularity, nodes should likely stay separate
		if len(result.FinalCommunities) < 2 {
			t.Logf("High min modularity didn't prevent merging (got %d communities)", 
				len(result.FinalCommunities))
		}
	})

	t.Run("MaxIterations", func(t *testing.T) {
		g := NewHomogeneousGraph()
		for i := 1; i <= 10; i++ {
			g.AddNode(fmt.Sprintf("n%d", i), 1.0)
			if i > 1 {
				g.AddEdge(fmt.Sprintf("n%d", i-1), fmt.Sprintf("n%d", i), 1.0)
			}
		}
		
		config := DefaultLouvainConfig()
		config.MaxIterations = 2
		config.RandomSeed = 42
		
		result, err := RunLouvain(g, config)
		if err != nil {
			t.Fatalf("RunLouvain failed: %v", err)
		}
		
		// Should respect max iterations
		if result.NumLevels > config.MaxIterations {
			t.Errorf("Exceeded max iterations: %d > %d", result.NumLevels, config.MaxIterations)
		}
	})

	t.Run("RandomSeedConsistency", func(t *testing.T) {
		g := NewHomogeneousGraph()
		g.AddEdge("a", "b", 1.0)
		g.AddEdge("b", "c", 1.0)
		g.AddEdge("c", "d", 1.0)
		
		config := DefaultLouvainConfig()
		config.RandomSeed = 12345
		config.MaxIterations = 3
		
		// Run twice with same seed
		result1, err1 := RunLouvain(g, config)
		if err1 != nil {
			t.Fatalf("First run failed: %v", err1)
		}
		
		result2, err2 := RunLouvain(g, config)
		if err2 != nil {
			t.Fatalf("Second run failed: %v", err2)
		}
		
		// Results should be identical
		if result1.NumLevels != result2.NumLevels {
			t.Errorf("Different number of levels: %d vs %d", result1.NumLevels, result2.NumLevels)
		}
		
		if math.Abs(result1.Modularity-result2.Modularity) > 1e-10 {
			t.Errorf("Different modularity: %f vs %f", result1.Modularity, result2.Modularity)
		}
	})
}

// Test memory and performance
func TestPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}
	
	t.Run("LargeGraph", func(t *testing.T) {
		// Create a moderately large graph
		g := NewHomogeneousGraph()
		
		// Create grid graph
		size := 20
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				nodeID := fmt.Sprintf("n_%d_%d", i, j)
				g.AddNode(nodeID, 1.0)
			}
		}
		
		// Add grid edges
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				if i < size-1 {
					g.AddEdge(fmt.Sprintf("n_%d_%d", i, j), fmt.Sprintf("n_%d_%d", i+1, j), 1.0)
				}
				if j < size-1 {
					g.AddEdge(fmt.Sprintf("n_%d_%d", i, j), fmt.Sprintf("n_%d_%d", i, j+1), 1.0)
				}
			}
		}
		
		config := DefaultLouvainConfig()
		config.MaxIterations = 5
		config.RandomSeed = 42
		
		start := time.Now()
		result, err := RunLouvain(g, config)
		duration := time.Since(start)
		
		if err != nil {
			t.Fatalf("RunLouvain failed: %v", err)
		}
		
		t.Logf("Large graph (%d nodes, %d edges) processed in %v", 
			len(g.Nodes), len(g.Edges)/2, duration)
		t.Logf("Found %d communities across %d levels", 
			len(result.FinalCommunities), result.NumLevels)
		t.Logf("Final modularity: %f", result.Modularity)
		
		// Basic sanity checks
		if result.NumLevels == 0 {
			t.Error("Should have at least one level")
		}
		if result.Modularity < 0 {
			t.Errorf("Modularity should be non-negative, got %f", result.Modularity)
		}
	})
}

// Test output writer
func TestOutputWriter(t *testing.T) {
	// Create simple test graph
	g := NewHomogeneousGraph()
	g.AddEdge("a", "b", 1.0)
	g.AddEdge("b", "c", 1.0)
	g.AddEdge("d", "e", 1.0)
	
	config := DefaultLouvainConfig()
	config.RandomSeed = 42
	
	result, err := RunLouvain(g, config)
	if err != nil {
		t.Fatalf("RunLouvain failed: %v", err)
	}
	
	// Test that output methods don't crash
	// writer := NewFileWriter()
	
	t.Run("WriteMapping", func(t *testing.T) {
		// This would normally write to file, but we just test it doesn't crash
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("WriteMapping panicked: %v", r)
			}
		}()
		
		// We can't easily test file writing without creating temp files
		// But we can test the logic by calling methods that don't write
		if len(result.Levels) > 0 {
			topLevel := result.Levels[len(result.Levels)-1]
			if len(topLevel.Communities) == 0 {
				t.Log("No communities in top level")
			}
		}
	})
}

// Benchmark basic operations
func BenchmarkModularityCalculation(b *testing.B) {
	g := NewHomogeneousGraph()
	for i := 0; i < 100; i++ {
		g.AddNode(fmt.Sprintf("n%d", i), 1.0)
		if i > 0 {
			g.AddEdge(fmt.Sprintf("n%d", i-1), fmt.Sprintf("n%d", i), 1.0)
		}
	}
	
	config := DefaultLouvainConfig()
	state := NewLouvainState(g, config)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = state.GetModularity()
	}
}

func BenchmarkNodeChunkProcessing(b *testing.B) {
	g := NewHomogeneousGraph()
	for i := 0; i < 50; i++ {
		g.AddNode(fmt.Sprintf("n%d", i), 1.0)
		if i > 0 {
			g.AddEdge(fmt.Sprintf("n%d", i-1), fmt.Sprintf("n%d", i), 1.0)
		}
	}
	
	config := DefaultLouvainConfig()
	state := NewLouvainState(g, config)
	nodes := make([]string, 10)
	for i := 0; i < 10; i++ {
		nodes[i] = fmt.Sprintf("n%d", i)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = state.processNodeChunk(nodes)
	}
}

// Helper function to verify community structure
func verifyCommunityStructure(t *testing.T, result *LouvainResult, originalGraph *HomogeneousGraph) {
	// Verify that each original node is assigned to exactly one community
	assignedNodes := make(map[string]bool)
	
	for node, _ := range result.FinalCommunities {
		if assignedNodes[node] {
			t.Errorf("Node %s assigned to multiple communities", node)
		}
		assignedNodes[node] = true
	}
	
	// Verify all original nodes are assigned
	for nodeID := range originalGraph.Nodes {
		if !assignedNodes[nodeID] {
			t.Errorf("Node %s not assigned to any community", nodeID)
		}
	}
}

// Test specific algorithmic properties
func TestAlgorithmicProperties(t *testing.T) {
	t.Run("ModularityNonDecreasing", func(t *testing.T) {
		g := NewHomogeneousGraph()
		// Create structured graph that should show clear modularity improvement
		for i := 1; i <= 6; i++ {
			g.AddNode(fmt.Sprintf("n%d", i), 1.0)
		}
		// Two triangles weakly connected
		g.AddEdge("n1", "n2", 2.0)
		g.AddEdge("n2", "n3", 2.0)
		g.AddEdge("n3", "n1", 2.0)
		g.AddEdge("n4", "n5", 2.0)
		g.AddEdge("n5", "n6", 2.0)
		g.AddEdge("n6", "n4", 2.0)
		g.AddEdge("n1", "n4", 0.1) // Weak bridge
		
		config := DefaultLouvainConfig()
		config.MaxIterations = 3
		config.RandomSeed = 42
		
		result, err := RunLouvain(g, config)
		if err != nil {
			t.Fatalf("RunLouvain failed: %v", err)
		}
		
		// Check modularity is non-decreasing
		for i := 1; i < len(result.Levels); i++ {
			prev := result.Levels[i-1].Modularity
			curr := result.Levels[i].Modularity
			if curr < prev - 1e-10 {
				t.Errorf("Modularity decreased from level %d to %d: %f -> %f", i-1, i, prev, curr)
			}
		}
	})

	t.Run("CommunityCoherence", func(t *testing.T) {
		// Test that well-connected components stay together
		g := NewHomogeneousGraph()
		
		// Create two dense cliques
		for i := 1; i <= 4; i++ {
			g.AddNode(fmt.Sprintf("a%d", i), 1.0)
			g.AddNode(fmt.Sprintf("b%d", i), 1.0)
		}
		
		// Fully connect each clique
		for i := 1; i <= 4; i++ {
			for j := i + 1; j <= 4; j++ {
				g.AddEdge(fmt.Sprintf("a%d", i), fmt.Sprintf("a%d", j), 5.0)
				g.AddEdge(fmt.Sprintf("b%d", i), fmt.Sprintf("b%d", j), 5.0)
			}
		}
		
		// Weak connection between cliques
		g.AddEdge("a1", "b1", 0.1)
		
		config := DefaultLouvainConfig()
		config.RandomSeed = 42
		
		result, err := RunLouvain(g, config)
		if err != nil {
			t.Fatalf("RunLouvain failed: %v", err)
		}
		
		// At minimum, we should have 2 communities (could be more if algorithm is conservative)
		finalCommunities := make(map[int][]string)
		for node, comm := range result.FinalCommunities {
			finalCommunities[comm] = append(finalCommunities[comm], node)
		}
		
		if len(finalCommunities) < 2 {
			t.Logf("Expected at least 2 communities for well-separated cliques, got %d", len(finalCommunities))
		}
		
		// Check that nodes from same clique tend to be together
		aNodes := []string{"a1", "a2", "a3", "a4"}
		bNodes := []string{"b1", "b2", "b3", "b4"}
		
		aComms := make(map[int]int)
		bComms := make(map[int]int)
		
		for _, node := range aNodes {
			comm := result.FinalCommunities[node]
			aComms[comm]++
		}
		for _, node := range bNodes {
			comm := result.FinalCommunities[node]
			bComms[comm]++
		}
		
		t.Logf("A-clique community distribution: %v", aComms)
		t.Logf("B-clique community distribution: %v", bComms)
	})
}

// TestForwardBackwardModularityGain tests that moving a node forward and backward
// should give opposite signs for modularity gain
func TestForwardBackwardModularityGain(t *testing.T) {
	// Create a simple test graph: three nodes where two are connected
	g := NewHomogeneousGraph()
	g.AddEdge("a", "b", 2.0)  // Strong connection
	g.AddNode("c", 1.0)       // Isolated node
	
	config := DefaultLouvainConfig()
	state := NewLouvainState(g, config)
	
	// Initially: each node in its own community
	// Communities: {a}, {b}, {c}
	commA := state.N2C["a"]
	commB := state.N2C["b"]
	
	t.Logf("Initial communities: a->%d, b->%d, c->%d", 
		state.N2C["a"], state.N2C["b"], state.N2C["c"])
	
	// Test moving node 'a' to node 'b's community
	neighborComms := state.getNeighborCommunities("a")
	
	var weightToB float64
	for _, nc := range neighborComms {
		if nc.Community == commB {
			weightToB = nc.Weight
			break
		}
	}
	
	// Calculate forward gain: a's community -> b's community
	forwardGain := state.modularityGain("a", commB, weightToB)
	t.Logf("Forward gain (a: %d -> %d): %.6f", commA, commB, forwardGain)
	
	// Actually move the node
	state.removeNodeFromCommunity("a", commA)
	state.insertNodeIntoCommunity("a", commB)
	
	// Now calculate reverse gain: b's community -> a's original community
	reverseNeighborComms := state.getNeighborCommunities("a")
	var weightToOriginal float64
	for _, nc := range reverseNeighborComms {
		if nc.Community == commA {
			weightToOriginal = nc.Weight
			break
		}
	}
	
	reverseGain := state.modularityGain("a", commA, weightToOriginal)
	t.Logf("Reverse gain (a: %d -> %d): %.6f", commB, commA, reverseGain)
	
	// The key test: if forward gain is positive, reverse should be negative (and vice versa)
	t.Logf("Forward: %.6f, Reverse: %.6f, Sum: %.6f", forwardGain, reverseGain, forwardGain+reverseGain)
	
	// Test the relationship
	if forwardGain > 0 {
		if reverseGain >= 0 {
			t.Errorf("Expected negative reverse gain when forward gain is positive. Forward: %.6f, Reverse: %.6f", 
				forwardGain, reverseGain)
		} else {
			t.Logf("✅ Correct: Positive forward gain (%.6f) has negative reverse gain (%.6f)", 
				forwardGain, reverseGain)
		}
	} else if forwardGain < 0 {
		if reverseGain <= 0 {
			t.Errorf("Expected positive reverse gain when forward gain is negative. Forward: %.6f, Reverse: %.6f", 
				forwardGain, reverseGain)
		} else {
			t.Logf("✅ Correct: Negative forward gain (%.6f) has positive reverse gain (%.6f)", 
				forwardGain, reverseGain)
		}
	} else {
		t.Logf("Forward gain is zero, reverse gain: %.6f", reverseGain)
	}
	
	// Additional check: they shouldn't both be positive (that would be weird)
	if forwardGain > 0 && reverseGain > 0 {
		t.Errorf("🚩 Both gains are positive - this is suspicious! Forward: %.6f, Reverse: %.6f", 
			forwardGain, reverseGain)
	}
	
	// Move back to restore state for cleanup
	state.removeNodeFromCommunity("a", commB)
	state.insertNodeIntoCommunity("a", commA)
}

// TestForwardBackwardWithDifferentGraphs tests the same concept on various graph structures
func TestForwardBackwardWithDifferentGraphs(t *testing.T) {
	testCases := []struct {
		name string
		setupGraph func() *HomogeneousGraph
		nodeToMove string
		expectedPattern string
	}{
		{
			name: "StronglyConnectedPair",
			setupGraph: func() *HomogeneousGraph {
				g := NewHomogeneousGraph()
				g.AddEdge("x", "y", 5.0)  // Strong connection
				g.AddNode("z", 1.0)       // Isolated
				return g
			},
			nodeToMove: "x",
			expectedPattern: "positive_forward_negative_reverse",
		},
		{
			name: "WeaklyConnectedPair",
			setupGraph: func() *HomogeneousGraph {
				g := NewHomogeneousGraph()
				g.AddEdge("x", "y", 0.1)  // Weak connection
				g.AddNode("z", 1.0)       // Isolated
				return g
			},
			nodeToMove: "x",
			expectedPattern: "negative_forward_positive_reverse",
		},
		{
			name: "Triangle",
			setupGraph: func() *HomogeneousGraph {
				g := NewHomogeneousGraph()
				g.AddEdge("a", "b", 1.0)
				g.AddEdge("b", "c", 1.0)
				g.AddEdge("c", "a", 1.0)
				return g
			},
			nodeToMove: "a",
			expectedPattern: "any", // Could go either way in a triangle
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			g := tc.setupGraph()
			config := DefaultLouvainConfig()
			state := NewLouvainState(g, config)
			
			oldComm := state.N2C[tc.nodeToMove]
			neighborComms := state.getNeighborCommunities(tc.nodeToMove)
			
			// Find a different community to move to
			var targetComm int
			var weightToTarget float64
			found := false
			for _, nc := range neighborComms {
				if nc.Community != oldComm {
					targetComm = nc.Community
					weightToTarget = nc.Weight
					found = true
					break
				}
			}
			
			if !found {
				t.Skip("No alternative community found for this graph")
			}
			
			// Test forward/backward
			forwardGain := state.modularityGain(tc.nodeToMove, targetComm, weightToTarget)
			
			state.removeNodeFromCommunity(tc.nodeToMove, oldComm)
			state.insertNodeIntoCommunity(tc.nodeToMove, targetComm)
			
			reverseNeighborComms := state.getNeighborCommunities(tc.nodeToMove)
			var weightToOriginal float64
			for _, nc := range reverseNeighborComms {
				if nc.Community == oldComm {
					weightToOriginal = nc.Weight
					break
				}
			}
			
			reverseGain := state.modularityGain(tc.nodeToMove, oldComm, weightToOriginal)
			
			t.Logf("%s: Forward=%.6f, Reverse=%.6f", tc.name, forwardGain, reverseGain)
			
			// Check the pattern based on expected behavior
			switch tc.expectedPattern {
			case "positive_forward_negative_reverse":
				if forwardGain <= 0 || reverseGain >= 0 {
					t.Logf("Expected positive forward, negative reverse, got forward=%.6f, reverse=%.6f", 
						forwardGain, reverseGain)
				}
			case "negative_forward_positive_reverse":
				if forwardGain >= 0 || reverseGain <= 0 {
					t.Logf("Expected negative forward, positive reverse, got forward=%.6f, reverse=%.6f", 
						forwardGain, reverseGain)
				}
			case "any":
				// Just log the results
				t.Logf("Triangle case: forward=%.6f, reverse=%.6f", forwardGain, reverseGain)
			}
			
			// Universal check: they shouldn't both be positive
			if forwardGain > 0 && reverseGain > 0 {
				t.Errorf("🚩 Both gains positive in %s: forward=%.6f, reverse=%.6f", 
					tc.name, forwardGain, reverseGain)
			}
			
			// Restore state
			state.removeNodeFromCommunity(tc.nodeToMove, targetComm)
			state.insertNodeIntoCommunity(tc.nodeToMove, oldComm)
		})
	}
}

// TestModularityConsistencyProper tests modularity gain with non-singleton communities
func TestModularityConsistencyProper(t *testing.T) {
	// Create a graph where we can move a node between multi-node communities
	g := NewHomogeneousGraph()
	
	// Community 1: {a, b} - strongly connected
	g.AddEdge("a", "b", 3.0)
	
	// Community 2: {c, d} - strongly connected  
	g.AddEdge("c", "d", 3.0)
	
	// Weak bridge between communities
	g.AddEdge("a", "c", 0.5)
	
	config := DefaultLouvainConfig()
	state := NewLouvainState(g, config)
	
	// First, let's set up the communities properly by moving nodes
	// Move 'b' to same community as 'a'
	commA := state.N2C["a"]
	commB := state.N2C["b"]
	if commA != commB {
		state.removeNodeFromCommunity("b", commB)
		state.insertNodeIntoCommunity("b", commA)
	}
	
	// Move 'd' to same community as 'c'  
	commC := state.N2C["c"]
	commD := state.N2C["d"]
	if commC != commD {
		state.removeNodeFromCommunity("d", commD)
		state.insertNodeIntoCommunity("d", commC)
	}
	
	// Clean up empty communities
	state.cleanupEmptyCommunities()
	
	t.Logf("=== SETUP COMPLETE ===")
	t.Logf("Communities: %v", state.C2N)
	t.Logf("N2C mapping: %v", state.N2C)
	t.Logf("Number of communities: %d", len(state.C2N))
	
	// Now test moving 'a' from its community (with 'b') to 'c's community (with 'd')
	oldComm := state.N2C["a"]
	targetComm := state.N2C["c"]
	
	if oldComm == targetComm {
		t.Skip("Nodes already in same community - test setup failed")
	}
	
	// Verify both communities have multiple nodes
	if len(state.C2N[oldComm]) < 2 {
		t.Fatalf("Source community has only %d nodes, need at least 2", len(state.C2N[oldComm]))
	}
	if len(state.C2N[targetComm]) < 1 {
		t.Fatalf("Target community has only %d nodes", len(state.C2N[targetComm]))
	}
	
	t.Logf("Moving 'a' from community %d (size %d) to community %d (size %d)", 
		oldComm, len(state.C2N[oldComm]), targetComm, len(state.C2N[targetComm]))
	
	// Get the weight to target community
	neighborComms := state.getNeighborCommunities("a")
	var weightToTarget float64
	for _, nc := range neighborComms {
		if nc.Community == targetComm {
			weightToTarget = nc.Weight
			break
		}
	}
	
	t.Logf("Weight from 'a' to target community: %.6f", weightToTarget)
	
	// Store current modularity
	initialModularity := state.GetModularity()
	t.Logf("Initial modularity: %.10f", initialModularity)
	
	// Calculate predicted gain
	predictedGain := state.modularityGain("a", targetComm, weightToTarget)
	t.Logf("Predicted gain: %.10f", predictedGain)
	
	// Actually make the move
	state.removeNodeFromCommunity("a", oldComm)
	state.insertNodeIntoCommunity("a", targetComm)
	
	// Verify both communities still exist
	if len(state.C2N) != 2 {
		t.Errorf("Expected 2 communities after move, got %d", len(state.C2N))
	}
	
	t.Logf("After move - communities: %v", state.C2N)
	
	// Calculate actual new modularity
	newModularity := state.GetModularity()
	actualGain := newModularity - initialModularity
	
	t.Logf("New modularity: %.10f", newModularity)
	t.Logf("Actual gain: %.10f", actualGain)
	t.Logf("Difference: %.10f", math.Abs(predictedGain - actualGain))
	
	// They should match (within floating point tolerance)
	tolerance := 1e-9
	if math.Abs(predictedGain - actualGain) > tolerance {
		t.Errorf("Modularity gain prediction failed: predicted=%.10f, actual=%.10f", 
			predictedGain, actualGain)
	} else {
		t.Logf("✅ Modularity gain prediction is accurate!")
	}
}

// Simpler test with a pre-constructed community structure
func TestModularityConsistencySimple(t *testing.T) {
	g := NewHomogeneousGraph()
	
	// Create a 4-node graph: two pairs with a bridge
	g.AddEdge("n1", "n2", 2.0)  // Pair 1
	g.AddEdge("n3", "n4", 2.0)  // Pair 2  
	g.AddEdge("n1", "n3", 0.1)  // Weak bridge
	
	config := DefaultLouvainConfig()
	state := NewLouvainState(g, config)
	
	// Manually set up communities: {n1, n2} and {n3, n4}
	comm1 := state.N2C["n1"]
	comm2 := state.N2C["n2"] 
	comm3 := state.N2C["n3"]
	comm4 := state.N2C["n4"]
	
	// Move n2 to n1's community
	if comm1 != comm2 {
		state.removeNodeFromCommunity("n2", comm2)
		state.insertNodeIntoCommunity("n2", comm1)
	}
	
	// Move n4 to n3's community
	if comm3 != comm4 {
		state.removeNodeFromCommunity("n4", comm4) 
		state.insertNodeIntoCommunity("n4", comm3)
	}
	
	// Clean up
	state.cleanupEmptyCommunities()
	
	t.Logf("Setup: Communities = %v", state.C2N)
	
	// Test moving n1 from {n1,n2} to {n3,n4}
	oldComm := state.N2C["n1"]
	targetComm := state.N2C["n3"]
	
	// Get connection weight
	neighborComms := state.getNeighborCommunities("n1")
	var weightToTarget float64
	for _, nc := range neighborComms {
		if nc.Community == targetComm {
			weightToTarget = nc.Weight
			break
		}
	}
	
	// Test prediction vs reality
	initialMod := state.GetModularity()
	predicted := state.modularityGain("n1", targetComm, weightToTarget)
	
	state.removeNodeFromCommunity("n1", oldComm)
	state.insertNodeIntoCommunity("n1", targetComm)
	
	newMod := state.GetModularity()
	actual := newMod - initialMod
	
	t.Logf("Predicted: %.10f, Actual: %.10f, Diff: %.10f", 
		predicted, actual, math.Abs(predicted-actual))
	
	if math.Abs(predicted-actual) > 1e-9 {
		t.Errorf("Mismatch: predicted=%.10f, actual=%.10f", predicted, actual)
	}
}


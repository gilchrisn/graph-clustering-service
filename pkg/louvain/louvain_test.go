package louvain

import (
	"fmt"
	"math"
	"testing"
)

// TestBasicLouvain tests the Louvain algorithm on a simple graph
func TestBasicLouvain(t *testing.T) {
	// Create a simple graph with two clear communities
	graph := createTestGraph()
	
	config := DefaultLouvainConfig()
	config.Verbose = true
	
	result, err := RunLouvain(graph, config)
	if err != nil {
		t.Fatalf("Louvain failed: %v", err)
	}
	
	// Verify results
	if result.NumLevels == 0 {
		t.Error("Expected at least one level")
	}
	
	if result.Modularity <= 0 {
		t.Errorf("Expected positive modularity, got %f", result.Modularity)
	}
	
	// Check that we have fewer communities than nodes
	finalLevel := result.Levels[len(result.Levels)-1]
	if len(finalLevel.Communities) >= len(graph.Nodes) {
		t.Error("No community detection occurred")
	}
	
	t.Logf("Final modularity: %f", result.Modularity)
	t.Logf("Number of levels: %d", result.NumLevels)
	t.Logf("Final communities: %d", len(finalLevel.Communities))
}

// TestSizeConstrainedLouvain tests Louvain with community size constraints
func TestSizeConstrainedLouvain(t *testing.T) {
	graph := createLargerTestGraph()
	
	config := DefaultLouvainConfig()
	config.MaxCommunitySize = 5
	
	result, err := RunLouvain(graph, config)
	if err != nil {
		t.Fatalf("Louvain failed: %v", err)
	}
	
	// Verify size constraints
	for _, level := range result.Levels {
		for _, nodes := range level.Communities {
			if len(nodes) > config.MaxCommunitySize {
				t.Errorf("Community size %d exceeds limit %d", 
					len(nodes), config.MaxCommunitySize)
			}
		}
	}
}

// TestModularityCalculation tests the modularity calculation
func TestModularityCalculation(t *testing.T) {
	graph := createTestGraph()
	config := DefaultLouvainConfig()
	state := NewLouvainState(graph, config)
	
	// Initial modularity (each node in its own community)
	initialMod := state.GetModularity()
	t.Logf("Initial modularity: %f", initialMod)
	
	// Put all nodes in one community
	singleComm := 0
	for nodeID := range state.N2C {
		state.removeNodeFromCommunity(nodeID, state.N2C[nodeID])
		state.insertNodeIntoCommunity(nodeID, singleComm)
	}
	
	singleCommMod := state.GetModularity()
	t.Logf("Single community modularity: %f", singleCommMod)
	
	// Modularity should be 0 for single community
	if math.Abs(singleCommMod) > 1e-9 {
		t.Errorf("Expected modularity 0 for single community, got %f", singleCommMod)
	}
}

// TestOutputGeneration tests the output file generation
func TestOutputGeneration(t *testing.T) {
	graph := createTestGraph()
	config := DefaultLouvainConfig()
	
	result, err := RunLouvain(graph, config)
	if err != nil {
		t.Fatalf("Louvain failed: %v", err)
	}
	
	// Test output generation
	writer := NewFileWriter()
	outputDir := "./testdata/output"
	prefix := "test"
	
	err = writer.WriteAll(result, graph, outputDir, prefix)
	if err != nil {
		t.Fatalf("Failed to write output: %v", err)
	}
	
	// Verify files were created
	// Note: In a real test, you'd check file contents
	t.Log("Output files generated successfully")
}

// TestGraphValidation tests graph validation
func TestGraphValidation(t *testing.T) {
	// Test empty graph
	emptyGraph := NewHomogeneousGraph()
	err := emptyGraph.Validate()
	if err == nil {
		t.Error("Expected error for empty graph")
	}
	
	// Test graph with invalid edge
	invalidGraph := NewHomogeneousGraph()
	invalidGraph.AddNode("a", 1.0)
	invalidGraph.Edges[EdgeKey{From: "a", To: "b"}] = 1.0 // b doesn't exist
	
	err = invalidGraph.Validate()
	if err == nil {
		t.Error("Expected error for invalid edge reference")
	}
	
	// Test non-symmetric graph
	asymmetricGraph := NewHomogeneousGraph()
	asymmetricGraph.AddNode("a", 1.0)
	asymmetricGraph.AddNode("b", 1.0)
	asymmetricGraph.Edges[EdgeKey{From: "a", To: "b"}] = 1.0
	// Missing reverse edge
	
	err = asymmetricGraph.Validate()
	if err == nil {
		t.Error("Expected error for asymmetric graph")
	}
}

// TestDeterminism tests that the algorithm is deterministic with fixed seed
func TestDeterminism(t *testing.T) {
	graph := createLargerTestGraph()
	
	config := DefaultLouvainConfig()
	config.RandomSeed = 42
	
	// Run algorithm twice with same seed
	result1, err1 := RunLouvain(graph, config)
	if err1 != nil {
		t.Fatalf("First run failed: %v", err1)
	}
	
	result2, err2 := RunLouvain(graph, config)
	if err2 != nil {
		t.Fatalf("Second run failed: %v", err2)
	}
	
	// Results should be identical
	if result1.Modularity != result2.Modularity {
		t.Errorf("Different modularity: %f vs %f", result1.Modularity, result2.Modularity)
	}
	
	if result1.NumLevels != result2.NumLevels {
		t.Errorf("Different number of levels: %d vs %d", result1.NumLevels, result2.NumLevels)
	}
}

// Helper functions for creating test graphs

func createTestGraph() *HomogeneousGraph {
	// Create a graph with two clear communities
	graph := NewHomogeneousGraph()
	
	// Community 1: a, b, c
	graph.AddEdge("a", "b", 1.0)
	graph.AddEdge("b", "c", 1.0)
	graph.AddEdge("a", "c", 1.0)
	
	// Community 2: d, e, f
	graph.AddEdge("d", "e", 1.0)
	graph.AddEdge("e", "f", 1.0)
	graph.AddEdge("d", "f", 1.0)
	
	// Weak link between communities
	graph.AddEdge("c", "d", 0.1)
	
	return graph
}

func createLargerTestGraph() *HomogeneousGraph {
	graph := NewHomogeneousGraph()
	
	// Create 3 communities of 5 nodes each
	communities := [][]string{
		{"a1", "a2", "a3", "a4", "a5"},
		{"b1", "b2", "b3", "b4", "b5"},
		{"c1", "c2", "c3", "c4", "c5"},
	}
	
	// Add strong intra-community edges
	for _, comm := range communities {
		for i := 0; i < len(comm); i++ {
			for j := i + 1; j < len(comm); j++ {
				graph.AddEdge(comm[i], comm[j], 1.0)
			}
		}
	}
	
	// Add weak inter-community edges
	graph.AddEdge("a5", "b1", 0.1)
	graph.AddEdge("b5", "c1", 0.1)
	
	return graph
}

// BenchmarkLouvain benchmarks the Louvain algorithm
func BenchmarkLouvain(b *testing.B) {
	// Create a larger graph for benchmarking
	graph := createBenchmarkGraph(100, 10)
	config := DefaultLouvainConfig()
	
	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		_, err := RunLouvain(graph, config)
		if err != nil {
			b.Fatalf("Louvain failed: %v", err)
		}
	}
}

func createBenchmarkGraph(numCommunities, nodesPerCommunity int) *HomogeneousGraph {
	graph := NewHomogeneousGraph()
	
	// Create communities
	for c := 0; c < numCommunities; c++ {
		// Add intra-community edges (dense)
		for i := 0; i < nodesPerCommunity; i++ {
			for j := i + 1; j < nodesPerCommunity; j++ {
				from := fmt.Sprintf("c%d_n%d", c, i)
				to := fmt.Sprintf("c%d_n%d", c, j)
				graph.AddEdge(from, to, 1.0)
			}
		}
	}
	
	// Add inter-community edges (sparse)
	for c1 := 0; c1 < numCommunities; c1++ {
		for c2 := c1 + 1; c2 < numCommunities; c2++ {
			// Add one edge between communities
			from := fmt.Sprintf("c%d_n0", c1)
			to := fmt.Sprintf("c%d_n0", c2)
			graph.AddEdge(from, to, 0.1)
		}
	}
	
	return graph
}
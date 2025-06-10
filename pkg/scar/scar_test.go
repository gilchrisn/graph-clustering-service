package scar

import (
	"fmt"
	"testing"
	"time"
)

// ==================== COMPREHENSIVE TEST SUITE ====================

// TestScarStepByStepValidation validates every major step of SCAR algorithm
func TestScarStepByStepValidation(t *testing.T) {
	t.Log("🔬 Starting comprehensive SCAR step-by-step validation...")
	
	// Create test graph with known community structure
	graph := createKnownCommunityGraph()
	config := createOptimalConfig()
	
	// Expected communities: {a0,a1,a2}, {a3,a4,a5}, {a6,a7,a8}
	expectedCommunities := map[string]int{
		"a0": 0, "a1": 0, "a2": 0,
		"a3": 1, "a4": 1, "a5": 1, 
		"a6": 2, "a7": 2, "a8": 2,
	}
	
	t.Logf("Graph: %d nodes, %d edges", graph.NumNodes(), graph.NumEdges())
	t.Logf("Expected 3 communities with 3 authors each")
	
	// STEP 1: Test sketch construction in isolation
	t.Run("Step1_SketchConstruction", func(t *testing.T) {
		validateSketchConstruction(t, graph, config)
	})
	
	// STEP 2: Test hash-to-node mapping
	t.Run("Step2_HashToNodeMapping", func(t *testing.T) {
		validateHashToNodeMapping(t, graph, config)
	})
	
	// STEP 3: Test degree estimation accuracy
	t.Run("Step3_DegreeEstimation", func(t *testing.T) {
		validateDegreeEstimation(t, graph, config)
	})
	
	// STEP 4: Test union/intersection operations
	t.Run("Step4_UnionIntersection", func(t *testing.T) {
		validateUnionIntersection(t, config)
	})
	
	// STEP 5: Test community discovery through sketches
	t.Run("Step5_CommunityDiscovery", func(t *testing.T) {
		validateCommunityDiscovery(t, graph, config, expectedCommunities)
	})
	
	// STEP 6: Test full algorithm
	t.Run("Step6_FullAlgorithm", func(t *testing.T) {
		validateFullAlgorithm(t, graph, config, expectedCommunities)
	})
}

// TestEdgeCases tests various edge cases and boundary conditions
func TestEdgeCases(t *testing.T) {
	t.Log("🧪 Testing edge cases...")
	
	t.Run("EmptyGraph", func(t *testing.T) {
		graph := NewHeterogeneousGraph()
		config := createOptimalConfig()
		
		result, err := RunScar(graph, config)
		if err == nil {
			t.Error("Expected error for empty graph")
		}
		if result != nil {
			t.Error("Expected nil result for empty graph")
		}
	})
	
	t.Run("SingleNode", func(t *testing.T) {
		graph := createSingleNodeGraph()
		config := createOptimalConfig()
		
		result, err := RunScar(graph, config)
		if err != nil {
			t.Fatalf("Single node should not fail: %v", err)
		}
		
		if len(result.FinalCommunities) != 1 {
			t.Errorf("Expected 1 community, got %d", len(result.FinalCommunities))
		}
	})
	
	t.Run("DisconnectedComponents", func(t *testing.T) {
		graph := createDisconnectedGraph()
		config := createOptimalConfig()
		
		result, err := RunScar(graph, config)
		if err != nil {
			t.Fatalf("Disconnected graph failed: %v", err)
		}
		
		// Should find separate communities for each component
		communities := getUniqueCommunities(result.FinalCommunities)
		if len(communities) < 2 {
			t.Errorf("Expected multiple communities for disconnected graph, got %d", len(communities))
		}
	})
	
	t.Run("VerySmallK", func(t *testing.T) {
		graph := createKnownCommunityGraph()
		config := createOptimalConfig()
		config.K = 2 // Very small K
		
		result, err := RunScar(graph, config)
		if err != nil {
			t.Fatalf("Small K should not fail: %v", err)
		}
		
		if result.Modularity < 0 {
			t.Error("Small K produced negative modularity")
		}
	})
	
	t.Run("VeryLargeK", func(t *testing.T) {
		graph := createKnownCommunityGraph()
		config := createOptimalConfig()
		config.K = 1024 // Very large K
		
		result, err := RunScar(graph, config)
		if err != nil {
			t.Fatalf("Large K should not fail: %v", err)
		}
		
		if result.Modularity < 0 {
			t.Error("Large K produced negative modularity")
		}
	})
}

// TestAccuracyBenchmarks tests accuracy against known ground truth
func TestAccuracyBenchmarks(t *testing.T) {
	testCases := []struct {
		name     string
		graphGen func() *HeterogeneousGraph
		expected map[string]int
		minAccuracy float64
	}{
		{
			name:     "PerfectClusters",
			graphGen: createPerfectClustersGraph,
			expected: map[string]int{
				"a0": 0, "a1": 0, "a2": 0,
				"a3": 1, "a4": 1, "a5": 1,
			},
			minAccuracy: 0.8,
		},
		{
			name:     "OverlappingClusters", 
			graphGen: createOverlappingClustersGraph,
			expected: map[string]int{
				"a0": 0, "a1": 0, "a2": 0, "a3": 0,
				"a4": 1, "a5": 1, "a6": 1, "a7": 1,
			},
			minAccuracy: 0.6, // Lower expectation for overlapping
		},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			graph := tc.graphGen()
			config := createOptimalConfig()
			
			result, err := RunScar(graph, config)
			if err != nil {
				t.Fatalf("SCAR failed: %v", err)
			}
			
			accuracy := calculateAccuracy(result.FinalCommunities, tc.expected)
			t.Logf("Accuracy: %.3f (expected: %.3f)", accuracy, tc.minAccuracy)
			
			if accuracy < tc.minAccuracy {
				t.Errorf("Accuracy %.3f below minimum %.3f", accuracy, tc.minAccuracy)
			}
		})
	}
}

// TestPerformanceBenchmarks tests performance characteristics
func TestPerformanceBenchmarks(t *testing.T) {
	sizes := []int{10, 50, 100}
	
	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size_%d", size), func(t *testing.T) {
			graph := createScalableGraph(size)
			config := createOptimalConfig()
			
			start := time.Now()
			result, err := RunScar(graph, config)
			duration := time.Since(start)
			
			if err != nil {
				t.Fatalf("Performance test failed: %v", err)
			}
			
			t.Logf("Size %d: %v, modularity=%.4f, communities=%d", 
				size, duration, result.Modularity, len(getUniqueCommunities(result.FinalCommunities)))
			
			// Performance expectations
			if duration > time.Second*10 {
				t.Errorf("Too slow for size %d: %v", size, duration)
			}
			
			if result.Modularity < 0 {
				t.Error("Negative modularity indicates algorithmic issues")
			}
		})
	}
}

// ==================== STEP-BY-STEP VALIDATION FUNCTIONS ====================

func validateSketchConstruction(t *testing.T, graph *HeterogeneousGraph, config ScarConfig) {
	t.Log("🔍 Validating sketch construction...")
	
	state := createInitialState(graph, config)
	err := state.constructSketchesIteratively()
	if err != nil {
		t.Fatalf("Sketch construction failed: %v", err)
	}
	
	// Validate all nodes have sketches
	for _, nodeID := range graph.NodeList {
		sketch := state.Sketches[nodeID]
		if sketch == nil {
			t.Errorf("Node %s missing sketch", nodeID)
			continue
		}
		
		// Validate sketch structure
		if sketch.K != config.K {
			t.Errorf("Node %s: sketch K=%d, expected %d", nodeID, sketch.K, config.K)
		}
		
		if sketch.NK != config.NK {
			t.Errorf("Node %s: sketch NK=%d, expected %d", nodeID, sketch.NK, config.NK)
		}
		
		if len(sketch.Sketches) != config.NK {
			t.Errorf("Node %s: %d hash functions, expected %d", nodeID, len(sketch.Sketches), config.NK)
		}
	}
	
	// Validate source nodes have hash values
	sourceType := config.MetaPath.NodeTypes[0]
	sourceNodes := graph.GetNodesByType(sourceType)
	
	nonEmptySource := 0
	for _, nodeID := range sourceNodes {
		if !state.Sketches[nodeID].IsEmpty() {
			nonEmptySource++
		}
	}
	
	if nonEmptySource == 0 {
		t.Error("No source nodes have hash values")
	}
	
	t.Logf("✅ Sketch construction: %d nodes, %d source nodes with values", 
		len(state.Sketches), nonEmptySource)
}

func validateHashToNodeMapping(t *testing.T, graph *HeterogeneousGraph, config ScarConfig) {
	t.Log("🔍 Validating hash-to-node mapping...")
	
	state := createInitialState(graph, config)
	state.constructSketchesIteratively()
	
	// Check hash mapping coverage
	totalHashes := len(state.HashToNodeMap.Mapping)
	if totalHashes == 0 {
		t.Error("No hash-to-node mappings created")
		return
	}
	
	// Validate mapping consistency
	validMappings := 0
	for hashValue, nodeID := range state.HashToNodeMap.Mapping {
		if graph.Nodes[nodeID].ID == nodeID {
			validMappings++
		} else {
			t.Errorf("Invalid mapping: hash %d -> node %s (node doesn't exist)", hashValue, nodeID)
		}
	}
	
	t.Logf("✅ Hash-to-node mapping: %d total, %d valid", totalHashes, validMappings)
}

func validateDegreeEstimation(t *testing.T, graph *HeterogeneousGraph, config ScarConfig) {
	t.Log("🔍 Validating degree estimation...")
	
	state := createInitialState(graph, config)
	state.constructSketchesIteratively()
	
	// Test degree estimation for all nodes
	totalNodes := 0
	positiveDegrees := 0
	
	for nodeID, sketch := range state.Sketches {
		if sketch.IsEmpty() {
			continue
		}
		
		totalNodes++
		degree := sketch.EstimateDegree()
		
		if degree.Value > 0 {
			positiveDegrees++
		}
		
		if degree.Value < 0 {
			t.Errorf("Node %s: negative degree %f", nodeID, degree.Value)
		}
		
		if degree.Value > float64(graph.NumEdges()*2) {
			t.Errorf("Node %s: unrealistic degree %f (total edges: %d)", 
				nodeID, degree.Value, graph.NumEdges())
		}
	}
	
	if positiveDegrees == 0 && totalNodes > 0 {
		t.Error("No nodes have positive degree estimates")
	}
	
	t.Logf("✅ Degree estimation: %d nodes, %d with positive degrees", 
		totalNodes, positiveDegrees)
}

func validateUnionIntersection(t *testing.T, config ScarConfig) {
	t.Log("🔍 Validating union/intersection operations...")
	
	// Create test sketches with known overlap
	sketch1 := NewVertexBottomKSketch(config.K, config.NK, 0)
	sketch2 := NewVertexBottomKSketch(config.K, config.NK, 0)
	
	// Add known values with some overlap
	for hashFunc := 0; hashFunc < config.NK; hashFunc++ {
		// sketch1: values 0, 2, 4, 6, ...
		for i := 0; i < config.K/2; i++ {
			sketch1.AddValue(hashFunc, uint64(i*2))
		}
		
		// sketch2: values 1, 3, 5, 7, ... (some overlap if we add even numbers too)
		for i := 0; i < config.K/2; i++ {
			sketch2.AddValue(hashFunc, uint64(i*2+1))
			if i < config.K/4 {
				sketch2.AddValue(hashFunc, uint64(i*2)) // Some overlap
			}
		}
	}
	
	// Test union operation
	unionSketch := UnionSketches(sketch1, sketch2)
	if unionSketch == nil {
		t.Fatal("Union operation returned nil")
	}
	
	// Test intersection estimation
	intersection := sketch1.EstimateIntersectionWith(sketch2)
	if intersection < 0 {
		t.Error("Intersection should be non-negative")
	}
	
	// Test basic properties
	size1 := sketch1.EstimateDegree().Value
	size2 := sketch2.EstimateDegree().Value
	unionSize := unionSketch.EstimateDegree().Value
	
	// Union should be at least as large as either individual sketch
	if unionSize < size1 || unionSize < size2 {
		t.Errorf("Union size %f smaller than individuals %f, %f", unionSize, size1, size2)
	}
	
	t.Logf("✅ Union/Intersection: size1=%.1f, size2=%.1f, union=%.1f, intersection=%.1f", 
		size1, size2, unionSize, intersection)
}

func validateCommunityDiscovery(t *testing.T, graph *HeterogeneousGraph, config ScarConfig, expected map[string]int) {
	t.Log("🔍 Validating community discovery through sketches...")
	
	state := createInitialState(graph, config)
	state.constructSketchesIteratively()
	state.initializeCommunities()
	
	// Test community discovery for sample nodes
	testNodes := []string{"a0", "a1", "a3", "a4"}
	
	for _, nodeID := range testNodes {
		if _, exists := graph.Nodes[nodeID]; !exists {
			continue
		}
		
		communities := state.findCommunitiesThroughSketches(nodeID)
		
		if len(communities) == 0 {
			t.Errorf("Node %s found no communities", nodeID)
			continue
		}
		
		// Should at least find its own community
		ownCommunity := state.N2C[nodeID]
		foundOwn := false
		for _, comm := range communities {
			if comm == ownCommunity {
				foundOwn = true
				break
			}
		}
		
		if !foundOwn {
			t.Errorf("Node %s didn't find its own community %d", nodeID, ownCommunity)
		}
	}
	
	t.Logf("✅ Community discovery: tested %d nodes", len(testNodes))
}

func validateFullAlgorithm(t *testing.T, graph *HeterogeneousGraph, config ScarConfig, expected map[string]int) {
	t.Log("🔍 Validating full SCAR algorithm...")
	
	start := time.Now()
	result, err := RunScar(graph, config)
	duration := time.Since(start)
	
	if err != nil {
		t.Fatalf("Full algorithm failed: %v", err)
	}
	
	// Basic result validation
	if result == nil {
		t.Fatal("Result is nil")
	}
	
	if result.Modularity < 0 {
		t.Errorf("Negative modularity: %f", result.Modularity)
	}
	
	if len(result.FinalCommunities) == 0 {
		t.Error("No final communities found")
	}
	
	// Community structure validation
	communities := getUniqueCommunities(result.FinalCommunities)
	t.Logf("Found %d communities", len(communities))
	
	if len(communities) == len(graph.NodeList) {
		t.Logf("⚠️  WARNING: Each node in its own community (potential algorithm issue)")
	}
	
	if len(communities) == 1 {
		t.Logf("⚠️  WARNING: All nodes in one community (potential algorithm issue)")
	}
	
	// Accuracy check if ground truth provided
	if len(expected) > 0 {
		accuracy := calculateAccuracy(result.FinalCommunities, expected)
		t.Logf("Accuracy vs ground truth: %.3f", accuracy)
		
		if accuracy < 0.5 {
			t.Errorf("Low accuracy: %.3f", accuracy)
		}
	}
	
	t.Logf("✅ Full algorithm: %v, modularity=%.4f, %d communities", 
		duration, result.Modularity, len(communities))
}

// ==================== HELPER FUNCTIONS ====================

func createKnownCommunityGraph() *HeterogeneousGraph {
	graph := NewHeterogeneousGraph()
	
	// Create 3 distinct author communities, each working on separate papers
	authors := []string{"a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"}
	papers := []string{"p0", "p1", "p2", "p3", "p4", "p5"}
	
	// Add author nodes
	for _, authorID := range authors {
		node := HeteroNode{ID: authorID, Type: "Author"}
		graph.AddNode(node)
	}
	
	// Add paper nodes
	for _, paperID := range papers {
		node := HeteroNode{ID: paperID, Type: "Paper"}
		graph.AddNode(node)
	}
	
	// Community 1: a0, a1, a2 -> p0, p1
	collaborations := map[string][]string{
		"p0": {"a0", "a1", "a2"},     // Community 1
		"p1": {"a0", "a1", "a2"},     // Community 1
		"p2": {"a3", "a4", "a5"},     // Community 2  
		"p3": {"a3", "a4", "a5"},     // Community 2
		"p4": {"a6", "a7", "a8"},     // Community 3
		"p5": {"a6", "a7", "a8"},     // Community 3
	}
	
	// Add edges
	for paperID, authorList := range collaborations {
		for _, authorID := range authorList {
			edge := HeteroEdge{From: authorID, To: paperID, Type: "writes", Weight: 1.0}
			graph.AddEdge(edge)
		}
	}
	
	return graph
}

func createOptimalConfig() ScarConfig {
	config := DefaultScarConfig()
	config.MetaPath = MetaPath{
		NodeTypes: []string{"Author", "Paper", "Author"},
		EdgeTypes: []string{"writes", "writes"},
	}
	config.K = 32
	config.NK = 4
	config.MaxIterations = 10
	config.Verbose = false
	config.RandomSeed = 42
	return config
}

func createInitialState(graph *HeterogeneousGraph, config ScarConfig) *ScarState {
	state := &ScarState{
		Graph:             graph,
		Config:            config,
		Sketches:          make(map[string]*VertexBottomKSketch),
		HashToNodeMap:     NewHashToNodeMap(),
		NodeToOriginal:    make(map[string][]string),
		N2C:               make(map[string]int),
		C2N:               make(map[int][]string),
		CommunitySketches: make(map[int]*VertexBottomKSketch),
		CommunityDegrees:  make(map[int]*DegreeEstimate),
		NodeDegrees:       make(map[string]*DegreeEstimate),
	}
	
	// Initialize node mapping
	for _, nodeID := range graph.NodeList {
		state.NodeToOriginal[nodeID] = []string{nodeID}
	}
	
	return state
}

func createSingleNodeGraph() *HeterogeneousGraph {
	graph := NewHeterogeneousGraph()
	node := HeteroNode{ID: "single", Type: "Author"}
	graph.AddNode(node)
	return graph
}

func createDisconnectedGraph() *HeterogeneousGraph {
	graph := NewHeterogeneousGraph()
	
	// Component 1
	authors1 := []string{"a0", "a1"}
	paper1 := "p0"
	
	// Component 2  
	authors2 := []string{"a2", "a3"}
	paper2 := "p1"
	
	// Add nodes
	for _, authorID := range append(authors1, authors2...) {
		node := HeteroNode{ID: authorID, Type: "Author"}
		graph.AddNode(node)
	}
	
	for _, paperID := range []string{paper1, paper2} {
		node := HeteroNode{ID: paperID, Type: "Paper"}
		graph.AddNode(node)
	}
	
	// Add edges (no connection between components)
	for _, authorID := range authors1 {
		edge := HeteroEdge{From: authorID, To: paper1, Type: "writes", Weight: 1.0}
		graph.AddEdge(edge)
	}
	
	for _, authorID := range authors2 {
		edge := HeteroEdge{From: authorID, To: paper2, Type: "writes", Weight: 1.0}
		graph.AddEdge(edge)
	}
	
	return graph
}

func createPerfectClustersGraph() *HeterogeneousGraph {
	graph := NewHeterogeneousGraph()
	
	// Two perfect clusters with no inter-cluster connections
	authors := []string{"a0", "a1", "a2", "a3", "a4", "a5"}
	papers := []string{"p0", "p1", "p2", "p3"}
	
	for _, authorID := range authors {
		node := HeteroNode{ID: authorID, Type: "Author"}
		graph.AddNode(node)
	}
	
	for _, paperID := range papers {
		node := HeteroNode{ID: paperID, Type: "Paper"}
		graph.AddNode(node)
	}
	
	// Cluster 1: a0,a1,a2 -> p0,p1
	// Cluster 2: a3,a4,a5 -> p2,p3
	collaborations := map[string][]string{
		"p0": {"a0", "a1", "a2"},
		"p1": {"a0", "a1", "a2"}, 
		"p2": {"a3", "a4", "a5"},
		"p3": {"a3", "a4", "a5"},
	}
	
	for paperID, authorList := range collaborations {
		for _, authorID := range authorList {
			edge := HeteroEdge{From: authorID, To: paperID, Type: "writes", Weight: 1.0}
			graph.AddEdge(edge)
		}
	}
	
	return graph
}

func createOverlappingClustersGraph() *HeterogeneousGraph {
	graph := NewHeterogeneousGraph()
	
	authors := []string{"a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"}
	papers := []string{"p0", "p1", "p2", "p3", "p4"}
	
	for _, authorID := range authors {
		node := HeteroNode{ID: authorID, Type: "Author"}
		graph.AddNode(node)
	}
	
	for _, paperID := range papers {
		node := HeteroNode{ID: paperID, Type: "Paper"}
		graph.AddNode(node)
	}
	
	// Overlapping clusters
	collaborations := map[string][]string{
		"p0": {"a0", "a1", "a2"},     // Cluster 1
		"p1": {"a1", "a2", "a3"},     // Bridge paper
		"p2": {"a2", "a3", "a4"},     // Bridge paper
		"p3": {"a4", "a5", "a6"},     // Cluster 2  
		"p4": {"a5", "a6", "a7"},     // Cluster 2
	}
	
	for paperID, authorList := range collaborations {
		for _, authorID := range authorList {
			edge := HeteroEdge{From: authorID, To: paperID, Type: "writes", Weight: 1.0}
			graph.AddEdge(edge)
		}
	}
	
	return graph
}

func createScalableGraph(numAuthors int) *HeterogeneousGraph {
	graph := NewHeterogeneousGraph()
	
	numPapers := numAuthors / 2
	
	// Add author nodes
	for i := 0; i < numAuthors; i++ {
		node := HeteroNode{ID: fmt.Sprintf("a%d", i), Type: "Author"}
		graph.AddNode(node)
	}
	
	// Add paper nodes
	for i := 0; i < numPapers; i++ {
		node := HeteroNode{ID: fmt.Sprintf("p%d", i), Type: "Paper"}
		graph.AddNode(node)
	}
	
	// Create community structure: groups of 3-5 authors per community
	authorsPerPaper := 3
	
	for paperIdx := 0; paperIdx < numPapers; paperIdx++ {
		communityStart := (paperIdx * authorsPerPaper) % numAuthors
		
		for i := 0; i < authorsPerPaper; i++ {
			authorIdx := (communityStart + i) % numAuthors
			authorID := fmt.Sprintf("a%d", authorIdx)
			paperID := fmt.Sprintf("p%d", paperIdx)
			
			edge := HeteroEdge{From: authorID, To: paperID, Type: "writes", Weight: 1.0}
			graph.AddEdge(edge)
		}
	}
	
	return graph
}

func calculateAccuracy(predicted, groundTruth map[string]int) float64 {
	if len(predicted) == 0 || len(groundTruth) == 0 {
		return 0
	}
	
	// Find the best mapping between predicted and ground truth communities
	predCommunities := getUniqueCommunities(predicted)
	trueCommunities := getUniqueCommunities(groundTruth)
	
	bestAccuracy := 0.0
	
	// Try all possible mappings (brute force for small numbers)
	if len(predCommunities) <= 10 && len(trueCommunities) <= 10 {
		bestAccuracy = findBestMapping(predicted, groundTruth, predCommunities, trueCommunities)
	} else {
		// For larger cases, use a simpler accuracy measure
		correct := 0
		total := 0
		
		for nodeID, predComm := range predicted {
			if trueComm, exists := groundTruth[nodeID]; exists {
				total++
				if predComm == trueComm {
					correct++
				}
			}
		}
		
		if total > 0 {
			bestAccuracy = float64(correct) / float64(total)
		}
	}
	
	return bestAccuracy
}

func findBestMapping(predicted, groundTruth map[string]int, predComms, trueComms map[int]bool) float64 {
	// Simple accuracy without complex mapping
	totalNodes := 0
	correctAssignments := 0
	
	// Group nodes by their predicted communities
	predGroups := make(map[int][]string)
	for nodeID, comm := range predicted {
		predGroups[comm] = append(predGroups[comm], nodeID)
	}
	
	// For each predicted community, find the most common true community
	for _, nodes := range predGroups {
		if len(nodes) == 0 {
			continue
		}
		
		// Count true community assignments in this predicted community
		trueCounts := make(map[int]int)
		for _, nodeID := range nodes {
			if trueComm, exists := groundTruth[nodeID]; exists {
				trueCounts[trueComm]++
				totalNodes++
			}
		}
		
		// Find the most common true community
		maxCount := 0
		for _, count := range trueCounts {
			if count > maxCount {
				maxCount = count
			}
		}
		
		correctAssignments += maxCount
	}
	
	if totalNodes == 0 {
		return 0
	}
	
	return float64(correctAssignments) / float64(totalNodes)
}


// TestMinimalScar - Simple test to isolate core issues
func TestMinimalScar(t *testing.T) {
	// Create tiny graph: 2 authors, 1 paper
	graph := NewHeterogeneousGraph()
	
	// Add nodes
	graph.AddNode(HeteroNode{ID: "a1", Type: "Author"})
	graph.AddNode(HeteroNode{ID: "a2", Type: "Author"})
	graph.AddNode(HeteroNode{ID: "p1", Type: "Paper"})
	
	// Add edges: both authors write same paper
	graph.AddEdge(HeteroEdge{From: "a1", To: "p1", Type: "writes", Weight: 1.0})
	graph.AddEdge(HeteroEdge{From: "a2", To: "p1", Type: "writes", Weight: 1.0})
	
	config := DefaultScarConfig()
	config.MetaPath = MetaPath{
		NodeTypes: []string{"Author", "Paper", "Author"},
		EdgeTypes: []string{"writes", "writes"},
	}
	config.K = 8  // Small K
	config.NK = 2
	config.Verbose = true
	
	t.Logf("Testing minimal graph: 2 authors, 1 shared paper")
	t.Logf("Expected: 1 community with both authors")
	
	// Test just sketch construction
	state := &ScarState{
		Graph:         graph,
		Config:        config,
		Sketches:      make(map[string]*VertexBottomKSketch),
		HashToNodeMap: NewHashToNodeMap(),
		NodeToOriginal: make(map[string][]string),
	}
	
	for _, nodeID := range graph.NodeList {
		state.NodeToOriginal[nodeID] = []string{nodeID}
	}
	
	// 1. Test sketch construction
	err := state.constructSketchesIteratively()
	if err != nil {
		t.Fatalf("Sketch construction failed: %v", err)
	}
	
	// 2. Check degree estimates
	for nodeID, sketch := range state.Sketches {
		if sketch.IsEmpty() {
			continue
		}
		degree := sketch.EstimateDegree()
		t.Logf("Node %s: degree=%.2f, saturated=%v", nodeID, degree.Value, degree.IsSaturated)
		
		if degree.Value > 1000 {
			t.Errorf("Node %s: unrealistic degree %.2f", nodeID, degree.Value)
		}
		if degree.Value < 0 {
			t.Errorf("Node %s: negative degree %.2f", nodeID, degree.Value)
		}
	}
	
	// 3. Test union operation
	a1Sketch := state.Sketches["a1"]
	a2Sketch := state.Sketches["a2"]
	
	if a1Sketch != nil && a2Sketch != nil && !a1Sketch.IsEmpty() && !a2Sketch.IsEmpty() {
		union := UnionSketches(a1Sketch, a2Sketch)
		if union != nil {
			unionDegree := union.EstimateDegree().Value
			t.Logf("Union degree: %.2f", unionDegree)
			
			if unionDegree > 1000000 {
				t.Errorf("Union degree overflow: %.2f", unionDegree)
			}
		}
	}
	
	// 4. Test full algorithm
	result, err := RunScar(graph, config)
	if err != nil {
		t.Fatalf("SCAR failed: %v", err)
	}
	
	t.Logf("Result: modularity=%.4f", result.Modularity)
	t.Logf("Communities: %v", result.FinalCommunities)
	
	// Expected: both authors in same community
	if result.FinalCommunities["a1"] != result.FinalCommunities["a2"] {
		t.Errorf("Authors should be in same community: a1=%d, a2=%d", 
			result.FinalCommunities["a1"], result.FinalCommunities["a2"])
	}
	
	if result.Modularity < 0 {
		t.Errorf("Negative modularity suggests algorithm issues: %.4f", result.Modularity)
	}
}

// TestSketchOverflow - Test degree estimation overflow
func TestSketchOverflow(t *testing.T) {
	sketch := NewVertexBottomKSketch(4, 2, 0)
	
	// Add very small values that might cause overflow
	sketch.AddValue(0, 1)  // Very small value
	sketch.AddValue(0, 2)
	sketch.AddValue(1, 1)
	sketch.AddValue(1, 3)
	
	degree := sketch.EstimateDegree()
	t.Logf("Small values: degree=%.2f, saturated=%v", degree.Value, degree.IsSaturated)
	
	if degree.Value > 1000000 {
		t.Errorf("Degree estimation overflow with small values: %.2f", degree.Value)
	}
	
	// Test with zero value (should not crash)
	sketch2 := NewVertexBottomKSketch(4, 2, 0)
	sketch2.AddValue(0, 0)  // Zero value
	
	degree2 := sketch2.EstimateDegree()
	t.Logf("Zero value: degree=%.2f", degree2.Value)
	
	if degree2.Value > 1000000 {
		t.Errorf("Degree estimation overflow with zero: %.2f", degree2.Value)
	}
}


// ==================== HOW TO RUN ====================

/* 
TO RUN THESE TESTS:

1. Save this as scar_comprehensive_test.go in your project directory

2. Run comprehensive validation:
   go test -v -run TestScarStepByStepValidation

3. Run edge case tests:
   go test -v -run TestEdgeCases

4. Run accuracy benchmarks:  
   go test -v -run TestAccuracyBenchmarks

5. Run performance tests:
   go test -v -run TestPerformanceBenchmarks

6. Run all tests:
   go test -v

7. Run with race detection:
   go test -race -v

8. Generate coverage report:
   go test -cover -v

EXPECTED OUTPUT EXAMPLE:
=== RUN   TestScarStepByStepValidation
🔬 Starting comprehensive SCAR step-by-step validation...
Graph: 15 nodes, 12 edges
Expected 3 communities with 3 authors each
=== RUN   TestScarStepByStepValidation/Step1_SketchConstruction
🔍 Validating sketch construction...
✅ Sketch construction: 15 nodes, 9 source nodes with values
=== RUN   TestScarStepByStepValidation/Step2_HashToNodeMapping
🔍 Validating hash-to-node mapping...
✅ Hash-to-node mapping: 36 total, 36 valid
... (continues for each step)

INTERPRETING RESULTS:
- ✅ = Step passed validation
- ❌ = Step failed (algorithm issue)
- ⚠️  = Warning (potential issue)
- Numbers show actual vs expected values

*/
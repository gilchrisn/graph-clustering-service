package materialization

import (
	// "context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// TestDataBuilder helps create test graphs and meta paths
type TestDataBuilder struct {
	graph    *models.HeterogeneousGraph
	metaPath *models.MetaPath
}

func NewTestDataBuilder() *TestDataBuilder {
	return &TestDataBuilder{
		graph: &models.HeterogeneousGraph{
			Nodes: make(map[string]models.Node),
			Edges: make([]models.Edge, 0),
		},
		metaPath: &models.MetaPath{},
	}
}

func (tdb *TestDataBuilder) AddNode(id, nodeType string, properties map[string]interface{}) *TestDataBuilder {
	if properties == nil {
		properties = make(map[string]interface{})
	}
	tdb.graph.Nodes[id] = models.Node{
		ID:         id,
		Type:       nodeType,
		Properties: properties,
	}
	return tdb
}

func (tdb *TestDataBuilder) AddEdge(from, to, edgeType string, weight float64) *TestDataBuilder {
	tdb.graph.Edges = append(tdb.graph.Edges, models.Edge{
		From:   from,
		To:     to,
		Type:   edgeType,
		Weight: weight,
	})
	return tdb
}

func (tdb *TestDataBuilder) SetMetaPath(id string, nodeSeq []string, edgeSeq []string) *TestDataBuilder {
	tdb.metaPath = &models.MetaPath{
		ID:           id,
		NodeSequence: nodeSeq,
		EdgeSequence: edgeSeq,
	}
	return tdb
}

func (tdb *TestDataBuilder) Build() (*models.HeterogeneousGraph, *models.MetaPath) {
	return tdb.graph, tdb.metaPath
}

// ==========================
// INPUT VALIDATION TESTS
// ==========================

func TestMaterializationEngine_InputValidation(t *testing.T) {
	tests := []struct {
		name          string
		setupFunc     func() (*models.HeterogeneousGraph, *models.MetaPath)
		expectError   bool
		errorContains string
	}{
		{
			name: "nil_graph",
			setupFunc: func() (*models.HeterogeneousGraph, *models.MetaPath) {
				return nil, &models.MetaPath{
					ID:           "test",
					NodeSequence: []string{"Author"},
					EdgeSequence: []string{},
				}
			},
			expectError:   true,
			errorContains: "heterogeneous graph cannot be nil",
		},
		{
			name: "nil_meta_path",
			setupFunc: func() (*models.HeterogeneousGraph, *models.MetaPath) {
				graph, _ := NewTestDataBuilder().
					AddNode("a1", "Author", nil).
					Build()
				return graph, nil
			},
			expectError:   true,
			errorContains: "meta path cannot be nil",
		},
		{
			name: "empty_graph_nodes",
			setupFunc: func() (*models.HeterogeneousGraph, *models.MetaPath) {
				return &models.HeterogeneousGraph{
						Nodes: make(map[string]models.Node),
						Edges: []models.Edge{},
					}, &models.MetaPath{
						ID:           "test",
						NodeSequence: []string{"Author"},
						EdgeSequence: []string{},
					}
			},
			expectError:   true,
			errorContains: "graph has no nodes",
		},
		{
			name: "empty_graph_edges",
			setupFunc: func() (*models.HeterogeneousGraph, *models.MetaPath) {
				graph, metaPath := NewTestDataBuilder().
					AddNode("a1", "Author", nil).
					SetMetaPath("test", []string{"Author", "Paper"}, []string{"writes"}).
					Build()
				return graph, metaPath
			},
			expectError:   true,
			errorContains: "graph has no edges",
		},
		{
			name: "missing_start_node_type",
			setupFunc: func() (*models.HeterogeneousGraph, *models.MetaPath) {
				graph, metaPath := NewTestDataBuilder().
					AddNode("p1", "Paper", nil).
					AddEdge("p1", "p2", "cites", 1.0).
					SetMetaPath("test", []string{"Author", "Paper"}, []string{"writes"}).
					Build()
				return graph, metaPath
			},
			expectError:   true,
			errorContains: "no nodes of starting type 'Author' found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			graph, metaPath := tt.setupFunc()
			config := DefaultMaterializationConfig()
			
			engine := NewMaterializationEngine(graph, metaPath, config, nil)
			result, err := engine.Materialize()

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				} else if tt.errorContains != "" && !contains(err.Error(), tt.errorContains) {
					t.Errorf("Expected error to contain '%s', got '%s'", tt.errorContains, err.Error())
				}
				if result == nil || result.Success {
					t.Errorf("Expected failed result")
				}
			} else {
				if err != nil {
					t.Errorf("Expected no error but got: %v", err)
				}
				if result == nil || !result.Success {
					t.Errorf("Expected successful result")
				}
			}
		})
	}
}

// ==========================
// INSTANCE GENERATION TESTS
// ==========================

func TestInstanceGenerator_SimpleLinearPath(t *testing.T) {
	// Author -> writes -> Paper
	graph, metaPath := NewTestDataBuilder().
		AddNode("a1", "Author", map[string]interface{}{"name": "Alice"}).
		AddNode("a2", "Author", map[string]interface{}{"name": "Bob"}).
		AddNode("p1", "Paper", map[string]interface{}{"title": "Paper1"}).
		AddNode("p2", "Paper", map[string]interface{}{"title": "Paper2"}).
		AddEdge("a1", "p1", "writes", 1.0).
		AddEdge("a2", "p2", "writes", 1.0).
		SetMetaPath("author-paper", []string{"Author", "Paper"}, []string{"writes"}).
		Build()

	config := DefaultTraversalConfig()
	generator := NewInstanceGenerator(graph, metaPath, config)

	instances, stats, err := generator.FindAllInstances(nil)

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(instances) != 2 {
		t.Errorf("Expected 2 instances, got %d", len(instances))
	}

	// Verify instance structure
	for _, instance := range instances {
		if !instance.IsValid() {
			t.Errorf("Invalid instance: %v", instance)
		}
		if len(instance.Nodes) != 2 {
			t.Errorf("Expected 2 nodes in instance, got %d", len(instance.Nodes))
		}
		if len(instance.Edges) != 1 {
			t.Errorf("Expected 1 edge in instance, got %d", len(instance.Edges))
		}
	}

	// Verify statistics
	if stats.StartingNodes != 2 {
		t.Errorf("Expected 2 starting nodes, got %d", stats.StartingNodes)
	}
}

func TestInstanceGenerator_ComplexPath(t *testing.T) {
	// Author -> writes -> Paper -> cites -> Paper
	graph, metaPath := NewTestDataBuilder().
		AddNode("a1", "Author", nil).
		AddNode("p1", "Paper", nil).
		AddNode("p2", "Paper", nil).
		AddNode("p3", "Paper", nil).
		AddEdge("a1", "p1", "writes", 2.0).
		AddEdge("p1", "p2", "cites", 3.0).
		AddEdge("p1", "p3", "cites", 1.5).
		SetMetaPath("author-paper-paper", 
			[]string{"Author", "Paper", "Paper"}, 
			[]string{"writes", "cites"}).
		Build()

	config := DefaultTraversalConfig()
	generator := NewInstanceGenerator(graph, metaPath, config)

	instances, _, err := generator.FindAllInstances(nil)

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(instances) != 2 {
		t.Errorf("Expected 2 instances, got %d", len(instances))
	}

	// Check accumulated weights
	expectedWeights := []float64{6.0, 3.0} // 2*3=6, 2*1.5=3
	actualWeights := make([]float64, len(instances))
	for i, instance := range instances {
		actualWeights[i] = instance.Weight
	}

	for _, expected := range expectedWeights {
		found := false
		for _, actual := range actualWeights {
			if math.Abs(actual-expected) < 0.001 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected weight %f not found in actual weights %v", expected, actualWeights)
		}
	}
}

func TestInstanceGenerator_CycleDetection(t *testing.T) {
	// Create a graph with potential cycles
	graph, metaPath := NewTestDataBuilder().
		AddNode("a1", "Author", nil).
		AddNode("a2", "Author", nil).
		AddNode("p1", "Paper", nil).
		AddEdge("a1", "p1", "writes", 1.0).
		AddEdge("p1", "a2", "writtenBy", 1.0).
		AddEdge("a2", "a1", "collaborates", 1.0).
		SetMetaPath("author-paper-author-author", 
			[]string{"Author", "Paper", "Author", "Author"}, 
			[]string{"writes", "writtenBy", "collaborates"}).
		Build()

	// Test with cycles disallowed
	configNoCycles := DefaultTraversalConfig()
	configNoCycles.AllowCycles = false
	generator := NewInstanceGenerator(graph, metaPath, configNoCycles)

	instances, stats, err := generator.FindAllInstances(nil)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(instances) != 0 {
		t.Errorf("Expected 0 instances with cycles disallowed, got %d", len(instances))
	}
	if stats.CyclesDetected == 0 {
		t.Errorf("Expected some cycles to be detected")
	}

	// Test with cycles allowed
	configWithCycles := DefaultTraversalConfig()
	configWithCycles.AllowCycles = true
	generator = NewInstanceGenerator(graph, metaPath, configWithCycles)

	instances, _, err = generator.FindAllInstances(nil)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(instances) != 1 {
		t.Errorf("Expected 1 instance with cycles allowed, got %d", len(instances))
	}
}

func TestInstanceGenerator_MemoryLimits(t *testing.T) {
	// Create a large graph that could exceed memory limits
	builder := NewTestDataBuilder()
	
	// Add many authors and papers
	for i := 0; i < 100; i++ {
		builder.AddNode(fmt.Sprintf("a%d", i), "Author", nil)
		builder.AddNode(fmt.Sprintf("p%d", i), "Paper", nil)
		builder.AddEdge(fmt.Sprintf("a%d", i), fmt.Sprintf("p%d", i), "writes", 1.0)
		
		// Cross-connect to create many possible paths
		if i > 0 {
			builder.AddEdge(fmt.Sprintf("p%d", i), fmt.Sprintf("p%d", i-1), "cites", 1.0)
		}
	}

	graph, metaPath := builder.
		SetMetaPath("test", []string{"Author", "Paper", "Paper"}, []string{"writes", "cites"}).
		Build()

	config := DefaultTraversalConfig()
	config.MaxInstances = 50 // Limit to 50 instances

	generator := NewInstanceGenerator(graph, metaPath, config)
	instances, _, err := generator.FindAllInstances(nil)

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(instances) > 50 {
		t.Errorf("Expected at most 50 instances due to limit, got %d", len(instances))
	}
}

func TestInstanceGenerator_TimeoutHandling(t *testing.T) {
	// Create a complex graph that takes time to traverse
	builder := NewTestDataBuilder()
	
	// Create a highly connected graph
	for i := 0; i < 50; i++ {
		builder.AddNode(fmt.Sprintf("a%d", i), "Author", nil)
		for j := 0; j < 20; j++ {
			builder.AddNode(fmt.Sprintf("p%d_%d", i, j), "Paper", nil)
			builder.AddEdge(fmt.Sprintf("a%d", i), fmt.Sprintf("p%d_%d", i, j), "writes", 1.0)
		}
	}

	graph, metaPath := builder.
		SetMetaPath("test", []string{"Author", "Paper"}, []string{"writes"}).
		Build()

	config := DefaultTraversalConfig()
	config.TimeoutSeconds = 1 // Very short timeout

	generator := NewInstanceGenerator(graph, metaPath, config)
	start := time.Now()
	_, stats, _ := generator.FindAllInstances(nil)
	duration := time.Since(start)

	// Should either complete quickly or timeout
	if duration > 2*time.Second {
		t.Errorf("Processing took too long: %v", duration)
	}

	// Check if timeout was properly detected
	if duration > time.Second && !stats.TimeoutOccurred {
		t.Errorf("Expected timeout to be detected for long operation")
	}
}

func TestInstanceGenerator_ParallelProcessing(t *testing.T) {
	// Create a graph suitable for parallel processing
	builder := NewTestDataBuilder()
	
	for i := 0; i < 20; i++ {
		builder.AddNode(fmt.Sprintf("a%d", i), "Author", nil)
		builder.AddNode(fmt.Sprintf("p%d", i), "Paper", nil)
		builder.AddEdge(fmt.Sprintf("a%d", i), fmt.Sprintf("p%d", i), "writes", 1.0)
	}

	graph, metaPath := builder.
		SetMetaPath("test", []string{"Author", "Paper"}, []string{"writes"}).
		Build()

	// Test with different parallelism levels
	parallelismLevels := []int{1, 2, 4, 8}
	
	for _, parallelism := range parallelismLevels {
		t.Run(fmt.Sprintf("parallelism_%d", parallelism), func(t *testing.T) {
			config := DefaultTraversalConfig()
			config.Parallelism = parallelism

			generator := NewInstanceGenerator(graph, metaPath, config)
			instances, stats, err := generator.FindAllInstances(nil)

			if err != nil {
				t.Fatalf("Expected no error with parallelism %d, got: %v", parallelism, err)
			}

			if len(instances) != 20 {
				t.Errorf("Expected 20 instances, got %d", len(instances))
			}

			// Verify worker utilization makes sense
			totalWork := 0
			for _, work := range stats.WorkerUtilization {
				totalWork += work
			}
			
			if parallelism > 1 && len(stats.WorkerUtilization) < 2 {
				t.Errorf("Expected multiple workers to be used with parallelism %d", parallelism)
			}
		})
	}
}

// ==========================
// HOMOGENEOUS BUILDER TESTS
// ==========================

func TestHomogeneousBuilder_DirectTraversal(t *testing.T) {
	metaPath := &models.MetaPath{
		ID:           "test",
		NodeSequence: []string{"Author", "Paper", "Author"},
		EdgeSequence: []string{"writes", "writtenBy"},
	}

	config := DefaultAggregationConfig()
	config.Interpretation = DirectTraversal
	builder := NewHomogeneousBuilder(metaPath, config)

	// Add test instances
	instances := []PathInstance{
		{Nodes: []string{"a1", "p1", "a2"}, Edges: []string{"writes", "writtenBy"}, Weight: 2.0},
		{Nodes: []string{"a1", "p2", "a3"}, Edges: []string{"writes", "writtenBy"}, Weight: 3.0},
		{Nodes: []string{"a2", "p3", "a1"}, Edges: []string{"writes", "writtenBy"}, Weight: 1.5},
	}

	for _, instance := range instances {
		builder.AddInstance(instance)
	}

	homogGraph, stats := builder.Build()

	// Verify graph structure
	if len(homogGraph.Nodes) != 3 {
		t.Errorf("Expected 3 nodes, got %d", len(homogGraph.Nodes))
	}

	expectedEdges := 3
	if len(homogGraph.Edges) != expectedEdges {
		t.Errorf("Expected %d edges, got %d", expectedEdges, len(homogGraph.Edges))
	}

	// Verify statistics
	if stats.InstancesAggregated != 3 {
		t.Errorf("Expected 3 instances aggregated, got %d", stats.InstancesAggregated)
	}
}

func TestHomogeneousBuilder_MeetingBased(t *testing.T) {
	metaPath := &models.MetaPath{
		ID:           "test",
		NodeSequence: []string{"Author", "Venue"},
		EdgeSequence: []string{"publishedAt"},
	}

	config := DefaultAggregationConfig()
	config.Interpretation = MeetingBased
	config.Symmetric = true
	builder := NewHomogeneousBuilder(metaPath, config)

	// Add instances that meet at venues (Author -> Venue)
	instances := []PathInstance{
		{Nodes: []string{"a1", "v1"}, Edges: []string{"publishedAt"}, Weight: 1.0},
		{Nodes: []string{"a2", "v1"}, Edges: []string{"publishedAt"}, Weight: 1.0},
		{Nodes: []string{"a3", "v1"}, Edges: []string{"publishedAt"}, Weight: 2.0},
		{Nodes: []string{"a1", "v2"}, Edges: []string{"publishedAt"}, Weight: 1.5},
		{Nodes: []string{"a2", "v2"}, Edges: []string{"publishedAt"}, Weight: 1.0},
	}

	for _, instance := range instances {
		builder.AddInstance(instance)
	}

	homogGraph, _ := builder.Build()

	// In meeting-based interpretation, we should have edges between authors
	// who meet at the same venue
	expectedPairs := []EdgeKey{
		{From: "a1", To: "a2"}, {From: "a2", To: "a1"}, // Meet at v1 and v2
		{From: "a1", To: "a3"}, {From: "a3", To: "a1"}, // Meet at v1
		{From: "a2", To: "a3"}, {From: "a3", To: "a2"}, // Meet at v1
	}

	for _, expectedEdge := range expectedPairs {
		if _, exists := homogGraph.Edges[expectedEdge]; !exists {
			t.Errorf("Expected edge %v not found", expectedEdge)
		}
	}

	// Check that weights are calculated correctly (should be > 0)
	for edge, weight := range homogGraph.Edges {
		if weight <= 0 {
			t.Errorf("Edge %v has non-positive weight: %f", edge, weight)
		}
	}
}

func TestHomogeneousBuilder_AggregationStrategies(t *testing.T) {
	metaPath := &models.MetaPath{
		ID:           "test",
		NodeSequence: []string{"Author", "Author"},
		EdgeSequence: []string{"collaborates"},
	}

	// Create test instances with different weights
	instances := []PathInstance{
		{Nodes: []string{"a1", "a2"}, Edges: []string{"collaborates"}, Weight: 2.0},
		{Nodes: []string{"a1", "a2"}, Edges: []string{"collaborates"}, Weight: 3.0},
		{Nodes: []string{"a1", "a2"}, Edges: []string{"collaborates"}, Weight: 1.0},
	}

	strategies := map[AggregationStrategy]float64{
		Count:   3.0,   // 3 instances
		Sum:     6.0,   // 2+3+1
		Average: 2.0,   // 6/3
		Maximum: 3.0,   // max(2,3,1)
		Minimum: 1.0,   // min(2,3,1)
	}

	for strategy, expectedWeight := range strategies {
		t.Run(fmt.Sprintf("strategy_%d", strategy), func(t *testing.T) {
			config := DefaultAggregationConfig()
			config.Strategy = strategy
			builder := NewHomogeneousBuilder(metaPath, config)

			for _, instance := range instances {
				builder.AddInstance(instance)
			}

			homogGraph, _ := builder.Build()
			
			edgeKey := EdgeKey{From: "a1", To: "a2"}
			actualWeight, exists := homogGraph.Edges[edgeKey]
			
			if !exists {
				t.Fatalf("Expected edge not found")
			}
			
			if math.Abs(actualWeight-expectedWeight) > 0.001 {
				t.Errorf("Expected weight %f, got %f", expectedWeight, actualWeight)
			}
		})
	}
}

// func TestHomogeneousBuilder_EdgeFiltering(t *testing.T) {
// 	metaPath := &models.MetaPath{
// 		ID:           "test",
// 		NodeSequence: []string{"Author", "Author"},
// 		EdgeSequence: []string{"collaborates"},
// 	}

// 	config := DefaultAggregationConfig()
// 	config.MinWeight = 2.0 // Filter edges below 2.0
// 	config.MaxEdges = 2    // Keep only top 2 edges

// 	builder := NewHomogeneousBuilder(metaPath, config)

// 	// Add instances with different weights - make sure to include the Edges field
// 	instances := []PathInstance{
// 		{Nodes: []string{"a1", "a2"}, Edges: []string{"collaborates"}, Weight: 3.0}, // Should be kept
// 		{Nodes: []string{"a2", "a3"}, Edges: []string{"collaborates"}, Weight: 2.5}, // Should be kept
// 		{Nodes: []string{"a3", "a4"}, Edges: []string{"collaborates"}, Weight: 1.0}, // Should be filtered (below minWeight)
// 		{Nodes: []string{"a4", "a5"}, Edges: []string{"collaborates"}, Weight: 2.2}, // Should be filtered (exceeds maxEdges)
// 	}

// 	for _, instance := range instances {
// 		builder.AddInstance(instance)
// 	}

// 	homogGraph, stats := builder.Build()

// 	// Should have exactly 2 edges due to MaxEdges limit
// 	if len(homogGraph.Edges) != 2 {
// 		t.Errorf("Expected 2 edges after filtering, got %d", len(homogGraph.Edges))
// 		// Debug: print actual edges
// 		for edge, weight := range homogGraph.Edges {
// 			t.Logf("Edge: %v, Weight: %f", edge, weight)
// 		}
// 	}

// 	// Check that the highest weight edges remain
// 	edge1 := EdgeKey{From: "a1", To: "a2"}
// 	edge2 := EdgeKey{From: "a2", To: "a3"}
	
// 	if weight1, exists := homogGraph.Edges[edge1]; !exists {
// 		t.Errorf("Expected high-weight edge %v to remain", edge1)
// 	} else if weight1 != 3.0 {
// 		t.Errorf("Expected edge weight 3.0, got %f", weight1)
// 	}
	
// 	if weight2, exists := homogGraph.Edges[edge2]; !exists {
// 		t.Errorf("Expected high-weight edge %v to remain", edge2)
// 	} else if weight2 != 2.5 {
// 		t.Errorf("Expected edge weight 2.5, got %f", weight2)
// 	}

// 	// Check that low-weight edges are filtered
// 	lowWeightEdge := EdgeKey{From: "a3", To: "a4"}
// 	if _, exists := homogGraph.Edges[lowWeightEdge]; exists {
// 		t.Errorf("Expected low-weight edge to be filtered")
// 	}

// 	// Verify filtering statistics
// 	if stats.EdgesFiltered == 0 {
// 		t.Errorf("Expected some edges to be filtered")
// 	}
// }

// ==========================
// WEIGHT CALCULATOR TESTS
// ==========================

func TestWeightCalculator_DegreeNormalization(t *testing.T) {
	// Create a simple graph for testing
	graph := &HomogeneousGraph{
		NodeType: "Author",
		Nodes: map[string]Node{
			"a1": {ID: "a1", Type: "Author", Degree: 0},
			"a2": {ID: "a2", Type: "Author", Degree: 0},
			"a3": {ID: "a3", Type: "Author", Degree: 0},
		},
		Edges: map[EdgeKey]float64{
			{From: "a1", To: "a2"}: 4.0,
			{From: "a1", To: "a3"}: 2.0,
			{From: "a2", To: "a3"}: 6.0,
		},
	}

	config := DefaultAggregationConfig()
	config.Normalization = DegreeNorm
	calculator := NewWeightCalculator(config)

	err := calculator.ProcessGraph(graph)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// After degree normalization, weights should be different
	for edge, weight := range graph.Edges {
		if weight <= 0 {
			t.Errorf("Edge %v has non-positive weight after normalization: %f", edge, weight)
		}
		if weight >= 6.0 {
			t.Errorf("Edge %v weight not normalized (still too high): %f", edge, weight)
		}
	}
}

func TestWeightCalculator_MaxNormalization(t *testing.T) {
	graph := &HomogeneousGraph{
		NodeType: "Author",
		Nodes: map[string]Node{
			"a1": {ID: "a1", Type: "Author"},
			"a2": {ID: "a2", Type: "Author"},
		},
		Edges: map[EdgeKey]float64{
			{From: "a1", To: "a2"}: 10.0,
			{From: "a2", To: "a1"}: 5.0,
		},
	}

	config := DefaultAggregationConfig()
	config.Normalization = MaxNorm
	calculator := NewWeightCalculator(config)

	err := calculator.ProcessGraph(graph)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// After max normalization, all weights should be in [0,1]
	for edge, weight := range graph.Edges {
		if weight < 0 || weight > 1 {
			t.Errorf("Edge %v weight not in [0,1] after max normalization: %f", edge, weight)
		}
	}

	// The highest weight should be 1.0
	maxWeight := 0.0
	for _, weight := range graph.Edges {
		if weight > maxWeight {
			maxWeight = weight
		}
	}
	if math.Abs(maxWeight-1.0) > 0.001 {
		t.Errorf("Expected max weight to be 1.0, got %f", maxWeight)
	}
}

func TestWeightCalculator_StandardNormalization(t *testing.T) {
	graph := &HomogeneousGraph{
		NodeType: "Author",
		Nodes: map[string]Node{
			"a1": {ID: "a1", Type: "Author"},
			"a2": {ID: "a2", Type: "Author"},
			"a3": {ID: "a3", Type: "Author"},
		},
		Edges: map[EdgeKey]float64{
			{From: "a1", To: "a2"}: 1.0,
			{From: "a2", To: "a3"}: 2.0,
			{From: "a1", To: "a3"}: 3.0,
		},
	}

	config := DefaultAggregationConfig()
	config.Normalization = StandardNorm
	calculator := NewWeightCalculator(config)

	err := calculator.ProcessGraph(graph)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// After standard normalization, mean should be ~0 and std ~1
	sum := 0.0
	count := 0
	for _, weight := range graph.Edges {
		sum += weight
		count++
	}
	mean := sum / float64(count)

	if math.Abs(mean) > 0.001 {
		t.Errorf("Expected mean to be ~0 after standard normalization, got %f", mean)
	}
}

func TestWeightCalculator_InvalidWeights(t *testing.T) {
	config := DefaultAggregationConfig()
	calculator := NewWeightCalculator(config)

	testCases := []struct {
		name    string
		weights map[EdgeKey]float64
		expectError bool
	}{
		{
			name: "nan_weight",
			weights: map[EdgeKey]float64{
				{From: "a1", To: "a2"}: math.NaN(),
			},
			expectError: true,
		},
		{
			name: "infinite_weight",
			weights: map[EdgeKey]float64{
				{From: "a1", To: "a2"}: math.Inf(1),
			},
			expectError: true,
		},
		{
			name: "negative_weight",
			weights: map[EdgeKey]float64{
				{From: "a1", To: "a2"}: -1.0,
			},
			expectError: true,
		},
		{
			name: "valid_weights",
			weights: map[EdgeKey]float64{
				{From: "a1", To: "a2"}: 1.0,
				{From: "a2", To: "a3"}: 2.5,
			},
			expectError: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			graph := &HomogeneousGraph{
				NodeType: "Author",
				Nodes: map[string]Node{
					"a1": {ID: "a1", Type: "Author"},
					"a2": {ID: "a2", Type: "Author"},
					"a3": {ID: "a3", Type: "Author"},
				},
				Edges: tc.weights,
			}

			err := calculator.ValidateWeights(graph)
			
			if tc.expectError && err == nil {
				t.Errorf("Expected error for %s but got none", tc.name)
			}
			if !tc.expectError && err != nil {
				t.Errorf("Expected no error for %s but got: %v", tc.name, err)
			}
		})
	}
}

// ==========================
// CONFIGURATION TESTS
// ==========================

func TestMaterializationConfig_Validation(t *testing.T) {
	tests := []struct {
		name        string
		configFunc  func() MaterializationConfig
		expectError bool
	}{
		{
			name: "valid_config",
			configFunc: func() MaterializationConfig {
				return DefaultMaterializationConfig()
			},
			expectError: false,
		},
		{
			name: "negative_min_weight",
			configFunc: func() MaterializationConfig {
				config := DefaultMaterializationConfig()
				config.Aggregation.MinWeight = -1.0
				return config
			},
			expectError: true,
		},
		{
			name: "negative_max_edges",
			configFunc: func() MaterializationConfig {
				config := DefaultMaterializationConfig()
				config.Aggregation.MaxEdges = -1
				return config
			},
			expectError: true,
		},
		{
			name: "zero_parallelism",
			configFunc: func() MaterializationConfig {
				config := DefaultMaterializationConfig()
				config.Traversal.Parallelism = 0
				return config
			},
			expectError: false, // Should default to 1
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := tt.configFunc()
			
			// Test with aggregation builder
			metaPath := &models.MetaPath{
				ID: "test",
				NodeSequence: []string{"Author"},
				EdgeSequence: []string{},
			}
			
			builder := NewHomogeneousBuilder(metaPath, config.Aggregation)
			err := builder.ValidateConfiguration()

			if tt.expectError && err == nil {
				t.Errorf("Expected validation error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Expected no validation error but got: %v", err)
			}
		})
	}
}

// ==========================
// INTEGRATION TESTS
// ==========================

func TestMaterializationEngine_EndToEnd_AuthorCollaboration(t *testing.T) {
	// Create a realistic author collaboration graph
	// Authors -> Papers -> Authors (collaboration through papers)
	builder := NewTestDataBuilder()
	
	// Add authors
	authors := []string{"alice", "bob", "charlie", "diana"}
	for _, author := range authors {
		builder.AddNode(author, "Author", map[string]interface{}{"name": author})
	}
	
	// Add papers
	papers := []string{"paper1", "paper2", "paper3", "paper4"}
	for _, paper := range papers {
		builder.AddNode(paper, "Paper", map[string]interface{}{"title": paper})
	}
	
	// Add authorship relationships - make sure all authors have at least one paper
	collaborations := map[string][]string{
		"paper1": {"alice", "bob"},      // Alice-Bob collaboration
		"paper2": {"bob", "charlie"},    // Bob-Charlie collaboration  
		"paper3": {"alice", "charlie"},  // Alice-Charlie collaboration
		"paper4": {"diana", "alice"},    // Diana-Alice collaboration (so diana appears in result)
	}
	
	for paper, paperAuthors := range collaborations {
		for _, author := range paperAuthors {
			builder.AddEdge(author, paper, "writes", 1.0)
			builder.AddEdge(paper, author, "writtenBy", 1.0)
		}
	}

	graph, metaPath := builder.
		SetMetaPath("author-collaboration", 
			[]string{"Author", "Paper", "Author"}, 
			[]string{"writes", "writtenBy"}).
		Build()

	// Configure for collaboration detection
	config := DefaultMaterializationConfig()
	config.Aggregation.Strategy = Count
	config.Aggregation.Symmetric = true

	// Track progress
	progressCallbacks := 0
	progressCb := func(current, total int, message string) {
		progressCallbacks++
		t.Logf("Progress: %d/%d - %s", current, total, message)
	}

	engine := NewMaterializationEngine(graph, metaPath, config, progressCb)
	result, err := engine.Materialize()

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if !result.Success {
		t.Fatalf("Expected successful materialization, got error: %s", result.Error)
	}

	// Verify the result
	homogGraph := result.HomogeneousGraph
	
	// Should have 4 author nodes now (with diana included)
	if len(homogGraph.Nodes) != 4 {
		t.Errorf("Expected 4 author nodes, got %d", len(homogGraph.Nodes))
	}

	// Should have collaboration edges
	expectedCollaborations := []struct {
		from, to string
	}{
		{"alice", "bob"},     // Through paper1
		{"bob", "alice"},     // Symmetric
		{"bob", "charlie"},   // Through paper2
		{"charlie", "bob"},   // Symmetric
		{"alice", "charlie"}, // Through paper3
		{"charlie", "alice"}, // Symmetric
		{"diana", "alice"},   // Through paper4
		{"alice", "diana"},   // Symmetric
	}

	for _, collab := range expectedCollaborations {
		edge := EdgeKey{From: collab.from, To: collab.to}
		if weight, exists := homogGraph.Edges[edge]; !exists {
			t.Errorf("Expected collaboration edge %v not found", edge)
		} else if weight <= 0 {
			t.Errorf("Collaboration edge %v has non-positive weight: %f", edge, weight)
		}
	}

	// Verify statistics
	if result.Statistics.RuntimeMS <= 0 {
		t.Errorf("Expected positive runtime, got %d", result.Statistics.RuntimeMS)
	}

	if result.Statistics.InstancesGenerated <= 0 {
		t.Errorf("Expected some instances to be generated")
	}

	if progressCallbacks == 0 {
		t.Errorf("Expected progress callbacks to be called")
	}

	t.Logf("Materialization completed: %d nodes, %d edges, %d instances",
		len(homogGraph.Nodes), len(homogGraph.Edges), result.Statistics.InstancesGenerated)
}

func TestMaterializationEngine_EndToEnd_CitationNetwork(t *testing.T) {
	// Create a paper citation network
	builder := NewTestDataBuilder()
	
	// Add papers
	papers := []string{"p1", "p2", "p3", "p4", "p5"}
	for _, paper := range papers {
		builder.AddNode(paper, "Paper", map[string]interface{}{"title": paper})
	}
	
	// Add citation relationships (p1 cites p2, etc.)
	citations := []struct{ from, to string }{
		{"p1", "p2"}, {"p1", "p3"},
		{"p2", "p3"}, {"p2", "p4"},
		{"p3", "p4"}, {"p3", "p5"},
		{"p4", "p5"},
	}
	
	for _, citation := range citations {
		builder.AddEdge(citation.from, citation.to, "cites", 1.0)
	}

	graph, metaPath := builder.
		SetMetaPath("paper-citation-cocitation", 
			[]string{"Paper", "Paper", "Paper"}, 
			[]string{"cites", "cites"}).
		Build()

	config := DefaultMaterializationConfig()
	config.Aggregation.Strategy = Count
	config.Aggregation.MinWeight = 1.0

	engine := NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if !result.Success {
		t.Fatalf("Expected successful materialization")
	}

	homogGraph := result.HomogeneousGraph

	// Verify that papers with citation paths are connected
	if len(homogGraph.Edges) == 0 {
		t.Errorf("Expected some edges in citation network")
	}

	// Papers that are 2-hops apart should be connected
	// e.g., p1 -> p2 -> p3, so (p1,p3) should be an edge
	expectedEdge := EdgeKey{From: "p1", To: "p3"}
	if _, exists := homogGraph.Edges[expectedEdge]; !exists {
		t.Errorf("Expected 2-hop citation edge %v not found", expectedEdge)
	}
}

func TestMaterializationEngine_BatchProcessing(t *testing.T) {
	// Create a graph with multiple possible meta paths
	graph, _ := NewTestDataBuilder().
		AddNode("a1", "Author", nil).
		AddNode("a2", "Author", nil).
		AddNode("p1", "Paper", nil).
		AddNode("v1", "Venue", nil).
		AddEdge("a1", "p1", "writes", 1.0).
		AddEdge("a2", "p1", "writes", 1.0).
		AddEdge("p1", "v1", "publishedAt", 1.0).
		Build()

	// Create multiple meta paths
	metaPaths := []*models.MetaPath{
		{
			ID:           "author-paper-author",
			NodeSequence: []string{"Author", "Paper", "Author"},
			EdgeSequence: []string{"writes", "writes"},
		},
		{
			ID:           "author-paper-venue",
			NodeSequence: []string{"Author", "Paper", "Venue"},
			EdgeSequence: []string{"writes", "publishedAt"},
		},
	}

	config := DefaultMaterializationConfig()
	
	progressCallbacks := 0
	progressCb := func(current, total int, message string) {
		progressCallbacks++
	}

	results, err := BatchMaterialize(graph, metaPaths, config, progressCb)

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(results) != len(metaPaths) {
		t.Errorf("Expected %d results, got %d", len(metaPaths), len(results))
	}

	// All results should be successful
	for i, result := range results {
		if !result.Success {
			t.Errorf("Result %d failed: %s", i, result.Error)
		}
	}

	if progressCallbacks == 0 {
		t.Errorf("Expected progress callbacks during batch processing")
	}
}

// ==========================
// EDGE CASE TESTS
// ==========================

func TestMaterializationEngine_EmptyResult(t *testing.T) {
	// Create a graph where the meta path has no instances
	graph, metaPath := NewTestDataBuilder().
		AddNode("a1", "Author", nil).
		AddNode("p1", "Paper", nil).
		AddEdge("a1", "p1", "writes", 1.0).
		SetMetaPath("impossible", 
			[]string{"Author", "Venue"}, // No venues in graph
			[]string{"publishedAt"}).
		Build()

	config := DefaultMaterializationConfig()
	engine := NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()

	if err != nil {
		t.Fatalf("Expected no error for empty result, got: %v", err)
	}

	if !result.Success {
		t.Fatalf("Expected successful result even with no instances")
	}

	// Result should have empty graph
	if len(result.HomogeneousGraph.Nodes) != 0 {
		t.Errorf("Expected 0 nodes for empty result, got %d", len(result.HomogeneousGraph.Nodes))
	}

	if len(result.HomogeneousGraph.Edges) != 0 {
		t.Errorf("Expected 0 edges for empty result, got %d", len(result.HomogeneousGraph.Edges))
	}
}

func TestMaterializationEngine_SingleNode(t *testing.T) {
	// Test with a meta path that returns to the same node
	graph, metaPath := NewTestDataBuilder().
		AddNode("a1", "Author", nil).
		AddNode("p1", "Paper", nil).
		AddEdge("a1", "p1", "writes", 1.0).
		AddEdge("p1", "a1", "writtenBy", 1.0).
		SetMetaPath("self-path", 
			[]string{"Author", "Paper", "Author"}, 
			[]string{"writes", "writtenBy"}).
		Build()

	config := DefaultMaterializationConfig()
	engine := NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if !result.Success {
		t.Fatalf("Expected successful result")
	}

	// Should have one node but no edges (self-loops removed by default)
	if len(result.HomogeneousGraph.Nodes) != 1 {
		t.Errorf("Expected 1 node, got %d", len(result.HomogeneousGraph.Nodes))
	}
}

func TestMaterializationEngine_MemoryEstimation(t *testing.T) {
	// Create a moderately sized graph
	builder := NewTestDataBuilder()
	for i := 0; i < 100; i++ {
		builder.AddNode(fmt.Sprintf("a%d", i), "Author", nil)
		builder.AddNode(fmt.Sprintf("p%d", i), "Paper", nil)
		builder.AddEdge(fmt.Sprintf("a%d", i), fmt.Sprintf("p%d", i), "writes", 1.0)
	}

	graph, metaPath := builder.
		SetMetaPath("test", []string{"Author", "Paper"}, []string{"writes"}).
		Build()

	config := DefaultMaterializationConfig()
	engine := NewMaterializationEngine(graph, metaPath, config, nil)

	// Test complexity estimation
	complexity, err := engine.EstimateComplexity()
	if err != nil {
		t.Fatalf("Expected no error estimating complexity, got: %v", err)
	}

	if complexity <= 0 {
		t.Errorf("Expected positive complexity estimate, got %d", complexity)
	}

	// Test memory estimation
	memoryMB, err := engine.GetMemoryEstimate()
	if err != nil {
		t.Fatalf("Expected no error estimating memory, got: %v", err)
	}

	if memoryMB <= 0 {
		t.Errorf("Expected positive memory estimate, got %d MB", memoryMB)
	}

	// Test feasibility check
	feasible, reason, err := engine.CanMaterialize(1000) // 1GB limit
	if err != nil {
		t.Fatalf("Expected no error checking feasibility, got: %v", err)
	}

	if !feasible {
		t.Logf("Materialization not feasible: %s", reason)
	}

	t.Logf("Complexity: %d instances, Memory: %d MB, Feasible: %v", 
		complexity, memoryMB, feasible)
}

// ==========================
// PERFORMANCE TESTS
// ==========================

func TestMaterializationEngine_Performance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	// Create a larger graph for performance testing
	builder := NewTestDataBuilder()
	numAuthors := 500
	numPapers := 1000
	
	// Add authors
	for i := 0; i < numAuthors; i++ {
		builder.AddNode(fmt.Sprintf("a%d", i), "Author", nil)
	}
	
	// Add papers
	for i := 0; i < numPapers; i++ {
		builder.AddNode(fmt.Sprintf("p%d", i), "Paper", nil)
	}
	
	// Add random authorship relationships
	for i := 0; i < numPapers; i++ {
		// Each paper has 1-3 authors
		numAuthorsForPaper := 1 + (i % 3)
		for j := 0; j < numAuthorsForPaper; j++ {
			authorIdx := (i*3 + j) % numAuthors
			builder.AddEdge(fmt.Sprintf("a%d", authorIdx), fmt.Sprintf("p%d", i), "writes", 1.0)
		}
	}

	graph, metaPath := builder.
		SetMetaPath("perf-test", 
			[]string{"Author", "Paper", "Author"}, 
			[]string{"writes", "writes"}).
		Build()

	config := DefaultMaterializationConfig()
	config.Traversal.Parallelism = runtime.NumCPU()
	config.Traversal.MaxInstances = 50000 // Limit for reasonable test time

	start := time.Now()
	engine := NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Expected no error in performance test, got: %v", err)
	}

	if !result.Success {
		t.Fatalf("Expected successful performance test")
	}

	t.Logf("Performance test completed in %v", duration)
	t.Logf("Generated %d instances, %d edges", 
		result.Statistics.InstancesGenerated, 
		len(result.HomogeneousGraph.Edges))
	t.Logf("Memory usage: %d MB", result.Statistics.MemoryPeakMB)

	// Performance expectations (adjust based on your requirements)
	if duration > 30*time.Second {
		t.Errorf("Performance test took too long: %v", duration)
	}
}

// ==========================
// FILE I/O TESTS
// ==========================

func TestMaterializationEngine_FileIO(t *testing.T) {
	// Create a simple test graph
	graph, metaPath := NewTestDataBuilder().
		AddNode("a1", "Author", nil).
		AddNode("a2", "Author", nil).
		AddNode("p1", "Paper", nil).
		AddEdge("a1", "p1", "writes", 1.0).
		AddEdge("a2", "p1", "writes", 1.0).
		SetMetaPath("file-test", 
			[]string{"Author", "Paper", "Author"}, 
			[]string{"writes", "writes"}).
		Build()

	config := DefaultMaterializationConfig()
	engine := NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "graph_materialization_test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Test different file formats
	formats := []string{".csv", ".json", ".edgelist", ".txt"}
	
	for _, format := range formats {
		t.Run(fmt.Sprintf("format_%s", format), func(t *testing.T) {
			outputPath := filepath.Join(tempDir, "graph"+format)
			
			err := SaveHomogeneousGraph(result.HomogeneousGraph, outputPath)
			if err != nil {
				t.Fatalf("Failed to save graph as %s: %v", format, err)
			}

			// Verify file was created and has content
			stat, err := os.Stat(outputPath)
			if err != nil {
				t.Fatalf("Output file not created: %v", err)
			}

			if stat.Size() == 0 {
				t.Errorf("Output file is empty")
			}

			t.Logf("Saved %s file: %d bytes", format, stat.Size())
		})
	}

	// Test saving materialization result
	resultPath := filepath.Join(tempDir, "result.json")
	err = SaveMaterializationResult(result, resultPath)
	if err != nil {
		t.Fatalf("Failed to save materialization result: %v", err)
	}

	// Verify result file
	stat, err := os.Stat(resultPath)
	if err != nil {
		t.Fatalf("Result file not created: %v", err)
	}

	if stat.Size() == 0 {
		t.Errorf("Result file is empty")
	}
}

// ==========================
// UTILITY FUNCTIONS
// ==========================

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || 
		(len(s) > len(substr) && 
			(s[:len(substr)] == substr || 
			 s[len(s)-len(substr):] == substr || 
			 findSubstring(s, substr))))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// ==========================
// BENCHMARKS
// ==========================

func BenchmarkInstanceGenerator_FindInstances(b *testing.B) {
	// Create benchmark graph
	builder := NewTestDataBuilder()
	for i := 0; i < 100; i++ {
		builder.AddNode(fmt.Sprintf("a%d", i), "Author", nil)
		builder.AddNode(fmt.Sprintf("p%d", i), "Paper", nil)
		builder.AddEdge(fmt.Sprintf("a%d", i), fmt.Sprintf("p%d", i), "writes", 1.0)
	}

	graph, metaPath := builder.
		SetMetaPath("bench", []string{"Author", "Paper"}, []string{"writes"}).
		Build()

	config := DefaultTraversalConfig()
	generator := NewInstanceGenerator(graph, metaPath, config)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := generator.FindAllInstances(nil)
		if err != nil {
			b.Fatalf("Benchmark failed: %v", err)
		}
	}
}

func BenchmarkHomogeneousBuilder_Build(b *testing.B) {
	metaPath := &models.MetaPath{
		ID:           "bench",
		NodeSequence: []string{"Author", "Author"},
		EdgeSequence: []string{"collaborates"},
	}

	// Create test instances
	instances := make([]PathInstance, 1000)
	for i := 0; i < 1000; i++ {
		instances[i] = PathInstance{
			Nodes:  []string{fmt.Sprintf("a%d", i%100), fmt.Sprintf("a%d", (i+1)%100)},
			Edges:  []string{"collaborates"},
			Weight: float64(i % 10),
		}
	}

	config := DefaultAggregationConfig()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builder := NewHomogeneousBuilder(metaPath, config)
		for _, instance := range instances {
			builder.AddInstance(instance)
		}
		_, _ = builder.Build()
	}
}

func BenchmarkWeightCalculator_ProcessGraph(b *testing.B) {
	// Create benchmark graph
	graph := &HomogeneousGraph{
		NodeType: "Author",
		Nodes:    make(map[string]Node),
		Edges:    make(map[EdgeKey]float64),
	}

	// Add nodes and edges
	for i := 0; i < 100; i++ {
		nodeID := fmt.Sprintf("a%d", i)
		graph.Nodes[nodeID] = Node{ID: nodeID, Type: "Author"}
		
		for j := i + 1; j < 100; j++ {
			edge := EdgeKey{From: fmt.Sprintf("a%d", i), To: fmt.Sprintf("a%d", j)}
			graph.Edges[edge] = float64(i + j)
		}
	}

	config := DefaultAggregationConfig()
	config.Normalization = DegreeNorm
	calculator := NewWeightCalculator(config)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Create a copy for each iteration
		graphCopy := &HomogeneousGraph{
			NodeType: graph.NodeType,
			Nodes:    make(map[string]Node),
			Edges:    make(map[EdgeKey]float64),
		}
		for k, v := range graph.Nodes {
			graphCopy.Nodes[k] = v
		}
		for k, v := range graph.Edges {
			graphCopy.Edges[k] = v
		}

		err := calculator.ProcessGraph(graphCopy)
		if err != nil {
			b.Fatalf("Benchmark failed: %v", err)
		}
	}
}


func TestVerifyMaterialization(t *testing.T) {
	// Create a simple test graph (use your existing test data)
	graph, metaPath := NewTestDataBuilder().
		AddNode("a1", "Author", nil).
		AddNode("a2", "Author", nil).
		AddNode("p1", "Paper", nil).
		AddEdge("a1", "p1", "writes", 1.0).
		AddEdge("a2", "p1", "writes", 1.0).
		AddEdge("p1", "a1", "writtenBy", 1.0).
		AddEdge("p1", "a2", "writtenBy", 1.0).
		SetMetaPath("test", 
			[]string{"Author", "Paper", "Author"}, 
			[]string{"writes", "writtenBy"}).
		Build()

	// Verify the materialization
	verifier := NewGraphVerifier()
	verifier.LoadFromObjects(graph, metaPath)
	
	config := DefaultMaterializationConfig()
	result, err := verifier.VerifyMaterialization(config)

	if err != nil {
		t.Fatalf("Verification failed: %v", err)
	}

	// Print summary
	PrintVerificationSummary(result)

	// Check critical requirements
	if !result.Passed {
		t.Errorf("Verification failed - check output above")
	}

	// Check specific things you care about
	nodeCountTest := findTest(result.TestResults, "symmetric_node_count")
	if nodeCountTest != nil && !nodeCountTest.Passed {
		t.Errorf("Node count test failed: %s", nodeCountTest.Actual)
	}

	traversabilityTest := findTest(result.TestResults, "meta_path_traversable")
	if traversabilityTest != nil && !traversabilityTest.Passed {
		t.Errorf("Meta path not traversable: %s", traversabilityTest.ErrorMsg)
	}
}


func TestVerifyMaterializationWithRealData(t *testing.T) {
	// Load YOUR actual graph and meta path files
	// Replace these paths with your actual file paths
	verifier := NewGraphVerifier()
	
	// For testing, use your actual files:
	// err := verifier.LoadFromFiles("path/to/your/graph.json", "path/to/your/metapath.json")
	// if err != nil {
	//     t.Fatalf("Failed to load files: %v", err)
	// }
	
	// For now, creating test data that matches your scenario
	graph, metaPath := NewTestDataBuilder().
		AddNode("a1", "Author", nil).
		AddNode("a2", "Author", nil).
		AddNode("p1", "Paper", nil).
		AddEdge("a1", "p1", "writes", 1.0).
		AddEdge("a2", "p1", "writes", 1.0).
		AddEdge("p1", "a1", "writtenBy", 1.0).
		AddEdge("p1", "a2", "writtenBy", 1.0).
		SetMetaPath("test", 
			[]string{"Author", "Paper", "Author"}, 
			[]string{"writes", "writtenBy"}).
		Build()
	verifier.LoadFromObjects(graph, metaPath)
	
	config := DefaultMaterializationConfig()
	result, err := verifier.VerifyMaterialization(config)

	if err != nil {
		t.Fatalf("Verification failed: %v", err)
	}

	// Print summary
	PrintVerificationSummary(result)

	// INVESTIGATE MISSING NODES
	if result.GraphStats.OriginalGraph.NodeCount > result.GraphStats.MaterializedGraph.NodeCount {
		fmt.Printf("\n🔍 INVESTIGATING MISSING NODES...\n")
		investigation := verifier.InvestigateMissingNodes()
		PrintMissingNodesReport(investigation)
	}

	// Check only critical failures (warnings/info are OK)
	criticalFailures := 0
	for _, test := range result.TestResults {
		if !test.Passed && test.Severity == "critical" {
			criticalFailures++
			t.Errorf("CRITICAL FAILURE - %s: %s", test.Name, test.Description)
		}
	}

	if criticalFailures > 0 {
		t.Errorf("Found %d critical failures", criticalFailures)
	}

	// Log success for the important stuff
	nodeCountTest := findTest(result.TestResults, "symmetric_node_count") 
	if nodeCountTest != nil {
		t.Logf("✅ Node count verification: %s", nodeCountTest.Actual)
	}
	
	traversabilityTest := findTest(result.TestResults, "meta_path_traversable")
	if traversabilityTest != nil {
		t.Logf("✅ Meta path traversable: %t", traversabilityTest.Passed)
	}
	
	t.Logf("✅ Materialized %d nodes, %d edges", 
		result.GraphStats.MaterializedGraph.NodeCount,
		result.GraphStats.MaterializedGraph.EdgeCount)
}

// Helper function to find a specific test result
func findTest(results []TestResult, testName string) *TestResult {
	for _, result := range results {
		if result.Name == testName {
			return &result
		}
	}
	return nil
}
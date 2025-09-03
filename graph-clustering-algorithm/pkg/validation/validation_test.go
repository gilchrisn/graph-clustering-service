package validation

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// TestLoadAndValidateGraph tests the graph loading and validation
func TestLoadAndValidateGraph(t *testing.T) {
	// Create test data directory
	testDir := "../../testdata"
	os.MkdirAll(testDir, 0755)
	defer os.RemoveAll(testDir)

	// Create test graph file
	testGraph := models.HeterogeneousGraph{
		Nodes: map[string]models.Node{
			"a1": {Type: "Author", Properties: map[string]interface{}{"name": "Alice"}},
			"p1": {Type: "Paper", Properties: map[string]interface{}{"title": "Test Paper"}},
		},
		Edges: []models.Edge{
			{From: "a1", To: "p1", Type: "writes", Weight: 1.0},
		},
	}

	graphFile := filepath.Join(testDir, "test_graph.json")
	data, _ := json.Marshal(testGraph)
	os.WriteFile(graphFile, data, 0644)

	// Test loading
	graph, err := LoadAndValidateGraph(graphFile)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(graph.Nodes) != 2 {
		t.Errorf("Expected 2 nodes, got %d", len(graph.Nodes))
	}

	if len(graph.Edges) != 1 {
		t.Errorf("Expected 1 edge, got %d", len(graph.Edges))
	}
}

// TestValidateGraphStructure tests various graph validation scenarios
func TestValidateGraphStructure(t *testing.T) {
	tests := []struct {
		name        string
		graph       *models.HeterogeneousGraph
		expectError bool
		errorField  string
	}{
		{
			name: "valid graph",
			graph: &models.HeterogeneousGraph{
				Nodes: map[string]models.Node{
					"a1": {Type: "Author"},
					"p1": {Type: "Paper"},
				},
				Edges: []models.Edge{
					{From: "a1", To: "p1", Type: "writes", Weight: 1.0},
				},
			},
			expectError: false,
		},
		{
			name: "empty nodes",
			graph: &models.HeterogeneousGraph{
				Nodes: map[string]models.Node{},
				Edges: []models.Edge{},
			},
			expectError: true,
			errorField:  "nodes",
		},
		{
			name: "invalid edge reference",
			graph: &models.HeterogeneousGraph{
				Nodes: map[string]models.Node{
					"a1": {Type: "Author"},
				},
				Edges: []models.Edge{
					{From: "a1", To: "nonexistent", Type: "writes", Weight: 1.0},
				},
			},
			expectError: true,
			errorField:  "edge",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.graph.PopulateTypeMaps()
			err := ValidateGraphStructure(tt.graph)
			
			if tt.expectError && err == nil {
				t.Errorf("Expected error but got none")
			}
			
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

// TestMetaPathValidation tests meta path validation
func TestMetaPathValidation(t *testing.T) {
	tests := []struct {
		name        string
		metaPath    *models.MetaPath
		expectError bool
	}{
		{
			name: "valid meta path",
			metaPath: &models.MetaPath{
				ID:           "test",
				NodeSequence: []string{"Author", "Paper"},
				EdgeSequence: []string{"writes"},
				Description:  "Test path",
			},
			expectError: false,
		},
		{
			name: "empty ID",
			metaPath: &models.MetaPath{
				ID:           "",
				NodeSequence: []string{"Author", "Paper"},
				EdgeSequence: []string{"writes"},
			},
			expectError: true,
		},
		{
			name: "mismatched sequences",
			metaPath: &models.MetaPath{
				ID:           "test",
				NodeSequence: []string{"Author", "Paper"},
				EdgeSequence: []string{"writes", "extra"},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.metaPath.Validate()
			
			if tt.expectError && err == nil {
				t.Errorf("Expected error but got none")
			}
			
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

// TestMetaPathGraphCompatibility tests meta path compatibility with graph
func TestMetaPathGraphCompatibility(t *testing.T) {
	// Create test graph
	graph := &models.HeterogeneousGraph{
		Nodes: map[string]models.Node{
			"a1": {Type: "Author"},
			"p1": {Type: "Paper"},
		},
		Edges: []models.Edge{
			{From: "a1", To: "p1", Type: "writes", Weight: 1.0},
		},
	}
	graph.PopulateTypeMaps()

	tests := []struct {
		name        string
		metaPath    *models.MetaPath
		expectError bool
	}{
		{
			name: "compatible meta path",
			metaPath: &models.MetaPath{
				ID:           "test",
				NodeSequence: []string{"Author", "Paper"},
				EdgeSequence: []string{"writes"},
			},
			expectError: false,
		},
		{
			name: "incompatible node type",
			metaPath: &models.MetaPath{
				ID:           "test",
				NodeSequence: []string{"Venue", "Paper"},
				EdgeSequence: []string{"writes"},
			},
			expectError: true,
		},
		{
			name: "incompatible edge type",
			metaPath: &models.MetaPath{
				ID:           "test",
				NodeSequence: []string{"Author", "Paper"},
				EdgeSequence: []string{"cites"},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateMetaPathAgainstGraph(tt.metaPath, graph)
			
			if tt.expectError && err == nil {
				t.Errorf("Expected error but got none")
			}
			
			if !tt.expectError && err != nil {
				t.Errorf("Expected no error but got: %v", err)
			}
		})
	}
}

// BenchmarkGraphValidation benchmarks the graph validation performance
func BenchmarkGraphValidation(b *testing.B) {
	// Create a larger test graph
	graph := &models.HeterogeneousGraph{
		Nodes: make(map[string]models.Node),
		Edges: make([]models.Edge, 0),
	}

	// Add 1000 nodes
	for i := 0; i < 1000; i++ {
		nodeID := fmt.Sprintf("n%d", i)
		nodeType := "Type" + fmt.Sprintf("%d", i%3) // 3 different types
		graph.Nodes[nodeID] = models.Node{
			Type:       nodeType,
			Properties: map[string]interface{}{"id": i},
		}
	}

	// Add 2000 edges
	for i := 0; i < 2000; i++ {
		from := fmt.Sprintf("n%d", i%1000)
		to := fmt.Sprintf("n%d", (i+1)%1000)
		graph.Edges = append(graph.Edges, models.Edge{
			From:   from,
			To:     to,
			Type:   "edge_type",
			Weight: 1.0,
		})
	}

	graph.PopulateTypeMaps()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ValidateGraphStructure(graph)
	}
}
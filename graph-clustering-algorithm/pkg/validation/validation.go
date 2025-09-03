package validation

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// LoadAndValidateGraph loads a graph from JSON file and validates its structure
func LoadAndValidateGraph(filePath string) (*models.HeterogeneousGraph, error) {
	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("graph file does not exist: %s", filePath)
	}

	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read graph file: %w", err)
	}

	// Parse JSON
	var graph models.HeterogeneousGraph
	if err := json.Unmarshal(data, &graph); err != nil {
		return nil, fmt.Errorf("failed to parse graph JSON: %w", err)
	}

	// Populate computed fields
	graph.PopulateTypeMaps()

	// Validate graph structure
	if err := ValidateGraphStructure(&graph); err != nil {
		return nil, fmt.Errorf("graph validation failed: %w", err)
	}

	return &graph, nil
}

// LoadAndValidateMetaPath loads a meta path from JSON file and validates it
func LoadAndValidateMetaPath(filePath string) (*models.MetaPath, error) {
	// Check if file exists
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("meta path file does not exist: %s", filePath)
	}

	// Read file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read meta path file: %w", err)
	}

	// Parse JSON
	var metaPath models.MetaPath
	if err := json.Unmarshal(data, &metaPath); err != nil {
		return nil, fmt.Errorf("failed to parse meta path JSON: %w", err)
	}

	// Validate meta path
	if err := metaPath.Validate(); err != nil {
		return nil, fmt.Errorf("meta path validation failed: %w", err)
	}

	return &metaPath, nil
}

// ValidateGraphStructure performs comprehensive validation of graph structure
func ValidateGraphStructure(graph *models.HeterogeneousGraph) error {
	var errors models.ValidationErrors

	// Check basic structure
	if graph.Nodes == nil {
		errors = append(errors, models.ValidationError{
			Field:   "nodes",
			Message: "nodes map cannot be nil",
		})
	}

	if graph.Edges == nil {
		errors = append(errors, models.ValidationError{
			Field:   "edges",
			Message: "edges array cannot be nil",
		})
	}

	if len(errors) > 0 {
		return errors
	}

	// Validate nodes
	if err := validateNodes(graph.Nodes); err != nil {
		if ve, ok := err.(models.ValidationErrors); ok {
			errors = append(errors, ve...)
		} else {
			errors = append(errors, models.ValidationError{
				Field:   "nodes",
				Message: err.Error(),
			})
		}
	}

	// Validate edges
	if err := validateEdges(graph.Edges, graph.Nodes); err != nil {
		if ve, ok := err.(models.ValidationErrors); ok {
			errors = append(errors, ve...)
		} else {
			errors = append(errors, models.ValidationError{
				Field:   "edges",
				Message: err.Error(),
			})
		}
	}

	// Check graph connectivity
	if err := validateGraphConnectivity(graph); err != nil {
		errors = append(errors, models.ValidationError{
			Field:   "connectivity",
			Message: err.Error(),
		})
	}

	if len(errors) > 0 {
		return errors
	}

	return nil
}

// validateNodes checks all nodes for consistency and required fields
func validateNodes(nodes map[string]models.Node) error {
	var errors models.ValidationErrors

	if len(nodes) == 0 {
		return models.ValidationError{
			Field:   "nodes",
			Message: "graph must contain at least one node",
		}
	}

	nodeTypes := make(map[string]bool)
	for nodeID, node := range nodes {
		// Validate node ID
		if strings.TrimSpace(nodeID) == "" {
			errors = append(errors, models.ValidationError{
				Field:   "node.id",
				Message: "node ID cannot be empty or whitespace",
				Value:   nodeID,
			})
			continue
		}

		// Validate node type
		if strings.TrimSpace(node.Type) == "" {
			errors = append(errors, models.ValidationError{
				Field:   "node.type",
				Message: "node type cannot be empty",
				Value:   nodeID,
			})
		} else {
			nodeTypes[node.Type] = true
		}

		// Validate properties (optional but should be a valid map)
		if node.Properties == nil {
			// Initialize empty properties if nil
			node.Properties = make(map[string]interface{})
		}

		// Set ID if not present in properties (for consistency)
		if node.ID == "" {
			node.ID = nodeID
		}
	}

	// Check if we have at least 2 different node types (for heterogeneous graph)
	if len(nodeTypes) < 2 {
		errors = append(errors, models.ValidationError{
			Field:   "node_types",
			Message: fmt.Sprintf("heterogeneous graph should have at least 2 node types, found: %d", len(nodeTypes)),
		})
	}

	if len(errors) > 0 {
		return errors
	}
	return nil
}

// validateEdges checks all edges for consistency and valid references
func validateEdges(edges []models.Edge, nodes map[string]models.Node) error {
	var errors models.ValidationErrors

	if len(edges) == 0 {
		return models.ValidationError{
			Field:   "edges",
			Message: "graph must contain at least one edge",
		}
	}

	edgeTypes := make(map[string]bool)
	seenEdges := make(map[string]bool) // To detect duplicates

	for i, edge := range edges {
		fieldPrefix := fmt.Sprintf("edge[%d]", i)

		// Validate from node
		if strings.TrimSpace(edge.From) == "" {
			errors = append(errors, models.ValidationError{
				Field:   fieldPrefix + ".from",
				Message: "from node ID cannot be empty",
			})
		} else if _, exists := nodes[edge.From]; !exists {
			errors = append(errors, models.ValidationError{
				Field:   fieldPrefix + ".from",
				Message: "from node does not exist in graph",
				Value:   edge.From,
			})
		}

		// Validate to node
		if strings.TrimSpace(edge.To) == "" {
			errors = append(errors, models.ValidationError{
				Field:   fieldPrefix + ".to",
				Message: "to node ID cannot be empty",
			})
		} else if _, exists := nodes[edge.To]; !exists {
			errors = append(errors, models.ValidationError{
				Field:   fieldPrefix + ".to",
				Message: "to node does not exist in graph",
				Value:   edge.To,
			})
		}

		// Validate edge type
		if strings.TrimSpace(edge.Type) == "" {
			errors = append(errors, models.ValidationError{
				Field:   fieldPrefix + ".type",
				Message: "edge type cannot be empty",
			})
		} else {
			edgeTypes[edge.Type] = true
		}

		// Validate weight (set default if zero)
		if edge.Weight == 0 {
			edge.Weight = 1.0 // Default weight
		} else if edge.Weight < 0 {
			errors = append(errors, models.ValidationError{
				Field:   fieldPrefix + ".weight",
				Message: "edge weight cannot be negative",
				Value:   fmt.Sprintf("%.2f", edge.Weight),
			})
		}

		// Check for self-loops
		if edge.From == edge.To {
			errors = append(errors, models.ValidationError{
				Field:   fieldPrefix,
				Message: "self-loops are not allowed",
				Value:   fmt.Sprintf("%s -> %s", edge.From, edge.To),
			})
		}

		// Check for duplicate edges (same from, to, type combination)
		edgeKey := fmt.Sprintf("%s|%s|%s", edge.From, edge.To, edge.Type)
		if seenEdges[edgeKey] {
			errors = append(errors, models.ValidationError{
				Field:   fieldPrefix,
				Message: "duplicate edge detected",
				Value:   edgeKey,
			})
		}
		seenEdges[edgeKey] = true
	}

	if len(errors) > 0 {
		return errors
	}
	return nil
}

// validateGraphConnectivity performs basic connectivity checks
func validateGraphConnectivity(graph *models.HeterogeneousGraph) error {
	// Find isolated nodes (nodes with no edges)
	nodeConnections := make(map[string]int)
	for nodeID := range graph.Nodes {
		nodeConnections[nodeID] = 0
	}

	for _, edge := range graph.Edges {
		nodeConnections[edge.From]++
		nodeConnections[edge.To]++
	}

	isolatedNodes := make([]string, 0)
	for nodeID, connections := range nodeConnections {
		if connections == 0 {
			isolatedNodes = append(isolatedNodes, nodeID)
		}
	}

	if len(isolatedNodes) > 0 {
		return fmt.Errorf("found %d isolated nodes (no connections): %v", 
			len(isolatedNodes), isolatedNodes)
	}

	return nil
}

// ValidateMetaPathAgainstGraph checks if meta path is compatible with graph structure
func ValidateMetaPathAgainstGraph(metaPath *models.MetaPath, graph *models.HeterogeneousGraph) error {
	var errors models.ValidationErrors

	// Check if all node types in meta path exist in graph
	graphNodeTypes := make(map[string]bool)
	for _, node := range graph.Nodes {
		graphNodeTypes[node.Type] = true
	}

	for i, nodeType := range metaPath.NodeSequence {
		if !graphNodeTypes[nodeType] {
			errors = append(errors, models.ValidationError{
				Field:   fmt.Sprintf("node_sequence[%d]", i),
				Message: "node type does not exist in graph",
				Value:   nodeType,
			})
		}
	}

	// Check if all edge types in meta path exist in graph
	graphEdgeTypes := make(map[string]bool)
	for _, edge := range graph.Edges {
		graphEdgeTypes[edge.Type] = true
	}

	for i, edgeType := range metaPath.EdgeSequence {
		if !graphEdgeTypes[edgeType] {
			errors = append(errors, models.ValidationError{
				Field:   fmt.Sprintf("edge_sequence[%d]", i),
				Message: "edge type does not exist in graph",
				Value:   edgeType,
			})
		}
	}

	// Check if meta path transitions are possible in the graph
	if err := validateMetaPathTransitions(metaPath, graph); err != nil {
		errors = append(errors, models.ValidationError{
			Field:   "meta_path_transitions",
			Message: err.Error(),
		})
	}

	if len(errors) > 0 {
		return errors
	}

	return nil
}

// validateMetaPathTransitions checks if each step in meta path is valid
func validateMetaPathTransitions(metaPath *models.MetaPath, graph *models.HeterogeneousGraph) error {
	for step := 0; step < len(metaPath.EdgeSequence); step++ {
		fromNodeType := metaPath.NodeSequence[step]
		toNodeType := metaPath.NodeSequence[step+1]
		requiredEdgeType := metaPath.EdgeSequence[step]

		// Check if this transition exists in the graph
		transitionExists := false
		for _, edge := range graph.Edges {
			fromNode, fromExists := graph.Nodes[edge.From]
			toNode, toExists := graph.Nodes[edge.To]

			if fromExists && toExists &&
				fromNode.Type == fromNodeType &&
				toNode.Type == toNodeType &&
				edge.Type == requiredEdgeType {
				transitionExists = true
				break
			}
		}

		if !transitionExists {
			return fmt.Errorf("transition %s -[%s]-> %s not found in graph at step %d",
				fromNodeType, requiredEdgeType, toNodeType, step)
		}
	}

	return nil
}


// ValidateFileFormat checks if files have correct extensions and are readable
func ValidateFileFormat(filePath string) error {
	// Check extension
	ext := strings.ToLower(filepath.Ext(filePath))
	if ext != ".json" {
		return fmt.Errorf("file must have .json extension, got: %s", ext)
	}

	// Check if file is readable
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("cannot open file: %w", err)
	}
	defer file.Close()

	// Try to parse as JSON
	decoder := json.NewDecoder(file)
	var temp interface{}
	if err := decoder.Decode(&temp); err != nil {
		return fmt.Errorf("file is not valid JSON: %w", err)
	}

	return nil
}

// ValidateOutputDirectory checks if output directory exists or can be created
func ValidateOutputDirectory(outputDir string) error {
	// Check if directory exists
	info, err := os.Stat(outputDir)
	if os.IsNotExist(err) {
		// Try to create it
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			return fmt.Errorf("cannot create output directory: %w", err)
		}
		return nil
	}

	if err != nil {
		return fmt.Errorf("cannot access output directory: %w", err)
	}

	if !info.IsDir() {
		return fmt.Errorf("output path exists but is not a directory: %s", outputDir)
	}

	// Check if directory is writable
	testFile := filepath.Join(outputDir, ".write_test")
	if err := os.WriteFile(testFile, []byte("test"), 0644); err != nil {
		return fmt.Errorf("output directory is not writable: %w", err)
	}
	os.Remove(testFile) // Clean up

	return nil
}
package materialization

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// SaveHomogeneousGraph saves a homogeneous graph to file
// Format is determined by file extension: .csv, .json, .edgelist, .txt
func SaveHomogeneousGraph(graph *HomogeneousGraph, outputPath string) error {
	if graph == nil {
		return fmt.Errorf("graph cannot be nil")
	}
	
	// Create directory if it doesn't exist
	dir := filepath.Dir(outputPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}
	
	// Determine format by extension
	ext := strings.ToLower(filepath.Ext(outputPath))
	
	switch ext {
	case ".csv":
		return SaveAsCSV(graph, outputPath)
	case ".json":
		return SaveAsJSON(graph, outputPath)
	case ".edgelist", ".txt", "":
		return SaveAsEdgeList(graph, outputPath)
	default:
		return SaveAsEdgeList(graph, outputPath) // Default to edge list
	}
}

// SaveAsEdgeList saves in standard graph format: first line = "nodes edges", then edge list
func SaveAsEdgeList(graph *HomogeneousGraph, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()
	
	// Write header: number_of_nodes number_of_edges
	_, err = fmt.Fprintf(file, "%d %d\n", len(graph.Nodes), len(graph.Edges))
	if err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}
	
	// Write edges: from to weight
	for edgeKey, weight := range graph.Edges {
		_, err = fmt.Fprintf(file, "%s %s %.6f\n", edgeKey.From, edgeKey.To, weight)
		if err != nil {
			return fmt.Errorf("failed to write edge: %w", err)
		}
	}
	
	return nil
}

// SaveAsCSV saves as CSV with header: from,to,weight
func SaveAsCSV(graph *HomogeneousGraph, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()
	
	writer := csv.NewWriter(file)
	defer writer.Flush()
	
	// Write header
	if err := writer.Write([]string{"from", "to", "weight"}); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}
	
	// Write edges
	for edgeKey, weight := range graph.Edges {
		record := []string{
			edgeKey.From,
			edgeKey.To,
			strconv.FormatFloat(weight, 'f', 6, 64),
		}
		if err := writer.Write(record); err != nil {
			return fmt.Errorf("failed to write CSV record: %w", err)
		}
	}
	
	return nil
}

// SaveAsJSON saves the complete graph with metadata as JSON
// SaveAsJSON saves the complete graph with metadata as JSON
func SaveAsJSON(graph *HomogeneousGraph, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	// Convert to JSON-friendly format
	type JSONEdge struct {
		From   string  `json:"from"`
		To     string  `json:"to"`
		Weight float64 `json:"weight"`
	}

	type JSONGraph struct {
		NodeType   string                 `json:"node_type"`
		Nodes      map[string]Node        `json:"nodes"`
		Edges      []JSONEdge             `json:"edges"`      // Convert map to slice
		Statistics GraphStatistics        `json:"statistics"`
		MetaPath   models.MetaPath        `json:"meta_path"`
	}

	// Convert edges map to slice
	var jsonEdges []JSONEdge
	for edgeKey, weight := range graph.Edges {
		jsonEdges = append(jsonEdges, JSONEdge{
			From:   edgeKey.From,
			To:     edgeKey.To,
			Weight: weight,
		})
	}

	jsonGraph := JSONGraph{
		NodeType:   graph.NodeType,
		Nodes:      graph.Nodes,
		Edges:      jsonEdges,
		Statistics: graph.Statistics,
		MetaPath:   graph.MetaPath,
	}

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	if err := encoder.Encode(jsonGraph); err != nil {
		return fmt.Errorf("failed to encode JSON: %w", err)
	}

	return nil
}

// SaveMaterializationResult saves the complete materialization result
// SaveMaterializationResult saves the complete materialization result
func SaveMaterializationResult(result *MaterializationResult, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	// Convert edges to JSON-friendly format
	type JSONEdge struct {
		From   string  `json:"from"`
		To     string  `json:"to"`
		Weight float64 `json:"weight"`
	}

	type JSONResult struct {
		HomogeneousGraph *struct {
			NodeType   string                 `json:"node_type"`
			Nodes      map[string]Node        `json:"nodes"`
			Edges      []JSONEdge             `json:"edges"`
			Statistics GraphStatistics        `json:"statistics"`
			MetaPath   models.MetaPath        `json:"meta_path"`
		} `json:"homogeneous_graph"`
		Statistics ProcessingStatistics  `json:"statistics"`
		Config     MaterializationConfig `json:"config"`
		Success    bool                  `json:"success"`
		Error      string                `json:"error,omitempty"`
	}

	jsonResult := &JSONResult{
		Statistics: result.Statistics,
		Config:     result.Config,
		Success:    result.Success,
		Error:      result.Error,
	}

	// Convert homogeneous graph if it exists
	if result.HomogeneousGraph != nil {
		var jsonEdges []JSONEdge
		for edgeKey, weight := range result.HomogeneousGraph.Edges {
			jsonEdges = append(jsonEdges, JSONEdge{
				From:   edgeKey.From,
				To:     edgeKey.To,
				Weight: weight,
			})
		}

		jsonResult.HomogeneousGraph = &struct {
			NodeType   string                 `json:"node_type"`
			Nodes      map[string]Node        `json:"nodes"`
			Edges      []JSONEdge             `json:"edges"`
			Statistics GraphStatistics        `json:"statistics"`
			MetaPath   models.MetaPath        `json:"meta_path"`
		}{
			NodeType:   result.HomogeneousGraph.NodeType,
			Nodes:      result.HomogeneousGraph.Nodes,
			Edges:      jsonEdges,
			Statistics: result.HomogeneousGraph.Statistics,
			MetaPath:   result.HomogeneousGraph.MetaPath,
		}
	}

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	if err := encoder.Encode(jsonResult); err != nil {
		return fmt.Errorf("failed to encode JSON: %w", err)
	}

	return nil
}


func SaveAsSimpleEdgeList(graph *HomogeneousGraph, outputPath string) error {
	if graph == nil {
		return fmt.Errorf("graph cannot be nil")
	}
	
	// Create directory if it doesn't exist
	dir := filepath.Dir(outputPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}
	
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()
	
	// Write edges: from to weight (or just from to if weight is 1.0)
	for edgeKey, weight := range graph.Edges {
		if weight == 1.0 {
			// Skip weight if it's 1.0 (SCAR format allows this)
			_, err = fmt.Fprintf(file, "%s %s\n", edgeKey.From, edgeKey.To)
		} else {
			_, err = fmt.Fprintf(file, "%s %s %.6f\n", edgeKey.From, edgeKey.To, weight)
		}
		if err != nil {
			return fmt.Errorf("failed to write edge: %w", err)
		}
	}
	
	return nil
}

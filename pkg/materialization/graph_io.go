package materialization

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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
func SaveAsJSON(graph *HomogeneousGraph, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	
	if err := encoder.Encode(graph); err != nil {
		return fmt.Errorf("failed to encode JSON: %w", err)
	}
	
	return nil
}

// SaveMaterializationResult saves the complete materialization result
func SaveMaterializationResult(result *MaterializationResult, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	
	if err := encoder.Encode(result); err != nil {
		return fmt.Errorf("failed to encode JSON: %w", err)
	}
	
	return nil
}
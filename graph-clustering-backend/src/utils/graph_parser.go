// utils/graph_parser.go
package utils

import (
	"fmt"
	"log"
	"strconv"
	"strings"
)

// EdgeData represents an edge in the graph
type EdgeData struct {
	Source string  `json:"source"`
	Target string  `json:"target"`
	Weight float64 `json:"weight"`
}

// HierarchyEntry represents a supernode in the hierarchy
type HierarchyEntry struct {
	ID         string   `json:"id"`
	ChildCount int      `json:"childCount"`
	Children   []string `json:"children"`
}

// MappingEntry represents a mapping from supernode to leaf nodes
type MappingEntry struct {
	ID        string `json:"id"`
	LeafCount int    `json:"leafCount"`
	Leaves    []int  `json:"leaves"`
}

// ParseHierarchyFile parses a hierarchy file - corrected to handle multiple supernodes
// Returns parsed hierarchy data or error if invalid
func ParseHierarchyFile(content string) (map[string]interface{}, error) {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	lineIndex := 0
	hierarchies := make(map[string]interface{})
	
	// Process file line by line to extract multiple supernodes
	for lineIndex < len(lines) {
		// Each section starts with a supernode ID
		if lineIndex >= len(lines) {
			break
		}
		id := strings.TrimSpace(lines[lineIndex])
		lineIndex++

		// parse out parent id and level
		parts := strings.Split(id, "_")
		if len(parts) < 2 {
			log.Printf("Invalid supernode ID format: %s", id)
			continue
		}
		
		parentCommunityID := parts[0]
		// strip the "l" and convert to number
		levelStr := parts[1]
		if !strings.HasPrefix(levelStr, "l") {
			log.Printf("Invalid level format in ID: %s", id)
			continue
		}
		parentLevel, err := strconv.Atoi(levelStr[1:])
		if err != nil {
			log.Printf("Invalid level number in ID: %s", id)
			continue
		}
		
		// Next line should be the child count
		if lineIndex >= len(lines) {
			break
		}
		childCountStr := strings.TrimSpace(lines[lineIndex])
		lineIndex++
		
		childCount, err := strconv.Atoi(childCountStr)
		if err != nil || childCount <= 0 {
			// Error
			log.Printf("Invalid child count for supernode %s: %s", id, childCountStr)
			// Skip invalid entries
			continue
		}

		// Collect child IDs
		children := make([]string, 0, childCount)
		for i := 0; i < childCount && lineIndex < len(lines); i++ {
			childIDStr := strings.TrimSpace(lines[lineIndex])
			lineIndex++
			childKey := fmt.Sprintf("%s_l%d_%s", parentCommunityID, parentLevel-1, childIDStr)
			children = append(children, childKey)
		}
		
		// Store this supernode data
		hierarchies[id] = map[string]interface{}{
			"id":         id,
			"childCount": childCount,
			"children":   children,
		}
	}
	
	return hierarchies, nil
}

// ParseEdgeFile parses edge file content
// Returns array of edge objects
func ParseEdgeFile(content string) ([]EdgeData, error) {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	var edges []EdgeData
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			source := parts[0]
			target := parts[1]
			weight := 1.0
			
			if len(parts) > 2 {
				if w, err := strconv.ParseFloat(parts[2], 64); err == nil {
					weight = w
				}
			}
			
			edges = append(edges, EdgeData{
				Source: source,
				Target: target,
				Weight: weight,
			})
		}
	}
	
	return edges, nil
}

// ParseMappingFile parses a mapping file - corrected to handle multiple supernodes
// Returns parsed mapping data or error if invalid
func ParseMappingFile(content string) (map[string]interface{}, error) {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	lineIndex := 0
	mappings := make(map[string]interface{})
	
	// Process file line by line to extract multiple supernode mappings
	for lineIndex < len(lines) {
		// Each section starts with a supernode ID
		if lineIndex >= len(lines) {
			break
		}
		id := strings.TrimSpace(lines[lineIndex])
		lineIndex++
		
		// Next line should be the leaf count
		if lineIndex >= len(lines) {
			break
		}
		leafCountStr := strings.TrimSpace(lines[lineIndex])
		lineIndex++
		
		leafCount, err := strconv.Atoi(leafCountStr)
		if err != nil || leafCount <= 0 {
			// Error
			log.Printf("Invalid leaf count for supernode %s: %s", id, leafCountStr)
			// Skip invalid entries
			continue
		}
		
		// Collect leaf node IDs
		leaves := make([]int, 0, leafCount)
		for i := 0; i < leafCount && lineIndex < len(lines); i++ {
			leafIDStr := strings.TrimSpace(lines[lineIndex])
			lineIndex++
			
			leafID, err := strconv.Atoi(leafIDStr)
			if err == nil {
				leaves = append(leaves, leafID)
			}
		}
		
		// Store this mapping data
		mappings[id] = map[string]interface{}{
			"id":        id,
			"leafCount": leafCount,
			"leaves":    leaves,
		}
	}
	
	return mappings, nil
}
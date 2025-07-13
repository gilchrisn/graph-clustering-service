package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ChildNode represents a child node with its visualization data
type ChildNode struct {
	ID       string  `json:"id"`
	PageRank float64 `json:"pagerank"`
	X        float64 `json:"x"`
	Y        float64 `json:"y"`
	Radius   float64 `json:"radius"`
	Label    string  `json:"label"`
	Level    int     `json:"level"`
}

// NodeViz matches the structure from your pipeline
type NodeViz struct {
	ID       string  `json:"id"`
	PageRank float64 `json:"pagerank"`
	X        float64 `json:"x"`
	Y        float64 `json:"y"`
	Radius   float64 `json:"radius"`
	Label    string  `json:"label"`
}

// LevelViz matches the structure from your pipeline
type LevelViz struct {
	Level int       `json:"level"`
	Nodes []NodeViz `json:"nodes"`
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	supernodeID := os.Args[1]
	
	// Default paths - you can make these configurable later
	baseDir := "results"
	algorithm := "materialization" // or "scar"
	
	// Allow optional algorithm parameter
	if len(os.Args) >= 3 {
		algorithm = os.Args[2]
	}

	fmt.Printf("🔍 Querying supernode: %s (algorithm: %s)\n", supernodeID, algorithm)
	fmt.Printf("📂 Looking in: %s\n\n", baseDir)

	// Load data and query
	children, err := getImmediateChildren(baseDir, algorithm, supernodeID)
	if err != nil {
		fmt.Printf("❌ Error: %v\n", err)
		os.Exit(1)
	}

	// Display results
	fmt.Printf("✅ Found %d immediate children for supernode %s:\n\n", len(children), supernodeID)
	
	for i, child := range children {
		fmt.Printf("Child %d:\n", i+1)
		fmt.Printf("  ID:       %s\n", child.ID)
		fmt.Printf("  Level:    %d\n", child.Level)
		fmt.Printf("  Position: (%.3f, %.3f)\n", child.X, child.Y)
		fmt.Printf("  Radius:   %.3f\n", child.Radius)
		fmt.Printf("  PageRank: %.6f\n", child.PageRank)
		fmt.Printf("  Label:    %s\n", child.Label)
		fmt.Printf("\n")
	}
}

func printUsage() {
	fmt.Printf("Hierarchy Query Tool\n")
	fmt.Printf("====================\n\n")
	fmt.Printf("Usage: %s <supernode_id> [algorithm]\n\n", os.Args[0])
	fmt.Printf("Arguments:\n")
	fmt.Printf("  supernode_id  - ID of the supernode to query (e.g., c0_l2_0)\n")
	fmt.Printf("  algorithm     - Algorithm type: 'materialization' or 'scar' (default: materialization)\n\n")
	fmt.Printf("Examples:\n")
	fmt.Printf("  %s c0_l2_0\n", os.Args[0])
	fmt.Printf("  %s c0_l1_5 scar\n", os.Args[0])
	fmt.Printf("\n")
	fmt.Printf("Expected directory structure:\n")
	fmt.Printf("  results/\n")
	fmt.Printf("    materialization/\n")
	fmt.Printf("      clustering/\n")
	fmt.Printf("        communities.hierarchy\n")
	fmt.Printf("      visualization/\n")
	fmt.Printf("        levels.json\n")
	fmt.Printf("    scar/\n")
	fmt.Printf("      clustering/\n")
	fmt.Printf("        communities_hierarchy.dat\n")
	fmt.Printf("      visualization/\n")
	fmt.Printf("        levels.json\n")
}

// getImmediateChildren returns immediate children of a supernode with their coordinates and radius
func getImmediateChildren(baseDir, algorithm, supernodeID string) ([]ChildNode, error) {
	// Load visualization data
	vizData, err := loadVisualizationData(baseDir, algorithm)
	if err != nil {
		return nil, fmt.Errorf("failed to load visualization data: %w", err)
	}

	// Determine the level of the supernode
	supernodeLevel := extractLevel(supernodeID)
	if supernodeLevel == 0 {
		return nil, fmt.Errorf("node %s is at leaf level (0), has no children", supernodeID)
	}

	// Child level is one level below
	childLevel := supernodeLevel - 1

	var children []string

	if supernodeLevel == 1 {
		// Level 1 supernodes contain leaf nodes - look in mapping file
		mapping, err := loadMappingData(baseDir, algorithm)
		if err != nil {
			return nil, fmt.Errorf("failed to load mapping data: %w", err)
		}

		leafNodes, exists := mapping[supernodeID]
		if !exists {
			return nil, fmt.Errorf("supernode %s not found in mapping", supernodeID)
		}
		children = leafNodes
	} else {
		// Level 2+ supernodes have supernode children - look in hierarchy file
		hierarchy, err := loadHierarchyData(baseDir, algorithm)
		if err != nil {
			return nil, fmt.Errorf("failed to load hierarchy data: %w", err)
		}

		supernodeChildren, exists := hierarchy[supernodeID]
		if !exists {
			return nil, fmt.Errorf("supernode %s not found in hierarchy", supernodeID)
		}
		children = supernodeChildren
	}

	// Get visualization data for child level
	levelVizData, exists := vizData[childLevel]
	if !exists {
		return nil, fmt.Errorf("no visualization data found for level %d", childLevel)
	}

	// Build result
	var result []ChildNode
	for _, rawChildID := range children {
		// Construct proper child ID based on level
		var lookupID string
		if childLevel == 0 {
			// Leaf nodes use their true ID (no prefix)
			lookupID = rawChildID
		} else {
			// Supernodes use community ID format: c0_l{level}_{id}
			lookupID = fmt.Sprintf("c0_l%d_%s", childLevel, rawChildID)
		}

		if vizNode, exists := levelVizData[lookupID]; exists {
			child := ChildNode{
				ID:       vizNode.ID,
				PageRank: vizNode.PageRank,
				X:        vizNode.X,
				Y:        vizNode.Y,
				Radius:   vizNode.Radius,
				Label:    vizNode.Label,
				Level:    childLevel,
			}
			result = append(result, child)
		} else {
			fmt.Printf("⚠️  Warning: child %s (lookup: %s) not found in visualization data for level %d\n", rawChildID, lookupID, childLevel)
		}
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("no children found with visualization data for supernode %s", supernodeID)
	}

	return result, nil
}

// loadVisualizationData loads the levels.json file
func loadVisualizationData(baseDir, algorithm string) (map[int]map[string]NodeViz, error) {
	vizPath := filepath.Join(baseDir, algorithm, "visualization", "levels.json")

	file, err := os.Open(vizPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var levels []LevelViz
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&levels)
	if err != nil {
		return nil, err
	}

	// Index by level and node ID for fast lookup
	vizData := make(map[int]map[string]NodeViz)
	for _, level := range levels {
		vizData[level.Level] = make(map[string]NodeViz)
		for _, node := range level.Nodes {
			vizData[level.Level][node.ID] = node
		}
	}

	return vizData, nil
}

// loadMappingData loads the mapping structure
func loadMappingData(baseDir, algorithm string) (map[string][]string, error) {
	clusteringDir := filepath.Join(baseDir, algorithm, "clustering")

	// Load mapping file
	var mappingFile string
	if algorithm == "materialization" {
		mappingFile = filepath.Join(clusteringDir, "communities.mapping")
	} else {
		mappingFile = filepath.Join(clusteringDir, "communities_mapping.dat")
	}

	mapping, err := parseMappingFile(mappingFile)
	if err != nil {
		return nil, err
	}

	return mapping, nil
}

// loadHierarchyData loads the hierarchy structure
func loadHierarchyData(baseDir, algorithm string) (map[string][]string, error) {
	clusteringDir := filepath.Join(baseDir, algorithm, "clustering")

	// Load hierarchy file
	var hierarchyFile string
	if algorithm == "materialization" {
		hierarchyFile = filepath.Join(clusteringDir, "communities.hierarchy")
	} else {
		hierarchyFile = filepath.Join(clusteringDir, "communities_hierarchy.dat")
	}

	hierarchy, err := parseHierarchyFile(hierarchyFile)
	if err != nil {
		return nil, err
	}

	return hierarchy, nil
}

// parseHierarchyFile parses the Louvain hierarchy file
func parseHierarchyFile(filename string) (map[string][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	hierarchy := make(map[string][]string)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		// Read parent ID
		parentID := strings.TrimSpace(scanner.Text())
		if parentID == "" {
			continue
		}

		// Read child count
		if !scanner.Scan() {
			break
		}
		countStr := strings.TrimSpace(scanner.Text())
		count, err := strconv.Atoi(countStr)
		if err != nil {
			return nil, fmt.Errorf("invalid child count for parent %s: %s", parentID, countStr)
		}

		// Read children
		children := make([]string, 0, count)
		for i := 0; i < count && scanner.Scan(); i++ {
			child := strings.TrimSpace(scanner.Text())
			children = append(children, child)
		}

		if len(children) != count {
			return nil, fmt.Errorf("expected %d children for parent %s, got %d", count, parentID, len(children))
		}

		hierarchy[parentID] = children
	}

	return hierarchy, scanner.Err()
}

// parseMappingFile parses the Louvain mapping file
// Format: community_id\nnode_count\nnode1\nnode2\n...
func parseMappingFile(filename string) (map[string][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	mapping := make(map[string][]string)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		// Read community ID
		communityID := strings.TrimSpace(scanner.Text())
		if communityID == "" {
			continue
		}

		// Read node count
		if !scanner.Scan() {
			break
		}
		countStr := strings.TrimSpace(scanner.Text())
		count, err := strconv.Atoi(countStr)
		if err != nil {
			return nil, fmt.Errorf("invalid node count for community %s: %s", communityID, countStr)
		}

		// Read nodes
		nodes := make([]string, 0, count)
		for i := 0; i < count && scanner.Scan(); i++ {
			node := strings.TrimSpace(scanner.Text())
			nodes = append(nodes, node)
		}

		if len(nodes) != count {
			return nil, fmt.Errorf("expected %d nodes for community %s, got %d", count, communityID, len(nodes))
		}

		mapping[communityID] = nodes
	}

	return mapping, scanner.Err()
}

// extractLevel extracts level from community ID (e.g., "c0_l1_0" -> 1)
func extractLevel(communityID string) int {
	parts := strings.Split(communityID, "_")
	if len(parts) >= 2 && strings.HasPrefix(parts[1], "l") {
		if level, err := strconv.Atoi(parts[1][1:]); err == nil {
			return level
		}
	}
	return 0
}
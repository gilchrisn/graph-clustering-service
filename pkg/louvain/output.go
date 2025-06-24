package louvain

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
)

// OutputWriter interface for flexible output generation
type OutputWriter interface {
	WriteMapping(result *LouvainResult, parser *GraphParser, path string) error
	WriteHierarchy(result *LouvainResult, parser *GraphParser, path string) error
	WriteRoot(result *LouvainResult, parser *GraphParser, path string) error
	WriteEdges(result *LouvainResult, parser *GraphParser, path string) error
	WriteAll(result *LouvainResult, parser *GraphParser, outputDir string, prefix string) error
}

// FileWriter implements OutputWriter for file-based output
type FileWriter struct{}

// NewFileWriter creates a new file-based output writer
func NewFileWriter() OutputWriter {
	return &FileWriter{}
}

// WriteAll writes all output files
func (fw *FileWriter) WriteAll(result *LouvainResult, parser *GraphParser, outputDir string, prefix string) error {
	// Debug output
	for level, levelInfo := range result.Levels {
		fmt.Printf("DEBUG: Level %d has %d communities\n", level, len(levelInfo.Communities))
		for commID, nodes := range levelInfo.Communities {
			if len(nodes) == 0 {
				fmt.Printf("DEBUG: Level %d community %d is empty\n", level, commID)
			}
		}
	}

	// Create output directory if it doesn't exist
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Write mapping file
	mappingPath := filepath.Join(outputDir, fmt.Sprintf("%s.mapping", prefix))
	if err := fw.WriteMapping(result, parser, mappingPath); err != nil {
		return fmt.Errorf("failed to write mapping: %w", err)
	}
	
	// Write hierarchy file
	hierarchyPath := filepath.Join(outputDir, fmt.Sprintf("%s.hierarchy", prefix))
	if err := fw.WriteHierarchy(result, parser, hierarchyPath); err != nil {
		return fmt.Errorf("failed to write hierarchy: %w", err)
	}
	
	// Write root file
	rootPath := filepath.Join(outputDir, fmt.Sprintf("%s.root", prefix))
	if err := fw.WriteRoot(result, parser, rootPath); err != nil {
		return fmt.Errorf("failed to write root: %w", err)
	}
	
	// Write edges file
	edgesPath := filepath.Join(outputDir, fmt.Sprintf("%s.edges", prefix))
	if err := fw.WriteEdges(result, parser, edgesPath); err != nil {
		return fmt.Errorf("failed to write edges: %w", err)
	}
	
	return nil
}

// WriteMapping writes the mapping from communities to original nodes
func (fw *FileWriter) WriteMapping(result *LouvainResult, parser *GraphParser, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Build the complete mapping from top-level communities to original nodes
	topLevel := result.Levels[len(result.Levels)-1]
	
	// Create sorted community list for consistent output
	var communityIDs []int
	for commID := range topLevel.Communities {
		communityIDs = append(communityIDs, commID)
	}
	sort.Ints(communityIDs)

	var nodeCount int
	
	// Write each community and its original nodes
	for _, commID := range communityIDs {
		// Get all original nodes in this top-level community
		originalNodes := fw.getOriginalNodes(result, parser, commID, len(result.Levels)-1)
		nodeCount += len(originalNodes)
		
		// Sort nodes for consistent output
		sort.Strings(originalNodes)
		
		// Write community identifier
		fmt.Fprintf(file, "c0_l%d_%d\n", len(result.Levels), commID)
		fmt.Fprintf(file, "%d\n", len(originalNodes))
		
		// Write nodes
		for _, node := range originalNodes {
			fmt.Fprintf(file, "%s\n", node)
		}
	}
	
	fmt.Printf("Total nodes written: %d\n", nodeCount)
	return nil
}

// WriteHierarchy writes the hierarchical community structure
func (fw *FileWriter) WriteHierarchy(result *LouvainResult, parser *GraphParser, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Write hierarchy for each level (except the first)
	for level := 1; level < len(result.Levels); level++ {
		levelInfo := result.Levels[level]
		
		// Get sorted community IDs
		var communityIDs []int
		for commID := range levelInfo.Communities {
			communityIDs = append(communityIDs, commID)
		}
		sort.Ints(communityIDs)
		
		// Write each community and its sub-communities
		for _, commID := range communityIDs {
			nodes := levelInfo.Communities[commID]
			
			// Community identifier
			fmt.Fprintf(file, "c0_l%d_%d\n", level+1, commID)
			
			// Find which communities from previous level are contained in this community
			prevLevel := result.Levels[level-1]
			subComms := make(map[int]bool)
			
			for _, node := range nodes {
				// Find which community this node belonged to in previous level
				if prevComm, exists := prevLevel.CommunityMap[node]; exists {
					subComms[prevComm] = true
				}
			}
			
			// Write sub-communities count and IDs
			fmt.Fprintf(file, "%d\n", len(subComms))
			
			// Write sorted sub-communities
			var subCommIDs []int
			for id := range subComms {
				subCommIDs = append(subCommIDs, id)
			}
			sort.Ints(subCommIDs)
			
			for _, subCommID := range subCommIDs {
				fmt.Fprintf(file, "%d\n", subCommID)
			}
		}
	}
	
	return nil
}

// WriteRoot writes the top-level communities
func (fw *FileWriter) WriteRoot(result *LouvainResult, parser *GraphParser, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Get top level communities
	topLevel := result.Levels[len(result.Levels)-1]
	
	// Sort community IDs
	var communityIDs []int
	for commID := range topLevel.Communities {
		communityIDs = append(communityIDs, commID)
	}
	sort.Ints(communityIDs)
	
	// Write root communities
	for _, commID := range communityIDs {
		fmt.Fprintf(file, "c0_l%d_%d\n", len(result.Levels), commID)
	}
	
	return nil
}

// WriteEdges writes the edges between communities at each level
func (fw *FileWriter) WriteEdges(result *LouvainResult, parser *GraphParser, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Write edges for each level
	for level := 1; level <= len(result.Levels); level++ {
		levelInfo := result.Levels[level-1]
		
		// Find edges between communities
		communityEdges := make(map[string]bool)
		
		// Use the graph from this level to find inter-community edges
		graph := levelInfo.Graph
		
		// For each edge in the graph, check if it connects different communities
		for i := 0; i < graph.NumNodes; i++ {
			comm1 := levelInfo.CommunityMap[i]
			
			neighbors := graph.GetNeighbors(i)
			for neighbor := range neighbors {
				comm2 := levelInfo.CommunityMap[neighbor]
				
				if comm1 != comm2 {
					// Create edge identifier (smaller ID first)
					if comm1 < comm2 {
						edgeID := fmt.Sprintf("c0_l%d_%d c0_l%d_%d", level, comm1, level, comm2)
						communityEdges[edgeID] = true
					} else {
						edgeID := fmt.Sprintf("c0_l%d_%d c0_l%d_%d", level, comm2, level, comm1)
						communityEdges[edgeID] = true
					}
				}
			}
		}
		
		// Write sorted edges
		var edges []string
		for edge := range communityEdges {
			edges = append(edges, edge)
		}
		sort.Strings(edges)
		
		for _, edge := range edges {
			fmt.Fprintf(file, "%s\n", edge)
		}
	}
	
	return nil
}

// getOriginalNodes recursively gets all original nodes in a community, converting to original IDs
func (fw *FileWriter) getOriginalNodes(result *LouvainResult, parser *GraphParser, commID int, level int) []string {
	if level == 0 {
		// Base case: convert normalized node IDs to original IDs
		nodes := result.Levels[0].Communities[commID]
		if nodes == nil {
			return []string{}
		}
		
		originalNodes := make([]string, 0, len(nodes))
		for _, normalizedNode := range nodes {
			if originalID, exists := parser.GetOriginalID(normalizedNode); exists {
				originalNodes = append(originalNodes, originalID)
			} else {
				fmt.Printf("WARNING: Could not find original ID for normalized node %d\n", normalizedNode)
			}
		}
		return originalNodes
	}
	
	// Recursive case: get nodes from lower levels
	var originalNodes []string
	nodes := result.Levels[level].Communities[commID]

	if nodes == nil {
		fmt.Printf("Community %d does not exist at level %d\n", commID, level)
		return []string{}
	}

	for _, node := range nodes {
		// Find which community this node belonged to in the previous level
		if prevComm, exists := result.Levels[level-1].CommunityMap[node]; exists {
			subNodes := fw.getOriginalNodes(result, parser, prevComm, level-1)
			originalNodes = append(originalNodes, subNodes...)
		} else {
			// This shouldn't happen in a well-formed hierarchy
			fmt.Printf("WARNING: Node %d at level %d not found in previous level\n", node, level)
		}
	}
	
	return originalNodes
}

// WriteOriginalMapping writes a simple mapping file with original node IDs
func (fw *FileWriter) WriteOriginalMapping(result *LouvainResult, parser *GraphParser, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Get final communities with original node IDs
	finalCommunities := make(map[string]int)
	for normalizedNode, community := range result.FinalCommunities {
		if originalID, exists := parser.GetOriginalID(normalizedNode); exists {
			finalCommunities[originalID] = community
		}
	}
	
	// Sort original node IDs for consistent output
	var originalIDs []string
	for originalID := range finalCommunities {
		originalIDs = append(originalIDs, originalID)
	}
	
	// Try to sort numerically if possible, otherwise lexicographically
	if fw.allIDsAreIntegers(originalIDs) {
		sort.Slice(originalIDs, func(i, j int) bool {
			a, _ := strconv.Atoi(originalIDs[i])
			b, _ := strconv.Atoi(originalIDs[j])
			return a < b
		})
	} else {
		sort.Strings(originalIDs)
	}
	
	// Write mapping
	for _, originalID := range originalIDs {
		community := finalCommunities[originalID]
		fmt.Fprintf(file, "%s %d\n", originalID, community)
	}
	
	return nil
}

// allIDsAreIntegers checks if all IDs can be parsed as integers
func (fw *FileWriter) allIDsAreIntegers(ids []string) bool {
	for _, id := range ids {
		if _, err := strconv.Atoi(id); err != nil {
			return false
		}
	}
	return true
}
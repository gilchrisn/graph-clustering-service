package louvain

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// OutputWriter interface for flexible output generation
type OutputWriter interface {
	WriteMapping(result *LouvainResult, originalGraph *HomogeneousGraph, path string) error
	WriteHierarchy(result *LouvainResult, path string) error
	WriteRoot(result *LouvainResult, path string) error
	WriteEdges(result *LouvainResult, originalGraph *HomogeneousGraph, path string) error
	WriteAll(result *LouvainResult, originalGraph *HomogeneousGraph, outputDir string, prefix string) error
}

// FileWriter implements OutputWriter for file-based output
type FileWriter struct{}

// NewFileWriter creates a new file-based output writer
func NewFileWriter() OutputWriter {
	return &FileWriter{}
}

// WriteAll writes all output files
func (fw *FileWriter) WriteAll(result *LouvainResult, originalGraph *HomogeneousGraph, outputDir string, prefix string) error {
	// Create output directory if it doesn't exist
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Write mapping file
	mappingPath := filepath.Join(outputDir, fmt.Sprintf("%s.mapping", prefix))
	if err := fw.WriteMapping(result, originalGraph, mappingPath); err != nil {
		return fmt.Errorf("failed to write mapping: %w", err)
	}
	
	// Write hierarchy file
	hierarchyPath := filepath.Join(outputDir, fmt.Sprintf("%s.hierarchy", prefix))
	if err := fw.WriteHierarchy(result, hierarchyPath); err != nil {
		return fmt.Errorf("failed to write hierarchy: %w", err)
	}
	
	// Write root file
	rootPath := filepath.Join(outputDir, fmt.Sprintf("%s.root", prefix))
	if err := fw.WriteRoot(result, rootPath); err != nil {
		return fmt.Errorf("failed to write root: %w", err)
	}
	
	// Write edges file
	edgesPath := filepath.Join(outputDir, fmt.Sprintf("%s.edges", prefix))
	if err := fw.WriteEdges(result, originalGraph, edgesPath); err != nil {
		return fmt.Errorf("failed to write edges: %w", err)
	}
	
	return nil
}

// WriteMapping writes the mapping from communities to original nodes
func (fw *FileWriter) WriteMapping(result *LouvainResult, originalGraph *HomogeneousGraph, path string) error {
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
	
	// Write each community and its original nodes
	for _, commID := range communityIDs {
		// Get all original nodes in this top-level community
		originalNodes := fw.getOriginalNodes(result, commID, len(result.Levels)-1)
		
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
	
	return nil
}

// WriteHierarchy writes the hierarchical community structure
func (fw *FileWriter) WriteHierarchy(result *LouvainResult, path string) error {
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
			fmt.Fprintf(file, "%d\n", len(nodes))
			
			// Write sub-communities (nodes from previous level)
			// Need to map these to their community IDs from previous level
			// prevLevel := result.Levels[level-1]
			subComms := make(map[int]bool)
			
			for _, node := range nodes {
				// Find which community this node belonged to in previous level
				if strings.HasPrefix(node, "c") {
					// Extract community ID from node name
					var prevCommID int
					fmt.Sscanf(node, "c%d", &prevCommID)
					subComms[prevCommID] = true
				}
			}
			
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
func (fw *FileWriter) WriteRoot(result *LouvainResult, path string) error {
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
func (fw *FileWriter) WriteEdges(result *LouvainResult, originalGraph *HomogeneousGraph, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// Write edges for each level
	for level := 1; level <= len(result.Levels); level++ {
		levelInfo := result.Levels[level-1]
		
		// Build node to community mapping for this level
		nodeToCommunity := make(map[string]int)
		for commID, nodes := range levelInfo.Communities {
			for _, node := range nodes {
				nodeToCommunity[node] = commID
			}
		}
		
		// Find edges between communities
		communityEdges := make(map[string]bool)
		
		// If this is the first level, use original graph
		if level == 1 {
			for edge := range originalGraph.Edges {
				comm1 := nodeToCommunity[edge.From]
				comm2 := nodeToCommunity[edge.To]
				
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
		} else {
			// Use the graph from this level
			for edge := range levelInfo.Graph.Edges {
				// Extract community IDs from node names
				var comm1, comm2 int
				fmt.Sscanf(edge.From, "c%d", &comm1)
				fmt.Sscanf(edge.To, "c%d", &comm2)
				
				if comm1 != comm2 {
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

// getOriginalNodes recursively gets all original nodes in a community
func (fw *FileWriter) getOriginalNodes(result *LouvainResult, commID int, level int) []string {
	if level == 0 {
		// Base case: return nodes directly
		return result.Levels[0].Communities[commID]
	}
	
	// Recursive case: get nodes from sub-communities
	var originalNodes []string
	nodes := result.Levels[level].Communities[commID]
	
	for _, node := range nodes {
		if strings.HasPrefix(node, "c") {
			// This is a community from previous level
			var subCommID int
			fmt.Sscanf(node, "c%d", &subCommID)
			subNodes := fw.getOriginalNodes(result, subCommID, level-1)
			originalNodes = append(originalNodes, subNodes...)
		} else {
			// This is an original node
			originalNodes = append(originalNodes, node)
		}
	}
	
	return originalNodes
}
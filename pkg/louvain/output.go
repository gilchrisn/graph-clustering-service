package louvain

import (
	"fmt"
	// "go/format"
	"os"
	"path/filepath"
	"sort"
	// "strconv"
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
type FileWriter struct {
	hierarchy map[string][]string // id -> children
	mapping   map[string][]string // id -> original nodes
	rootID    string               // root community ID
}

// NewFileWriter creates a new file-based output writer
func NewFileWriter() OutputWriter {
	return &FileWriter{
		hierarchy: make(map[string][]string),
		mapping:   make(map[string][]string),
		rootID:    "c0_l0_0", // Default root ID
	}
}

// WriteAll writes all output files
func (fw *FileWriter) WriteAll(result *LouvainResult, parser *GraphParser, outputDir string, prefix string) error {
	fmt.Printf("=== COMPLETE LOUVAIN RESULT DEBUG ===\n")
	fmt.Printf("Total Levels: %d\n", len(result.Levels))
	fmt.Printf("Final Communities: %v\n", result.FinalCommunities)
	fmt.Printf("Modularity: %f\n", result.Modularity)
	fmt.Printf("NumLevels: %d\n", result.NumLevels)
	
	fmt.Printf("\n=== PARSER DEBUG ===\n")
	fmt.Printf("NumNodes: %d\n", parser.NumNodes)
	fmt.Printf("OriginalToNormalized: %v\n", parser.OriginalToNormalized)
	fmt.Printf("NormalizedToOriginal: %v\n", parser.NormalizedToOriginal)
	
	fmt.Printf("\n=== LEVEL BY LEVEL DEBUG ===\n")
	for i, level := range result.Levels {
		fmt.Printf("\n--- LEVEL %d ---\n", i)
		fmt.Printf("Level: %d\n", level.Level)
		fmt.Printf("NumCommunities: %d\n", level.NumCommunities)
		fmt.Printf("NumMoves: %d\n", level.NumMoves)
		fmt.Printf("Modularity: %f\n", level.Modularity)
		
		fmt.Printf("Communities (commID -> nodes):\n")
		for commID, nodes := range level.Communities {
			fmt.Printf("  %d: %v\n", commID, nodes)
		}
		
		fmt.Printf("CommunityMap (node -> commID):\n")
		for node, commID := range level.CommunityMap {
			fmt.Printf("  %d: %d\n", node, commID)
		}
		
		fmt.Printf("SuperNodeToCommMap exists: %t\n", level.SuperNodeToCommMap != nil)
		if level.SuperNodeToCommMap != nil {
			fmt.Printf("SuperNodeToCommMap (superNode -> prevCommID):\n")
			for superNode, prevComm := range level.SuperNodeToCommMap {
				fmt.Printf("  %d: %d\n", superNode, prevComm)
			}
		}
		
		if level.Graph != nil {
			fmt.Printf("Graph NumNodes: %d\n", level.Graph.NumNodes)
			fmt.Printf("Graph TotalWeight: %f\n", level.Graph.TotalWeight)
		} else {
			fmt.Printf("Graph: nil\n")
		}
	}
	
	fmt.Printf("\n=== BUILDING STRUCTURES ===\n")
	// Build the two data structures
	fw.buildStructures(result, parser)
	
	fmt.Printf("\n=== BUILT HIERARCHY ===\n")
	for id, children := range fw.hierarchy {
		fmt.Printf("%s: %v\n", id, children)
	}
	
	fmt.Printf("\n=== BUILT MAPPING ===\n")
	for id, nodes := range fw.mapping {
		fmt.Printf("%s: %v\n", id, nodes)
	}
	
	fmt.Printf("\n=== CREATING OUTPUT FILES ===\n")
	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Write files
	files := map[string]func() error{
		"mapping":   func() error { return fw.WriteMapping(result, parser, filepath.Join(outputDir, prefix+".mapping")) },
		"hierarchy": func() error { return fw.WriteHierarchy(result, parser, filepath.Join(outputDir, prefix+".hierarchy")) },
		"root":      func() error { return fw.WriteRoot(result, parser, filepath.Join(outputDir, prefix+".root")) },
		"edges":     func() error { return fw.WriteEdges(result, parser, filepath.Join(outputDir, prefix+".edges")) },
	}
	
	for name, writeFunc := range files {
		if err := writeFunc(); err != nil {
			return fmt.Errorf("failed to write %s: %w", name, err)
		}
	}
	
	return nil
}

// buildStructures builds hierarchy and mapping
func (fw *FileWriter) buildStructures(result *LouvainResult, parser *GraphParser) {
	level := 0
	for level < len(result.Levels) {
		if level == 0 {
			for commID, nodes := range result.Levels[level].Communities {
				formattedID := fmt.Sprintf("c0_l%d_%d", level+1, commID)
				fmt.Printf("Processing community %s with nodes %v\n", formattedID, nodes)
				fw.mapping[formattedID] = []string{}
				for _, node := range nodes {
					fw.mapping[formattedID] = append(fw.mapping[formattedID], parser.NormalizedToOriginal[node])

				}
			}
		} else {
			for commID, nodes := range result.Levels[level].Communities {
				formattedID := fmt.Sprintf("c0_l%d_%d", level+1, commID)
				fmt.Printf("Processing community %s with nodes %v\n", formattedID, nodes)
				fw.mapping[formattedID] = []string{}
				fw.hierarchy[formattedID] = []string{}
				for _, node := range nodes {
					childOriginalID := result.Levels[level-1].SuperNodeToCommMap[node]
					fmt.Printf("  Child original ID: %d from %d\n", childOriginalID, node)
					formattedChildID := fmt.Sprintf("c0_l%d_%d", level, childOriginalID)
					fw.mapping[formattedID] = append(fw.mapping[formattedID], fw.mapping[formattedChildID]...)
					fw.hierarchy[formattedID] = append(fw.hierarchy[formattedID], formattedChildID)
				}
			}
		}
		level++
	}

	// if top level has > 1 community, merge them into a single root community
	if len(result.Levels) > 0 && len(result.Levels[len(result.Levels)-1].Communities) > 1 {
		rootID := fmt.Sprintf("c0_l%d_0", len(result.Levels))
		fw.rootID = rootID
		fw.hierarchy[rootID] = []string{}
		fw.mapping[rootID] = []string{}
		for commID, _ := range result.Levels[len(result.Levels)-1].Communities {
			formattedID := fmt.Sprintf("c0_l%d_%d", len(result.Levels)-1, commID)
			fw.hierarchy[rootID] = append(fw.hierarchy[rootID], formattedID)
			fw.mapping[rootID] = append(fw.mapping[rootID], fw.mapping[formattedID]...)
		}
	} else {
		// Get the id of the only community in the top level
		if len(result.Levels) > 0 && len(result.Levels[len(result.Levels)-1].Communities) == 1 {
			for commID := range result.Levels[len(result.Levels)-1].Communities {
				fw.rootID = fmt.Sprintf("c0_l%d_%d", len(result.Levels)-1, commID)
				break
			}
		} else {
			fw.rootID = "c0_l0_0" // Default root ID if no communities
		}
	}
}


// WriteMapping writes the mapping from communities to original nodes
func (fw *FileWriter) WriteMapping(result *LouvainResult, parser *GraphParser, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// SOMEHOW SORT

	// for id, nodes := range fw.mapping {
	for id, nodes := range fw.mapping {
		// Sort nodes for consistent output
		sort.Strings(nodes)
		
		// Write each community mapping
		fmt.Fprintf(file, "%s\n", id)
		fmt.Fprintf(file, "%d\n", len(nodes))
		// Write the nodes in the community
		for _, node := range nodes {
			fmt.Fprintf(file, "%s\n", node)
		}
	}
	 
	return nil
}

// WriteHierarchy writes the hierarchical community structure
func (fw *FileWriter) WriteHierarchy(result *LouvainResult, parser *GraphParser, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// SOMEHOW SORT

	for id, children := range fw.hierarchy {
		// Sort children for consistent output
		sort.Strings(children)
		
		// Write each community and its children
		fmt.Fprintf(file, "%s\n", id)
		fmt.Fprintf(file, "%d\n", len(children))
		for _, child := range children {
			fmt.Fprintf(file, "%s\n", child)
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

	// Write the root community ID
	fmt.Fprintf(file, "%s\n", fw.rootID)

	return nil
	
}


// WriteEdges writes the edges between communities at each level
func (fw *FileWriter) WriteEdges(result *LouvainResult, parser *GraphParser, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	// For each level, find edges between communities
	for level := 0; level < len(result.Levels); level++ {
		levelInfo := result.Levels[level]
		
		if levelInfo.Graph == nil {
			continue
		}
		
		communityEdges := make(map[string]bool)
		graph := levelInfo.Graph
		
		for i := 0; i < graph.NumNodes; i++ {
			comm1, exists1 := levelInfo.CommunityMap[i]
			if !exists1 {
				continue
			}
			
			neighbors := graph.GetNeighbors(i)
			for neighbor := range neighbors {
				comm2, exists2 := levelInfo.CommunityMap[neighbor]
				if !exists2 || comm1 == comm2 {
					continue
				}
				
				id1 := fmt.Sprintf("c0_l%d_%d", level, comm1)
				id2 := fmt.Sprintf("c0_l%d_%d", level, comm2)
				
				var edgeID string
				if id1 < id2 {
					edgeID = fmt.Sprintf("%s %s", id1, id2)
				} else {
					edgeID = fmt.Sprintf("%s %s", id2, id1)
				}
				
				communityEdges[edgeID] = true
			}
		}
		
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
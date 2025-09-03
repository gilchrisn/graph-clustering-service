// sketch_louvain_result.go - COMPLETE REWRITE

package scar

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"math"
)

// LevelResult stores the result data for a single level
type LevelResult struct {
	partition        []int64                          // partition[nodeId] = communityId
	sketches         map[int64]*VertexBottomKSketch  // sketches[nodeId] = sketch
	hashMap          map[uint32]int64                // hashMap[hash] = nodeId
	communityToNodes map[int64][]int64               // community -> member nodes
	commToNewNode    map[int64]int64                 // old community -> new super-node
	adjacencyList    map[int64][]WeightedEdge               // adjacency list for each community
}

// SketchLouvainResult stores the complete hierarchical clustering result
type SketchLouvainResult struct {
	levels    []LevelResult         // levels[i] = result for level i
	hierarchy map[string][]string   // community ID -> children community IDs
	mapping   map[string][]int64    // community ID -> original nodes
	rootID    string                // root community ID
}

// NewSketchLouvainResult creates a new hierarchical result
func NewSketchLouvainResult() *SketchLouvainResult {
	return &SketchLouvainResult{
		levels: make([]LevelResult, 0),
	}
}

// AddLevel adds a new level to the hierarchical result
func (slr *SketchLouvainResult) AddLevel(
	partition []int64,
	sketches map[int64]*VertexBottomKSketch,
	hashMap map[uint32]int64,
	communityToNodes map[int64][]int64,
	commToNewNode map[int64]int64,
	adjacencyList map[int64][]WeightedEdge, 
) {
	// Make copies to avoid reference issues
	partitionCopy := make([]int64, len(partition))
	copy(partitionCopy, partition)
	
	sketchesCopy := make(map[int64]*VertexBottomKSketch)
	for nodeId, sketch := range sketches {
		sketchesCopy[nodeId] = sketch
	}
	
	hashMapCopy := make(map[uint32]int64)
	for hash, nodeId := range hashMap {
		hashMapCopy[hash] = nodeId
	}
	
	communityToNodesCopy := make(map[int64][]int64)
	for commId, nodes := range communityToNodes {
		nodesCopy := make([]int64, len(nodes))
		copy(nodesCopy, nodes)
		communityToNodesCopy[commId] = nodesCopy
	}
	
	commToNewNodeCopy := make(map[int64]int64)
	for oldComm, newNode := range commToNewNode {
		commToNewNodeCopy[oldComm] = newNode
	}
	
	adjacencyListCopy := make(map[int64][]WeightedEdge)
	for nodeId, edges := range adjacencyList {
		edgesCopy := make([]WeightedEdge, len(edges))
		for i, edge := range edges {
			edgesCopy[i] = WeightedEdge{
				neighbor: edge.neighbor,
				weight:   edge.weight,
			}
		}
		adjacencyListCopy[nodeId] = edgesCopy
	}

	levelResult := LevelResult{
		partition:        partitionCopy,
		sketches:         sketchesCopy,
		hashMap:          hashMapCopy,
		communityToNodes: communityToNodesCopy,
		commToNewNode:    commToNewNodeCopy,
		adjacencyList:    adjacencyListCopy,
	}

	slr.levels = append(slr.levels, levelResult)
	
	fmt.Printf("Added level %d with %d nodes, %d communities\n", 
		len(slr.levels)-1, len(partition), len(communityToNodes))
}


func (slr *SketchLouvainResult) writeSketchGraphFiles(config SCARConfig) error {
	if len(slr.levels) == 0 {
		return fmt.Errorf("no levels available for sketch graph output")
	}
	
	firstLevel := slr.levels[0]
	
	if err := slr.writeEdgeListFile(config, firstLevel.adjacencyList); err != nil {
		return fmt.Errorf("failed to write edge list: %v", err)
	}
	
	if err := slr.writeAttributesFile(config, firstLevel.adjacencyList); err != nil {
		return fmt.Errorf("failed to write attributes: %v", err)
	}
	
	return nil
}

func (slr *SketchLouvainResult) writeEdgeListFile(config SCARConfig, adjacencyList map[int64][]WeightedEdge) error {
	filename := fmt.Sprintf("%s_graph.txt", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	edgeSet := make(map[string]bool)
	
	var nodeIDs []int64
	for nodeId := range adjacencyList {
		nodeIDs = append(nodeIDs, nodeId)
	}
	sort.Slice(nodeIDs, func(i, j int) bool { return nodeIDs[i] < nodeIDs[j] })
	
	for _, nodeId := range nodeIDs {
		for _, edge := range adjacencyList[nodeId] {
			var edgeKey string
			if nodeId < edge.neighbor {
				edgeKey = fmt.Sprintf("%d_%d", nodeId, edge.neighbor)
			} else {
				edgeKey = fmt.Sprintf("%d_%d", edge.neighbor, nodeId)
			}
			
			if edgeSet[edgeKey] {
				continue
			}
			edgeSet[edgeKey] = true
			
			if config.SketchGraphWeights {
				fmt.Fprintf(file, "%d %d %.6f\n", nodeId, edge.neighbor, edge.weight)
			} else {
				fmt.Fprintf(file, "%d %d\n", nodeId, edge.neighbor)
			}
		}
	}
	return nil
}

func (slr *SketchLouvainResult) writeAttributesFile(config SCARConfig, adjacencyList map[int64][]WeightedEdge) error {
	filename := fmt.Sprintf("%s_attributes.txt", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	nodeCount := int64(len(adjacencyList))
	
	edgeSet := make(map[string]bool)
	edgeCount := int64(0)
	
	for nodeId, edges := range adjacencyList {
		for _, edge := range edges {
			var edgeKey string
			if nodeId < edge.neighbor {
				edgeKey = fmt.Sprintf("%d_%d", nodeId, edge.neighbor)
			} else {
				edgeKey = fmt.Sprintf("%d_%d", edge.neighbor, nodeId)
			}
			
			if !edgeSet[edgeKey] {
				edgeSet[edgeKey] = true
				edgeCount++
			}
		}
	}
	
	fmt.Fprintf(file, "n = %d\n", nodeCount)
	fmt.Fprintf(file, "m = %d\n", edgeCount)
	return nil
}

// BuildHierarchy builds the hierarchy and mapping structures from stored data
func (slr *SketchLouvainResult) BuildHierarchy() {
	slr.hierarchy = make(map[string][]string)
	slr.mapping = make(map[string][]int64)
	
	for level := 0; level < len(slr.levels); level++ {
		fmt.Printf("Building hierarchy for level %d\n", level)
		if level == 0 {
			// Level 0: map communities to original nodes
			for commID, nodes := range slr.levels[level].communityToNodes {
				formattedID := fmt.Sprintf("c0_l%d_%d", level+1, commID)
				slr.mapping[formattedID] = make([]int64, len(nodes))
				copy(slr.mapping[formattedID], nodes)
			}
		} else {
			// Level N: map communities to child communities using stored mappings
			for commID, superNodes := range slr.levels[level].communityToNodes {
				formattedID := fmt.Sprintf("c0_l%d_%d", level+1, commID)
				slr.mapping[formattedID] = []int64{}
				slr.hierarchy[formattedID] = []string{}
				
				// Use the stored commToNewNode mapping from previous level
				prevLevelMapping := slr.levels[level-1].commToNewNode
				
				for _, superNode := range superNodes {
					// Find which old community this super-node came from
					for oldComm, newNode := range prevLevelMapping {
						if newNode == superNode {
							formattedChildID := fmt.Sprintf("c0_l%d_%d", level, oldComm)

							slr.hierarchy[formattedID] = append(slr.hierarchy[formattedID], fmt.Sprintf("%d", oldComm))
							if childMapping, exists := slr.mapping[formattedChildID]; exists {
								slr.mapping[formattedID] = append(slr.mapping[formattedID], childMapping...)
							}
							break
						}
					}
				}
			}
		}
	}

	// Determine root ID
	if len(slr.levels) > 0 {
		lastLevel := len(slr.levels) - 1
		lastLevelCommunities := slr.levels[lastLevel].communityToNodes
		
		if len(lastLevelCommunities) > 1 {
			// Multiple communities at top level - create virtual root
			slr.rootID = fmt.Sprintf("c0_l%d_0", len(slr.levels)+1)
			slr.hierarchy[slr.rootID] = []string{}
			slr.mapping[slr.rootID] = []int64{}
			
			for commID := range lastLevelCommunities {
				formattedID := fmt.Sprintf("c0_l%d_%d", lastLevel+1, commID)
				slr.hierarchy[slr.rootID] = append(slr.hierarchy[slr.rootID], fmt.Sprintf("%d", commID))
				if nodes, exists := slr.mapping[formattedID]; exists {
					slr.mapping[slr.rootID] = append(slr.mapping[slr.rootID], nodes...)
				}
			}
		} else {
			// Single community at top level
			for commID := range lastLevelCommunities {
				slr.rootID = fmt.Sprintf("c0_l%d_%d", lastLevel+1, commID)
				break
			}
		}
	} else {
		slr.rootID = "c0_l1_0" // Default root
	}
}

// WriteFiles writes the hierarchical result to output files
func (slr *SketchLouvainResult) WriteFiles(config SCARConfig) error {
	fmt.Println("\n=== WRITING HIERARCHICAL RESULT FILES ===")
	
	// Build the hierarchy structures first
	slr.BuildHierarchy()
	
	// SCAR HIERARCHY + SKETCH MODE: Write all hierarchy files + .sketch
	fmt.Println("SCAR sketch output mode: writing hierarchy files + .sketch file")
	
	if err := slr.writeMappingFile(config); err != nil {
		return fmt.Errorf("failed to write mapping file: %v", err)
	}
	
	if err := slr.writeHierarchyFile(config); err != nil {
		return fmt.Errorf("failed to write hierarchy file: %v", err)
	}
	
	if err := slr.writeRootFile(config); err != nil {
		return fmt.Errorf("failed to write root file: %v", err)
	}
	
	if config.SketchOutput {
		if err := slr.writeSketchFile(config); err != nil {
			return fmt.Errorf("failed to write sketch file: %v", err)
		}
	}

	if config.WriteSketchGraph {
		if err := slr.writeSketchGraphFiles(config); err != nil {
			return fmt.Errorf("failed to write sketch graph files: %v", err)
		}
	}
	fmt.Println("Output files written successfully")
	return nil
}

// writeMappingFile writes the mapping from communities to original nodes
func (slr *SketchLouvainResult) writeMappingFile(config SCARConfig) error {
	filename := fmt.Sprintf("%s_mapping.dat", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Printf("Writing mapping file: %s\n", filename)
	
	// Sort community IDs for consistent output
	var communityIDs []string
	for id := range slr.mapping {
		communityIDs = append(communityIDs, id)
	}
	sort.Strings(communityIDs)
	
	for _, id := range communityIDs {
		nodes := slr.mapping[id]
		// Sort nodes for consistent output
		sort.Slice(nodes, func(i, j int) bool { return nodes[i] < nodes[j] })
		
		// Write community mapping
		fmt.Fprintf(file, "%s\n", id)
		fmt.Fprintf(file, "%d\n", len(nodes))
		for _, node := range nodes {
			fmt.Fprintf(file, "%d\n", node)
		}
	}
	
	return nil
}

// writeHierarchyFile writes the hierarchical community structure
func (slr *SketchLouvainResult) writeHierarchyFile(config SCARConfig) error {
	filename := fmt.Sprintf("%s_hierarchy.dat", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Printf("Writing hierarchy file: %s\n", filename)
	
	// Sort community IDs for consistent output  
	var communityIDs []string
	for id := range slr.hierarchy {
		communityIDs = append(communityIDs, id)
	}
	sort.Strings(communityIDs)
	
	for _, id := range communityIDs {
		children := slr.hierarchy[id]
		// Sort children for consistent output
		sort.Strings(children)
		
		// Write community hierarchy
		fmt.Fprintf(file, "%s\n", id)
		fmt.Fprintf(file, "%d\n", len(children))
		for _, child := range children {
			fmt.Fprintf(file, "%s\n", child)
		}
	}
	
	return nil
}

// writeRootFile writes the root community identifier
func (slr *SketchLouvainResult) writeRootFile(config SCARConfig) error {
	filename := fmt.Sprintf("%s_root.dat", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Printf("Writing root file: %s\n", filename)
	
	// Write the root community ID
	fmt.Fprintf(file, "%s\n", slr.rootID)
	
	return nil
}

// writeSketchFile writes the sketch data for all levels
func (slr *SketchLouvainResult) writeSketchFile(config SCARConfig) error {
	filename := fmt.Sprintf("%s.sketch", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Printf("Writing sketch file: %s\n", filename)
	
	// Write header with parameters
	fmt.Fprintf(file, "k=%d nk=%d th=%.3f pro=%s path=%s\n", 
		config.K, config.NK, config.Threshold, 
		filepath.Base(config.PropertyFile), filepath.Base(config.PathFile))
	
	// Write sketches for all levels
    for level, levelResult := range slr.levels {
        // Build reverse mapping: normalized ID -> original community ID
        var reverseMapping map[int64]int64
        if level > 0 {
            reverseMapping = make(map[int64]int64)
            prevLevelMapping := slr.levels[level-1].commToNewNode
            for originalComm, normalizedId := range prevLevelMapping {
                reverseMapping[normalizedId] = originalComm
            }
        }
        
        var nodeIDs []int64
        for nodeId := range levelResult.sketches {
            nodeIDs = append(nodeIDs, nodeId)
        }
        sort.Slice(nodeIDs, func(i, j int) bool { return nodeIDs[i] < nodeIDs[j] })
        
        for _, nodeId := range nodeIDs {
            sketch := levelResult.sketches[nodeId]
            if sketch != nil {
                var formattedId string
                if level == 0 {
                    // Level 0: just node ID
                    formattedId = fmt.Sprintf("%d", nodeId)
                } else {
                    // Level 1+: map back to original community ID

                    originalId := reverseMapping[nodeId]
                    formattedId = fmt.Sprintf("c0_l%d_%d", level, originalId)
                }
                slr.writeNodeSketch(file, formattedId, nodeId, level, sketch, config.NK)
            }
        }
    }
    
    return nil
}


// writeNodeSketch writes a single node's sketch to file
func (slr *SketchLouvainResult) writeNodeSketch(
	file *os.File,
	formattedId string,
	nodeId int64,  // ← Keep original nodeId for hash lookup
	level int,
	sketch *VertexBottomKSketch,
	nk int64,
) {
	fmt.Fprintf(file, "%s\n", formattedId)  // ← Use formatted ID

	// Find this node's own hashes from the hashToNodeMap 
	levelResult := slr.levels[level]
	var nodeOwnHashes []uint32
	for hash, mappedNodeId := range levelResult.hashMap {
		if mappedNodeId == nodeId {
			nodeOwnHashes = append(nodeOwnHashes, hash)
		}
	}
	
	// Sort for consistent output
	sort.Slice(nodeOwnHashes, func(i, j int) bool { return nodeOwnHashes[i] < nodeOwnHashes[j] })
	
	// Write node's own hashes
	var hashStrs []string
	for _, hash := range nodeOwnHashes {
		hashStrs = append(hashStrs, fmt.Sprintf("%d", hash))
	}
	if len(hashStrs) > 0 {
		fmt.Fprintf(file, "%s\n", strings.Join(hashStrs, ","))
	} else {
		fmt.Fprintf(file, "\n")
	}
	
	// Write sketch layers
	for layer := int64(0); layer < nk; layer++ {
		layerSketch := sketch.GetSketch(layer)
		var sketchStrs []string
		
		for _, val := range layerSketch {
			if val != math.MaxUint32 {
				sketchStrs = append(sketchStrs, fmt.Sprintf("%d", val))
			}
		}
		
		if len(sketchStrs) > 0 {
			fmt.Fprintf(file, "%s\n", strings.Join(sketchStrs, ","))
		} else {
			fmt.Fprintf(file, "\n")
		}
	}
	

}
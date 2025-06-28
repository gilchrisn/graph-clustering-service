package scar

import (
	"fmt"
)

func (pt *MultiLevelPartitionTracker) recordAggregation(
    oldCommunity []int64,
    commToNewNode map[int64]int64,
	currentSketches map[int64]*VertexBottomKSketch,
) {
    fmt.Printf("\n=== RECORDING LEVEL %d AGGREGATION ===\n", pt.currentLevel)
    
    newMapping := LevelMapping{
        superNodeToChildNodes:    make(map[int64][]int64),
        superNodeToOriginalNodes: make(map[int64][]int64),
        numNodes:                int64(len(commToNewNode)),
    }

    // Save sketches from current level before aggregation
    newMapping.levelSketches = make(map[int64]*VertexBottomKSketch)
    for nodeId, sketch := range currentSketches {
        newMapping.levelSketches[nodeId] = sketch
    }

    if pt.currentLevel == 0 {
        // First aggregation: original nodes -> super-nodes
        fmt.Println("First level: mapping original nodes to super-nodes")
        
        // Group original nodes by community
        communityToNodes := make(map[int64][]int64)
        for node, comm := range oldCommunity {
            communityToNodes[comm] = append(communityToNodes[comm], int64(node))
        }
        
        // Create mappings
        for comm, nodes := range communityToNodes {
            if superNodeId, exists := commToNewNode[comm]; exists {
                newMapping.superNodeToChildNodes[superNodeId] = nodes
                newMapping.superNodeToOriginalNodes[superNodeId] = nodes // Same at level 0
                
                fmt.Printf("Super-node %d <- Community %d: nodes %v\n", 
                    superNodeId, comm, nodes)
            }
        }
    } else {
        // Subsequent aggregations: super-nodes -> new super-nodes
        fmt.Printf("Level %d: mapping super-nodes to new super-nodes\n", pt.currentLevel)
        
        previousLevel := pt.levelMappings[pt.currentLevel-1]
        
        // Group old super-nodes by their new community
        communityToSuperNodes := make(map[int64][]int64)
        for superNode, comm := range oldCommunity {
            communityToSuperNodes[comm] = append(communityToSuperNodes[comm], int64(superNode))
        }
        
        // Create mappings
        for comm, oldSuperNodes := range communityToSuperNodes {
            if newSuperNodeId, exists := commToNewNode[comm]; exists {
                // Track which old super-nodes form this new one
                newMapping.superNodeToChildNodes[newSuperNodeId] = oldSuperNodes
                
                // Flatten to get all original nodes
                var allOriginalNodes []int64
                for _, oldSuperNode := range oldSuperNodes {
                    if origNodes, exists := previousLevel.superNodeToOriginalNodes[oldSuperNode]; exists {
                        allOriginalNodes = append(allOriginalNodes, origNodes...)
                    }
                }
                newMapping.superNodeToOriginalNodes[newSuperNodeId] = allOriginalNodes
                
                fmt.Printf("New super-node %d <- Old super-nodes %v (contains %d original nodes)\n",
                    newSuperNodeId, oldSuperNodes, len(allOriginalNodes))
            }
        }
    }

	
    
    // Save this level's mapping
    pt.levelMappings = append(pt.levelMappings, newMapping)
    pt.currentLevel++
    
    fmt.Printf("Saved level %d mapping: %d super-nodes\n", 
        pt.currentLevel-1, newMapping.numNodes)
}

// Reconstruct the complete hierarchy from any level
func (pt *MultiLevelPartitionTracker) reconstructHierarchy(finalCommunity []int64) HierarchicalPartition {
    fmt.Println("\n=== RECONSTRUCTING HIERARCHY ===")
    
    result := HierarchicalPartition{
        OriginalGraphSize: pt.originalGraphSize,
        NumLevels:        pt.currentLevel + 1, // +1 for final level
        Levels:           make([]LevelInfo, 0),
    }
    
    // Add all aggregation levels
    for i, mapping := range pt.levelMappings {
        levelInfo := LevelInfo{
            Level:                   i,
            NumNodes:               mapping.numNodes,
            SuperNodeToChildNodes:   mapping.superNodeToChildNodes,
            SuperNodeToOriginalNodes: mapping.superNodeToOriginalNodes,
        }
        result.Levels = append(result.Levels, levelInfo)
        
        fmt.Printf("Level %d: %d super-nodes\n", i, mapping.numNodes)
    }
    
    // Add final level (leaf communities)
    finalLevel := LevelInfo{
        Level:    pt.currentLevel,
        NumNodes: int64(len(finalCommunity)),
        Communities: finalCommunity,
    }
    result.Levels = append(result.Levels, finalLevel)
    
    fmt.Printf("Final level %d: %d communities\n", pt.currentLevel, len(finalCommunity))
    
    return result
}

// Get final partition by tracing back to original nodes
func (pt *MultiLevelPartitionTracker) getFinalPartition(finalCommunity []int64) []int64 {
    fmt.Println("\n=== GETTING FINAL PARTITION ===")
    
    originalPartition := make([]int64, pt.originalGraphSize)
    
    if pt.currentLevel == 0 {
        // No aggregation happened
        return finalCommunity
    }
    
    // Get the last level mapping
    lastLevel := pt.levelMappings[pt.currentLevel-1]
    
    // Map each final community assignment back to original nodes
    for superNodeId, finalCommId := range finalCommunity {
        if originalNodes, exists := lastLevel.superNodeToOriginalNodes[int64(superNodeId)]; exists {
            for _, origNode := range originalNodes {
                originalPartition[origNode] = finalCommId
                fmt.Printf("Original node %d -> Community %d\n", origNode, finalCommId)
            }
        }
    }
    
    return originalPartition
}


func (pt *MultiLevelPartitionTracker) traceNode(originalNodeId int64) {
    fmt.Printf("\n=== TRACING NODE %d THROUGH HIERARCHY ===\n", originalNodeId)
    
    currentId := originalNodeId
    fmt.Printf("Level -1 (Original): Node %d\n", currentId)
    
    for level := 0; level < pt.currentLevel; level++ {
        mapping := pt.levelMappings[level]
        
        // Find which super-node contains our current node
        found := false
        for superNodeId, nodes := range mapping.superNodeToOriginalNodes {
            for _, node := range nodes {
                if node == originalNodeId {
                    fmt.Printf("Level %d: Node %d is in super-node %d\n", 
                        level, originalNodeId, superNodeId)
                    currentId = superNodeId
                    found = true
                    break
                }
            }
            if found {
                break
            }
        }
    }
}
package scar

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"math"
)

// OutputWriter handles writing results to files
type OutputWriter struct{}

func NewOutputWriter() *OutputWriter {
	return &OutputWriter{}
}

// WriteResults writes the old simple format for backward compatibility
func (ow *OutputWriter) WriteResults(outputFile string, community []int64, n int64) error {
	file, err := os.Create(outputFile)
	if err != nil {
		return err
	}
	defer file.Close()
	
	for i := int64(0); i < n; i++ {
		if community[i] != -1 {
			fmt.Fprintf(file, "%d %d\n", i, community[i])
		}
	}
	
	return nil
}

// WriteLouvainResults writes results based on config.SketchOutput flag
func (ow *OutputWriter) WriteLouvainResults(
	config SCARConfig,
	finalCommunity []int64,
	partitionTracker *MultiLevelPartitionTracker,
	graph *GraphStructure,
	sketchManager *SketchManager,
) error {
	fmt.Println("\n=== WRITING LOUVAIN RESULTS ===")
	
	if config.SketchOutput {
		// SCAR HIERARCHY + SKETCH MODE: Write all hierarchy files + .sketch
		fmt.Println("SCAR sketch output mode: writing hierarchy files + .sketch file")
		
		if err := ow.writeMappingFile(config, finalCommunity, partitionTracker); err != nil {
			return fmt.Errorf("failed to write mapping file: %v", err)
		}
		
		if err := ow.writeHierarchyFile(config, partitionTracker, finalCommunity); err != nil {
			return fmt.Errorf("failed to write hierarchy file: %v", err)
		}
		
		if err := ow.writeRootFile(config, finalCommunity, partitionTracker); err != nil {
			return fmt.Errorf("failed to write root file: %v", err)
		}
		
		if err := ow.writeSketchFile(config, sketchManager, partitionTracker); err != nil {
			return fmt.Errorf("failed to write sketch file: %v", err)
		}
		
	} else {
		// BASIC MODE: Write only basic output.txt
		fmt.Println("Basic output mode: writing simple output file")
		
		if err := ow.WriteResults(config.OutputFile, finalCommunity, int64(len(finalCommunity))); err != nil {
			return fmt.Errorf("failed to write basic output: %v", err)
		}
	}
	
	fmt.Println("Output files written successfully")
	return nil
}
// writeMappingFile writes the {prefix}_mapping.dat file
func (ow *OutputWriter) writeMappingFile(
	config SCARConfig,
	finalCommunity []int64,
	partitionTracker *MultiLevelPartitionTracker,
) error {
	filename := fmt.Sprintf("%s_mapping.dat", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Printf("Writing mapping file: %s\n", filename)
	
	if partitionTracker.currentLevel == 0 {
		// No hierarchy - write simple node->community mapping
		communityToNodes := make(map[int64][]int64)
		for nodeId, comm := range finalCommunity {
			communityToNodes[comm] = append(communityToNodes[comm], int64(nodeId))
		}
		
		for commId, nodes := range communityToNodes {
			// Start from level 1, not 0
			fmt.Fprintf(file, "c0_l1_%d\n", commId)
			fmt.Fprintf(file, "%d\n", len(nodes))
			for _, nodeId := range nodes {
				fmt.Fprintf(file, "%d\n", nodeId)
			}
		}
		return nil
	}
	
	// Write hierarchy - traverse all levels
	for level, mapping := range partitionTracker.levelMappings {
		for superNodeId, originalNodes := range mapping.superNodeToOriginalNodes {
			// Add 1 to level to start from 1
			fmt.Fprintf(file, "c0_l%d_%d\n", level+1, superNodeId)
			fmt.Fprintf(file, "%d\n", len(originalNodes))
			for _, nodeId := range originalNodes {
				fmt.Fprintf(file, "%d\n", nodeId)
			}
		}
	}
	
	// Write final level communities
	finalLevel := partitionTracker.currentLevel
	finalCommunityToNodes := make(map[int64][]int64)
	
	// Get final partition mapped back to original nodes
	if finalLevel > 0 {
		lastLevelMapping := partitionTracker.levelMappings[finalLevel-1]
		for superNodeId, finalCommId := range finalCommunity {
			if originalNodes, exists := lastLevelMapping.superNodeToOriginalNodes[int64(superNodeId)]; exists {
				finalCommunityToNodes[finalCommId] = append(finalCommunityToNodes[finalCommId], originalNodes...)
			}
		}
	} else {
		for nodeId, comm := range finalCommunity {
			finalCommunityToNodes[comm] = append(finalCommunityToNodes[comm], int64(nodeId))
		}
	}
	
	for commId, nodes := range finalCommunityToNodes {
		// Add 1 to level to start from 1
		fmt.Fprintf(file, "c0_l%d_%d\n", finalLevel+1, commId)
		fmt.Fprintf(file, "%d\n", len(nodes))
		for _, nodeId := range nodes {
			fmt.Fprintf(file, "%d\n", nodeId)
		}
	}

	// Only create virtual root if we have a multi-level hierarchy AND multiple communities at the highest level
	// Don't create virtual root if:
	// 1. No hierarchy (currentLevel == 0) - already handled above
	// 2. Only one community at the final level
	if finalLevel > 0 && len(finalCommunityToNodes) > 1 {
		// Add virtual root that contains all final communities  
		allNodes := make([]int64, 0)
		for _, nodes := range finalCommunityToNodes {
			allNodes = append(allNodes, nodes...)
		}
		// Add 1 to level and then +1 for virtual root
		fmt.Fprintf(file, "c0_l%d_0\n", finalLevel+2)
		fmt.Fprintf(file, "%d\n", len(allNodes))
		for _, nodeId := range allNodes {
			fmt.Fprintf(file, "%d\n", nodeId)
		}
	}
		
	return nil
}

// writeHierarchyFile writes the {prefix}_hierarchy.dat file
func (ow *OutputWriter) writeHierarchyFile(
	config SCARConfig,
	partitionTracker *MultiLevelPartitionTracker,
	finalCommunity []int64,
) error {
	filename := fmt.Sprintf("%s_hierarchy.dat", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Printf("Writing hierarchy file: %s\n", filename)
	
	// Write parent-child relationships for each level
	for level, mapping := range partitionTracker.levelMappings {
		for superNodeId, childNodes := range mapping.superNodeToChildNodes {
			// Add 1 to level to start from 1
			fmt.Fprintf(file, "c0_l%d_%d\n", level+1, superNodeId)
			fmt.Fprintf(file, "%d\n", len(childNodes))
			for _, childId := range childNodes {
				if level == 0 {
					// Level 0: children are original nodes
					fmt.Fprintf(file, "%d\n", childId)
				} else {
					// Higher levels: children are from previous level
					// level is already adjusted, so use level instead of level-1
					fmt.Fprintf(file, "c0_l%d_%d\n", level, childId)
				}
			}
		}
	}
	
	// Add virtual root level if multiple final communities exist AND we have hierarchy
	uniqueComms := make(map[int64]bool)
	for _, comm := range finalCommunity {
		uniqueComms[comm] = true
	}
	
	// Only create virtual root if we have hierarchy AND multiple communities at final level
	if partitionTracker.currentLevel > 0 && len(uniqueComms) > 1 {
		// Virtual root contains all final communities
		// Add 1 to currentLevel and then +1 for virtual root
		fmt.Fprintf(file, "c0_l%d_0\n", partitionTracker.currentLevel+2)
		fmt.Fprintf(file, "%d\n", len(uniqueComms))
		for comm := range uniqueComms {
			// Add 1 to currentLevel 
			fmt.Fprintf(file, "c0_l%d_%d\n", partitionTracker.currentLevel+1, comm)
		}
	}
	
	return nil
}

// writeRootFile writes the {prefix}_root.dat file
func (ow *OutputWriter) writeRootFile(
	config SCARConfig,
	finalCommunity []int64,
	partitionTracker *MultiLevelPartitionTracker,
) error {
	filename := fmt.Sprintf("%s_root.dat", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Printf("Writing root file: %s\n", filename)
	
    // Find unique final communities
    uniqueComms := make(map[int64]bool)
    for _, comm := range finalCommunity {
        uniqueComms[comm] = true
    }
    
    // Only add virtual root level if we have hierarchy AND multiple final communities
    if partitionTracker.currentLevel > 0 && len(uniqueComms) > 1 {
        // Create virtual root that contains all final communities
        // Add 1 to currentLevel and then +1 for virtual root
        fmt.Fprintf(file, "c0_l%d_0\n", partitionTracker.currentLevel+2)
    } else {
        // Single final community or no hierarchy - it's already the root
        finalLevel := partitionTracker.currentLevel
        for comm := range uniqueComms {
            // Add 1 to finalLevel to start from 1
            fmt.Fprintf(file, "c0_l%d_%d\n", finalLevel+1, comm)
        }
    }
    
    return nil
}

// writeSketchFile writes the {prefix}.sketch file with complete hierarchy traversal
func (ow *OutputWriter) writeSketchFile(
	config SCARConfig,
	sketchManager *SketchManager,
	partitionTracker *MultiLevelPartitionTracker,
) error {
	filename := fmt.Sprintf("%s.sketch", config.Prefix)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Printf("Writing sketch file: %s\n", filename)
	
	// Write header with parameters
	propertyFileBase := filepath.Base(config.PropertyFile)
	pathFileBase := filepath.Base(config.PathFile)
	fmt.Fprintf(file, "k=%d nk=%d th=%.3f pro=%s path=%s\n", 
		config.K, config.NK, config.Threshold, propertyFileBase, pathFileBase)
	
	// Write sketches for all hierarchy levels
	if partitionTracker.currentLevel == 0 {
		// No hierarchy - write original nodes at level 0
		for nodeId, sketch := range sketchManager.vertexSketches {
			ow.writeNodeSketch(file, nodeId, 0, sketch, config.NK)
		}
	} else {
		// Write complete hierarchy
		ow.writeHierarchicalSketches(file, sketchManager, partitionTracker, config)
	}
	
	return nil
}

// writeHierarchicalSketches writes sketches for all levels in the hierarchy
func (ow *OutputWriter) writeHierarchicalSketches(
    file *os.File,
    sketchManager *SketchManager,
    partitionTracker *MultiLevelPartitionTracker,
    config SCARConfig,
) {
    // Write sketches for each level using preserved sketches
    for level := 0; level <= partitionTracker.currentLevel; level++ {
        fmt.Printf("Writing sketches for level %d\n", level)
        
        if level == partitionTracker.currentLevel {
            // Final level - use current sketch manager
            for nodeId, sketch := range sketchManager.vertexSketches {
                ow.writeNodeSketch(file, nodeId, level, sketch, config.NK)
            }
        } else {
            // Historical levels - use preserved sketches
            if level < len(partitionTracker.levelMappings) {
                mapping := partitionTracker.levelMappings[level]
                for nodeId, sketch := range mapping.levelSketches {
                    ow.writeNodeSketch(file, nodeId, level, sketch, config.NK)
                }
            }
        }
    }
}

// writeNodeSketch writes a single node's sketch to file
func (ow *OutputWriter) writeNodeSketch(
	file *os.File,
	nodeId int64,
	level int,
	sketch *VertexBottomKSketch,
	nk int64,
) {
	fmt.Fprintf(file, "%d level=%d\n", nodeId, level)
	
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
			fmt.Fprintf(file, "\n") // Empty line for empty sketch
		}
	}
}
package scar

import (
	"fmt"
	"os"
)

// OutputWriter handles writing results to files
type OutputWriter struct{}

func NewOutputWriter() *OutputWriter {
	return &OutputWriter{}
}

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

func (ow *OutputWriter) ReconstructEdges(
	edgelistFile string,
	nodesInCommunity []int64,
	communitySketches map[int64][]uint32,
	hashToNodeMap map[uint32]int64,
) error {
	file, err := os.Create(edgelistFile)
	if err != nil {
		return err
	}
	defer file.Close()
	
	uniqueEdges := make(map[string]bool)
	
	for nodeIndex, currentNode := range nodesInCommunity {
		if sketches, exists := communitySketches[int64(nodeIndex)]; exists {
			for _, sketchValue := range sketches {
				if neighborNode, exists := hashToNodeMap[sketchValue]; exists {
					if currentNode != neighborNode {
						src, dst := currentNode, neighborNode
						if src > dst {
							src, dst = dst, src
						}
						edgeKey := fmt.Sprintf("%d_%d", src, dst)
						if !uniqueEdges[edgeKey] {
							uniqueEdges[edgeKey] = true
							fmt.Fprintf(file, "%d %d\n", src, dst)
						}
					}
				}
			}
		}
	}
	
	fmt.Printf("Reconstructed %d edges\n", len(uniqueEdges))
	return nil
}

// Legacy functions for backward compatibility
func reconstructEdges(edgelistFile string, nodesInCommunity []int64, communitySketches map[int64][]uint32, hashToNodeMap map[uint32]int64) {
	writer := NewOutputWriter()
	if err := writer.ReconstructEdges(edgelistFile, nodesInCommunity, communitySketches, hashToNodeMap); err != nil {
		fmt.Printf("Error reconstructing edges: %v\n", err)
	}
}

func writeOutput(outputFile string, community []int64, n int64) {
	writer := NewOutputWriter()
	if err := writer.WriteResults(outputFile, community, n); err != nil {
		fmt.Printf("Error writing output: %v\n", err)
	}
}
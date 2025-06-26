package scar

import (
	"fmt"
	"time"
)

// LouvainSCAR implements traditional Louvain algorithm with sketch-based modularity
type LouvainSCAR struct {
	graphReader       *GraphReader
	fileReader        *FileReader
	sketchComputer    *SketchComputer
	outputWriter      *OutputWriter
	modCalculator     *ModularityCalculator
}

func NewLouvainSCAR() *LouvainSCAR {
	return &LouvainSCAR{
		graphReader:   NewGraphReader(),
		fileReader:    NewFileReader(),
		sketchComputer: NewSketchComputer(),
		outputWriter:  NewOutputWriter(),
		modCalculator: NewModularityCalculator(),
	}
}

func (ls *LouvainSCAR) RunWithConfig(config SCARConfig) error {
	fmt.Println("Start reading graph")
	
	// Phase 1: Read graph and compute sketches (same as before)
	graph, sketches, nodeHashValue, wholeWeight, err := ls.initializeGraph(config)
	if err != nil {
		return err
	}
	
	n := graph.n
	
	// Phase 2: Traditional Louvain with sketch-based modularity
	startTime := time.Now()
	community := make([]int64, n)
	
	// Initialize: each node in its own community
	for i := int64(0); i < n; i++ {
		community[i] = i
	}
	
	// Create node-to-sketch mapping for valid nodes only
	nodeToSketch := ls.createNodeSketchMapping(n, sketches, nodeHashValue, config.K, config.NK)
	
	// Run Louvain phases
	totalImprovement := true
	phase := 0
	
	for totalImprovement && phase < 10 {
		fmt.Printf("=== Louvain Phase %d ===\n", phase)
		
		// Phase 1: Local optimization (traditional Louvain node moves)
		improvement := true
		localIter := 0
		
		for improvement && localIter < 100 {
			improvement = false
			nodesProcessed := 0
			nodesMoved := 0
			
			// For each node, try to move to best neighboring community
			for nodeId := int64(0); nodeId < n; nodeId++ {
				// Skip nodes without sketches
				if _, hasSketch := nodeToSketch[nodeId]; !hasSketch {
					continue
				}
				
				nodesProcessed++
				currentCommunity := community[nodeId]
				
				// Find neighboring communities through sketch similarity
				neighborCommunities := ls.findNeighboringCommunities(nodeId, nodeToSketch, community, config.K)
				
				if len(neighborCommunities) == 0 {
					continue
				}
				
				// Calculate gain for moving to each neighboring community
				bestCommunity := currentCommunity
				bestGain := 0.0
				
				for neighborComm := range neighborCommunities {
					if neighborComm == currentCommunity {
						continue
					}
					
					// Calculate modularity gain using E-function approximation
					gain := ls.calculateModularityGain(nodeId, currentCommunity, neighborComm, nodeToSketch, community, wholeWeight, config.K, config.NK)
					
					if gain > bestGain {
						bestGain = gain
						bestCommunity = neighborComm
					}
				}
				
				// Move node if beneficial
				if bestCommunity != currentCommunity {
					community[nodeId] = bestCommunity
					improvement = true
					nodesMoved++
				}
			}
			
			localIter++
			fmt.Printf("Local iter %d: %d nodes processed, %d moved\n", localIter, nodesProcessed, nodesMoved)
		}
		
		if !improvement {
			totalImprovement = false
			break
		}
		
		// Phase 2: Community aggregation (create super-graph)
		graph, nodeToSketch, community = ls.aggregateCommunities(graph, nodeToSketch, community, config.K, config.NK)
		n = graph.n
		
		phase++
	}
	
	fmt.Printf("Louvain completed in %v\n", time.Since(startTime))
	
	// Write results
	err = ls.outputWriter.WriteResults(config.OutputFile, community, n)
	if err != nil {
		return fmt.Errorf("failed to write output: %v", err)
	}
	
	// Calculate final modularity
	finalModularity := ls.calculateFinalModularity(community, nodeToSketch, wholeWeight, config.K)
	fmt.Printf("Final Modularity: %f\n", finalModularity)
	
	return nil
}

// Find neighboring communities for a node based on sketch similarity
func (ls *LouvainSCAR) findNeighboringCommunities(
	nodeId int64,
	nodeToSketch map[int64][]uint32,
	community []int64,
	k int64,
) map[int64]bool {
	neighborCommunities := make(map[int64]bool)
	nodeSketch := nodeToSketch[nodeId]
	
	// For each sketch value, find other nodes with same sketch value
	for _, sketchValue := range nodeSketch {
		// Find all nodes that have this sketch value
		for otherNodeId, otherSketch := range nodeToSketch {
			if otherNodeId == nodeId {
				continue
			}
			
			// Check if other node has this sketch value
			for _, otherSketchValue := range otherSketch {
				if sketchValue == otherSketchValue {
					neighborCommunities[community[otherNodeId]] = true
					break
				}
			}
		}
	}
	
	return neighborCommunities
}

// Calculate modularity gain for moving a node (using E-function approximation)
func (ls *LouvainSCAR) calculateModularityGain(
	nodeId, fromComm, toComm int64,
	nodeToSketch map[int64][]uint32,
	community []int64,
	wholeWeight float64,
	k, nk int64,
) float64 {
	nodeSketch := nodeToSketch[nodeId]
	
	// Estimate degree of the node
	nodeDegree := ls.estimateNodeDegree(nodeSketch, k, nk)
	
	// Estimate edges to/from communities
	edgesToFrom := ls.estimateEdgesToCommunity(nodeId, fromComm, nodeToSketch, community, k)
	edgesToTo := ls.estimateEdgesToCommunity(nodeId, toComm, nodeToSketch, community, k)
	
	// Estimate community degrees
	fromCommDegree := ls.estimateCommunityDegree(fromComm, nodeToSketch, community, k, nk)
	toCommDegree := ls.estimateCommunityDegree(toComm, nodeToSketch, community, k, nk)
	
	// Calculate gain similar to traditional Louvain but with sketch estimates
	// Traditional: gain = (k_i_in_new - k_i_in_old) - k_i * (sum_tot_new - sum_tot_old) / (2*m)
	// SCAR version: use E-function style calculation
	
	gainFrom := float64(edgesToFrom) - (nodeDegree * fromCommDegree) / (2 * wholeWeight)
	gainTo := float64(edgesToTo) - (nodeDegree * toCommDegree) / (2 * wholeWeight)
	
	return gainTo - gainFrom
}

// Helper functions for sketch-based estimations
func (ls *LouvainSCAR) estimateNodeDegree(nodeSketch []uint32, k, nk int64) float64 {
	if len(nodeSketch) == 0 {
		return 1.0
	}
	
	// Estimate degree from sketch (similar to original SCAR logic)
	totalDegree := 0.0
	sketches := make([][]uint32, nk)
	
	// Group sketches by nk
	for i, sketch := range nodeSketch {
		layerIdx := int64(i) / k
		if layerIdx < nk {
			sketches[layerIdx] = append(sketches[layerIdx], sketch)
		}
	}
	
	for _, layerSketch := range sketches {
		if len(layerSketch) > 0 {
			maxSketch := layerSketch[len(layerSketch)-1]
			if maxSketch > 0 && int64(len(layerSketch)) >= k-1 {
				degree := float64(uint32(0xFFFFFFFF)) / float64(maxSketch) * float64(k-1)
				totalDegree += degree
			} else {
				totalDegree += float64(len(layerSketch))
			}
		}
	}
	
	return totalDegree / float64(nk)
}

func (ls *LouvainSCAR) estimateEdgesToCommunity(
	nodeId, targetComm int64,
	nodeToSketch map[int64][]uint32,
	community []int64,
	k int64,
) float64 {
	nodeSketch := nodeToSketch[nodeId]
	edgeCount := 0.0
	
	// Count sketch intersections with nodes in target community
	for otherNodeId, otherSketch := range nodeToSketch {
		if community[otherNodeId] != targetComm || otherNodeId == nodeId {
			continue
		}
		
		// Count sketch intersections
		intersections := ls.countSketchIntersections(nodeSketch, otherSketch)
		edgeCount += float64(intersections) / float64(k) // Normalize by k
	}
	
	return edgeCount
}

func (ls *LouvainSCAR) estimateCommunityDegree(
	commId int64,
	nodeToSketch map[int64][]uint32,
	community []int64,
	k, nk int64,
) float64 {
	totalDegree := 0.0
	
	for nodeId, nodeSketch := range nodeToSketch {
		if community[nodeId] == commId {
			totalDegree += ls.estimateNodeDegree(nodeSketch, k, nk)
		}
	}
	
	return totalDegree
}

func (ls *LouvainSCAR) countSketchIntersections(sketch1, sketch2 []uint32) int {
	intersections := 0
	i, j := 0, 0
	
	for i < len(sketch1) && j < len(sketch2) {
		if sketch1[i] == sketch2[j] {
			intersections++
			i++
			j++
		} else if sketch1[i] < sketch2[j] {
			i++
		} else {
			j++
		}
	}
	
	return intersections
}

// Initialize graph and compute sketches (reuse existing logic)
func (ls *LouvainSCAR) initializeGraph(config SCARConfig) (*GraphStructure, []uint32, []uint32, float64, error) {
	// Read graph
	graph, err := ls.graphReader.ReadFromFile(config.GraphFile)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	
	n := graph.n
	pathLength := int64(10)
	
	// Initialize sketch arrays
	oldSketches := make([]uint32, pathLength*n*config.K*config.NK)
	for i := range oldSketches {
		oldSketches[i] = 0xFFFFFFFF // math.MaxUint32
	}
	
	nodeHashValue := make([]uint32, n*config.NK)
	
	// Read properties and path
	vertexProperties, err := ls.fileReader.ReadProperties(config.PropertyFile, n)
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
	}
	
	path, actualPathLength, err := ls.fileReader.ReadPath(config.PathFile)
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
	}
	pathLength = actualPathLength
	
	// Compute sketches
	ls.sketchComputer.ComputeForGraph(graph, oldSketches, path, pathLength, vertexProperties, nodeHashValue, config.K, config.NK)
	
	sketches := oldSketches[(pathLength-1)*n*config.K*config.NK:]
	
	// Add 1 to all sketches (original logic)
	for i := range sketches {
		if sketches[i] != 0xFFFFFFFF {
			sketches[i]++
		}
	}
	
	// Calculate whole weight
	wholeWeight := 0.0
	for i := int64(0); i < n; i++ {
		if nodeHashValue[i*config.NK] != 0 {
			degree := ls.estimateNodeDegreeFromSketches(i, sketches, config.K, config.NK, n)
			wholeWeight += degree
		}
	}
	wholeWeight /= 2.0
	
	return graph, sketches, nodeHashValue, wholeWeight, nil
}

func (ls *LouvainSCAR) estimateNodeDegreeFromSketches(nodeId int64, sketches []uint32, k, nk, n int64) float64 {
	totalDegree := 0.0
	
	for j := int64(0); j < nk; j++ {
		flag := false
		var currentSketch uint32 = 0
		
		for ki := int64(0); ki < k; ki++ {
			sketchVal := sketches[j*n*k+nodeId*k+ki]
			if sketchVal != 0 {
				currentSketch = sketchVal
			} else {
				flag = true
				totalDegree += float64(ki - 1)
				break
			}
		}
		
		if !flag && currentSketch != 0 {
			degree := float64(0xFFFFFFFF) / float64(currentSketch) * float64(k-1)
			totalDegree += degree
		}
	}
	
	return totalDegree / float64(nk)
}

// Create mapping from node ID to its sketch values
func (ls *LouvainSCAR) createNodeSketchMapping(n int64, sketches []uint32, nodeHashValue []uint32, k, nk int64) map[int64][]uint32 {
	nodeToSketch := make(map[int64][]uint32)
	
	for i := int64(0); i < n; i++ {
		// Only include nodes that have valid sketches
		if nodeHashValue[i*nk] != 0 && len(sketches) > int(i*k+1) && sketches[i*k+1] != 0 {
			var nodeSketch []uint32
			
			// Collect all sketch values for this node across all nk layers
			for j := int64(0); j < nk; j++ {
				for ki := int64(0); ki < k; ki++ {
					idx := j*n*k + i*k + ki
					if int(idx) < len(sketches) && sketches[idx] != 0 {
						nodeSketch = append(nodeSketch, sketches[idx])
					}
				}
			}
			
			if len(nodeSketch) > 0 {
				nodeToSketch[i] = nodeSketch
			}
		}
	}
	
	return nodeToSketch
}

// Aggregate communities into super-nodes (Phase 2 of Louvain)
func (ls *LouvainSCAR) aggregateCommunities(
	graph *GraphStructure, 
	nodeToSketch map[int64][]uint32, 
	community []int64,
	k, nk int64,
) (*GraphStructure, map[int64][]uint32, []int64) {
	// Create community mapping
	communityNodes := make(map[int64][]int64)
	for nodeId, comm := range community {
		communityNodes[comm] = append(communityNodes[comm], int64(nodeId))
	}
	
	// Create new graph where each community becomes a super-node
	newNodeCount := int64(len(communityNodes))
	newGraph := &GraphStructure{
		V: make([]SymmetricVertex, newNodeCount),
		n: newNodeCount,
		m: 0,
	}
	
	newNodeToSketch := make(map[int64][]uint32)
	newCommunity := make([]int64, newNodeCount)
	
	// Map old community IDs to new node IDs
	commToNewNode := make(map[int64]int64)
	newNodeId := int64(0)
	for oldComm := range communityNodes {
		commToNewNode[oldComm] = newNodeId
		newCommunity[newNodeId] = newNodeId // Each super-node starts in its own community
		newNodeId++
	}
	
	// Create super-node sketches by merging constituent node sketches
	for oldComm, nodes := range communityNodes {
		newId := commToNewNode[oldComm]
		var mergedSketch []uint32
		
		// Merge sketches from all nodes in this community
		for _, nodeId := range nodes {
			if sketch, exists := nodeToSketch[nodeId]; exists {
				mergedSketch = append(mergedSketch, sketch...)
			}
		}
		
		// Sort and deduplicate
		if len(mergedSketch) > 0 {
			// Simple sort and limit to k*nk values
			// In a more sophisticated version, you'd properly merge sketches
			if len(mergedSketch) > int(k*nk) {
				mergedSketch = mergedSketch[:k*nk]
			}
			newNodeToSketch[newId] = mergedSketch
		}
	}
	
	// Build super-graph edges based on sketch similarities
	// This is simplified - in practice you'd track actual inter-community edges
	for newId1 := int64(0); newId1 < newNodeCount; newId1++ {
		var neighbors []int64
		
		if sketch1, exists := newNodeToSketch[newId1]; exists {
			for newId2 := newId1 + 1; newId2 < newNodeCount; newId2++ {
				if sketch2, exists := newNodeToSketch[newId2]; exists {
					// If sketches intersect, create edge
					if ls.countSketchIntersections(sketch1, sketch2) > 0 {
						neighbors = append(neighbors, newId2)
						// Add reverse edge
						newGraph.V[newId2].neighbors = append(newGraph.V[newId2].neighbors, newId1)
						newGraph.V[newId2].degree++
					}
				}
			}
		}
		
		newGraph.V[newId1].neighbors = neighbors
		newGraph.V[newId1].degree = int64(len(neighbors))
		newGraph.m += int64(len(neighbors))
	}
	
	return newGraph, newNodeToSketch, newCommunity
}

// Calculate final modularity
func (ls *LouvainSCAR) calculateFinalModularity(
	community []int64,
	nodeToSketch map[int64][]uint32,
	wholeWeight float64,
	k int64,
) float64 {
	// Group nodes by community
	communityNodes := make(map[int64][]int64)
	for nodeId, comm := range community {
		if _, hasSketch := nodeToSketch[int64(nodeId)]; hasSketch {
			communityNodes[comm] = append(communityNodes[comm], int64(nodeId))
		}
	}
	
	totalModularity := 0.0
	
	for _, nodes := range communityNodes {
		if len(nodes) <= 1 {
			continue
		}
		
		// Calculate internal edges and total degree for this community
		internalEdges := 0.0
		totalDegree := 0.0
		
		for i, nodeId1 := range nodes {
			sketch1 := nodeToSketch[nodeId1]
			nodeDegree := ls.estimateNodeDegree(sketch1, k, 4) // Assume nk=4
			totalDegree += nodeDegree
			
			// Count internal edges
			for j := i + 1; j < len(nodes); j++ {
				nodeId2 := nodes[j]
				sketch2 := nodeToSketch[nodeId2]
				intersections := ls.countSketchIntersections(sketch1, sketch2)
				internalEdges += float64(intersections) / float64(k)
			}
		}
		
		// Modularity contribution: (internal edges / total edges) - (degree sum / 2*total edges)^2
		if wholeWeight > 0 {
			deltaQ := (internalEdges / wholeWeight) - (totalDegree / (2 * wholeWeight)) * (totalDegree / (2 * wholeWeight))
			totalModularity += deltaQ
		}
	}
	
	return totalModularity
}
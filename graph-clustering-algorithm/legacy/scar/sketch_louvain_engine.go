package scar

import (
	"fmt"
	"math"
	// "time"
	"sort"
)

// SketchLouvainEngine implements Louvain algorithm with sketch-based operations
type SketchLouvainEngine struct {
	sketchLouvainState *SketchLouvainState
	graph              *GraphStructure
	config             SCARConfig
	result             *SketchLouvainResult
}

func NewSketchLouvainEngine(config SCARConfig) *SketchLouvainEngine {
	return &SketchLouvainEngine{
		config: config,
	}
}

// RunLouvain executes the sketch-based Louvain algorithm
func (sle *SketchLouvainEngine) RunLouvain() error {

	// Phase 1: Initialize graph, sketches and result
	err := sle.InitializeGraphAndSketches()

	if err != nil {
		return err
	}

	// // print degree of each node
	// fmt.Println("Node degrees:")
	// for nodeId := int64(0); nodeId < sle.sketchLouvainState.n; nodeId++ {
	// 	if sketch := sle.sketchLouvainState.GetVertexSketch(nodeId); sketch != nil {
	// 		// Estimate degree using sketch cardinality	
	// 		estimatedDegree := sketch.EstimateCardinality()
	// 		fmt.Printf("Node sketch %d: %s\n", nodeId, sketch.String())
	// 		fmt.Printf("Node %d: estimated degree %.4f\n", nodeId, estimatedDegree)
	// 	} else {		
	// 		fmt.Printf("Node %d: no sketch available\n", nodeId)
	// 	}
	// }

	// Phase 2: Run Louvain algorithm
	// startTime := time.Now()



	// Run Louvain phases
	totalImprovement := true
	phase := 0

	for totalImprovement && phase < 3 {
		// fmt.Printf("=== Louvain Phase %d ===\n", phase)
		// sle.sketchLouvainState.PrintState(fmt.Sprintf("Phase %d", phase), "SKETCHES")

		// Phase 1: Local optimization
		totalImprovement = false
		improvement := true
		localIter := 0

		for improvement && localIter < 10 {
			improvement = false
			nodesProcessed := 0
			nodesMoved := 0

			// For each node, try to move to best neighboring community
			for nodeId := int64(0); nodeId < sle.sketchLouvainState.n; nodeId++ {
				// Skip nodes without sketches
				if sle.sketchLouvainState.GetVertexSketch(nodeId) == nil {
					continue
				}

				nodesProcessed++

				currentCommunity := sle.sketchLouvainState.GetNodeCommunity(nodeId)

				// Find neighboring communities through sketch similarity
				neighborCommunities := sle.sketchLouvainState.FindNeighboringCommunities(nodeId)

// ======================================= NON-DETERMINISTIC ======================================
																// if len(neighborCommunities) == 0 {
																// 	continue
																// }

																// // Calculate gain for moving to each neighboring community
																// bestCommunity := currentCommunity
																// bestGain := sle.calculateModularityGain(nodeId, currentCommunity)

																// for neighborComm, weight := range neighborCommunities {
																// 	if neighborComm == currentCommunity {
																// 		continue
																// 	}

																// 	// Calculate modularity gain using sketch-based estimation
																// 	gain := sle.calculateModularityGain(nodeId, neighborComm, weight)

																// 	if gain > bestGain {
																// 		bestGain = gain
																// 		bestCommunity = neighborComm
																// 	}
																// }
// ====================================== DETERMINISTIC FOR REPRODUCIBILITY ======================================
	// Sort communities for deterministic processing
var sortedCommunities []int64
for commId := range neighborCommunities {
    sortedCommunities = append(sortedCommunities, commId)
}
sort.Slice(sortedCommunities, func(i, j int) bool {
    return sortedCommunities[i] < sortedCommunities[j]
})

// Calculate gain for moving to each neighboring community
bestCommunity := currentCommunity
// bestGain := sle.calculateModularityGain(nodeId, currentCommunity, neighborCommunities[currentCommunity]) 

bestGain := 0.0
// fallback 
if (sle.sketchLouvainState.GetVertexSketch(nodeId).IsSketchFull()) {
	bestGain = sle.calculateModularityGain(nodeId, currentCommunity, neighborCommunities[currentCommunity]) 
} else {
	bestGain = sle.calculateModularityGain(nodeId, currentCommunity, neighborCommunities[currentCommunity])
}

// fmt.Printf("Node %d: current community %d, best gain so far: %f\n", nodeId, currentCommunity, bestGain)
for _, neighborComm := range sortedCommunities {
	if neighborComm == currentCommunity {
		continue
	}


    // Use the weight from FindNeighboringCommunities
    edgesToComm := neighborCommunities[neighborComm]
	
    gain := sle.calculateModularityGain(nodeId, neighborComm, edgesToComm)
	// fmt.Printf("Node %d: neighbor community %d, gain: %f\n", nodeId, neighborComm, gain)

    if gain > bestGain {
        bestGain = gain
        bestCommunity = neighborComm
    }
}
// ===================================== END OF DETERMINISTIC ======================================

				// Move node if beneficial
				if bestCommunity != currentCommunity {
					// fmt.Printf("Node %d: moving from community %d to %d (gain: %f)\n", nodeId, currentCommunity, bestCommunity, bestGain)
					sle.sketchLouvainState.MoveNode(nodeId, currentCommunity, bestCommunity)
					totalImprovement = true
					improvement = true
					nodesMoved++
				}
			}

			fmt.Printf("Local iteration %d: %d nodes processed, %d moved\n", localIter, nodesProcessed, nodesMoved)
			localIter++
			// fmt.Printf("Local iter %d: %d nodes processed, %d moved\n", localIter, nodesProcessed, nodesMoved)
		}

		// fmt.Printf("\n=== HIERARCHY LEVEL %d END ===\n", phase)
		// fmt.Printf("Total improvement achieved: %t\n", totalImprovement)

		// fmt.Printf("Phase %d: Improvement %t\n", phase, improvement)

		// Phase 2: Community aggregation (create super-graph)
		
		if !totalImprovement {
			if true {
				// Force all nodes into a single community
				fmt.Println("No improvement in phase 0, forcing all nodes into a single community...")

				singleCommunityId := int64(0)
				var allNodes []int64

				// Collect all nodes that currently belong to any community
				for nodeId := int64(0); nodeId < sle.sketchLouvainState.n; nodeId++ {
					if sle.sketchLouvainState.GetVertexSketch(nodeId) == nil {
						continue
					}
					allNodes = append(allNodes, nodeId)
					sle.sketchLouvainState.nodeToCommunity[nodeId] = singleCommunityId
				}

				// Replace community map
				sle.sketchLouvainState.communityToNodes = map[int64][]int64{
					singleCommunityId: allNodes,
				}

				// Reset active communities
				sle.sketchLouvainState.activeCommunities = map[int64]bool{
					singleCommunityId: true,
				}

				// Update the community sketch for this single community
				sle.sketchLouvainState.sketchManager.UpdateCommunitySketch(singleCommunityId, allNodes)

				// Now aggregate as usual
				sle.aggregateCommunities()

				break
			} 
			
		} 

		sle.aggregateCommunities()
		
		phase++
	}

	// fmt.Printf("Louvain completed in %v\n", time.Since(startTime))
	// fmt.Printf("Total hierarchy levels: %d\n", phase)


	return sle.result.WriteFiles(sle.config)
}

// InitializeGraphAndSketches reads graph and computes initial sketches
func (sle *SketchLouvainEngine) InitializeGraphAndSketches() error {
	// Read graph
	graphReader := NewGraphReader()
	graph, err := graphReader.ReadFromFile(sle.config.GraphFile)
	if err != nil {
		return err
	}
	sle.graph = graph
	propK := sle.config.K + 1

	n := graph.n
	pathLength := int64(10)

	// Initialize flat sketch arrays (for compatibility with existing sketch computation)
	oldSketches := make([]uint32, pathLength*n*propK*sle.config.NK)
	for i := range oldSketches {
		oldSketches[i] = math.MaxUint32
	}

	nodeHashValue := make([]uint32, n*sle.config.NK)

	// Read properties and path
	fileReader := NewFileReader()
	// fmt.Printf("About to read properties from: '%s'\n", sle.config.PropertyFile)
	// fmt.Printf("About to read path from: '%s'\n", sle.config.PathFile)

	vertexProperties, err := fileReader.ReadProperties(sle.config.PropertyFile, n)
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
	}

	path, actualPathLength, err := fileReader.ReadPath(sle.config.PathFile)
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
	}
	pathLength = actualPathLength

	// Compute sketches using existing logic
	sketchComputer := NewSketchComputer(graph.n)
	sketchComputer.ComputeForGraph(graph, oldSketches, path, pathLength, vertexProperties, nodeHashValue, propK, sle.config.NK, sle.config.NumWorkers)

	sketches := oldSketches[(pathLength-1)*n*propK*sle.config.NK:]

	// Add 1 to all sketches (original logic)
	for i := range sketches {
		if sketches[i] != math.MaxUint32 {
			sketches[i]++
		}
	}

	// Convert flat arrays to VertexBottomKSketch objects
	newSketchManager := sle.convertPropSketchesToFinal(sketches, nodeHashValue, n, propK)

	// Initialize community manager
	sle.sketchLouvainState = NewSketchLouvainState(sle.graph.n, newSketchManager)

	// // Print final sketch state
	// fmt.Println("Here are the vertex sketches and their cardinalities:")
	// for i := int64(0); i < n; i++ {
	// 	if nodeHashValue[i*sle.config.NK] != 0 {
	// 		if sketch := sle.sketchLouvainState.GetVertexSketch(i); sketch != nil {
	// 			fmt.Printf("Vertex %d: %s\n", i, sketch.String())
	// 			fmt.Printf("  Estimated cardinality: %.4f\n", sketch.EstimateCardinality())
	// 		}
	// 	}
	// }
	// fmt.Println()

	// Initialize communities (each node in its own community)
	for nodeId := int64(0); nodeId < n; nodeId++ {
		// Skip nodes without sketches
		if sle.sketchLouvainState.GetVertexSketch(nodeId) == nil {
			sle.sketchLouvainState.nodeToCommunity[nodeId] = -1 // Not in any community
			continue
		}

		// Each node starts in its own community
		sle.sketchLouvainState.nodeToCommunity[nodeId] = nodeId
		sle.sketchLouvainState.communityToNodes[nodeId] = []int64{nodeId}
		sle.sketchLouvainState.activeCommunities[nodeId] = true

		// Create initial community sketch (copy of node sketch)
		sle.sketchLouvainState.sketchManager.UpdateCommunitySketch(nodeId, []int64{nodeId})
	}

	
	sle.sketchLouvainState.BuildSketchAdjacencyList(sle.config.NumWorkers)
	
	sle.sketchLouvainState.CalculateWholeWeight()
	sle.result = NewSketchLouvainResult()

	// Print final sketch state
	// fmt.Println("\n=== FINAL SKETCHES (after +1) ===")
	// for i := int64(0); i < n; i++ {
	// 	if nodeHashValue[i*sle.config.NK] != 0 {
	// 		if sketch := sle.sketchLouvainState.GetVertexSketch(i); sketch != nil {
	// 			fmt.Println(sketch.String())
	// 		}
	// 	}
	// }
	// fmt.Println()

	return nil
}


// convertPropSketchesToFinal converts sketches generated during propagation
// at size propK = K+1 into final bottom-k sketches of size sle.config.K.
// During the conversion we drop the node's own seed hash(es) so they don't
// consume sketch slots. Because propagation maintained sorted order, we can
// take the first K non-self values per layer.
func (sle *SketchLouvainEngine) convertPropSketchesToFinal(
    propSketches []uint32,
    nodeHashValue []uint32,
    n int64,
    propK int64,
) *SketchManager {
    finalK := sle.config.K
    nk := sle.config.NK

    mgr := NewSketchManager(finalK, nk)

    // Pre-build per-node self-hash set (small; nk is tiny, so map cost is low).
    selfHashes := make([]map[uint32]struct{}, n)
    for i := int64(0); i < n; i++ {
        m := make(map[uint32]struct{}, nk)
        for j := int64(0); j < nk; j++ {
            hv := nodeHashValue[i*nk+j]
            if hv != 0 {
                m[hv] = struct{}{}
            }
        }
        selfHashes[i] = m
    }

    for i := int64(0); i < n; i++ {
        // Skip nodes that were never seeded (no hash in layer 0).
        if nodeHashValue[i*nk] == 0 {
            continue
        }

        vsk := NewVertexBottomKSketch(i, finalK, nk)

        for j := int64(0); j < nk; j++ {
            // Collect candidates from propagated sketch (propK wide).
            out := make([]uint32, 0, finalK)
            base := j*n*propK + i*propK

            for ki := int64(0); ki < propK; ki++ {
                idx := base + ki
                if int(idx) >= len(propSketches) {
                    break
                }
                v := propSketches[idx]
                if v == math.MaxUint32 {
                    break // remaining are empty
                }
                // Drop self-hash.
                if _, isSelf := selfHashes[i][v]; isSelf {
                    continue
                }
                // Optional tiny dedup within layer (comment out if unwanted):
                if len(out) > 0 && v == out[len(out)-1] {
                    continue
                }
                out = append(out, v)
                if len(out) == int(finalK) {
                    break
                }
            }

            // Pad if under-filled.
            for len(out) < int(finalK) {
                out = append(out, math.MaxUint32)
            }

            // Install into the sketch.
            copy(vsk.sketches[j], out)
        }

        vsk.UpdateFilledCount()

        if vsk.GetFilledCount() > 0 {
            mgr.vertexSketches[i] = vsk
        }

        // Rebuild hash→node map from node seeds (what others will see).
        for j := int64(0); j < nk; j++ {
            hv := nodeHashValue[i*nk+j]
            if hv != 0 {
                mgr.hashToNodeMap[hv] = i
            }
        }
    }

    return mgr
}


// // convertToVertexSketches converts flat arrays to VertexBottomKSketch objects
// func (sle *SketchLouvainEngine) convertToVertexSketches(sketches []uint32, nodeHashValue []uint32, n int64) *SketchManager {
// 	newSketchManager := NewSketchManager(sle.config.K, sle.config.NK)

// 	for i := int64(0); i < n; i++ {
// 		if nodeHashValue[i*sle.config.NK] != 0 {
// 			// Create vertex sketch
// 			sketch := NewVertexBottomKSketch(i, sle.config.K, sle.config.NK)

// 			// Prepare all sketch data at once
// 			allSketchData := make([][]uint32, sle.config.NK)
// 			for j := int64(0); j < sle.config.NK; j++ {
// 				layerSketch := make([]uint32, sle.config.K)
// 				for ki := int64(0); ki < sle.config.K; ki++ {
// 					idx := j*n*sle.config.K + i*sle.config.K + ki
// 					if int(idx) < len(sketches) {
// 						layerSketch[ki] = sketches[idx]
// 					} else {
// 						layerSketch[ki] = math.MaxUint32
// 					}
// 				}
// 				allSketchData[j] = layerSketch
// 			}

// 			// Set it all at once
// 			sketch.SetCompleteSketch(allSketchData)

// 			// Build hash to node mapping
// 			for j := int64(0); j < sle.config.NK; j++ {
// 				hashValue := nodeHashValue[i*sle.config.NK+j]
// 				if hashValue != 0 {
// 					newSketchManager.hashToNodeMap[hashValue] = i
// 				}
// 			}

// 			// Remove node's own hash values from its sketch
// 			nodeOwnHashes := make(map[uint32]bool)
// 			for j := int64(0); j < sle.config.NK; j++ {
// 				ownHashValue := nodeHashValue[i*sle.config.NK+j]
// 				if ownHashValue != 0 {
// 					nodeOwnHashes[ownHashValue] = true
// 				}
// 			}

// 			// Clean each layer's sketch
// 			for j := int64(0); j < sle.config.NK; j++ {
// 				cleanedSketch := make([]uint32, sle.config.K)
// 				cleanIdx := 0

// 				for ki := int64(0); ki < sle.config.K; ki++ {
// 					val := sketch.sketches[j][ki]
// 					if val != math.MaxUint32 && !nodeOwnHashes[val] {
// 						if cleanIdx < int(sle.config.K) {
// 							cleanedSketch[cleanIdx] = val
// 							cleanIdx++
// 						}
// 					}
// 				}

// 				// Fill remaining with MaxUint32
// 				for cleanIdx < int(sle.config.K) {
// 					cleanedSketch[cleanIdx] = math.MaxUint32
// 					cleanIdx++
// 				}

// 				sketch.sketches[j] = cleanedSketch
// 			}

// 			sketch.UpdateFilledCount()

// 			// fmt.Println("Created sketch for node", i, ":", sketch.String())
// 			if sketch.GetFilledCount() > 0 {
// 				newSketchManager.vertexSketches[i] = sketch
// 			}
// 		}
// 	}
// 	return newSketchManager
// }


// calculateModularityGain calculates modularity gain for moving a node
func (sle *SketchLouvainEngine) calculateModularityGain(
	nodeId, toComm int64, edgesToTo float64,
) float64 {
	sketch := sle.sketchLouvainState.GetVertexSketch(nodeId)
	if sketch == nil {
		return 0.0
	}

	// Get total weight of the graph
	wholeWeight := sle.sketchLouvainState.GetTotalWeight()

	// Estimate degree of the node
	nodeDegree := sle.sketchLouvainState.EstimateCardinality(nodeId)

	// edgesToFrom := sle.sketchLouvainState.EstimateEdgesToCommunity(nodeId, sle.sketchLouvainState.GetNodeCommunity(nodeId))
	// fromCommDegree := sle.sketchLouvainState.EstimateCommunityCardinality(sle.sketchLouvainState.GetNodeCommunity(nodeId))

	// Estimate community degrees
	toCommDegree := sle.sketchLouvainState.EstimateCommunityCardinality(toComm)

    // // gain := edgesToTo - nodeDegree * toCommDegree / (2 * wholeWeight) 
	// gain := edgesToTo - edgesToFrom + nodeDegree * (fromCommDegree - toCommDegree - nodeDegree) / (2 * wholeWeight)
	// // // Print all component for debugging
	// fmt.Printf("Moving node %d to community %d: edgesToTo: %.4f, edgesToFrom: %.4f, nodeDegree: %.4f, "+
	// 	" fromCommDegree: %.4f, toCommDegree: %.4f, wholeWeight: %.4f, gain: %.4f\n",
	// 	nodeId, toComm, edgesToTo, edgesToFrom,	 nodeDegree, fromCommDegree, toCommDegree, wholeWeight, gain)

	gain := edgesToTo - nodeDegree * toCommDegree / (2 * wholeWeight)
	// Print all components for debugging
	if nodeId < 10 {
		fmt.Printf("Moving node %d to community %d: edgesToTo: %.4f, "+
			" nodeDegree: %.4f, toCommDegree: %.4f, wholeWeight: %.4f, gain: %.4f\n",
			nodeId, toComm, edgesToTo, nodeDegree, toCommDegree, wholeWeight, gain)		
	}


	
	return gain
}

func (sle *SketchLouvainEngine) aggregateCommunities() error {
	// fmt.Println("\n=== COMMUNITY AGGREGATION PHASE ===")

	// Step 1: Get current state before aggregation
	prevState := sle.sketchLouvainState  // SAVE THE PREVIOUS STATE HERE
	communityNodes := prevState.GetCommunityToNodesMap()
	
	// Step 2: Create mapping from old community IDs to new super-node IDs
	commToNewNode := make(map[int64]int64)
	newNodeId := int64(0)
// ===================================================== NON-DETERMINISTIC =====================================================
										// for oldComm := range communityNodes {
										// 	commToNewNode[oldComm] = newNodeId
										// 	newNodeId++
										// }
// ===================================================== DETERMINISTIC FOR REPRODUCIBILITY =====================================================
										// Sort communities before assigning super-node IDs
										var sortedComms []int64
										for commId := range communityNodes {
											sortedComms = append(sortedComms, commId)
										}
										sort.Slice(sortedComms, func(i, j int) bool { return sortedComms[i] < sortedComms[j] })

										for _, oldComm := range sortedComms {  // Instead of: for oldComm := range communityNodes
											commToNewNode[oldComm] = newNodeId
											newNodeId++
										}
// ==============================

	// Step 3: Record the current level BEFORE aggregation (with the mapping)
	sle.recordCurrentLevel(communityNodes, commToNewNode, sle.sketchLouvainState.sketchAdjacencyList)

	newNodeCount := int64(len(communityNodes))

	// Step 4: Create new SketchManager for super-graph
	newSketchManager := NewSketchManager(sle.config.K, sle.config.NK)

	// Step 5: Create super-node sketches
	superNodeSketches := make(map[int64]*VertexBottomKSketch)

	for oldComm, _ := range communityNodes {
		newId := commToNewNode[oldComm]

		communitySketch := prevState.GetCommunitySketch(oldComm)  // USE prevState HERE TOO
		communitySketch.nodeId = newId

		superNodeSketches[newId] = communitySketch
		newSketchManager.vertexSketches[newId] = communitySketch
	}

	// Step 6: Rebuild hash-to-node mapping for super-nodes
	newHashToNodeMap := make(map[uint32]int64)
	for hash, originalNodeId := range prevState.sketchManager.hashToNodeMap {  // USE prevState HERE TOO
		oldComm := prevState.GetNodeCommunity(originalNodeId)
		if newId, exists := commToNewNode[oldComm]; exists {
			newHashToNodeMap[hash] = newId
		}
	}

	newSketchManager.hashToNodeMap = newHashToNodeMap

	// Step 7: Create NEW sketchLouvainState for super-graph (level N+1)
	sle.sketchLouvainState = NewSketchLouvainState(newNodeCount, newSketchManager)

	// Step 8: Initialize super-node communities
	// Each super-node starts in its own community at the new level
	for superNodeId := int64(0); superNodeId < newNodeCount; superNodeId++ {
		// Initialize community membership
		sle.sketchLouvainState.nodeToCommunity[superNodeId] = superNodeId
		sle.sketchLouvainState.communityToNodes[superNodeId] = []int64{superNodeId}
		sle.sketchLouvainState.activeCommunities[superNodeId] = true

		// Create community sketch for this single-member community
		// (This copies the super-node sketch as the community sketch)
		sle.sketchLouvainState.sketchManager.UpdateCommunitySketch(superNodeId, []int64{superNodeId})
	}

	// Step 9: Build adjacency list for the super-graph
	err := sle.sketchLouvainState.BuildSuperNodeAdjacencyFromPreviousLevel(prevState, commToNewNode)  // PASS prevState
	if err != nil {
		return fmt.Errorf("failed to build super-node adjacency: %w", err)
	}

	// Step 10: Calculate total weight of the super-graph
	sle.sketchLouvainState.CalculateWholeWeight()

	// fmt.Println("Super-graph community manager initialized")

	return nil
}

// recordCurrentLevel records the current level state
func (sle *SketchLouvainEngine) recordCurrentLevel(
	communityToNodes map[int64][]int64,
	commToNewNode map[int64]int64,
	sketchAdjacencyList map[int64][]WeightedEdge,
) {
	community := sle.sketchLouvainState.GetNodesToCommunityMap()
	sketches := sle.sketchLouvainState.GetAllVertexSketches()
	hashMap := sle.sketchLouvainState.sketchManager.hashToNodeMap

	sle.result.AddLevel(community, sketches, hashMap, communityToNodes, commToNewNode, sketchAdjacencyList)
}



// GetSketchLouvainState returns the current state of the SketchLouvainEngine
func (sle *SketchLouvainEngine) GetSketchLouvainState() *SketchLouvainState {
	return sle.sketchLouvainState
}
package scar

import (
	"fmt"
	"time"
	"math"
)

// SketchLouvainEngine implements Louvain algorithm with sketch-based operations
type SketchLouvainEngine struct {
	sketchManager *SketchManager
	graph         *GraphStructure
	config        SCARConfig
	partitionTracker *MultiLevelPartitionTracker
	originalGraphSize int64
}

func NewSketchLouvainEngine(config SCARConfig) *SketchLouvainEngine {
	return &SketchLouvainEngine{
		sketchManager: NewSketchManager(config.K, config.NK),
		config:        config,
	}
}

// RunLouvain executes the sketch-based Louvain algorithm
func (sle *SketchLouvainEngine) RunLouvain() error {
	fmt.Println("Start reading graph")
	
	// Phase 1: Initialize graph and sketches
	err := sle.initializeGraphAndSketches()
	if err != nil {
		return err
	}
	
	n := sle.graph.n
	wholeWeight := sle.calculateWholeWeight()

	sle.originalGraphSize = n
	sle.partitionTracker = NewMultiLevelPartitionTracker(sle.originalGraphSize)
	
	// Phase 2: Run Louvain algorithm
	startTime := time.Now()
	community := make([]int64, n)
	
	// Initialize: each node in its own community
	for i := int64(0); i < n; i++ {
		community[i] = i
	}
	
	// Create node-to-sketch mapping
	nodeToSketch := sle.sketchManager.CreateNodeToSketchMapping()
	
	// Run Louvain phases
	totalImprovement := true
	phase := 0
	
	for totalImprovement && phase < 10 {
		fmt.Printf("=== Louvain Phase %d ===\n", phase)
		
		
		sle.printCommunityState("INITIAL", community, nodeToSketch, phase)

		// Phase 1: Local optimization
		totalImprovement = false
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
				neighborCommunities := sle.findNeighboringCommunities(nodeId, nodeToSketch, community)
				
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
					
					// Calculate modularity gain using sketch-based estimation
					gain := sle.calculateModularityGain(nodeId, currentCommunity, neighborComm, nodeToSketch, community, wholeWeight)
					if gain > bestGain {
						fmt.Printf("Node %d: Found better community %d with gain %f\n", nodeId, neighborComm, gain)
						bestGain = gain
						bestCommunity = neighborComm
					}
				}
				
				// Move node if beneficial
				if bestCommunity != currentCommunity {
					fmt.Printf("Node %d: Moving from community %d to %d with gain %f\n", nodeId, currentCommunity, bestCommunity, bestGain)
					community[nodeId] = bestCommunity
					totalImprovement = true
					improvement = true
					nodesMoved++
				}
			}
			
			localIter++
			fmt.Printf("Local iter %d: %d nodes processed, %d moved\n", localIter, nodesProcessed, nodesMoved)
		}

		// Print community mapping of nodes
		sle.printCommunityState("FINAL", community, nodeToSketch, phase)
		
		fmt.Printf("\n=== HIERARCHY LEVEL %d END ===\n", phase)
		fmt.Printf("Total improvement achieved: %t\n", totalImprovement)

		fmt.Printf("Phase %d: Improvement %t\n", phase, improvement)



		communityMap := make(map[int64][]int64)
		for nodeId, commId := range community {
			if _, exists := communityMap[commId]; !exists {
				communityMap[commId] = []int64{}
			}
			communityMap[commId] = append(communityMap[commId], int64(nodeId))
		}

		
		if !totalImprovement {
			break
		}
		
		// Phase 2: Community aggregation (create super-graph)
		sle.graph, nodeToSketch, community = sle.aggregateCommunities(nodeToSketch, community)
		n = sle.graph.n
		
		wholeWeight = sle.calculateWholeWeight()
		fmt.Printf("Updated whole weight after aggregation: %f\n", wholeWeight)
		
		phase++
	}
	
	fmt.Printf("Louvain completed in %v\n", time.Since(startTime))
	fmt.Printf("Total hierarchy levels: %d\n", phase)

	// Write results - trace back to original nodes if we did aggregation
	var finalPartition []int64
	// var outputSize int64

	if sle.partitionTracker.currentLevel > 0 {  
		// We did aggregation, need to trace back to original nodes
		fmt.Println("Tracing back partition to original nodes...")
		finalPartition = sle.partitionTracker.getFinalPartition(community)
		// outputSize = sle.originalGraphSize
	} else {
		// No aggregation happened, community array is already for original nodes
		fmt.Println("No aggregation occurred, using direct community mapping...")
		finalPartition = community
		// outputSize = n
	}

	// Write results using new output writer
	outputWriter := NewOutputWriter()
	
	// Call the new method that handles both modes
	err = outputWriter.WriteLouvainResults(
		sle.config,
		finalPartition,
		sle.partitionTracker,
		sle.graph,
		sle.sketchManager,
	)

	if err != nil {
		return fmt.Errorf("failed to write output: %v", err)
	}
	
	// Calculate final modularity
	finalModularity := sle.calculateFinalModularity(community, nodeToSketch, wholeWeight)
	fmt.Printf("Final Modularity: %f\n", finalModularity)

	
	sle.printHierarchySummary()
	
	return nil
}

// initializeGraphAndSketches reads graph and computes initial sketches
func (sle *SketchLouvainEngine) initializeGraphAndSketches() error {
	// Read graph
	graphReader := NewGraphReader()
	graph, err := graphReader.ReadFromFile(sle.config.GraphFile)
	if err != nil {
		return err
	}
	sle.graph = graph
	
	n := graph.n
	pathLength := int64(10)
	
	// Initialize flat sketch arrays (for compatibility with existing sketch computation)
	oldSketches := make([]uint32, pathLength*n*sle.config.K*sle.config.NK)
	for i := range oldSketches {
		oldSketches[i] = math.MaxUint32
	}
	
	nodeHashValue := make([]uint32, n*sle.config.NK)
	
	// Read properties and path
	fileReader := NewFileReader()
	fmt.Printf("About to read properties from: '%s'\n", sle.config.PropertyFile)
	fmt.Printf("About to read path from: '%s'\n", sle.config.PathFile)
	
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
	sketchComputer := NewSketchComputer()
	sketchComputer.ComputeForGraph(graph, oldSketches, path, pathLength, vertexProperties, nodeHashValue, sle.config.K, sle.config.NK)
	
	sketches := oldSketches[(pathLength-1)*n*sle.config.K*sle.config.NK:]
	
	// Add 1 to all sketches (original logic)
	for i := range sketches {
		if sketches[i] != math.MaxUint32 {
			sketches[i]++
		}
	}
	
	// Convert flat arrays to VertexBottomKSketch objects
	sle.convertToVertexSketches(sketches, nodeHashValue, n)
	
	// Print final sketch state
	fmt.Println("\n=== FINAL SKETCHES (after +1) ===")
	for i := int64(0); i < n; i++ {
		if nodeHashValue[i*sle.config.NK] != 0 {
			if sketch := sle.sketchManager.GetVertexSketch(i); sketch != nil {
				fmt.Println(sketch.String())
			}
		}
	}
	fmt.Println()
	
	return nil
}

// convertToVertexSketches converts flat arrays to VertexBottomKSketch objects
func (sle *SketchLouvainEngine) convertToVertexSketches(sketches []uint32, nodeHashValue []uint32, n int64) {
	for i := int64(0); i < n; i++ {
		if nodeHashValue[i*sle.config.NK] != 0 {
			// Extract layer values for hash mapping
			layerValues := make([]uint32, sle.config.NK)
			for j := int64(0); j < sle.config.NK; j++ {
				layerValues[j] = nodeHashValue[i*sle.config.NK+j] - 1 // Remove the +1 added during initialization
			}
			
			// Create vertex sketch
			sketch := NewVertexBottomKSketch(i, sle.config.K, sle.config.NK)
			
			// Fill sketch data from flat array
			for j := int64(0); j < sle.config.NK; j++ {
				layerSketch := make([]uint32, sle.config.K)
				for ki := int64(0); ki < sle.config.K; ki++ {
					idx := j*n*sle.config.K + i*sle.config.K + ki
					if int(idx) < len(sketches) {
						layerSketch[ki] = sketches[idx]
					} else {
						layerSketch[ki] = math.MaxUint32
					}
				}
				sketch.sketches[j] = layerSketch
			}
			
			sle.sketchManager.vertexSketches[i] = sketch
			
			// Build hash to node mapping
			for j := int64(0); j < sle.config.NK; j++ {
				hashValue := nodeHashValue[i*sle.config.NK+j]
				if hashValue != 0 {
					sle.sketchManager.hashToNodeMap[hashValue] = i
				}
			}

			// Remove node's own hash values from its sketch
			nodeOwnHashes := make(map[uint32]bool)
			for j := int64(0); j < sle.config.NK; j++ {
				ownHashValue := nodeHashValue[i*sle.config.NK+j]
				if ownHashValue != 0 {
					nodeOwnHashes[ownHashValue] = true
				}
			}

			// Clean each layer's sketch
			for j := int64(0); j < sle.config.NK; j++ {
				cleanedSketch := make([]uint32, sle.config.K)
				cleanIdx := 0
				
				for ki := int64(0); ki < sle.config.K; ki++ {
					val := sketch.sketches[j][ki]
					if val != math.MaxUint32 && !nodeOwnHashes[val] {
						if cleanIdx < int(sle.config.K) {
							cleanedSketch[cleanIdx] = val
							cleanIdx++
						}
					}
				}
				
				// Fill remaining with MaxUint32
				for cleanIdx < int(sle.config.K) {
					cleanedSketch[cleanIdx] = math.MaxUint32
					cleanIdx++
				}
				
				sketch.sketches[j] = cleanedSketch
			}

		}
	}
}

// calculateWholeWeight calculates total weight of the graph
func (sle *SketchLouvainEngine) calculateWholeWeight() float64 {
	wholeWeight := 0.0
	for nodeId, sketch := range sle.sketchManager.vertexSketches {
		
		degree := sketch.EstimateCardinality()
		wholeWeight += degree
		fmt.Printf("Node %d estimated degree: %f\n", nodeId, degree)
	}
	wholeWeight /= 2.0
	fmt.Printf("Total graph weight: %f\n", wholeWeight)
	return wholeWeight
}

// findNeighboringCommunities finds communities that are neighbors based on sketch similarity
func (sle *SketchLouvainEngine) findNeighboringCommunities(
	nodeId int64,
	nodeToSketch map[int64][]uint32,
	community []int64,
) map[int64]bool {
	neighborCommunities := make(map[int64]bool)
	nodeSketch := nodeToSketch[nodeId]
	
	sketch := sle.sketchManager.GetVertexSketch(nodeId)
	
	if sketch != nil && !sketch.IsSketchFull() {
		for _, sketchValue := range nodeSketch {
			if sketchValue != math.MaxUint32 {
				if otherNodeId, exists := sle.sketchManager.GetNodeFromHash(sketchValue); exists {
					if otherNodeId != nodeId {
						neighborCommunities[community[otherNodeId]] = true
					}
				}
			}
		}
		return neighborCommunities
	}
	
	// SKETCH METHOD: Use probabilistic intersection counting
	for otherNodeId, otherSketch := range nodeToSketch {
		if otherNodeId == nodeId {
			continue
		}
		
		// Check if other node has this sketch value
		for _, sketchValue := range nodeSketch {
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

// calculateModularityGain calculates modularity gain for moving a node
func (sle *SketchLouvainEngine) calculateModularityGain(
	nodeId, fromComm, toComm int64,
	nodeToSketch map[int64][]uint32,
	community []int64,
	wholeWeight float64,
) float64 {
	sketch := sle.sketchManager.GetVertexSketch(nodeId)
	if sketch == nil {
		return 0.0
	}
	
	// Estimate degree of the node
	nodeDegree := sketch.EstimateCardinality()
	
	// Estimate edges to/from communities
	edgesToFrom := sle.estimateEdgesToCommunity(nodeId, fromComm, nodeToSketch, community)
	edgesToTo := sle.estimateEdgesToCommunity(nodeId, toComm, nodeToSketch, community)
	
	// Estimate community degrees
	fromCommDegree := sle.estimateCommunityDegree(fromComm, nodeToSketch, community)
	toCommDegree := sle.estimateCommunityDegree(toComm, nodeToSketch, community)
	
	fmt.Printf("Node %d: edgesToFrom=%f, edgesToTo=%f, fromCommDegree=%f, toCommDegree=%f, wholeWeight=%f\n", 
		nodeId, edgesToFrom, edgesToTo, fromCommDegree, toCommDegree, wholeWeight)
	
	// Calculate gain similar to traditional Louvain but with sketch estimates
	gain := edgesToTo - edgesToFrom + (nodeDegree * (fromCommDegree - toCommDegree - nodeDegree) / (2 * wholeWeight))
	if gain < 0 {
		return 0.0
	}
	return gain / wholeWeight
}

// estimateEdgesToCommunity estimates edges from a node to a community
func (sle *SketchLouvainEngine) estimateEdgesToCommunity(
	nodeId, targetComm int64,
	nodeToSketch map[int64][]uint32,
	community []int64,
) float64 {
	nodeSketch := sle.sketchManager.GetVertexSketch(nodeId)
	if nodeSketch == nil {
		return 0.0
	}
	
	if !nodeSketch.IsSketchFull() {
		// EXACT METHOD: Direct counting
		edgeCount := 0.0
		nodeSketchLayer := nodeSketch.GetSketch(0)
		
		for _, sketchValue := range nodeSketchLayer {
			if sketchValue != math.MaxUint32 {
				if otherNodeId, exists := sle.sketchManager.GetNodeFromHash(sketchValue); exists {
					if otherNodeId != nodeId && community[otherNodeId] == targetComm {
						edgeCount += 1.0
					}
				}
			}
		}
		return edgeCount
	}
	
	// SKETCH METHOD: Inclusion-Exclusion
	// |sketch(node) ∩ sketch(community)| = |sketch(node)| + |sketch(community)| - |sketch(node) ∪ sketch(community)|
	
	// Step 1: Get node degree estimate
	nodeDegree := nodeSketch.EstimateCardinality()
	
	// Step 2: Get community degree estimate  
	communityDegree := sle.estimateCommunityDegree(targetComm, nodeToSketch, community)
	
	// Step 3: Create union sketch and estimate its cardinality
	communitySketch := sle.calculateCommunitySketch(targetComm, community)
	unionSketch := sle.createUnionSketch(nodeSketch, communitySketch)
	unionDegree := unionSketch.EstimateCardinality()
	
	// Step 4: Apply inclusion-exclusion
	intersectionSize := nodeDegree + communityDegree - unionDegree
	
	return math.Max(0, intersectionSize)  // Ensure non-negative
}

// estimateCommunityDegree estimates total degree of a community
func (sle *SketchLouvainEngine) estimateCommunityDegree(
	commId int64,
	nodeToSketch map[int64][]uint32,
	community []int64,
) float64 {
	// Step 1: Create merged community sketch
	communitySketch := sle.calculateCommunitySketch(commId, community)
	
	// Step 2: Estimate based on sketch fullness
	return communitySketch.EstimateCardinality()
}

// countSketchIntersections counts intersections between two sketches
func (sle *SketchLouvainEngine) countSketchIntersections(sketch1, sketch2 []uint32) int {
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

// aggregateCommunities creates super-graph from communities
func (sle *SketchLouvainEngine) aggregateCommunities(
	nodeToSketch map[int64][]uint32,
	community []int64,
) (*GraphStructure, map[int64][]uint32, []int64) {
	fmt.Println("\n=== COMMUNITY AGGREGATION PHASE ===")
	
	// Step 1: Create community mapping (which nodes belong to which community)
	communityNodes := make(map[int64][]int64)
	for nodeId, comm := range community {
		if _, hasSketch := nodeToSketch[int64(nodeId)]; hasSketch {
			communityNodes[comm] = append(communityNodes[comm], int64(nodeId))
		}
	}
	
	fmt.Printf("Found %d communities to aggregate\n", len(communityNodes))
	
	// Step 2: Create mapping from old community IDs to new super-node IDs
	commToNewNode := make(map[int64]int64)
	newNodeId := int64(0)
	for oldComm := range communityNodes {
		commToNewNode[oldComm] = newNodeId
		fmt.Printf("Community %d -> Super-node %d (contains %d nodes)\n", 
			oldComm, newNodeId, len(communityNodes[oldComm]))
		newNodeId++
	}
	
	newNodeCount := int64(len(communityNodes))
	
	// Step 3: Create new SketchManager for super-graph
	newSketchManager := NewSketchManager(sle.config.K, sle.config.NK)
	
	// Step 4: Create super-node sketches using PROPER Bottom-K union
	superNodeSketches := make(map[int64]*VertexBottomKSketch)
	
	for oldComm, nodes := range communityNodes {
		newId := commToNewNode[oldComm]
		
		fmt.Printf("\nCreating super-node %d from community %d (%d nodes)\n", 
			newId, oldComm, len(nodes))
		
		// Use existing calculateCommunitySketch function
		communitySketch := sle.calculateCommunitySketch(oldComm, community)
		communitySketch.nodeId = newId
		
		superNodeSketches[newId] = communitySketch
		newSketchManager.vertexSketches[newId] = communitySketch
	}
	
	// Step 5: Rebuild hash-to-node mapping for super-nodes
	newHashToNodeMap := make(map[uint32]int64)
	
	for superNodeId, superSketch := range superNodeSketches {
		// Get all sketch values from all layers
		for layer := int64(0); layer < sle.config.NK; layer++ {
			layerSketch := superSketch.GetSketch(layer)
			for _, hashValue := range layerSketch {
				if hashValue != math.MaxUint32 {
					// Check if this hash already maps to a different super-node
					if _, exists := newHashToNodeMap[hashValue]; !exists {
						newHashToNodeMap[hashValue] = superNodeId
					}
				}
			}
		}
	}
	
	newSketchManager.hashToNodeMap = newHashToNodeMap
	
	// Step 6: Create new nodeToSketch mapping (for compatibility with existing code)
	newNodeToSketch := make(map[int64][]uint32)
	
	for superNodeId, superSketch := range superNodeSketches {
		// Extract all non-max values from all layers
		allValues := superSketch.GetAllSketches()
		if len(allValues) > 0 {
			newNodeToSketch[superNodeId] = allValues
		}
	}
	
	// Step 7: Build super-graph structure based on sketch intersections
	newGraph := &GraphStructure{
		V: make([]SymmetricVertex, newNodeCount),
		n: newNodeCount,
		m: 0,
	}
	
	// First, build a reverse index: hash -> list of super-nodes containing it
	hashToSuperNodes := make(map[uint32][]int64)
	
	for superNodeId, superSketch := range superNodeSketches {
		for layer := int64(0); layer < sle.config.NK; layer++ {
			layerSketch := superSketch.GetSketch(layer)
			for _, hashValue := range layerSketch {
				if hashValue != math.MaxUint32 {
					hashToSuperNodes[hashValue] = append(hashToSuperNodes[hashValue], superNodeId)
				}
			}
		}
	}
	
	// Now find edges: if a hash appears in multiple super-nodes, they're connected
	// edgeSet := make(map[string]bool)
	adjacencyList := make(map[int64]map[int64]bool)
	
	for _, superNodes := range hashToSuperNodes {
		if len(superNodes) > 1 {
			// This hash connects these super-nodes
			for i := 0; i < len(superNodes); i++ {
				for j := i + 1; j < len(superNodes); j++ {
					node1, node2 := superNodes[i], superNodes[j]
					
					// Initialize adjacency maps if needed
					if adjacencyList[node1] == nil {
						adjacencyList[node1] = make(map[int64]bool)
					}
					if adjacencyList[node2] == nil {
						adjacencyList[node2] = make(map[int64]bool)
					}
					
					// Add edge (undirected)
					adjacencyList[node1][node2] = true
					adjacencyList[node2][node1] = true
				
				}
			}
		}
	}
	
	// Convert adjacency list to graph structure
	for superNodeId := int64(0); superNodeId < newNodeCount; superNodeId++ {
		neighbors := make([]int64, 0)
		if adjMap, exists := adjacencyList[superNodeId]; exists {
			for neighbor := range adjMap {
				neighbors = append(neighbors, neighbor)
			}
		}
		
		newGraph.V[superNodeId].neighbors = neighbors
		newGraph.V[superNodeId].degree = int64(len(neighbors))
		newGraph.m += int64(len(neighbors))
		
		if len(neighbors) > 0 {
			fmt.Printf("Super-node %d has neighbors: %v\n", superNodeId, neighbors)
		}
	}
	
	// Correct edge count (since we counted each edge twice)
	newGraph.m /= 2
	
	fmt.Printf("\nSuper-graph: %d nodes, %d edges\n", newNodeCount, newGraph.m)
	
	// Step 8: Create new community array (each super-node starts in its own community)
	newCommunity := make([]int64, newNodeCount)
	for i := int64(0); i < newNodeCount; i++ {
		newCommunity[i] = i
	}
	
	// Step 9: Record aggregation in partition tracker
	sle.partitionTracker.recordAggregation(community, commToNewNode, sle.sketchManager.vertexSketches)
	
	// Step 10: Update the main sketch manager
	sle.sketchManager = newSketchManager
	
	return newGraph, newNodeToSketch, newCommunity
}
	


// calculateFinalModularity calculates the final modularity score
func (sle *SketchLouvainEngine) calculateFinalModularity(
	community []int64,
	nodeToSketch map[int64][]uint32,
	wholeWeight float64,
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
			sketch1 := sle.sketchManager.GetVertexSketch(nodeId1)
			if sketch1 == nil {
				continue
			}
			
			nodeDegree := sketch1.EstimateCardinality()
			totalDegree += nodeDegree
			
			// Count internal edges
			for j := i + 1; j < len(nodes); j++ {
				nodeId2 := nodes[j]
				sketch1Values := nodeToSketch[nodeId1]
				sketch2Values := nodeToSketch[nodeId2]
				
				intersections := sle.countSketchIntersections(sketch1Values, sketch2Values)
				internalEdges += float64(intersections) / float64(sle.config.K)
			}
		}
		
		// Modularity contribution
		if wholeWeight > 0 {
			deltaQ := (internalEdges / wholeWeight) - (totalDegree / (2 * wholeWeight)) * (totalDegree / (2 * wholeWeight))
			totalModularity += deltaQ
		}
	}
	
	return totalModularity
}


// This function is MISSING - we need to create it
func (sle *SketchLouvainEngine) createUnionSketch(sketch1, sketch2 *VertexBottomKSketch) *VertexBottomKSketch {
	unionSketch := NewVertexBottomKSketch(-1, sle.config.K, sle.config.NK)
	
	// Union ALL layers, not just one
	for layer := 0; layer < int(sle.config.NK); layer++ {
		layer1 := sketch1.GetSketch(int64(layer))
		layer2 := sketch2.GetSketch(int64(layer))
		unionSketch.sketches[layer] = sketch1.bottomKUnion(layer1, layer2)  // Use existing bottomKUnion
	}
	
	return unionSketch
}

func (sle *SketchLouvainEngine) calculateCommunitySketch(
	targetComm int64,
	community []int64,
) *VertexBottomKSketch {
	// Start with empty community sketch
	communitySketch := NewVertexBottomKSketch(targetComm, sle.config.K, sle.config.NK)
	
	// Union each node's sketch using your existing method
	for nodeId, comm := range community {
		if comm == targetComm {
			nodeSketch := sle.sketchManager.GetVertexSketch(int64(nodeId))
			if nodeSketch != nil {
				for layer := int64(0); layer < sle.config.NK; layer++ {
					nodeLayer := nodeSketch.GetSketch(layer)
					communitySketch.UnionWithLayer(layer, nodeLayer)  
				}
			}
		}
	}
	
	return communitySketch
}


func (sle *SketchLouvainEngine) printHierarchySummary() {
	if sle.partitionTracker.currentLevel == 0 {
		fmt.Println("\n=== NO HIERARCHY (No aggregation occurred) ===")
		return
	}
	
	fmt.Println("\n=== HIERARCHY SUMMARY ===")
	fmt.Printf("Total levels: %d\n", sle.partitionTracker.currentLevel)
	
	for i, levelMapping := range sle.partitionTracker.levelMappings {
		fmt.Printf("\nLevel %d:\n", i)
		fmt.Printf("  Number of super-nodes: %d\n", levelMapping.numNodes)
		fmt.Printf("  Super-node sizes:\n")
		
		for superNodeId, originalNodes := range levelMapping.superNodeToOriginalNodes {
			fmt.Printf("    Super-node %d: %d original nodes\n", superNodeId, len(originalNodes))
		}
	}
	
	// Example: trace a few nodes
	fmt.Println("\n=== SAMPLE NODE TRACES ===")
	for i := int64(0); i < int64(min(5, int(sle.originalGraphSize))); i++ {
		sle.partitionTracker.traceNode(i)
	}
}



// Helper function to print community state
func (sle *SketchLouvainEngine) printCommunityState(stateType string, community []int64, nodeToSketch map[int64][]uint32, level int) {
	fmt.Printf("\n=== %s STATE - LEVEL %d ===\n", stateType, level)
	
	// Group nodes by community
	communityMap := make(map[int64][]int64)
	for nodeId, commId := range community {
		if _, hasSketch := nodeToSketch[int64(nodeId)]; hasSketch {
			if _, exists := communityMap[commId]; !exists {
				communityMap[commId] = []int64{}
			}
			communityMap[commId] = append(communityMap[commId], int64(nodeId))
		}
	}

	fmt.Printf("Number of communities: %d\n", len(communityMap))
	fmt.Printf("Community assignments:\n")
	for commId, nodes := range communityMap {
		fmt.Printf("  Community %d (%d nodes): ", commId, len(nodes))
		for i, nodeId := range nodes {
			if i > 0 {
				fmt.Printf(", ")
			}
			fmt.Printf("%d", nodeId)
			// Limit output for readability
			if i >= 9 && len(nodes) > 10 {
				fmt.Printf("... (+%d more)", len(nodes)-10)
				break
			}
		}
		fmt.Println()
	}
	fmt.Printf("=== END %s STATE ===\n\n", stateType)
}
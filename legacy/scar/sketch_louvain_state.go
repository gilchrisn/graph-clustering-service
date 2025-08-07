package scar



import (
	"fmt"
	// "sync"
	"sort"
	"strings"
	"math"
)
// SketchLouvainState manages community membership and delegates sketch operations
type SketchLouvainState struct {
	nodeToCommunity   []int64               // node -> community mapping
	communityToNodes  map[int64][]int64     // community -> nodes reverse mapping  
	activeCommunities map[int64]bool        // which communities exist
	sketchManager     *SketchManager        // delegate sketch operations
	totalWeight       float64               // total weight of the graph (if needed)
	n                  int64                  // number of nodes in the graph
	sketchAdjacencyList map[int64][]WeightedEdge   // adjacency list for sketch neighbors
}

func NewSketchLouvainState(n int64, sketchManager *SketchManager) *SketchLouvainState {
	// create new sketch louvain state
	sls := &SketchLouvainState{
		nodeToCommunity:   make([]int64, n), // initialize with -1 (no community)
		communityToNodes:  make(map[int64][]int64),	 // community ID -> nodes mapping
		activeCommunities: make(map[int64]bool), // track active communities
		sketchManager:     sketchManager,
		totalWeight:       0.0,
		n:                 n,
		sketchAdjacencyList: make(map[int64][]WeightedEdge), // adjacency list for sketch neighbors
	}

	return sls
}

type WeightedEdge struct {
    neighbor int64
    weight   float64
}

func (we WeightedEdge) GetNeighbor() int64 {
	return we.neighbor
}

// Get community ID for a node
func (sls *SketchLouvainState) GetNodeCommunity(nodeId int64) int64 {
	if nodeId >= int64(len(sls.nodeToCommunity)) {
		return -1
	}
	return sls.nodeToCommunity[nodeId]
}

// Get all nodes in a community
func (sls *SketchLouvainState) GetCommunityNodes(commId int64) []int64 {
	return sls.communityToNodes[commId]
}

// GetNodesToCommunityMap returns the mapping of nodes to their communities
func (sls *SketchLouvainState) GetNodesToCommunityMap() []int64 {
	return sls.nodeToCommunity
}

// GetCommunityToNodesMap returns the mapping of communities to their member nodes
func (sls *SketchLouvainState) GetCommunityToNodesMap() map[int64][]int64 {
	return sls.communityToNodes
}

// Get all active communities
func (sls *SketchLouvainState) GetActiveCommunities() map[int64]bool {
	return sls.activeCommunities
}

// GetSketchAdjacencyList returns the adjacency list for sketch neighbors
func (sls *SketchLouvainState) GetSketchAdjacencyList() map[int64][]WeightedEdge {
	return sls.sketchAdjacencyList
}

// GetVertexSketch returns the sketch for a specific node
func (sls *SketchLouvainState) GetVertexSketch(nodeId int64) *VertexBottomKSketch {
	return sls.sketchManager.GetVertexSketch(nodeId)
}

// GetTotalWeight returns the total weight of the graph
func (sls *SketchLouvainState) GetTotalWeight() float64 {
	return sls.totalWeight
}

// Move node from one community to another
func (sls *SketchLouvainState) MoveNode(nodeId, fromComm, toComm int64) {
	// Update node mapping
	sls.nodeToCommunity[nodeId] = toComm
	
	// Update reverse mappings
	// Remove from old community
	if fromNodes, exists := sls.communityToNodes[fromComm]; exists {
		newFromNodes := make([]int64, 0, len(fromNodes)-1)
		for _, id := range fromNodes {
			if id != nodeId {
				newFromNodes = append(newFromNodes, id)
			}
		}
		if len(newFromNodes) == 0 {
			// Community is now empty
			delete(sls.communityToNodes, fromComm)
			delete(sls.activeCommunities, fromComm)
			sls.sketchManager.DeleteCommunitySketch(fromComm)
		} else {
			sls.communityToNodes[fromComm] = newFromNodes
			// Update community sketch for remaining nodes
			sls.sketchManager.UpdateCommunitySketch(fromComm, newFromNodes)
		}
	}
	
	// Add to new community
	sls.communityToNodes[toComm] = append(sls.communityToNodes[toComm], nodeId)
	sls.activeCommunities[toComm] = true
	// Update community sketch with new member
	sls.sketchManager.UpdateCommunitySketch(toComm, sls.communityToNodes[toComm])
}

// Get community sketch via delegation
func (sls *SketchLouvainState) GetCommunitySketch(commId int64) *VertexBottomKSketch {
	return sls.sketchManager.GetCommunitySketch(commId)
}

// Estimate cardinality of a node's sketch via delegation
func (sls *SketchLouvainState) EstimateCardinality(nodeId int64) float64 {
	sketch := sls.sketchManager.GetVertexSketch(nodeId)
	if sketch == nil {
		return 0.0 // No sketch available
	}
	
	if sketch.IsSketchFull() {
		return sketch.EstimateCardinality() // Use sketch cardinality if full
	} else {
		// If sketch is not full, estimate using adjacency list
		neighbors := sls.GetSketchNeighbors(nodeId)
		totalWeight := 0.0
		for _, edge := range neighbors {
			if (nodeId == edge.neighbor) {
				totalWeight += edge.weight * 2 // Avoid double counting self-loops
			} else {
				totalWeight += edge.weight
			}
		}
		return totalWeight // Return sum of weights as estimate
	}
}

// Estimate community cardinality via delegation
func (sls *SketchLouvainState) EstimateCommunityCardinality(commId int64) float64 {
	degreeSum := 0.0
	for _, nodeId := range sls.communityToNodes[commId] {
		degreeSum += sls.EstimateCardinality(nodeId)
	}
	return degreeSum
}

func (sls *SketchLouvainState) EstimateEdgesToCommunity(nodeId, commId int64) float64 {
	// Get the sketch for the node
	nodeSketch := sls.GetVertexSketch(nodeId)
	if nodeSketch == nil {
		return 0.0
	}

	// Get the sketch for the community
	communitySketch := sls.GetCommunitySketch(commId)
	if communitySketch == nil {
		return 0.0
	}

	if !nodeSketch.IsSketchFull() {
		return sls.countExactEdgesToCommunity(nodeId, commId)
	} else if communitySketch.IsSketchFull() {
		return sls.estimateEdgesViaInclusion(nodeSketch, communitySketch, commId)
	} else {
		return sls.countExactEdgesToCommunityHybrid(nodeId, commId)
	}
}


func (sls *SketchLouvainState) countExactEdgesToCommunity(nodeId, commId int64) float64 {
	edgeCount := 0.0
	for _, edge := range sls.GetSketchNeighbors(nodeId) {
		if sls.GetNodeCommunity(edge.neighbor) == commId {
			edgeCount += edge.weight
		}
	}
	return edgeCount
}

func (sls *SketchLouvainState) estimateEdgesViaInclusion(nodeSketch, communitySketch *VertexBottomKSketch, commId int64) float64 {
	nodeDegree := nodeSketch.EstimateCardinality()
	communityDegree := sls.EstimateCommunityCardinality(commId)
	
	unionSketch := nodeSketch.UnionWith(communitySketch)
	if unionSketch == nil {
		return 0.0
	}
	
	unionDegree := unionSketch.EstimateCardinality()
	intersectionSize := nodeDegree + communityDegree - unionDegree

	return math.Max(0, intersectionSize)
}

func (sls *SketchLouvainState) countExactEdgesToCommunityHybrid(nodeId, commId int64) float64 {
	edgeWeight := 0.0
	
	// Iterate through all members of the community
	for _, memberNode := range sls.communityToNodes[commId] {
		// Check adjacency list of this community member for edge to target node
		for _, edge := range sls.GetSketchNeighbors(memberNode) {
			if edge.neighbor == nodeId {
				edgeWeight += edge.weight
				break // Found the edge, no need to continue
			}
		}
	}
	
	return edgeWeight
}

func (sls *SketchLouvainState) EstimateEdgesBetweenCommunities(nodeId, targetComm int64) float64 {
    // This is just a clear wrapper around the existing method
    // The existing EstimateEdgesToCommunity already handles:
    // - Non-full sketches: exact counting via adjacency list ✓
    // - Full sketches: inclusion-exclusion estimation ✓  
    // - Mixed cases: non-full handled first ✓
    return sls.EstimateEdgesToCommunity(nodeId, targetComm)
}


// FindNeighborCommunities finds communities of neighbors for a node. Returns all the neighboring communities and weight to each community.
func (sls *SketchLouvainState) FindNeighboringCommunities(nodeId int64) map[int64]float64 {
	neighborComms := make(map[int64]float64)
	
	// If sketch is full, just return all existing communities
	sketch := sls.sketchManager.GetVertexSketch(nodeId)
	if sketch != nil && sketch.IsSketchFull() {
		// Iterate over all communities
		for commId := range sls.activeCommunities {
			neighborComms[commId] = sls.EstimateEdgesToCommunity(nodeId, commId)
		}
		return neighborComms
	}

	// Get neighbors from the sketch adjacency list
	neighbors := sls.GetSketchNeighbors(nodeId)
	for _, edge := range neighbors {
		neighborId := edge.neighbor
		neighborComm := sls.GetNodeCommunity(neighborId)
		if neighborComm == -1 {
			continue // Skip if neighbor has no community
		}
		// Add or update the community weight
		if _, exists := neighborComms[neighborComm]; !exists {
			neighborComms[neighborComm] = 0.0 // Initialize weight if not present
		}
		neighborComms[neighborComm] += edge.weight // Sum weights
	}

	return neighborComms
}

// GetNodeFromHash returns the node ID for a hash value
func (sls *SketchLouvainState) GetNodeFromHash(hashValue uint32) (int64, bool) {
	return sls.sketchManager.GetNodeFromHash(hashValue)
}

// GetAllVertexSketches returns all vertex sketches
func (sls *SketchLouvainState) GetAllVertexSketches() map[int64]*VertexBottomKSketch {
	return sls.sketchManager.vertexSketches
}

// GetAllCommunitySketches returns all community sketches
func (sls *SketchLouvainState) GetAllCommunitySketches() map[int64]*VertexBottomKSketch {
	return sls.sketchManager.communitySketches
}

func (sls *SketchLouvainState) CalculateWholeWeight() {
	// Calculate the total weight of the graph based on the community sketches
	totalWeight := 0.0

	// for each vertex sketch
	for nodeId, sketch := range sls.sketchManager.vertexSketches {
		if sketch == nil {
			continue // Skip if no sketch exists for this node
		}
		if sketch.IsSketchFull() {
			// If sketch is full, use its cardinality as the weight
			totalWeight += sketch.EstimateCardinality()
		} else {
			// If sketch is not full, use adjacency list to estimate weight
			neighbors := sls.GetSketchNeighbors(nodeId)
			for _, edge := range neighbors {
				if (nodeId == edge.neighbor) {
					totalWeight += edge.weight * 2 
				} else {
					totalWeight += edge.weight
				}
			}
		}
	}
	sls.totalWeight = totalWeight / 2.0
}

func (sls *SketchLouvainState) BuildSketchAdjacencyList(numWorkers int) {
	fmt.Printf("Building sketch adjacency list with %d workers...\n", numWorkers)
	
	// Get all nodes with sketches
	allNodes := make([]int64, 0, len(sls.sketchManager.vertexSketches))
	for nodeId := range sls.sketchManager.vertexSketches {
		allNodes = append(allNodes, nodeId)
	}
	
	if len(allNodes) == 0 {
		fmt.Println("No nodes with sketches found")
		return
	}
	
	// Initialize adjacency list
	sls.sketchAdjacencyList = make(map[int64][]WeightedEdge)
	
	// Separate nodes by sketch fullness
	nonFullNodes := make([]int64, 0)
	fullNodes := make([]int64, 0)
	
	for _, nodeId := range allNodes {
		sketch := sls.sketchManager.GetVertexSketch(nodeId)
		if sketch != nil {
			if sketch.IsSketchFull() {
				fullNodes = append(fullNodes, nodeId)
			} else {
				nonFullNodes = append(nonFullNodes, nodeId)
			}
		}
	}
	
	// Phase 1: Handle non-full sketches (direct mapping) - NO parallelization needed here
	for _, nodeId := range nonFullNodes {
		sketch := sls.sketchManager.GetVertexSketch(nodeId)
		if sketch == nil {
			continue
		}
		
		// Use map to accumulate
		neighborWeights := make(map[int64]float64)
		hashes := sketch.GetLayerHashes(0) // Get hashes from first layer only
		
		for _, hash := range hashes {
			if neighborNodeId, exists := sls.sketchManager.GetNodeFromHash(hash); exists {
				neighborWeights[neighborNodeId] += 1.0
			}
		}
		
		// Convert to WeightedEdge slice
		weightedEdges := make([]WeightedEdge, 0, len(neighborWeights))
		for neighborId, weight := range neighborWeights {
			weightedEdges = append(weightedEdges, WeightedEdge{
				neighbor: neighborId,
				weight:   weight,
			})
		}
		
		sls.sketchAdjacencyList[nodeId] = weightedEdges
	}
	
	// // Phase 2: Handle full sketches with parallelization
	// reverseMap := make(map[uint32][]int64)
	// var reverseMapMutex sync.Mutex  // ONE mutex for the entire map
	
	// var wg sync.WaitGroup
	// nodesPerWorker := len(fullNodes) / numWorkers
	// if nodesPerWorker == 0 {
	// 	nodesPerWorker = 1
	// }
	
	// for worker := 0; worker < numWorkers; worker++ {
	// 	start := worker * nodesPerWorker
	// 	end := start + nodesPerWorker
	// 	if worker == numWorkers-1 {
	// 		end = len(fullNodes) // Last worker handles remaining nodes
	// 	}
		
	// 	wg.Add(1)
	// 	go func(startIdx, endIdx int) {
	// 		defer wg.Done()
			
	// 		for i := startIdx; i < endIdx; i++ {
	// 			nodeId := fullNodes[i]
	// 			sketch := sls.sketchManager.GetVertexSketch(nodeId)
	// 			if sketch == nil {
	// 				continue
	// 			}
				
	// 			// Get hashes from first layer only
	// 			layerHashes := sketch.GetLayerHashes(0)
				
	// 			for _, hash := range layerHashes {
	// 				// Use single mutex for entire map
	// 				reverseMapMutex.Lock()
	// 				reverseMap[hash] = append(reverseMap[hash], nodeId)
	// 				reverseMapMutex.Unlock()
	// 			}
	// 		}
	// 	}(start, end)
	// }
	
	// wg.Wait()
	
	// // Build adjacency lists for full sketches (sequential)
	// for _, nodes := range reverseMap {
	// 	if len(nodes) < 2 {
	// 		continue
	// 	}
		
	// 	for i := 0; i < len(nodes); i++ {
	// 		for j := i + 1; j < len(nodes); j++ {
	// 			u, v := nodes[i], nodes[j]
	// 			weight := 1.0
				
	// 			sls.sketchAdjacencyList[u] = append(sls.sketchAdjacencyList[u], WeightedEdge{neighbor: v, weight: weight})
	// 			sls.sketchAdjacencyList[v] = append(sls.sketchAdjacencyList[v], WeightedEdge{neighbor: u, weight: weight})
	// 		}
	// 	}
	// }
	
}

// BuildSuperNodeAdjacencyFromPreviousLevel builds the adjacency list for super-nodes
// using the connectivity information from the previous level
func (sls *SketchLouvainState) BuildSuperNodeAdjacencyFromPreviousLevel(
	prevState *SketchLouvainState,
	communityMapping map[int64]int64, // oldCommunityId -> newSuperNodeId
) error {
	fmt.Printf("Building super-node adjacency from previous level...\n")
	
	// Initialize adjacency list
	sls.sketchAdjacencyList = make(map[int64][]WeightedEdge)
	
	// Temporary adjacency accumulator: superNodeA -> superNodeB -> weight
	adjacency := make(map[int64]map[int64]float64)
	for _, superNodeId := range communityMapping {
		adjacency[superNodeId] = make(map[int64]float64)
	}
	
	// Separate communities by sketch fullness
	nonFullCommunities := make([]int64, 0)
	fullCommunities := make([]int64, 0)
	
	for communityId, nodes := range prevState.GetCommunityToNodesMap() {
		if len(nodes) == 0 {
			continue
		}
		
		// Use the COMMUNITY sketch, not the first node's sketch
		communitySketch := prevState.GetCommunitySketch(communityId)
		if communitySketch == nil {
			continue
		}
		
		if communitySketch.IsSketchFull() {
			fullCommunities = append(fullCommunities, communityId)
		} else {
			nonFullCommunities = append(nonFullCommunities, communityId)
		}
	}
	
	// Process non-full sketch communities (exact method)
	for _, communityId := range nonFullCommunities {
		nodes := prevState.GetCommunityNodes(communityId)
		currentSupernode := communityMapping[communityId]
		// fmt.Printf("For each node in community %d (%d) with members %v:\n", communityId, currentSupernode, nodes)

		for _, nodeId := range nodes {
			neighbors := prevState.GetSketchNeighbors(nodeId)
			// fmt.Printf("Node %d's neighbors: %v\n", nodeId, neighbors)
			for _, edge := range neighbors {
				neighborCommunity := prevState.GetNodeCommunity(edge.neighbor)
				if neighborCommunity == -1 {
					continue
				}
				otherSupernode := communityMapping[neighborCommunity]
				// fmt.Printf("  Neighbor %d his in %d (%d) with weight %.1f\n", edge.neighbor, neighborCommunity, otherSupernode, edge.weight)
				if currentSupernode == otherSupernode {
					if (nodeId == edge.neighbor) {
						adjacency[currentSupernode][currentSupernode] += edge.weight
					} else {
						adjacency[currentSupernode][currentSupernode] += edge.weight / 2.0
					}
				} else {
					adjacency[currentSupernode][otherSupernode] += edge.weight
				}
			}
		}
	}
	
	// // Process full sketch communities (estimation method)
	// for i, communityA := range fullCommunities {
	// 	for j, communityB := range fullCommunities {
	// 		if i > j {
	// 			continue // Skip duplicate pairs
	// 		}
			
	// 		superNodeA := communityMapping[communityA]
	// 		superNodeB := communityMapping[communityB]
			
	// 		// Check if they are neighbors using sketch intersection
	// 		areNeighbors := false
	// 		if communityA == communityB {
	// 			areNeighbors = true // Self-loop
	// 		} else {
	// 			// Check sketch intersection like leaf level
	// 			sketchA := prevState.GetCommunitySketch(communityA)
	// 			sketchB := prevState.GetCommunitySketch(communityB)
	// 			if sketchA != nil && sketchB != nil {
	// 				intersection := sketchA.IntersectWith(sketchB)
	// 				areNeighbors = len(intersection) > 0
	// 			}
	// 		}
			
	// 		if !areNeighbors {
	// 			continue
	// 		}
			
	// 		// Estimate edges between communities
	// 		edges := prevState.EstimateEdgesBetweenCommunities(communityA, communityB)
			
	// 		if edges > 0 {
	// 			if communityA == communityB {
	// 				adjacency[superNodeA][superNodeA] += edges
	// 			} else {
	// 				adjacency[superNodeA][superNodeB] += edges
	// 				adjacency[superNodeB][superNodeA] += edges
	// 			}
	// 		}
	// 	}
	// }
	
	// Convert to WeightedEdge format
	for superNodeA, neighbors := range adjacency {
		weightedEdges := make([]WeightedEdge, 0)
		for superNodeB, weight := range neighbors {
			if weight > 0 {
				weightedEdges = append(weightedEdges, WeightedEdge{
					neighbor: superNodeB,
					weight:   weight,
				})
			}
		}
		if len(weightedEdges) > 0 {
			sls.sketchAdjacencyList[superNodeA] = weightedEdges
		}
	}
	
	// fmt.Printf("Super-node adjacency built: %d nodes with edges\n", len(sls.sketchAdjacencyList))
	// for nodeId, edges := range sls.sketchAdjacencyList {
	// 	if len(edges) > 0 {
	// 		fmt.Printf("  Node %d: %d neighbors\n", nodeId, len(edges))
	// 		if len(edges) > 20 {
	// 			fmt.Printf("    Neighbors: ")
	// 			for i, edge := range edges {
	// 				if i >= 5 {
	// 					fmt.Printf("...+%d more", len(edges)-5)
	// 					break
	// 				}
	// 				fmt.Printf("(%d,%.1f) ", edge.neighbor, edge.weight)
	// 			}
	// 			fmt.Println()
	// 		} else {
	// 			fmt.Printf("    Neighbors: ")
	// 			for _, edge := range edges {
	// 				fmt.Printf("(%d,%.1f) ", edge.neighbor, edge.weight)
	// 			}
	// 			fmt.Println()	
	// 		}
	// 	}
	// }
	return nil
}


// Get sketch neighbors for a node
func (sls *SketchLouvainState) GetSketchNeighbors(nodeId int64) []WeightedEdge {
	if weightedEdges, exists := sls.sketchAdjacencyList[nodeId]; exists {
		return weightedEdges
	}
	return []WeightedEdge{}
}

// ============================================= DEBUGGING =============================================


// PrintState prints the state of the SCAR Louvain algorithm execution
// mode can be "ALL", "COMMUNITIES", "SKETCHES", or "GRAPH"
func (sls *SketchLouvainState) PrintState(label string, mode string) {
	fmt.Printf("\n" + strings.Repeat("=", 80) + "\n")
	fmt.Printf("SCAR STATE: %s (Mode: %s)\n", label, mode)
	fmt.Printf(strings.Repeat("=", 80) + "\n")
	
	if mode == "GRAPH" || mode == "ALL" {
		sls.printGraph()
	}
	
	if mode == "COMMUNITIES" || mode == "ALL" {
		sls.printCommunities()
	}
	
	if mode == "SKETCHES" || mode == "ALL" {
		sls.printSketches()
	}
	
	fmt.Printf(strings.Repeat("=", 80) + "\n")
}

// printGraph prints the graph structure using sketch adjacency
func (sls *SketchLouvainState) printGraph() {
    fmt.Printf("GRAPH STRUCTURE:\n")
    fmt.Printf("  Nodes: %d\n", sls.n)
    fmt.Printf("  Total weight: %.4f\n", sls.totalWeight)
    fmt.Printf("  Sketch adjacency entries: %d\n", len(sls.sketchAdjacencyList))
    fmt.Printf("\n")
    
    // Get all nodes with sketches
    nodeIDs := make([]int64, 0, len(sls.sketchManager.vertexSketches))
    for nodeId := range sls.sketchManager.vertexSketches {
        nodeIDs = append(nodeIDs, nodeId)
    }
    sort.Slice(nodeIDs, func(i, j int) bool { return nodeIDs[i] < nodeIDs[j] })
    
    for _, nodeId := range nodeIDs {
        sketch := sls.sketchManager.GetVertexSketch(nodeId)
        weightedEdges := sls.GetSketchNeighbors(nodeId)
        
        // Calculate total weighted degree from adjacency list
        totalWeight := 0.0
        for _, edge := range weightedEdges {
            totalWeight += edge.weight
        }
        
        sketchDegree := 0.0
        if sketch != nil {
            sketchDegree = sketch.EstimateCardinality()
        }
        
        fmt.Printf("  Node %d: sketch_degree=%.2f, weighted_degree=%.2f, neighbors=%d\n", 
            nodeId, sketchDegree, totalWeight, len(weightedEdges))
        
        // Show first few weighted edges
        if len(weightedEdges) > 0 {
            fmt.Printf("    Weighted edges: ")
            maxShow := 20
            for i, edge := range weightedEdges {
                if i >= maxShow {
                    fmt.Printf("...+%d more", len(weightedEdges)-maxShow)
                    break
                }
                fmt.Printf("(%d,%.1f)", edge.neighbor, edge.weight)
                if i < len(weightedEdges)-1 && i < maxShow-1 {
                    fmt.Printf(", ")
                }
            }
            fmt.Printf("\n")
        }
    }
    fmt.Printf("\n")
}

// printCommunities prints the community structure
func (sls *SketchLouvainState) printCommunities() {
	fmt.Printf("COMMUNITY STRUCTURE:\n")
	fmt.Printf("  Active communities: %d\n", len(sls.activeCommunities))
	fmt.Printf("  Total nodes with communities: %d\n", len(sls.nodeToCommunity))
	fmt.Printf("\n")
	
	// Print N2C (node -> community)
	fmt.Printf("N2C (Node -> Community):\n")
	nodeIDs := make([]int64, 0, len(sls.nodeToCommunity))
	for i := int64(0); i < int64(len(sls.nodeToCommunity)); i++ {
		if sls.nodeToCommunity[i] != -1 {  // Only show nodes with valid communities
			nodeIDs = append(nodeIDs, i)
		}
	}
	sort.Slice(nodeIDs, func(i, j int) bool { return nodeIDs[i] < nodeIDs[j] })
	
	for _, nodeId := range nodeIDs {
		commId := sls.nodeToCommunity[nodeId]
		fmt.Printf("  Node %d -> Community %d\n", nodeId, commId)
	}
	fmt.Printf("\n")
	
	// Print C2N (community -> nodes)
	fmt.Printf("C2N (Community -> Nodes):\n")
	communityIDs := make([]int64, 0, len(sls.communityToNodes))
	for commID := range sls.communityToNodes {
		communityIDs = append(communityIDs, commID)
	}
	sort.Slice(communityIDs, func(i, j int) bool { return communityIDs[i] < communityIDs[j] })
	
	for _, commID := range communityIDs {
		nodes := sls.communityToNodes[commID]
		nodesCopy := make([]int64, len(nodes))
		copy(nodesCopy, nodes)
		sort.Slice(nodesCopy, func(i, j int) bool { return nodesCopy[i] < nodesCopy[j] })
		fmt.Printf("  Community %d -> Nodes %v (count: %d)\n", commID, nodesCopy, len(nodes))
	}
	fmt.Printf("\n")
	
	// Print estimated community cardinalities
	fmt.Printf("Estimated Community Cardinalities:\n")
	for _, commID := range communityIDs {
		cardinality := sls.EstimateCommunityCardinality(commID)
		fmt.Printf("  Community %d -> Estimated Cardinality: %.4f\n", commID, cardinality)
	}
	fmt.Printf("\n")
	
	// Summary table
	fmt.Printf("COMMUNITY SUMMARY:\n")
	fmt.Printf("  %-10s %-10s %-15s %-10s\n", "Community", "Nodes", "Est_Cardinality", "Active")
	fmt.Printf("  %s\n", strings.Repeat("-", 50))
	
	for _, commID := range communityIDs {
		nodeCount := len(sls.communityToNodes[commID])
		cardinality := sls.EstimateCommunityCardinality(commID)
		isActive := sls.activeCommunities[commID]
		
		fmt.Printf("  %-10d %-10d %-15.4f %-10t\n", 
			commID, nodeCount, cardinality, isActive)
	}
	fmt.Printf("\n")
}

// printSketches prints detailed sketch information with actual sketch contents
func (sls *SketchLouvainState) printSketches() {
	fmt.Printf("SKETCH INFORMATION:\n")
	fmt.Printf("  Total vertex sketches: %d\n", len(sls.sketchManager.vertexSketches))
	fmt.Printf("  Total community sketches: %d\n", len(sls.sketchManager.communitySketches))
	fmt.Printf("  Hash-to-node mappings: %d\n", len(sls.sketchManager.hashToNodeMap))
	fmt.Printf("\n")
	
	// Print vertex sketches with full contents
	fmt.Printf("VERTEX SKETCHES (DETAILED):\n")
	nodeIDs := make([]int64, 0, len(sls.sketchManager.vertexSketches))
	for nodeId := range sls.sketchManager.vertexSketches {
		nodeIDs = append(nodeIDs, nodeId)
	}
	sort.Slice(nodeIDs, func(i, j int) bool { return nodeIDs[i] < nodeIDs[j] })
	
	for _, nodeId := range nodeIDs {
		sketch := sls.sketchManager.GetVertexSketch(nodeId)
		if sketch != nil {
			cardinality := sls.EstimateCardinality(nodeId)
			filledCount := sketch.GetFilledCount()
			isFull := sketch.IsSketchFull()
			
			fmt.Printf("  Node %d: cardinality=%.4f, filled=%d/%d, full=%t\n", 
				nodeId, cardinality, filledCount, sketch.k, isFull)
			
			// Print the actual sketch contents
			fmt.Printf("    %s\n", sketch.String())
			
			// Show which nodes this sketch connects to
			allSketches := sketch.GetAllSketches()
			connectedNodes := []int64{}
			for _, hash := range allSketches {
				if connectedNode, exists := sls.sketchManager.GetNodeFromHash(hash); exists {
					if connectedNode != nodeId {
						connectedNodes = append(connectedNodes, connectedNode)
					}
				}
			}
			if len(connectedNodes) > 0 {
				sort.Slice(connectedNodes, func(i, j int) bool { return connectedNodes[i] < connectedNodes[j] })
				fmt.Printf("    Connected to nodes: %v\n", connectedNodes)
			}
			fmt.Printf("\n")
		}
	}
	
	// Print community sketches with full contents
	fmt.Printf("COMMUNITY SKETCHES (DETAILED):\n")
	communityIDs := make([]int64, 0, len(sls.sketchManager.communitySketches))
	for commId := range sls.sketchManager.communitySketches {
		communityIDs = append(communityIDs, commId)
	}
	sort.Slice(communityIDs, func(i, j int) bool { return communityIDs[i] < communityIDs[j] })
	
	for _, commId := range communityIDs {
		sketch := sls.sketchManager.GetCommunitySketch(commId)
		if sketch != nil {
			cardinality := sls.EstimateCommunityCardinality(commId)
			filledCount := sketch.GetFilledCount()
			isFull := sketch.IsSketchFull()
			
			fmt.Printf("  Community %d: cardinality=%.4f, filled=%d/%d, full=%t\n", 
				commId, cardinality, filledCount, sketch.k, isFull)
			
			// Print the actual sketch contents
			fmt.Printf("    %s\n", sketch.String())
			
			// Show which nodes this community sketch connects to
			allSketches := sketch.GetAllSketches()
			connectedNodes := []int64{}
			for _, hash := range allSketches {
				if connectedNode, exists := sls.sketchManager.GetNodeFromHash(hash); exists {
					connectedNodes = append(connectedNodes, connectedNode)
				}
			}
			if len(connectedNodes) > 0 {
				sort.Slice(connectedNodes, func(i, j int) bool { return connectedNodes[i] < connectedNodes[j] })
				if len(connectedNodes) > 20 {
					fmt.Printf("    Connected to nodes: %v...+%d more\n", 
						connectedNodes[:20], len(connectedNodes)-20)
				} else {
					fmt.Printf("    Connected to nodes: %v\n", connectedNodes)
				}
			}
			fmt.Printf("\n")
		}
	}
	
	// Print hash-to-node mappings
	fmt.Printf("HASH-TO-NODE MAPPINGS:\n")
	hashValues := make([]uint32, 0, len(sls.sketchManager.hashToNodeMap))
	for hash := range sls.sketchManager.hashToNodeMap {
		hashValues = append(hashValues, hash)
	}
	sort.Slice(hashValues, func(i, j int) bool { return hashValues[i] < hashValues[j] })
	
	// Print all mappings if reasonable, otherwise truncate
	maxToPrint := 50
	for i, hash := range hashValues {
		if i >= maxToPrint {
			fmt.Printf("  ... and %d more mappings\n", len(hashValues)-maxToPrint)
			break
		}
		nodeId := sls.sketchManager.hashToNodeMap[hash]
		fmt.Printf("  Hash %d -> Node %d\n", hash, nodeId)
	}
	fmt.Printf("\n")
}

// PrintSketchDetails prints detailed sketch contents for specific nodes
func (sls *SketchLouvainState) PrintSketchDetails(nodeIds []int64) {
    fmt.Printf("\n" + strings.Repeat("-", 60) + "\n")
    fmt.Printf("DETAILED SKETCH CONTENTS\n")
    fmt.Printf(strings.Repeat("-", 60) + "\n")
    
    for _, nodeId := range nodeIds {
        sketch := sls.sketchManager.GetVertexSketch(nodeId)
        if sketch != nil {
            fmt.Printf("Node %d:\n", nodeId)
            fmt.Printf("  %s\n", sketch.String())
            fmt.Printf("  Cardinality: %.4f\n", sketch.EstimateCardinality())
            fmt.Printf("  Filled: %d/%d\n", sketch.GetFilledCount(), sketch.k)
            fmt.Printf("  Full: %t\n", sketch.IsSketchFull())
            
            // Show community membership
            community := sls.GetNodeCommunity(nodeId)
            fmt.Printf("  Community: %d\n", community)
            
            // Show weighted sketch neighbors
            weightedEdges := sls.GetSketchNeighbors(nodeId)
            if len(weightedEdges) > 0 {
                fmt.Printf("  Weighted neighbors (%d): ", len(weightedEdges))
                for i, edge := range weightedEdges {
                    fmt.Printf("(%d,%.1f)", edge.neighbor, edge.weight)
                    if i < len(weightedEdges)-1 && i < 10 {
                        fmt.Printf(", ")
                    }
                    if i >= 10 {
                        fmt.Printf("...+%d more", len(weightedEdges)-10)
                        break
                    }
                }
                fmt.Printf("\n")
            }
            
            // Show neighboring communities
            neighborComms := sls.FindNeighboringCommunities(nodeId)
            if len(neighborComms) > 0 {
                fmt.Printf("  Neighboring communities: ")
                for commId, weight := range neighborComms {
                    fmt.Printf("(comm_%d,%.1f) ", commId, weight)
                }
                fmt.Printf("\n")
            }
            
            fmt.Printf("\n")
        } else {
            fmt.Printf("Node %d: No sketch found\n\n", nodeId)
        }
    }
    fmt.Printf(strings.Repeat("-", 60) + "\n")
}

// PrintSketchComparison prints a comparison between two node sketches
func (sls *SketchLouvainState) PrintSketchComparison(nodeId1, nodeId2 int64) {
	fmt.Printf("\n" + strings.Repeat("-", 70) + "\n")
	fmt.Printf("SKETCH COMPARISON: Node %d vs Node %d\n", nodeId1, nodeId2)
	fmt.Printf(strings.Repeat("-", 70) + "\n")
	
	sketch1 := sls.sketchManager.GetVertexSketch(nodeId1)
	sketch2 := sls.sketchManager.GetVertexSketch(nodeId2)
	
	if sketch1 == nil || sketch2 == nil {
		fmt.Printf("Cannot compare - one or both sketches missing\n")
		return
	}
	
	fmt.Printf("Node %d: %s\n", nodeId1, sketch1.String())
	fmt.Printf("Node %d: %s\n", nodeId2, sketch2.String())
	
	// Calculate intersection
	intersection := sketch1.IntersectWith(sketch2)
	fmt.Printf("Intersection: %v (size: %d)\n", intersection, len(intersection))
	
	// Calculate union
	unionSketch := sketch1.UnionWith(sketch2)
	if unionSketch != nil {
		fmt.Printf("Union: %s\n", unionSketch.String())
		fmt.Printf("Union cardinality: %.4f\n", unionSketch.EstimateCardinality())
	}
	
	// Show communities
	comm1 := sls.GetNodeCommunity(nodeId1)
	comm2 := sls.GetNodeCommunity(nodeId2)
	fmt.Printf("Communities: Node %d -> %d, Node %d -> %d\n", nodeId1, comm1, nodeId2, comm2)
	
	fmt.Printf(strings.Repeat("-", 70) + "\n")
}
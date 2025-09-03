package scar

import (
	"fmt"
	"math"

	"strings" // for debugging purposes
)

// SketchGraph represents a graph with sketch-based operations (PARALLEL to Graph)
type SketchGraph struct {
	NumNodes    int         `json:"num_nodes"`
	TotalWeight float64     `json:"total_weight"`
	degrees       []float64 
	
	// Sketch-based data structures
	sketchManager   *SketchManager
	adjacencyList   [][]WeightedEdge  // For non-full sketches (exact edges)
}

// WeightedEdge represents an edge with weight
type WeightedEdge struct {
	Neighbor int
	Weight   float64
}

// SketchManager manages all vertex sketches
type SketchManager struct {
	vertexSketches map[int64]*VertexBottomKSketch
	hashToNodeMap  map[uint32]int64
    nodeToHashMap  map[int64]*VertexBottomKSketch  
	k              int64
	nk             int64
}

// VertexBottomKSketch represents a Bottom-K sketch for a single vertex
type VertexBottomKSketch struct {
	sketches    [][]uint32 // [nk][k] array - one sketch per layer
	k           int64      // sketch size
	nk          int64      // number of layers
	nodeId      int64      // which node this sketch belongs to
	filledCount int64      // filled count per layer
}

// NewSketchManager creates a new sketch manager
func NewSketchManager(k, nk int64) *SketchManager {
	return &SketchManager{
		vertexSketches: make(map[int64]*VertexBottomKSketch),
		hashToNodeMap:  make(map[uint32]int64),
        nodeToHashMap:  make(map[int64]*VertexBottomKSketch), 
		k:              k,
		nk:             nk,
	}
}

// GetVertexSketch returns the sketch for a vertex
func (sm *SketchManager) GetVertexSketch(nodeId int64) *VertexBottomKSketch {
	return sm.vertexSketches[nodeId]
}

// GetNodeFromHash returns the node ID for a hash value
func (sm *SketchManager) GetNodeFromHash(hashValue uint32) (int64, bool) {
	nodeId, exists := sm.hashToNodeMap[hashValue]
	return nodeId, exists
}

// GetNodeToHashMap returns a map of node IDs to their identifying hash sketches
func (sm *SketchManager) GetNodeToHashMap() map[int64]*VertexBottomKSketch {
	nodeToHashMap := make(map[int64]*VertexBottomKSketch)
	for nodeID, sketch := range sm.nodeToHashMap {
		nodeToHashMap[nodeID] = sketch
	}
	return nodeToHashMap
}

// NewVertexBottomKSketch creates a new bottom-k sketch
func NewVertexBottomKSketch(nodeId, k, nk int64) *VertexBottomKSketch {
	sketches := make([][]uint32, nk)
	for i := range sketches {
		sketches[i] = make([]uint32, k)
		for j := range sketches[i] {
			sketches[i][j] = math.MaxUint32
		}
	}
	
	return &VertexBottomKSketch{
		sketches:    sketches,
		k:           k,
		nk:          nk,
		nodeId:      nodeId,
		filledCount: 0,
	}
}

// IsSketchFull checks if sketch has enough values for estimation
func (vbs *VertexBottomKSketch) IsSketchFull() bool {
	return vbs.filledCount >= vbs.k
}

// EstimateCardinality estimates the cardinality using Bottom-K sketch
func (vbs *VertexBottomKSketch) EstimateCardinality() float64 {
	if vbs.filledCount < vbs.k {
		return float64(vbs.filledCount) // Not full, use exact count
	} else {
		// Full sketch, use estimation formula
		sum := uint32(0)
		for layer := int64(0); layer < vbs.nk; layer++ {
			if vbs.sketches[layer][vbs.k-1] != math.MaxUint32 {
				sum += vbs.sketches[layer][vbs.k-1]
			}
		}
		
		if vbs.nk == 0 {
			return 0.0
		}
		
		return float64(vbs.k-1) * float64(math.MaxUint32) / (float64(sum) * float64(vbs.nk))
	}
}

// GetSketch returns the sketch for a specific layer
func (vbs *VertexBottomKSketch) GetSketch(layer int64) []uint32 {
	if layer >= vbs.nk {
		return nil
	}
	return vbs.sketches[layer]
}

// GetLayerHashes returns the non-max hashes for a specific layer
func (vbs *VertexBottomKSketch) GetLayerHashes(layer int64) []uint32 {
	if layer < 0 || layer >= vbs.nk {
		return nil
	}
	
	hashes := make([]uint32, 0, vbs.k)
	for _, hash := range vbs.sketches[layer] {
		if hash != math.MaxUint32 {
			hashes = append(hashes, hash)
		}
	}
	return hashes
}

// GetFilledCount returns the filled count for this sketch
func (vbs *VertexBottomKSketch) GetFilledCount() int64 {
	return vbs.filledCount
}

// UpdateFilledCount updates the filled count
func (vbs *VertexBottomKSketch) UpdateFilledCount() {
	count := int64(0)
	for i := int64(0); i < vbs.k; i++ {
		if vbs.sketches[0][i] != math.MaxUint32 {
			count++
		} else {
			break
		}
	}
	vbs.filledCount = count
}

// GetK returns the size of the sketch
func (vbs *VertexBottomKSketch) GetK() int64 {
	return vbs.k
}

// GetNk returns the number of layers in the sketch
func (vbs *VertexBottomKSketch) GetNk() int64 {
	return vbs.nk
}

// NewSketchGraph creates a new sketch graph
func NewSketchGraph(numNodes int) *SketchGraph {
	return &SketchGraph{
		NumNodes:      numNodes,
		TotalWeight:   0.0,
		degrees:       make([]float64, numNodes),
		sketchManager: NewSketchManager(10, 4), // Default k=10, nk=4
		adjacencyList: make([][]WeightedEdge, numNodes),
	}
}

// GetDegree returns the degree of a node (SKETCH-BASED implementation of graph.Degrees[])
// func (sg *SketchGraph) GetDegree(node int) float64 {
// 	if node < 0 || node >= sg.NumNodes {
// 		return 0.0
// 	}
	
// 	sketch := sg.sketchManager.GetVertexSketch(int64(node))
// 	if sketch != nil && sketch.IsSketchFull() {
// 		return sketch.EstimateCardinality() // Sketch estimation
// 	} else {
// 		// Exact calculation from adjacency list
// 		degree := 0.0
// 		for _, edge := range sg.adjacencyList[node] {
// 			if edge.Neighbor == node {
// 				degree += edge.Weight * 2 // Self-loop counts double
// 			} else {
// 				degree += edge.Weight
// 			}
// 		}
// 		return degree
// 	}
// }

func (sg *SketchGraph) GetDegree(node int) float64 {
	if node < 0 || node >= sg.NumNodes {
		return 0.0
	}
	return sg.degrees[node]
}

func (sg *SketchGraph) CalculateAndStoreDegrees() {
	for node := 0; node < sg.NumNodes; node++ {
		sketch := sg.sketchManager.GetVertexSketch(int64(node))
		if sketch != nil && sketch.IsSketchFull() {
			// Use sketch estimation
			sg.degrees[node] = sketch.EstimateCardinality()
		} else {
			// Use exact calculation from adjacency list
			degree := 0.0
			for _, edge := range sg.adjacencyList[node] {
				if edge.Neighbor == node {
					degree += edge.Weight * 2 // Self-loop counts double
				} else {
					degree += edge.Weight
				}
			}
			sg.degrees[node] = degree
		}
	}
}


// GetEdgeWeight returns the weight of edge between two nodes (SKETCH-BASED)
func (sg *SketchGraph) GetEdgeWeight(u, v int) float64 {
	if u < 0 || u >= sg.NumNodes || v < 0 || v >= sg.NumNodes {
		return 0.0
	}
	
	// Look in adjacency list for exact edges
	for _, edge := range sg.adjacencyList[u] {
		if edge.Neighbor == v {
			return edge.Weight
		}
	}
	
	return 0.0
}

// GetNeighbors returns neighbors and their edge weights (SKETCH-BASED implementation of graph.GetNeighbors())
func (sg *SketchGraph) GetNeighbors(node int) ([]int, []float64) {
	if node < 0 || node >= sg.NumNodes {
		return nil, nil
	}
	
	sketch := sg.sketchManager.GetVertexSketch(int64(node))
	if sketch != nil && sketch.IsSketchFull() {
		// Use sketch-based neighbor discovery
		return sg.getSketchNeighbors(node)
	} else {
		// Use exact adjacency list
		neighbors := make([]int, len(sg.adjacencyList[node]))
		weights := make([]float64, len(sg.adjacencyList[node]))
		
		for i, edge := range sg.adjacencyList[node] {
			neighbors[i] = edge.Neighbor
			weights[i] = edge.Weight
		}
		
		return neighbors, weights
	}
}

// getSketchNeighbors finds neighbors using sketch hashes
func (sg *SketchGraph) getSketchNeighbors(node int) ([]int, []float64) {
	sketch := sg.sketchManager.GetVertexSketch(int64(node))
	if sketch == nil {
		return nil, nil
	}
	
	neighborWeights := make(map[int]float64)
	
	// Get hashes from first layer only (for simplicity)
	hashes := sketch.GetLayerHashes(0)
	for _, hash := range hashes {
		if neighborNodeId, exists := sg.sketchManager.GetNodeFromHash(hash); exists {
			neighborWeights[int(neighborNodeId)] += 1.0
		}
	}
	
	// Convert to slices
	neighbors := make([]int, 0, len(neighborWeights))
	weights := make([]float64, 0, len(neighborWeights))
	
	for neighbor, weight := range neighborWeights {
		neighbors = append(neighbors, neighbor)
		weights = append(weights, weight)
	}
	
	return neighbors, weights
}

// Validate checks graph consistency (PARALLEL to Graph.Validate())
func (sg *SketchGraph) Validate() error {
	if sg.NumNodes <= 0 {
		return fmt.Errorf("graph must have positive number of nodes")
	}
	
	if sg.sketchManager == nil {
		return fmt.Errorf("sketch manager cannot be nil")
	}
	
	// Validate adjacency list
	for i := 0; i < sg.NumNodes; i++ {
		for _, edge := range sg.adjacencyList[i] {
			if edge.Neighbor < 0 || edge.Neighbor >= sg.NumNodes {
				return fmt.Errorf("invalid neighbor %d for node %d", edge.Neighbor, i)
			}
			
			if edge.Weight <= 0 {
				return fmt.Errorf("non-positive weight %f for edge %d-%d", edge.Weight, i, edge.Neighbor)
			}
		}
	}
	
	// Validate sketch consistency
	sketchCount := len(sg.sketchManager.vertexSketches)
	if sketchCount == 0 {
		return fmt.Errorf("no sketches found - ensure preprocessing completed successfully")
	}
	
	return nil
}


// UnionWith returns the union of this sketch with another sketch (NEEDED for inclusion-exclusion)
func (vbs *VertexBottomKSketch) UnionWith(other *VertexBottomKSketch) *VertexBottomKSketch {
	if other == nil || vbs.nk != other.nk || vbs.k != other.k {
		return nil // Invalid input or incompatible dimensions
	}
	
	unionSketch := NewVertexBottomKSketch(vbs.nodeId, vbs.k, vbs.nk)
	for layer := int64(0); layer < vbs.nk; layer++ {
		unionSketch.sketches[layer] = vbs.bottomKUnion(vbs.sketches[layer], other.sketches[layer])
	}
	unionSketch.UpdateFilledCount()
	return unionSketch
}

// UnionWithLayer performs Bottom-K union with another sketch layer (NEEDED for community sketch updates)
func (vbs *VertexBottomKSketch) UnionWithLayer(layer int64, otherSketch []uint32) {
	if layer >= vbs.nk {
		return
	}
	
	currentSketch := vbs.sketches[layer]
	vbs.sketches[layer] = vbs.bottomKUnion(currentSketch, otherSketch)
	vbs.UpdateFilledCount()
}

// bottomKUnion performs Bottom-K union of two sketches (CORE operation)
func (vbs *VertexBottomKSketch) bottomKUnion(sketch1, sketch2 []uint32) []uint32 {
	result := make([]uint32, vbs.k)
	for i := range result {
		result[i] = math.MaxUint32
	}
	
	i, j, t := 0, 0, 0
	
	for t < int(vbs.k) {
		val1 := uint32(math.MaxUint32)
		val2 := uint32(math.MaxUint32)

		if i < len(sketch1) {
			val1 = sketch1[i]
		}
		if j < len(sketch2) {
			val2 = sketch2[j]
		}
		
		if val1 == val2 && val1 != math.MaxUint32 {
			result[t] = val1
			t++
			i++
			j++
		} else if i < len(sketch1) && (j >= len(sketch2) || val1 < val2) {
			result[t] = val1
			t++
			i++
		} else if j < len(sketch2) {
			result[t] = val2
			t++
			j++
		} else {
			break
		}
	}
	
	return result
}


// AggregateFromPreviousLevel creates sketch aggregation for super-nodes
func (sg *SketchGraph) AggregateFromPreviousLevel(prevGraph *SketchGraph, commToSuper map[int]int, comm *Community) error {
	// Step 1: Create super-node sketches (union of member sketches)
	for commId, superNodeId := range commToSuper {
		memberNodes := comm.CommunityNodes[commId]
		if len(memberNodes) == 0 {
			continue
		}
		
		// Get first valid sketch to initialize dimensions
		var firstSketch *VertexBottomKSketch
		for _, nodeId := range memberNodes {
			if sketch := prevGraph.sketchManager.GetVertexSketch(int64(nodeId)); sketch != nil {
				firstSketch = sketch
				break
			}
		}
		
		if firstSketch == nil {
			continue
		}
		
		// Create super-node sketch as union of all member sketches
		superSketch := NewVertexBottomKSketch(int64(superNodeId), firstSketch.k, firstSketch.nk)
		
		for _, nodeId := range memberNodes {
			nodeSketch := prevGraph.sketchManager.GetVertexSketch(int64(nodeId))
			if nodeSketch != nil {
				for layer := int64(0); layer < nodeSketch.nk; layer++ {
					superSketch.UnionWithLayer(layer, nodeSketch.GetSketch(layer))
				}
			}
		}
		
		sg.sketchManager.vertexSketches[int64(superNodeId)] = superSketch

		sg.degrees[superNodeId] = comm.CommunityWeights[commId]	
	}
	
	// Step 2: Build hash-to-node mapping for super-nodes
	for hash, originalNodeId := range prevGraph.sketchManager.hashToNodeMap {
		originalComm := comm.NodeToCommunity[int(originalNodeId)]
		if superNodeId, exists := commToSuper[originalComm]; exists {
			sg.sketchManager.hashToNodeMap[hash] = int64(superNodeId)
		}
	}

    for commId, superNodeId := range commToSuper {
        if communitySketch := comm.communitySketches[commId]; communitySketch != nil {
            sg.sketchManager.nodeToHashMap[int64(superNodeId)] = communitySketch
        }
    }
	
	// Step 3: Build adjacency for super-nodes (exact method aggregation)
	totalWeight := 0.0
	adjacency := make(map[int]map[int]float64)
	
	for _, superNodeId := range commToSuper {
		adjacency[superNodeId] = make(map[int]float64)
	}
	
	for node := 0; node < prevGraph.NumNodes; node++ {
		if _, exists := commToSuper[comm.NodeToCommunity[node]]; exists {
			totalWeight += prevGraph.degrees[node]
		}
	}

	// Step 3a: Create reverse mapping and separate super-nodes by sketch fullness
	superToComm := make(map[int]int)
	nonFullSuperNodes := make([]int, 0)

	for commId, superNodeId := range commToSuper {
		superToComm[superNodeId] = commId // Create reverse mapping
		
		memberNodes := comm.CommunityNodes[commId]
		if len(memberNodes) == 0 {
			continue
		}
		
		// Check if SUPER-NODE sketch is full
		superNodeSketch := sg.sketchManager.GetVertexSketch(int64(superNodeId))
		if superNodeSketch == nil || !superNodeSketch.IsSketchFull() {
			nonFullSuperNodes = append(nonFullSuperNodes, superNodeId)
		}
		// Note: Full super-nodes are skipped for adjacency building
	}

	// Step 3b: Only process nodes from non-full super-nodes
	for _, superNodeId := range nonFullSuperNodes {
		communityId := superToComm[superNodeId]
		memberNodes := comm.CommunityNodes[communityId]
		currentSupernode := superNodeId
		
		for _, nodeId := range memberNodes {
			// Get neighbors using exact adjacency (since super-node sketch not full)
			neighbors, weights := prevGraph.GetNeighbors(nodeId)
			for i, neighbor := range neighbors {
				neighborComm := comm.NodeToCommunity[neighbor]
				otherSupernode, exists := commToSuper[neighborComm]
				if !exists {
					continue
				}
				
				if currentSupernode == otherSupernode {
					if nodeId == neighbor {
						// Actual self-loop: full weight
						adjacency[currentSupernode][otherSupernode] += weights[i]
					} else {
						// Internal community edge: half weight
						adjacency[currentSupernode][otherSupernode] += weights[i] / 2.0
					}
				} else {
					// Edge between different super-nodes: full weight
					adjacency[currentSupernode][otherSupernode] += weights[i]
				}
			}
		}
	}

	
	// Step 4: Convert to adjacency list format (NO division by 2.0)
	for superNodeA, neighbors := range adjacency {
		for superNodeB, weight := range neighbors {
			if weight > 0 {
				sg.addAdjacencyEdge(superNodeA, superNodeB, weight)
			}
		}
	}
	
	sg.TotalWeight = totalWeight / 2.0 // Undirected graph
	
	return nil
}

// EstimateEdgesToCommunity 
func (sg *SketchGraph) EstimateEdgesToCommunity(node, commId int, comm *Community) float64 {
	if node < 0 || node >= sg.NumNodes || commId < 0 {
		return 0.0
	}
	
	nodeSketch := sg.sketchManager.GetVertexSketch(int64(node))
	if nodeSketch == nil {
		return 0.0
	}

	
	communitySketch := comm.communitySketches[commId]
	if communitySketch == nil {
		return 0.0
	}

	if nodeSketch.IsSketchFull() && comm.NodeToCommunity[node] == commId && len(comm.CommunityNodes[commId]) == 1 && len(comm.communitySketches) == 1 {
		return 0.0
	}
	
	if !nodeSketch.IsSketchFull() {
		return sg.countExactEdgesToCommunity(node, commId, comm)
	} else {
		return sg.estimateEdgesViaInclusion(nodeSketch, communitySketch, commId, comm)
	}
}


// countExactEdgesToCommunity counts edges directly via adjacency list
func (sg *SketchGraph) countExactEdgesToCommunity(node, commId int, comm *Community) float64 {
	edgeCount := 0.0
	for _, edge := range sg.adjacencyList[node] {
		if edge.Neighbor < len(comm.NodeToCommunity) && comm.NodeToCommunity[edge.Neighbor] == commId {
			edgeCount += edge.Weight
		}
	}
	return edgeCount
}



// estimateEdgesViaInclusion uses inclusion-exclusion principle
func (sg *SketchGraph) estimateEdgesViaInclusion(nodeSketch, communitySketch *VertexBottomKSketch, commId int, comm *Community) float64 {
    nodeDegree := nodeSketch.EstimateCardinality()
    communitySize := float64(len(comm.CommunityNodes[commId])) 
    
    unionSketch := nodeSketch.UnionWith(communitySketch)
    if unionSketch == nil {
        return 0.0
    }
    
    unionDegree := unionSketch.EstimateCardinality()
    edges := nodeDegree + communitySize - unionDegree

    return math.Max(0, edges)
}

// FindNeighboringCommunities finds all communities neighboring a given node
func (sg *SketchGraph) FindNeighboringCommunities(node int, comm *Community) map[int]float64 {
	neighborComms := make(map[int]float64)
	
	nodeSketch := sg.sketchManager.GetVertexSketch(int64(node))
	if nodeSketch != nil && nodeSketch.IsSketchFull() {
		for commId := 0; commId < comm.NumCommunities; commId++ {
			if len(comm.CommunityNodes[commId]) > 0 {
				edgeWeight := sg.EstimateEdgesToCommunity(node, commId, comm)
				neighborComms[commId] = edgeWeight
				
			}
		}
	} else {
		// EXACT METHOD: Only check actual neighbors
		for _, edge := range sg.adjacencyList[node] {
			neighborComm := comm.NodeToCommunity[edge.Neighbor]
			neighborComms[neighborComm] += edge.Weight
		}
	}
	
	// Always include current community
	currentComm := comm.NodeToCommunity[node]
	if _, exists := neighborComms[currentComm]; !exists {
		neighborComms[currentComm] = 0.0
	}
	
	return neighborComms
}

// EstimateCommunityCardinality estimates the cardinality of a community
func (sg *SketchGraph) EstimateCommunityCardinality(commId int, comm *Community) float64 {
	communitySketch := comm.communitySketches[commId]
	if communitySketch == nil {
		return 0.0
	}

	degreeSum := 0.0
	for _, nodeId := range comm.CommunityNodes[commId] {
		degreeSum += sg.degrees[nodeId]
	}
	return degreeSum
}
// ================================================================================================
// Add a node's identifying hash sketch to a community sketch
func (sg *SketchGraph) addNodeHashToCommunitySketch(nodeId, commId int, comm *Community) {
    nodeHashSketch := sg.sketchManager.nodeToHashMap[int64(nodeId)]
    if nodeHashSketch == nil {
        return
    }
    
    communitySketch := comm.communitySketches[commId]
    if communitySketch == nil {
        // No existing community sketch, just reference the node's hash sketch
        comm.communitySketches[commId] = nodeHashSketch
        return
    }
    
    // Union node's hash sketch into existing community sketch
    for layer := int64(0); layer < communitySketch.nk; layer++ {
        communitySketch.UnionWithLayer(layer, nodeHashSketch.GetSketch(layer))
    }
    communitySketch.UpdateFilledCount()
}

// Remove a node's identifying hash sketch from a community sketch
func (sg *SketchGraph) removeNodeHashFromCommunitySketch(nodeId, commId int, comm *Community) {
    nodeHashSketch := sg.sketchManager.nodeToHashMap[int64(nodeId)]
    if nodeHashSketch == nil {
        return
    }
    
    communitySketch := comm.communitySketches[commId]
    if communitySketch == nil {
        return
    }
    
    // Remove node's hashes from each layer
    for layer := int64(0); layer < communitySketch.nk; layer++ {
        nodeLayerHashes := nodeHashSketch.GetSketch(layer)
        communityLayer := communitySketch.sketches[layer]
        
        // Remove each hash from this layer
        for _, hashToRemove := range nodeLayerHashes {
            if hashToRemove == math.MaxUint32 {
                break
            }
            sg.removeHashFromSortedArray(communityLayer, hashToRemove)
        }
    }
    
    communitySketch.UpdateFilledCount()
}

// Helper function to remove specific hash from sorted array
func (sg *SketchGraph) removeHashFromSortedArray(sortedArray []uint32, hashToRemove uint32) {
    for i := 0; i < len(sortedArray); i++ {
        if sortedArray[i] == hashToRemove {
            // Shift elements left
            for j := i; j < len(sortedArray)-1; j++ {
                sortedArray[j] = sortedArray[j+1]
            }
            sortedArray[len(sortedArray)-1] = math.MaxUint32
            break
        }
        if sortedArray[i] == math.MaxUint32 {
            break // Hash not found
        }
    }
}

// addAdjacencyEdge adds an edge to the adjacency list (helper function)
func (sg *SketchGraph) addAdjacencyEdge(u, v int, weight float64) {
	// Check if edge already exists and accumulate weight
	for i := range sg.adjacencyList[u] {
		if sg.adjacencyList[u][i].Neighbor == v {
			sg.adjacencyList[u][i].Weight += weight
			return
		}
	}
	
	// Add new edge
	sg.adjacencyList[u] = append(sg.adjacencyList[u], WeightedEdge{
		Neighbor: v,
		Weight:   weight,
	})
}


// buildAdjacencyList builds the adjacency list from sketches
func (sg *SketchGraph) buildAdjacencyList(rawGraph *RawGraph, nodeMapping *NodeMapping) {
	// Only build adjacency for target nodes using compressed IDs
	for _, compressedId := range nodeMapping.OriginalToCompressed {
		sketch := sg.sketchManager.GetVertexSketch(int64(compressedId))
		if sketch == nil {
			continue
		}
		
		if !sketch.IsSketchFull() {
			neighborWeights := make(map[int]float64)
			hashes := sketch.GetLayerHashes(0) // Use first layer
			
			for _, hash := range hashes {
				if compressedNeighborId, exists := sg.sketchManager.GetNodeFromHash(hash); exists {
					neighborWeights[int(compressedNeighborId)] += 1.0
				}
			}
			
			// Convert to WeightedEdge slice using compressed IDs
			for neighbor, weight := range neighborWeights {
				sg.adjacencyList[compressedId] = append(sg.adjacencyList[compressedId], WeightedEdge{
					Neighbor: neighbor,
					Weight:   weight,
				})
			}
		}
	}

	sg.CalculateAndStoreDegrees()
	
	// Calculate total weight using stored degrees
	totalWeight := 0.0
	for i := 0; i < sg.NumNodes; i++ {
		totalWeight += sg.degrees[i]
	}
	sg.TotalWeight = totalWeight / 2.0 // Undirected graph
}









// ============================ DEBUGGING ============================


// GetNeighborsInclusionExclusion returns ALL nodes as neighbors with inclusion-exclusion weights
func (sg *SketchGraph) GetNeighborsInclusionExclusion(node int, threshold float64) ([]int, []float64) {
    if node < 0 || node >= sg.NumNodes {
        return nil, nil
    }
    
    nodeSketch := sg.sketchManager.GetVertexSketch(int64(node))
    if nodeSketch == nil || !nodeSketch.IsSketchFull() {
        // Fall back to exact method for non-full sketches
        return sg.getExactNeighbors(node)
    }
    
    // Create weighted clique: estimate edge weight to ALL other nodes
    neighbors := make([]int, 0, sg.NumNodes-1)
    weights := make([]float64, 0, sg.NumNodes-1)
    
    for otherNode := 0; otherNode < sg.NumNodes; otherNode++ {
        if otherNode == node {
            continue // Skip self
        }
        
        otherSketch := sg.sketchManager.GetVertexSketch(int64(otherNode))
        if otherSketch == nil {
            continue
        }
        
        // Estimate edge weight using inclusion-exclusion
        weight := sg.estimateEdgeWeightInclusionExclusion(nodeSketch, otherSketch)
        
        // Only include if weight is above threshold
        if weight > threshold {
            neighbors = append(neighbors, otherNode)
            weights = append(weights, weight)
        }
    }
    
    return neighbors, weights
}


// estimateEdgeWeightInclusionExclusion estimates edge weight between two nodes using sketches
func (sg *SketchGraph) estimateEdgeWeightInclusionExclusion(sketchA, sketchB *VertexBottomKSketch) float64 {
    if sketchA == nil || sketchB == nil {
        return 0.0
    }
    
    // Get individual cardinalities
    cardinalityA := sketchA.EstimateCardinality()
    cardinalityB := sketchB.EstimateCardinality()
    
    // Compute union
    unionSketch := sketchA.UnionWith(sketchB)
    if unionSketch == nil {
        return 0.0
    }
    
    unionCardinality := unionSketch.EstimateCardinality()
    
    // Inclusion-exclusion: |A ∩ B| = |A| + |B| - |A ∪ B|
    intersection := cardinalityA + cardinalityB - unionCardinality
    
    // Return non-negative intersection as edge weight
    return math.Max(0, intersection)
}

// getExactNeighbors falls back to adjacency list for non-full sketches
func (sg *SketchGraph) getExactNeighbors(node int) ([]int, []float64) {
    neighbors := make([]int, len(sg.adjacencyList[node]))
    weights := make([]float64, len(sg.adjacencyList[node]))
    
    for i, edge := range sg.adjacencyList[node] {
        neighbors[i] = edge.Neighbor
        weights[i] = edge.Weight
    }
    
    return neighbors, weights
}

// SetSketchManager sets the sketch manager
func (sg *SketchGraph) SetSketchManager(sm *SketchManager) {
	sg.sketchManager = sm
}

// SetSketchValue sets a value in the sketch at specific layer and index
func (vbs *VertexBottomKSketch) SetSketchValue(layer, index int64, value uint32) {
	if layer >= 0 && layer < vbs.nk && index >= 0 && index < vbs.k {
		vbs.sketches[layer][index] = value
	}
}

// SetVertexSketch stores a vertex sketch in the manager
func (sm *SketchManager) SetVertexSketch(nodeId int64, sketch *VertexBottomKSketch) {
	sm.vertexSketches[nodeId] = sketch
}

// SetHashToNode sets a hash-to-node mapping
func (sm *SketchManager) SetHashToNode(hash uint32, nodeId int64) {
	sm.hashToNodeMap[hash] = nodeId
}

// SetNodeToHashMap sets a node-to-hash mapping
func (sm *SketchManager) SetNodeToHashMap(nodeId int64, sketch *VertexBottomKSketch) {
	sm.nodeToHashMap[nodeId] = sketch
}

// SetTotalWeight sets the total weight of the sketch graph
func (sg *SketchGraph) SetTotalWeight(weight float64) {
	if weight < 0 {
		weight = 0.0 // Ensure non-negative weight
	}
	sg.TotalWeight = weight
}

// AddEdgeToAdjacencyList adds an edge to the adjacency list
func (sg *SketchGraph) AddEdgeToAdjacencyList(from, to int, weight float64) {
	sg.adjacencyList[from] = append(sg.adjacencyList[from], WeightedEdge{
		Neighbor: to,
		Weight:   weight,
	})
}

// GetSketchManager returns the sketch manager (for testing)
func (sg *SketchGraph) GetSketchManager() *SketchManager {
	return sg.sketchManager
}

// GetFilledCount returns the filled count for a vertex sketch (for testing)
func (sg *SketchGraph) GetFilledCount(nodeId int64) int64 {
	sketch := sg.sketchManager.GetVertexSketch(nodeId)
	if sketch != nil {
		return sketch.filledCount
	}
	return 0
}

// GetCommunitySketches returns the community sketches (for testing)
func (sg *SketchGraph) GetCommunitySketches() map[int]*VertexBottomKSketch {
	communitySketches := make(map[int]*VertexBottomKSketch)
	for commId, sketch := range sg.sketchManager.vertexSketches {
		if sketch != nil {
			communitySketches[int(commId)] = sketch
		}
	}
	return communitySketches
}

func (sg *SketchGraph) PrintDebug() {
	fmt.Println("=== SKETCH GRAPH DEBUG ===")
	fmt.Printf("Nodes: %d\n", sg.NumNodes)
	fmt.Printf("Total Weight: %.2f\n", sg.TotalWeight)
	fmt.Printf("Sketch Manager - k: %d, nk: %d\n", sg.sketchManager.k, sg.sketchManager.nk)
	
	// Print vertex sketches
	fmt.Println("\n--- VERTEX SKETCHES ---")
	if len(sg.sketchManager.vertexSketches) == 0 {
		fmt.Println("No vertex sketches found!")
	} else {
		for nodeId, sketch := range sg.sketchManager.vertexSketches {
			fmt.Printf("Node %d: filled=%d, full=%v\n", nodeId, sketch.filledCount, sketch.IsSketchFull())
			
			for layer := int64(0); layer < sketch.nk; layer++ {
				fmt.Printf("  Layer %d: [", layer)
				hashes := sketch.GetLayerHashes(layer)
				
				if len(hashes) == 0 {
					fmt.Printf("empty")
				} else {
					// Convert hashes to node owners
					owners := []string{}
					for _, hash := range hashes {
						if ownerNode, exists := sg.sketchManager.GetNodeFromHash(hash); exists {
							owners = append(owners, fmt.Sprintf("N%d", ownerNode))
						} else {
							owners = append(owners, fmt.Sprintf("?%d", hash))
						}
					}
					fmt.Printf("%s", strings.Join(owners, ", "))
				}
				fmt.Printf("]\n")
			}
		}
	}
	
	// Print hash-to-node mapping
	fmt.Println("\n--- HASH TO NODE MAPPING ---")
	if len(sg.sketchManager.hashToNodeMap) == 0 {
		fmt.Println("No hash mappings found!")
	} else {
		fmt.Printf("Total mappings: %d\n", len(sg.sketchManager.hashToNodeMap))
		// Show first 10 mappings as sample
		count := 0
		for hash, nodeId := range sg.sketchManager.hashToNodeMap {
			if count < 10 {
				fmt.Printf("  Hash %d -> Node %d\n", hash, nodeId)
				count++
			} else if count == 10 {
				fmt.Printf("  ... and %d more mappings\n", len(sg.sketchManager.hashToNodeMap)-10)
				break
			}
		}
	}
	
	// Print adjacency list
	fmt.Println("\n--- ADJACENCY LIST ---")
	hasAdjacency := false
	for i := 0; i < sg.NumNodes; i++ {
		if len(sg.adjacencyList[i]) > 0 {
			hasAdjacency = true
			fmt.Printf("Node %d -> ", i)
			for j, edge := range sg.adjacencyList[i] {
				if j > 0 { fmt.Printf(", ") }
				fmt.Printf("%d(%.1f)", edge.Neighbor, edge.Weight)
			}
			fmt.Println()
		}
	}
	if !hasAdjacency {
		fmt.Println("No adjacency list entries found!")
	}
	
	// Print node degrees
	fmt.Println("\n--- NODE DEGREES ---")
	for i := 0; i < sg.NumNodes; i++ { // Show all nodes
		degree := sg.degrees[i]
		if degree > 0 {
			fmt.Printf("Node %d: degree %.2f\n", i, degree)
		}
	}
	
	fmt.Println("=== END SKETCH GRAPH DEBUG ===")
}


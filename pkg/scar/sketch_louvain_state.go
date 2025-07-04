package scar

// import (
// 	"fmt"
// )
// SketchLouvainState manages community membership and delegates sketch operations
type SketchLouvainState struct {
	nodeToCommunity   []int64               // node -> community mapping
	communityToNodes  map[int64][]int64     // community -> nodes reverse mapping  
	activeCommunities map[int64]bool        // which communities exist
	sketchManager     *SketchManager        // delegate sketch operations
	totalWeight       float64               // total weight of the graph (if needed)
	n                  int64                  // number of nodes in the graph
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
	}

	return sls
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

// Estimate community cardinality via delegation
func (sls *SketchLouvainState) EstimateCommunityCardinality(commId int64) float64 {
	memberNodes := sls.GetCommunityNodes(commId)
	return sls.sketchManager.EstimateCommunityCardinality(commId, memberNodes)
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

	for _, sketch := range sls.sketchManager.vertexSketches {
		totalWeight += sketch.EstimateCardinality()
	}
	sls.totalWeight = totalWeight / 2
}
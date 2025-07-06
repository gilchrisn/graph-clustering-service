package scar

import (
	"fmt"
	// "math"
	// "math/rand"
)
// SketchManager manages all vertex sketches and provides operations
type SketchManager struct {
	vertexSketches map[int64]*VertexBottomKSketch
	hashToNodeMap  map[uint32]int64
	communitySketches map[int64]*VertexBottomKSketch // For community sketches
	k              int64
	nk             int64
}

func NewSketchManager(k, nk int64) *SketchManager {
	return &SketchManager{
		vertexSketches: make(map[int64]*VertexBottomKSketch),
		hashToNodeMap:  make(map[uint32]int64),
		communitySketches: make(map[int64]*VertexBottomKSketch),
		k:              k,
		nk:             nk,
	}
}

// CreateVertexSketch creates a new sketch for a vertex
func (sm *SketchManager) CreateVertexSketch(nodeId int64, layerValues []uint32) {
	sketch := NewVertexBottomKSketch(nodeId, sm.k, sm.nk)
	sketch.Initialize(layerValues)
	sm.vertexSketches[nodeId] = sketch
	
	// Build hash to node mapping (original + 1 for hash values)
	for _, val := range layerValues {
		sm.hashToNodeMap[val+1] = nodeId
		fmt.Printf("HashToNodeMap[%d] = %d\n", val+1, nodeId)
	}
}

// GetVertexSketch returns the sketch for a vertex
func (sm *SketchManager) GetVertexSketch(nodeId int64) *VertexBottomKSketch {
	return sm.vertexSketches[nodeId]
}

// GetCommunitySketch returns the sketch for a community	
func (sm *SketchManager) GetCommunitySketch(communityId int64) *VertexBottomKSketch {
	return sm.communitySketches[communityId]
}

// DeleteCommunitySketch deletes the sketch for a community
func (sm *SketchManager) DeleteCommunitySketch(commId int64) {
	delete(sm.communitySketches, commId)
}

// UnionVertexSketch performs union operation on a vertex sketch
func (sm *SketchManager) UnionVertexSketch(targetNodeId, sourceNodeId int64, layer int64) {
	targetSketch := sm.vertexSketches[targetNodeId]
	sourceSketch := sm.vertexSketches[sourceNodeId]
	
	if targetSketch != nil && sourceSketch != nil {
		sourceLayerSketch := sourceSketch.GetSketch(layer)
		targetSketch.UnionWithLayer(layer, sourceLayerSketch)
	}
}

// AddOneToAllSketches adds 1 to all sketch values
func (sm *SketchManager) AddOneToAllSketches() {
	for _, sketch := range sm.vertexSketches {
		sketch.AddOne()
	}
}

// GetNodeFromHash returns the node ID for a hash value
func (sm *SketchManager) GetNodeFromHash(hashValue uint32) (int64, bool) {
	nodeId, exists := sm.hashToNodeMap[hashValue]
	return nodeId, exists
}

// CreateNodeToSketchMapping creates mapping from node to sketch values (excluding self)
func (sm *SketchManager) CreateNodeToSketchMapping() map[int64][]uint32 {
	nodeToSketch := make(map[int64][]uint32)
	
	for nodeId, sketch := range sm.vertexSketches {
		allSketches := sketch.GetAllSketches()
		var filteredSketches []uint32
		
		for _, sketchValue := range allSketches {
			if otherNodeId, exists := sm.hashToNodeMap[sketchValue]; exists {
				if otherNodeId != nodeId { // Exclude self-loops
					filteredSketches = append(filteredSketches, sketchValue)
				}
			}
		}
		
		if len(filteredSketches) > 0 {
			nodeToSketch[nodeId] = filteredSketches
		}
	}
	
	return nodeToSketch
}

func (sm *SketchManager) UpdateCommunitySketch(commId int64, memberNodes []int64) {
	communitySketch := NewVertexBottomKSketch(commId, sm.k, sm.nk)
	
	// Union all member node sketches
	for _, nodeId := range memberNodes {
		nodeSketch := sm.GetVertexSketch(nodeId)
		if nodeSketch != nil {
			for layer := int64(0); layer < sm.nk; layer++ {
				communitySketch.UnionWithLayer(layer, nodeSketch.GetSketch(layer))
			}
		}
	}
	
	sm.communitySketches[commId] = communitySketch
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
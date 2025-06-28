package scar

import (
	"fmt"
	"math"
	// "math/rand"
)
// SketchManager manages all vertex sketches and provides operations
type SketchManager struct {
	vertexSketches map[int64]*VertexBottomKSketch
	hashToNodeMap  map[uint32]int64
	k              int64
	nk             int64
}

func NewSketchManager(k, nk int64) *SketchManager {
	return &SketchManager{
		vertexSketches: make(map[int64]*VertexBottomKSketch),
		hashToNodeMap:  make(map[uint32]int64),
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

// updateForSuperGraph rebuilds sketch manager for super-graph nodes
func (sm *SketchManager) updateForSuperGraph(newNodeToSketch map[int64][]uint32) {
	fmt.Printf("=== UPDATING SKETCH MANAGER FOR SUPER-GRAPH ===\n")
	
	// Clear old mappings
	oldVertexCount := len(sm.vertexSketches)
	oldHashCount := len(sm.hashToNodeMap)
	sm.vertexSketches = make(map[int64]*VertexBottomKSketch)
	sm.hashToNodeMap = make(map[uint32]int64)
	
	// Rebuild with super-nodes
	for superNodeId, sketchValues := range newNodeToSketch {
		// Create new vertex sketch for super-node
		superSketch := NewVertexBottomKSketch(superNodeId, sm.k, sm.nk)
		
		// Set first layer sketch values (using only layer 0 for simplicity)
		if len(sketchValues) > 0 {
			copyLen := len(sketchValues)
			if copyLen > int(sm.k) {
				copyLen = int(sm.k)
			}
			copy(superSketch.sketches[0], sketchValues[:copyLen])
		}
		
		sm.vertexSketches[superNodeId] = superSketch
		
		// Rebuild hash mapping
		for _, hashValue := range sketchValues {
			if hashValue != math.MaxUint32 {
				sm.hashToNodeMap[hashValue] = superNodeId
			}
		}
		
		fmt.Printf("SuperNode %d: sketch values %v\n", superNodeId, sketchValues[:min(len(sketchValues), 5)])
	}
	
	fmt.Printf("Updated sketch manager: %d->%d vertices, %d->%d hash mappings\n", 
		oldVertexCount, len(sm.vertexSketches), oldHashCount, len(sm.hashToNodeMap))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
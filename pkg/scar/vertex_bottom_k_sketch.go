package scar

import (
	"fmt"
	"math"
)

// VertexBottomKSketch represents a Bottom-K sketch for a single vertex
type VertexBottomKSketch struct {
	sketches [][]uint32 // [nk][k] array - one sketch per layer
	k        int64      // sketch size
	nk       int64      // number of layers
	nodeId   int64      // which node this sketch belongs to
}

func NewVertexBottomKSketch(nodeId, k, nk int64) *VertexBottomKSketch {
	sketches := make([][]uint32, nk)
	for i := range sketches {
		sketches[i] = make([]uint32, k)
		for j := range sketches[i] {
			sketches[i][j] = math.MaxUint32
		}
	}
	
	return &VertexBottomKSketch{
		sketches: sketches,
		k:        k,
		nk:       nk,
		nodeId:   nodeId,
	}
}

// Initialize sketch with uniform random values for each layer
func (vbs *VertexBottomKSketch) Initialize(layerValues []uint32) {
	if len(layerValues) != int(vbs.nk) {
		panic("layerValues length must equal nk")
	}
	
	for layer := int64(0); layer < vbs.nk; layer++ {
		vbs.sketches[layer][0] = layerValues[layer]
		// Rest remain MaxUint32
	}
}

// GetSketch returns the sketch for a specific layer
func (vbs *VertexBottomKSketch) GetSketch(layer int64) []uint32 {
	if layer >= vbs.nk {
		return nil
	}
	return vbs.sketches[layer]
}

// GetAllSketches returns flattened view of all sketches
func (vbs *VertexBottomKSketch) GetAllSketches() []uint32 {
	result := make([]uint32, 0, vbs.k*vbs.nk)
	for layer := int64(0); layer < vbs.nk; layer++ {
		for i := int64(0); i < vbs.k; i++ {
			if vbs.sketches[layer][i] != math.MaxUint32 {
				result = append(result, vbs.sketches[layer][i])
			}
		}
	}
	return result
}

// Union performs Bottom-K union with another sketch layer
func (vbs *VertexBottomKSketch) UnionWithLayer(layer int64, otherSketch []uint32) {
	if layer >= vbs.nk {
		return
	}
	
	currentSketch := vbs.sketches[layer]
	vbs.sketches[layer] = vbs.bottomKUnion(currentSketch, otherSketch)
}

// bottomKUnion performs Bottom-K union of two sketches
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

// IsSketchFull checks if any layer has a full sketch
func (vbs *VertexBottomKSketch) IsSketchFull() bool {
	for layer := int64(0); layer < vbs.nk; layer++ {
		nonMaxCount := 0
		for i := int64(0); i < vbs.k; i++ {
			if vbs.sketches[layer][i] != math.MaxUint32 {
				nonMaxCount++
			} else {
				break
			}
		}
		if int64(nonMaxCount) >= vbs.k {
			return true
		}
	}
	return false
}

// EstimateCardinality estimates the cardinality using Bottom-K sketch
func (vbs *VertexBottomKSketch) EstimateCardinality() float64 {
	totalEstimate := 0.0
	
	nonMaxCount := 0
	for i := int64(0); i < vbs.k; i++ {
		if vbs.sketches[0][i] != math.MaxUint32 {
			nonMaxCount++
		} else {
			break
		}
	}
	
	if int64(nonMaxCount) < vbs.k {
		// Not full, use exact count
		totalEstimate += float64(nonMaxCount)
	} else {
		// Full sketch, use estimation formula
		rK := float64(vbs.sketches[0][vbs.k-1])
		if rK > 0 {
			estimate := float64(vbs.k-1) * float64(math.MaxUint32) / rK
			totalEstimate += estimate
		}
	}
	
	return totalEstimate
}

// AddOne adds 1 to all non-max values in all sketches
func (vbs *VertexBottomKSketch) AddOne() {
	for layer := int64(0); layer < vbs.nk; layer++ {
		for i := int64(0); i < vbs.k; i++ {
			if vbs.sketches[layer][i] != math.MaxUint32 {
				vbs.sketches[layer][i]++
			}
		}
	}
}

// String returns a string representation for debugging
func (vbs *VertexBottomKSketch) String() string {
	result := fmt.Sprintf("Node %d sketches: ", vbs.nodeId)
	for layer := int64(0); layer < vbs.nk; layer++ {
		result += fmt.Sprintf("layer%d=[", layer)
		for i := int64(0); i < vbs.k; i++ {
			if vbs.sketches[layer][i] == math.MaxUint32 {
				result += "MAX"
			} else {
				result += fmt.Sprintf("%d", vbs.sketches[layer][i])
			}
			if i < vbs.k-1 {
				result += ","
			}
		}
		result += "] "
	}
	return result
}

// ContainsHash checks if the sketch contains a specific hash value
func (vbs *VertexBottomKSketch) ContainsHash(targetHash uint32) bool {
	for layer := int64(0); layer < vbs.nk; layer++ {
		for _, hash := range vbs.sketches[layer] {
			if hash == targetHash {
				return true
			}
			if hash > targetHash {
				// Sketches are sorted, so we can stop early
				break
			}
		}
	}
	return false
}
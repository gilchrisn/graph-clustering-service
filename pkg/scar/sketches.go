package scar

import (
	"fmt"
	"math"
	"sort"
)

const UINT64_MAX = math.MaxUint64

// NewVertexBottomKSketch creates multi-sketch with nK independent hash functions
func NewVertexBottomKSketch(k int, nk int, pathPos int) *VertexBottomKSketch {
	sketches := make([][]uint64, nk)
	for i := range sketches {
		sketches[i] = make([]uint64, 0, k)
	}
	
	return &VertexBottomKSketch{
		Sketches: sketches,
		K:        k,
		NK:       nk,
		PathPos:  pathPos,
	}
}

// AddValue adds hash value to specific hash function sketch
func (s *VertexBottomKSketch) AddValue(hashFunc int, value uint64) {
	if hashFunc < 0 || hashFunc >= s.NK || s.K <= 0 {
		return
	}

	sketch := s.Sketches[hashFunc]
	
	// If we have less than K values, just add it
	if len(sketch) < s.K {
		sketch = append(sketch, value)
		// Keep sorted for efficiency
		if len(sketch) > 1 && sketch[len(sketch)-1] < sketch[len(sketch)-2] {
			sort.Slice(sketch, func(i, j int) bool { return sketch[i] < sketch[j] })
		}
		s.Sketches[hashFunc] = sketch
		return
	}

	// If the value is smaller than the largest (K-th) value, replace it
	if value < sketch[s.K-1] {
		sketch[s.K-1] = value
		// Re-sort to maintain order
		sort.Slice(sketch, func(i, j int) bool { return sketch[i] < sketch[j] })
		s.Sketches[hashFunc] = sketch
	}
}

// EstimateUnionWith calculates union size for inclusion-exclusion
func (s *VertexBottomKSketch) EstimateUnionWith(other *VertexBottomKSketch) float64 {
	if other == nil || s.NK != other.NK {
		return 0
	}
	
	unionSketch := UnionSketches(s, other)
	if unionSketch == nil {
		return 0
	}
	
	return unionSketch.EstimateDegree().Value
}

// Sophisticated degree estimation with saturated/undersaturated handling
func (s *VertexBottomKSketch) EstimateDegree() *DegreeEstimate {
	if s.NK == 0 {
		return &DegreeEstimate{Value: 0, IsSaturated: false, SketchSize: 0}
	}
	
	totalEstimate := 0.0
	saturatedCount := 0
	totalSketchSize := 0
	
	for i := 0; i < s.NK; i++ {
		sketch := s.Sketches[i]
		sketchSize := len(sketch)
		totalSketchSize += sketchSize
		
		if sketchSize >= s.K-1 {
			// Saturated sketch - use sophisticated estimation
			if sketchSize > 0 {
				currentSketch := sketch[sketchSize-1]
				if currentSketch > 0 {
					degree := float64(UINT64_MAX) / float64(currentSketch) * float64(s.K-1)
					totalEstimate += degree
					saturatedCount++
				}
			}
		} else {
			// Undersaturated sketch - use count
			totalEstimate += float64(sketchSize)
		}
	}
	
	avgEstimate := totalEstimate / float64(s.NK)
	isSaturated := saturatedCount > s.NK/2 // Majority saturated
	
	return &DegreeEstimate{
		Value:      avgEstimate,
		IsSaturated: isSaturated,
		SketchSize: totalSketchSize / s.NK,
	}
}

// Complex intersection calculation with two methods// Complex intersection calculation using inclusion-exclusion principle
func (s *VertexBottomKSketch) EstimateIntersectionWith(other *VertexBottomKSketch) float64 {
	if other == nil || s.NK != other.NK {
		return 0
	}
	
	// Get individual estimates
	size1 := s.EstimateDegree().Value
	size2 := other.EstimateDegree().Value
	
	// For very small sketches, use direct overlap counting
	if s.Size() < s.K/2 && other.Size() < other.K/2 {
		return s.countDirectOverlap(other)
	}
	
	// Use inclusion-exclusion: |A ∩ B| = |A| + |B| - |A ∪ B|
	unionSize := s.EstimateUnionWith(other)
	intersection := size1 + size2 - unionSize
	
	// Ensure non-negative result
	if intersection < 0 {
		intersection = 0
	}
	
	return intersection
}

// Helper function for direct overlap counting
func (s *VertexBottomKSketch) countDirectOverlap(other *VertexBottomKSketch) float64 {
	totalOverlap := 0.0
	
	for i := 0; i < s.NK; i++ {
		sketch1 := s.Sketches[i]
		sketch2 := other.Sketches[i]
		
		overlapCount := 0
		sketch1Map := make(map[uint64]bool)
		for _, val := range sketch1 {
			sketch1Map[val] = true
		}
		
		for _, val := range sketch2 {
			if sketch1Map[val] {
				overlapCount++
			}
		}
		
		totalOverlap += float64(overlapCount)
	}
	
	return totalOverlap / float64(s.NK)
}

// Helper function to create union of two hash function sketches
func (s *VertexBottomKSketch) createUnionForHashFunc(sketch1, sketch2 []uint64) []uint64 {
	// Merge all values
	allValues := make([]uint64, 0, len(sketch1)+len(sketch2))
	allValues = append(allValues, sketch1...)
	allValues = append(allValues, sketch2...)
	
	// Remove duplicates and sort
	uniqueValues := removeDuplicatesUint64(allValues)
	
	// Keep only K smallest values
	if len(uniqueValues) > s.K {
		uniqueValues = uniqueValues[:s.K]
	}
	
	return uniqueValues
}

// Helper function to estimate degree for single hash function
func (s *VertexBottomKSketch) estimateDegreeForHashFunc(sketch []uint64) float64 {
	if len(sketch) == 0 {
		return 0
	}
	
	if len(sketch) < s.K {
		// Undersaturated
		return float64(len(sketch))
	} else {
		// Saturated
		maxValue := sketch[len(sketch)-1]
		if maxValue > 0 {
			return float64(UINT64_MAX) / float64(maxValue) * float64(s.K-1)
		}
	}
	
	return 0
}

// Get all hash values for adjacency discovery
func (s *VertexBottomKSketch) GetAllHashValues() []uint64 {
	var allValues []uint64
	
	for i := 0; i < s.NK; i++ {
		allValues = append(allValues, s.Sketches[i]...)
	}
	
	// Remove duplicates and sort
	uniqueValues := removeDuplicatesUint64(allValues)
	sort.Slice(uniqueValues, func(i, j int) bool { return uniqueValues[i] < uniqueValues[j] })
	
	return uniqueValues
}

// Union sketches with proper nK handling
func UnionSketches(s1, s2 *VertexBottomKSketch) *VertexBottomKSketch {
	if s1 == nil && s2 == nil {
		return nil
	}
	if s1 == nil {
		return s2.Clone()
	}
	if s2 == nil {
		return s1.Clone()
	}
	
	if s1.NK != s2.NK {
		return nil // Incompatible sketches
	}

	k := s1.K
	if s2.K < k {
		k = s2.K
	}

	union := NewVertexBottomKSketch(k, s1.NK, s1.PathPos)

	// Union each hash function independently
	for i := 0; i < s1.NK; i++ {
		sketch1 := s1.Sketches[i]
		sketch2 := s2.Sketches[i]
		
		// Merge all values from both sketches
		allValues := make([]uint64, 0, len(sketch1)+len(sketch2))
		allValues = append(allValues, sketch1...)
		allValues = append(allValues, sketch2...)

		// Remove duplicates and sort
		uniqueValues := removeDuplicatesUint64(allValues)
		sort.Slice(uniqueValues, func(i, j int) bool { return uniqueValues[i] < uniqueValues[j] })

		// Keep only the K smallest values
		numToKeep := k
		if len(uniqueValues) < numToKeep {
			numToKeep = len(uniqueValues)
		}

		union.Sketches[i] = uniqueValues[:numToKeep]
	}
	
	return union
}

// Clone creates a deep copy of the multi-sketch
func (s *VertexBottomKSketch) Clone() *VertexBottomKSketch {
	clone := NewVertexBottomKSketch(s.K, s.NK, s.PathPos)
	
	for i := 0; i < s.NK; i++ {
		clone.Sketches[i] = make([]uint64, len(s.Sketches[i]))
		copy(clone.Sketches[i], s.Sketches[i])
	}
	
	return clone
}

// Size returns the total number of values across all hash functions
func (s *VertexBottomKSketch) Size() int {
	total := 0
	for i := 0; i < s.NK; i++ {
		total += len(s.Sketches[i])
	}
	return total
}

// IsEmpty returns true if all sketches are empty
func (s *VertexBottomKSketch) IsEmpty() bool {
	for i := 0; i < s.NK; i++ {
		if len(s.Sketches[i]) > 0 {
			return false
		}
	}
	return true
}

// Independent hash functions for nK sketches
func GenerateIndependentHashValue(nodeID string, hashFunc int, seed int64) uint64 {
	// Use different constants for each hash function to ensure independence
	hashConstants := []uint64{
		0x9e3779b97f4a7c15, 0x85ebca6b7b4a4c3d, 0xc2b2ae35b3b4c3d5, 0xa2b4c7d5e3f4a5b7,
		0xb7e9c4d2f1a3b5c7, 0xd4c5b8e2f7a9c1d3, 0xf2a7c9e1b5d3f4a6, 0xa9c7e4f1d2b6c8e5,
	}
	
	// Ensure we have enough constants
	if hashFunc >= len(hashConstants) {
		hashFunc = hashFunc % len(hashConstants)
	}
	
	hash := uint64(seed) ^ hashConstants[hashFunc]
	
	// Hash the node ID with the specific hash function
	for _, char := range nodeID {
		hash = hash*31 + uint64(char)
		hash ^= hash >> 16
		hash *= 0x85ebca6b7b4a4c3d
		hash ^= hash >> 13
		hash *= 0xc2b2ae35b3b4c3d5
		hash ^= hash >> 16
	}
	
	// Mix in the hash function index for additional independence
	hash ^= uint64(hashFunc) * hashConstants[hashFunc]
	hash ^= hash >> 16
	
	return hash
}

// Merge sketches for community construction
func MergeSketches(sketches []*VertexBottomKSketch) *VertexBottomKSketch {
	if len(sketches) == 0 {
		return nil
	}

	// Find parameters from first non-nil sketch
	var k, nk, pathPos int
	for _, sketch := range sketches {
		if sketch != nil {
			k = sketch.K
			nk = sketch.NK
			pathPos = sketch.PathPos
			break
		}
	}

	merged := NewVertexBottomKSketch(k, nk, pathPos)

	// Merge each hash function independently
	for hashFunc := 0; hashFunc < nk; hashFunc++ {
		var allValues []uint64
		
		// Collect all values from this hash function across all sketches
		for _, sketch := range sketches {
			if sketch != nil && hashFunc < len(sketch.Sketches) {
				allValues = append(allValues, sketch.Sketches[hashFunc]...)
			}
		}
		
		// Remove duplicates and sort
		uniqueValues := removeDuplicatesUint64(allValues)
		sort.Slice(uniqueValues, func(i, j int) bool { return uniqueValues[i] < uniqueValues[j] })
		
		// Keep only K smallest values
		numToKeep := k
		if len(uniqueValues) < numToKeep {
			numToKeep = len(uniqueValues)
		}
		
		if numToKeep > 0 {
			merged.Sketches[hashFunc] = uniqueValues[:numToKeep]
		}
	}

	return merged
}

// Validate multi-sketch structure
func (s *VertexBottomKSketch) ValidateSketch() error {
	if s.K <= 0 {
		return fmt.Errorf("invalid K value: %d", s.K)
	}
	
	if s.NK <= 0 {
		return fmt.Errorf("invalid NK value: %d", s.NK)
	}
	
	if len(s.Sketches) != s.NK {
		return fmt.Errorf("sketch count (%d) doesn't match NK (%d)", len(s.Sketches), s.NK)
	}

	for i, sketch := range s.Sketches {
		if len(sketch) > s.K {
			return fmt.Errorf("sketch %d contains more values (%d) than K (%d)", i, len(sketch), s.K)
		}

		// Check if values are sorted
		for j := 1; j < len(sketch); j++ {
			if sketch[j] < sketch[j-1] {
				return fmt.Errorf("sketch %d values are not sorted", i)
			}
		}
	}

	return nil
}

// Get statistics for multi-sketch
func (s *VertexBottomKSketch) GetStatistics() map[string]interface{} {
	stats := make(map[string]interface{})
	
	totalSize := 0
	saturatedCount := 0
	
	for i := 0; i < s.NK; i++ {
		sketchSize := len(s.Sketches[i])
		totalSize += sketchSize
		
		if sketchSize >= s.K-1 {
			saturatedCount++
		}
	}
	
	degree := s.EstimateDegree()
	
	stats["total_size"] = totalSize
	stats["avg_size"] = float64(totalSize) / float64(s.NK)
	stats["saturated_count"] = saturatedCount
	stats["saturation_ratio"] = float64(saturatedCount) / float64(s.NK)
	stats["estimated_degree"] = degree.Value
	stats["is_majority_saturated"] = degree.IsSaturated
	stats["k"] = s.K
	stats["nk"] = s.NK
	stats["path_pos"] = s.PathPos
	
	return stats
}

// Helper function to remove duplicates from uint64 slice
func removeDuplicatesUint64(values []uint64) []uint64 {
	if len(values) <= 1 {
		return values
	}
	
	// Sort first
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })

	result := make([]uint64, 0, len(values))
	result = append(result, values[0])

	for i := 1; i < len(values); i++ {
		if values[i] != values[i-1] {
			result = append(result, values[i])
		}
	}

	return result
}

// Create sketch from hash values with proper hash function assignment
func BuildSketchFromHashValues(hashValues map[uint64]string, k int, nk int, pathPos int) *VertexBottomKSketch {
	sketch := NewVertexBottomKSketch(k, nk, pathPos)
	
	// Distribute hash values across hash functions based on their value
	for hashValue := range hashValues {
		hashFunc := int(hashValue) % nk
		sketch.AddValue(hashFunc, hashValue)
	}
	
	return sketch
}

// Check for overlap in sketches (for adjacency discovery)
func HasSketchOverlap(s1, s2 *VertexBottomKSketch, threshold float64) bool {
	if s1 == nil || s2 == nil || s1.NK != s2.NK {
		return false
	}
	
	overlapCount := 0
	totalChecked := 0
	
	for i := 0; i < s1.NK; i++ {
		sketch1 := s1.Sketches[i]
		sketch2 := s2.Sketches[i]
		
		// Create map for faster lookup
		sketch1Map := make(map[uint64]bool)
		for _, val := range sketch1 {
			sketch1Map[val] = true
		}
		
		// Check for overlaps
		for _, val := range sketch2 {
			totalChecked++
			if sketch1Map[val] {
				overlapCount++
			}
		}
	}
	
	if totalChecked == 0 {
		return false
	}
	
	overlapRatio := float64(overlapCount) / float64(totalChecked)
	return overlapRatio >= threshold
}


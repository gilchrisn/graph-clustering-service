package main

import (
    "fmt"
    "math"
    "math/rand"
    "sort"
)

// ============================================================================
// VBK (Bottom-K Sketch) DATA STRUCTURES
// ============================================================================

type BottomKSketch struct {
    Hashes []uint32
    K      int
    Filled int
}

type VBKNode struct {
    ID     int
    Hash   uint32
    Sketch *BottomKSketch
}

// ============================================================================
// VBK SKETCH OPERATIONS
// ============================================================================

func newBottomKSketch(k int) *BottomKSketch {
    return &BottomKSketch{
        Hashes: make([]uint32, k),
        K:      k,
        Filled: 0,
    }
}

func (s *BottomKSketch) addHash(hash uint32) {
    if s.Filled < s.K {
        s.Hashes[s.Filled] = hash
        s.Filled++
        if s.Filled == s.K {
            sort.Slice(s.Hashes, func(i, j int) bool {
                return s.Hashes[i] < s.Hashes[j]
            })
        }
    } else {
        // Sketch is full, check if we should replace
        if hash < s.Hashes[s.K-1] {
            s.Hashes[s.K-1] = hash
            // Re-sort to maintain order
            for i := s.K - 1; i > 0 && s.Hashes[i] < s.Hashes[i-1]; i-- {
                s.Hashes[i], s.Hashes[i-1] = s.Hashes[i-1], s.Hashes[i]
            }
        }
    }
}

func (s *BottomKSketch) estimateCardinality() float64 {
    if s.Filled < s.K {
        return float64(s.Filled)
    }
    
    if s.Filled == 0 {
        return 0.0
    }
    
    kthValue := s.Hashes[s.K-1]
    return float64(s.K-1) * float64(math.MaxUint32) / float64(kthValue)
}

func unionSketches(s1, s2 *BottomKSketch) *BottomKSketch {
    if s1 == nil || s2 == nil {
        return nil
    }
    
    union := newBottomKSketch(s1.K)
    
    // Merge both sketches
    allHashes := make([]uint32, 0, s1.Filled+s2.Filled)
    for i := 0; i < s1.Filled; i++ {
        allHashes = append(allHashes, s1.Hashes[i])
    }
    for i := 0; i < s2.Filled; i++ {
        allHashes = append(allHashes, s2.Hashes[i])
    }
    
    // Sort and deduplicate
    sort.Slice(allHashes, func(i, j int) bool {
        return allHashes[i] < allHashes[j]
    })
    
    prev := uint32(math.MaxUint32)
    for _, hash := range allHashes {
        if hash != prev && union.Filled < union.K {
            union.Hashes[union.Filled] = hash
            union.Filled++
            prev = hash
        }
    }
    
    return union
}

// ============================================================================
// VBK UNION METHOD ESTIMATOR
// ============================================================================

type VBKUnionEstimator struct {
    K     int
    nodes map[int]*VBKNode
}

func NewVBKUnionEstimator(k int) *VBKUnionEstimator {
    return &VBKUnionEstimator{
        K:     k,
        nodes: make(map[int]*VBKNode),
    }
}

func (e *VBKUnionEstimator) BuildGraph(graph *Graph, rng *rand.Rand) error {
    e.nodes = make(map[int]*VBKNode)
    
    // Generate unique hashes for each node
    nodeHashes := make(map[int]uint32)
    usedHashes := make(map[uint32]bool)
    
    for i := 0; i < graph.NumNodes; i++ {
        for {
            hash := rng.Uint32()
            if hash != math.MaxUint32 && !usedHashes[hash] {
                nodeHashes[i] = hash
                usedHashes[hash] = true
                break
            }
        }
    }
    
    // Build sketches for each node
    for nodeId := 0; nodeId < graph.NumNodes; nodeId++ {
        sketch := newBottomKSketch(e.K)
        
        // Add self hash
        sketch.addHash(nodeHashes[nodeId])
        
        // Add neighbor hashes
        for neighbor := range graph.Edges[nodeId] {
            sketch.addHash(nodeHashes[neighbor])
        }
        
        e.nodes[nodeId] = &VBKNode{
            ID:     nodeId,
            Hash:   nodeHashes[nodeId],
            Sketch: sketch,
        }
    }
    
    return nil
}

func (e *VBKUnionEstimator) EstimateEdges(commA, commB []int) float64 {
    if len(commA) == 0 || len(commB) == 0 {
        return 0.0
    }
    
    // Union all sketches in community A
    var unionA *BottomKSketch
    for _, nodeId := range commA {
        if node := e.nodes[nodeId]; node != nil {
            if unionA == nil {
                unionA = newBottomKSketch(node.Sketch.K)
                for i := 0; i < node.Sketch.Filled; i++ {
                    unionA.addHash(node.Sketch.Hashes[i])
                }
            } else {
                for i := 0; i < node.Sketch.Filled; i++ {
                    unionA.addHash(node.Sketch.Hashes[i])
                }
            }
        }
    }
    
    // Create community B identifier sketch
    identifierB := newBottomKSketch(len(commB))
    for _, nodeId := range commB {
        if node := e.nodes[nodeId]; node != nil {
            identifierB.addHash(node.Hash)
        }
    }
    
    if unionA == nil || identifierB.Filled == 0 {
        return 0.0
    }
    
    // Apply inclusion-exclusion: |A ∩ B| = |A| + |B| - |A ∪ B|
    cardinalityA := unionA.estimateCardinality()
    cardinalityB := float64(len(commB))
    
    unionAB := unionSketches(unionA, identifierB)
    if unionAB == nil {
        return 0.0
    }
    
    unionCardinality := unionAB.estimateCardinality()
    intersection := cardinalityA + cardinalityB - unionCardinality
    
    return math.Max(0, intersection)
}

func (e *VBKUnionEstimator) GetMethodName() string {
    return fmt.Sprintf("vbk_u_%d", e.K)
}

func (e *VBKUnionEstimator) GetParameters() map[string]interface{} {
    return map[string]interface{}{
        "K": e.K,
    }
}

// ============================================================================
// VBK SUM METHOD ESTIMATOR
// ============================================================================

type VBKSumEstimator struct {
    K     int
    nodes map[int]*VBKNode
}

func NewVBKSumEstimator(k int) *VBKSumEstimator {
    return &VBKSumEstimator{
        K:     k,
        nodes: make(map[int]*VBKNode),
    }
}

func (e *VBKSumEstimator) BuildGraph(graph *Graph, rng *rand.Rand) error {
    e.nodes = make(map[int]*VBKNode)
    
    // Generate unique hashes for each node
    nodeHashes := make(map[int]uint32)
    usedHashes := make(map[uint32]bool)
    
    for i := 0; i < graph.NumNodes; i++ {
        for {
            hash := rng.Uint32()
            if hash != math.MaxUint32 && !usedHashes[hash] {
                nodeHashes[i] = hash
                usedHashes[hash] = true
                break
            }
        }
    }
    
    // Build sketches for each node
    for nodeId := 0; nodeId < graph.NumNodes; nodeId++ {
        sketch := newBottomKSketch(e.K)
        
        // Add self hash
        sketch.addHash(nodeHashes[nodeId])
        
        // Add neighbor hashes
        for neighbor := range graph.Edges[nodeId] {
            sketch.addHash(nodeHashes[neighbor])
        }
        
        e.nodes[nodeId] = &VBKNode{
            ID:     nodeId,
            Hash:   nodeHashes[nodeId],
            Sketch: sketch,
        }
    }
    
    return nil
}

func (e *VBKSumEstimator) EstimateEdges(commA, commB []int) float64 {
    totalEdges := 0.0
    
    // For each node in A, estimate edges to the entire community B
    for _, nodeAId := range commA {
        nodeA := e.nodes[nodeAId]
        if nodeA == nil {
            continue
        }
        
        // Estimate edges from nodeA to community B using inclusion-exclusion
        edgesToCommB := e.estimateNodeToCommunityEdges(nodeA, commB)
        totalEdges += edgesToCommB
    }
    
    return totalEdges
}

func (e *VBKSumEstimator) estimateNodeToCommunityEdges(nodeA *VBKNode, commB []int) float64 {
    if nodeA == nil || len(commB) == 0 {
        return 0.0
    }
    
    // Create identifier sketch for community B
    identifierB := newBottomKSketch(len(commB))
    for _, nodeBId := range commB {
        if nodeB := e.nodes[nodeBId]; nodeB != nil {
            identifierB.addHash(nodeB.Hash)
        }
    }
    
    if identifierB.Filled == 0 {
        return 0.0
    }
    
    // Apply inclusion-exclusion: |A ∩ B| = |A| + |B| - |A ∪ B|
    cardinalityA := nodeA.Sketch.estimateCardinality()
    cardinalityB := float64(len(commB))
    
    unionAB := unionSketches(nodeA.Sketch, identifierB)
    if unionAB == nil {
        return 0.0
    }
    
    unionCardinality := unionAB.estimateCardinality()
    intersection := cardinalityA + cardinalityB - unionCardinality
    
    return math.Max(0, intersection)
}

func (e *VBKSumEstimator) GetMethodName() string {
    return fmt.Sprintf("vbk_s_%d", e.K)
}

func (e *VBKSumEstimator) GetParameters() map[string]interface{} {
    return map[string]interface{}{
        "K": e.K,
    }
}
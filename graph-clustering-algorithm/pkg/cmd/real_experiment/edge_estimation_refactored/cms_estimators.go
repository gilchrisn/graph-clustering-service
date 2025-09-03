package main

import (
    "hash/fnv"
    "math"
    "math/rand"
	"fmt"
)

// ============================================================================
// COUNT-MIN SKETCH DATA STRUCTURES
// ============================================================================

type CountMinSketch struct {
    Depth      int        // d - number of hash functions
    Width      int        // w - number of buckets per hash function
    Matrix     [][]uint32 // d x w matrix of counters
    HashSeeds  []uint32   // d hash seeds for independent hash functions
}

type CMSNode struct {
    ID     int
    Hash   uint32
    Sketch *CountMinSketch
}

// ============================================================================
// COUNT-MIN SKETCH OPERATIONS
// ============================================================================

func newCountMinSketch(depth, width int, rng *rand.Rand) *CountMinSketch {
    cms := &CountMinSketch{
        Depth:     depth,
        Width:     width,
        Matrix:    make([][]uint32, depth),
        HashSeeds: make([]uint32, depth),
    }
    
    // Initialize matrix
    for i := 0; i < depth; i++ {
        cms.Matrix[i] = make([]uint32, width)
        cms.HashSeeds[i] = rng.Uint32()
    }
    
    return cms
}

// FNV-based hash function with seed
func (cms *CountMinSketch) hash(item uint32, seed uint32) uint32 {
    h := fnv.New32a()
    // Combine item with seed
    combined := (uint64(item) << 32) | uint64(seed)
    h.Write([]byte{
        byte(combined >> 56), byte(combined >> 48),
        byte(combined >> 40), byte(combined >> 32),
        byte(combined >> 24), byte(combined >> 16),
        byte(combined >> 8), byte(combined),
    })
    return h.Sum32() % uint32(cms.Width)
}

func (cms *CountMinSketch) Add(itemHash uint32, count uint32) {
    for i := 0; i < cms.Depth; i++ {
        bucket := cms.hash(itemHash, cms.HashSeeds[i])
        cms.Matrix[i][bucket] += count
    }
}

func (cms *CountMinSketch) Estimate(itemHash uint32) uint32 {
    if cms.Depth == 0 {
        return 0
    }
    
    minCount := cms.Matrix[0][cms.hash(itemHash, cms.HashSeeds[0])]
    for i := 1; i < cms.Depth; i++ {
        bucket := cms.hash(itemHash, cms.HashSeeds[i])
        count := cms.Matrix[i][bucket]
        if count < minCount {
            minCount = count
        }
    }
    return minCount
}

func sumSketches(sketches ...*CountMinSketch) *CountMinSketch {
    if len(sketches) == 0 {
        return nil
    }
    
    first := sketches[0]
    result := &CountMinSketch{
        Depth:     first.Depth,
        Width:     first.Width,
        Matrix:    make([][]uint32, first.Depth),
        HashSeeds: make([]uint32, first.Depth),
    }
    
    // Copy hash seeds from first sketch
    copy(result.HashSeeds, first.HashSeeds)
    
    // Initialize result matrix
    for i := 0; i < first.Depth; i++ {
        result.Matrix[i] = make([]uint32, first.Width)
    }
    
    // Sum all matrices element-wise
    for _, sketch := range sketches {
        if sketch.Depth != first.Depth || sketch.Width != first.Width {
            continue // Skip incompatible sketches
        }
        for i := 0; i < sketch.Depth; i++ {
            for j := 0; j < sketch.Width; j++ {
                result.Matrix[i][j] += sketch.Matrix[i][j]
            }
        }
    }
    
    return result
}

// ============================================================================
// CMS SUM OF SKETCHES ESTIMATOR
// ============================================================================

type CMSSumSketchesEstimator struct {
    Depth int // d
    Width int // w
    nodes map[int]*CMSNode
}

func NewCMSSumSketchesEstimator(depth, width int) *CMSSumSketchesEstimator {
    return &CMSSumSketchesEstimator{
        Depth: depth,
        Width: width,
        nodes: make(map[int]*CMSNode),
    }
}

func (e *CMSSumSketchesEstimator) BuildGraph(graph *Graph, rng *rand.Rand) error {
    e.nodes = make(map[int]*CMSNode)
    
    // Generate unique hashes for each node
    nodeHashes := make(map[int]uint32)
    usedHashes := make(map[uint32]bool)
    
    for i := 0; i < graph.NumNodes; i++ {
        for {
            hash := rng.Uint32()
            if hash != 0 && hash != math.MaxUint32 && !usedHashes[hash] {
                nodeHashes[i] = hash
                usedHashes[hash] = true
                break
            }
        }
    }
    
    // Build CMS for each node
    for nodeId := 0; nodeId < graph.NumNodes; nodeId++ {
        sketch := newCountMinSketch(e.Depth, e.Width, rng)
        
        // Add neighbor hashes to the sketch (NOT including self)
        for neighbor := range graph.Edges[nodeId] {
            sketch.Add(nodeHashes[neighbor], 1)
        }
        
        e.nodes[nodeId] = &CMSNode{
            ID:     nodeId,
            Hash:   nodeHashes[nodeId],
            Sketch: sketch,
        }
    }
    
    return nil
}

func (e *CMSSumSketchesEstimator) EstimateEdges(commA, commB []int) float64 {
    if len(commA) == 0 || len(commB) == 0 {
        return 0.0
    }
    
    // Collect all sketches from community A
    commASketches := make([]*CountMinSketch, 0, len(commA))
    for _, nodeId := range commA {
        if node := e.nodes[nodeId]; node != nil {
            commASketches = append(commASketches, node.Sketch)
        }
    }
    
    if len(commASketches) == 0 {
        return 0.0
    }
    
    // Sum all sketches from community A
    communityASketch := sumSketches(commASketches...)
    if communityASketch == nil {
        return 0.0
    }
    
    // Query the summed sketch for each node in community B
    totalWeight := 0.0
    for _, nodeBId := range commB {
        if nodeB := e.nodes[nodeBId]; nodeB != nil {
            count := communityASketch.Estimate(nodeB.Hash)
            totalWeight += float64(count)
        }
    }
    
    return totalWeight
}

func (e *CMSSumSketchesEstimator) GetMethodName() string {
    return fmt.Sprintf("cms_ss_%d_%d", e.Depth, e.Width)
}

func (e *CMSSumSketchesEstimator) GetParameters() map[string]interface{} {
    return map[string]interface{}{
        "depth": e.Depth,
        "width": e.Width,
    }
}

// ============================================================================
// CMS INDIVIDUAL ESTIMATES ESTIMATOR
// ============================================================================

type CMSIndividualEstimator struct {
    Depth int // d
    Width int // w
    nodes map[int]*CMSNode
}

func NewCMSIndividualEstimator(depth, width int) *CMSIndividualEstimator {
    return &CMSIndividualEstimator{
        Depth: depth,
        Width: width,
        nodes: make(map[int]*CMSNode),
    }
}

func (e *CMSIndividualEstimator) BuildGraph(graph *Graph, rng *rand.Rand) error {
    e.nodes = make(map[int]*CMSNode)
    
    // Generate unique hashes for each node
    nodeHashes := make(map[int]uint32)
    usedHashes := make(map[uint32]bool)
    
    for i := 0; i < graph.NumNodes; i++ {
        for {
            hash := rng.Uint32()
            if hash != 0 && hash != math.MaxUint32 && !usedHashes[hash] {
                nodeHashes[i] = hash
                usedHashes[hash] = true
                break
            }
        }
    }
    
    // Build CMS for each node
    for nodeId := 0; nodeId < graph.NumNodes; nodeId++ {
        sketch := newCountMinSketch(e.Depth, e.Width, rng)
        
        // Add neighbor hashes to the sketch (NOT including self)
        for neighbor := range graph.Edges[nodeId] {
            sketch.Add(nodeHashes[neighbor], 1)
        }
        
        e.nodes[nodeId] = &CMSNode{
            ID:     nodeId,
            Hash:   nodeHashes[nodeId],
            Sketch: sketch,
        }
    }
    
    return nil
}

func (e *CMSIndividualEstimator) EstimateEdges(commA, commB []int) float64 {
    totalWeight := 0.0
    
    for _, nodeAId := range commA {
        nodeA := e.nodes[nodeAId]
        if nodeA == nil {
            continue
        }
        
        for _, nodeBId := range commB {
            nodeB := e.nodes[nodeBId]
            if nodeB == nil {
                continue
            }
            
            count := nodeA.Sketch.Estimate(nodeB.Hash)
            totalWeight += float64(count)
        }
    }
    
    return totalWeight
}

func (e *CMSIndividualEstimator) GetMethodName() string {
    return fmt.Sprintf("cms_i_%d_%d", e.Depth, e.Width)
}

func (e *CMSIndividualEstimator) GetParameters() map[string]interface{} {
    return map[string]interface{}{
        "depth": e.Depth,
        "width": e.Width,
    }
}
package scar

import "math"

type UintE = uint32
type UintT = uint32

const UINT_E_MAX = math.MaxUint32

// SCARConfig holds configuration for SCAR algorithm
type SCARConfig struct {
	GraphFile    string
	PropertyFile string
	PathFile     string
	OutputFile   string
	EdgesFile    string
	Prefix       string // For output files
	K            int64
	NK           int64
	Threshold    float64
	UseLouvain   bool
	SketchOutput bool // Whether to output sketches
}

type MultiLevelPartitionTracker struct {
    // Stores mappings at each level
    levelMappings []LevelMapping
    
    // Original graph size
    originalGraphSize int64
    
    // Current level
    currentLevel int
}

type LevelMapping struct {
    // Maps super-node ID to its constituent node IDs from previous level
    superNodeToChildNodes map[int64][]int64
    
    // Maps super-node ID to original node IDs (flattened)
    superNodeToOriginalNodes map[int64][]int64
    
    // Number of nodes at this level
    numNodes int64

	levelSketches map[int64]*VertexBottomKSketch // sketches for this level
}


type HierarchicalPartition struct {
    OriginalGraphSize int64
    NumLevels        int
    Levels           []LevelInfo
}

type LevelInfo struct {
    Level                   int
    NumNodes               int64
    SuperNodeToChildNodes   map[int64][]int64  // Who merged to form each super-node
    SuperNodeToOriginalNodes map[int64][]int64  // Original nodes in each super-node
    Communities            []int64             // Only for final level
}

func NewMultiLevelPartitionTracker(originalSize int64) *MultiLevelPartitionTracker {
    return &MultiLevelPartitionTracker{
        levelMappings:     make([]LevelMapping, 0),
        originalGraphSize: originalSize,
        currentLevel:     0,
    }
}



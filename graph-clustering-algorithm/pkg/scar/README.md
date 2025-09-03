# SCAR Algorithm API Documentation

## Overview
SCAR (Sketch-based Community detection with Approximate Refinement) algorithm for large-scale community detection using Bottom-K sketches.

## Main Function

```go
func Run(graphFile, propertiesFile, pathFile string, config *Config, ctx context.Context) (*Result, error)
```

### Input

**File Formats:**

**Graph File** (edge list):
```
# Each line: source_node destination_node
0 1
1 2  
2 3
```

**Properties File** (node types):
```
# Each line: node_id type_id
0 0
1 1
2 0  
```

**Path File** (type sequence):
```
# Each line: type_id (path pattern)
0
1
0
```

**Configuration:**
```go
type Config struct {
    // Standard Louvain parameters
    MaxLevels()          int     // max hierarchy levels (default: 10)
    MaxIterations()      int     // max iterations per level (default: 100)
    MinModularityGain()  float64 // min gain threshold (default: 1e-6)
    
    // SCAR-specific parameters  
    K()         int64   // sketch size (default: 10)
    NK()        int64   // number of sketch layers (default: 4)
    Threshold() float64 // sketch fullness threshold (default: 0.5)
    
    // Graph storage
    StoreGraphsAtEachLevel() bool // store SketchGraph objects (default: false)
}
```

### Output

```go
type Result struct {
    Levels           []LevelInfo     // hierarchy information per level
    FinalCommunities map[int]int     // originalNodeID -> communityID
    Modularity       float64         // final modularity score  
    NumLevels        int            // number of hierarchy levels
    Statistics       Statistics      // performance metrics
    NodeMapping      *NodeMapping    // original to compressed mapping
}

type LevelInfo struct {
    Level          int             // hierarchy level
    Communities    map[int][]int   // communityID -> [originalNodeIDs]
    Modularity     float64         // modularity at this level
    NumCommunities int            // number of communities
    NumMoves       int            // node moves in this level
    RuntimeMS      int64          // level runtime
    
    // Hierarchy tracking
    CommunityToSuperNode map[int]int // community -> super-node mapping
    SuperNodeToCommunity map[int]int // super-node -> community mapping
    
    // Graph storage (only if StoreGraphsAtEachLevel=true)
    SketchGraph *SketchGraph `json:"-"` // sketch graph at this level
}

type NodeMapping struct {
    OriginalToCompressed map[int]int  // original -> compressed node IDs
    CompressedToOriginal []int        // compressed -> original node IDs
    NumTargetNodes       int          // number of target type nodes
}
```

## Usage Example

```go
// Basic usage
config := scar.NewConfig()
config.Set("scar.k", 64)
config.Set("algorithm.max_levels", 5)

result, err := scar.Run(graphFile, propertiesFile, pathFile, config, ctx)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Final modularity: %.4f\n", result.Modularity)
community := result.FinalCommunities[originalNodeID]
```

### Graph Storage
```go
// Enable graph storage
config.Set("output.store_graphs_at_each_level", true)

result, err := scar.Run(graphFile, propertiesFile, pathFile, config, ctx)

// Access stored graphs
for level, levelInfo := range result.Levels {
    if levelInfo.SketchGraph != nil {
        degree := levelInfo.SketchGraph.GetDegree(node)
        neighbors, weights := levelInfo.SketchGraph.GetNeighbors(node)
    }
}
```

## Key Differences from Louvain

1. **Input**: Takes file paths instead of Graph object
2. **Preprocessing**: Performs sketch computation on input graph
3. **Scalability**: Uses probabilistic sketches for large graphs
4. **Node Mapping**: Compresses graph to target nodes only
5. **Accuracy**: Trades some accuracy for scalability
6. **Graph Storage**: Stores `SketchGraph` objects with sketch-based operations

## Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scar.k` | int64 | 10 | Bottom-K sketch size |
| `scar.nk` | int64 | 4 | Number of sketch layers |
| `scar.threshold` | float64 | 0.5 | Sketch fullness threshold |
| `algorithm.max_levels` | int | 10 | Maximum hierarchy levels |
| `algorithm.max_iterations` | int | 100 | Max iterations per level |
| `algorithm.min_modularity_gain` | float64 | 1e-6 | Minimum gain threshold |
| `output.store_graphs_at_each_level` | bool | false | Store SketchGraph at each level |

## File Format Details

**Graph File**: Standard edge list format. Supports weighted graphs with third column for weights.

**Properties File**: Maps each node to a type ID. Nodes not in file default to type 0.

**Path File**: Defines the sequence of node types to consider. SCAR processes nodes matching the first type in the path.
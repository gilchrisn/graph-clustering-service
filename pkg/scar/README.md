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

type NodeMapping struct {
    OriginalToCompressed map[int]int  // original -> compressed node IDs
    CompressedToOriginal []int        // compressed -> original node IDs
    NumTargetNodes       int          // number of target type nodes
}
```

## Usage Example

```go
// Prepare input files
graphFile := "graph.txt"
propertiesFile := "properties.txt" 
pathFile := "path.txt"

// Configure algorithm
config := scar.NewConfig()
config.Set("scar.k", 64)           // sketch size
config.Set("scar.nk", 8)           // sketch layers
config.Set("algorithm.max_levels", 5)

// Run clustering
ctx := context.Background()
result, err := scar.Run(graphFile, propertiesFile, pathFile, config, ctx)
if err != nil {
    log.Fatal(err)
}

// Access results
fmt.Printf("Final modularity: %.4f\n", result.Modularity)
fmt.Printf("Target nodes processed: %d\n", result.NodeMapping.NumTargetNodes)

// Get community for original node ID
community := result.FinalCommunities[originalNodeID]

// Map between original and compressed IDs
compressedID := result.NodeMapping.OriginalToCompressed[originalNodeID]
originalID := result.NodeMapping.CompressedToOriginal[compressedID]
```

## Key Differences from Louvain

1. **Input**: Takes file paths instead of Graph object
2. **Preprocessing**: Performs sketch computation on input graph
3. **Scalability**: Uses probabilistic sketches for large graphs
4. **Node Mapping**: Compresses graph to target nodes only
5. **Accuracy**: Trades some accuracy for scalability

## Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scar.k` | int64 | 10 | Bottom-K sketch size |
| `scar.nk` | int64 | 4 | Number of sketch layers |
| `scar.threshold` | float64 | 0.5 | Sketch fullness threshold |
| `algorithm.max_levels` | int | 10 | Maximum hierarchy levels |
| `algorithm.max_iterations` | int | 100 | Max iterations per level |
| `algorithm.min_modularity_gain` | float64 | 1e-6 | Minimum gain threshold |

## File Format Details

**Graph File**: Standard edge list format. Supports weighted graphs with third column for weights.

**Properties File**: Maps each node to a type ID. Nodes not in file default to type 0.

**Path File**: Defines the sequence of node types to consider. SCAR processes nodes matching the first type in the path.

## Performance Notes

- Use larger `k` values (64-512) for better accuracy
- Use multiple layers (`nk` = 4-8) for robust estimation  
- SCAR is designed for graphs with millions of nodes
- Memory usage scales with sketch size, not graph size


# SCAR: Sketch-based Community Detection

## Command Line Usage

```bash
# Basic community detection
./scar.exe "graph.txt" -pro="properties.txt" -path="path.txt" -louvain

# SCAR hierarchy output (for PPRViz integration)
./scar.exe "graph.txt" -pro="properties.txt" -path="path.txt" -louvain -sketch-output
```

### Parameters
- `-k=<int>`: Sketch size (default: 10)
- `-nk=<int>`: Number of sketch layers (default: 4)
- `-th=<float>`: Threshold (default: 0.5)
- `-prefix=<string>`: Output prefix (default: graph filename)

## Input Files

### Graph File (`graph.txt`)
```
0 1
1 2
2 0
3 4
```

### Properties File (`properties.txt`)
```
0 0
1 1
2 0
3 0
4 1
```

### Path File (`path.txt`)
```
0
1
0
```

## Output Files

### Basic Mode (no `-sketch-output`)
- **`output.txt`**: Simple node-community pairs
```
0 0
1 0
2 1
3 1
```

### SCAR Mode (`-sketch-output`)
- **`{prefix}_mapping.dat`**: Hierarchy mapping
- **`{prefix}_hierarchy.dat`**: Parent-child relationships
- **`{prefix}_root.dat`**: Top-level communities
- **`{prefix}.sketch`**: Node sketches with levels

#### Example Output Files

**`network_mapping.dat`**:
```
c0_l0_0
3
0
1
2
c0_l1_0
5
0
1
2
3
4
```

**`network_hierarchy.dat`**:
```
c0_l0_0
3
0
1
2
c0_l1_0
2
c0_l0_0
c0_l0_1
```

**`network_root.dat`**:
```
c0_l1_0
```

**`network.sketch`**:
```
k=64 nk=4 th=0.5 pro=properties.txt path=path.txt
0 level=0
1000000,1000001,1000002
2000000,2000001,2000002
3000000,3000001,3000002
4000000,4000001,4000002
```

## API Usage

```go
package main

import "github.com/gilchrisn/graph-clustering-service/pkg/scar"

func main() {
    config := scar.SCARConfig{
        GraphFile:     "network.txt",
        PropertyFile:  "properties.txt",
        PathFile:      "path.txt",
        K:             64,
        NK:            4,
        Threshold:     0.5,
        UseLouvain:    true,
        SketchOutput:  true,  // For SCAR hierarchy output
    }
    
    engine := scar.NewSketchLouvainEngine(config)
    err := engine.RunLouvain()
    if err != nil {
        panic(err)
    }
}
```

## Pipeline Integration

### With PPRViz
```bash
# Step 1: Run SCAR
./scar.exe "network.txt" -pro="props.txt" -path="path.txt" -louvain -sketch-output

# Step 2: Run PPRViz
./pprviz -mode query -scar-dir ./ -scar-prefix network -embed
```

### Batch Processing
```bash
for graph in *.txt; do
    base=$(basename "$graph" .txt)
    ./scar.exe "$graph" -pro="${base}_props.txt" -path="${base}_path.txt" -louvain -sketch-output
done
```

## Configuration

```go
type SCARConfig struct {
    GraphFile     string  // Input graph file
    PropertyFile  string  // Node properties file
    PathFile      string  // Path specification file
    OutputFile    string  // Basic output file
    Prefix        string  // Output prefix
    K             int64   // Sketch size
    NK            int64   // Number of sketch layers
    Threshold     float64 // Algorithm threshold
    UseLouvain    bool    // Enable Louvain algorithm
    SketchOutput  bool    // Enable SCAR hierarchy output
}
```
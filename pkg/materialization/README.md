# Graph Materialization Service

Converts heterogeneous graphs (multiple node/edge types) into homogeneous graphs (single node type) using meta-path traversal. Perfect for feeding into community detection algorithms like Louvain, Leiden, or SCAR.

## Quick Start

```go
import "github.com/gilchrisn/graph-clustering-service/pkg/materialization"

// Convert SCAR-format input directly to edge list
err := materialization.SCARToMaterialization(
    "graph.txt",      // Input graph edges
    "properties.txt", // Node type assignments  
    "path.txt",       // Meta-path specification
    "output.txt"      // Edge list output
)
```

## Input Formats

### Option 1: SCAR Format (Simple)

**Graph File** (`graph.txt`) - Edge list:
```
0 1
1 2
2 0
3 4
4 5
```

**Properties File** (`properties.txt`) - Node types:
```
0 0
1 1
2 0
3 0
4 1
5 1
```

**Path File** (`path.txt`) - Meta-path as type sequence:
```
0
1
0
```

### Option 2: JSON Format (Advanced)

**Graph JSON**:
```json
{
  "nodes": {
    "alice": {"id": "alice", "type": "Author", "properties": {}},
    "paper1": {"id": "paper1", "type": "Paper", "properties": {}},
    "bob": {"id": "bob", "type": "Author", "properties": {}}
  },
  "edges": [
    {"from": "alice", "to": "paper1", "type": "writes", "weight": 1.0},
    {"from": "bob", "to": "paper1", "type": "writes", "weight": 1.0}
  ]
}
```

**Meta-path JSON**:
```json
{
  "id": "author-collaboration",
  "node_sequence": ["Author", "Paper", "Author"],
  "edge_sequence": ["writes", "writes"]
}
```

## Output Format

Simple edge list compatible with most community detection tools:

```
alice bob 2.000000
alice charlie 1.000000
bob charlie 1.000000
diana alice 1.000000
```

## API Reference

### Basic Usage

```go
// Parse SCAR input
graph, metaPath, err := materialization.ParseSCARInput("graph.txt", "properties.txt", "path.txt")

// Configure materialization
config := materialization.DefaultMaterializationConfig()
config.Aggregation.Strategy = materialization.Count
config.Aggregation.Symmetric = true

// Run materialization
engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
result, err := engine.Materialize()

// Save as edge list
err = materialization.SaveAsSimpleEdgeList(result.HomogeneousGraph, "output.txt")
```

### Configuration Options

```go
type MaterializationConfig struct {
    Traversal   TraversalConfig   // Path finding settings
    Aggregation AggregationConfig // Edge weight calculation
    Progress    ProgressConfig    // Progress reporting
}

// Key settings:
config.Aggregation.Strategy = Count     // Count, Sum, Average, Maximum, Minimum
config.Aggregation.Symmetric = true     // Force symmetric edges
config.Traversal.MaxInstances = 1000000 // Memory limit
config.Traversal.AllowCycles = false    // Prevent cycles in paths
```

### Advanced Pipeline

```go
// Custom progress tracking
progressCb := func(current, total int, message string) {
    fmt.Printf("Progress: %d/%d - %s\n", current, total, message)
}

engine := materialization.NewMaterializationEngine(graph, metaPath, config, progressCb)
result, err := engine.Materialize()

// Multiple output formats
materialization.SaveAsSimpleEdgeList(result.HomogeneousGraph, "edges.txt")
materialization.SaveAsCSV(result.HomogeneousGraph, "edges.csv") 
materialization.SaveAsJSON(result.HomogeneousGraph, "graph.json")
materialization.SaveMaterializationResult(result, "detailed_result.json")
```

## Common Use Cases

### Author Collaboration Network
**Input**: Authors write Papers  
**Meta-path**: Author → Paper → Author  
**Output**: Author-Author collaboration edges

### Citation Co-occurrence
**Input**: Papers cite Papers  
**Meta-path**: Paper → Paper → Paper  
**Output**: Papers connected by shared citations

### User-Item Similarity  
**Input**: Users rate Items  
**Meta-path**: User → Item → User  
**Output**: User-User similarity based on shared items

## Pipeline Integration

### With Louvain Community Detection

```bash
# Step 1: Materialize graph
./materialization -graph="network.txt" -props="props.txt" -path="path.txt" -output="edges.txt"

# Step 2: Run Louvain
./louvain edges.txt
```

### With SCAR Clustering

```bash
# Step 1: Materialize
./materialization -graph="data.txt" -props="types.txt" -path="metapath.txt" -output="homogeneous.txt"

# Step 2: Run SCAR  
./scar.exe "homogeneous.txt" -louvain -sketch-output
```

### Programmatic Pipeline

```go
func runPipeline(graphFile, propsFile, pathFile, outputDir string) error {
    // Materialize
    graph, metaPath, err := materialization.ParseSCARInput(graphFile, propsFile, pathFile)
    if err != nil {
        return err
    }
    
    config := materialization.DefaultMaterializationConfig()
    engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
    result, err := engine.Materialize()
    if err != nil {
        return err
    }
    
    // Save edge list
    edgeFile := filepath.Join(outputDir, "edges.txt")
    err = materialization.SaveAsSimpleEdgeList(result.HomogeneousGraph, edgeFile)
    if err != nil {
        return err
    }
    
    // Run community detection
    cmd := exec.Command("./louvain", edgeFile)
    cmd.Dir = outputDir
    return cmd.Run()
}
```

## Performance Tips

### For Large Graphs (>100K nodes)
```go
config.Traversal.MaxInstances = 500000    // Reduce memory usage
config.Aggregation.MinWeight = 0.1        // Filter weak edges
config.Aggregation.MaxEdges = 1000000     // Limit output size
```

### For High Quality Results
```go
config.Aggregation.Strategy = Average     // Better than Count
config.Traversal.AllowCycles = false      // Cleaner paths
config.Aggregation.Normalization = DegreeNorm // Normalize weights
```

### For Speed
```go
config.Traversal.MaxInstances = 100000    // Early termination
config.Aggregation.Strategy = Count       // Fastest aggregation
config.Progress.EnableProgress = false    // Disable progress reporting
```

## Error Handling

Common issues and solutions:

```go
// Check input validity
if err := graph.Validate(); err != nil {
    log.Fatalf("Invalid graph: %v", err)
}

// Handle memory limits
estimated, err := engine.GetMemoryEstimate()
if estimated > maxMemoryMB {
    config.Traversal.MaxInstances = 50000 // Reduce limit
}

// Verify meta-path traversability  
generator := materialization.NewInstanceGenerator(graph, metaPath, config.Traversal)
if err := generator.ValidateMetaPathTraversability(); err != nil {
    log.Fatalf("Meta-path not traversable: %v", err)
}
```

## Build and Install

```bash
# Build the library
go build ./pkg/materialization

# Build CLI tool (if available)
go build -o materialization ./cmd/materialization

# Run tests
go test ./pkg/materialization/...

# Run with verification
go test ./pkg/materialization/ -run TestVerifyMaterialization
```

## Examples

See `materialization_test.go` for complete examples including:
- Author collaboration networks
- Citation networks  
- User-item recommendation graphs
- Performance benchmarks
- Verification and debugging tools

## License

MIT License - see LICENSE file for details.
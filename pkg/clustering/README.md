# Clustering Package

A Go package for heterogeneous graph clustering with two distinct approaches:

1. **Materialization + Louvain**: Traditional approach that materializes the heterogeneous graph into a homogeneous graph, then applies the Louvain algorithm
2. **SCAR**: Sketch-based Community detection with Approximated Resistance - memory-efficient approach using probabilistic sketches

## Installation

```bash
go get github.com/gilchrisn/graph-clustering-service/pkg/clustering
```

## Quick Start

### Import the package

```go
import "github.com/gilchrisn/graph-clustering-service/pkg/clustering"
```

### Approach 1: Materialization + Louvain

```go
// Create configuration
config := clustering.DefaultMaterializationConfig()
config.GraphFile = "data/my_graph.json"
config.MetaPathFile = "data/my_meta_path.json"
config.OutputDir = "output/materialization"

// Run clustering
result, err := clustering.RunMaterializationClustering(config)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Found %d communities with modularity %.3f\n", 
    result.NumCommunities, result.Modularity)
```

### Approach 2: SCAR

```go
// Create configuration
config := clustering.DefaultScarConfig()
config.GraphFile = "data/my_graph.json"
config.MetaPathFile = "data/my_meta_path.json"
config.OutputDir = "output/scar"

// Run clustering
result, err := clustering.RunScarClustering(config)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Found %d communities with modularity %.3f\n", 
    result.NumCommunities, result.Modularity)
```

## Configuration

### MaterializationClusteringConfig

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `GraphFile` | string | Path to heterogeneous graph JSON file | Required |
| `MetaPathFile` | string | Path to meta path JSON file | Required |
| `OutputDir` | string | Output directory for results | Required |
| `AggregationStrategy` | enum | How to aggregate path instances (Count, Sum, Average, Maximum, Minimum) | Count |
| `MetaPathInterpretation` | enum | How to interpret meta paths (DirectTraversal, MeetingBased) | MeetingBased |
| `Symmetric` | bool | Force symmetric edges | true |
| `MinWeight` | float64 | Filter edges below this weight | 1.0 |
| `MaxEdges` | int | Keep only top-k edges (0 = no limit) | 0 |
| `MaxInstances` | int | Memory safety limit for path instances | 1,000,000 |
| `TimeoutSeconds` | int | Processing timeout | 300 |
| `TraversalParallelism` | int | Number of parallel workers | 4 |
| `LouvainMaxIterations` | int | Maximum Louvain iterations | 1 |
| `LouvainMinModularity` | float64 | Minimum modularity improvement | 0.001 |
| `RandomSeed` | int64 | For reproducibility | 42 |
| `Verbose` | bool | Enable verbose output | false |

### ScarClusteringConfig

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `GraphFile` | string | Path to heterogeneous graph JSON file | Required |
| `MetaPathFile` | string | Path to meta path JSON file | Required |
| `OutputDir` | string | Output directory for results | Required |
| `K` | int | Sketch size | 64 |
| `NK` | int | Number of independent hash functions | 8 |
| `MaxIterations` | int | Maximum algorithm iterations | 50 |
| `MinModularity` | float64 | Minimum modularity improvement | 1e-6 |
| `RandomSeed` | int64 | For reproducibility | 42 |
| `ParallelEnabled` | bool | Enable parallel processing | true |
| `NumWorkers` | int | Number of worker goroutines | 4 |
| `BatchSize` | int | Nodes per batch | 100 |
| `UpdateBuffer` | int | Channel buffer size | 10,000 |
| `Verbose` | bool | Enable verbose output | false |

## Output Files

Both approaches generate standardized output files:

### Materialization + Louvain Output
- `communities.mapping` - Community assignments (node → community)
- `communities.hierarchy` - Hierarchical community structure
- `communities.root` - Top-level communities
- `communities.edges` - Inter-community edges

### SCAR Output
- `scar.root` - Top-level structure
- `hierarchy-output/` - Directory with hierarchical structure per level
- `mapping-output/` - Directory with community mappings per level
- `edges-output/` - Directory with inter-community edges per level

## Result Structure

Both functions return a `ClusteringResult` struct:

```go
type ClusteringResult struct {
    Success            bool
    Error              string
    Approach           string
    Runtime            time.Duration
    MemoryPeakMB       int64
    Communities        map[string]int    // node_id -> community_id
    Modularity         float64
    NumCommunities     int
    NumLevels          int
    TotalIterations    int
    OutputFiles        OutputFiles
    AlgorithmDetails   interface{}       // MaterializationDetails or ScarDetails
}
```

## Verification

Verify that clustering completed successfully and outputs are valid:

```go
// Verify the clustering result
if err := clustering.VerifyClusteringResult(result); err != nil {
    log.Printf("Verification failed: %v", err)
}

// Verify specific output files
if err := clustering.VerifyMaterializationOutput(outputDir, "communities"); err != nil {
    log.Printf("Output verification failed: %v", err)
}
```

## Complete Example

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/gilchrisn/graph-clustering-service/pkg/clustering"
    "github.com/gilchrisn/graph-clustering-service/pkg/materialization"
)

func main() {
    // Configure materialization approach
    matConfig := clustering.DefaultMaterializationConfig()
    matConfig.GraphFile = "data/dblp_graph.json"
    matConfig.MetaPathFile = "data/author_coauthorship.json"
    matConfig.OutputDir = "output/materialization"
    matConfig.AggregationStrategy = materialization.Count
    matConfig.Symmetric = true
    matConfig.Verbose = true

    // Run materialization clustering
    result1, err := clustering.RunMaterializationClustering(matConfig)
    if err != nil {
        log.Fatalf("Materialization failed: %v", err)
    }

    // Configure SCAR approach
    scarConfig := clustering.DefaultScarConfig()
    scarConfig.GraphFile = "data/dblp_graph.json"
    scarConfig.MetaPathFile = "data/author_coauthorship.json"
    scarConfig.OutputDir = "output/scar"
    scarConfig.K = 128
    scarConfig.NK = 16
    scarConfig.Verbose = true

    // Run SCAR clustering
    result2, err := clustering.RunScarClustering(scarConfig)
    if err != nil {
        log.Fatalf("SCAR failed: %v", err)
    }

    // Compare results
    fmt.Printf("Materialization: %d communities, modularity %.3f, runtime %v\n",
        result1.NumCommunities, result1.Modularity, result1.Runtime)
    fmt.Printf("SCAR: %d communities, modularity %.3f, runtime %v\n",
        result2.NumCommunities, result2.Modularity, result2.Runtime)

    // Verify outputs
    if err := clustering.VerifyClusteringResult(result1); err != nil {
        log.Printf("Materialization verification failed: %v", err)
    }
    if err := clustering.VerifyClusteringResult(result2); err != nil {
        log.Printf("SCAR verification failed: %v", err)
    }

    fmt.Println("Clustering completed successfully!")
}
```

## Input File Formats

### Graph File (JSON)
```json
{
  "nodes": {
    "a1": {"type": "Author", "name": "Alice"},
    "p1": {"type": "Paper", "title": "ML Paper"},
    "v1": {"type": "Venue", "name": "ICML"}
  },
  "edges": [
    {"from": "a1", "to": "p1", "type": "writes", "weight": 1.0},
    {"from": "p1", "to": "v1", "type": "published_in", "weight": 1.0}
  ]
}
```

### Meta Path File (JSON)
```json
{
  "id": "author_coauthorship",
  "node_sequence": ["Author", "Paper", "Author"],
  "edge_sequence": ["writes", "writes"],
  "description": "Authors connected through co-authored papers"
}
```

## Testing

Run the test suite:

```bash
go test ./pkg/clustering
```

Run benchmarks:

```bash
go test -bench=. ./pkg/clustering
```

## Performance Comparison

| Aspect | Materialization + Louvain | SCAR |
|--------|---------------------------|------|
| **Memory Usage** | High (materializes full graph) | Low (sketch-based) |
| **Speed** | Moderate (depends on materialization) | Fast (parallel sketches) |
| **Accuracy** | High (exact computation) | Good (probabilistic approximation) |
| **Scalability** | Limited by memory | High (constant memory per node) |
| **Use Case** | Small to medium graphs, accuracy critical | Large graphs, efficiency critical |

## When to Use Which Approach

### Use Materialization + Louvain when:
- Graph is small to medium sized (< 100K nodes)
- Memory is abundant
- Accuracy is critical
- You need exact community detection results

### Use SCAR when:
- Graph is large (> 100K nodes)
- Memory is limited
- Speed is important
- Approximate results are acceptable
- You want to scale to very large graphs

## License

This package is part of the graph clustering service project.
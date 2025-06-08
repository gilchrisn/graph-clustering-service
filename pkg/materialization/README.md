# Materialization Module

The materialization module converts heterogeneous graphs into homogeneous graphs using meta path instances, enabling traditional graph algorithms like Louvain clustering.

## Features

- **Meta Path Traversal**: Find all instances of a given meta path in the graph
- **Two Interpretation Modes**: DirectTraversal and MeetingBased
- **Multiple Aggregation Strategies**: Count, Sum, Average, Maximum, Minimum
- **Weight Normalization**: Degree, Max, Standard normalization options
- **Memory Management**: Configurable limits and progress tracking
- **Parallel Processing**: Multi-threaded instance generation

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/gilchrisn/graph-clustering-service/pkg/materialization"
    "github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

func main() {
    // Load validated data
    graph, _ := validation.LoadAndValidateGraph("data/graph.json")
    metaPath, _ := validation.LoadAndValidateMetaPath("data/meta_path.json")
    
    // Configure materialization
    config := materialization.DefaultMaterializationConfig()
    config.Aggregation.Strategy = materialization.Count
    config.Aggregation.Interpretation = materialization.DirectTraversal
    
    // Create engine and materialize
    engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
    result, err := engine.Materialize()
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Generated %d edges from %d instances\n", 
        len(result.HomogeneousGraph.Edges), result.Statistics.InstancesGenerated)
}
```

## Core Concepts

### Meta Path Instances
A meta path instance is a concrete path in the graph that follows the meta path pattern:

```
Meta Path: Author → Paper → Author
Instance:  Alice → "ML Paper" → Bob
```

### Interpretation Modes

#### 1. DirectTraversal (Default)
Connects start and end nodes directly:
- `Alice → "ML Paper" → Bob` creates edge `Alice ↔ Bob`
- Weight = aggregation of all instances between Alice and Bob

#### 2. MeetingBased  
Connects nodes that meet at intermediate nodes:
- `Alice → "ICML"`, `Bob → "ICML"` creates edge `Alice ↔ Bob`
- Weight = combination of their relationships to "ICML"

### Aggregation Strategies

```go
type AggregationStrategy int

const (
    Count   AggregationStrategy = iota // Count instances
    Sum                               // Sum instance weights
    Average                           // Average instance weights  
    Maximum                           // Maximum instance weight
    Minimum                           // Minimum instance weight
)
```

## Configuration

### Full Configuration Example

```go
config := materialization.MaterializationConfig{
    Traversal: materialization.TraversalConfig{
        Strategy:       materialization.BFS,
        MaxPathLength:  10,
        AllowCycles:    false,
        MaxInstances:   1000000,
        TimeoutSeconds: 300,
        Parallelism:    4,
    },
    Aggregation: materialization.AggregationConfig{
        Strategy:      materialization.Count,
        Interpretation: materialization.DirectTraversal,
        Normalization: materialization.NoNormalization,
        MinWeight:     1.0,
        MaxEdges:      0, // No limit
        Symmetric:     true,
    },
    Progress: materialization.ProgressConfig{
        EnableProgress: true,
        ReportInterval: 10000,
    },
}
```

### Key Parameters

- **MaxInstances**: Memory limit for path instances
- **TimeoutSeconds**: Maximum processing time
- **Parallelism**: Number of worker threads
- **MinWeight**: Filter edges below threshold
- **MaxEdges**: Keep only top-k edges
- **Symmetric**: Force symmetric edges

## Usage Patterns

### Basic Materialization

```go
func basicMaterialization(graph *models.HeterogeneousGraph, metaPath *models.MetaPath) {
    config := materialization.DefaultMaterializationConfig()
    
    progressCb := func(current, total int, message string) {
        fmt.Printf("\rProgress: %d/%d - %s", current, total, message)
    }
    
    engine := materialization.NewMaterializationEngine(graph, metaPath, config, progressCb)
    result, err := engine.Materialize()
    if err != nil {
        log.Fatal(err)
    }
    
    // Save result
    materialization.SaveHomogeneousGraph(result.HomogeneousGraph, "output/graph.edgelist")
}
```

### Memory-Conscious Materialization

```go
func checkFeasibility(engine *materialization.MaterializationEngine) {
    canMaterialize, reason, err := engine.CanMaterialize(1000) // 1GB limit
    if err != nil {
        log.Fatal(err)
    }
    
    if !canMaterialize {
        fmt.Printf("Cannot materialize: %s\n", reason)
        return
    }
    
    // Proceed with materialization
    result, err := engine.Materialize()
    // ...
}
```

## Output Formats

### Homogeneous Graph Structure

```go
type HomogeneousGraph struct {
    NodeType   string                 // "Author" for symmetric paths
    Nodes      map[string]Node        // All nodes in result
    Edges      map[EdgeKey]float64    // (from,to) -> weight  
    Statistics GraphStatistics        // Graph metrics
    MetaPath   models.MetaPath        // Original meta path
}
```

### Supported Output Formats

```go
// Edge list format (default)
materialization.SaveHomogeneousGraph(graph, "output/graph.edgelist")

// CSV format  
materialization.SaveHomogeneousGraph(graph, "output/graph.csv")

// JSON format
materialization.SaveHomogeneousGraph(graph, "output/graph.json")
```

### Edge List Format
```
6 8
alice bob 3.000000
alice charlie 1.000000
bob charlie 2.000000
```

## Integration with Louvain

The materialized homogeneous graph can be directly used with the Louvain module:

```go
// Materialize graph
result, err := engine.Materialize()
if err != nil {
    log.Fatal(err)
}

// Convert to Louvain format and run clustering
louvainGraph := convertToLouvainGraph(result.HomogeneousGraph)
louvainConfig := louvain.DefaultLouvainConfig()
communities, err := louvain.RunLouvain(louvainGraph, louvainConfig)
```

This forms the complete pipeline: `Heterogeneous Graph → Materialization → Louvain → Communities`
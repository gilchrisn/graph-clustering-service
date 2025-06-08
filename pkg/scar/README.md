# SCAR Module

The SCAR (Sketch-based Community detection for heterogeAR graphs) module implements sketch-based community detection that works directly on heterogeneous graphs without requiring full materialization.

## Features

- **Direct Heterogeneous Processing**: Works on heterogeneous graphs without materialization
- **Memory Efficient**: Uses Bottom-K sketches instead of storing all path instances
- **Scalable**: O(k×n) complexity where k is sketch size, n is number of nodes
- **Meta Path Aware**: Uses meta paths to define similarity between nodes
- **Multi-Hash Functions**: Uses multiple independent hash functions for accuracy
- **Three-Phase Optimization**: Initial, quick, and sophisticated merging strategies

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/gilchrisn/graph-clustering-service/pkg/scar"
    "github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

func main() {
    // Load heterogeneous graph
    graph, _ := validation.LoadAndValidateGraph("data/graph.json")
    
    // Convert to SCAR format (helper function needed)
    scarGraph := convertToScarGraph(graph)
    
    // Configure SCAR
    config := scar.DefaultScarConfig()
    config.MetaPath = scar.MetaPath{
        NodeTypes: []string{"Author", "Paper", "Author"},
        EdgeTypes: []string{"writes", "writes"},
    }
    config.K = 64   // Sketch size
    config.NK = 8   // Number of hash functions
    config.Verbose = true
    
    // Run SCAR
    result, err := scar.RunScar(scarGraph, config)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Found %d communities with modularity %.6f\n",
        len(result.FinalCommunities), result.Modularity)
}
```

## Core Concepts

### Bottom-K Sketches
SCAR uses Bottom-K sketches to efficiently estimate node similarities without storing all path instances:

```go
type VertexBottomKSketch struct {
    Sketches [][]uint64 // NK independent sketches, each of size K
    K        int        // Sketch size  
    NK       int        // Number of independent hash functions
    PathPos  int        // Position in meta path
}
```

### Multi-Hash Functions
Uses NK independent hash functions to improve estimation accuracy:
- Each sketch has NK independent hash functions
- Final estimates averaged across all hash functions
- Reduces variance in similarity estimations

### Three-Phase Merging

1. **Initial Phase**: Quality optimization with sophisticated calculations
2. **Quick Phase**: Fast degree-based merging for efficiency  
3. **Sophisticated Phase**: E-function based merging for accuracy

## Configuration

### Full Configuration Example

```go
config := scar.ScarConfig{
    K:                64,           // Sketch size (larger = more accurate)
    NK:               8,            // Number of hash functions  
    MetaPath:         metaPath,     // Meta path definition
    MaxIterations:    100,          // Maximum iterations per level
    MinModularity:    1e-6,         // Convergence threshold
    RandomSeed:       42,           // For reproducible results
    Verbose:          true,         // Enable detailed output
    ProgressCallback: progressCb,   // Progress monitoring
}
```

### Key Parameters

- **K (Sketch Size)**: Larger values increase accuracy but use more memory
- **NK (Hash Functions)**: More functions improve accuracy, typically 4-16
- **MetaPath**: Defines which nodes are considered similar
- **RandomSeed**: Set for reproducible results

## Meta Path Definition

```go
type MetaPath struct {
    NodeTypes []string // Sequence of node types
    EdgeTypes []string // Sequence of edge types  
}

// Example: Author collaboration through papers
metaPath := scar.MetaPath{
    NodeTypes: []string{"Author", "Paper", "Author"},
    EdgeTypes: []string{"writes", "writes"},
}

// Example: Papers in same venue
metaPath := scar.MetaPath{
    NodeTypes: []string{"Paper", "Venue", "Paper"},
    EdgeTypes: []string{"published_in", "published_in"},
}
```

## Usage Patterns

### Basic Community Detection

```go
func runScarClustering(graph *scar.HeterogeneousGraph) {
    config := scar.DefaultScarConfig()
    config.MetaPath = scar.MetaPath{
        NodeTypes: []string{"Author", "Paper", "Author"},
        EdgeTypes: []string{"writes", "writes"},
    }
    config.K = 32
    config.Verbose = true
    
    result, err := scar.RunScar(graph, config)
    if err != nil {
        log.Fatal(err)
    }
    
    // Analyze results
    fmt.Printf("SCAR Results:\n")
    fmt.Printf("- Modularity: %.6f\n", result.Modularity)
    fmt.Printf("- Levels: %d\n", result.NumLevels)
    fmt.Printf("- Runtime: %v\n", result.Statistics.TotalDuration)
    
    // Print community assignments
    communities := make(map[int][]string)
    for node, comm := range result.FinalCommunities {
        communities[comm] = append(communities[comm], node)
    }
    
    for commID, nodes := range communities {
        fmt.Printf("Community %d: %v\n", commID, nodes)
    }
}
```

### Progress Monitoring

```go
config.ProgressCallback = func(level int, iteration int, modularity float64, nodes int) {
    fmt.Printf("Level %d, Iteration %d: modularity=%.6f, nodes=%d\n", 
        level, iteration, modularity, nodes)
}
```

### Memory-Constrained Setup

```go
// For large graphs, use smaller sketches
config := scar.DefaultScarConfig()
config.K = 16          // Smaller sketch size
config.NK = 4          // Fewer hash functions
config.MaxIterations = 20  // Faster convergence
```

### High-Accuracy Setup

```go
// For maximum accuracy
config := scar.DefaultScarConfig()
config.K = 128         // Larger sketch size
config.NK = 16         // More hash functions
config.MinModularity = 1e-8  // Tighter convergence
```

## Performance Characteristics

### Memory Usage
- **Sketches**: O(K × NK × N) where N is number of nodes
- **Community Data**: O(N + C) where C is number of communities
- **Total**: Much lower than materialization approach

### Time Complexity
- **Sketch Construction**: O(K × NK × |instances|)
- **Optimization**: O(iterations × N × average_degree)
- **Total**: O(K × NK × N) typically

### Comparison with Materialization

| Aspect | SCAR | Materialization + Louvain |
|--------|------|---------------------------|
| Memory | O(K×N) | O(M²) for dense paths |
| Time | O(K×N) | O(M×N + M²) |
| Accuracy | ~95-99% | 100% (exact) |
| Scalability | Excellent | Limited by memory |

## Integration with Validation

```go
func scarPipeline(graphFile, metaPathFile string) {
    // Step 1: Load and validate
    graph, err := validation.LoadAndValidateGraph(graphFile)
    if err != nil {
        log.Fatal(err)
    }
    
    metaPath, err := validation.LoadAndValidateMetaPath(metaPathFile)
    if err != nil {
        log.Fatal(err)
    }
    
    // Step 2: Convert formats
    scarGraph := convertToScarGraph(graph)
    scarMetaPath := convertToScarMetaPath(metaPath)
    
    // Step 3: Run SCAR
    config := scar.DefaultScarConfig()
    config.MetaPath = scarMetaPath
    config.Verbose = true
    
    result, err := scar.RunScar(scarGraph, config)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("SCAR completed: %.6f modularity\n", result.Modularity)
}
```

## Output Structure

### SCAR Result

```go
type ScarResult struct {
    Levels           []LevelInfo         // Hierarchy of levels
    FinalCommunities map[string]int      // Final node -> community mapping  
    Modularity       float64             // Final modularity score
    NumLevels        int                 // Number of hierarchy levels
    Statistics       ScarStats           // Performance statistics
    HierarchyLevels  []map[string][]string // For output compatibility
    MappingLevels    []map[string][]string // For output compatibility
}
```

### Statistics

```go
type ScarStats struct {
    TotalLevels     int           // Number of levels processed
    TotalIterations int           // Total iterations across levels
    TotalDuration   time.Duration // Total runtime
    FinalModularity float64       // Final modularity achieved
    InitialNodes    int           // Starting number of nodes
    InitialEdges    int           // Starting number of edges
    FinalNodes      int           // Final number of nodes
}
```

## Output Files

SCAR generates Louvain-compatible output files:

```go
// Write all output files  
err := scar.WriteAll(result, graph, "output/", "scar_result")

// Generated files:
// output/hierarchy-output/scar_result_*.dat
// output/mapping-output/scar_result_*.dat  
// output/scar_result.root
// output/edges-output/scar_result_*.dat
```

## Testing and Validation

```bash
# Run SCAR tests
go test ./pkg/scar/

# Run with verbose output
go test -v ./pkg/scar/

# Test specific fixes
go test -run=TestFixedScarImplementation ./pkg/scar/

# Performance benchmarks
go test -bench=. ./pkg/scar/
```

This makes SCAR suitable for large-scale heterogeneous graph analysis where traditional materialization approaches would be memory-prohibitive.
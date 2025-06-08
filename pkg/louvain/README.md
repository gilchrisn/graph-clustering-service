# Louvain Module

The Louvain module implements the traditional Louvain algorithm for community detection in homogeneous graphs. It works on the output of the materialization module to find hierarchical community structures.

## Features

- **Multi-level Community Detection**: Hierarchical community detection with multiple levels
- **Modularity Optimization**: Uses modularity as the objective function
- **Configurable Parameters**: Community size limits, convergence thresholds, iteration limits
- **Progress Tracking**: Real-time progress callbacks and statistics
- **Output Compatibility**: Standard output formats for visualization and analysis
- **Memory Efficient**: Optimized for large graphs with millions of edges

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/gilchrisn/graph-clustering-service/pkg/louvain"
)

func main() {
    // Create or load a homogeneous graph
    graph := louvain.NewHomogeneousGraph()
    
    // Add nodes and edges
    graph.AddEdge("a", "b", 1.0)
    graph.AddEdge("b", "c", 1.0)
    graph.AddEdge("c", "d", 1.0)
    graph.AddEdge("d", "e", 0.1) // Weak connection
    graph.AddEdge("e", "f", 1.0)
    
    // Configure algorithm
    config := louvain.DefaultLouvainConfig()
    config.Verbose = true
    
    // Run Louvain
    result, err := louvain.RunLouvain(graph, config)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Found %d communities with modularity %.4f\n", 
        len(result.Levels[len(result.Levels)-1].Communities), result.Modularity)
}
```

## Algorithm Overview

The Louvain algorithm works in two main phases:

### Phase 1: Local Optimization
- Each node starts in its own community
- Iteratively move nodes to neighboring communities that maximize modularity gain
- Continue until no beneficial moves remain

### Phase 2: Network Aggregation  
- Create super-graph where communities become nodes
- Edge weights between super-nodes = sum of inter-community edges
- Repeat Phase 1 on the super-graph

### Multi-level Process
Repeat phases until no more improvement is possible, creating a hierarchy of communities.

## Configuration

### Full Configuration Example

```go
config := louvain.LouvainConfig{
    MaxCommunitySize:  100,        // Limit community size (0 = no limit)
    MinModularity:     0.00001,    // Convergence threshold
    MaxIterations:     100,        // Maximum iterations per level
    NumWorkers:        4,          // Parallel processing workers
    ChunkSize:         32,         // Node processing chunk size
    RandomSeed:        42,         // For reproducible results
    Verbose:           true,       // Enable progress output
    ProgressCallback:  progressCb, // Custom progress tracking
}
```

### Key Parameters

- **MaxCommunitySize**: Prevents communities from growing too large
- **MinModularity**: Stops when modularity improvement falls below threshold
- **MaxIterations**: Safety limit to prevent infinite loops
- **RandomSeed**: Set for reproducible results (-1 for random)
- **NumWorkers**: Parallel processing (0 = auto-detect CPU cores)

## Input Format

### HomogeneousGraph Structure

```go
type HomogeneousGraph struct {
    Nodes       map[string]Node        // Node ID -> Node data
    Edges       map[EdgeKey]float64    // (from,to) -> weight
    TotalWeight float64                // Sum of all edge weights
    NodeList    []string               // Ordered node list
}

type Node struct {
    ID         string                 // Unique identifier
    Weight     float64                // Node weight  
    Degree     float64                // Weighted degree
    Properties map[string]interface{} // Additional properties
}
```

### Loading from Edge List

```go
func loadFromEdgeList(filename string) (*louvain.HomogeneousGraph, error) {
    graph := louvain.NewHomogeneousGraph()
    
    // Read file line by line
    // First line: num_nodes num_edges
    // Subsequent lines: from to weight
    
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    scanner := bufio.NewScanner(file)
    lineNum := 0
    
    for scanner.Scan() {
        lineNum++
        line := strings.TrimSpace(scanner.Text())
        
        if lineNum == 1 {
            // Parse header
            parts := strings.Fields(line)
            fmt.Printf("Expected %s nodes and %s edges\n", parts[0], parts[1])
            continue
        }
        
        // Parse edge: from to weight
        parts := strings.Fields(line)
        if len(parts) >= 2 {
            from := parts[0]
            to := parts[1]
            weight := 1.0
            
            if len(parts) >= 3 {
                weight, _ = strconv.ParseFloat(parts[2], 64)
            }
            
            graph.AddEdge(from, to, weight)
        }
    }
    
    return graph, nil
}
```

## Usage Patterns

### Basic Community Detection

```go
func basicCommunityDetection() {
    graph := createTestGraph()
    
    config := louvain.DefaultLouvainConfig()
    config.Verbose = true
    config.RandomSeed = 42
    
    result, err := louvain.RunLouvain(graph, config)
    if err != nil {
        log.Fatal(err)
    }
    
    // Print results
    fmt.Printf("Modularity: %.6f\n", result.Modularity)
    fmt.Printf("Levels: %d\n", result.NumLevels)
    
    // Print final communities
    finalLevel := result.Levels[len(result.Levels)-1]
    for commID, nodes := range finalLevel.Communities {
        fmt.Printf("Community %d: %v\n", commID, nodes)
    }
}
```

### Progress Monitoring

```go
config.ProgressCallback = func(level, iteration int, message string) {
    if level >= 0 {
        fmt.Printf("Level %d: %s\n", level, message)
    } else {
        fmt.Printf("  %s\n", message)
    }
}
```

### Size-Constrained Communities

```go
config := louvain.DefaultLouvainConfig()
config.MaxCommunitySize = 50  // Maximum 50 nodes per community

result, err := louvain.RunLouvain(graph, config)
if err != nil {
    log.Fatal(err)
}

// Verify size constraints
for _, level := range result.Levels {
    for commID, nodes := range level.Communities {
        if len(nodes) > 50 {
            fmt.Printf("Warning: Community %d has %d nodes\n", commID, len(nodes))
        }
    }
}
```

## Output Structure

### LouvainResult

```go
type LouvainResult struct {
    Levels           []LevelInfo          // Hierarchy of levels
    FinalCommunities map[string]int       // Final node -> community mapping
    Modularity       float64              // Final modularity score
    NumLevels        int                  // Number of hierarchy levels
    Statistics       LouvainStats         // Performance statistics
}
```

### LevelInfo

```go
type LevelInfo struct {
    Level          int                  // Level number (0-based)
    Communities    map[int][]string     // Community ID -> nodes
    CommunityMap   map[string]int       // Node -> community mapping
    Graph          *HomogeneousGraph    // Graph at this level
    Modularity     float64              // Modularity at this level
    NumCommunities int                  // Number of communities
    NumMoves       int                  // Number of node moves
}
```

## Output Files

The Louvain module generates several output files compatible with standard network analysis tools:

### File Types Generated

```go
// Write all output files
writer := louvain.NewFileWriter()
err := writer.WriteAll(result, graph, "output/", "communities")

// Generated files:
// output/communities.mapping   - Community to nodes mapping
// output/communities.hierarchy - Hierarchical structure  
// output/communities.root      - Top-level communities
// output/communities.edges     - Edges between communities
```

## Integration with Materialization

Complete pipeline from heterogeneous to communities:

```go
func fullPipeline(graphFile, metaPathFile string) {
    // Step 1: Validation
    graph, err := validation.LoadAndValidateGraph(graphFile)
    if err != nil {
        log.Fatal(err)
    }
    
    metaPath, err := validation.LoadAndValidateMetaPath(metaPathFile)
    if err != nil {
        log.Fatal(err)
    }
    
    // Step 2: Materialization
    config := materialization.DefaultMaterializationConfig()
    engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
    matResult, err := engine.Materialize()
    if err != nil {
        log.Fatal(err)
    }
    
    // Step 3: Convert to Louvain format
    louvainGraph := convertToLouvainGraph(matResult.HomogeneousGraph)
    
    // Step 4: Run Louvain
    louvainConfig := louvain.DefaultLouvainConfig()
    louvainConfig.Verbose = true
    
    communities, err := louvain.RunLouvain(louvainGraph, louvainConfig)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Pipeline complete: %.6f modularity, %d communities\n",
        communities.Modularity, len(communities.FinalCommunities))
}
```

## Testing and Benchmarks

```bash
# Run Louvain tests
go test ./pkg/louvain/

# Run with verbose output
go test -v ./pkg/louvain/

# Performance benchmarks
go test -bench=. ./pkg/louvain/

# Test determinism
go test -run=TestDeterminism ./pkg/louvain/
```
# Louvain Algorithm - Complete API Documentation

This document provides comprehensive API documentation for the Louvain community detection algorithm implementation, including input/output formats, data structures, function interfaces, and pipeline integration guidelines.

## Table of Contents

1. [Core API Overview](#core-api-overview)
2. [Data Structures](#data-structures)
3. [Input/Output API](#inputoutput-api)
4. [Algorithm API](#algorithm-api)
5. [Configuration API](#configuration-api)
6. [Pipeline Integration](#pipeline-integration)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)
9. [Examples](#examples)

---

## Core API Overview

### Package Structure
```go
package louvain

// Core algorithm
func RunLouvain(graph *NormalizedGraph, config LouvainConfig) (*LouvainResult, error)

// Input parsing
func NewGraphParser() *GraphParser
func (p *GraphParser) ParseEdgeList(filename string) (*ParseResult, error)

// Output generation
func NewFileWriter() OutputWriter
func (fw *FileWriter) WriteAll(result *LouvainResult, parser *GraphParser, outputDir string, prefix string) error

// Graph construction
func NewNormalizedGraph(numNodes int) *NormalizedGraph
func NewHomogeneousGraph() *HomogeneousGraph
```

### Pipeline Flow
```
Input File → GraphParser → NormalizedGraph → RunLouvain → LouvainResult → OutputWriter → Output Files
```

---

## Data Structures

### Core Graph Types

#### `NormalizedGraph`
Primary graph representation with integer node indices.

```go
type NormalizedGraph struct {
    NumNodes     int         // Number of nodes in the graph
    Degrees      []float64   // Node degrees (includes 2×self-loop weight)
    Weights      []float64   // Node weights (default 1.0)
    Adjacency    [][]int     // Adjacency list (neighbor indices)
    EdgeWeights  [][]float64 // Edge weights corresponding to adjacency
    TotalWeight  float64     // Sum of all edge weights
}

// Methods
func (g *NormalizedGraph) AddEdge(from, to int, weight float64)
func (g *NormalizedGraph) GetNeighbors(nodeID int) map[int]float64
func (g *NormalizedGraph) GetNodeDegree(nodeID int) float64
func (g *NormalizedGraph) GetEdgeWeight(from, to int) float64
func (g *NormalizedGraph) Clone() *NormalizedGraph
func (g *NormalizedGraph) Validate() error
```

#### `HomogeneousGraph` (Legacy)
Alternative graph representation with string node IDs.

```go
type HomogeneousGraph struct {
    Nodes       map[string]Node        // Node ID → Node mapping
    Edges       map[EdgeKey]float64    // Edge → Weight mapping
    TotalWeight float64                // Sum of all edge weights
    NodeList    []string               // Ordered list of node IDs
}

// Methods (similar to NormalizedGraph but with string IDs)
func (g *HomogeneousGraph) AddNode(nodeID string, weight float64)
func (g *HomogeneousGraph) AddEdge(from, to string, weight float64)
```

### Input/Output Types

#### `GraphParser`
Handles conversion between original node IDs and normalized indices.

```go
type GraphParser struct {
    OriginalToNormalized map[string]int // Original ID → Normalized index
    NormalizedToOriginal map[int]string // Normalized index → Original ID
    NumNodes             int            // Total number of nodes
}

// Methods
func (p *GraphParser) ParseEdgeList(filename string) (*ParseResult, error)
func (p *GraphParser) GetOriginalID(normalizedID int) (string, bool)
func (p *GraphParser) GetNormalizedID(originalID string) (int, bool)
func (p *GraphParser) ConvertCommunityMapping(normalizedMapping map[int]int) map[string]int
```

#### `ParseResult`
Combined result of parsing input.

```go
type ParseResult struct {
    Graph   *NormalizedGraph // Parsed and normalized graph
    Parser  *GraphParser     // Parser with ID mappings
}
```

### Algorithm Configuration

#### `LouvainConfig`
Configuration parameters for the algorithm.

```go
type LouvainConfig struct {
    MaxCommunitySize  int              // Maximum nodes per community (0 = no limit)
    MinModularity     float64          // Minimum modularity improvement threshold
    MaxIterations     int              // Maximum optimization levels
    NumWorkers        int              // Parallel processing workers
    ChunkSize         int              // Node processing chunk size
    RandomSeed        int64            // Random seed for reproducibility
    Verbose           bool             // Enable detailed logging
    ProgressCallback  ProgressCallback // Progress reporting function
}

// Default configuration
func DefaultLouvainConfig() LouvainConfig
```

#### `ProgressCallback`
Function type for progress reporting.

```go
type ProgressCallback func(level, iteration int, message string)
```

### Algorithm Results

#### `LouvainResult`
Complete algorithm output.

```go
type LouvainResult struct {
    Levels           []LevelInfo          // Hierarchical community structure
    FinalCommunities map[int]int          // Final normalized node → community mapping
    Modularity       float64              // Final modularity score
    NumLevels        int                  // Number of hierarchy levels
    Statistics       LouvainStats         // Performance statistics
    Parser           *GraphParser         // Parser for ID conversion (set by output)
}
```

#### `LevelInfo`
Information about one hierarchy level.

```go
type LevelInfo struct {
    Level          int                  // Level number (0, 1, 2, ...)
    Communities    map[int][]int        // Community ID → Node list mapping
    CommunityMap   map[int]int          // Node → Community mapping
    Graph          *NormalizedGraph     // Graph at this level
    Modularity     float64              // Modularity at this level
    NumCommunities int                  // Number of communities
    NumMoves       int                  // Number of node moves made
}
```

#### `LouvainStats`
Performance and execution statistics.

```go
type LouvainStats struct {
    TotalIterations   int                // Total optimization iterations
    TotalMoves        int                // Total node moves across all levels
    RuntimeMS         int64              // Total runtime in milliseconds
    MemoryPeakMB      int64              // Peak memory usage in MB
    LevelStats        []LevelStats       // Per-level statistics
}

type LevelStats struct {
    Level             int                // Level number
    Iterations        int                // Iterations at this level
    Moves             int                // Moves at this level
    InitialModularity float64            // Starting modularity
    FinalModularity   float64            // Ending modularity
    RuntimeMS         int64              // Level runtime in milliseconds
}
```

### Output Interface

#### `OutputWriter`
Interface for flexible output generation.

```go
type OutputWriter interface {
    WriteMapping(result *LouvainResult, parser *GraphParser, path string) error
    WriteHierarchy(result *LouvainResult, parser *GraphParser, path string) error
    WriteRoot(result *LouvainResult, parser *GraphParser, path string) error
    WriteEdges(result *LouvainResult, parser *GraphParser, path string) error
    WriteAll(result *LouvainResult, parser *GraphParser, outputDir string, prefix string) error
}
```

---

## Input/Output API

### Input Processing

#### Edge List Format
```
# Comments start with #
from_node to_node [weight]
```

- **Node IDs**: Strings or numbers
- **Weight**: Optional floating point (defaults to 1.0)
- **Self-loops**: Supported (`node node weight`)
- **Comments**: Lines starting with `#`

#### Parsing Functions

```go
// Create new parser
parser := NewGraphParser()

// Parse edge list file
result, err := parser.ParseEdgeList("input.txt")
if err != nil {
    return fmt.Errorf("parsing failed: %w", err)
}

graph := result.Graph   // NormalizedGraph for algorithm
parser := result.Parser // GraphParser for ID conversion
```

#### Manual Graph Construction

```go
// Create empty graph
graph := NewNormalizedGraph(5) // 5 nodes

// Add edges (nodes must be 0-indexed integers)
graph.AddEdge(0, 1, 2.5)  // Node 0 to Node 1, weight 2.5
graph.AddEdge(1, 2, 1.0)  // Node 1 to Node 2, weight 1.0
graph.AddEdge(0, 0, 1.5)  // Self-loop on Node 0, weight 1.5

// Validate graph
if err := graph.Validate(); err != nil {
    return fmt.Errorf("invalid graph: %w", err)
}
```

### Output Generation

#### File-Based Output

```go
// Create output writer
writer := NewFileWriter()

// Write all output files
err := writer.WriteAll(result, parser, "output_dir", "prefix")
// Creates: prefix.mapping, prefix.hierarchy, prefix.root, prefix.edges

// Write individual files
err = writer.WriteMapping(result, parser, "communities.txt")
err = writer.WriteHierarchy(result, parser, "hierarchy.txt")
```

#### Generated Files

1. **`prefix.mapping`**: Community membership with original node IDs
2. **`prefix.hierarchy`**: Hierarchical community structure
3. **`prefix.root`**: Top-level communities only
4. **`prefix.edges`**: Inter-community connections
5. **`simple_mapping.txt`**: Simple node→community pairs

#### Programmatic Access

```go
// Get final communities with original node IDs
originalCommunities := parser.ConvertCommunityMapping(result.FinalCommunities)
for nodeID, communityID := range originalCommunities {
    fmt.Printf("Node %s is in community %d\n", nodeID, communityID)
}

// Access hierarchical structure
for level, levelInfo := range result.Levels {
    fmt.Printf("Level %d: %d communities, modularity %.4f\n", 
        level, levelInfo.NumCommunities, levelInfo.Modularity)
}
```

---

## Algorithm API

### Basic Usage

```go
// Configure algorithm
config := DefaultLouvainConfig()
config.MaxIterations = 10
config.MinModularity = 0.001
config.Verbose = true

// Run algorithm
result, err := RunLouvain(graph, config)
if err != nil {
    return fmt.Errorf("algorithm failed: %w", err)
}

fmt.Printf("Final modularity: %.4f\n", result.Modularity)
fmt.Printf("Number of levels: %d\n", result.NumLevels)
```

### Advanced Configuration

```go
config := LouvainConfig{
    MaxCommunitySize: 1000,           // Limit community size
    MinModularity:    0.0001,         // Fine-grained optimization
    MaxIterations:    20,             // Deep hierarchy
    NumWorkers:       8,              // Parallel processing
    ChunkSize:        64,             // Processing batch size
    RandomSeed:       42,             // Reproducible results
    Verbose:          true,           // Detailed logging
    ProgressCallback: func(level, iteration int, message string) {
        fmt.Printf("[Level %d, Iter %d] %s\n", level, iteration, message)
    },
}
```

### Result Analysis

```go
// Final statistics
stats := result.Statistics
fmt.Printf("Total runtime: %d ms\n", stats.RuntimeMS)
fmt.Printf("Total iterations: %d\n", stats.TotalIterations)
fmt.Printf("Peak memory: %d MB\n", stats.MemoryPeakMB)

// Per-level analysis
for _, levelStats := range stats.LevelStats {
    fmt.Printf("Level %d: %.4f → %.4f modularity (%d moves)\n",
        levelStats.Level, 
        levelStats.InitialModularity,
        levelStats.FinalModularity,
        levelStats.Moves)
}

// Community size distribution
finalLevel := result.Levels[len(result.Levels)-1]
sizes := make([]int, 0)
for _, nodes := range finalLevel.Communities {
    sizes = append(sizes, len(nodes))
}
fmt.Printf("Community sizes: %v\n", sizes)
```

---

## Configuration API

### Default Configuration

```go
config := DefaultLouvainConfig()
// Returns:
// MaxCommunitySize: 0 (no limit)
// MinModularity: 0.001
// MaxIterations: 1
// NumWorkers: 4
// ChunkSize: 32
// RandomSeed: -1 (time-based)
// Verbose: false
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MaxCommunitySize` | `int` | `0` | Maximum nodes per community (0 = no limit) |
| `MinModularity` | `float64` | `0.001` | Minimum modularity improvement to continue |
| `MaxIterations` | `int` | `1` | Maximum hierarchy levels to explore |
| `NumWorkers` | `int` | `4` | Number of parallel workers (unused currently) |
| `ChunkSize` | `int` | `32` | Node processing batch size |
| `RandomSeed` | `int64` | `-1` | Random seed (-1 = time-based) |
| `Verbose` | `bool` | `false` | Enable detailed console output |
| `ProgressCallback` | `func` | `nil` | Progress reporting function |

### Tuning Guidelines

#### For Large Graphs (>10K nodes)
```go
config.ChunkSize = 128        // Larger batches
config.MinModularity = 0.0001 // Finer optimization
config.MaxIterations = 5      // Limit levels for performance
```

#### For High-Quality Results
```go
config.MinModularity = 0.00001 // Very fine optimization
config.MaxIterations = 20      // Deep hierarchy
config.RandomSeed = 42         // Reproducible results
```

#### For Fast Prototyping
```go
config.MinModularity = 0.01   // Coarse optimization
config.MaxIterations = 3      // Shallow hierarchy
config.Verbose = true         // Monitor progress
```

---

## Pipeline Integration

### REST API Integration

```go
type LouvainService struct {
    parser *GraphParser
    writer OutputWriter
}

func (s *LouvainService) ProcessGraph(inputData []byte, config LouvainConfig) (*LouvainResult, error) {
    // Parse input
    tempFile := "temp_input.txt"
    if err := ioutil.WriteFile(tempFile, inputData, 0644); err != nil {
        return nil, err
    }
    defer os.Remove(tempFile)
    
    parseResult, err := s.parser.ParseEdgeList(tempFile)
    if err != nil {
        return nil, fmt.Errorf("parsing failed: %w", err)
    }
    
    // Run algorithm
    result, err := RunLouvain(parseResult.Graph, config)
    if err != nil {
        return nil, fmt.Errorf("algorithm failed: %w", err)
    }
    
    result.Parser = parseResult.Parser
    return result, nil
}
```

### Stream Processing

```go
type StreamProcessor struct {
    graph  *NormalizedGraph
    parser *GraphParser
    nodeMap map[string]int
}

func (sp *StreamProcessor) AddEdge(from, to string, weight float64) error {
    // Get or create normalized indices
    fromIdx, exists := sp.nodeMap[from]
    if !exists {
        fromIdx = len(sp.nodeMap)
        sp.nodeMap[from] = fromIdx
        sp.parser.OriginalToNormalized[from] = fromIdx
        sp.parser.NormalizedToOriginal[fromIdx] = from
    }
    
    toIdx, exists := sp.nodeMap[to]
    if !exists {
        toIdx = len(sp.nodeMap)
        sp.nodeMap[to] = toIdx
        sp.parser.OriginalToNormalized[to] = toIdx
        sp.parser.NormalizedToOriginal[toIdx] = to
    }
    
    // Resize graph if needed
    if max(fromIdx, toIdx) >= sp.graph.NumNodes {
        sp.resizeGraph(max(fromIdx, toIdx) + 1)
    }
    
    sp.graph.AddEdge(fromIdx, toIdx, weight)
    return nil
}

func (sp *StreamProcessor) Process(config LouvainConfig) (*LouvainResult, error) {
    return RunLouvain(sp.graph, config)
}
```

### Batch Processing

```go
type BatchProcessor struct {
    inputDir   string
    outputDir  string
    config     LouvainConfig
}

func (bp *BatchProcessor) ProcessAll() error {
    files, err := filepath.Glob(filepath.Join(bp.inputDir, "*.txt"))
    if err != nil {
        return err
    }
    
    for _, file := range files {
        basename := filepath.Base(file)
        name := strings.TrimSuffix(basename, filepath.Ext(basename))
        
        if err := bp.processFile(file, name); err != nil {
            log.Printf("Failed to process %s: %v", file, err)
            continue
        }
    }
    
    return nil
}

func (bp *BatchProcessor) processFile(inputFile, outputPrefix string) error {
    parser := NewGraphParser()
    parseResult, err := parser.ParseEdgeList(inputFile)
    if err != nil {
        return err
    }
    
    result, err := RunLouvain(parseResult.Graph, bp.config)
    if err != nil {
        return err
    }
    
    writer := NewFileWriter()
    return writer.WriteAll(result, parseResult.Parser, bp.outputDir, outputPrefix)
}
```

### Message Queue Integration

```go
type QueueConsumer struct {
    queue  MessageQueue
    parser *GraphParser
    writer OutputWriter
}

func (qc *QueueConsumer) Start() {
    for message := range qc.queue.Messages() {
        if err := qc.processMessage(message); err != nil {
            log.Printf("Processing failed: %v", err)
            message.Nack()
        } else {
            message.Ack()
        }
    }
}

func (qc *QueueConsumer) processMessage(msg *Message) error {
    var request struct {
        GraphData string          `json:"graph_data"`
        Config    LouvainConfig   `json:"config"`
        OutputKey string          `json:"output_key"`
    }
    
    if err := json.Unmarshal(msg.Body, &request); err != nil {
        return err
    }
    
    // Process graph
    result, err := qc.processGraphData(request.GraphData, request.Config)
    if err != nil {
        return err
    }
    
    // Store result
    return qc.storeResult(request.OutputKey, result)
}
```

---

## Error Handling

### Error Types

```go
// Input errors
var (
    ErrInvalidGraphFormat = errors.New("invalid graph format")
    ErrMissingNodes      = errors.New("edge references unknown nodes")
    ErrInvalidWeight     = errors.New("invalid edge weight")
    ErrEmptyGraph        = errors.New("graph has no nodes")
)

// Algorithm errors  
var (
    ErrInvalidState      = errors.New("invalid algorithm state")
    ErrNoImprovement     = errors.New("no modularity improvement possible")
    ErrMemoryLimit       = errors.New("memory limit exceeded")
)

// Output errors
var (
    ErrOutputDirectory   = errors.New("cannot create output directory")
    ErrFileWrite        = errors.New("cannot write output file")
    ErrInvalidMapping   = errors.New("invalid node ID mapping")
)
```

### Error Handling Patterns

```go
// Comprehensive error handling
func ProcessGraphWithRecovery(inputFile string, config LouvainConfig) (*LouvainResult, error) {
    // Parse input with validation
    parser := NewGraphParser()
    parseResult, err := parser.ParseEdgeList(inputFile)
    if err != nil {
        return nil, fmt.Errorf("input parsing failed: %w", err)
    }
    
    // Validate graph structure
    if err := parseResult.Graph.Validate(); err != nil {
        return nil, fmt.Errorf("invalid graph structure: %w", err)
    }
    
    // Run algorithm with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
    defer cancel()
    
    resultChan := make(chan *LouvainResult, 1)
    errorChan := make(chan error, 1)
    
    go func() {
        result, err := RunLouvain(parseResult.Graph, config)
        if err != nil {
            errorChan <- err
        } else {
            resultChan <- result
        }
    }()
    
    select {
    case result := <-resultChan:
        result.Parser = parseResult.Parser
        return result, nil
    case err := <-errorChan:
        return nil, fmt.Errorf("algorithm failed: %w", err)
    case <-ctx.Done():
        return nil, fmt.Errorf("algorithm timeout: %w", ctx.Err())
    }
}
```

### Validation Functions

```go
// Validate input before processing
func ValidateInput(graph *NormalizedGraph) error {
    if graph.NumNodes == 0 {
        return ErrEmptyGraph
    }
    
    if graph.TotalWeight <= 0 {
        return fmt.Errorf("invalid total weight: %f", graph.TotalWeight)
    }
    
    return graph.Validate()
}

// Validate result integrity
func ValidateResult(result *LouvainResult, originalNodeCount int) error {
    if len(result.FinalCommunities) != originalNodeCount {
        return fmt.Errorf("missing community assignments: got %d, expected %d",
            len(result.FinalCommunities), originalNodeCount)
    }
    
    if result.Modularity < -1.0 || result.Modularity > 1.0 {
        return fmt.Errorf("invalid modularity: %f", result.Modularity)
    }
    
    return nil
}
```

---

## Performance Considerations

### Memory Usage

```go
// Estimate memory requirements
func EstimateMemoryUsage(numNodes, numEdges int) int64 {
    // Graph storage
    graphMemory := int64(numNodes) * 64                    // Node arrays
    graphMemory += int64(numEdges) * 32                    // Edge storage
    
    // Algorithm state
    stateMemory := int64(numNodes) * 16                    // Community mappings
    stateMemory += int64(numNodes) * 8                     // Community stats
    
    // Working memory (2x for super-graph creation)
    workingMemory := (graphMemory + stateMemory) * 2
    
    return graphMemory + stateMemory + workingMemory
}

// Memory-efficient processing for large graphs
func ProcessLargeGraph(inputFile string, config LouvainConfig) error {
    // Check available memory
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    availableMemory := int64(m.Sys)
    
    // Estimate requirements
    nodeCount, edgeCount := estimateGraphSize(inputFile)
    requiredMemory := EstimateMemoryUsage(nodeCount, edgeCount)
    
    if requiredMemory > availableMemory*8/10 { // Use 80% of available memory
        return fmt.Errorf("insufficient memory: need %d MB, have %d MB",
            requiredMemory/1024/1024, availableMemory/1024/1024)
    }
    
    // Process with memory monitoring
    return processWithMemoryMonitoring(inputFile, config)
}
```

### Performance Tuning

```go
// Optimize configuration for graph size
func OptimizeConfig(numNodes, numEdges int) LouvainConfig {
    config := DefaultLouvainConfig()
    
    if numNodes > 100000 {
        // Large graph optimizations
        config.ChunkSize = 256
        config.MinModularity = 0.0001
        config.MaxIterations = 5
    } else if numNodes > 10000 {
        // Medium graph optimizations  
        config.ChunkSize = 128
        config.MinModularity = 0.001
        config.MaxIterations = 10
    } else {
        // Small graph - prioritize quality
        config.ChunkSize = 32
        config.MinModularity = 0.00001
        config.MaxIterations = 20
    }
    
    return config
}

// Parallel processing setup
func SetupParallelProcessing(numFiles int) *BatchProcessor {
    maxWorkers := runtime.NumCPU()
    if numFiles < maxWorkers {
        maxWorkers = numFiles
    }
    
    return &BatchProcessor{
        Workers:   maxWorkers,
        QueueSize: maxWorkers * 2,
    }
}
```

---

## Examples

### Complete Processing Pipeline

```go
package main

import (
    "fmt"
    "log"
    "path/filepath"
    "github.com/your-org/louvain"
)

func main() {
    // Setup
    inputFile := "social_network.txt"
    outputDir := "results"
    prefix := "communities"
    
    // Parse input
    parser := louvain.NewGraphParser()
    parseResult, err := parser.ParseEdgeList(inputFile)
    if err != nil {
        log.Fatalf("Parsing failed: %v", err)
    }
    
    fmt.Printf("Loaded graph: %d nodes, %.0f total weight\n",
        parseResult.Graph.NumNodes, parseResult.Graph.TotalWeight)
    
    // Configure algorithm
    config := louvain.DefaultLouvainConfig()
    config.MaxIterations = 10
    config.Verbose = true
    config.ProgressCallback = func(level, iteration int, message string) {
        fmt.Printf("[L%d I%d] %s\n", level, iteration, message)
    }
    
    // Run algorithm
    result, err := louvain.RunLouvain(parseResult.Graph, config)
    if err != nil {
        log.Fatalf("Algorithm failed: %v", err)
    }
    
    // Print results
    fmt.Printf("\nResults:\n")
    fmt.Printf("  Final modularity: %.6f\n", result.Modularity)
    fmt.Printf("  Hierarchy levels: %d\n", result.NumLevels)
    fmt.Printf("  Runtime: %d ms\n", result.Statistics.RuntimeMS)
    
    // Analyze community structure
    finalLevel := result.Levels[len(result.Levels)-1]
    fmt.Printf("  Final communities: %d\n", finalLevel.NumCommunities)
    
    sizes := make([]int, 0)
    for _, nodes := range finalLevel.Communities {
        sizes = append(sizes, len(nodes))
    }
    fmt.Printf("  Community sizes: %v\n", sizes)
    
    // Write output files
    writer := louvain.NewFileWriter()
    result.Parser = parseResult.Parser
    
    err = writer.WriteAll(result, parseResult.Parser, outputDir, prefix)
    if err != nil {
        log.Fatalf("Output failed: %v", err)
    }
    
    fmt.Printf("\nOutput files created in %s/:\n", outputDir)
    fmt.Printf("  %s.mapping\n", prefix)
    fmt.Printf("  %s.hierarchy\n", prefix)
    fmt.Printf("  %s.root\n", prefix)
    fmt.Printf("  %s.edges\n", prefix)
    
    // Write simple mapping for easy comparison
    simplePath := filepath.Join(outputDir, "simple_mapping.txt")
    err = writer.WriteOriginalMapping(result, parseResult.Parser, simplePath)
    if err != nil {
        log.Printf("Warning: Could not write simple mapping: %v", err)
    } else {
        fmt.Printf("  simple_mapping.txt\n")
    }
}
```

### Custom Output Format

```go
// Custom JSON output writer
type JSONWriter struct{}

func (jw *JSONWriter) WriteJSON(result *LouvainResult, parser *GraphParser, path string) error {
    // Convert to original node IDs
    communities := make(map[string][]string)
    for commID, nodes := range result.Levels[len(result.Levels)-1].Communities {
        communityName := fmt.Sprintf("community_%d", commID)
        originalNodes := make([]string, 0, len(nodes))
        
        for _, normalizedNode := range nodes {
            if originalID, exists := parser.GetOriginalID(normalizedNode); exists {
                originalNodes = append(originalNodes, originalID)
            }
        }
        communities[communityName] = originalNodes
    }
    
    output := struct {
        Modularity  float64             `json:"modularity"`
        NumLevels   int                 `json:"num_levels"`
        Communities map[string][]string `json:"communities"`
        Statistics  LouvainStats        `json:"statistics"`
    }{
        Modularity:  result.Modularity,
        NumLevels:   result.NumLevels,
        Communities: communities,
        Statistics:  result.Statistics,
    }
    
    file, err := os.Create(path)
    if err != nil {
        return err
    }
    defer file.Close()
    
    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")
    return encoder.Encode(output)
}
```

### Integration with External Systems

```go
// Database integration
type DatabaseStore struct {
    db *sql.DB
}

func (ds *DatabaseStore) StoreCommunities(result *LouvainResult, parser *GraphParser, experimentID string) error {
    tx, err := ds.db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    // Store experiment metadata
    _, err = tx.Exec(`
        INSERT INTO experiments (id, modularity, num_levels, runtime_ms)
        VALUES (?, ?, ?, ?)`,
        experimentID, result.Modularity, result.NumLevels, result.Statistics.RuntimeMS)
    if err != nil {
        return err
    }
    
    // Store community assignments
    stmt, err := tx.Prepare(`
        INSERT INTO community_assignments (experiment_id, node_id, community_id)
        VALUES (?, ?, ?)`)
    if err != nil {
        return err
    }
    defer stmt.Close()
    
    originalCommunities := parser.ConvertCommunityMapping(result.FinalCommunities)
    for nodeID, communityID := range originalCommunities {
        _, err = stmt.Exec(experimentID, nodeID, communityID)
        if err != nil {
            return err
        }
    }
    
    return tx.Commit()
}
```

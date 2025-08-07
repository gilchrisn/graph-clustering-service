# Louvain Algorithm API Documentation

## Overview
Louvain algorithm for community detection in networks using modularity optimization with hierarchical clustering.

## Main Function

```go
func Run(graph *Graph, config *Config, ctx context.Context) (*Result, error)
```

### Input

**Graph Structure:**
```go
type Graph struct {
    NumNodes    int           // Number of nodes
    Adjacency   [][]int       // adjacency[i] = neighbors of node i  
    Weights     [][]float64   // weights[i][j] = weight to adjacency[i][j]
    Degrees     []float64     // weighted degree of each node
    TotalWeight float64       // sum of all edge weights
}
```

**Configuration:**
```go
type Config struct {
    MaxLevels()          int     // max hierarchy levels (default: 10)
    MaxIterations()      int     // max iterations per level (default: 100)  
    MinModularityGain()  float64 // min gain threshold (default: 1e-6)
    RandomSeed()         int64   // random seed
}
```

### Output

```go
type Result struct {
    Levels           []LevelInfo     // hierarchy information per level
    FinalCommunities map[int]int     // nodeID -> communityID  
    Modularity       float64         // final modularity score
    NumLevels        int            // number of hierarchy levels
    Statistics       Statistics      // performance metrics
}

type LevelInfo struct {
    Level          int             // hierarchy level
    Communities    map[int][]int   // communityID -> [nodeIDs]
    Modularity     float64         // modularity at this level
    NumCommunities int            // number of communities
    NumMoves       int            // node moves in this level
    RuntimeMS      int64          // level runtime
}
```

## Usage Example

```go
// Create graph
graph := NewGraph(numNodes)
graph.AddEdge(0, 1, 1.0)
graph.AddEdge(1, 2, 1.0)
// ... add more edges

// Configure algorithm  
config := NewConfig()
config.Set("algorithm.max_levels", 5)
config.Set("algorithm.min_modularity_gain", 1e-5)

// Run clustering
ctx := context.Background()
result, err := louvain.Run(graph, config, ctx)
if err != nil {
    log.Fatal(err)
}

// Access results
fmt.Printf("Final modularity: %.4f\n", result.Modularity)
fmt.Printf("Number of communities: %d\n", len(result.FinalCommunities))

// Get community for specific node
community := result.FinalCommunities[nodeID]

// Get hierarchy path for node
path := result.GetHierarchyPath(nodeID)
```

## Key Methods

### Graph Construction
```go
func NewGraph(numNodes int) *Graph
func (g *Graph) AddEdge(u, v int, weight float64) error
```

### Configuration  
```go
func NewConfig() *Config
func (c *Config) Set(key string, value interface{})
func (c *Config) LoadFromFile(path string) error
```

### Result Analysis
```go
func (r *Result) GetHierarchyPath(nodeID int) []int
func (r *Result) GetCommunityHierarchy(nodeID int) []int  
func (r *Result) GetAllHierarchyPaths() map[int][]int
```

## Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `algorithm.max_levels` | int | 10 | Maximum hierarchy levels |
| `algorithm.max_iterations` | int | 100 | Max iterations per level |
| `algorithm.min_modularity_gain` | float64 | 1e-6 | Minimum gain threshold |
| `algorithm.random_seed` | int64 | timestamp | Random seed |
| `logging.level` | string | "info" | Log level |
| `logging.enable_progress` | bool | true | Enable progress logs |

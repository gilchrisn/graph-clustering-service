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
    
    // Graph storage
    StoreGraphsAtEachLevel() bool // store Graph objects (default: false)
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
    Communities    map[int][]int   // communityID -> [originalNodeIDs]
    Modularity     float64         // modularity at this level
    NumCommunities int            // number of communities
    NumMoves       int            // node moves in this level
    RuntimeMS      int64          // level runtime
    
    // Hierarchy tracking
    CommunityToSuperNode map[int]int // community -> super-node mapping
    SuperNodeToCommunity map[int]int // super-node -> community mapping
    
    // Graph storage (only if StoreGraphsAtEachLevel=true)
    Graph *Graph `json:"-"` // graph structure at this level
}
```

## Usage Example

```go
// Basic usage
graph := NewGraph(numNodes)
graph.AddEdge(0, 1, 1.0)

config := NewConfig()
config.Set("algorithm.max_levels", 5)

result, err := louvain.Run(graph, config, ctx)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Final modularity: %.4f\n", result.Modularity)
community := result.FinalCommunities[nodeID]
path := result.GetHierarchyPath(nodeID)
```

### Graph Storage
```go
// Enable graph storage
config.Set("output.store_graphs_at_each_level", true)

result, err := louvain.Run(graph, config, ctx)

// Access stored graphs
for level, levelInfo := range result.Levels {
    if levelInfo.Graph != nil {
        degree := levelInfo.Graph.Degrees[node]
        neighbors, weights := levelInfo.Graph.GetNeighbors(node)
    }
}
```

## Key Methods

```go
// Graph construction
func NewGraph(numNodes int) *Graph
func (g *Graph) AddEdge(u, v int, weight float64) error
func (g *Graph) GetNeighbors(node int) ([]int, []float64)

// Configuration  
func NewConfig() *Config
func (c *Config) Set(key string, value interface{})

// Result analysis
func (r *Result) GetHierarchyPath(nodeID int) []int
func (r *Result) GetCommunityHierarchy(nodeID int) []int  
```

## Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `algorithm.max_levels` | int | 10 | Maximum hierarchy levels |
| `algorithm.max_iterations` | int | 100 | Max iterations per level |
| `algorithm.min_modularity_gain` | float64 | 1e-6 | Minimum gain threshold |
| `algorithm.random_seed` | int64 | timestamp | Random seed |
| `output.store_graphs_at_each_level` | bool | false | Store Graph at each level |
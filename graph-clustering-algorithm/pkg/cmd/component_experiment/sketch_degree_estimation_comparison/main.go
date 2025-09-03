package main

import (
    "encoding/json"
    "fmt"
    "math"
    "math/rand"
    "os"
    "sort"
    "time"
    
    "github.com/gilchrisn/graph-clustering-service/pkg/scar" 
)

type ExperimentConfig struct {
    NumNodes   int     `json:"num_nodes"`
    EdgeProb   float64 `json:"edge_prob"`
    K          int64   `json:"k"`
    NK         int64   `json:"nk"`
    Seed       int64   `json:"seed"`
    RunID      int     `json:"run_id"`
}

type NodeResult struct {
    NodeID               int     `json:"node_id"`
    TrueDegree          float64 `json:"true_degree"`
    SketchEstimate      float64 `json:"sketch_estimate"`
    EdgeByEdgeEstimate  float64 `json:"edge_by_edge_estimate"`
    SketchIsFull        bool    `json:"sketch_is_full"`
}

type ExperimentResult struct {
    Config      ExperimentConfig `json:"config"`
    NodeResults []NodeResult     `json:"node_results"`
    GraphInfo   GraphInfo        `json:"graph_info"`
    RuntimeMS   int64           `json:"runtime_ms"`
}

type GraphInfo struct {
    NumNodes    int     `json:"num_nodes"`
    NumEdges    int     `json:"num_edges"`
    AvgDegree   float64 `json:"avg_degree"`
    Density     float64 `json:"density"`
}

type AllResults struct {
    Timestamp string             `json:"timestamp"`
    Results   []ExperimentResult `json:"results"`
}

// RawGraph represents the input graph structure
type RawGraph struct {
    NumNodes  int
    Adjacency [][]int64
}

func main() {
    configs := []ExperimentConfig{
      // Edge probability 0.2
        {NumNodes: 1000, EdgeProb: 0.2, K: 2, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.2, K: 4, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.2, K: 8, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.2, K: 16, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.2, K: 32, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.2, K: 64, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.2, K: 128, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.2, K: 256, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.2, K: 512, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.2, K: 1024, NK: 1},
        
        // Edge probability 0.4
        {NumNodes: 1000, EdgeProb: 0.4, K: 2, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.4, K: 4, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.4, K: 8, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.4, K: 16, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.4, K: 32, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.4, K: 64, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.4, K: 128, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.4, K: 256, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.4, K: 512, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.4, K: 1024, NK: 1},
        
        // Edge probability 0.6
        {NumNodes: 1000, EdgeProb: 0.6, K: 2, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.6, K: 4, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.6, K: 8, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.6, K: 16, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.6, K: 32, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.6, K: 64, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.6, K: 128, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.6, K: 256, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.6, K: 512, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.6, K: 1024, NK: 1},
        
        // Edge probability 0.8
        {NumNodes: 1000, EdgeProb: 0.8, K: 2, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.8, K: 4, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.8, K: 8, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.8, K: 16, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.8, K: 32, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.8, K: 64, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.8, K: 128, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.8, K: 256, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.8, K: 512, NK: 1},
        {NumNodes: 1000, EdgeProb: 0.8, K: 1024, NK: 1},
    }
    
    allResults := AllResults{
        Timestamp: time.Now().Format("2006-01-02 15:04:05"),
        Results:   make([]ExperimentResult, 0),
    }
    
    runsPerConfig := 3
    
    for configIdx, config := range configs {
        fmt.Printf("Config %d/%d: N=%d, p=%.1f, K=%d, NK=%d\n",
            configIdx+1, len(configs), config.NumNodes, config.EdgeProb, config.K, config.NK)
        
        for run := 0; run < runsPerConfig; run++ {
            config.Seed = int64(configIdx*1000 + run)
            config.RunID = run
            
            result, err := runExperiment(config)
            if err != nil {
                fmt.Printf("  Run %d failed: %v\n", run+1, err)
                continue
            }
            
            allResults.Results = append(allResults.Results, result)
            fmt.Printf("  Run %d: Sketch MAE=%.2f, Edge MAE=%.2f\n", 
                run+1, calculateMAE(result.NodeResults, "sketch"), 
                calculateMAE(result.NodeResults, "edge"))
        }
    }
    
    // Save results
    saveResults(allResults, "degree_estimation_results.json")
    fmt.Printf("\nCompleted %d experiments. Results saved.\n", len(allResults.Results))
    fmt.Println("Run 'python plot.py' to visualize.")
}

func runExperiment(config ExperimentConfig) (ExperimentResult, error) {
    startTime := time.Now()
    
    // Step 1: Generate pure Erdős-Rényi graph
    rng := rand.New(rand.NewSource(config.Seed))
    rawGraph := generateErdosRenyiGraph(config.NumNodes, config.EdgeProb, rng)
    
    // Step 2: Build sketch graph using the cleaner approach
    sketchGraph, err := buildSketchGraphFromRawGraph(rawGraph, config, rng)
    if err != nil {
        return ExperimentResult{}, err
    }
    
    // DEBUG: Print graph info
    actualEdges := 0
    for _, neighbors := range rawGraph.Adjacency {
        actualEdges += len(neighbors)
    }
    actualEdges /= 2 // Undirected graph
    
    // fmt.Printf("    DEBUG: Generated ER graph - nodes: %d, edges: %d\n", rawGraph.NumNodes, actualEdges)
    // fmt.Printf("    DEBUG: SketchGraph built - nodes: %d, total_weight: %.2f\n", 
        // sketchGraph.NumNodes, sketchGraph.TotalWeight)
    
    // Step 3: Calculate estimates for each node
    nodeResults := make([]NodeResult, 0)
    
    for nodeID := 0; nodeID < rawGraph.NumNodes; nodeID++ {
        // True degree
        trueDegree := float64(len(rawGraph.Adjacency[nodeID]))
        
        // Sketch-based estimate
        sketchEstimate := sketchGraph.GetDegree(nodeID)
        
        // Edge-by-edge estimate  
        _, weights := sketchGraph.GetNeighborsInclusionExclusion(nodeID, 0.1)
        edgeByEdgeEstimate := 0.0
        for _, weight := range weights {
            edgeByEdgeEstimate += weight
        }
        
        // Check if sketch is full
        sketch := sketchGraph.GetSketchManager().GetVertexSketch(int64(nodeID))
        sketchIsFull := sketch != nil && sketch.IsSketchFull()
        
        nodeResults = append(nodeResults, NodeResult{
            NodeID:              nodeID,
            TrueDegree:          trueDegree,
            SketchEstimate:      sketchEstimate,
            EdgeByEdgeEstimate:  edgeByEdgeEstimate,
            SketchIsFull:        sketchIsFull,
        })
    }
    
    return ExperimentResult{
        Config:      config,
        NodeResults: nodeResults,
        GraphInfo: GraphInfo{
            NumNodes:  rawGraph.NumNodes,
            NumEdges:  actualEdges,
            AvgDegree: float64(2 * actualEdges) / float64(rawGraph.NumNodes),
            Density:   float64(actualEdges) / float64(rawGraph.NumNodes*(rawGraph.NumNodes-1)/2),
        },
        RuntimeMS: time.Since(startTime).Milliseconds(),
    }, nil
}

// generateErdosRenyiGraph creates RawGraph for Erdos-Renyi graph (adapted from inspiration code)
func generateErdosRenyiGraph(numNodes int, edgeProbability float64, rng *rand.Rand) *RawGraph {
    adjacency := make([][]int64, numNodes)
    
    for i := 0; i < numNodes; i++ {
        for j := i + 1; j < numNodes; j++ {
            if rng.Float64() < edgeProbability {
                adjacency[i] = append(adjacency[i], int64(j))
                adjacency[j] = append(adjacency[j], int64(i))
            }
        }
    }
    
    return &RawGraph{
        NumNodes:  numNodes,
        Adjacency: adjacency,
    }
}

// buildSketchGraphFromRawGraph creates a SketchGraph with manually built sketches (adapted from inspiration code)
func buildSketchGraphFromRawGraph(rawGraph *RawGraph, config ExperimentConfig, rng *rand.Rand) (*scar.SketchGraph, error) {
    n := rawGraph.NumNodes
    finalK := config.K
    nk := config.NK
    
    // Create sketch graph
    sketchGraph := scar.NewSketchGraph(n)
    sketchManager := scar.NewSketchManager(finalK, nk)
    sketchGraph.SetSketchManager(sketchManager)
    
    // Step 1: Generate unique random hashes for each node
    nodeHashes := make([]uint32, n)
    usedHashes := make(map[uint32]bool)
    
    for i := 0; i < n; i++ {
        for {
            hash := rng.Uint32()
            if hash != math.MaxUint32 && !usedHashes[hash] {
                nodeHashes[i] = hash
                usedHashes[hash] = true
                break
            }
        }
        
        // Set hash-to-node mapping
        sketchManager.SetHashToNode(nodeHashes[i], int64(i))
    }
    
    // Step 2: Build bottom-k sketches for each node by unioning neighbor hashes
    for nodeId := 0; nodeId < n; nodeId++ {
        // Collect neighbor hashes (including self)
        neighborHashes := []uint32{nodeHashes[nodeId]}
        for _, neighbor := range rawGraph.Adjacency[nodeId] {
            neighborHashes = append(neighborHashes, nodeHashes[neighbor])
        }
        
        // Sort to get bottom-k
        sort.Slice(neighborHashes, func(i, j int) bool {
            return neighborHashes[i] < neighborHashes[j]
        })
        
        // Create bottom-k sketch
        sketch := scar.NewVertexBottomKSketch(int64(nodeId), finalK, nk)
        
        // Fill each layer with bottom-k hashes
        for layer := int64(0); layer < nk; layer++ {
            for i := int64(0); i < finalK && i < int64(len(neighborHashes)); i++ {
                sketch.SetSketchValue(layer, i, neighborHashes[i])
            }
        }
        
        sketch.UpdateFilledCount()
        sketchManager.SetVertexSketch(int64(nodeId), sketch)
    }
    
    // Step 3: Build exact adjacency list for non-full sketches
    totalWeight := 0.0
    for nodeId := 0; nodeId < n; nodeId++ {
        for _, neighbor := range rawGraph.Adjacency[nodeId] {
            sketchGraph.AddEdgeToAdjacencyList(nodeId, int(neighbor), 1.0)
        }
        totalWeight += sketchGraph.GetDegree(nodeId)
    }
    
    sketchGraph.SetTotalWeight(totalWeight / 2.0) // Undirected graph
    
    return sketchGraph, nil
}

func calculateMAE(results []NodeResult, method string) float64 {
    sum := 0.0
    count := 0
    
    for _, result := range results {
        var estimate float64
        if method == "sketch" {
            estimate = result.SketchEstimate
        } else {
            estimate = result.EdgeByEdgeEstimate
        }
        
        if !math.IsInf(estimate, 0) && !math.IsNaN(estimate) {
            sum += math.Abs(result.TrueDegree - estimate)
            count++
        }
    }
    
    if count == 0 {
        return math.Inf(1)
    }
    return sum / float64(count)
}

func saveResults(results AllResults, filename string) {
    file, _ := os.Create(filename)
    defer file.Close()
    
    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")
    encoder.Encode(results)
}
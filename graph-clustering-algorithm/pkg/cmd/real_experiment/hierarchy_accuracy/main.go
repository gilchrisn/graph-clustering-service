package main

import (
    // "context"
    "encoding/json"
    "fmt"
    "math"
    "math/rand"
    "os"
    "sort"
    "strings"
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

type LevelResult struct {
    Level                int     `json:"level"`
    NumCommunities       int     `json:"num_communities"`
    NumComparisons       int     `json:"num_comparisons"`
    MAE                  float64 `json:"mae"`
    RelativeError        float64 `json:"relative_error"`  // Changed from RMSE
    MeanAbsolutePercErr  float64 `json:"mean_absolute_percent_error"`
    SketchFullNodes      int     `json:"sketch_full_nodes"`
    TotalNodes           int     `json:"total_nodes"`
    MaxError             float64 `json:"max_error"`
    MinError             float64 `json:"min_error"`
}

type ExperimentResult struct {
    Config         ExperimentConfig `json:"config"`
    LevelResults   []LevelResult    `json:"level_results"`
    GraphInfo      GraphInfo        `json:"graph_info"`
    ClusteringInfo ClusteringInfo   `json:"clustering_info"`
    RuntimeMS      int64           `json:"runtime_ms"`
}

type GraphInfo struct {
    NumNodes    int     `json:"num_nodes"`
    NumEdges    int     `json:"num_edges"`
    AvgDegree   float64 `json:"avg_degree"`
    Density     float64 `json:"density"`
}

type ClusteringInfo struct {
    NumLevels        int     `json:"num_levels"`
    FinalModularity  float64 `json:"final_modularity"`
    FinalCommunities int     `json:"final_communities"`
    TotalMoves       int     `json:"total_moves"`
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
    // Define K values to test
    kValues := []int64{2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
    
    // Fixed parameters
    numNodes := 1000
    edgeProb := 0.3
    nk := int64(1)
    runsPerConfig := 5
    
    allResults := AllResults{
        Timestamp: time.Now().Format("2006-01-02 15:04:05"),
        Results:   make([]ExperimentResult, 0),
    }
    
    // Generate graphs once per run, then test all K values on each graph
    for run := 0; run < runsPerConfig; run++ {
        fmt.Printf("Run %d/%d: Generating graph... ", run+1, runsPerConfig)
        
        // Generate one graph per run (seed based only on run number)
        graphSeed := int64(run * 1000)
        rng := rand.New(rand.NewSource(graphSeed))
        originalGraph := generateErdosRenyiGraph(numNodes, edgeProb, rng)
        
        fmt.Printf("done (N=%d, E=%d)\n", numNodes, countEdges(originalGraph))
        
        // Test all K values on this same graph
        for kIdx, k := range kValues {
            fmt.Printf("  K=%d (%d/%d): ", k, kIdx+1, len(kValues))
            
            config := ExperimentConfig{
                NumNodes: numNodes,
                EdgeProb: edgeProb,
                K:        k,
                NK:       nk,
                Seed:     graphSeed, // Same seed for graph generation
                RunID:    run,
            }
            
            result, err := runExperimentOnGraph(originalGraph, config)
            if err != nil {
                fmt.Printf("F")
                continue
            }
            
            allResults.Results = append(allResults.Results, result)
            fmt.Printf("done\n")
        }
    }
    
    fmt.Printf("\nAll experiments completed. Writing detailed results to file...\n")
    
    // Write detailed statistics to file
    writeDetailedResultsToFile(allResults, "scar_detailed_results.txt")
    
    // Print brief summary to stdout
    printBriefSummary(allResults)
    
    // Save results
    saveResults(allResults, "scar_edge_weight_results.json")
    fmt.Printf("\nCompleted %d experiments. Results saved to scar_edge_weight_results.json\n", len(allResults.Results))
}


func runExperimentOnGraph(originalGraph *RawGraph, config ExperimentConfig) (ExperimentResult, error) {
    startTime := time.Now()
    
    // Calculate graph info (moved from runExperiment)
    numEdges := countEdges(originalGraph)
    
    graphInfo := GraphInfo{
        NumNodes:  originalGraph.NumNodes,
        NumEdges:  numEdges,
        AvgDegree: float64(2 * numEdges) / float64(originalGraph.NumNodes),
        Density:   float64(numEdges) / float64(originalGraph.NumNodes*(originalGraph.NumNodes-1)/2),
    }
    
    // Run SCAR algorithm on the existing graph
    result, err := runSCARFromRawGraph(originalGraph, config)
    if err != nil {
        return ExperimentResult{}, err
    }
    
    fmt.Printf(" Levels: %d, Communities: %d, Modularity: %.4f, Moves: %d, Time: %dms",
        result.NumLevels, len(result.FinalCommunities), result.Modularity, 
        result.Statistics.TotalMoves, time.Since(startTime).Milliseconds())

    // Analyze edge weight estimation at each level
    levelResults := analyzeEdgeWeightEstimation(originalGraph, result, config)
    
    return ExperimentResult{
        Config:         config,
        LevelResults:   levelResults,
        GraphInfo:      graphInfo,
        ClusteringInfo: ClusteringInfo{
            NumLevels:        result.NumLevels,
            FinalModularity:  result.Modularity,
            FinalCommunities: len(result.FinalCommunities),
            TotalMoves:       result.Statistics.TotalMoves,
        },
        RuntimeMS: time.Since(startTime).Milliseconds(),
    }, nil
}

// Helper function to count edges
func countEdges(graph *RawGraph) int {
    numEdges := 0
    for _, neighbors := range graph.Adjacency {
        numEdges += len(neighbors)
    }
    return numEdges / 2 // Undirected graph
}



func runExperiment(config ExperimentConfig) (ExperimentResult, error) {
    startTime := time.Now()
    
    // Step 1: Generate Erdos-Renyi graph
    rng := rand.New(rand.NewSource(config.Seed))
    originalGraph := generateErdosRenyiGraph(config.NumNodes, config.EdgeProb, rng)
    
    // Calculate graph info
    numEdges := 0
    for _, neighbors := range originalGraph.Adjacency {
        numEdges += len(neighbors)
    }
    numEdges /= 2 // Undirected graph
    
    graphInfo := GraphInfo{
        NumNodes:  originalGraph.NumNodes,
        NumEdges:  numEdges,
        AvgDegree: float64(2 * numEdges) / float64(originalGraph.NumNodes),
        Density:   float64(numEdges) / float64(originalGraph.NumNodes*(originalGraph.NumNodes-1)/2),
    }
    
    // Step 2: Run modified SCAR algorithm
    result, err := runSCARFromRawGraph(originalGraph, config)
    if err != nil {
        return ExperimentResult{}, err
    }
    fmt.Printf(" Levels: %d, Communities: %d, Modularity: %.4f, Moves: %d, Time: %dms\n",
        result.NumLevels, len(result.FinalCommunities), result.Modularity, result.Statistics.TotalMoves, time.Since(startTime).Milliseconds())

    // Step 3: Analyze edge weight estimation at each level
    levelResults := analyzeEdgeWeightEstimation(originalGraph, result, config)
    
    return ExperimentResult{
        Config:       config,
        LevelResults: levelResults,
        GraphInfo:    graphInfo,
        ClusteringInfo: ClusteringInfo{
            NumLevels:        result.NumLevels,
            FinalModularity:  result.Modularity,
            FinalCommunities: len(result.FinalCommunities),
            TotalMoves:       result.Statistics.TotalMoves,
        },
        RuntimeMS: time.Since(startTime).Milliseconds(),
    }, nil
}

// generateErdosRenyiGraph creates RawGraph for Erdos-Renyi graph
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

// Modified buildSketchGraphFromRawGraph to use a separate RNG for sketch generation
func buildSketchGraphFromRawGraph(rawGraph *RawGraph, config ExperimentConfig, rng *rand.Rand) (*scar.SketchGraph, error) {
    n := rawGraph.NumNodes
    finalK := config.K
    nk := config.NK
    
    // Create sketch graph
    sketchGraph := scar.NewSketchGraph(n)
    sketchManager := scar.NewSketchManager(finalK, nk)
    sketchGraph.SetSketchManager(sketchManager)
    
    // Step 1: Generate unique random hashes for each node
    // Use a different seed for sketch generation to ensure different hashes for different K values
    sketchSeed := config.Seed + int64(config.K*10000) // Different seed for different K values
    sketchRng := rand.New(rand.NewSource(sketchSeed))
    
    nodeHashes := make([]uint32, n)
    usedHashes := make(map[uint32]bool)
    
    for i := 0; i < n; i++ {
        for {
            hash := sketchRng.Uint32()
            if hash != math.MaxUint32 && !usedHashes[hash] {
                nodeHashes[i] = hash
                usedHashes[hash] = true
                break
            }
        }
        
        // Set hash-to-node mapping
        sketchManager.SetHashToNode(nodeHashes[i], int64(i))
    }
    
    // Rest of the function remains the same...
    // (Step 2: Build bottom-k sketches, Step 3: Build adjacency list, etc.)
    
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

        identifyingHashSketch := scar.NewVertexBottomKSketch(int64(nodeId), finalK, nk)
        for layer := int64(0); layer < nk; layer++ {
            identifyingHashSketch.SetSketchValue(layer, 0, nodeHashes[nodeId])
            // Fill remaining positions with MaxUint32
            for i := int64(1); i < finalK; i++ {
                identifyingHashSketch.SetSketchValue(layer, i, math.MaxUint32)
            }
        }
        identifyingHashSketch.UpdateFilledCount()
        sketchManager.SetNodeToHashMap(int64(nodeId), identifyingHashSketch)    
    }
    
    // Step 3: Build exact adjacency list for non-full sketches
    for nodeId := 0; nodeId < n; nodeId++ {
        for _, neighbor := range rawGraph.Adjacency[nodeId] {
            sketchGraph.AddEdgeToAdjacencyList(nodeId, int(neighbor), 1.0)
        }
    }

    sketchGraph.CalculateAndStoreDegrees()

    // Now calculate total weight
    totalWeight := 0.0
    for nodeId := 0; nodeId < n; nodeId++ {
        totalWeight += sketchGraph.GetDegree(nodeId)
    }
    sketchGraph.SetTotalWeight(totalWeight / 2.0)
    
    return sketchGraph, nil
}

// Modified Run function that takes RawGraph instead of file paths
func runSCARFromRawGraph(rawGraph *RawGraph, config ExperimentConfig) (*scar.Result, error) {
    // Create SCAR config
    scarConfig := scar.NewConfig()
    scarConfig.Set("scar.k", config.K)
    scarConfig.Set("scar.nk", config.NK)
    scarConfig.Set("algorithm.random_seed", config.Seed)
    scarConfig.Set("algorithm.max_levels", 5)
    scarConfig.Set("algorithm.max_iterations", 50)
    scarConfig.Set("algorithm.min_modularity_gain", -100.0)
    scarConfig.Set("output.store_graphs_at_each_level", true)
    
    logger := scarConfig.CreateLogger()
    
    // Build sketch graph using the approach from user's code
    rng := rand.New(rand.NewSource(config.Seed))
    graph, err := buildSketchGraphFromRawGraph(rawGraph, config, rng)
    if err != nil {
        return nil, fmt.Errorf("sketch preprocessing failed: %w", err)
    }
    
    logger.Info().
        Int("nodes", graph.NumNodes).
        Float64("total_weight", graph.TotalWeight).
        Msg("Sketch preprocessing completed, starting clustering")
    
    // Validate input graph
    if err := graph.Validate(); err != nil {
        return nil, fmt.Errorf("invalid graph: %w", err)
    }
    
    result := &scar.Result{
        Levels:     make([]scar.LevelInfo, 0),
        Statistics: scar.Statistics{LevelStats: make([]scar.LevelStats, 0)},
    }
    
    // Initialize community structure
    comm := scar.NewCommunity(graph)
    currentGraph := graph
    
    // Track mapping from current level nodes back to original nodes
    nodeToOriginal := make([][]int, graph.NumNodes)
    for i := 0; i < graph.NumNodes; i++ {
        nodeToOriginal[i] = []int{i} // Initially, each node maps to itself
    }
    
    // Main hierarchical loop (simplified version of the original Run function)
    for level := 0; level < scarConfig.MaxLevels(); level++ {
        levelStart := time.Now()
        initialMod := scar.CalculateModularity(currentGraph, comm)
        
        logger.Info().
            Int("level", level).
            Int("nodes", currentGraph.NumNodes).
            Float64("initial_modularity", initialMod).
            Msg("Starting level")
        
        // Phase 1: Local optimization
        improvement, moves, err := scar.OneLevel(currentGraph, comm, scarConfig, logger, nil)
        if err != nil {
            return nil, fmt.Errorf("local optimization failed at level %d: %w", level, err)
        }
        
        finalMod := scar.CalculateModularity(currentGraph, comm)
        levelTime := time.Since(levelStart)
        
        // Record level information
        levelInfo := scar.LevelInfo{
            Level:          level,
            Communities:    make(map[int][]int),
            Modularity:     finalMod,
            NumCommunities: 0,
            NumMoves:       moves,
            RuntimeMS:      levelTime.Milliseconds(),
            CommunityToSuperNode: make(map[int]int),
            SuperNodeToCommunity: make(map[int]int),
            SketchGraph:    currentGraph, // Store the current sketch graph
        }
        
        // Build communities map
        for c := 0; c < comm.NumCommunities; c++ {
            if len(comm.CommunityNodes[c]) > 0 {
                levelInfo.Communities[c] = make([]int, len(comm.CommunityNodes[c]))
                copy(levelInfo.Communities[c], comm.CommunityNodes[c])
            }
        }
        
        levelInfo.NumCommunities = len(levelInfo.Communities)
        
        result.Levels = append(result.Levels, levelInfo)
        result.Statistics.TotalMoves += moves
        
        // Record level statistics
        levelStats := scar.LevelStats{
            Level:             level,
            Moves:             moves,
            InitialModularity: initialMod,
            FinalModularity:   finalMod,
            RuntimeMS:         levelTime.Milliseconds(),
        }
        result.Statistics.LevelStats = append(result.Statistics.LevelStats, levelStats)
        
        // Check termination conditions
        if !improvement {
            logger.Info().Int("level", level).Msg("No improvement, stopping")
            break
        }
        
        if levelInfo.NumCommunities == 1 {
            logger.Info().Int("level", level).Msg("Single community remaining, stopping")
            break
        }
        
        // Phase 2: Create super-graph
        superGraph, communityMapping, superToComm, commToSuper, err := scar.AggregateGraph(currentGraph, comm, logger)
        if err != nil {
            return nil, fmt.Errorf("aggregation failed at level %d: %w", level, err)
        }
        
        // Populate hierarchy tracking
        currentLevelIndex := len(result.Levels) - 1
        
        for commID := range result.Levels[currentLevelIndex].Communities {
            if superNodeID, exists := commToSuper[commID]; exists {
                result.Levels[currentLevelIndex].CommunityToSuperNode[commID] = superNodeID
            }
        }
        
        for superNodeID, commID := range superToComm {
            if _, exists := result.Levels[currentLevelIndex].Communities[commID]; exists {
                result.Levels[currentLevelIndex].SuperNodeToCommunity[superNodeID] = commID
            }
        }
        
        // Check if compression occurred
        if superGraph.NumNodes >= currentGraph.NumNodes {
            logger.Info().Msg("No compression achieved, stopping")
            break
        }
        
        // Update node-to-original mapping for next level
        newNodeToOriginal := make([][]int, superGraph.NumNodes)
        for superNodeID, originalNodesList := range communityMapping {
            newNodeToOriginal[superNodeID] = make([]int, 0)
            for _, currentLevelNode := range originalNodesList {
                newNodeToOriginal[superNodeID] = append(newNodeToOriginal[superNodeID], nodeToOriginal[currentLevelNode]...)
            }
        }
        nodeToOriginal = newNodeToOriginal
        
        // Prepare for next level
        currentGraph = superGraph
        comm = scar.NewCommunity(currentGraph)
    }
    
    // Finalize results
    result.NumLevels = len(result.Levels)
    result.Modularity = scar.CalculateModularity(currentGraph, comm)
    result.Statistics.RuntimeMS = time.Since(time.Now()).Milliseconds()
    
    // Build final community assignments
    result.FinalCommunities = make(map[int]int)
    for compressedNode := 0; compressedNode < currentGraph.NumNodes; compressedNode++ {
        finalCommID := comm.NodeToCommunity[compressedNode]
        for _, originalNode := range nodeToOriginal[compressedNode] {
            result.FinalCommunities[originalNode] = finalCommID
        }
    }
    
    return result, nil
}

func analyzeEdgeWeightEstimation(originalGraph *RawGraph, result *scar.Result, config ExperimentConfig) []LevelResult {
    levelResults := make([]LevelResult, 0)
    
    // Analyze all levels except the last one (leaf level)
    for levelIdx := 0; levelIdx < len(result.Levels); levelIdx++ {
        level := result.Levels[levelIdx]

        if level.SketchGraph == nil {
            continue
        }
        
        
        levelResult := analyzeLevelEdgeWeights(originalGraph, result, levelIdx)
        levelResults = append(levelResults, levelResult)
    }
    
    return levelResults
}

func analyzeLevelEdgeWeights(originalGraph *RawGraph, result *scar.Result, levelIdx int) LevelResult {
    level := result.Levels[levelIdx]
    sketchGraph := level.SketchGraph
    
    // Get exact aggregated weights by simulating community aggregation on original graph
    exactWeights := aggregateOriginalGraph(originalGraph, result, levelIdx)
    
    var errors []float64
    var relativeErrors []float64  // Changed from percentErrors
    var percentErrors []float64   // Keep for MAPE calculation
    sketchFullCount := 0
    totalNodes := sketchGraph.NumNodes
    minError := math.Inf(1)
    maxError := 0.0
    
    // Compare all community pairs
    communities := level.Communities
    for commA := range communities {
        for commB := range communities {
            if commA >= commB {
                continue // Avoid duplicates and self-loops
            }
            
            // Get exact weight from aggregated original graph
            exactKey := fmt.Sprintf("%d-%d", commA, commB)
            exactWeight := exactWeights[exactKey]

            // Get member nodes for each community
            membersA := result.Levels[levelIdx].Communities[commA]
            membersB := result.Levels[levelIdx].Communities[commB]

            // Get sketch-based estimate using inclusion-exclusion
            sketchEstimate := estimateEdgesBetweenCommunities(sketchGraph, membersA, membersB)
            
            // Calculate absolute error
            absError := math.Abs(exactWeight - sketchEstimate)
            errors = append(errors, absError)
            
            if absError < minError {
                minError = absError
            }
            if absError > maxError {
                maxError = absError
            }
            
            // Calculate relative error and percentage error (if exact weight > 0)
            if exactWeight > 0 {
                relativeError := absError / exactWeight
                relativeErrors = append(relativeErrors, relativeError)
                
                percentError := relativeError * 100
                percentErrors = append(percentErrors, percentError)
            }
        }
    }
    
    // Count sketch full nodes
    for nodeID := 0; nodeID < totalNodes; nodeID++ {
        sketch := sketchGraph.GetSketchManager().GetVertexSketch(int64(nodeID))
        if sketch != nil && sketch.IsSketchFull() {
            sketchFullCount++
        }
    }
    
    if minError == math.Inf(1) {
        minError = 0.0
    }
    
    // Calculate metrics
    mae := calculateMean(errors)
    meanRelativeError := calculateMean(relativeErrors) 
    mape := calculateMean(percentErrors)
    
    return LevelResult{
        Level:               levelIdx,
        NumCommunities:      level.NumCommunities,
        NumComparisons:      len(errors),
        MAE:                 mae,
        RelativeError:       meanRelativeError,  
        MeanAbsolutePercErr: mape,
        SketchFullNodes:     sketchFullCount,
        TotalNodes:          totalNodes,
        MaxError:            maxError,
        MinError:            minError,
    }
}


// aggregateOriginalGraph simulates community aggregation on the original graph
func aggregateOriginalGraph(originalGraph *RawGraph, result *scar.Result, level int) map[string]float64 {
    exactWeights := make(map[string]float64)
    
    communities := result.Levels[level].Communities
    
    for commA := range communities {
        for commB := range communities {
            if commA >= commB {
                continue
            }
            
            // Get original nodes for each community
            originalNodesA := getOriginalNodesForCommunity(result, level, commA)
            originalNodesB := getOriginalNodesForCommunity(result, level, commB)
            
            // Sum exact edges between these node sets in original graph
            weight := sumEdgesBetweenSets(originalGraph, originalNodesA, originalNodesB)
            
            key := fmt.Sprintf("%d-%d", commA, commB)
            exactWeights[key] = weight
        }
    }
    
    return exactWeights
}

// getOriginalNodesForCommunity traces back to original nodes using hierarchy
func getOriginalNodesForCommunity(result *scar.Result, level int, communityID int) []int {
    // For level 0, communities contain original node IDs directly
    if level == 0 {
        return result.Levels[level].Communities[communityID]
    }
    
    // For higher levels, need to trace back through hierarchy
    originalNodes := make([]int, 0)
    
    // Get super-nodes in this community at this level
    superNodes := result.Levels[level].Communities[communityID]
    
    // For each super-node, find its corresponding community in previous level
    for _, superNode := range superNodes {
        // Find which community this super-node came from in previous level
        prevLevel := level - 1
        prevLevelInfo := result.Levels[prevLevel]
        
        // Find the community that maps to this super-node
        for prevCommID, prevSuperNodeID := range prevLevelInfo.CommunityToSuperNode {
            if prevSuperNodeID == superNode {
                // Recursively get original nodes from previous level
                prevOriginalNodes := getOriginalNodesForCommunity(result, prevLevel, prevCommID)
                originalNodes = append(originalNodes, prevOriginalNodes...)
                break
            }
        }
    }
    
    return originalNodes
}

// sumEdgesBetweenSets counts edges between two sets of nodes in original graph
func sumEdgesBetweenSets(originalGraph *RawGraph, setA, setB []int) float64 {
    // Create lookup map for setB for efficiency
    setBLookup := make(map[int]bool)
    for _, nodeB := range setB {
        setBLookup[nodeB] = true
    }
    
    edgeSum := 0.0
    
    // For each node in setA, count edges to nodes in setB
    for _, nodeA := range setA {
        if nodeA >= len(originalGraph.Adjacency) {
            continue
        }
        
        for _, neighbor := range originalGraph.Adjacency[nodeA] {
            if setBLookup[int(neighbor)] {
                edgeSum += 1.0 // Each edge has weight 1.0 in our Erdos-Renyi graph
            }
        }
    }
    
    return edgeSum
}
func estimateEdgesBetweenCommunities(sketchGraph *scar.SketchGraph, membersA, membersB []int) float64 {
    // Create lookup map for community B for efficient checking
    communityBLookup := make(map[int]bool)
    for _, nodeB := range membersB {
        communityBLookup[nodeB] = true
    }
    
    totalEdges := 0.0
    
    // For each node in community A, estimate edges to community B
    for _, nodeA := range membersA {
        edges := estimateNodeEdgesToCommunitySet(sketchGraph, nodeA, membersB, communityBLookup)
        totalEdges += edges
    }
    
    return totalEdges
}

// Estimate edges from a single node to a set of community members
// Uses the same pattern as EstimateEdgesToCommunity but works with arbitrary node sets
func estimateNodeEdgesToCommunitySet(sketchGraph *scar.SketchGraph, node int, communityMembers []int, communityLookup map[int]bool) float64 {
    if node < 0 || node >= sketchGraph.NumNodes {
        return 0.0
    }
    
    nodeSketch := sketchGraph.GetSketchManager().GetVertexSketch(int64(node))
    if nodeSketch == nil {
        return 0.0
    }
    
    // Use sketch estimation if node has full sketch, otherwise exact counting
    if !nodeSketch.IsSketchFull() {
        return countExactEdgesToCommunitySet(sketchGraph, node, communityLookup)
    } else {
        return estimateEdgesToCommunitySetViaInclusion(sketchGraph, node, communityMembers, nodeSketch)
    }
}

// Exact counting fallback - count edges directly from adjacency list
func countExactEdgesToCommunitySet(sketchGraph *scar.SketchGraph, node int, communityLookup map[int]bool) float64 {
    neighbors, weights := sketchGraph.GetNeighbors(node)
    
    edgeCount := 0.0
    for i, neighbor := range neighbors {
        if communityLookup[neighbor] {
            edgeCount += weights[i]
        }
    }
    
    return edgeCount
}

// Sketch-based estimation using inclusion-exclusion
func estimateEdgesToCommunitySetViaInclusion(sketchGraph *scar.SketchGraph, node int, communityMembers []int, nodeSketch *scar.VertexBottomKSketch) float64 {
    // Create identifying sketch for the community members
    communityIdentifyingSketch := createCommunityIdentifyingSketch(sketchGraph, communityMembers)
    if communityIdentifyingSketch == nil {
        // Fallback to exact if we can't create community sketch
        communityLookup := make(map[int]bool)
        for _, member := range communityMembers {
            communityLookup[member] = true
        }
        return countExactEdgesToCommunitySet(sketchGraph, node, communityLookup)
    }
    
    // Apply inclusion-exclusion: |node_neighbors ∩ community| = |node_neighbors| + |community| - |union|
    nodeDegree := nodeSketch.EstimateCardinality()
    communitySize := float64(len(communityMembers))
    
    unionSketch := nodeSketch.UnionWith(communityIdentifyingSketch)
    if unionSketch == nil {
        // Fallback to exact if union fails
        communityLookup := make(map[int]bool)
        for _, member := range communityMembers {
            communityLookup[member] = true
        }
        return countExactEdgesToCommunitySet(sketchGraph, node, communityLookup)
    }
    
    unionSize := unionSketch.EstimateCardinality()
    intersection := nodeDegree + communitySize - unionSize
    
    return math.Max(0, intersection)
}

// Create a sketch containing the identifying hashes of community members
func createCommunityIdentifyingSketch(sketchGraph *scar.SketchGraph, communityMembers []int) *scar.VertexBottomKSketch {
    if len(communityMembers) == 0 {
        return nil
    }
    
    var result *scar.VertexBottomKSketch
    
    // Find first valid identifying sketch to initialize dimensions
    for _, nodeID := range communityMembers {
        identifyingSketch := sketchGraph.GetSketchManager().GetNodeToHashMap()[int64(nodeID)]
        if identifyingSketch != nil {
            // Create copy for the union
            result = scar.NewVertexBottomKSketch(int64(-1), identifyingSketch.GetK(), identifyingSketch.GetNk())
            for layer := int64(0); layer < identifyingSketch.GetNk(); layer++ {
                for i := int64(0); i < identifyingSketch.GetK(); i++ {
                    result.SetSketchValue(layer, i, identifyingSketch.GetSketch(layer)[i])
                }
            }
            result.UpdateFilledCount()
            break
        }
    }
    
    if result == nil {
        return nil
    }
    
    // Union with remaining identifying sketches
    for _, nodeID := range communityMembers {
        identifyingSketch := sketchGraph.GetSketchManager().GetNodeToHashMap()[int64(nodeID)]
        if identifyingSketch != nil {
            result = result.UnionWith(identifyingSketch)
            if result == nil {
                return nil
            }
        }
    }
    
    return result
}

func unionNodeSketches(sketchGraph *scar.SketchGraph, nodeIDs []int) *scar.VertexBottomKSketch {
    if len(nodeIDs) == 0 {
        return nil
    }
    
    // Get first valid sketch to initialize dimensions
    var result *scar.VertexBottomKSketch
    for _, nodeID := range nodeIDs {
        sketch := sketchGraph.GetSketchManager().GetVertexSketch(int64(nodeID))
        if sketch != nil {
            // Create copy for the union
            result = scar.NewVertexBottomKSketch(int64(-1), sketch.GetK(), sketch.GetNk())
            for layer := int64(0); layer < sketch.GetNk(); layer++ {
                for i := int64(0); i < sketch.GetK(); i++ {
                    result.SetSketchValue(layer, i, sketch.GetSketch(layer)[i])
                }
            }
            result.UpdateFilledCount()
            break
        }
    }
    
    if result == nil {
        return nil
    }
    
    // Union with remaining sketches
    for _, nodeID := range nodeIDs {
        sketch := sketchGraph.GetSketchManager().GetVertexSketch(int64(nodeID))
        if sketch != nil {
            result = result.UnionWith(sketch)
            if result == nil {
                return nil
            }
        }
    }
    
    return result
}

func unionWithIdentifyingHashes(neighborhoodSketch *scar.VertexBottomKSketch, sketchGraph *scar.SketchGraph, nodeIDs []int) *scar.VertexBottomKSketch {
    if neighborhoodSketch == nil {
        return nil
    }
    
    // Start with copy of neighborhood sketch
    result := neighborhoodSketch.UnionWith(neighborhoodSketch) // Creates a copy
    
    // Union with each node's identifying hash sketch
    for _, nodeID := range nodeIDs {
        identifyingSketch := sketchGraph.GetSketchManager().GetNodeToHashMap()[int64(nodeID)]
        if identifyingSketch != nil {
            result = result.UnionWith(identifyingSketch)
            if result == nil {
                return nil
            }
        }
    }
    
    return result
}

func calculateMean(values []float64) float64 {
    if len(values) == 0 {
        return 0.0
    }
    
    sum := 0.0
    for _, v := range values {
        sum += v
    }
    return sum / float64(len(values))
}

func calculateRMSE(errors []float64) float64 {
    if len(errors) == 0 {
        return 0.0
    }
    
    sumSquares := 0.0
    for _, err := range errors {
        sumSquares += err * err
    }
    return math.Sqrt(sumSquares / float64(len(errors)))
}

func saveResults(results AllResults, filename string) {
    file, err := os.Create(filename)
    if err != nil {
        fmt.Printf("Error creating results file: %v\n", err)
        return
    }
    defer file.Close()
    
    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")
    encoder.Encode(results)
}


func writeDetailedResultsToFile(results AllResults, filename string) {
    file, err := os.Create(filename)
    if err != nil {
        fmt.Printf("Error creating detailed results file: %v\n", err)
        return
    }
    defer file.Close()
    
    fmt.Fprintf(file, "SCAR EDGE WEIGHT ESTIMATION EXPERIMENT RESULTS\n")
    fmt.Fprintf(file, "Generated: %s\n", results.Timestamp)
    fmt.Fprintf(file, "Total Experiments: %d\n", len(results.Results))
    fmt.Fprintf(file, "%s\n\n", strings.Repeat("=", 120))
    
    // Group results by configuration
    configGroups := make(map[string][]ExperimentResult)
    
    for _, result := range results.Results {
        key := fmt.Sprintf("N%d_p%.1f_K%d_NK%d", 
            result.Config.NumNodes, result.Config.EdgeProb, 
            result.Config.K, result.Config.NK)
        configGroups[key] = append(configGroups[key], result)
    }
    
    // Write detailed statistics for each configuration
    for configKey, configResults := range configGroups {
        if len(configResults) == 0 {
            continue
        }
        
        config := configResults[0].Config
        fmt.Fprintf(file, "%s\n", strings.Repeat("=", 120))
        fmt.Fprintf(file, "CONFIG: N=%d, p=%.1f, K=%d, NK=%d (%s)\n", 
            config.NumNodes, config.EdgeProb, config.K, config.NK, configKey)
        fmt.Fprintf(file, "%s\n", strings.Repeat("=", 120))
        
        // Find maximum number of levels across all runs for this config
        maxLevels := 0
        for _, result := range configResults {
            if len(result.LevelResults) > maxLevels {
                maxLevels = len(result.LevelResults)
            }
        }
        
        // Create header - changed from RMSE to RelErr
        header := fmt.Sprintf("%-4s %-8s %-12s", "Run", "Levels", "Modularity")
        for level := 0; level < maxLevels; level++ {
            header += fmt.Sprintf(" %-12s", fmt.Sprintf("L%d_MAE", level))
        }
        for level := 0; level < maxLevels; level++ {
            header += fmt.Sprintf(" %-12s", fmt.Sprintf("L%d_RelErr", level))  // Changed from RMSE
        }
        for level := 0; level < maxLevels; level++ {
            header += fmt.Sprintf(" %-8s", fmt.Sprintf("L%d_Comp", level))
        }
        
        fmt.Fprintf(file, "%s\n", header)
        fmt.Fprintf(file, "%s\n", strings.Repeat("-", len(header)))
        
        // Print data for each run
        for i, result := range configResults {
            row := fmt.Sprintf("%-4d %-8d %-12.6f", 
                i+1, len(result.LevelResults), result.ClusteringInfo.FinalModularity)
            
            // MAE columns
            for level := 0; level < maxLevels; level++ {
                if level < len(result.LevelResults) {
                    row += fmt.Sprintf(" %-12.6f", result.LevelResults[level].MAE)
                } else {
                    row += fmt.Sprintf(" %-12s", "N/A")
                }
            }
            
            // RelativeError columns (changed from RMSE)
            for level := 0; level < maxLevels; level++ {
                if level < len(result.LevelResults) {
                    row += fmt.Sprintf(" %-12.6f", result.LevelResults[level].RelativeError)
                } else {
                    row += fmt.Sprintf(" %-12s", "N/A")
                }
            }
            
            // Comparison columns
            for level := 0; level < maxLevels; level++ {
                if level < len(result.LevelResults) {
                    row += fmt.Sprintf(" %-8d", result.LevelResults[level].NumComparisons)
                } else {
                    row += fmt.Sprintf(" %-8s", "N/A")
                }
            }
            
            fmt.Fprintf(file, "%s\n", row)
        }
        
        // Summary statistics for this config
        fmt.Fprintf(file, "\n%s\n", strings.Repeat("-", len(header)))
        
        // Calculate averages for each level
        avgRow := fmt.Sprintf("%-4s %-8s %-12s", "AVG", "", "")
        
        // Average MAE for each level
        for level := 0; level < maxLevels; level++ {
            levelMAEs := make([]float64, 0)
            for _, result := range configResults {
                if level < len(result.LevelResults) {
                    levelMAEs = append(levelMAEs, result.LevelResults[level].MAE)
                }
            }
            if len(levelMAEs) > 0 {
                avgRow += fmt.Sprintf(" %-12.6f", calculateMean(levelMAEs))
            } else {
                avgRow += fmt.Sprintf(" %-12s", "N/A")
            }
        }
        
        // Average RelativeError for each level (changed from RMSE)
        for level := 0; level < maxLevels; level++ {
            levelRelErrs := make([]float64, 0)
            for _, result := range configResults {
                if level < len(result.LevelResults) {
                    levelRelErrs = append(levelRelErrs, result.LevelResults[level].RelativeError)
                }
            }
            if len(levelRelErrs) > 0 {
                avgRow += fmt.Sprintf(" %-12.6f", calculateMean(levelRelErrs))
            } else {
                avgRow += fmt.Sprintf(" %-12s", "N/A")
            }
        }
        
        // Average comparisons for each level
        for level := 0; level < maxLevels; level++ {
            levelComps := make([]int, 0)
            for _, result := range configResults {
                if level < len(result.LevelResults) {
                    levelComps = append(levelComps, result.LevelResults[level].NumComparisons)
                }
            }
            if len(levelComps) > 0 {
                avgRow += fmt.Sprintf(" %-8.0f", calculateMean(convertIntSliceToFloat(levelComps)))
            } else {
                avgRow += fmt.Sprintf(" %-8s", "N/A")
            }
        }
        
        fmt.Fprintf(file, "%s\n", avgRow)
        
        // Standard deviation row
        stdRow := fmt.Sprintf("%-4s %-8s %-12s", "STD", "", "")
        
        // Standard deviation MAE for each level
        for level := 0; level < maxLevels; level++ {
            levelMAEs := make([]float64, 0)
            for _, result := range configResults {
                if level < len(result.LevelResults) {
                    levelMAEs = append(levelMAEs, result.LevelResults[level].MAE)
                }
            }
            if len(levelMAEs) > 0 {
                stdRow += fmt.Sprintf(" %-12.6f", calculateStdDev(levelMAEs))
            } else {
                stdRow += fmt.Sprintf(" %-12s", "N/A")
            }
        }
        
        // Standard deviation RelativeError for each level (changed from RMSE)
        for level := 0; level < maxLevels; level++ {
            levelRelErrs := make([]float64, 0)
            for _, result := range configResults {
                if level < len(result.LevelResults) {
                    levelRelErrs = append(levelRelErrs, result.LevelResults[level].RelativeError)
                }
            }
            if len(levelRelErrs) > 0 {
                stdRow += fmt.Sprintf(" %-12.6f", calculateStdDev(levelRelErrs))
            } else {
                stdRow += fmt.Sprintf(" %-12s", "N/A")
            }
        }
        
        // Standard deviation comparisons for each level
        for level := 0; level < maxLevels; level++ {
            levelComps := make([]int, 0)
            for _, result := range configResults {
                if level < len(result.LevelResults) {
                    levelComps = append(levelComps, result.LevelResults[level].NumComparisons)
                }
            }
            if len(levelComps) > 0 {
                stdRow += fmt.Sprintf(" %-8.6f", calculateStdDev(convertIntSliceToFloat(levelComps)))
            } else {
                stdRow += fmt.Sprintf(" %-8s", "N/A")
            }
        }
        
        fmt.Fprintf(file, "%s\n\n", stdRow)
    }
}

func printBriefSummary(results AllResults) {
    fmt.Printf("\nBRIEF SUMMARY:\n")
    fmt.Printf("Total Experiments: %d\n", len(results.Results))
    
    // Calculate overall MAE
    allMAEs := make([]float64, 0)
    for _, result := range results.Results {
        for _, lr := range result.LevelResults {
            allMAEs = append(allMAEs, lr.MAE)
        }
    }
    
    if len(allMAEs) > 0 {
        fmt.Printf("Overall MAE: %.6f ± %.6f\n", calculateMean(allMAEs), calculateStdDev(allMAEs))
    }
    
    fmt.Printf("\nDetailed results written to: scar_detailed_results.txt\n")
    fmt.Printf("JSON results written to: scar_edge_weight_results.json\n")
}

func printFinalSummary(results AllResults) {
    // This function is no longer used - replaced by writeDetailedResultsToFile and printBriefSummary
}

func convertIntSliceToFloat(intSlice []int) []float64 {
    floatSlice := make([]float64, len(intSlice))
    for i, v := range intSlice {
        floatSlice[i] = float64(v)
    }
    return floatSlice
}

func calculateStdDev(values []float64) float64 {
    if len(values) <= 1 {
        return 0.0
    }
    
    mean := calculateMean(values)
    sumSquaredDiffs := 0.0
    
    for _, v := range values {
        diff := v - mean
        sumSquaredDiffs += diff * diff
    }
    
    variance := sumSquaredDiffs / float64(len(values)-1)
    return math.Sqrt(variance)
}
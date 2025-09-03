package main

import (
    "encoding/json"
    "fmt"
    "math"
    "math/rand"
    "os"
    "sort"
    "strings"
    "time"
)

// ============================================================================
// SIMPLIFIED DATA STRUCTURES FOR EXPERIMENT
// ============================================================================

type Graph struct {
    NumNodes  int
    Edges     map[int]map[int]bool // adjacency as sets for fast lookup
    Communities [][]int            // predefined communities
}

type BottomKSketch struct {
    Hashes []uint32
    K      int
    Filled int
}

type Node struct {
    ID     int
    Hash   uint32
    Sketch *BottomKSketch
}

// ============================================================================
// EXPERIMENT CONFIGURATION
// ============================================================================

type ExperimentConfig struct {
    // Graph parameters
    NumNodes       int     `json:"num_nodes"`
    EdgeProb       float64 `json:"edge_prob,omitempty"`
    NumCommunities int     `json:"num_communities"`
    
    // For planted partition
    PIntra         float64 `json:"p_intra,omitempty"`
    PInter         float64 `json:"p_inter,omitempty"`
    
    // Sketch parameters
    K              int     `json:"k"`
    
    // Experiment parameters
    GraphType      string  `json:"graph_type"` // "erdos_renyi", "planted_partition"
    Repetitions    int     `json:"repetitions"`
    Seed           int64   `json:"seed"`
}

type ComparisonResult struct {
    CommunityA    []int   `json:"community_a"`
    CommunityB    []int   `json:"community_b"`
    
    TrueWeight    float64 `json:"true_weight"`
    UnionEstimate float64 `json:"union_estimate"`
    SumEstimate   float64 `json:"sum_estimate"`
    
    UnionError    float64 `json:"union_error"`
    SumError      float64 `json:"sum_error"`
    
    // Context for analysis
    CommunityASize   int     `json:"community_a_size"`
    CommunityBSize   int     `json:"community_b_size"`
    EdgeMultiplicity float64 `json:"edge_multiplicity"`
    LocalDensity     float64 `json:"local_density"`
    K                int     `json:"k"`
}

type ExperimentResult struct {
    Config      ExperimentConfig   `json:"config"`
    Comparisons []ComparisonResult `json:"comparisons"`
    Summary     Summary            `json:"summary"`
    RuntimeMS   int64             `json:"runtime_ms"`
}

type Summary struct {
    UnionMAE           float64 `json:"union_mae"`
    SumMAE             float64 `json:"sum_mae"`
    UnionMaxError      float64 `json:"union_max_error"`
    SumMaxError        float64 `json:"sum_max_error"`
    DensityCorrelation float64 `json:"density_correlation"`
    SizeCorrelation    float64 `json:"size_correlation"`
    TotalComparisons   int     `json:"total_comparisons"`
}

// ============================================================================
// GRAPH GENERATION
// ============================================================================

func generateErdosRenyi(numNodes int, edgeProb float64, rng *rand.Rand) *Graph {
    graph := &Graph{
        NumNodes: numNodes,
        Edges:    make(map[int]map[int]bool),
    }
    
    // Initialize adjacency
    for i := 0; i < numNodes; i++ {
        graph.Edges[i] = make(map[int]bool)
    }
    
    // Generate edges
    for i := 0; i < numNodes; i++ {
        for j := i + 1; j < numNodes; j++ {
            if rng.Float64() < edgeProb {
                graph.Edges[i][j] = true
                graph.Edges[j][i] = true
            }
        }
    }
    
    // Generate random communities (for testing)
    numComms := max(1, numNodes/20)
    communities := make([][]int, numComms)
    for i := 0; i < numNodes; i++ {
        commId := rng.Intn(numComms)
        communities[commId] = append(communities[commId], i)
    }
    graph.Communities = communities
    
    return graph
}

func generatePlantedPartition(numNodes, numCommunities int, pIntra, pInter float64, rng *rand.Rand) *Graph {
    graph := &Graph{
        NumNodes: numNodes,
        Edges:    make(map[int]map[int]bool),
    }
    
    // Initialize adjacency
    for i := 0; i < numNodes; i++ {
        graph.Edges[i] = make(map[int]bool)
    }
    
    // Assign nodes to communities
    nodesPerComm := numNodes / numCommunities
    communities := make([][]int, numCommunities)
    nodeToComm := make([]int, numNodes)
    
    for i := 0; i < numNodes; i++ {
        commId := i / nodesPerComm
        if commId >= numCommunities {
            commId = numCommunities - 1
        }
        communities[commId] = append(communities[commId], i)
        nodeToComm[i] = commId
    }
    
    // Generate edges based on community membership
    for i := 0; i < numNodes; i++ {
        for j := i + 1; j < numNodes; j++ {
            prob := pInter // Default inter-community
            if nodeToComm[i] == nodeToComm[j] {
                prob = pIntra // Same community
            }
            
            if rng.Float64() < prob {
                graph.Edges[i][j] = true
                graph.Edges[j][i] = true
            }
        }
    }
    
    graph.Communities = communities
    return graph
}

// ============================================================================
// SKETCH OPERATIONS
// ============================================================================

func newBottomKSketch(k int) *BottomKSketch {
    return &BottomKSketch{
        Hashes: make([]uint32, k),
        K:      k,
        Filled: 0,
    }
}

func (s *BottomKSketch) addHash(hash uint32) {
    if s.Filled < s.K {
        s.Hashes[s.Filled] = hash
        s.Filled++
        if s.Filled == s.K {
            sort.Slice(s.Hashes, func(i, j int) bool {
                return s.Hashes[i] < s.Hashes[j]
            })
        }
    } else {
        // Sketch is full, check if we should replace
        if hash < s.Hashes[s.K-1] {
            s.Hashes[s.K-1] = hash
            // Re-sort to maintain order
            for i := s.K - 1; i > 0 && s.Hashes[i] < s.Hashes[i-1]; i-- {
                s.Hashes[i], s.Hashes[i-1] = s.Hashes[i-1], s.Hashes[i]
            }
        }
    }
}

func (s *BottomKSketch) estimateCardinality() float64 {
    if s.Filled < s.K {
        return float64(s.Filled)
    }
    
    if s.Filled == 0 {
        return 0.0
    }
    
    kthValue := s.Hashes[s.K-1]
    return float64(s.K-1) * float64(math.MaxUint32) / float64(kthValue)
}

func unionSketches(s1, s2 *BottomKSketch) *BottomKSketch {
    if s1 == nil || s2 == nil {
        return nil
    }
    
    union := newBottomKSketch(s1.K)
    
    // Merge both sketches
    allHashes := make([]uint32, 0, s1.Filled+s2.Filled)
    for i := 0; i < s1.Filled; i++ {
        allHashes = append(allHashes, s1.Hashes[i])
    }
    for i := 0; i < s2.Filled; i++ {
        allHashes = append(allHashes, s2.Hashes[i])
    }
    
    // Sort and deduplicate
    sort.Slice(allHashes, func(i, j int) bool {
        return allHashes[i] < allHashes[j]
    })
    
    prev := uint32(math.MaxUint32)
    for _, hash := range allHashes {
        if hash != prev && union.Filled < union.K {
            union.Hashes[union.Filled] = hash
            union.Filled++
            prev = hash
        }
    }
    
    return union
}

// ============================================================================
// SKETCH GRAPH CONSTRUCTION
// ============================================================================

func buildSketchGraph(graph *Graph, k int, rng *rand.Rand) map[int]*Node {
    nodes := make(map[int]*Node)
    
    // Step 1: Generate unique hashes for each node
    nodeHashes := make(map[int]uint32)
    usedHashes := make(map[uint32]bool)
    
    for i := 0; i < graph.NumNodes; i++ {
        for {
            hash := rng.Uint32()
            if hash != math.MaxUint32 && !usedHashes[hash] {
                nodeHashes[i] = hash
                usedHashes[hash] = true
                break
            }
        }
    }
    
    // Step 2: Build sketches for each node
    for nodeId := 0; nodeId < graph.NumNodes; nodeId++ {
        sketch := newBottomKSketch(k)
        
        // Add self hash
        sketch.addHash(nodeHashes[nodeId])
        
        // Add neighbor hashes
        for neighbor := range graph.Edges[nodeId] {
            sketch.addHash(nodeHashes[neighbor])
        }
        
        nodes[nodeId] = &Node{
            ID:     nodeId,
            Hash:   nodeHashes[nodeId],
            Sketch: sketch,
        }
    }
    
    return nodes
}

// ============================================================================
// TWO ESTIMATION METHODS
// ============================================================================

// Method 1: Union of Member Sketches (SCAR's approach)
func estimateEdgesUnionMethod(nodes map[int]*Node, commA, commB []int) float64 {
    if len(commA) == 0 || len(commB) == 0 {
        return 0.0
    }
    
    // Union all sketches in community A
    var unionA *BottomKSketch
    for _, nodeId := range commA {
        if node := nodes[nodeId]; node != nil {
            if unionA == nil {
                unionA = newBottomKSketch(node.Sketch.K)
                for i := 0; i < node.Sketch.Filled; i++ {
                    unionA.addHash(node.Sketch.Hashes[i])
                }
            } else {
                for i := 0; i < node.Sketch.Filled; i++ {
                    unionA.addHash(node.Sketch.Hashes[i])
                }
            }
        }
    }
    
    // Create community B identifier sketch (contains identifying hashes of B members)
    identifierB := newBottomKSketch(len(commB))
    for _, nodeId := range commB {
        if node := nodes[nodeId]; node != nil {
            identifierB.addHash(node.Hash)
        }
    }
    
    if unionA == nil || identifierB.Filled == 0 {
        return 0.0
    }
    
    // Apply inclusion-exclusion: |A ∩ B| = |A| + |B| - |A ∪ B|
    cardinalityA := unionA.estimateCardinality()
    cardinalityB := float64(len(commB))
    
    unionAB := unionSketches(unionA, identifierB)
    if unionAB == nil {
        return 0.0
    }
    
    unionCardinality := unionAB.estimateCardinality()
    intersection := cardinalityA + cardinalityB - unionCardinality
    
    return math.Max(0, intersection)
}

// Method 2: Sum of Member Estimations (CORRECTED)
func estimateEdgesSumMethod(nodes map[int]*Node, commA, commB []int) float64 {
    totalEdges := 0.0
    
    // For each node in A, estimate edges to the entire community B using inclusion-exclusion
    for _, nodeAId := range commA {
        nodeA := nodes[nodeAId]
        if nodeA == nil {
            continue
        }
        
        // Estimate edges from nodeA to community B using inclusion-exclusion
        edgesToCommB := estimateNodeToCommunityEdges(nodeA, commB, nodes)
        totalEdges += edgesToCommB
    }
    
    return totalEdges
}

// Helper function: Estimate edges from a single node to a community using inclusion-exclusion
func estimateNodeToCommunityEdges(nodeA *Node, commB []int, nodes map[int]*Node) float64 {
    if nodeA == nil || len(commB) == 0 {
        return 0.0
    }
    
    // Create identifier sketch for community B (contains identifying hashes of B members)
    identifierB := newBottomKSketch(len(commB))
    for _, nodeBId := range commB {
        if nodeB := nodes[nodeBId]; nodeB != nil {
            identifierB.addHash(nodeB.Hash)
        }
    }
    
    if identifierB.Filled == 0 {
        return 0.0
    }
    
    // Apply inclusion-exclusion: |A ∩ B| = |A| + |B| - |A ∪ B|
    cardinalityA := nodeA.Sketch.estimateCardinality()
    cardinalityB := float64(len(commB))
    
    // Union nodeA's sketch with community B's identifier sketch
    unionAB := unionSketches(nodeA.Sketch, identifierB)
    if unionAB == nil {
        return 0.0
    }
    
    unionCardinality := unionAB.estimateCardinality()
    intersection := cardinalityA + cardinalityB - unionCardinality
    
    return math.Max(0, intersection)
}

// ============================================================================
// GROUND TRUTH CALCULATION
// ============================================================================

func calculateTrueWeight(graph *Graph, commA, commB []int) float64 {
    commBLookup := make(map[int]bool)
    for _, nodeB := range commB {
        commBLookup[nodeB] = true
    }
    
    weight := 0.0
    for _, nodeA := range commA {
        for neighbor := range graph.Edges[nodeA] {
            if commBLookup[neighbor] {
                weight += 1.0
            }
        }
    }
    
    return weight
}

func calculateEdgeMultiplicity(graph *Graph, commA, commB []int) float64 {
    if len(commA) == 0 || len(commB) == 0 {
        return 0.0
    }
    
    commBLookup := make(map[int]bool)
    for _, nodeB := range commB {
        commBLookup[nodeB] = true
    }
    
    uniqueConnections := make(map[int]bool)
    totalEdges := 0
    
    for _, nodeA := range commA {
        for neighbor := range graph.Edges[nodeA] {
            if commBLookup[neighbor] {
                uniqueConnections[neighbor] = true
                totalEdges++
            }
        }
    }
    
    if len(uniqueConnections) == 0 {
        return 0.0
    }
    
    return float64(totalEdges) / float64(len(uniqueConnections))
}

func calculateLocalDensity(graph *Graph, commA, commB []int) float64 {
    totalPossible := len(commA) * len(commB)
    if totalPossible == 0 {
        return 0.0
    }
    
    actualEdges := calculateTrueWeight(graph, commA, commB)
    return actualEdges / float64(totalPossible)
}

// ============================================================================
// EXPERIMENT EXECUTION
// ============================================================================

func main() {
    experiments := createExperiments()
    allResults := make([]ExperimentResult, 0)
    
    for i, config := range experiments {
        fmt.Printf("Experiment %d/%d: %s (N=%d, K=%d)\n", 
            i+1, len(experiments), config.GraphType, config.NumNodes, config.K)
        
        result, err := runExperiment(config)
        if err != nil {
            fmt.Printf("  FAILED: %v\n", err)
            continue
        }
        
        allResults = append(allResults, result)
        fmt.Printf("  Union MAE: %.4f, Sum MAE: %.4f, Comparisons: %d\n",
            result.Summary.UnionMAE, result.Summary.SumMAE, result.Summary.TotalComparisons)
    }
    
    // Save results and generate report
    saveResults(allResults, "controlled_experiment_results.json")
    generateReport(allResults, "controlled_experiment_report.txt")
    
    fmt.Printf("\nCompleted %d experiments. Results saved.\n", len(allResults))
}

func createExperiments() []ExperimentConfig {
    experiments := make([]ExperimentConfig, 0)
    
    // Density series - fixed size, varying density
    densities := []float64{0.05, 0.1, 0.2, 0.4, 0.6, 0.8}
    kValues := []int{8, 16, 32, 64, 128}
    
    for _, k := range kValues {
        for _, density := range densities {
            experiments = append(experiments, ExperimentConfig{
                NumNodes:       300,
                EdgeProb:       density,
                NumCommunities: 10,
                K:              k,
                GraphType:      "erdos_renyi",
                Repetitions:    5,
                Seed:           int64(k*1000) + int64(density*100),
            })
        }
    }
    
    // Size series - fixed density, varying size
    sizes := []int{100, 200, 400, 800, 1600, 3200}
    
    for _, k := range kValues {
        for _, size := range sizes {
            experiments = append(experiments, ExperimentConfig{
                NumNodes:       size,
                EdgeProb:       0.1, // Fixed density
                NumCommunities: max(1, size/20),
                K:              k,
                GraphType:      "erdos_renyi",
                Repetitions:    5,
                Seed:           int64(k*10000) + int64(size),
            })
        }
    }
    
    // Planted partition series - controlled community structure
    for _, k := range kValues {
        experiments = append(experiments, ExperimentConfig{
            NumNodes:       500,
            NumCommunities: 10,
            PIntra:         0.7, // High internal density
            PInter:         0.05, // Low external density
            K:              k,
            GraphType:      "planted_partition",
            Repetitions:    3,
            Seed:           int64(k*100000),
        })
    }
    
    return experiments
}

func runExperiment(config ExperimentConfig) (ExperimentResult, error) {
    startTime := time.Now()
    allComparisons := make([]ComparisonResult, 0)
    
    for rep := 0; rep < config.Repetitions; rep++ {
        rng := rand.New(rand.NewSource(config.Seed + int64(rep)))
        
        // Generate graph
        var graph *Graph
        switch config.GraphType {
        case "erdos_renyi":
            graph = generateErdosRenyi(config.NumNodes, config.EdgeProb, rng)
        case "planted_partition":
            graph = generatePlantedPartition(config.NumNodes, config.NumCommunities, 
                config.PIntra, config.PInter, rng)
        default:
            return ExperimentResult{}, fmt.Errorf("unknown graph type: %s", config.GraphType)
        }
        
        // Build sketch graph
        nodes := buildSketchGraph(graph, config.K, rng)
        
        // Test all community pairs
        for i, commA := range graph.Communities {
            for j, commB := range graph.Communities {
                if i >= j || len(commA) == 0 || len(commB) == 0 {
                    continue
                }
                
                // Calculate ground truth
                trueWeight := calculateTrueWeight(graph, commA, commB)
                
                // Method 1: Union estimation
                unionEstimate := estimateEdgesUnionMethod(nodes, commA, commB)
                
                // Method 2: Sum estimation
                sumEstimate := estimateEdgesSumMethod(nodes, commA, commB)
                
                // Calculate metrics
                multiplicity := calculateEdgeMultiplicity(graph, commA, commB)
                localDensity := calculateLocalDensity(graph, commA, commB)
                
                comparison := ComparisonResult{
                    CommunityA:       commA,
                    CommunityB:       commB,
                    TrueWeight:       trueWeight,
                    UnionEstimate:    unionEstimate,
                    SumEstimate:      sumEstimate,
                    UnionError:       math.Abs(unionEstimate - trueWeight),
                    SumError:         math.Abs(sumEstimate - trueWeight),
                    CommunityASize:   len(commA),
                    CommunityBSize:   len(commB),
                    EdgeMultiplicity: multiplicity,
                    LocalDensity:     localDensity,
                    K:                config.K,
                }
                
                allComparisons = append(allComparisons, comparison)
            }
        }
    }
    
    // Calculate summary statistics
    summary := calculateSummary(allComparisons)
    
    return ExperimentResult{
        Config:      config,
        Comparisons: allComparisons,
        Summary:     summary,
        RuntimeMS:   time.Since(startTime).Milliseconds(),
    }, nil
}

// ============================================================================
// ANALYSIS FUNCTIONS
// ============================================================================

func calculateSummary(comparisons []ComparisonResult) Summary {
    if len(comparisons) == 0 {
        return Summary{}
    }
    
    unionErrors := make([]float64, len(comparisons))
    sumErrors := make([]float64, len(comparisons))
    densities := make([]float64, len(comparisons))
    sizes := make([]float64, len(comparisons))
    
    unionMaxError := 0.0
    sumMaxError := 0.0
    
    for i, comp := range comparisons {
        unionErrors[i] = comp.UnionError
        sumErrors[i] = comp.SumError
        densities[i] = comp.LocalDensity
        sizes[i] = float64(comp.CommunityASize + comp.CommunityBSize)
        
        if comp.UnionError > unionMaxError {
            unionMaxError = comp.UnionError
        }
        if comp.SumError > sumMaxError {
            sumMaxError = comp.SumError
        }
    }
    
    return Summary{
        UnionMAE:           calculateMean(unionErrors),
        SumMAE:             calculateMean(sumErrors),
        UnionMaxError:      unionMaxError,
        SumMaxError:        sumMaxError,
        DensityCorrelation: calculateCorrelation(densities, unionErrors),
        SizeCorrelation:    calculateCorrelation(sizes, sumErrors),
        TotalComparisons:   len(comparisons),
    }
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

func calculateCorrelation(x, y []float64) float64 {
    if len(x) != len(y) || len(x) == 0 {
        return 0.0
    }
    
    meanX := calculateMean(x)
    meanY := calculateMean(y)
    
    numerator := 0.0
    sumXSq := 0.0
    sumYSq := 0.0
    
    for i := 0; i < len(x); i++ {
        dx := x[i] - meanX
        dy := y[i] - meanY
        numerator += dx * dy
        sumXSq += dx * dx
        sumYSq += dy * dy
    }
    
    denominator := math.Sqrt(sumXSq * sumYSq)
    if denominator == 0 {
        return 0.0
    }
    
    return numerator / denominator
}

// ============================================================================
// RESULTS SAVING AND REPORTING
// ============================================================================

func saveResults(results []ExperimentResult, filename string) {
    file, err := os.Create(filename)
    if err != nil {
        fmt.Printf("Error saving results: %v\n", err)
        return
    }
    defer file.Close()
    
    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")
    encoder.Encode(results)
}

func generateReport(results []ExperimentResult, filename string) {
    file, err := os.Create(filename)
    if err != nil {
        fmt.Printf("Error creating report: %v\n", err)
        return
    }
    defer file.Close()
    
    fmt.Fprintf(file, "CONTROLLED SCAR ESTIMATION METHOD COMPARISON\n")
    fmt.Fprintf(file, "Generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))
    fmt.Fprintf(file, "%s\n\n", strings.Repeat("=", 80))
    
    // Group results by type
    densityResults := make([]ExperimentResult, 0)
    sizeResults := make([]ExperimentResult, 0)
    plantedResults := make([]ExperimentResult, 0)
    
    for _, result := range results {
        switch {
        case result.Config.GraphType == "erdos_renyi" && result.Config.EdgeProb > 0:
            if result.Config.NumNodes == 300 { // Fixed size series
                densityResults = append(densityResults, result)
            } else { // Variable size series
                sizeResults = append(sizeResults, result)
            }
        case result.Config.GraphType == "planted_partition":
            plantedResults = append(plantedResults, result)
        }
    }
    
    // Test 1: Density scaling hypothesis
    fmt.Fprintf(file, "TEST 1: UNION METHOD ERROR vs DENSITY\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    fmt.Fprintf(file, "Hypothesis: Union method error should correlate with graph density\n\n")
    
    if len(densityResults) > 0 {
        analyzeDensityScaling(file, densityResults)
    } else {
        fmt.Fprintf(file, "No density results available\n")
    }
    
    // Test 2: Size scaling hypothesis  
    fmt.Fprintf(file, "\nTEST 2: SUM METHOD ERROR vs COMMUNITY SIZE\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    fmt.Fprintf(file, "Hypothesis: Sum method error should correlate with community size\n\n")
    
    if len(sizeResults) > 0 {
        analyzeSizeScaling(file, sizeResults)
    } else {
        fmt.Fprintf(file, "No size results available\n")
    }
    
    // Test 3: Controlled community structure
    fmt.Fprintf(file, "\nTEST 3: PLANTED PARTITION ANALYSIS\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    fmt.Fprintf(file, "Analysis: High intra-community density vs low inter-community density\n\n")
    
    if len(plantedResults) > 0 {
        analyzePlantedPartition(file, plantedResults)
    } else {
        fmt.Fprintf(file, "No planted partition results available\n")
    }
    
    // Overall conclusions
    fmt.Fprintf(file, "\nOVERALL CONCLUSIONS\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    generateConclusions(file, results)
}

func analyzeDensityScaling(file *os.File, results []ExperimentResult) {
    fmt.Fprintf(file, "%-8s %-4s %-12s %-12s %-12s\n", 
        "Density", "K", "Union_MAE", "Sum_MAE", "Correlation")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 55))
    
    // Group by K value for cleaner presentation
    kGroups := make(map[int][]ExperimentResult)
    for _, result := range results {
        kGroups[result.Config.K] = append(kGroups[result.Config.K], result)
    }
    
    for k, kResults := range kGroups {
        // Sort by density
        sort.Slice(kResults, func(i, j int) bool {
            return kResults[i].Config.EdgeProb < kResults[j].Config.EdgeProb
        })
        
        for _, result := range kResults {
            fmt.Fprintf(file, "%-8.3f %-4d %-12.6f %-12.6f %-12.4f\n",
                result.Config.EdgeProb, k, result.Summary.UnionMAE, 
                result.Summary.SumMAE, result.Summary.DensityCorrelation)
        }
        fmt.Fprintf(file, "%s\n", strings.Repeat("-", 55))
    }
}

func analyzeSizeScaling(file *os.File, results []ExperimentResult) {
    fmt.Fprintf(file, "%-8s %-4s %-12s %-12s %-12s\n", 
        "Size", "K", "Union_MAE", "Sum_MAE", "Size_Corr")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 55))
    
    // Group by K value
    kGroups := make(map[int][]ExperimentResult)
    for _, result := range results {
        kGroups[result.Config.K] = append(kGroups[result.Config.K], result)
    }
    
    for k, kResults := range kGroups {
        // Sort by size
        sort.Slice(kResults, func(i, j int) bool {
            return kResults[i].Config.NumNodes < kResults[j].Config.NumNodes
        })
        
        for _, result := range kResults {
            fmt.Fprintf(file, "%-8d %-4d %-12.6f %-12.6f %-12.4f\n",
                result.Config.NumNodes, k, result.Summary.UnionMAE, 
                result.Summary.SumMAE, result.Summary.SizeCorrelation)
        }
        fmt.Fprintf(file, "%s\n", strings.Repeat("-", 55))
    }
}

func analyzePlantedPartition(file *os.File, results []ExperimentResult) {
    fmt.Fprintf(file, "%-4s %-12s %-12s %-12s %-12s\n", 
        "K", "Union_MAE", "Sum_MAE", "Density_Corr", "Multiplicity")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 60))
    
    for _, result := range results {
        // Calculate average multiplicity for this experiment
        avgMultiplicity := 0.0
        if len(result.Comparisons) > 0 {
            totalMult := 0.0
            for _, comp := range result.Comparisons {
                totalMult += comp.EdgeMultiplicity
            }
            avgMultiplicity = totalMult / float64(len(result.Comparisons))
        }
        
        fmt.Fprintf(file, "%-4d %-12.6f %-12.6f %-12.4f %-12.4f\n",
            result.Config.K, result.Summary.UnionMAE, result.Summary.SumMAE,
            result.Summary.DensityCorrelation, avgMultiplicity)
    }
}

func generateConclusions(file *os.File, results []ExperimentResult) {
    // Aggregate all data for cross-experiment analysis
    allUnionErrors := make([]float64, 0)
    allSumErrors := make([]float64, 0)
    allDensities := make([]float64, 0)
    allSizes := make([]float64, 0)
    allMultiplicities := make([]float64, 0)
    
    for _, result := range results {
        for _, comp := range result.Comparisons {
            allUnionErrors = append(allUnionErrors, comp.UnionError)
            allSumErrors = append(allSumErrors, comp.SumError)
            allDensities = append(allDensities, comp.LocalDensity)
            allSizes = append(allSizes, float64(comp.CommunityASize + comp.CommunityBSize))
            allMultiplicities = append(allMultiplicities, comp.EdgeMultiplicity)
        }
    }
    
    if len(allUnionErrors) == 0 {
        fmt.Fprintf(file, "No data available for analysis\n")
        return
    }
    
    // Calculate key correlations
    unionDensityCorr := calculateCorrelation(allDensities, allUnionErrors)
    unionMultiplicityCorr := calculateCorrelation(allMultiplicities, allUnionErrors)
    sumSizeCorr := calculateCorrelation(allSizes, allSumErrors)
    
    fmt.Fprintf(file, "Cross-Experiment Analysis (Total Comparisons: %d)\n\n", len(allUnionErrors))
    
    fmt.Fprintf(file, "1. DENSITY HYPOTHESIS VALIDATION:\n")
    fmt.Fprintf(file, "   Union Error vs Density Correlation: %.4f\n", unionDensityCorr)
    fmt.Fprintf(file, "   Union Error vs Multiplicity Correlation: %.4f\n", unionMultiplicityCorr)
    
    if unionDensityCorr > 0.5 || unionMultiplicityCorr > 0.5 {
        fmt.Fprintf(file, "   ✓ CONFIRMED: Union method error scales with density/multiplicity\n")
    } else if unionDensityCorr > 0.2 || unionMultiplicityCorr > 0.2 {
        fmt.Fprintf(file, "   ? PARTIAL: Weak positive correlation observed\n")
    } else {
        fmt.Fprintf(file, "   ✗ NOT CONFIRMED: No significant density correlation\n")
    }
    
    fmt.Fprintf(file, "\n2. SIZE HYPOTHESIS VALIDATION:\n")
    fmt.Fprintf(file, "   Sum Error vs Community Size Correlation: %.4f\n", sumSizeCorr)
    
    if sumSizeCorr > 0.5 {
        fmt.Fprintf(file, "   ✓ CONFIRMED: Sum method error scales with community size\n")
    } else if sumSizeCorr > 0.2 {
        fmt.Fprintf(file, "   ? PARTIAL: Weak positive correlation observed\n")
    } else {
        fmt.Fprintf(file, "   ✗ NOT CONFIRMED: No significant size correlation\n")
    }
    
    fmt.Fprintf(file, "\n3. METHOD COMPARISON:\n")
    unionOverallMAE := calculateMean(allUnionErrors)
    sumOverallMAE := calculateMean(allSumErrors)
    
    fmt.Fprintf(file, "   Union Method Overall MAE: %.6f\n", unionOverallMAE)
    fmt.Fprintf(file, "   Sum Method Overall MAE: %.6f\n", sumOverallMAE)
    
    if unionOverallMAE < sumOverallMAE {
        fmt.Fprintf(file, "   → Union method has lower overall error\n")
    } else {
        fmt.Fprintf(file, "   → Sum method has lower overall error\n")
    }
    
    // K-scaling analysis
    fmt.Fprintf(file, "\n4. SKETCH SIZE (K) SCALING:\n")
    analyzeKScaling(file, results)
}

func analyzeKScaling(file *os.File, results []ExperimentResult) {
    // Group results by graph configuration (excluding K)
    configGroups := make(map[string][]ExperimentResult)
    
    for _, result := range results {
        key := fmt.Sprintf("%s_N%d_p%.2f", result.Config.GraphType, 
            result.Config.NumNodes, result.Config.EdgeProb)
        configGroups[key] = append(configGroups[key], result)
    }
    
    for configKey, configResults := range configGroups {
        if len(configResults) < 2 {
            continue // Need multiple K values to analyze scaling
        }
        
        // Sort by K
        sort.Slice(configResults, func(i, j int) bool {
            return configResults[i].Config.K < configResults[j].Config.K
        })
        
        fmt.Fprintf(file, "   Config: %s\n", configKey)
        fmt.Fprintf(file, "   %-6s %-12s %-12s\n", "K", "Union_MAE", "Sum_MAE")
        
        for _, result := range configResults {
            fmt.Fprintf(file, "   %-6d %-12.6f %-12.6f\n",
                result.Config.K, result.Summary.UnionMAE, result.Summary.SumMAE)
        }
        
        // Check if union error stays constant (modeling error) vs sum error decreases (estimation error)
        firstUnion := configResults[0].Summary.UnionMAE
        lastUnion := configResults[len(configResults)-1].Summary.UnionMAE
        
        firstSum := configResults[0].Summary.SumMAE
        lastSum := configResults[len(configResults)-1].Summary.SumMAE
        
        unionChange := math.Abs(lastUnion - firstUnion) / firstUnion
        sumChange := math.Abs(lastSum - firstSum) / firstSum
        
        fmt.Fprintf(file, "   Union change: %.1f%%, Sum change: %.1f%%\n", 
            unionChange*100, sumChange*100)
        
        if unionChange < 0.2 && sumChange > 0.2 {
            fmt.Fprintf(file, "   ✓ Pattern matches theory: Union constant, Sum decreases\n")
        } else {
            fmt.Fprintf(file, "   ? Pattern unclear or opposite to theory\n")
        }
        fmt.Fprintf(file, "\n")
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
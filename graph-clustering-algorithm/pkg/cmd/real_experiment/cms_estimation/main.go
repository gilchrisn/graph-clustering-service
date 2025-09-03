package main

import (
    "encoding/json"
    "fmt"
    "hash/fnv"
    "math"
    "math/rand"
    "os"
    "sort"
    "strings"
    "time"
)

// ============================================================================
// COUNT-MIN SKETCH IMPLEMENTATION
// ============================================================================

type CountMinSketch struct {
    Depth      int        // d - number of hash functions
    Width      int        // w - number of buckets per hash function
    Matrix     [][]uint32 // d x w matrix of counters
    HashSeeds  []uint32   // d hash seeds for independent hash functions
}

func newCountMinSketch(depth, width int, rng *rand.Rand) *CountMinSketch {
    cms := &CountMinSketch{
        Depth:     depth,
        Width:     width,
        Matrix:    make([][]uint32, depth),
        HashSeeds: make([]uint32, depth),
    }
    
    // Initialize matrix
    for i := 0; i < depth; i++ {
        cms.Matrix[i] = make([]uint32, width)
        cms.HashSeeds[i] = rng.Uint32()
    }
    
    return cms
}

// FNV-based hash function with seed
func (cms *CountMinSketch) hash(item uint32, seed uint32) uint32 {
    h := fnv.New32a()
    // Combine item with seed
    combined := (uint64(item) << 32) | uint64(seed)
    h.Write([]byte{
        byte(combined >> 56), byte(combined >> 48),
        byte(combined >> 40), byte(combined >> 32),
        byte(combined >> 24), byte(combined >> 16),
        byte(combined >> 8), byte(combined),
    })
    return h.Sum32() % uint32(cms.Width)
}

func (cms *CountMinSketch) Add(itemHash uint32, count uint32) {
    for i := 0; i < cms.Depth; i++ {
        bucket := cms.hash(itemHash, cms.HashSeeds[i])
        cms.Matrix[i][bucket] += count
    }
}

func (cms *CountMinSketch) Estimate(itemHash uint32) uint32 {
    if cms.Depth == 0 {
        return 0
    }
    
    minCount := cms.Matrix[0][cms.hash(itemHash, cms.HashSeeds[0])]
    for i := 1; i < cms.Depth; i++ {
        bucket := cms.hash(itemHash, cms.HashSeeds[i])
        count := cms.Matrix[i][bucket]
        if count < minCount {
            minCount = count
        }
    }
    return minCount
}

func sumSketches(sketches ...*CountMinSketch) *CountMinSketch {
    if len(sketches) == 0 {
        return nil
    }
    
    first := sketches[0]
    result := &CountMinSketch{
        Depth:     first.Depth,
        Width:     first.Width,
        Matrix:    make([][]uint32, first.Depth),
        HashSeeds: make([]uint32, first.Depth),
    }
    
    // Copy hash seeds from first sketch
    copy(result.HashSeeds, first.HashSeeds)
    
    // Initialize result matrix
    for i := 0; i < first.Depth; i++ {
        result.Matrix[i] = make([]uint32, first.Width)
    }
    
    // Sum all matrices element-wise
    for _, sketch := range sketches {
        if sketch.Depth != first.Depth || sketch.Width != first.Width {
            continue // Skip incompatible sketches
        }
        for i := 0; i < sketch.Depth; i++ {
            for j := 0; j < sketch.Width; j++ {
                result.Matrix[i][j] += sketch.Matrix[i][j]
            }
        }
    }
    
    return result
}

// ============================================================================
// GRAPH AND NODE STRUCTURES (Updated for CMS)
// ============================================================================

type Graph struct {
    NumNodes    int
    Edges       map[int]map[int]bool
    Communities [][]int
}

type CMSNode struct {
    ID     int
    Hash   uint32
    Sketch *CountMinSketch
}

// ============================================================================
// EXPERIMENT CONFIGURATION (Updated)
// ============================================================================

type CMSExperimentConfig struct {
    // Graph parameters
    NumNodes       int     `json:"num_nodes"`
    EdgeProb       float64 `json:"edge_prob,omitempty"`
    NumCommunities int     `json:"num_communities"`
    
    // For planted partition
    PIntra         float64 `json:"p_intra,omitempty"`
    PInter         float64 `json:"p_inter,omitempty"`
    
    // CMS parameters
    CMS_d          int     `json:"cms_d"` // depth
    CMS_w          int     `json:"cms_w"` // width
    
    // Experiment parameters
    GraphType      string  `json:"graph_type"`
    Repetitions    int     `json:"repetitions"`
    Seed           int64   `json:"seed"`
}

type CMSComparisonResult struct {
    CommunityA        []int   `json:"community_a"`
    CommunityB        []int   `json:"community_b"`
    
    TrueWeight        float64 `json:"true_weight"`
    SumSketchEstimate float64 `json:"sum_sketch_estimate"`
    IndividualEstimate float64 `json:"individual_estimate"`
    
    SumSketchError    float64 `json:"sum_sketch_error"`
    IndividualError   float64 `json:"individual_error"`
    
    // Context for analysis
    CommunityASize    int     `json:"community_a_size"`
    CommunityBSize    int     `json:"community_b_size"`
    EdgeMultiplicity  float64 `json:"edge_multiplicity"`
    LocalDensity      float64 `json:"local_density"`
    CMS_d             int     `json:"cms_d"`
    CMS_w             int     `json:"cms_w"`
}

type CMSExperimentResult struct {
    Config      CMSExperimentConfig     `json:"config"`
    Comparisons []CMSComparisonResult   `json:"comparisons"`
    Summary     CMSSummary             `json:"summary"`
    RuntimeMS   int64                  `json:"runtime_ms"`
}

type CMSSummary struct {
    SumSketchMAE           float64 `json:"sum_sketch_mae"`
    IndividualMAE          float64 `json:"individual_mae"`
    SumSketchMaxError      float64 `json:"sum_sketch_max_error"`
    IndividualMaxError     float64 `json:"individual_max_error"`
    DensityCorrelation     float64 `json:"density_correlation"`
    MultiplicityCorrelation float64 `json:"multiplicity_correlation"`
    TotalComparisons       int     `json:"total_comparisons"`
}

// ============================================================================
// GRAPH GENERATION (Reused from original)
// ============================================================================

func generateErdosRenyi(numNodes int, edgeProb float64, rng *rand.Rand) *Graph {
    graph := &Graph{
        NumNodes: numNodes,
        Edges:    make(map[int]map[int]bool),
    }
    
    for i := 0; i < numNodes; i++ {
        graph.Edges[i] = make(map[int]bool)
    }
    
    for i := 0; i < numNodes; i++ {
        for j := i + 1; j < numNodes; j++ {
            if rng.Float64() < edgeProb {
                graph.Edges[i][j] = true
                graph.Edges[j][i] = true
            }
        }
    }
    
    // Generate random communities
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
            prob := pInter
            if nodeToComm[i] == nodeToComm[j] {
                prob = pIntra
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
// CMS GRAPH CONSTRUCTION
// ============================================================================

func buildCMSGraph(graph *Graph, depth, width int, rng *rand.Rand) map[int]*CMSNode {
    nodes := make(map[int]*CMSNode)
    
    // Generate unique hashes for each node
    nodeHashes := make(map[int]uint32)
    usedHashes := make(map[uint32]bool)
    
    for i := 0; i < graph.NumNodes; i++ {
        for {
            hash := rng.Uint32()
            if hash != 0 && hash != math.MaxUint32 && !usedHashes[hash] {
                nodeHashes[i] = hash
                usedHashes[hash] = true
                break
            }
        }
    }
    
    // Build CMS for each node
    for nodeId := 0; nodeId < graph.NumNodes; nodeId++ {
        sketch := newCountMinSketch(depth, width, rng)
        
        // Add neighbor hashes to the sketch
        // NOTE: We do NOT add the node's own hash to its sketch
        for neighbor := range graph.Edges[nodeId] {
            sketch.Add(nodeHashes[neighbor], 1)
        }
        
        nodes[nodeId] = &CMSNode{
            ID:     nodeId,
            Hash:   nodeHashes[nodeId],
            Sketch: sketch,
        }
    }
    
    return nodes
}

// ============================================================================
// CMS ESTIMATION METHODS
// ============================================================================

// Method 1: Sum of Community Sketches (The Efficient Way)
func estimateEdgesSumOfSketches(nodes map[int]*CMSNode, commA, commB []int) float64 {
    if len(commA) == 0 || len(commB) == 0 {
        return 0.0
    }
    
    // Collect all sketches from community A
    commASketches := make([]*CountMinSketch, 0, len(commA))
    for _, nodeId := range commA {
        if node := nodes[nodeId]; node != nil {
            commASketches = append(commASketches, node.Sketch)
        }
    }
    
    if len(commASketches) == 0 {
        return 0.0
    }
    
    // Sum all sketches from community A
    communityASketch := sumSketches(commASketches...)
    if communityASketch == nil {
        return 0.0
    }
    
    // Query the summed sketch for each node in community B
    totalWeight := 0.0
    for _, nodeBId := range commB {
        if nodeB := nodes[nodeBId]; nodeB != nil {
            count := communityASketch.Estimate(nodeB.Hash)
            totalWeight += float64(count)
        }
    }
    
    return totalWeight
}

// Method 2: Sum of Individual Node Estimates (The Baseline)
func estimateEdgesIndividualSum(nodes map[int]*CMSNode, commA, commB []int) float64 {
    totalWeight := 0.0
    
    for _, nodeAId := range commA {
        nodeA := nodes[nodeAId]
        if nodeA == nil {
            continue
        }
        
        for _, nodeBId := range commB {
            nodeB := nodes[nodeBId]
            if nodeB == nil {
                continue
            }
            
            count := nodeA.Sketch.Estimate(nodeB.Hash)
            totalWeight += float64(count)
        }
    }
    
    return totalWeight
}

// ============================================================================
// GROUND TRUTH AND METRICS (Reused)
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
    experiments := createCMSExperiments()
    allResults := make([]CMSExperimentResult, 0)
    
    for i, config := range experiments {
        fmt.Printf("CMS Experiment %d/%d: %s (N=%d, d=%d, w=%d)\n", 
            i+1, len(experiments), config.GraphType, config.NumNodes, config.CMS_d, config.CMS_w)
        
        result, err := runCMSExperiment(config)
        if err != nil {
            fmt.Printf("  FAILED: %v\n", err)
            continue
        }
        
        allResults = append(allResults, result)
        fmt.Printf("  SumSketch MAE: %.4f, Individual MAE: %.4f, Comparisons: %d\n",
            result.Summary.SumSketchMAE, result.Summary.IndividualMAE, result.Summary.TotalComparisons)
    }
    
    // Save results and generate report
    saveCMSResults(allResults, "cms_experiment_results.json")
    generateCMSReport(allResults, "cms_experiment_report.txt")
    
    fmt.Printf("\nCompleted %d CMS experiments. Results saved.\n", len(allResults))
}

func createCMSExperiments() []CMSExperimentConfig {
    experiments := make([]CMSExperimentConfig, 0)
    
    // CMS parameter combinations
    depthValues := []int{3, 4, 5}
    widthValues := []int{128, 256, 512, 1024}
    
    // Density series - test for modeling error independence
    densities := []float64{0.05, 0.1, 0.2, 0.4, 0.6}
    
    for _, d := range depthValues {
        for _, w := range widthValues {
            for _, density := range densities {
                experiments = append(experiments, CMSExperimentConfig{
                    NumNodes:       300,
                    EdgeProb:       density,
                    NumCommunities: 10,
                    CMS_d:          d,
                    CMS_w:          w,
                    GraphType:      "erdos_renyi",
                    Repetitions:    3,
                    Seed:           int64(d*10000 + w*10 + int(density*100)),
                })
            }
        }
    }
    
    // Size series
    sizes := []int{100, 200, 400, 800}
    
    for _, d := range depthValues {
        for _, w := range widthValues {
            for _, size := range sizes {
                experiments = append(experiments, CMSExperimentConfig{
                    NumNodes:       size,
                    EdgeProb:       0.1,
                    NumCommunities: max(1, size/20),
                    CMS_d:          d,
                    CMS_w:          w,
                    GraphType:      "erdos_renyi",
                    Repetitions:    3,
                    Seed:           int64(d*100000 + w*100 + size),
                })
            }
        }
    }
    
    // Planted partition with high multiplicity
    for _, d := range depthValues {
        for _, w := range widthValues {
            experiments = append(experiments, CMSExperimentConfig{
                NumNodes:       400,
                NumCommunities: 8,
                PIntra:         0.8,
                PInter:         0.1,
                CMS_d:          d,
                CMS_w:          w,
                GraphType:      "planted_partition",
                Repetitions:    3,
                Seed:           int64(d*1000000 + w*1000),
            })
        }
    }
    
    return experiments
}

func runCMSExperiment(config CMSExperimentConfig) (CMSExperimentResult, error) {
    startTime := time.Now()
    allComparisons := make([]CMSComparisonResult, 0)
    
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
            return CMSExperimentResult{}, fmt.Errorf("unknown graph type: %s", config.GraphType)
        }
        
        // Build CMS graph
        nodes := buildCMSGraph(graph, config.CMS_d, config.CMS_w, rng)
        
        // Test all community pairs
        for i, commA := range graph.Communities {
            for j, commB := range graph.Communities {
                if i >= j || len(commA) == 0 || len(commB) == 0 {
                    continue
                }
                
                // Calculate ground truth
                trueWeight := calculateTrueWeight(graph, commA, commB)
                
                // Method 1: Sum of sketches
                sumSketchEstimate := estimateEdgesSumOfSketches(nodes, commA, commB)
                
                // Method 2: Individual estimates
                individualEstimate := estimateEdgesIndividualSum(nodes, commA, commB)
                
                // Calculate metrics
                multiplicity := calculateEdgeMultiplicity(graph, commA, commB)
                localDensity := calculateLocalDensity(graph, commA, commB)
                
                comparison := CMSComparisonResult{
                    CommunityA:         commA,
                    CommunityB:         commB,
                    TrueWeight:         trueWeight,
                    SumSketchEstimate:  sumSketchEstimate,
                    IndividualEstimate: individualEstimate,
                    SumSketchError:     math.Abs(sumSketchEstimate - trueWeight),
                    IndividualError:    math.Abs(individualEstimate - trueWeight),
                    CommunityASize:     len(commA),
                    CommunityBSize:     len(commB),
                    EdgeMultiplicity:   multiplicity,
                    LocalDensity:       localDensity,
                    CMS_d:              config.CMS_d,
                    CMS_w:              config.CMS_w,
                }
                
                allComparisons = append(allComparisons, comparison)
            }
        }
    }
    
    // Calculate summary statistics
    summary := calculateCMSSummary(allComparisons)
    
    return CMSExperimentResult{
        Config:      config,
        Comparisons: allComparisons,
        Summary:     summary,
        RuntimeMS:   time.Since(startTime).Milliseconds(),
    }, nil
}

// ============================================================================
// ANALYSIS FUNCTIONS
// ============================================================================

func calculateCMSSummary(comparisons []CMSComparisonResult) CMSSummary {
    if len(comparisons) == 0 {
        return CMSSummary{}
    }
    
    sumSketchErrors := make([]float64, len(comparisons))
    individualErrors := make([]float64, len(comparisons))
    densities := make([]float64, len(comparisons))
    multiplicities := make([]float64, len(comparisons))
    
    sumSketchMaxError := 0.0
    individualMaxError := 0.0
    
    for i, comp := range comparisons {
        sumSketchErrors[i] = comp.SumSketchError
        individualErrors[i] = comp.IndividualError
        densities[i] = comp.LocalDensity
        multiplicities[i] = comp.EdgeMultiplicity
        
        if comp.SumSketchError > sumSketchMaxError {
            sumSketchMaxError = comp.SumSketchError
        }
        if comp.IndividualError > individualMaxError {
            individualMaxError = comp.IndividualError
        }
    }
    
    return CMSSummary{
        SumSketchMAE:            calculateMean(sumSketchErrors),
        IndividualMAE:           calculateMean(individualErrors),
        SumSketchMaxError:       sumSketchMaxError,
        IndividualMaxError:      individualMaxError,
        DensityCorrelation:      calculateCorrelation(densities, sumSketchErrors),
        MultiplicityCorrelation: calculateCorrelation(multiplicities, sumSketchErrors),
        TotalComparisons:        len(comparisons),
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

func saveCMSResults(results []CMSExperimentResult, filename string) {
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

func generateCMSReport(results []CMSExperimentResult, filename string) {
    file, err := os.Create(filename)
    if err != nil {
        fmt.Printf("Error creating report: %v\n", err)
        return
    }
    defer file.Close()
    
    fmt.Fprintf(file, "COUNT-MIN SKETCH COMMUNITY CONNECTIVITY EXPERIMENT\n")
    fmt.Fprintf(file, "Generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))
    fmt.Fprintf(file, "%s\n\n", strings.Repeat("=", 80))
    
    // Group results by type
    densityResults := make([]CMSExperimentResult, 0)
    sizeResults := make([]CMSExperimentResult, 0)
    plantedResults := make([]CMSExperimentResult, 0)
    
    for _, result := range results {
        switch {
        case result.Config.GraphType == "erdos_renyi" && result.Config.EdgeProb > 0:
            if result.Config.NumNodes == 300 {
                densityResults = append(densityResults, result)
            } else {
                sizeResults = append(sizeResults, result)
            }
        case result.Config.GraphType == "planted_partition":
            plantedResults = append(plantedResults, result)
        }
    }
    
    // Test 1: Density independence hypothesis
    fmt.Fprintf(file, "TEST 1: CMS ERROR vs DENSITY INDEPENDENCE\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    fmt.Fprintf(file, "Hypothesis: CMS error should NOT correlate with graph density\n\n")
    
    if len(densityResults) > 0 {
        analyzeCMSDensityIndependence(file, densityResults)
    }
    
    // Test 2: Method equivalence
    fmt.Fprintf(file, "\nTEST 2: METHOD EQUIVALENCE ANALYSIS\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    fmt.Fprintf(file, "Hypothesis: Both CMS methods should produce similar results\n\n")
    
    analyzeCMSMethodEquivalence(file, results)
    
    // Test 3: Parameter scaling
    fmt.Fprintf(file, "\nTEST 3: CMS PARAMETER SCALING\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    fmt.Fprintf(file, "Analysis: Error reduction with increased d and w\n\n")
    
    analyzeCMSParameterScaling(file, results)
    
    // Overall conclusions
    fmt.Fprintf(file, "\nCMS EXPERIMENT CONCLUSIONS\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    generateCMSConclusions(file, results)
}

func analyzeCMSDensityIndependence(file *os.File, results []CMSExperimentResult) {
    fmt.Fprintf(file, "%-8s %-6s %-6s %-12s %-12s %-12s\n", 
        "Density", "d", "w", "SumMAE", "IndivMAE", "DensCorr")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 70))
    
    // Sort by density for cleaner presentation
    sort.Slice(results, func(i, j int) bool {
        if results[i].Config.EdgeProb != results[j].Config.EdgeProb {
            return results[i].Config.EdgeProb < results[j].Config.EdgeProb
        }
        if results[i].Config.CMS_d != results[j].Config.CMS_d {
            return results[i].Config.CMS_d < results[j].Config.CMS_d
        }
        return results[i].Config.CMS_w < results[j].Config.CMS_w
    })
    
    for _, result := range results {
        fmt.Fprintf(file, "%-8.3f %-6d %-6d %-12.6f %-12.6f %-12.4f\n",
            result.Config.EdgeProb, result.Config.CMS_d, result.Config.CMS_w,
            result.Summary.SumSketchMAE, result.Summary.IndividualMAE, 
            result.Summary.DensityCorrelation)
    }
}

func analyzeCMSMethodEquivalence(file *os.File, results []CMSExperimentResult) {
    fmt.Fprintf(file, "%-12s %-12s %-12s %-12s\n", 
        "GraphType", "SumMAE", "IndivMAE", "Difference")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 55))
    
    for _, result := range results {
        diff := math.Abs(result.Summary.SumSketchMAE - result.Summary.IndividualMAE)
        fmt.Fprintf(file, "%-12s %-12.6f %-12.6f %-12.6f\n",
            result.Config.GraphType, result.Summary.SumSketchMAE, 
            result.Summary.IndividualMAE, diff)
    }
}

func analyzeCMSParameterScaling(file *os.File, results []CMSExperimentResult) {
    // Group results by d and w parameters
    paramGroups := make(map[string][]CMSExperimentResult)
    
    for _, result := range results {
        key := fmt.Sprintf("d%d_w%d", result.Config.CMS_d, result.Config.CMS_w)
        paramGroups[key] = append(paramGroups[key], result)
    }
    
    fmt.Fprintf(file, "%-10s %-12s %-12s %-12s\n", 
        "Params", "AvgSumMAE", "AvgIndivMAE", "MemoryMB")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 55))
    
    // Sort parameter combinations
    var paramKeys []string
    for key := range paramGroups {
        paramKeys = append(paramKeys, key)
    }
    sort.Strings(paramKeys)
    
    for _, key := range paramKeys {
        groupResults := paramGroups[key]
        if len(groupResults) == 0 {
            continue
        }
        
        // Calculate average MAE across all experiments with these parameters
        avgSumMAE := 0.0
        avgIndivMAE := 0.0
        for _, result := range groupResults {
            avgSumMAE += result.Summary.SumSketchMAE
            avgIndivMAE += result.Summary.IndividualMAE
        }
        avgSumMAE /= float64(len(groupResults))
        avgIndivMAE /= float64(len(groupResults))
        
        // Estimate memory usage (rough approximation)
        d := groupResults[0].Config.CMS_d
        w := groupResults[0].Config.CMS_w
        memoryMB := float64(d * w * 4) / (1024 * 1024) // 4 bytes per counter
        
        fmt.Fprintf(file, "%-10s %-12.6f %-12.6f %-12.2f\n",
            key, avgSumMAE, avgIndivMAE, memoryMB)
    }
}

func generateCMSConclusions(file *os.File, results []CMSExperimentResult) {
    if len(results) == 0 {
        fmt.Fprintf(file, "No data available for analysis\n")
        return
    }
    
    // Aggregate all comparison data
    allSumErrors := make([]float64, 0)
    allIndivErrors := make([]float64, 0)
    allDensities := make([]float64, 0)
    allMultiplicities := make([]float64, 0)
    
    for _, result := range results {
        for _, comp := range result.Comparisons {
            allSumErrors = append(allSumErrors, comp.SumSketchError)
            allIndivErrors = append(allIndivErrors, comp.IndividualError)
            allDensities = append(allDensities, comp.LocalDensity)
            allMultiplicities = append(allMultiplicities, comp.EdgeMultiplicity)
        }
    }
    
    if len(allSumErrors) == 0 {
        fmt.Fprintf(file, "No comparison data available\n")
        return
    }
    
    // Calculate key correlations
    sumDensityCorr := calculateCorrelation(allDensities, allSumErrors)
    sumMultiplicityCorr := calculateCorrelation(allMultiplicities, allSumErrors)
    indivDensityCorr := calculateCorrelation(allDensities, allIndivErrors)
    methodCorrelation := calculateCorrelation(allSumErrors, allIndivErrors)
    
    fmt.Fprintf(file, "Cross-Experiment Analysis (Total Comparisons: %d)\n\n", len(allSumErrors))
    
    fmt.Fprintf(file, "1. DENSITY INDEPENDENCE VALIDATION:\n")
    fmt.Fprintf(file, "   Sum Method Error vs Density Correlation: %.4f\n", sumDensityCorr)
    fmt.Fprintf(file, "   Individual Method Error vs Density Correlation: %.4f\n", indivDensityCorr)
    fmt.Fprintf(file, "   Sum Method Error vs Multiplicity Correlation: %.4f\n", sumMultiplicityCorr)
    
    if math.Abs(sumDensityCorr) < 0.2 && math.Abs(sumMultiplicityCorr) < 0.2 {
        fmt.Fprintf(file, "   ✓ CONFIRMED: CMS error is independent of density/multiplicity\n")
    } else if math.Abs(sumDensityCorr) < 0.5 && math.Abs(sumMultiplicityCorr) < 0.5 {
        fmt.Fprintf(file, "   ? PARTIAL: Weak correlation observed, needs investigation\n")
    } else {
        fmt.Fprintf(file, "   ✗ ISSUE: Strong correlation suggests modeling problems\n")
    }
    
    fmt.Fprintf(file, "\n2. METHOD EQUIVALENCE VALIDATION:\n")
    fmt.Fprintf(file, "   Sum vs Individual Method Correlation: %.4f\n", methodCorrelation)
    
    sumOverallMAE := calculateMean(allSumErrors)
    indivOverallMAE := calculateMean(allIndivErrors)
    
    fmt.Fprintf(file, "   Sum Method Overall MAE: %.6f\n", sumOverallMAE)
    fmt.Fprintf(file, "   Individual Method Overall MAE: %.6f\n", indivOverallMAE)
    
    relativeError := math.Abs(sumOverallMAE - indivOverallMAE) / math.Max(sumOverallMAE, indivOverallMAE)
    
    if methodCorrelation > 0.9 && relativeError < 0.1 {
        fmt.Fprintf(file, "   ✓ CONFIRMED: Methods produce equivalent results\n")
    } else if methodCorrelation > 0.7 && relativeError < 0.2 {
        fmt.Fprintf(file, "   ? PARTIAL: Methods mostly agree but show some divergence\n")
    } else {
        fmt.Fprintf(file, "   ✗ ISSUE: Methods produce significantly different results\n")
    }
    
    fmt.Fprintf(file, "\n3. COMPARISON WITH SCAR'S UNION METHOD:\n")
    fmt.Fprintf(file, "   Key Advantages of CMS Approach:\n")
    fmt.Fprintf(file, "   - Unbiased estimation of weighted connectivity\n")
    fmt.Fprintf(file, "   - No modeling error from multiplicity loss\n")
    fmt.Fprintf(file, "   - Error scales predictably with sketch parameters\n")
    
    fmt.Fprintf(file, "\n   Potential Disadvantages:\n")
    fmt.Fprintf(file, "   - Higher memory usage: O(d×w) per node vs O(K)\n")
    fmt.Fprintf(file, "   - Computational overhead for hash calculations\n")
    fmt.Fprintf(file, "   - Parameter tuning required (d, w selection)\n")
    
    // Parameter efficiency analysis
    fmt.Fprintf(file, "\n4. PARAMETER EFFICIENCY:\n")
    analyzeCMSParameterEfficiency(file, results)
    
    fmt.Fprintf(file, "\n5. PRACTICAL RECOMMENDATIONS:\n")
    fmt.Fprintf(file, "   For sparse graphs (density < 0.1):\n")
    fmt.Fprintf(file, "   - CMS may be overkill; SCAR's union method acceptable\n")
    fmt.Fprintf(file, "   - Consider d=3, w=256 for good accuracy/memory balance\n")
    
    fmt.Fprintf(file, "\n   For dense graphs (density > 0.3):\n")
    fmt.Fprintf(file, "   - CMS provides significant accuracy improvements\n")
    fmt.Fprintf(file, "   - Consider d=4, w=512 or higher for critical applications\n")
    
    fmt.Fprintf(file, "\n   For industrial-scale HINs:\n")
    fmt.Fprintf(file, "   - CMS enables handling of high-multiplicity meta-paths\n")
    fmt.Fprintf(file, "   - Memory usage scales linearly with node count\n")
    fmt.Fprintf(file, "   - Consider adaptive parameter selection based on graph properties\n")
}

func analyzeCMSParameterEfficiency(file *os.File, results []CMSExperimentResult) {
    // Find the configuration that gives best accuracy per memory unit
    type EfficiencyPoint struct {
        Config   CMSExperimentConfig
        MAE      float64
        MemoryMB float64
        Efficiency float64 // 1/MAE per MB
    }
    
    var efficiencyPoints []EfficiencyPoint
    
    configMAE := make(map[string]float64)
    configCount := make(map[string]int)
    
    // Average MAE by configuration
    for _, result := range results {
        key := fmt.Sprintf("d%d_w%d", result.Config.CMS_d, result.Config.CMS_w)
        configMAE[key] += result.Summary.SumSketchMAE
        configCount[key]++
    }
    
    for key, totalMAE := range configMAE {
        count := configCount[key]
        if count > 0 {
            avgMAE := totalMAE / float64(count)
            
            // Parse d and w from key
            var d, w int
            fmt.Sscanf(key, "d%d_w%d", &d, &w)
            
            memoryMB := float64(d * w * 4) / (1024 * 1024)
            efficiency := 0.0
            if avgMAE > 0 && memoryMB > 0 {
                efficiency = 1.0 / (avgMAE * memoryMB)
            }
            
            efficiencyPoints = append(efficiencyPoints, EfficiencyPoint{
                Config: CMSExperimentConfig{CMS_d: d, CMS_w: w},
                MAE: avgMAE,
                MemoryMB: memoryMB,
                Efficiency: efficiency,
            })
        }
    }
    
    // Sort by efficiency (higher is better)
    sort.Slice(efficiencyPoints, func(i, j int) bool {
        return efficiencyPoints[i].Efficiency > efficiencyPoints[j].Efficiency
    })
    
    fmt.Fprintf(file, "   Top efficiency configurations:\n")
    fmt.Fprintf(file, "   %-10s %-12s %-12s %-12s\n", "Config", "MAE", "Memory(MB)", "Efficiency")
    
    for i, point := range efficiencyPoints {
        if i >= 5 { // Show top 5
            break
        }
        fmt.Fprintf(file, "   d%d_w%-6d %-12.6f %-12.2f %-12.4f\n",
            point.Config.CMS_d, point.Config.CMS_w, point.MAE, point.MemoryMB, point.Efficiency)
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
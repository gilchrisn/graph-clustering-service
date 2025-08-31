package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"bufio"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/utils"
)

// Configuration for edge weight distribution experiment
type ExperimentConfig struct {
	SampleInterval     int      `json:"sample_interval"`     // Sample every N moves
	KValues           []int    `json:"k_values"`            // SCAR k values to test
	DistanceMetrics   []string `json:"distance_metrics"`    // ["mae", "wasserstein"]
	MaxSamples        int      `json:"max_samples"`         // Maximum number of samples to collect
}

// EdgeWeightDistributionSample captures edge weight calculations at a specific move
type EdgeWeightDistributionSample struct {
	MoveID              int                            `json:"move_id"`
	Node                int                            `json:"node"`
	CurrentCommunity    int                            `json:"current_community"`
	NumCommunities      int                            `json:"num_communities"`
	LouvainEdgeWeights  []float64                      `json:"louvain_edge_weights"`   // Edge weights to each community
	ScarEdgeWeights     map[int][]float64              `json:"scar_edge_weights"`      // k -> edge weights
	DistanceMeasures    map[int]DistanceMeasures       `json:"distance_measures"`      // k -> distances
}

// DistanceMeasures contains various distance metrics
type DistanceMeasures struct {
	MAE                float64 `json:"mae"`                 // Mean Absolute Error
	WassersteinP1      float64 `json:"wasserstein_p1"`
	CosineSimilarity   float64 `json:"cosine_similarity"`   // For reference
	PearsonCorrelation float64 `json:"pearson_correlation"` // For reference
}

// EdgeWeightDistributionExperiment contains all experiment data
type EdgeWeightDistributionExperiment struct {
	Config             ExperimentConfig             `json:"config"`
	GraphInfo          GraphInfo                    `json:"graph_info"`
	Samples            []EdgeWeightDistributionSample `json:"samples"`
	AggregatedMetrics  map[int]AggregatedMetrics    `json:"aggregated_metrics"` // k -> aggregated results
}

type GraphInfo struct {
	NumNodes      int     `json:"num_nodes"`
	NumEdges      int     `json:"num_edges"`
	TotalWeight   float64 `json:"total_weight"`
	AvgDegree     float64 `json:"avg_degree"`
}

type AggregatedMetrics struct {
	K                    int     `json:"k"`
	NumSamples          int     `json:"num_samples"`
	AvgMAE              float64 `json:"avg_mae"`
	AvgWassersteinP1    float64 `json:"avg_wasserstein_p1"`
	AvgCosineSimilarity float64 `json:"avg_cosine_similarity"`
	AvgPearsonCorr      float64 `json:"avg_pearson_correlation"`
	StdMAE              float64 `json:"std_mae"`
	StdWassersteinP1    float64 `json:"std_wasserstein_p1"`
}

// SCAR graph cache for different k values
type ScarGraphCache struct {
	graphs      map[int]*scar.SketchGraph     // k -> SketchGraph
	comms       map[int]*scar.Community       // k -> Community (for tracking state)
	nodeMappings map[int]*scar.NodeMapping    // k -> NodeMapping (for conversion)
	matToScar   map[int]map[int]int           // k -> (matIndex -> scarCompressedIndex)
}

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("🎯 EDGE WEIGHT DISTRIBUTION ANALYSIS")
	fmt.Println("====================================")
	fmt.Println("Comparing edge weight estimation distributions: Louvain vs SCAR(k)")

	// Experiment configuration
	expConfig := ExperimentConfig{
		SampleInterval:  5,  // Sample every 5 moves
		KValues:        []int{2, 16, 64, 128, 256, 512, 1024},
		DistanceMetrics: []string{"mae", "wasserstein"},
		MaxSamples:     5000, // Limit samples for manageable output
	}

	// Step 1: Materialize reference graph
	fmt.Println("\nStep 1: Materializing reference graph...")
	materializedGraph, nodeMapping, err := materializeReferenceGraph(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to materialize graph: %v", err)
	}
	fmt.Printf("✅ Materialized graph: %d nodes, %.2f total weight\n", 
		materializedGraph.NumNodes, materializedGraph.TotalWeight)

	// Step 2: Get Louvain moves for replication
	fmt.Println("\nStep 2: Running Louvain to get move sequence...")
	louvainMoves, err := runLouvainForMoves(materializedGraph)
	if err != nil {
		log.Fatalf("Failed to get Louvain moves: %v", err)
	}
	fmt.Printf("✅ Louvain produced %d moves\n", len(louvainMoves))

	// Step 3: Build SCAR graphs for different k values
	fmt.Println("\nStep 3: Building SCAR graphs for different k values...")
	scarCache, err := buildScarGraphCache(graphFile, propertiesFile, pathFile, expConfig.KValues, nodeMapping)
	if err != nil {
		log.Fatalf("Failed to build SCAR graphs: %v", err)
	}
	fmt.Printf("✅ Built SCAR graphs for k values: %v\n", expConfig.KValues)

	// Step 4: Run edge weight distribution experiment
	fmt.Println("\nStep 4: Running edge weight distribution experiment...")
	experiment := runEdgeWeightDistributionExperiment(materializedGraph, louvainMoves, scarCache, expConfig)

	// Step 5: Export results
	fmt.Println("\nStep 5: Exporting results...")
	if err := exportEdgeWeightDistributionExperiment(experiment); err != nil {
		log.Fatalf("Failed to export results: %v", err)
	}

	// Step 6: Display summary
	displayExperimentSummary(experiment)
}

func runLouvainForMoves(graph *louvain.Graph) ([]utils.MoveEvent, error) {
	config := louvain.NewConfig()
	config.Set("algorithm.random_seed", int64(42))
	config.Set("algorithm.max_iterations", 100)
	config.Set("algorithm.min_modularity_gain", 1e-6)
	config.Set("analysis.track_moves", true)
	config.Set("analysis.output_file", "temp_louvain_moves.jsonl")

	_, err := louvain.Run(graph, config, context.Background())
	if err != nil {
		return nil, err
	}

	moves, err := loadMovesFromJSONL("temp_louvain_moves.jsonl")
	if err != nil {
		return nil, err
	}

	// Clean up temp file
	os.Remove("temp_louvain_moves.jsonl")
	return moves, nil
}

func buildScarGraphCache(graphFile, propertiesFile, pathFile string, kValues []int, 
	materializedNodeMapping map[string]int) (*ScarGraphCache, error) {
	
	cache := &ScarGraphCache{
		graphs:       make(map[int]*scar.SketchGraph),
		comms:        make(map[int]*scar.Community),
		nodeMappings: make(map[int]*scar.NodeMapping),
	}

	for _, k := range kValues {
		fmt.Printf("  Building SCAR graph for k=%d...", k)
		
		config := scar.NewConfig()
		config.Set("algorithm.random_seed", int64(42))
		config.Set("scar.k", int64(k))
		config.Set("scar.nk", int64(1))

		// Build sketch graph using existing logic
		sketchGraph, scarNodeMapping, err := scar.BuildSketchGraph(graphFile, propertiesFile, pathFile, config, 
			config.CreateLogger())
		if err != nil {
			return nil, fmt.Errorf("failed to build SCAR graph for k=%d: %w", k, err)
		}

		cache.graphs[k] = sketchGraph
		cache.comms[k] = scar.NewCommunity(sketchGraph)
		cache.nodeMappings[k] = scarNodeMapping
		
		fmt.Printf(" ✅ (%d nodes)\n", sketchGraph.NumNodes)
	}

	return cache, nil
}

func runEdgeWeightDistributionExperiment(graph *louvain.Graph, moves []utils.MoveEvent, 
	scarCache *ScarGraphCache, config ExperimentConfig) *EdgeWeightDistributionExperiment {
	
	experiment := &EdgeWeightDistributionExperiment{
		Config: config,
		GraphInfo: GraphInfo{
			NumNodes:    graph.NumNodes,
			TotalWeight: graph.TotalWeight,
			AvgDegree:   (graph.TotalWeight * 2) / float64(graph.NumNodes),
		},
		Samples:           make([]EdgeWeightDistributionSample, 0),
		AggregatedMetrics: make(map[int]AggregatedMetrics),
	}

	// Initialize communities
	louvainComm := louvain.NewCommunity(graph)
	
	// Track move index
	sampleCount := 0
	
	fmt.Printf("  Sampling every %d moves (max %d samples)...\n", config.SampleInterval, config.MaxSamples)

	for moveIdx, move := range moves {
		// Apply the move to all communities first
		louvain.MoveNode(graph, louvainComm, move.Node, 
			louvainComm.NodeToCommunity[move.Node], move.ToComm)
		
		// Apply move to SCAR communities as well (direct node mapping)
		for k, scarComm := range scarCache.comms {
			// Direct mapping: materialized index = SCAR compressed index
			if move.Node < scarCache.graphs[k].NumNodes {
				scar.MoveNode(scarCache.graphs[k], scarComm, move.Node, 
					scarComm.NodeToCommunity[move.Node], move.ToComm)
			}
		}

		// Sample edge weight distributions at specified intervals
		if moveIdx%config.SampleInterval == 0 && sampleCount < config.MaxSamples {
			if moveIdx+1 < len(moves) { // Look ahead to next move
				nextMove := moves[moveIdx+1]
				sample := captureEdgeWeightDistribution(graph, nextMove.Node, louvainComm, scarCache, config.KValues)
				sample.MoveID = moveIdx + 1
				experiment.Samples = append(experiment.Samples, sample)
				sampleCount++
				
				if sampleCount%10 == 0 {
					fmt.Printf("    Captured %d samples...\n", sampleCount)
				}
			}
		}
	}

	// Calculate aggregated metrics
	experiment.AggregatedMetrics = calculateAggregatedMetrics(experiment.Samples, config.KValues)
	
	fmt.Printf("  ✅ Captured %d total samples\n", len(experiment.Samples))
	return experiment
}

func captureEdgeWeightDistribution(graph *louvain.Graph, node int, louvainComm *louvain.Community,
	scarCache *ScarGraphCache, kValues []int) EdgeWeightDistributionSample {
	
	sample := EdgeWeightDistributionSample{
		Node:             node,
		CurrentCommunity: louvainComm.NodeToCommunity[node],
		NumCommunities:   louvainComm.NumCommunities,
		ScarEdgeWeights:  make(map[int][]float64),
		DistanceMeasures: make(map[int]DistanceMeasures),
	}

	// First, find communities active in Louvain
	louvainActiveCommunities := getActiveCommunityIDs(louvainComm)

	// Calculate SCAR edge weights for each k value
	for _, k := range kValues {
		scarGraph := scarCache.graphs[k]
		scarComm := scarCache.comms[k]
		
		if node < scarGraph.NumNodes {
			// Find communities active in SCAR
			scarActiveCommunities := getActiveCommunityIDsScar(scarComm)
			
			// Find intersection: communities active in BOTH
			commonCommunities := intersection(louvainActiveCommunities, scarActiveCommunities)
			
			if len(commonCommunities) > 0 {
				// Calculate edge weights for common communities only
				sample.LouvainEdgeWeights = calculateEdgeWeightsForCommunities(graph, louvainComm, node, commonCommunities)
				sample.ScarEdgeWeights[k] = calculateScarEdgeWeightsForCommunities(scarGraph, scarComm, node, commonCommunities)
				
				// Now both arrays have the same communities in the same order
				sample.DistanceMeasures[k] = calculateDistanceMeasures(sample.LouvainEdgeWeights, sample.ScarEdgeWeights[k])
				
			} else {
				// No common communities - invalid measurement
				sample.DistanceMeasures[k] = DistanceMeasures{
					MAE:                math.Inf(1),
					WassersteinP1:      math.Inf(1),
					CosineSimilarity:   0.0,
					PearsonCorrelation: 0.0,
				}
				fmt.Printf("Warning: No common active communities for k=%d\n", k)
			}
		}
	}

	return sample
}

// Get list of active community IDs for Louvain
func getActiveCommunityIDs(comm *louvain.Community) []int {
	active := make([]int, 0)
	for commID := 0; commID < comm.NumCommunities; commID++ {
		if len(comm.CommunityNodes[commID]) > 0 {
			active = append(active, commID)
		}
	}
	sort.Ints(active)
	return active
}

// Get list of active community IDs for SCAR
func getActiveCommunityIDsScar(comm *scar.Community) []int {
	active := make([]int, 0)
	for commID := 0; commID < comm.NumCommunities; commID++ {
		if len(comm.CommunityNodes[commID]) > 0 {
			active = append(active, commID)
		}
	}
	sort.Ints(active)
	return active
}

// Find intersection of two sorted int slices
func intersection(a, b []int) []int {
	result := make([]int, 0)
	i, j := 0, 0
	
	for i < len(a) && j < len(b) {
		if a[i] == b[j] {
			result = append(result, a[i])
			i++
			j++
		} else if a[i] < b[j] {
			i++
		} else {
			j++
		}
	}
	return result
}

// Calculate edge weights for specific community IDs (in order)
func calculateEdgeWeightsForCommunities(graph *louvain.Graph, comm *louvain.Community, node int, communityIDs []int) []float64 {
	edgeWeights := make([]float64, len(communityIDs))
	
	for i, targetComm := range communityIDs {
		// Calculate edge weight to target community using exact method
		edgeWeight := 0.0
		neighbors, weights := graph.GetNeighbors(node)
		for j, neighbor := range neighbors {
			if comm.NodeToCommunity[neighbor] == targetComm {
				edgeWeight += weights[j]
			}
		}
		edgeWeights[i] = edgeWeight
	}
	
	return edgeWeights
}

// Calculate SCAR edge weights for specific community IDs (in order)
func calculateScarEdgeWeightsForCommunities(graph *scar.SketchGraph, comm *scar.Community, node int, communityIDs []int) []float64 {
	edgeWeights := make([]float64, len(communityIDs))
	
	for i, targetComm := range communityIDs {
		// Use SCAR's sketch-based edge weight estimation
		edgeWeight := graph.EstimateEdgesToCommunity(node, targetComm, comm)
		edgeWeights[i] = edgeWeight
	}
	
	return edgeWeights
}

// Calculate edge weights to ALL communities (Louvain exact method)
func calculateAllCommunityEdgeWeights(graph *louvain.Graph, comm *louvain.Community, node int) []float64 {
	edgeWeights := make([]float64, comm.NumCommunities)
	
	for targetComm := 0; targetComm < comm.NumCommunities; targetComm++ {
		if len(comm.CommunityNodes[targetComm]) == 0 {
			edgeWeights[targetComm] = 0.0
			continue
		}
		
		// Calculate edge weight to target community using exact method
		edgeWeight := 0.0
		neighbors, weights := graph.GetNeighbors(node)
		for i, neighbor := range neighbors {
			if comm.NodeToCommunity[neighbor] == targetComm {
				edgeWeight += weights[i]
			}
		}
		
		edgeWeights[targetComm] = edgeWeight
	}
	
	return edgeWeights
}

// Calculate SCAR edge weights to all communities (sketch-based method)
func calculateScarAllCommunityEdgeWeights(graph *scar.SketchGraph, comm *scar.Community, node int) []float64 {
	edgeWeights := make([]float64, comm.NumCommunities)
	
	for targetComm := 0; targetComm < comm.NumCommunities; targetComm++ {
		if len(comm.CommunityNodes[targetComm]) == 0 {
			edgeWeights[targetComm] = 0.0
			continue
		}
		
		// Use SCAR's sketch-based edge weight estimation
		edgeWeight := graph.EstimateEdgesToCommunity(node, targetComm, comm)
		edgeWeights[targetComm] = edgeWeight
	}
	
	return edgeWeights
}

// Calculate various distance measures between two distributions
func calculateDistanceMeasures(p, q []float64) DistanceMeasures {
	return DistanceMeasures{
		MAE:                meanAbsoluteError(p, q),
		WassersteinP1:      wassersteinP1(p, q),
		CosineSimilarity:   cosineSimilarity(p, q),
		PearsonCorrelation: pearsonCorrelation(p, q),
	}
}

// Mean Absolute Error: MAE = (1/n) * Σ |p(i) - q(i)|
func meanAbsoluteError(p, q []float64) float64 {
	if len(p) != len(q) || len(p) == 0 {
		return math.Inf(1)
	}
	
	sum := 0.0
	for i := 0; i < len(p); i++ {
		sum += math.Abs(p[i] - q[i])
	}
	
	return sum / float64(len(p))
}

// Normalize a distribution (handle negative values by shifting and scaling)
func normalize(dist []float64) []float64 {
	if len(dist) == 0 {
		return dist
	}
	
	// Shift to make all values non-negative
	minVal := dist[0]
	for _, v := range dist {
		if v < minVal {
			minVal = v
		}
	}
	
	// Shift and calculate sum
	sum := 0.0
	shifted := make([]float64, len(dist))
	for i, v := range dist {
		shifted[i] = v - minVal + 1e-10 // Add small epsilon to avoid zeros
		sum += shifted[i]
	}
	
	// Normalize
	if sum == 0 {
		return shifted
	}
	
	normalized := make([]float64, len(shifted))
	for i, v := range shifted {
		normalized[i] = v / sum
	}
	
	return normalized
}

// Wasserstein P1 distance (1st order)
func wassersteinP1(p, q []float64) float64 {
	if len(p) != len(q) {
		return math.Inf(1)
	}
	
	// Normalize distributions first
	pNorm := normalize(p)
	qNorm := normalize(q)
	
	// Calculate cumulative distributions
	cdfP := make([]float64, len(pNorm))
	cdfQ := make([]float64, len(qNorm))
	
	cdfP[0] = pNorm[0]
	cdfQ[0] = qNorm[0]
	
	for i := 1; i < len(pNorm); i++ {
		cdfP[i] = cdfP[i-1] + pNorm[i]
		cdfQ[i] = cdfQ[i-1] + qNorm[i]
	}
	
	// Wasserstein distance is the L1 distance between CDFs
	distance := 0.0
	for i := 0; i < len(cdfP); i++ {
		distance += math.Abs(cdfP[i] - cdfQ[i])
	}
	
	return distance
}

// Cosine similarity
func cosineSimilarity(p, q []float64) float64 {
	if len(p) != len(q) {
		return 0.0
	}
	
	dotProduct := 0.0
	normP := 0.0
	normQ := 0.0
	
	for i := 0; i < len(p); i++ {
		dotProduct += p[i] * q[i]
		normP += p[i] * p[i]
		normQ += q[i] * q[i]
	}
	
	if normP == 0 || normQ == 0 {
		return 0.0
	}
	
	return dotProduct / (math.Sqrt(normP) * math.Sqrt(normQ))
}

// Pearson correlation coefficient
func pearsonCorrelation(p, q []float64) float64 {
	if len(p) != len(q) || len(p) < 2 {
		return 0.0
	}
	
	n := float64(len(p))
	
	// Calculate means
	meanP := 0.0
	meanQ := 0.0
	for i := 0; i < len(p); i++ {
		meanP += p[i]
		meanQ += q[i]
	}
	meanP /= n
	meanQ /= n
	
	// Calculate covariance and variances
	covariance := 0.0
	varP := 0.0
	varQ := 0.0
	
	for i := 0; i < len(p); i++ {
		diffP := p[i] - meanP
		diffQ := q[i] - meanQ
		covariance += diffP * diffQ
		varP += diffP * diffP
		varQ += diffQ * diffQ
	}
	
	if varP == 0 || varQ == 0 {
		return 0.0
	}
	
	return covariance / math.Sqrt(varP*varQ)
}

func calculateAggregatedMetrics(samples []EdgeWeightDistributionSample, kValues []int) map[int]AggregatedMetrics {
	aggregated := make(map[int]AggregatedMetrics)
	
	for _, k := range kValues {
		metrics := AggregatedMetrics{K: k}
		
		maeValues := make([]float64, 0)
		wsValues := make([]float64, 0)
		cosValues := make([]float64, 0)
		pearsonValues := make([]float64, 0)
		
		for _, sample := range samples {
			if measures, exists := sample.DistanceMeasures[k]; exists {
				if !math.IsInf(measures.MAE, 1) && !math.IsNaN(measures.MAE) {
					maeValues = append(maeValues, measures.MAE)
				}
				if !math.IsInf(measures.WassersteinP1, 1) && !math.IsNaN(measures.WassersteinP1) {
					wsValues = append(wsValues, measures.WassersteinP1)
				}
				cosValues = append(cosValues, measures.CosineSimilarity)
				pearsonValues = append(pearsonValues, measures.PearsonCorrelation)
			}
		}
		
		metrics.NumSamples = len(maeValues)
		metrics.AvgMAE = average(maeValues)
		metrics.AvgWassersteinP1 = average(wsValues)
		metrics.AvgCosineSimilarity = average(cosValues)
		metrics.AvgPearsonCorr = average(pearsonValues)
		metrics.StdMAE = standardDeviation(maeValues)
		metrics.StdWassersteinP1 = standardDeviation(wsValues)
		
		aggregated[k] = metrics
	}
	
	return aggregated
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func standardDeviation(values []float64) float64 {
	if len(values) < 2 {
		return 0.0
	}
	
	mean := average(values)
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values) - 1)
	return math.Sqrt(variance)
}

func exportEdgeWeightDistributionExperiment(experiment *EdgeWeightDistributionExperiment) error {
	// Export main experiment data
	file, err := os.Create("edge_weight_distribution_experiment.json")
	if err != nil {
		return fmt.Errorf("failed to create edge_weight_distribution_experiment.json: %w", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(experiment); err != nil {
		return fmt.Errorf("failed to encode experiment: %w", err)
	}
	
	// Export distance tables (one for each metric) in modularity table format
	for _, metric := range []string{"mae", "wasserstein_p1", "cosine_similarity"} {
		table := createDistanceMetricTable(experiment, metric)
		filename := fmt.Sprintf("edge_weight_distance_table_%s.json", metric)
		
		tableFile, err := os.Create(filename)
		if err != nil {
			return fmt.Errorf("failed to create %s: %w", filename, err)
		}
		defer tableFile.Close()
		
		tableEncoder := json.NewEncoder(tableFile)
		tableEncoder.SetIndent("", "  ")
		if err := tableEncoder.Encode(table); err != nil {
			return fmt.Errorf("failed to encode %s: %w", filename, err)
		}
		
		fmt.Printf("✅ %s table exported to %s\n", metric, filename)
	}
	
	fmt.Println("✅ Edge weight distribution experiment exported to edge_weight_distribution_experiment.json")
	return nil
}

// DistanceMetricTable in modularity table format: rows=k_values, cols=moves
type DistanceMetricTable struct {
	MetricName       string              `json:"metric_name"`
	Config           ExperimentConfig    `json:"config"`
	GraphInfo        GraphInfo           `json:"graph_info"`
	MaxSamples       int                 `json:"max_samples"`
	KValueInfo       []KValueInfo        `json:"k_value_info"`      // Info about each k value
	SampleMoves      []int               `json:"sample_moves"`      // Move numbers that were sampled
	DistanceData     [][]float64         `json:"distance_data"`     // [k_index][sample_index] = distance
	Summary          MetricSummary       `json:"summary"`
}

type KValueInfo struct {
	KValue         int     `json:"k_value"`
	NumSamples     int     `json:"num_samples"`
	AvgDistance    float64 `json:"avg_distance"`
	FinalDistance  float64 `json:"final_distance"`
	BestDistance   float64 `json:"best_distance"`
	Improvement    float64 `json:"improvement"`    // (initial - final)
}

type MetricSummary struct {
	BestConvergingK    int     `json:"best_converging_k"`      // K with lowest final distance
	WorstConvergingK   int     `json:"worst_converging_k"`     // K with highest final distance
	OverallImprovement float64 `json:"overall_improvement"`    // Best vs worst final distance
	ConvergenceTrend   string  `json:"convergence_trend"`      // "improving", "stable", "degrading"
}

func createDistanceMetricTable(experiment *EdgeWeightDistributionExperiment, metricName string) *DistanceMetricTable {
	table := &DistanceMetricTable{
		MetricName:   metricName,
		Config:       experiment.Config,
		GraphInfo:    experiment.GraphInfo,
		MaxSamples:   len(experiment.Samples),
		KValueInfo:   make([]KValueInfo, 0),
		SampleMoves:  make([]int, len(experiment.Samples)),
		DistanceData: make([][]float64, len(experiment.Config.KValues)),
	}
	
	// Extract sample move numbers
	for i, sample := range experiment.Samples {
		table.SampleMoves[i] = sample.MoveID
	}
	
	// Extract distance data for this metric: [k_index][sample_index] = distance
	for kIdx, k := range experiment.Config.KValues {
		table.DistanceData[kIdx] = make([]float64, len(experiment.Samples))
		
		distances := make([]float64, 0, len(experiment.Samples))
		
		for sampleIdx, sample := range experiment.Samples {
			var distance float64 = 0.0
			
			if measures, exists := sample.DistanceMeasures[k]; exists {
				switch metricName {
				case "mae":
					distance = measures.MAE
				case "wasserstein_p1":
					distance = measures.WassersteinP1
				case "cosine_similarity":
					distance = measures.CosineSimilarity
				case "pearson_correlation":
					distance = measures.PearsonCorrelation
				}
				
				// Handle infinite/NaN values
				if math.IsInf(distance, 1) || math.IsInf(distance, -1) || math.IsNaN(distance) {
					distance = 0.0
				}
			}
			
			table.DistanceData[kIdx][sampleIdx] = distance
			distances = append(distances, distance)
		}
		
		// Calculate k value info
		kInfo := KValueInfo{
			KValue:     k,
			NumSamples: len(distances),
		}
		
		if len(distances) > 0 {
			kInfo.AvgDistance = average(distances)
			kInfo.FinalDistance = distances[len(distances)-1]
			kInfo.BestDistance = findBestDistance(distances, metricName)
			kInfo.Improvement = distances[0] - distances[len(distances)-1] // For distance metrics, lower is better
		}
		
		table.KValueInfo = append(table.KValueInfo, kInfo)
	}
	
	// Calculate summary
	table.Summary = calculateMetricSummary(table.KValueInfo, metricName)
	
	return table
}

func findBestDistance(distances []float64, metricName string) float64 {
	if len(distances) == 0 {
		return 0.0
	}
	
	best := distances[0]
	for _, d := range distances {
		if metricName == "cosine_similarity" || metricName == "pearson_correlation" {
			// Higher is better for similarity metrics
			if d > best {
				best = d
			}
		} else {
			// Lower is better for distance metrics
			if d < best {
				best = d
			}
		}
	}
	return best
}

func calculateMetricSummary(kValueInfos []KValueInfo, metricName string) MetricSummary {
	if len(kValueInfos) == 0 {
		return MetricSummary{}
	}
	
	summary := MetricSummary{}
	
	// Find best and worst converging k values
	bestIdx := 0
	worstIdx := 0
	
	for i, kInfo := range kValueInfos {
		if metricName == "cosine_similarity" || metricName == "pearson_correlation" {
			// Higher final distance is better for similarity
			if kInfo.FinalDistance > kValueInfos[bestIdx].FinalDistance {
				bestIdx = i
			}
			if kInfo.FinalDistance < kValueInfos[worstIdx].FinalDistance {
				worstIdx = i
			}
		} else {
			// Lower final distance is better for distance metrics
			if kInfo.FinalDistance < kValueInfos[bestIdx].FinalDistance {
				bestIdx = i
			}
			if kInfo.FinalDistance > kValueInfos[worstIdx].FinalDistance {
				worstIdx = i
			}
		}
	}
	
	summary.BestConvergingK = kValueInfos[bestIdx].KValue
	summary.WorstConvergingK = kValueInfos[worstIdx].KValue
	summary.OverallImprovement = math.Abs(kValueInfos[bestIdx].FinalDistance - kValueInfos[worstIdx].FinalDistance)
	
	// Determine convergence trend
	firstHalf := kValueInfos[:len(kValueInfos)/2]
	secondHalf := kValueInfos[len(kValueInfos)/2:]
	
	avgFirst := 0.0
	avgSecond := 0.0
	
	for _, kInfo := range firstHalf {
		avgFirst += kInfo.FinalDistance
	}
	avgFirst /= float64(len(firstHalf))
	
	for _, kInfo := range secondHalf {
		avgSecond += kInfo.FinalDistance
	}
	avgSecond /= float64(len(secondHalf))
	
	if metricName == "cosine_similarity" || metricName == "pearson_correlation" {
		if avgSecond > avgFirst+0.01 {
			summary.ConvergenceTrend = "improving"
		} else if avgSecond < avgFirst-0.01 {
			summary.ConvergenceTrend = "degrading"
		} else {
			summary.ConvergenceTrend = "stable"
		}
	} else {
		if avgSecond < avgFirst-0.01 {
			summary.ConvergenceTrend = "improving"
		} else if avgSecond > avgFirst+0.01 {
			summary.ConvergenceTrend = "degrading"
		} else {
			summary.ConvergenceTrend = "stable"
		}
	}
	
	return summary
}

func displayExperimentSummary(experiment *EdgeWeightDistributionExperiment) {
	fmt.Println("\n📊 EDGE WEIGHT DISTRIBUTION EXPERIMENT SUMMARY")
	fmt.Println("===============================================")
	
	fmt.Printf("Graph: %d nodes, %.2f total weight\n", 
		experiment.GraphInfo.NumNodes, experiment.GraphInfo.TotalWeight)
	fmt.Printf("Samples: %d (every %d moves)\n", 
		len(experiment.Samples), experiment.Config.SampleInterval)
	fmt.Printf("K values tested: %v\n", experiment.Config.KValues)
	
	// Display sample tables for key metrics
	for _, metric := range []string{"mae", "wasserstein_p1"} {
		fmt.Printf("\n📋 %s TABLE (First 10 samples):\n", strings.ToUpper(strings.ReplaceAll(metric, "_", " ")))
		fmt.Printf("%-8s", "K")
		
		// Show first 10 sample moves
		sampleCount := min(10, len(experiment.Samples))
		for i := 0; i < sampleCount; i++ {
			fmt.Printf(" %-10s", fmt.Sprintf("Move%d", experiment.Samples[i].MoveID))
		}
		fmt.Println()
		
		fmt.Printf("%s", "────────")
		for i := 0; i < sampleCount; i++ {
			fmt.Printf(" %s", "──────────")
		}
		fmt.Println()
		
		// Display data for each k
		for _, k := range experiment.Config.KValues {
			fmt.Printf("%-8d", k)
			
			for i := 0; i < sampleCount; i++ {
				if i < len(experiment.Samples) {
					sample := experiment.Samples[i]
					if measures, exists := sample.DistanceMeasures[k]; exists {
						var value float64
						switch metric {
						case "mae":
							value = measures.MAE
						case "wasserstein_p1":
							value = measures.WassersteinP1
						}
						
						if math.IsInf(value, 1) || math.IsInf(value, -1) || math.IsNaN(value) {
							fmt.Printf(" %-10s", "∞")
						} else {
							fmt.Printf(" %-10.6f", value)
						}
					} else {
						fmt.Printf(" %-10s", "N/A")
					}
				}
			}
			fmt.Println()
		}
	}
	
	fmt.Println("\nAVERAGE EDGE WEIGHT DISTANCE MEASURES:")
	fmt.Printf("%-8s %-12s %-12s %-12s\n", "K", "MAE", "Wasserst", "Cosine")
	fmt.Println("--------------------------------------------------")
	
	// Sort k values for consistent display
	sortedKs := make([]int, 0, len(experiment.AggregatedMetrics))
	for k := range experiment.AggregatedMetrics {
		sortedKs = append(sortedKs, k)
	}
	sort.Ints(sortedKs)
	
	for _, k := range sortedKs {
		metrics := experiment.AggregatedMetrics[k]
		fmt.Printf("%-8d %-12.6f %-12.6f %-12.6f\n", 
			k, metrics.AvgMAE, metrics.AvgWassersteinP1, metrics.AvgCosineSimilarity)
	}
	
	fmt.Println("\n💡 INTERPRETATION:")
	fmt.Println("   • Lower MAE/Wasserstein = more similar edge weight distributions")
	fmt.Println("   • Higher Cosine similarity = more similar edge weight distributions")
	fmt.Println("   • As k → ∞, expect SCAR edge weight estimates → Louvain exact edge weights")
	fmt.Println("   • This measures how well SCAR's sketch-based edge weight estimation works")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Materialize the reference graph from SCAR input files
func materializeReferenceGraph(graphFile, propertiesFile, pathFile string) (*louvain.Graph, map[string]int, error) {
	// Parse SCAR input
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
	if err != nil {
		return nil, nil, fmt.Errorf("parse failed: %w", err)
	}

	// Run materialization
	config := materialization.DefaultMaterializationConfig()
	config.Aggregation.Strategy = materialization.Average
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()
	if err != nil {
		return nil, nil, fmt.Errorf("materialization failed: %w", err)
	}

	// Convert to Louvain graph (same logic as the working code)
	hgraph := result.HomogeneousGraph
	nodeList := make([]string, 0, len(hgraph.Nodes))
	for nodeID := range hgraph.Nodes {
		nodeList = append(nodeList, nodeID)
	}

	// Sort nodes (integer-aware sorting)
	allIntegers := true
	for _, nodeID := range nodeList {
		if _, err := strconv.Atoi(nodeID); err != nil {
			allIntegers = false
			break
		}
	}

	if allIntegers {
		sort.Slice(nodeList, func(i, j int) bool {
			a, _ := strconv.Atoi(nodeList[i])
			b, _ := strconv.Atoi(nodeList[j])
			return a < b
		})
	} else {
		sort.Strings(nodeList)
	}

	// Build node mapping
	nodeMapping := make(map[string]int)
	for i, originalID := range nodeList {
		nodeMapping[originalID] = i
	}

	// Create Louvain graph
	louvainGraph := louvain.NewGraph(len(nodeList))
	processedEdges := make(map[string]bool)

	for edgeKey, weight := range hgraph.Edges {
		fromIdx, fromExists := nodeMapping[edgeKey.From]
		toIdx, toExists := nodeMapping[edgeKey.To]

		if !fromExists || !toExists {
			continue
		}

		// Avoid duplicate edges
		var canonicalID string
		if fromIdx <= toIdx {
			canonicalID = fmt.Sprintf("%d-%d", fromIdx, toIdx)
		} else {
			canonicalID = fmt.Sprintf("%d-%d", toIdx, fromIdx)
		}

		if !processedEdges[canonicalID] {
			louvainGraph.AddEdge(fromIdx, toIdx, weight)
			processedEdges[canonicalID] = true
		}
	}

	return louvainGraph, nodeMapping, nil
}

// loadMovesFromJSONL loads moves from JSONL format (one JSON object per line)
func loadMovesFromJSONL(filename string) ([]utils.MoveEvent, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s: %w", filename, err)
	}
	defer file.Close()

	var moves []utils.MoveEvent
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		var move utils.MoveEvent
		if err := json.Unmarshal(scanner.Bytes(), &move); err != nil {
			continue // Skip malformed lines
		}
		moves = append(moves, move)
	}

	return moves, scanner.Err()
}
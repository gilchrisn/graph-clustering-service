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

// Configuration for move selection quality experiment
type ExperimentConfig struct {
	KValues               []int    // SCAR k values to test
	NumRuns               int      // Number of experiment runs
	SampleInterval        int      // Sample every N moves
	MaxSamples            int      // Maximum samples per run
	MinCommunities        int      // Minimum communities for valid sample
	RandomSeeds           []int64  // Seeds for each run (auto-generated)
	OutputPrefix          string   // Prefix for output files
	EnableDetailedLogging bool     // Enable detailed SCAR calculation logging
	LogSampleInterval     int      // Log every N samples (1 = all samples)
}

// MoveSelectionQualitySample captures move selection comparison at a specific point
type MoveSelectionQualitySample struct {
	RunID                  int                            `json:"run_id"`
	MoveID                 int                            `json:"move_id"`
	Node                   int                            `json:"node"`
	CurrentCommunity       int                            `json:"current_community"`
	LouvainTargetCommunity int                            `json:"louvain_target_community"`
	LouvainGain           float64                         `json:"louvain_gain"`
	NumActiveCommunities   int                            `json:"num_active_communities"`
	ScarRankings          map[int]*CommunityRanking      `json:"scar_rankings"` // k -> ranking info
}

// CommunityRanking contains the ranking of communities by modularity gain
type CommunityRanking struct {
	Communities      []int     `json:"communities"`        // sorted by gain (best to worst)
	Gains           []float64 `json:"gains"`              // corresponding gains
	TargetRank      int       `json:"target_rank"`        // rank of Louvain's target (1=best, 2=second best, etc.)
	NormalizedScore float64   `json:"normalized_score"`   // (best_gain - target_gain) / (best_gain - worst_gain)
	TargetGain      float64   `json:"target_gain"`        // gain for Louvain's target community
	BestGain        float64   `json:"best_gain"`          // highest gain found by SCAR
	WorstGain       float64   `json:"worst_gain"`         // lowest gain found by SCAR
}

// AggregatedExperimentResults contains results from multiple runs
type AggregatedExperimentResults struct {
	Config                  ExperimentConfig              `json:"config"`
	GraphInfo              GraphInfo                     `json:"graph_info"`
	AllSamples             []MoveSelectionQualitySample  `json:"all_samples"`          // All samples from all runs
	RunSummaries           []RunSummary                  `json:"run_summaries"`        // Summary for each run
	AggregatedMetrics      map[int]*AggregatedMoveMetrics `json:"aggregated_metrics"`  // k -> aggregated across runs
	BestKAnalysis          BestKAnalysis                 `json:"best_k_analysis"`      // Analysis of optimal k
}

type RunSummary struct {
	RunID             int                           `json:"run_id"`
	RandomSeed        int64                         `json:"random_seed"`
	NumSamples        int                           `json:"num_samples"`
	RunMetrics        map[int]*MoveQualityMetrics   `json:"run_metrics"`  // k -> metrics for this run
}

type GraphInfo struct {
	NumNodes      int     `json:"num_nodes"`
	NumEdges      int     `json:"num_edges"`
	TotalWeight   float64 `json:"total_weight"`
	AvgDegree     float64 `json:"avg_degree"`
}

// AggregatedMoveMetrics contains metrics aggregated across multiple runs
type AggregatedMoveMetrics struct {
	K                     int                `json:"k"`
	TotalSamples         int                `json:"total_samples"`         // Total samples across all runs
	AvgTargetRank        float64            `json:"avg_target_rank"`       // Average across runs
	StdTargetRank        float64            `json:"std_target_rank"`       // Standard deviation across runs
	AvgPerfectMatchRate  float64            `json:"avg_perfect_match_rate"`
	StdPerfectMatchRate  float64            `json:"std_perfect_match_rate"`
	AvgTop3Rate          float64            `json:"avg_top3_rate"`
	StdTop3Rate          float64            `json:"std_top3_rate"`
	AvgNormalizedScore   float64            `json:"avg_normalized_score"`
	ConsistencyScore     float64            `json:"consistency_score"`     // How consistent across runs (lower std = more consistent)
	RunBreakdown         []RunMetricBreakdown `json:"run_breakdown"`       // Per-run breakdown
}

type RunMetricBreakdown struct {
	RunID              int     `json:"run_id"`
	TargetRank         float64 `json:"target_rank"`
	PerfectMatchRate   float64 `json:"perfect_match_rate"`
	Top3Rate           float64 `json:"top3_rate"`
	NormalizedScore    float64 `json:"normalized_score"`
}

// MoveQualityMetrics contains quality metrics for a specific k value in a single run
type MoveQualityMetrics struct {
	K                    int                `json:"k"`
	NumSamples          int                `json:"num_samples"`
	AvgTargetRank       float64            `json:"avg_target_rank"`
	AvgNormalizedScore  float64            `json:"avg_normalized_score"`
	TopKPercentage      map[int]float64    `json:"top_k_percentage"`    // percentage of times target was in top-k
	PerfectMatches      int                `json:"perfect_matches"`     // times SCAR picked same as Louvain
	PerfectMatchRate    float64            `json:"perfect_match_rate"`  // perfect_matches / num_samples
	RankDistribution    map[int]int        `json:"rank_distribution"`   // rank -> count
}

type BestKAnalysis struct {
	BestKByAvgRank       int     `json:"best_k_by_avg_rank"`
	BestKByPerfectRate   int     `json:"best_k_by_perfect_rate"`
	BestKByConsistency   int     `json:"best_k_by_consistency"`
	RecommendedK         int     `json:"recommended_k"`
	RecommendedKScore    float64 `json:"recommended_k_score"`
	KEfficiencyRanking   []KEfficiency `json:"k_efficiency_ranking"`
}

type KEfficiency struct {
	K              int     `json:"k"`
	EfficiencyScore float64 `json:"efficiency_score"`
	AvgRank        float64 `json:"avg_rank"`
	PerfectRate    float64 `json:"perfect_rate"`
	ConsistencyScore float64 `json:"consistency_score"`
}

// DetailedSketchLog captures detailed SCAR calculations for debugging
type DetailedSketchLog struct {
	RunID                  int                         `json:"run_id"`
	MoveID                 int                         `json:"move_id"`
	SampleIndex            int                         `json:"sample_index"`
	Node                   int                         `json:"node"`
	LouvainTargetCommunity int                         `json:"louvain_target_community"`
	LouvainGain           float64                      `json:"louvain_gain"`
	NumActiveCommunities   int                         `json:"num_active_communities"`
	ScarCalculations      map[int]*ScarCalculationLog  `json:"scar_calculations"` // k -> detailed calc
}

type ScarCalculationLog struct {
	K                     int                       `json:"k"`
	NodeDegree           float64                   `json:"node_degree"`
	NodeSketchFull       bool                      `json:"node_sketch_full"`
	CommunityCalculations []CommunityCalculationLog `json:"community_calculations"`
	FinalRanking         []int                     `json:"final_ranking"`        // communities sorted by gain
	FinalGains           []float64                 `json:"final_gains"`          // corresponding gains
	TargetRank           int                       `json:"target_rank"`
	TargetGain           float64                   `json:"target_gain"`
}

type CommunityCalculationLog struct {
	CommunityID          int     `json:"community_id"`
	CommunitySize        int     `json:"community_size"`
	CommunitySketchFull  bool    `json:"community_sketch_full"`
	EdgeWeightEstimate   float64 `json:"edge_weight_estimate"`
	ModularityGain       float64 `json:"modularity_gain"`
	CommunityTotalWeight float64 `json:"community_total_weight"`
	EstimationMethod     string  `json:"estimation_method"`  // "exact" or "sketch"
}

// SCAR graph cache for different k values
type ScarGraphCache struct {
	graphs      map[int]*scar.SketchGraph     // k -> SketchGraph
	comms       map[int]*scar.Community       // k -> Community (for tracking state)
	nodeMappings map[int]*scar.NodeMapping    // k -> NodeMapping (for conversion)
}

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("🎯 MOVE SELECTION QUALITY ANALYSIS")
	fmt.Println("==================================")
	fmt.Println("Multi-run experiment comparing Louvain vs SCAR(k) move selection quality")

	// ============================================================================
	// EXPERIMENT CONFIGURATION - EDIT THESE PARAMETERS
	// ============================================================================
	config := &ExperimentConfig{
		KValues:        []int{2, 16, 64,128, 256, 512},  // SCAR k values to test
		NumRuns:        10,                                   // Number of independent runs
		SampleInterval: 1,                                  // Sample every N moves
		MaxSamples:     10000,                                 // Max samples per run
		MinCommunities: 3,                                   // Min communities for valid sample
		OutputPrefix:   "move_selection",                    // Output file prefix
		EnableDetailedLogging: true,                         // Log detailed SCAR calculations
		LogSampleInterval: 1,                                // Log every N samples (1 = all samples)
	}
	
	// Alternative configurations (uncomment one to use):
	
	// QUICK TEST (fast, fewer samples)
	// config.KValues = []int{8, 32, 128}
	// config.NumRuns = 3
	// config.MaxSamples = 100
	// config.SampleInterval = 20
	// config.EnableDetailedLogging = false  // Disable for quick tests
	
	// COMPREHENSIVE (research quality, takes longer)
	// config.KValues = []int{2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
	// config.NumRuns = 10
	// config.MaxSamples = 1000
	// config.SampleInterval = 5
	// config.EnableDetailedLogging = true   // Enable for research
	// config.LogSampleInterval = 10         // Log every 10th sample to reduce file size
	
	// PRODUCTION (balanced, reliable)
	// config.KValues = []int{4, 8, 16, 32, 64, 128}
	// config.NumRuns = 7
	// config.MaxSamples = 750
	// config.SampleInterval = 8
	// config.EnableDetailedLogging = true   // Enable for debugging fluctuations
	// config.LogSampleInterval = 1          // Log every sample for detailed analysis
	// ============================================================================
	
	// Generate random seeds for reproducibility
	config.RandomSeeds = make([]int64, config.NumRuns)
	for i := 0; i < config.NumRuns; i++ {
		config.RandomSeeds[i] = int64(42 + i*1000) // Deterministic but different
	}

	fmt.Printf("Configuration: %d runs, k values: %v, %d max samples per run\n", 
		config.NumRuns, config.KValues, config.MaxSamples)
	
	if config.EnableDetailedLogging {
		fmt.Printf("🔍 Simple debug logging enabled: Will create %s_debug_runX.txt files\n", config.OutputPrefix)
		fmt.Printf("   Logging every %d samples in human-readable format\n", config.LogSampleInterval)
		fmt.Printf("   Same Louvain moves applied to all k values for fair comparison\n")
	}

	// Step 1: Materialize reference graph
	fmt.Println("\nStep 1: Materializing reference graph...")
	materializedGraph, nodeMapping, err := materializeReferenceGraph(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to materialize graph: %v", err)
	}
	fmt.Printf("✅ Materialized graph: %d nodes, %.2f total weight\n", 
		materializedGraph.NumNodes, materializedGraph.TotalWeight)

	// Step 2: Generate shared Louvain moves (same for all runs)
	fmt.Printf("\nStep 2: Generating shared Louvain move sequence...\n")
	sharedMoves, err := runLouvainForMoves(materializedGraph, 42) // Fixed seed for reproducible moves
	if err != nil {
		log.Fatalf("Failed to generate shared Louvain moves: %v", err)
	}
	fmt.Printf("✅ Generated %d moves (will be applied to all SCAR variants)\n", len(sharedMoves))

	// Step 3: Run multiple experiments with shared moves
	fmt.Printf("\nStep 3: Running %d experiment iterations with shared moves...\n", config.NumRuns)
	
	results := &AggregatedExperimentResults{
		Config: *config,
		GraphInfo: GraphInfo{
			NumNodes:    materializedGraph.NumNodes,
			TotalWeight: materializedGraph.TotalWeight,
			AvgDegree:   (materializedGraph.TotalWeight * 2) / float64(materializedGraph.NumNodes),
		},
		AllSamples:   make([]MoveSelectionQualitySample, 0),
		RunSummaries: make([]RunSummary, 0),
	}

	// Run each experiment iteration
	for runID := 0; runID < config.NumRuns; runID++ {
		fmt.Printf("\n  Run %d/%d (seed: %d)...\n", runID+1, config.NumRuns, config.RandomSeeds[runID])
		
		runSamples, runMetrics, err := runSingleExperiment(materializedGraph, nodeMapping, 
			graphFile, propertiesFile, pathFile, config, runID, config.RandomSeeds[runID], sharedMoves)
		if err != nil {
			log.Printf("Warning: Run %d failed: %v", runID, err)
			continue
		}
		
		// Add run ID to all samples
		for i := range runSamples {
			runSamples[i].RunID = runID
		}
		
		results.AllSamples = append(results.AllSamples, runSamples...)
		results.RunSummaries = append(results.RunSummaries, RunSummary{
			RunID:      runID,
			RandomSeed: config.RandomSeeds[runID],
			NumSamples: len(runSamples),
			RunMetrics: runMetrics,
		})
		
		fmt.Printf("    ✅ Captured %d samples\n", len(runSamples))
	}

	// Step 4: Aggregate results across runs
	fmt.Println("\nStep 4: Aggregating results across runs...")
	results.AggregatedMetrics = aggregateResultsAcrossRuns(results.RunSummaries, config.KValues)
	results.BestKAnalysis = analyzeBestK(results.AggregatedMetrics, config.KValues)

	// Step 5: Export results
	fmt.Println("\nStep 5: Exporting aggregated results...")
	if err := exportAggregatedResults(results); err != nil {
		log.Fatalf("Failed to export results: %v", err)
	}

	// Step 6: Display summary
	displayAggregatedSummary(results)
}

func runSingleExperiment(graph *louvain.Graph, nodeMapping map[string]int, 
	graphFile, propertiesFile, pathFile string, config *ExperimentConfig, runID int, seed int64, 
	sharedMoves []utils.MoveEvent) ([]MoveSelectionQualitySample, map[int]*MoveQualityMetrics, error) {

	// Use shared moves if provided, otherwise generate new ones
	var louvainMoves []utils.MoveEvent
	var err error
	
	if sharedMoves != nil {
		fmt.Printf("    Using shared Louvain moves (%d moves)\n", len(sharedMoves))
		louvainMoves = sharedMoves
	} else {
		fmt.Printf("    Generating Louvain moves for run %d\n", runID)
		louvainMoves, err = runLouvainForMoves(graph, seed)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to get Louvain moves: %w", err)
		}
	}

	// Build SCAR graphs for different k values (using run-specific seed)
	scarCache, err := buildScarGraphCache(graphFile, propertiesFile, pathFile, config.KValues, seed)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build SCAR graphs: %w", err)
	}

	// Run experiment
	samples := runMoveSelectionExperiment(graph, louvainMoves, scarCache, config, runID)
	
	// Calculate metrics for this run
	runMetrics := calculateMoveQualityMetrics(samples, config.KValues)
	
	return samples, runMetrics, nil
}

func runLouvainForMoves(graph *louvain.Graph, seed int64) ([]utils.MoveEvent, error) {
	config := louvain.NewConfig()
	config.Set("algorithm.random_seed", seed)
	config.Set("algorithm.max_iterations", 100)
	config.Set("algorithm.min_modularity_gain", 1e-6)
	config.Set("analysis.track_moves", true)
	config.Set("analysis.output_file", fmt.Sprintf("temp_louvain_moves_%d.jsonl", seed))

	_, err := louvain.Run(graph, config, context.Background())
	if err != nil {
		return nil, err
	}

	moves, err := loadMovesFromJSONL(fmt.Sprintf("temp_louvain_moves_%d.jsonl", seed))
	if err != nil {
		return nil, err
	}

	// Clean up temp file
	os.Remove(fmt.Sprintf("temp_louvain_moves_%d.jsonl", seed))
	return moves, nil
}

func buildScarGraphCache(graphFile, propertiesFile, pathFile string, kValues []int, seed int64) (*ScarGraphCache, error) {
	cache := &ScarGraphCache{
		graphs:       make(map[int]*scar.SketchGraph),
		comms:        make(map[int]*scar.Community),
		nodeMappings: make(map[int]*scar.NodeMapping),
	}

	for _, k := range kValues {
		config := scar.NewConfig()
		config.Set("algorithm.random_seed", seed)
		config.Set("scar.k", int64(k))
		config.Set("scar.nk", int64(1))

		sketchGraph, scarNodeMapping, err := scar.BuildSketchGraph(graphFile, propertiesFile, pathFile, config, 
			config.CreateLogger())
		if err != nil {
			return nil, fmt.Errorf("failed to build SCAR graph for k=%d: %w", k, err)
		}

		cache.graphs[k] = sketchGraph
		cache.comms[k] = scar.NewCommunity(sketchGraph)
		cache.nodeMappings[k] = scarNodeMapping
	}

	return cache, nil
}

func runMoveSelectionExperiment(graph *louvain.Graph, moves []utils.MoveEvent, 
	scarCache *ScarGraphCache, config *ExperimentConfig, runID int) []MoveSelectionQualitySample {
	
	samples := make([]MoveSelectionQualitySample, 0)
	var simpleLogger *SimpleLogger
	
	// Initialize simple logging if enabled
	if config.EnableDetailedLogging {
		logFilename := fmt.Sprintf("%s_debug_run%d.txt", config.OutputPrefix, runID)
		var err error
		simpleLogger, err = NewSimpleLogger(logFilename)
		if err != nil {
			fmt.Printf("Warning: Failed to create simple logger: %v\n", err)
		} else {
			defer simpleLogger.Close()
		}
	}
	
	// Initialize communities
	louvainComm := louvain.NewCommunity(graph)
	sampleCount := 0

	for moveIdx, move := range moves {
		// Apply the move to all communities
		louvain.MoveNode(graph, louvainComm, move.Node, 
			louvainComm.NodeToCommunity[move.Node], move.ToComm)
		
		// Apply move to SCAR communities as well
		for k, scarComm := range scarCache.comms {
			if move.Node < scarCache.graphs[k].NumNodes {
				scar.MoveNode(scarCache.graphs[k], scarComm, move.Node, 
					scarComm.NodeToCommunity[move.Node], move.ToComm)
			}
		}

		// Sample move selection quality at specified intervals
		if moveIdx%config.SampleInterval == 0 && sampleCount < config.MaxSamples {
			if moveIdx+1 < len(moves) { // Look ahead to next move to predict
				nextMove := moves[moveIdx+1]
				
				// Only sample if we have enough communities for meaningful ranking
				activeCommunities := countActiveCommunities(louvainComm)
				if activeCommunities >= config.MinCommunities {
					sample := captureMoveSelectionQuality(graph, nextMove, louvainComm, scarCache, config.KValues)
					sample.MoveID = moveIdx + 1
					samples = append(samples, sample)
					
					// Log in simple format if enabled
					if simpleLogger != nil && sampleCount%config.LogSampleInterval == 0 {
						simpleLogger.LogMove(runID, moveIdx+1, sampleCount, nextMove.Node, 
							nextMove.ToComm, nextMove.Gain, sample.ScarRankings)
					}
					
					sampleCount++
				}
			}
		}
	}
	
	return samples
}

func countActiveCommunities(comm *louvain.Community) int {
	active := 0
	for c := 0; c < comm.NumCommunities; c++ {
		if len(comm.CommunityNodes[c]) > 0 {
			active++
		}
	}
	return active
}

func captureMoveSelectionQuality(graph *louvain.Graph, nextMove utils.MoveEvent, louvainComm *louvain.Community,
	scarCache *ScarGraphCache, kValues []int) MoveSelectionQualitySample {
	
	sample := MoveSelectionQualitySample{
		Node:                   nextMove.Node,
		CurrentCommunity:       louvainComm.NodeToCommunity[nextMove.Node],
		LouvainTargetCommunity: nextMove.ToComm,
		LouvainGain:           nextMove.Gain,
		NumActiveCommunities:   countActiveCommunities(louvainComm),
		ScarRankings:          make(map[int]*CommunityRanking),
	}

	// For each k value, calculate community rankings by modularity gain
	for _, k := range kValues {
		scarGraph := scarCache.graphs[k]
		scarComm := scarCache.comms[k]
		
		if nextMove.Node < scarGraph.NumNodes {
			ranking := calculateCommunityRanking(scarGraph, scarComm, nextMove.Node, nextMove.ToComm)
			sample.ScarRankings[k] = ranking
		}
	}

	return sample
}

func calculateCommunityRanking(graph *scar.SketchGraph, comm *scar.Community, node int, targetComm int) *CommunityRanking {
	type CommunityGain struct {
		CommunityID int
		Gain        float64
	}

	// Find all active communities and calculate gains
	communityGains := make([]CommunityGain, 0)
	
	for commID := 0; commID < comm.NumCommunities; commID++ {
		if len(comm.CommunityNodes[commID]) > 0 {
			// Calculate modularity gain for moving to this community
			edgeWeight := graph.EstimateEdgesToCommunity(node, commID, comm)
			gain := scar.CalculateModularityGain(graph, comm, node, commID, edgeWeight)
			
			communityGains = append(communityGains, CommunityGain{
				CommunityID: commID,
				Gain:        gain,
			})
		}
	}

	// Sort by gain (descending - best first)
	sort.Slice(communityGains, func(i, j int) bool {
		if communityGains[i].Gain == communityGains[j].Gain {
			return communityGains[i].CommunityID < communityGains[j].CommunityID // Tiebreaker
		}
		return communityGains[i].Gain > communityGains[j].Gain
	})

	// Extract sorted lists
	communities := make([]int, len(communityGains))
	gains := make([]float64, len(communityGains))
	targetRank := -1
	targetGain := 0.0
	
	for i, cg := range communityGains {
		communities[i] = cg.CommunityID
		gains[i] = cg.Gain
		
		if cg.CommunityID == targetComm {
			targetRank = i + 1 // 1-indexed ranking
			targetGain = cg.Gain
		}
	}

	// Calculate normalized score
	normalizedScore := 0.0
	if len(gains) > 1 {
		bestGain := gains[0]
		worstGain := gains[len(gains)-1]
		
		if bestGain != worstGain {
			normalizedScore = (bestGain - targetGain) / (bestGain - worstGain)
		}
	}

	ranking := &CommunityRanking{
		Communities:     communities,
		Gains:          gains,
		TargetRank:     targetRank,
		NormalizedScore: normalizedScore,
		TargetGain:     targetGain,
	}

	if len(gains) > 0 {
		ranking.BestGain = gains[0]
		ranking.WorstGain = gains[len(gains)-1]
	}

	return ranking
}

// SimpleLogger handles logging in human-readable format
type SimpleLogger struct {
	file *os.File
}

func NewSimpleLogger(filename string) (*SimpleLogger, error) {
	file, err := os.Create(filename)
	if err != nil {
		return nil, err
	}
	
	logger := &SimpleLogger{file: file}
	
	// Write header
	fmt.Fprintf(file, "# SCAR Move Selection Debug Log\n")
	fmt.Fprintf(file, "# Format: [SAMPLE] Run Move Node → TargetComm | Louvain: gain | k=X: rank gain method | ...\n")
	fmt.Fprintf(file, "# method: E=exact, S=sketch | rank: 1=perfect match\n\n")
	
	return logger, nil
}

func (sl *SimpleLogger) LogMove(runID, moveID, sampleIdx, node, targetComm int, louvainGain float64, 
	scarResults map[int]*CommunityRanking) {
	
	fmt.Fprintf(sl.file, "[%03d] R%d M%04d N%04d→C%02d | Louvain: %+.4f", 
		sampleIdx, runID, moveID, node, targetComm, louvainGain)
		
	// Sort k values for consistent output
	kValues := make([]int, 0, len(scarResults))
	for k := range scarResults {
		kValues = append(kValues, k)
	}
	sort.Ints(kValues)
	
	for _, k := range kValues {
		ranking := scarResults[k]
		if ranking != nil && ranking.TargetRank > 0 {
			method := "E" // exact
			// We'd need to determine if sketch was used - for now assume exact
			
			fmt.Fprintf(sl.file, " | k=%d: %d/%d %+.4f %s", 
				k, ranking.TargetRank, len(ranking.Communities), ranking.TargetGain, method)
		}
	}
	
	fmt.Fprintf(sl.file, "\n")
}

func (sl *SimpleLogger) Close() error {
	return sl.file.Close()
}

// captureDetailedSketchCalculations captures detailed SCAR calculations for debugging
func captureDetailedSketchCalculations(graph *louvain.Graph, nextMove utils.MoveEvent, louvainComm *louvain.Community,
	scarCache *ScarGraphCache, kValues []int, runID, sampleIndex int) *DetailedSketchLog {
	
	log := &DetailedSketchLog{
		RunID:                  runID,
		MoveID:                 nextMove.Node, // This should be the move ID, but we're using node for now
		SampleIndex:            sampleIndex,
		Node:                   nextMove.Node,
		LouvainTargetCommunity: nextMove.ToComm,
		LouvainGain:           nextMove.Gain,
		NumActiveCommunities:   countActiveCommunities(louvainComm),
		ScarCalculations:      make(map[int]*ScarCalculationLog),
	}

	// For each k value, capture detailed calculations
	for _, k := range kValues {
		scarGraph := scarCache.graphs[k]
		scarComm := scarCache.comms[k]
		
		if nextMove.Node < scarGraph.NumNodes {
			calcLog := captureDetailedCommunityCalculations(scarGraph, scarComm, nextMove.Node, nextMove.ToComm)
			log.ScarCalculations[k] = calcLog
		}
	}

	return log
}

// captureDetailedCommunityCalculations captures the detailed calculation process for one k value
func captureDetailedCommunityCalculations(graph *scar.SketchGraph, comm *scar.Community, node int, targetComm int) *ScarCalculationLog {
	calcLog := &ScarCalculationLog{
		K:                     extractKFromGraph(graph), // Helper function to extract k
		NodeDegree:           graph.GetDegree(node),
		NodeSketchFull:       isNodeSketchFull(graph, node), // Helper function
		CommunityCalculations: make([]CommunityCalculationLog, 0),
		FinalRanking:         make([]int, 0),
		FinalGains:           make([]float64, 0),
	}
	
	type CommunityGain struct {
		CommunityID int
		Gain        float64
		CalcLog     CommunityCalculationLog
	}

	// Calculate gains for all active communities and capture details
	communityGains := make([]CommunityGain, 0)
	
	for commID := 0; commID < comm.NumCommunities; commID++ {
		if len(comm.CommunityNodes[commID]) > 0 {
			// Calculate edge weight estimate
			edgeWeight := graph.EstimateEdgesToCommunity(node, commID, comm)
			
			// Calculate modularity gain
			gain := scar.CalculateModularityGain(graph, comm, node, commID, edgeWeight)
			
			// Determine estimation method based on node sketch
			estimationMethod := "exact"
			if isNodeSketchFull(graph, node) {
				estimationMethod = "sketch"
			}
			
			// Create community calculation log
			commCalcLog := CommunityCalculationLog{
				CommunityID:          commID,
				CommunitySize:        len(comm.CommunityNodes[commID]),
				CommunitySketchFull:  isCommunitySketchFull(comm, commID), // Helper function
				EdgeWeightEstimate:   edgeWeight,
				ModularityGain:       gain,
				CommunityTotalWeight: comm.CommunityWeights[commID],
				EstimationMethod:     estimationMethod,
			}
			
			communityGains = append(communityGains, CommunityGain{
				CommunityID: commID,
				Gain:        gain,
				CalcLog:     commCalcLog,
			})
		}
	}
	
	// Sort by gain (descending - best first)
	sort.Slice(communityGains, func(i, j int) bool {
		if communityGains[i].Gain == communityGains[j].Gain {
			return communityGains[i].CommunityID < communityGains[j].CommunityID
		}
		return communityGains[i].Gain > communityGains[j].Gain
	})
	
	// Extract sorted results and find target rank
	for i, cg := range communityGains {
		calcLog.FinalRanking = append(calcLog.FinalRanking, cg.CommunityID)
		calcLog.FinalGains = append(calcLog.FinalGains, cg.Gain)
		calcLog.CommunityCalculations = append(calcLog.CommunityCalculations, cg.CalcLog)
		
		if cg.CommunityID == targetComm {
			calcLog.TargetRank = i + 1 // 1-indexed
			calcLog.TargetGain = cg.Gain
		}
	}
	
	return calcLog
}

// Helper functions to work with SCAR's internal state
func extractKFromGraph(graph *scar.SketchGraph) int {
	// This might need to be implemented based on SCAR's API
	// For now, return a placeholder
	return 0 // You'll need to modify this based on actual SCAR API
}

func isNodeSketchFull(graph *scar.SketchGraph, node int) bool {
	// This would check if the node's sketch is full
	// Implementation depends on SCAR's internal API
	return false // Placeholder
}

func isCommunitySketchFull(comm *scar.Community, commID int) bool {
	// This would check if the community's sketch is full
	// Implementation depends on SCAR's internal API
	return false // Placeholder
}

func calculateMoveQualityMetrics(samples []MoveSelectionQualitySample, kValues []int) map[int]*MoveQualityMetrics {
	metrics := make(map[int]*MoveQualityMetrics)
	
	for _, k := range kValues {
		metric := &MoveQualityMetrics{
			K:                k,
			TopKPercentage:   make(map[int]float64),
			RankDistribution: make(map[int]int),
		}
		
		ranks := make([]int, 0)
		normalizedScores := make([]float64, 0)
		perfectMatches := 0
		
		for _, sample := range samples {
			if ranking, exists := sample.ScarRankings[k]; exists && ranking != nil && ranking.TargetRank > 0 {
				ranks = append(ranks, ranking.TargetRank)
				normalizedScores = append(normalizedScores, ranking.NormalizedScore)
				
				// Count perfect matches (rank 1)
				if ranking.TargetRank == 1 {
					perfectMatches++
				}
				
				// Track rank distribution
				metric.RankDistribution[ranking.TargetRank]++
			}
		}
		
		metric.NumSamples = len(ranks)
		metric.PerfectMatches = perfectMatches
		
		if metric.NumSamples > 0 {
			// Calculate averages
			rankSum := 0
			for _, rank := range ranks {
				rankSum += rank
			}
			metric.AvgTargetRank = float64(rankSum) / float64(metric.NumSamples)
			
			scoreSum := 0.0
			for _, score := range normalizedScores {
				scoreSum += score
			}
			metric.AvgNormalizedScore = scoreSum / float64(metric.NumSamples)
			
			metric.PerfectMatchRate = float64(perfectMatches) / float64(metric.NumSamples)
			
			// Calculate top-k percentages
			for _, topK := range []int{1, 3, 5, 10} {
				topKCount := 0
				for _, rank := range ranks {
					if rank <= topK {
						topKCount++
					}
				}
				metric.TopKPercentage[topK] = float64(topKCount) / float64(metric.NumSamples)
			}
		}
		
		metrics[k] = metric
	}
	
	return metrics
}

func aggregateResultsAcrossRuns(runSummaries []RunSummary, kValues []int) map[int]*AggregatedMoveMetrics {
	aggregated := make(map[int]*AggregatedMoveMetrics)
	
	for _, k := range kValues {
		metric := &AggregatedMoveMetrics{
			K:            k,
			RunBreakdown: make([]RunMetricBreakdown, 0),
		}
		
		// Collect metrics from all runs
		targetRanks := make([]float64, 0)
		perfectRates := make([]float64, 0)
		top3Rates := make([]float64, 0)
		normalizedScores := make([]float64, 0)
		totalSamples := 0
		
		for _, runSummary := range runSummaries {
			if runMetric, exists := runSummary.RunMetrics[k]; exists {
				targetRanks = append(targetRanks, runMetric.AvgTargetRank)
				perfectRates = append(perfectRates, runMetric.PerfectMatchRate)
				top3Rates = append(top3Rates, runMetric.TopKPercentage[3])
				normalizedScores = append(normalizedScores, runMetric.AvgNormalizedScore)
				totalSamples += runMetric.NumSamples
				
				metric.RunBreakdown = append(metric.RunBreakdown, RunMetricBreakdown{
					RunID:            runSummary.RunID,
					TargetRank:       runMetric.AvgTargetRank,
					PerfectMatchRate: runMetric.PerfectMatchRate,
					Top3Rate:         runMetric.TopKPercentage[3],
					NormalizedScore:  runMetric.AvgNormalizedScore,
				})
			}
		}
		
		metric.TotalSamples = totalSamples
		
		if len(targetRanks) > 0 {
			metric.AvgTargetRank = average(targetRanks)
			metric.StdTargetRank = standardDeviation(targetRanks)
			metric.AvgPerfectMatchRate = average(perfectRates)
			metric.StdPerfectMatchRate = standardDeviation(perfectRates)
			metric.AvgTop3Rate = average(top3Rates)
			metric.StdTop3Rate = standardDeviation(top3Rates)
			metric.AvgNormalizedScore = average(normalizedScores)
			
			// Consistency score: lower standard deviation = more consistent
			metric.ConsistencyScore = 1.0 / (1.0 + metric.StdTargetRank)
		}
		
		aggregated[k] = metric
	}
	
	return aggregated
}

func analyzeBestK(aggregated map[int]*AggregatedMoveMetrics, kValues []int) BestKAnalysis {
	analysis := BestKAnalysis{
		KEfficiencyRanking: make([]KEfficiency, 0),
	}
	
	bestAvgRankK := kValues[0]
	bestPerfectRateK := kValues[0]
	bestConsistencyK := kValues[0]
	
	// Find individual bests
	for _, k := range kValues {
		if metric, exists := aggregated[k]; exists {
			if aggregated[bestAvgRankK].AvgTargetRank > metric.AvgTargetRank {
				bestAvgRankK = k
			}
			if aggregated[bestPerfectRateK].AvgPerfectMatchRate < metric.AvgPerfectMatchRate {
				bestPerfectRateK = k
			}
			if aggregated[bestConsistencyK].ConsistencyScore < metric.ConsistencyScore {
				bestConsistencyK = k
			}
			
			// Calculate efficiency score
			efficiency := metric.AvgPerfectMatchRate * 0.4 + (1.0/metric.AvgTargetRank) * 0.3 + metric.ConsistencyScore * 0.3
			analysis.KEfficiencyRanking = append(analysis.KEfficiencyRanking, KEfficiency{
				K:               k,
				EfficiencyScore: efficiency,
				AvgRank:        metric.AvgTargetRank,
				PerfectRate:    metric.AvgPerfectMatchRate,
				ConsistencyScore: metric.ConsistencyScore,
			})
		}
	}
	
	// Sort by efficiency
	sort.Slice(analysis.KEfficiencyRanking, func(i, j int) bool {
		return analysis.KEfficiencyRanking[i].EfficiencyScore > analysis.KEfficiencyRanking[j].EfficiencyScore
	})
	
	analysis.BestKByAvgRank = bestAvgRankK
	analysis.BestKByPerfectRate = bestPerfectRateK
	analysis.BestKByConsistency = bestConsistencyK
	
	if len(analysis.KEfficiencyRanking) > 0 {
		analysis.RecommendedK = analysis.KEfficiencyRanking[0].K
		analysis.RecommendedKScore = analysis.KEfficiencyRanking[0].EfficiencyScore
	}
	
	return analysis
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

func exportAggregatedResults(results *AggregatedExperimentResults) error {
	// Export main aggregated results
	filename := fmt.Sprintf("%s_aggregated_results.json", results.Config.OutputPrefix)
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create %s: %w", filename, err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(results); err != nil {
		return fmt.Errorf("failed to encode results: %w", err)
	}
	
	fmt.Printf("✅ Aggregated results exported to %s\n", filename)
	return nil
}

func displayAggregatedSummary(results *AggregatedExperimentResults) {
	fmt.Println("\n📊 AGGREGATED MOVE SELECTION QUALITY SUMMARY")
	fmt.Println("=============================================")
	
	config := results.Config
	fmt.Printf("Experiment: %d runs, k values: %v\n", config.NumRuns, config.KValues)
	fmt.Printf("Total samples: %d across all runs\n", len(results.AllSamples))
	fmt.Printf("Graph: %d nodes, %.2f total weight\n", 
		results.GraphInfo.NumNodes, results.GraphInfo.TotalWeight)
	
	fmt.Println("\nAGGREGATED METRICS (MEAN ± STD):")
	fmt.Printf("%-6s %-12s %-12s %-12s %-12s %-12s\n", 
		"K", "Avg Rank", "Perfect%", "Top3%", "Consistency", "Samples")
	fmt.Println(strings.Repeat("-", 80))
	
	for _, k := range config.KValues {
		if metric, exists := results.AggregatedMetrics[k]; exists {
			fmt.Printf("%-6d %-12s %-12s %-12s %-12.3f %-12d\n",
				k,
				fmt.Sprintf("%.2f±%.2f", metric.AvgTargetRank, metric.StdTargetRank),
				fmt.Sprintf("%.1f±%.1f", metric.AvgPerfectMatchRate*100, metric.StdPerfectMatchRate*100),
				fmt.Sprintf("%.1f±%.1f", metric.AvgTop3Rate*100, metric.StdTop3Rate*100),
				metric.ConsistencyScore,
				metric.TotalSamples)
		}
	}
	
	// Best K analysis
	analysis := results.BestKAnalysis
	fmt.Printf("\n🏆 BEST K VALUES:\n")
	fmt.Printf("   Best average rank: k=%d (%.2f)\n", analysis.BestKByAvgRank, 
		results.AggregatedMetrics[analysis.BestKByAvgRank].AvgTargetRank)
	fmt.Printf("   Best perfect rate: k=%d (%.1f%%)\n", analysis.BestKByPerfectRate,
		results.AggregatedMetrics[analysis.BestKByPerfectRate].AvgPerfectMatchRate*100)
	fmt.Printf("   Most consistent: k=%d (score: %.3f)\n", analysis.BestKByConsistency,
		results.AggregatedMetrics[analysis.BestKByConsistency].ConsistencyScore)
	fmt.Printf("   🎯 RECOMMENDED: k=%d (efficiency score: %.3f)\n", 
		analysis.RecommendedK, analysis.RecommendedKScore)
	
	fmt.Println("\n💡 INTERPRETATION:")
	fmt.Println("   • Lower average rank = better move selection quality")
	fmt.Println("   • Higher perfect% = more often SCAR makes same choice as Louvain")
	fmt.Println("   • Higher consistency = more reliable across different runs")
	fmt.Println("   • Recommended K balances accuracy, efficiency, and consistency")
	// fmt.Println("="*80)
}

// Helper functions (materializeReferenceGraph, loadMovesFromJSONL) remain the same as before
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

	// Convert to Louvain graph
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
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/utils"
)

// DebugSession holds the complete state for interactive debugging
type DebugSession struct {
	// Graph data
	MaterializedGraph *louvain.Graph
	NodeMapping       map[string]int
	ReverseMapping    map[int]string
	
	// Algorithm states
	LouvainComm     *louvain.Community
	ScarGraphs      map[int]*scar.SketchGraph  // k -> SketchGraph
	ScarComms       map[int]*scar.Community    // k -> Community
	ScarMappings    map[int]*scar.NodeMapping  // k -> NodeMapping
	
	// Move sequence
	Moves           []utils.MoveEvent
	CurrentMoveIdx  int
	
	// Configuration
	KValues         []int
	DebugSeed       int64
	
	// File logging
	LogFile         *os.File
}

// CommunityDistributionData holds the comparison data for visualization
type CommunityDistributionData struct {
	MoveIdx              int                    `json:"move_idx"`
	KValue               int                    `json:"k_value"`
	NumCommonCommunities int                    `json:"num_common_communities"`
	LouvainWeights       []float64              `json:"louvain_weights"`
	ScarWeights          []float64              `json:"scar_weights"`
	CommunityIDs         []int                  `json:"community_ids"`
	DistributionStats    DistributionStats      `json:"distribution_stats"`
	WeightComparisons    []CommunityComparison  `json:"weight_comparisons"`
}

// DistributionStats provides statistical comparison
type DistributionStats struct {
	LouvainStats    BasicStats  `json:"louvain_stats"`
	ScarStats       BasicStats  `json:"scar_stats"`
	CorrelationCoef float64     `json:"correlation_coefficient"`
	KLDivergence    float64     `json:"kl_divergence"`
	JensenShannon   float64     `json:"jensen_shannon"`
	CosineSimilarity float64    `json:"cosine_similarity"`
}

// BasicStats for a distribution
type BasicStats struct {
	Mean       float64 `json:"mean"`
	Median     float64 `json:"median"`
	StdDev     float64 `json:"std_dev"`
	Min        float64 `json:"min"`
	Max        float64 `json:"max"`
	Skewness   float64 `json:"skewness"`
	Kurtosis   float64 `json:"kurtosis"`
}

// CommunityComparison for individual community analysis
type CommunityComparison struct {
	CommunityID     int     `json:"community_id"`
	LouvainWeight   float64 `json:"louvain_weight"`
	ScarWeight      float64 `json:"scar_weight"`
	AbsoluteDiff    float64 `json:"absolute_diff"`
	RelativeDiff    float64 `json:"relative_diff"`
	CommunitySize   int     `json:"community_size"`
	ErrorMagnitude  string  `json:"error_magnitude"`
}

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("📊 COMMUNITY WEIGHT DISTRIBUTION DEBUGGER")
	fmt.Println("==========================================")
	fmt.Println("Compare Louvain vs SCAR community weight estimations")
	
	// Initialize debug session
	session, err := NewDebugSession(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to initialize debug session: %v", err)
	}
	defer session.Close()
	
	fmt.Printf("✅ Loaded graph: %d nodes, %d moves\n", session.MaterializedGraph.NumNodes, len(session.Moves))
	fmt.Printf("📊 SCAR k values: %v\n", session.KValues)
	fmt.Printf("📝 Logging to: community_debugger.log\n")
	
	// Start interactive session
	session.RunInteractiveSession()
}

func NewDebugSession(graphFile, propertiesFile, pathFile string) (*DebugSession, error) {
	session := &DebugSession{
		KValues:   []int{2, 16, 64, 128, 256, 512, 1024}, // Default k values to test
		DebugSeed: 42,
	}
	
	// Initialize log file
	logFile, err := os.Create("community_debugger.log")
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %w", err)
	}
	session.LogFile = logFile
	
	session.log("=== COMMUNITY WEIGHT DISTRIBUTION DEBUGGER SESSION STARTED ===")
	
	// Step 1: Load materialized graph
	session.log("Loading materialized graph...")
	materializedGraph, nodeMapping, err := materializeReferenceGraph(graphFile, propertiesFile, pathFile)
	if err != nil {
		return nil, fmt.Errorf("failed to materialize graph: %w", err)
	}
	session.MaterializedGraph = materializedGraph
	session.NodeMapping = nodeMapping
	session.log(fmt.Sprintf("✅ Materialized graph: %d nodes, %.2f total weight", 
		materializedGraph.NumNodes, materializedGraph.TotalWeight))
	
	// Build reverse mapping
	session.ReverseMapping = make(map[int]string)
	for originalID, idx := range nodeMapping {
		session.ReverseMapping[idx] = originalID
	}
	
	// Step 2: Generate Louvain moves
	session.log("Generating Louvain move sequence...")
	moves, err := generateLouvainMoves(materializedGraph, session.DebugSeed)
	if err != nil {
		return nil, fmt.Errorf("failed to generate moves: %w", err)
	}
	session.Moves = moves
	session.log(fmt.Sprintf("✅ Generated %d moves", len(moves)))
	
	// Step 3: Initialize algorithm states
	session.log("Initializing algorithm states...")
	if err := session.initializeStates(graphFile, propertiesFile, pathFile); err != nil {
		return nil, fmt.Errorf("failed to initialize states: %w", err)
	}
	session.log("✅ Algorithm states initialized")
	
	return session, nil
}

func (s *DebugSession) log(message string) {
	fmt.Println(message)
	if s.LogFile != nil {
		timestamp := time.Now().Format("15:04:05")
		fmt.Fprintf(s.LogFile, "[%s] %s\n", timestamp, message)
	}
}

func (s *DebugSession) Close() {
	if s.LogFile != nil {
		s.log("=== COMMUNITY WEIGHT DISTRIBUTION DEBUGGER SESSION ENDED ===")
		s.LogFile.Close()
	}
}

func (s *DebugSession) initializeStates(graphFile, propertiesFile, pathFile string) error {
	// Initialize Louvain
	s.LouvainComm = louvain.NewCommunity(s.MaterializedGraph)
	
	// Initialize SCAR for different k values
	s.ScarGraphs = make(map[int]*scar.SketchGraph)
	s.ScarComms = make(map[int]*scar.Community)
	s.ScarMappings = make(map[int]*scar.NodeMapping)
	
	for _, k := range s.KValues {
		config := scar.NewConfig()
		config.Set("algorithm.random_seed", s.DebugSeed)
		config.Set("scar.k", int64(k))
		config.Set("scar.nk", int64(1))
		
		sketchGraph, scarNodeMapping, err := scar.BuildSketchGraph(graphFile, propertiesFile, pathFile, config, 
			config.CreateLogger())
		if err != nil {
			return fmt.Errorf("failed to build SCAR graph for k=%d: %w", k, err)
		}
		
		s.ScarGraphs[k] = sketchGraph
		s.ScarComms[k] = scar.NewCommunity(sketchGraph)
		s.ScarMappings[k] = scarNodeMapping
	}
	
	return nil
}

func (s *DebugSession) RunInteractiveSession() {
	scanner := bufio.NewScanner(os.Stdin)
	
	fmt.Println("\n🎮 INTERACTIVE MODE")
	fmt.Println("Core Commands (Community Weight Distribution Focus):")
	fmt.Println("  step [n]              - Execute next n moves (default: 1)")
	fmt.Println("  goto <move>           - Jump to specific move number")
	fmt.Println("  compare k=<k>         - Compare weight distributions for k value")
	fmt.Println("  compare-all           - Compare distributions for all k values")
	fmt.Println("  export k=<k>          - Export distribution data to JSON")
	fmt.Println("  stats k=<k>           - Show detailed distribution statistics")
	fmt.Println("  sketches k=<k>        - Show current community sketch states")
	fmt.Println("  sync                  - Verify Louvain/SCAR synchronization")
	fmt.Println("  status                - Show current session status")
	fmt.Println("  help                  - Show this help")
	fmt.Println("  quit                  - Exit")
	fmt.Printf("📝 All commands logged to: community_debugger.log\n")
	fmt.Println()
	
	for {
		// Show current status in prompt
		nextMoveInfo := ""
		if s.CurrentMoveIdx < len(s.Moves) {
			nextMove := s.Moves[s.CurrentMoveIdx]
			nextMoveInfo = fmt.Sprintf(" next:N%d→C%d", nextMove.Node, nextMove.ToComm)
		}
		
		fmt.Printf("move[%d/%d%s]> ", s.CurrentMoveIdx, len(s.Moves), nextMoveInfo)
		
		if !scanner.Scan() {
			break
		}
		
		command := strings.TrimSpace(scanner.Text())
		if command == "" {
			continue
		}
		
		s.log(fmt.Sprintf("USER COMMAND: %s", command))
		
		if err := s.executeCommand(command); err != nil {
			fmt.Printf("Error: %v\n", err)
			s.log(fmt.Sprintf("ERROR: %v", err))
		}
	}
}

func (s *DebugSession) executeCommand(command string) error {
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return nil
	}
	
	switch parts[0] {
	case "step":
		steps := 1
		if len(parts) > 1 {
			if n, err := strconv.Atoi(parts[1]); err == nil {
				steps = n
			}
		}
		return s.executeSteps(steps)
		
	case "goto":
		if len(parts) < 2 {
			return fmt.Errorf("usage: goto <move_number>")
		}
		moveIdx, err := strconv.Atoi(parts[1])
		if err != nil {
			return fmt.Errorf("invalid move number: %s", parts[1])
		}
		return s.gotoMove(moveIdx)
		
	case "compare":
		if len(parts) < 2 || !strings.HasPrefix(parts[1], "k=") {
			return fmt.Errorf("usage: compare k=<k_value>")
		}
		kStr := strings.TrimPrefix(parts[1], "k=")
		k, err := strconv.Atoi(kStr)
		if err != nil {
			return fmt.Errorf("invalid k value: %s", kStr)
		}
		return s.compareDistribution(k)
		
	case "compare-all":
		return s.compareAllDistributions()
		
	case "export":
		if len(parts) < 2 || !strings.HasPrefix(parts[1], "k=") {
			return fmt.Errorf("usage: export k=<k_value>")
		}
		kStr := strings.TrimPrefix(parts[1], "k=")
		k, err := strconv.Atoi(kStr)
		if err != nil {
			return fmt.Errorf("invalid k value: %s", kStr)
		}
		return s.exportDistribution(k)
		
	case "stats":
		if len(parts) < 2 || !strings.HasPrefix(parts[1], "k=") {
			return fmt.Errorf("usage: stats k=<k_value>")
		}
		kStr := strings.TrimPrefix(parts[1], "k=")
		k, err := strconv.Atoi(kStr)
		if err != nil {
			return fmt.Errorf("invalid k value: %s", kStr)
		}
		return s.showDetailedStats(k)
		
	case "sketches":
		if len(parts) < 2 || !strings.HasPrefix(parts[1], "k=") {
			return fmt.Errorf("usage: sketches k=<k_value>")
		}
		kStr := strings.TrimPrefix(parts[1], "k=")
		k, err := strconv.Atoi(kStr)
		if err != nil {
			return fmt.Errorf("invalid k value: %s", kStr)
		}
		return s.showCommunitySketchStates(k)
		
	case "sync":
		return s.checkSynchronization()
		
	case "status":
		return s.showStatus()
		
	case "help":
		fmt.Println("\n📊 COMMUNITY WEIGHT DISTRIBUTION DEBUGGER")
		fmt.Println("Core Commands:")
		fmt.Println("  step [n]              - Execute next n moves")
		fmt.Println("  goto <move>           - Jump to specific move")
		fmt.Println("  compare k=<k>         - Compare weight distributions")
		fmt.Println("  compare-all           - Compare all k values")
		fmt.Println("  export k=<k>          - Export distribution data")
		fmt.Println("  stats k=<k>           - Detailed distribution statistics")
		fmt.Println("  sketches k=<k>        - Show community sketch states")
		fmt.Println("  sync                  - Check synchronization")
		fmt.Println("  status                - Session status")
		return nil
		
	case "quit", "exit":
		fmt.Println("Goodbye!")
		os.Exit(0)
		
	default:
		return fmt.Errorf("unknown command: %s (type 'help' for available commands)", parts[0])
	}
	
	return nil
}

// === CORE DISTRIBUTION COMPARISON METHODS ===

func (s *DebugSession) compareDistribution(k int) error {
	_, exists := s.ScarGraphs[k]
	if !exists {
		return fmt.Errorf("k=%d not available in current session", k)
	}
	
	fmt.Printf("\n📊 COMMUNITY WEIGHT DISTRIBUTION COMPARISON (k=%d, Move %d)\n", k, s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 80))
	
	s.log(fmt.Sprintf("=== DISTRIBUTION COMPARISON (k=%d, Move %d) ===", k, s.CurrentMoveIdx))
	
	// Collect distribution data
	distData, err := s.collectDistributionData(k)
	if err != nil {
		return fmt.Errorf("failed to collect distribution data: %w", err)
	}
	
	// Display the comparison
	s.displayDistributionComparison(distData)
	
	return nil
}

func (s *DebugSession) collectDistributionData(k int) (*CommunityDistributionData, error) {
	scarGraph := s.ScarGraphs[k]
	scarComm := s.ScarComms[k]
	louvainComm := s.LouvainComm
	
	// Find common active communities
	louvainActive := s.getActiveCommunityIDs(louvainComm)
	scarActive := s.getActiveCommunityIDsScar(scarComm)
	commonCommunities := intersection(louvainActive, scarActive)
	
	if len(commonCommunities) == 0 {
		return nil, fmt.Errorf("no common active communities found")
	}
	
	// Collect weights for common communities
	louvainWeights := make([]float64, len(commonCommunities))
	scarWeights := make([]float64, len(commonCommunities))
	communityComparisons := make([]CommunityComparison, len(commonCommunities))
	
	for i, commID := range commonCommunities {
		// Louvain exact weight
		louvainWeight := louvainComm.CommunityWeights[commID]
		
		// SCAR estimated weight
		scarWeight := scarGraph.EstimateCommunityCardinality(commID, scarComm)
		
		// Store weights
		louvainWeights[i] = louvainWeight
		scarWeights[i] = scarWeight
		
		// Calculate comparison metrics
		absDiff := math.Abs(scarWeight - louvainWeight)
		relDiff := 0.0
		if louvainWeight != 0 {
			relDiff = absDiff / louvainWeight * 100.0
		}
		
		// Determine error magnitude
		errorMagnitude := "EXCELLENT"
		if relDiff > 50.0 {
			errorMagnitude = "CRITICAL"
		} else if relDiff > 20.0 {
			errorMagnitude = "HIGH"
		} else if relDiff > 10.0 {
			errorMagnitude = "MODERATE"
		} else if relDiff > 5.0 {
			errorMagnitude = "LOW"
		}
		
		communityComparisons[i] = CommunityComparison{
			CommunityID:   commID,
			LouvainWeight: louvainWeight,
			ScarWeight:    scarWeight,
			AbsoluteDiff:  absDiff,
			RelativeDiff:  relDiff,
			CommunitySize: len(louvainComm.CommunityNodes[commID]),
			ErrorMagnitude: errorMagnitude,
		}
	}
	
	// Calculate distribution statistics
	distStats := s.calculateDistributionStats(louvainWeights, scarWeights)
	
	return &CommunityDistributionData{
		MoveIdx:              s.CurrentMoveIdx,
		KValue:               k,
		NumCommonCommunities: len(commonCommunities),
		LouvainWeights:       louvainWeights,
		ScarWeights:          scarWeights,
		CommunityIDs:         commonCommunities,
		DistributionStats:    distStats,
		WeightComparisons:    communityComparisons,
	}, nil
}

func (s *DebugSession) displayDistributionComparison(data *CommunityDistributionData) {
	fmt.Printf("📈 DISTRIBUTION OVERVIEW:\n")
	fmt.Printf("  Communities Compared: %d\n", data.NumCommonCommunities)
	fmt.Printf("  K Value: %d\n", data.KValue)
	fmt.Printf("  Move: %d\n", data.MoveIdx)
	
	// Statistical summary
	stats := data.DistributionStats
	fmt.Printf("\n📊 STATISTICAL SUMMARY:\n")
	fmt.Printf("  Correlation: %.4f\n", stats.CorrelationCoef)
	fmt.Printf("  KL Divergence: %.6f\n", stats.KLDivergence)
	fmt.Printf("  Jensen-Shannon: %.6f\n", stats.JensenShannon)
	fmt.Printf("  Cosine Similarity: %.4f\n", stats.CosineSimilarity)
	
	// Distribution comparison table
	fmt.Printf("\n📋 WEIGHT COMPARISON (Top 15 by Error):\n")
	fmt.Printf("%-6s %-15s %-15s %-12s %-10s %-6s %s\n", 
		"Comm", "Louvain", "SCAR", "Abs Diff", "Rel Diff%", "Size", "Error")
	fmt.Println(strings.Repeat("-", 80))
	
	// Sort by relative difference for better analysis
	sortedComparisons := make([]CommunityComparison, len(data.WeightComparisons))
	copy(sortedComparisons, data.WeightComparisons)
	sort.Slice(sortedComparisons, func(i, j int) bool {
		return sortedComparisons[i].RelativeDiff > sortedComparisons[j].RelativeDiff
	})
	
	// Show worst errors first
	maxShow := min(15, len(sortedComparisons))
	for i := 0; i < maxShow; i++ {
		comp := sortedComparisons[i]
		status := "✅"
		if comp.ErrorMagnitude == "CRITICAL" {
			status = "🔴"
		} else if comp.ErrorMagnitude == "HIGH" {
			status = "🟠"
		} else if comp.ErrorMagnitude == "MODERATE" {
			status = "🟡"
		}
		
		fmt.Printf("%-6d %-15.6f %-15.6f %-12.6f %-10.1f %-6d %s %s\n",
			comp.CommunityID, comp.LouvainWeight, comp.ScarWeight, 
			comp.AbsoluteDiff, comp.RelativeDiff, comp.CommunitySize,
			status, comp.ErrorMagnitude)
	}
	
	if len(sortedComparisons) > maxShow {
		fmt.Printf("... and %d more communities\n", len(sortedComparisons) - maxShow)
	}
	
	// Error analysis
	s.analyzeDistributionErrors(data)
}

func (s *DebugSession) analyzeDistributionErrors(data *CommunityDistributionData) {
	fmt.Printf("\n🔍 ERROR ANALYSIS:\n")
	
	// Count errors by magnitude
	errorCounts := make(map[string]int)
	totalError := 0.0
	maxError := 0.0
	maxErrorComm := -1
	
	for _, comp := range data.WeightComparisons {
		errorCounts[comp.ErrorMagnitude]++
		totalError += comp.AbsoluteDiff
		if comp.AbsoluteDiff > maxError {
			maxError = comp.AbsoluteDiff
			maxErrorComm = comp.CommunityID
		}
	}
	
	fmt.Printf("  Error Distribution:\n")
	for _, level := range []string{"EXCELLENT", "LOW", "MODERATE", "HIGH", "CRITICAL"} {
		count := errorCounts[level]
		if count > 0 {
			percentage := float64(count) / float64(len(data.WeightComparisons)) * 100.0
			status := "✅"
			if level == "CRITICAL" {
				status = "🔴"
			} else if level == "HIGH" {
				status = "🟠"
			}
			fmt.Printf("    %s %s: %d communities (%.1f%%)\n", status, level, count, percentage)
		}
	}
	
	fmt.Printf("  Total Absolute Error: %.6f\n", totalError)
	fmt.Printf("  Average Error: %.6f\n", totalError/float64(len(data.WeightComparisons)))
	fmt.Printf("  Worst Error: %.6f (Community %d)\n", maxError, maxErrorComm)
	
	// Recommendations
	fmt.Printf("\n💡 RECOMMENDATIONS:\n")
	if errorCounts["CRITICAL"] > 0 {
		fmt.Printf("  🔴 CRITICAL: %d communities have >50%% error - sketch may be unreliable\n", errorCounts["CRITICAL"])
	}
	if errorCounts["HIGH"] > len(data.WeightComparisons)/4 {
		fmt.Printf("  🟠 HIGH: Many communities show significant error - consider larger k\n")
	}
	if data.DistributionStats.CorrelationCoef < 0.8 {
		fmt.Printf("  ⚠️  LOW CORRELATION: %.4f - distributions are quite different\n", data.DistributionStats.CorrelationCoef)
	}
	if errorCounts["EXCELLENT"] > len(data.WeightComparisons)*3/4 {
		fmt.Printf("  ✅ GOOD: Most communities show excellent estimation accuracy\n")
	}
}

func (s *DebugSession) compareAllDistributions() error {
	fmt.Printf("\n📊 DISTRIBUTION COMPARISON FOR ALL K VALUES (Move %d)\n", s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 80))
	
	fmt.Printf("%-6s %-10s %-12s %-12s %-12s %-12s\n", 
		"K", "NumComm", "Correlation", "KL-Div", "Jensen-S", "Cosine")
	fmt.Println(strings.Repeat("-", 70))
	
	for _, k := range s.KValues {
		distData, err := s.collectDistributionData(k)
		if err != nil {
			fmt.Printf("%-6d ERROR: %v\n", k, err)
			continue
		}
		
		stats := distData.DistributionStats
		status := "✅"
		if stats.CorrelationCoef < 0.8 {
			status = "⚠️"
		}
		if stats.CorrelationCoef < 0.6 {
			status = "🔴"
		}
		
		fmt.Printf("%-6d %-10d %-12.4f %-12.6f %-12.6f %-12.4f %s\n",
			k, distData.NumCommonCommunities, stats.CorrelationCoef, 
			stats.KLDivergence, stats.JensenShannon, stats.CosineSimilarity, status)
	}
	
	fmt.Printf("\n💡 SUMMARY:\n")
	fmt.Printf("  ✅ Correlation > 0.8: Good approximation\n")
	fmt.Printf("  ⚠️  Correlation 0.6-0.8: Moderate approximation\n")
	fmt.Printf("  🔴 Correlation < 0.6: Poor approximation\n")
	
	return nil
}

func (s *DebugSession) exportDistribution(k int) error {
	distData, err := s.collectDistributionData(k)
	if err != nil {
		return err
	}
	
	filename := fmt.Sprintf("community_distribution_k%d_move%d.json", k, s.CurrentMoveIdx)
	
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filename, err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(distData); err != nil {
		return fmt.Errorf("failed to encode data: %w", err)
	}
	
	fmt.Printf("✅ Distribution data exported to: %s\n", filename)
	s.log(fmt.Sprintf("Exported distribution data to: %s", filename))
	
	return nil
}

func (s *DebugSession) showDetailedStats(k int) error {
	distData, err := s.collectDistributionData(k)
	if err != nil {
		return err
	}
	
	fmt.Printf("\n📈 DETAILED DISTRIBUTION STATISTICS (k=%d, Move %d)\n", k, s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 70))
	
	stats := distData.DistributionStats
	
	// Louvain statistics
	fmt.Printf("🎯 LOUVAIN DISTRIBUTION:\n")
	fmt.Printf("  Mean: %.6f\n", stats.LouvainStats.Mean)
	fmt.Printf("  Median: %.6f\n", stats.LouvainStats.Median)
	fmt.Printf("  Std Dev: %.6f\n", stats.LouvainStats.StdDev)
	fmt.Printf("  Range: [%.6f - %.6f]\n", stats.LouvainStats.Min, stats.LouvainStats.Max)
	fmt.Printf("  Skewness: %.4f\n", stats.LouvainStats.Skewness)
	fmt.Printf("  Kurtosis: %.4f\n", stats.LouvainStats.Kurtosis)
	
	// SCAR statistics
	fmt.Printf("\n🎨 SCAR DISTRIBUTION (k=%d):\n", k)
	fmt.Printf("  Mean: %.6f\n", stats.ScarStats.Mean)
	fmt.Printf("  Median: %.6f\n", stats.ScarStats.Median)
	fmt.Printf("  Std Dev: %.6f\n", stats.ScarStats.StdDev)
	fmt.Printf("  Range: [%.6f - %.6f]\n", stats.ScarStats.Min, stats.ScarStats.Max)
	fmt.Printf("  Skewness: %.4f\n", stats.ScarStats.Skewness)
	fmt.Printf("  Kurtosis: %.4f\n", stats.ScarStats.Kurtosis)
	
	// Comparison metrics
	fmt.Printf("\n📊 COMPARISON METRICS:\n")
	fmt.Printf("  Pearson Correlation: %.6f\n", stats.CorrelationCoef)
	fmt.Printf("  KL Divergence: %.6f\n", stats.KLDivergence)
	fmt.Printf("  Jensen-Shannon: %.6f\n", stats.JensenShannon)
	fmt.Printf("  Cosine Similarity: %.6f\n", stats.CosineSimilarity)
	
	// Interpretation
	fmt.Printf("\n🔍 INTERPRETATION:\n")
	if stats.CorrelationCoef > 0.95 {
		fmt.Printf("  ✅ EXCELLENT correlation - SCAR closely matches Louvain\n")
	} else if stats.CorrelationCoef > 0.8 {
		fmt.Printf("  ✅ GOOD correlation - SCAR provides reasonable approximation\n")
	} else if stats.CorrelationCoef > 0.6 {
		fmt.Printf("  ⚠️  MODERATE correlation - some discrepancies present\n")
	} else {
		fmt.Printf("  🔴 POOR correlation - significant differences detected\n")
	}
	
	if stats.KLDivergence < 0.1 {
		fmt.Printf("  ✅ Low KL divergence - distributions are very similar\n")
	} else if stats.KLDivergence > 1.0 {
		fmt.Printf("  🔴 High KL divergence - distributions are quite different\n")
	}
	
	return nil
}

// NEW: Fixed sketch state display
func (s *DebugSession) showCommunitySketchStates(k int) error {
	scarGraph, exists := s.ScarGraphs[k]
	if !exists {
		return fmt.Errorf("k=%d not available", k)
	}
	
	scarComm := s.ScarComms[k]
	
	fmt.Printf("\n🎨 COMMUNITY SKETCH STATES (k=%d, Move %d)\n", k, s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 70))
	
	// Get active communities and communities with sketches
	activeCommunities := s.getActiveCommunityIDsScar(scarComm)
	// sketchCommunities := scarComm.GetAllCommunitySketchIDs()
	totalWithSketches := scarComm.GetCommunitySketchCount()
	
	fmt.Printf("Active Communities: %d\n", len(activeCommunities))
	fmt.Printf("Communities with Sketches: %d\n", totalWithSketches)
	fmt.Printf("K Value: %d\n", k)
	
	// Show detailed community sketch information
	fmt.Printf("\n📊 DETAILED COMMUNITY SKETCH STATUS:\n")
	fmt.Printf("%-6s %-8s %-12s %-12s %-8s %-15s %s\n", 
		"Comm", "Size", "HasSketch", "IsFull", "Filled", "EstWeight", "Status")
	fmt.Println(strings.Repeat("-", 80))
	
	maxShow := len(activeCommunities)
	for i := 0; i < maxShow; i++ {
		commID := activeCommunities[i]
		size := len(scarComm.CommunityNodes[commID])
		estWeight := scarGraph.EstimateCommunityCardinality(commID, scarComm)
		
		// Now we can access community sketch info using the new methods!
		hasSketch := scarComm.HasCommunitySketch(commID)
		isFull := false
		filledCount := int64(0)
		
		if hasSketch {
			sketch := scarComm.GetCommunitySketch(commID)
			if sketch != nil {
				isFull = sketch.IsSketchFull()
				filledCount = sketch.GetFilledCount()
			}
		}
		
		status := "✅ FULL"
		if !hasSketch {
			status = "❌ NO SKETCH"
		} else if !isFull {
			status = "⚠️ PARTIAL"
		}
		
		fmt.Printf("%-6d %-8d %-12v %-12v %-8d %-15.6f %s\n",
			commID, size, hasSketch, isFull, filledCount, estWeight, status)
	}
	
	if len(activeCommunities) > maxShow {
		fmt.Printf("... and %d more communities\n", len(activeCommunities)-maxShow)
	}
	
	// Summary statistics
	withSketches := 0
	fullSketches := 0
	partialSketches := 0
	
	for _, commID := range activeCommunities {
		if scarComm.HasCommunitySketch(commID) {
			withSketches++
			sketch := scarComm.GetCommunitySketch(commID)
			if sketch != nil && sketch.IsSketchFull() {
				fullSketches++
			} else {
				partialSketches++
			}
		}
	}
	
	fmt.Printf("\n📈 SKETCH SUMMARY:\n")
	fmt.Printf("  Total communities: %d\n", len(activeCommunities))
	fmt.Printf("  With sketches: %d (%.1f%%)\n", 
		withSketches, float64(withSketches)/float64(len(activeCommunities))*100)
	fmt.Printf("  Full sketches: %d (%.1f%%)\n", 
		fullSketches, float64(fullSketches)/float64(max(withSketches, 1))*100)
	fmt.Printf("  Partial sketches: %d (%.1f%%)\n", 
		partialSketches, float64(partialSketches)/float64(max(withSketches, 1))*100)
	
	// Quality assessment
	fmt.Printf("\n🔍 SKETCH QUALITY ASSESSMENT:\n")
	fullRatio := float64(fullSketches) / float64(max(withSketches, 1))
	if fullRatio > 0.8 {
		fmt.Printf("  ✅ EXCELLENT: %.1f%% of sketches are full - high estimation accuracy expected\n", fullRatio*100)
	} else if fullRatio > 0.5 {
		fmt.Printf("  ⚠️ MODERATE: %.1f%% of sketches are full - mixed estimation accuracy\n", fullRatio*100)
	} else {
		fmt.Printf("  🔴 POOR: %.1f%% of sketches are full - low estimation accuracy expected\n", fullRatio*100)
	}
	
	// Export detailed sketch information to file
	debugFilename := fmt.Sprintf("community_sketches_detailed_k%d_move%d.txt", k, s.CurrentMoveIdx)
	debugFile, err := os.Create(debugFilename)
	if err != nil {
		fmt.Printf("Warning: Could not create debug file: %v\n", err)
	} else {
		defer debugFile.Close()
		
		fmt.Fprintf(debugFile, "COMMUNITY SKETCHES DETAILED DEBUG (k=%d, Move %d)\n", k, s.CurrentMoveIdx)
		fmt.Fprintf(debugFile, "=====================================================\n")
		fmt.Fprintf(debugFile, "Timestamp: %s\n\n", time.Now().Format("2006-01-02 15:04:05"))
		
		fmt.Fprintf(debugFile, "SUMMARY STATISTICS:\n")
		fmt.Fprintf(debugFile, "==================\n")
		fmt.Fprintf(debugFile, "Active Communities: %d\n", len(activeCommunities))
		fmt.Fprintf(debugFile, "Communities with Sketches: %d (%.1f%%)\n", 
			withSketches, float64(withSketches)/float64(len(activeCommunities))*100)
		fmt.Fprintf(debugFile, "Full Sketches: %d (%.1f%%)\n", 
			fullSketches, float64(fullSketches)/float64(max(withSketches, 1))*100)
		fmt.Fprintf(debugFile, "Partial Sketches: %d (%.1f%%)\n\n", 
			partialSketches, float64(partialSketches)/float64(max(withSketches, 1))*100)
		
		fmt.Fprintf(debugFile, "DETAILED COMMUNITY INFORMATION:\n")
		fmt.Fprintf(debugFile, "===============================\n")
		
		for _, commID := range activeCommunities {
			size := len(scarComm.CommunityNodes[commID])
			estWeight := scarGraph.EstimateCommunityCardinality(commID, scarComm)
			hasSketch := scarComm.HasCommunitySketch(commID)
			
			fmt.Fprintf(debugFile, "\nCommunity %d:\n", commID)
			fmt.Fprintf(debugFile, "  Size: %d nodes\n", size)
			fmt.Fprintf(debugFile, "  Estimated Weight: %.6f\n", estWeight)
			fmt.Fprintf(debugFile, "  Has Sketch: %v\n", hasSketch)
			
			if hasSketch {
				sketch := scarComm.GetCommunitySketch(commID)
				if sketch != nil {
					fmt.Fprintf(debugFile, "  Sketch Full: %v\n", sketch.IsSketchFull())
					fmt.Fprintf(debugFile, "  Filled Count: %d\n", sketch.GetFilledCount())
					
					// Show sketch layer information
					for layer := int64(0); layer < sketch.GetNk(); layer++ {
						hashes := sketch.GetLayerHashes(layer)
						fmt.Fprintf(debugFile, "  Layer %d: %d hashes", layer, len(hashes))
						
						if len(hashes) > 0 {
							fmt.Fprintf(debugFile, " [")
							maxHashes := len(hashes)
							for i := 0; i < maxHashes; i++ {
								if i > 0 {
									fmt.Fprintf(debugFile, ", ")
								}
								fmt.Fprintf(debugFile, "%d", hashes[i])
							}
							fmt.Fprintf(debugFile, "]")
						}
						fmt.Fprintf(debugFile, "\n")
					}
				} else {
					fmt.Fprintf(debugFile, "  ERROR: Sketch is nil!\n")
				}
			}
		}
		
		fmt.Printf("✅ Detailed community sketch debug exported to: %s\n", debugFilename)
	}
	
	s.log(fmt.Sprintf("Community sketch analysis: %d communities, %d with sketches, %d full", 
		len(activeCommunities), withSketches, fullSketches))
	
	return nil
}

// === SESSION MANAGEMENT METHODS ===

func (s *DebugSession) executeSteps(steps int) error {
	for i := 0; i < steps && s.CurrentMoveIdx < len(s.Moves); i++ {
		move := s.Moves[s.CurrentMoveIdx]
		
		s.log(fmt.Sprintf("Executing move %d: Node %d (%s) → Community %d (gain: %.6f)", 
			s.CurrentMoveIdx, move.Node, s.ReverseMapping[move.Node], move.ToComm, move.Gain))
		
		// Apply move to Louvain
		currentLouvainComm := s.LouvainComm.NodeToCommunity[move.Node]
		louvain.MoveNode(s.MaterializedGraph, s.LouvainComm, move.Node, currentLouvainComm, move.ToComm)
		
		// Apply move to all SCAR variants
		for k, scarComm := range s.ScarComms {
			if move.Node < s.ScarGraphs[k].NumNodes {
				currentScarComm := scarComm.NodeToCommunity[move.Node]
				scar.MoveNode(s.ScarGraphs[k], scarComm, move.Node, currentScarComm, move.ToComm)
			}
		}
		
		s.CurrentMoveIdx++
		
		if steps == 1 {
			fmt.Printf("✅ Executed move %d: Node %d → Community %d (gain: %.6f)\n", 
				s.CurrentMoveIdx-1, move.Node, move.ToComm, move.Gain)
		}
	}
	
	if steps > 1 {
		fmt.Printf("✅ Executed %d moves (now at move %d)\n", steps, s.CurrentMoveIdx)
		s.log(fmt.Sprintf("✅ Executed %d moves (now at move %d)", steps, s.CurrentMoveIdx))
	}
	
	return nil
}

func (s *DebugSession) gotoMove(targetMove int) error {
	if targetMove < 0 || targetMove > len(s.Moves) {
		return fmt.Errorf("move %d out of range [0, %d]", targetMove, len(s.Moves))
	}
	
	if targetMove < s.CurrentMoveIdx {
		// Need to restart from beginning
		fmt.Printf("⏪ Resetting to beginning and replaying to move %d...\n", targetMove)
		if err := s.resetToBeginning(); err != nil {
			return err
		}
	}
	
	// Execute moves to reach target
	remaining := targetMove - s.CurrentMoveIdx
	if remaining > 0 {
		fmt.Printf("⏩ Executing %d moves to reach move %d...\n", remaining, targetMove)
		return s.executeSteps(remaining)
	}
	
	fmt.Printf("📍 Now at move %d\n", s.CurrentMoveIdx)
	return nil
}

func (s *DebugSession) resetToBeginning() error {
	s.CurrentMoveIdx = 0
	
	// Reinitialize Louvain
	s.LouvainComm = louvain.NewCommunity(s.MaterializedGraph)
	
	// Reinitialize SCAR communities
	for k, sketchGraph := range s.ScarGraphs {
		s.ScarComms[k] = scar.NewCommunity(sketchGraph)
	}
	
	return nil
}

func (s *DebugSession) checkSynchronization() error {
	fmt.Printf("\n🔄 SYNCHRONIZATION CHECK (Move %d)\n", s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 40))
	
	allSynced := true
	for _, k := range s.KValues {
		synced := s.verifySynchronization(k)
		status := "✅ SYNCED"
		if !synced {
			status = "❌ DESYNCED"
			allSynced = false
		}
		fmt.Printf("k=%d: %s\n", k, status)
	}
	
	if allSynced {
		fmt.Println("\n🎉 All SCAR variants synchronized with Louvain")
	} else {
		fmt.Println("\n⚠️  Some SCAR variants are out of sync!")
	}
	
	return nil
}

func (s *DebugSession) verifySynchronization(k int) bool {
	scarComm := s.ScarComms[k]
	louvainComm := s.LouvainComm
	
	maxNodes := len(louvainComm.NodeToCommunity)
	if len(scarComm.NodeToCommunity) < maxNodes {
		maxNodes = len(scarComm.NodeToCommunity)
	}
	
	for node := 0; node < maxNodes; node++ {
		if louvainComm.NodeToCommunity[node] != scarComm.NodeToCommunity[node] {
			return false
		}
	}
	
	return true
}

func (s *DebugSession) showStatus() error {
	fmt.Printf("\n📋 SESSION STATUS\n")
	fmt.Println(strings.Repeat("=", 30))
	fmt.Printf("Current Move: %d/%d\n", s.CurrentMoveIdx, len(s.Moves))
	fmt.Printf("Graph Nodes: %d\n", s.MaterializedGraph.NumNodes)
	fmt.Printf("Graph Edges: %.0f\n", s.MaterializedGraph.TotalWeight)
	fmt.Printf("K Values: %v\n", s.KValues)
	fmt.Printf("Random Seed: %d\n", s.DebugSeed)
	
	// Show active communities
	louvainActive := len(s.getActiveCommunityIDs(s.LouvainComm))
	fmt.Printf("Louvain Active Communities: %d\n", louvainActive)
	
	for _, k := range s.KValues {
		scarActive := len(s.getActiveCommunityIDsScar(s.ScarComms[k]))
		fmt.Printf("SCAR k=%d Active Communities: %d\n", k, scarActive)
	}
	
	if s.CurrentMoveIdx < len(s.Moves) {
		nextMove := s.Moves[s.CurrentMoveIdx]
		fmt.Printf("\nNext Move: Node %d → Community %d (gain: %.6f)\n", 
			nextMove.Node, nextMove.ToComm, nextMove.Gain)
	} else {
		fmt.Printf("\nNo more moves available\n")
	}
	
	return nil
}

// === HELPER FUNCTIONS ===

func (s *DebugSession) getActiveCommunityIDs(comm *louvain.Community) []int {
	active := make([]int, 0)
	for commID := 0; commID < comm.NumCommunities; commID++ {
		if len(comm.CommunityNodes[commID]) > 0 {
			active = append(active, commID)
		}
	}
	sort.Ints(active)
	return active
}

func (s *DebugSession) getActiveCommunityIDsScar(comm *scar.Community) []int {
	active := make([]int, 0)
	for commID := 0; commID < comm.NumCommunities; commID++ {
		if len(comm.CommunityNodes[commID]) > 0 {
			active = append(active, commID)
		}
	}
	sort.Ints(active)
	return active
}

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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// === STATISTICAL CALCULATION METHODS ===

func (s *DebugSession) calculateDistributionStats(louvainWeights, scarWeights []float64) DistributionStats {
	return DistributionStats{
		LouvainStats:    s.calculateBasicStats(louvainWeights),
		ScarStats:       s.calculateBasicStats(scarWeights),
		CorrelationCoef: pearsonCorrelation(louvainWeights, scarWeights),
		KLDivergence:    klDivergence(normalize(louvainWeights), normalize(scarWeights)),
		JensenShannon:   jensenShannonDivergence(normalize(louvainWeights), normalize(scarWeights)),
		CosineSimilarity: cosineSimilarity(louvainWeights, scarWeights),
	}
}

func (s *DebugSession) calculateBasicStats(values []float64) BasicStats {
	if len(values) == 0 {
		return BasicStats{}
	}
	
	// Sort for median calculation
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	// Basic calculations
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	
	// Median
	median := sorted[len(sorted)/2]
	if len(sorted)%2 == 0 {
		median = (sorted[len(sorted)/2-1] + sorted[len(sorted)/2]) / 2.0
	}
	
	// Variance and standard deviation
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))
	stdDev := math.Sqrt(variance)
	
	// Skewness and kurtosis
	skewness := 0.0
	kurtosis := 0.0
	for _, v := range values {
		normalized := (v - mean) / stdDev
		skewness += normalized * normalized * normalized
		kurtosis += normalized * normalized * normalized * normalized
	}
	skewness /= float64(len(values))
	kurtosis = kurtosis/float64(len(values)) - 3.0 // Excess kurtosis
	
	return BasicStats{
		Mean:     mean,
		Median:   median,
		StdDev:   stdDev,
		Min:      sorted[0],
		Max:      sorted[len(sorted)-1],
		Skewness: skewness,
		Kurtosis: kurtosis,
	}
}

// === STATISTICAL FUNCTIONS ===
func normalize(dist []float64) []float64 {
	if len(dist) == 0 {
		return dist
	}
	
	minVal := dist[0]
	for _, v := range dist {
		if v < minVal {
			minVal = v
		}
	}
	
	sum := 0.0
	shifted := make([]float64, len(dist))
	for i, v := range dist {
		shifted[i] = v - minVal + 1e-10
		sum += shifted[i]
	}
	
	if sum == 0 {
		return shifted
	}
	
	normalized := make([]float64, len(shifted))
	for i, v := range shifted {
		normalized[i] = v / sum
	}
	
	return normalized
}

func klDivergence(p, q []float64) float64 {
	if len(p) != len(q) {
		return math.Inf(1)
	}
	
	kl := 0.0
	for i := 0; i < len(p); i++ {
		if p[i] > 0 && q[i] > 0 {
			kl += p[i] * math.Log(p[i]/q[i])
		} else if p[i] > 0 && q[i] == 0 {
			return math.Inf(1)
		}
	}
	return kl
}

func jensenShannonDivergence(p, q []float64) float64 {
	if len(p) != len(q) {
		return math.Inf(1)
	}
	
	m := make([]float64, len(p))
	for i := 0; i < len(p); i++ {
		m[i] = (p[i] + q[i]) / 2.0
	}
	
	klPM := klDivergence(p, m)
	klQM := klDivergence(q, m)
	
	if math.IsInf(klPM, 1) || math.IsInf(klQM, 1) {
		return math.Inf(1)
	}
	
	return 0.5*klPM + 0.5*klQM
}

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

func pearsonCorrelation(p, q []float64) float64 {
	if len(p) != len(q) || len(p) < 2 {
		return 0.0
	}
	
	n := float64(len(p))
	
	meanP := 0.0
	meanQ := 0.0
	for i := 0; i < len(p); i++ {
		meanP += p[i]
		meanQ += q[i]
	}
	meanP /= n
	meanQ /= n
	
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

// === GRAPH SETUP FUNCTIONS ===
func materializeReferenceGraph(graphFile, propertiesFile, pathFile string) (*louvain.Graph, map[string]int, error) {
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
	if err != nil {
		return nil, nil, fmt.Errorf("parse failed: %w", err)
	}

	config := materialization.DefaultMaterializationConfig()
	config.Aggregation.Strategy = materialization.Average
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()
	if err != nil {
		return nil, nil, fmt.Errorf("materialization failed: %w", err)
	}

	hgraph := result.HomogeneousGraph
	nodeList := make([]string, 0, len(hgraph.Nodes))
	for nodeID := range hgraph.Nodes {
		nodeList = append(nodeList, nodeID)
	}

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

	nodeMapping := make(map[string]int)
	for i, originalID := range nodeList {
		nodeMapping[originalID] = i
	}

	louvainGraph := louvain.NewGraph(len(nodeList))
	processedEdges := make(map[string]bool)

	for edgeKey, weight := range hgraph.Edges {
		fromIdx, fromExists := nodeMapping[edgeKey.From]
		toIdx, toExists := nodeMapping[edgeKey.To]

		if !fromExists || !toExists {
			continue
		}

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

func generateLouvainMoves(graph *louvain.Graph, seed int64) ([]utils.MoveEvent, error) {
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

	os.Remove(fmt.Sprintf("temp_louvain_moves_%d.jsonl", seed))
	return moves, nil
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
			continue
		}
		moves = append(moves, move)
	}

	return moves, scanner.Err()
}
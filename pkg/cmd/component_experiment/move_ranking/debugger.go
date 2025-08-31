package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
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

// EdgeWeightBreakdown - moved to package level
type EdgeWeightBreakdown struct {
	Method              string
	NodeDegree          float64
	CommunityDegree     float64
	UnionDegree         float64
	Intersection        float64
	ExactCount          float64
	SketchFull          bool
	CommunitySketchFull bool
}

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
	
	// State snapshots (for performance - cache states at key points)
	StateSnapshots  map[int]*StateSnapshot
	
	// File logging
	LogFile         *os.File
}

// StateSnapshot captures algorithm state at a specific move
type StateSnapshot struct {
	MoveIdx         int
	LouvainState    *louvain.Community
	ScarStates      map[int]*scar.Community
	CommunityStats  CommunityStats
}

// CommunityStats provides overview of community structure
type CommunityStats struct {
	NumActive           int
	LargestSize         int
	SmallestSize        int
	AverageSize         float64
	SizeDistribution    map[int]int  // size -> count
	CommunityNodeCounts []int        // ordered by community ID
}

// MoveAnalysis provides detailed analysis of a specific move
type MoveAnalysis struct {
	MoveIdx                int
	Move                   utils.MoveEvent
	BeforeStats            CommunityStats
	AfterStats             CommunityStats
	NodeContext            NodeContext
	ScarRankings          map[int]*CommunityRanking
	SynchronizationStatus  map[int]bool  // k -> is_synced
}

// NodeContext provides context about the node being moved
type NodeContext struct {
	NodeID              int
	OriginalID          string
	Degree              float64
	CurrentCommunity    int
	TargetCommunity     int
	CurrentCommSize     int
	TargetCommSize      int
	NeighborCommunities []int
	IsInterCommMove     bool
}

// CommunityRanking matches the one from your main code
type CommunityRanking struct {
	Communities      []int     `json:"communities"`
	Gains           []float64 `json:"gains"`
	TargetRank      int       `json:"target_rank"`
	NormalizedScore float64   `json:"normalized_score"`
	TargetGain      float64   `json:"target_gain"`
	BestGain        float64   `json:"best_gain"`
	WorstGain       float64   `json:"worst_gain"`
}

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("🔍 INTERACTIVE MOVE DEBUGGER")
	fmt.Println("============================")
	fmt.Println("Step-by-step analysis of Louvain vs SCAR move selection")
	
	// Initialize debug session
	session, err := NewDebugSession(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to initialize debug session: %v", err)
	}
	defer session.Close() // Ensure log file is closed
	
	fmt.Printf("✅ Loaded graph: %d nodes, %d moves\n", session.MaterializedGraph.NumNodes, len(session.Moves))
	fmt.Printf("📊 SCAR k values: %v\n", session.KValues)
	fmt.Printf("📝 Logging to: move_debugger.log\n")
	
	// Start interactive session
	session.RunInteractiveSession()
}

func NewDebugSession(graphFile, propertiesFile, pathFile string) (*DebugSession, error) {
	session := &DebugSession{
		KValues:        []int{64, 128, 256, 512}, // Default k values to test
		DebugSeed:      42,
		StateSnapshots: make(map[int]*StateSnapshot),
	}
	
	// Initialize log file
	logFile, err := os.Create("move_debugger.log")
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %w", err)
	}
	session.LogFile = logFile
	
	session.log("=== MOVE DEBUGGER SESSION STARTED ===")
	
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

// log writes a message to both console and log file
func (s *DebugSession) log(message string) {
	fmt.Println(message)
	if s.LogFile != nil {
		timestamp := time.Now().Format("15:04:05")
		fmt.Fprintf(s.LogFile, "[%s] %s\n", timestamp, message)
	}
}

// Close closes the log file
func (s *DebugSession) Close() {
	if s.LogFile != nil {
		s.log("=== MOVE DEBUGGER SESSION ENDED ===")
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
	fmt.Println("Commands:")
	fmt.Println("  step [n]           - Execute next n moves (default: 1)")
	fmt.Println("  goto <move>        - Jump to specific move number")  
	fmt.Println("  analyze            - Analyze current move in detail")
	fmt.Println("  inspect k=<k>      - Deep inspect SCAR ranking for specific k")
	fmt.Println("  community-health   - Analyze community structure health")
	fmt.Println("  debug-sketch k=<k> - Export detailed SCAR debug info to file")
	fmt.Println("  compare-gains k=<k> - Compare SCAR vs Louvain gain calculations")
	fmt.Println("  communities        - Show community statistics")
	fmt.Println("  ranking [k]        - Show SCAR ranking for k (default: all k)")
	fmt.Println("  sync               - Check Louvain/SCAR synchronization")
	fmt.Println("  snapshot           - Save current state as snapshot")
	fmt.Println("  find-problem       - Auto-detect where ranking degrades")
	fmt.Println("  help               - Show this help")
	fmt.Println("  quit               - Exit")
	fmt.Printf("📝 All commands are logged to: move_debugger.log\n")
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
		
	case "analyze":
		return s.analyzeCurrentMove()
		
	case "inspect":
		if len(parts) < 2 || !strings.HasPrefix(parts[1], "k=") {
			return fmt.Errorf("usage: inspect k=<k_value>")
		}
		kStr := strings.TrimPrefix(parts[1], "k=")
		k, err := strconv.Atoi(kStr)
		if err != nil {
			return fmt.Errorf("invalid k value: %s", kStr)
		}
		return s.deepInspectScarRanking(k)
		
	case "community-health":
		return s.analyzeCommunityHealth()
		
	case "debug-sketch":
		if len(parts) < 2 || !strings.HasPrefix(parts[1], "k=") {
			return fmt.Errorf("usage: debug-sketch k=<k_value>")
		}
		kStr := strings.TrimPrefix(parts[1], "k=")
		k, err := strconv.Atoi(kStr)
		if err != nil {
			return fmt.Errorf("invalid k value: %s", kStr)
		}
		return s.exportSketchDebugInfo(k)
		
	case "compare-gains":
		if len(parts) < 2 || !strings.HasPrefix(parts[1], "k=") {
			return fmt.Errorf("usage: compare-gains k=<k_value>")
		}
		kStr := strings.TrimPrefix(parts[1], "k=")
		k, err := strconv.Atoi(kStr)
		if err != nil {
			return fmt.Errorf("invalid k value: %s", kStr)
		}
		return s.compareGainCalculations(k)
		
	case "communities":
		return s.showCommunityStats()
		
	case "ranking":
		k := -1
		if len(parts) > 1 {
			if n, err := strconv.Atoi(parts[1]); err == nil {
				k = n
			}
		}
		return s.showRankings(k)
		
	case "sync":
		return s.checkSynchronization()
		
	case "snapshot":
		return s.saveSnapshot()
		
	case "find-problem":
		return s.findProblemArea()
		
	case "help":
		fmt.Println("\nAvailable commands:")
		fmt.Println("  step [n]           - Execute next n moves")
		fmt.Println("  goto <move>        - Jump to specific move")
		fmt.Println("  analyze            - Detailed move analysis")
		fmt.Println("  inspect k=<k>      - Deep SCAR ranking inspection")
		fmt.Println("  community-health   - Community structure analysis")
		fmt.Println("  debug-sketch k=<k> - Export SCAR debug info to file")
		fmt.Println("  compare-gains k=<k> - Compare SCAR vs Louvain gains")
		fmt.Println("  communities        - Community statistics")
		fmt.Println("  ranking [k]        - SCAR rankings")
		fmt.Println("  sync               - Check synchronization")
		fmt.Println("  find-problem       - Auto-detect issues")
		return nil
		
	case "quit", "exit":
		fmt.Println("Goodbye!")
		os.Exit(0)
		
	default:
		return fmt.Errorf("unknown command: %s (type 'help' for available commands)", parts[0])
	}
	
	return nil
}

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

func (s *DebugSession) analyzeCurrentMove() error {
	if s.CurrentMoveIdx >= len(s.Moves) {
		return fmt.Errorf("no more moves to analyze")
	}
	
	move := s.Moves[s.CurrentMoveIdx]
	
	fmt.Printf("\n📋 MOVE ANALYSIS #%d\n", s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 50))
	
	// Basic move info
	originalID := s.ReverseMapping[move.Node]
	fmt.Printf("Node: %d (original: %s)\n", move.Node, originalID)
	fmt.Printf("From Community: %d → To Community: %d\n", 
		s.LouvainComm.NodeToCommunity[move.Node], move.ToComm)
	fmt.Printf("Louvain Gain: %.6f\n", move.Gain)
	
	// Node context
	degree := s.MaterializedGraph.Degrees[move.Node]  // Use Degrees field
	currentCommSize := len(s.LouvainComm.CommunityNodes[s.LouvainComm.NodeToCommunity[move.Node]])
	targetCommSize := len(s.LouvainComm.CommunityNodes[move.ToComm])
	
	fmt.Printf("Node Degree: %.2f\n", degree)
	fmt.Printf("Current Community Size: %d\n", currentCommSize)
	fmt.Printf("Target Community Size: %d\n", targetCommSize)
	
	// Log to file
	s.log(fmt.Sprintf("ANALYSIS: Move %d - Node %d (%s) → %d, gain: %.6f, degree: %.2f", 
		s.CurrentMoveIdx, move.Node, originalID, move.ToComm, move.Gain, degree))
	
	// SCAR rankings analysis
	fmt.Printf("\n🎯 SCAR RANKINGS:\n")
	for _, k := range s.KValues {
		ranking := s.calculateScarRanking(k, move.Node, move.ToComm)
		if ranking != nil {
			status := "✅"
			if ranking.TargetRank > 1 {
				status = "❌"
			}
			rankInfo := fmt.Sprintf("k=%d: %s Rank %d/%d (gain: %.6f)", 
				k, status, ranking.TargetRank, len(ranking.Communities), ranking.TargetGain)
			fmt.Println(rankInfo)
			s.log(rankInfo)
		}
	}
	
	return nil
}

func (s *DebugSession) calculateScarRanking(k, node, targetComm int) *CommunityRanking {
	scarGraph := s.ScarGraphs[k]
	scarComm := s.ScarComms[k]
	
	if node >= scarGraph.NumNodes {
		return nil
	}
	
	type CommunityGain struct {
		CommunityID int
		Gain        float64
	}
	
	communityGains := make([]CommunityGain, 0)
	
	for commID := 0; commID < scarComm.NumCommunities; commID++ {
		if len(scarComm.CommunityNodes[commID]) > 0 {
			// Calculate modularity gain using SCAR's method
			edgeWeight := scarGraph.EstimateEdgesToCommunity(node, commID, scarComm)
			gain := scar.CalculateModularityGain(scarGraph, scarComm, node, commID, edgeWeight)
			
			communityGains = append(communityGains, CommunityGain{
				CommunityID: commID,
				Gain:        gain,
			})
		}
	}
	
	// Sort by gain (descending)
	sort.Slice(communityGains, func(i, j int) bool {
		if communityGains[i].Gain == communityGains[j].Gain {
			return communityGains[i].CommunityID < communityGains[j].CommunityID
		}
		return communityGains[i].Gain > communityGains[j].Gain
	})
	
	// Build ranking
	communities := make([]int, len(communityGains))
	gains := make([]float64, len(communityGains))
	targetRank := -1
	targetGain := 0.0
	
	for i, cg := range communityGains {
		communities[i] = cg.CommunityID
		gains[i] = cg.Gain
		
		if cg.CommunityID == targetComm {
			targetRank = i + 1
			targetGain = cg.Gain
		}
	}
	
	ranking := &CommunityRanking{
		Communities: communities,
		Gains:      gains,
		TargetRank: targetRank,
		TargetGain: targetGain,
	}
	
	if len(gains) > 0 {
		ranking.BestGain = gains[0]
		ranking.WorstGain = gains[len(gains)-1]
		
		if ranking.BestGain != ranking.WorstGain {
			ranking.NormalizedScore = (ranking.BestGain - targetGain) / (ranking.BestGain - ranking.WorstGain)
		}
	}
	
	return ranking
}

func (s *DebugSession) showCommunityStats() error {
	stats := s.calculateCommunityStats(s.LouvainComm)
	
	fmt.Printf("\n📊 COMMUNITY STATISTICS (Move %d)\n", s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 40))
	fmt.Printf("Active Communities: %d\n", stats.NumActive)
	fmt.Printf("Largest Community: %d nodes\n", stats.LargestSize)
	fmt.Printf("Smallest Community: %d nodes\n", stats.SmallestSize)
	fmt.Printf("Average Size: %.2f nodes\n", stats.AverageSize)
	
	fmt.Println("\nSize Distribution:")
	sizes := make([]int, 0, len(stats.SizeDistribution))
	for size := range stats.SizeDistribution {
		sizes = append(sizes, size)
	}
	sort.Ints(sizes)
	
	for _, size := range sizes {
		count := stats.SizeDistribution[size]
		fmt.Printf("  %d nodes: %d communities\n", size, count)
	}
	
	return nil
}

func (s *DebugSession) calculateCommunityStats(comm *louvain.Community) CommunityStats {
	stats := CommunityStats{
		SizeDistribution: make(map[int]int),
	}
	
	totalNodes := 0
	activeCommunities := 0
	minSize := int(^uint(0) >> 1) // Max int
	maxSize := 0
	
	for commID := 0; commID < comm.NumCommunities; commID++ {
		size := len(comm.CommunityNodes[commID])
		if size > 0 {
			activeCommunities++
			totalNodes += size
			
			if size < minSize {
				minSize = size
			}
			if size > maxSize {
				maxSize = size
			}
			
			stats.SizeDistribution[size]++
		}
		stats.CommunityNodeCounts = append(stats.CommunityNodeCounts, size)
	}
	
	stats.NumActive = activeCommunities
	stats.LargestSize = maxSize
	stats.SmallestSize = minSize
	
	if activeCommunities > 0 {
		stats.AverageSize = float64(totalNodes) / float64(activeCommunities)
	}
	
	return stats
}

// Overloaded version for SCAR communities
func (s *DebugSession) calculateScarCommunityStats(comm *scar.Community) CommunityStats {
	stats := CommunityStats{
		SizeDistribution: make(map[int]int),
	}
	
	totalNodes := 0
	activeCommunities := 0
	minSize := int(^uint(0) >> 1) // Max int
	maxSize := 0
	
	for commID := 0; commID < comm.NumCommunities; commID++ {
		size := len(comm.CommunityNodes[commID])
		if size > 0 {
			activeCommunities++
			totalNodes += size
			
			if size < minSize {
				minSize = size
			}
			if size > maxSize {
				maxSize = size
			}
			
			stats.SizeDistribution[size]++
		}
		stats.CommunityNodeCounts = append(stats.CommunityNodeCounts, size)
	}
	
	stats.NumActive = activeCommunities
	stats.LargestSize = maxSize
	stats.SmallestSize = minSize
	
	if activeCommunities > 0 {
		stats.AverageSize = float64(totalNodes) / float64(activeCommunities)
	}
	
	return stats
}

func (s *DebugSession) showRankings(specificK int) error {
	if s.CurrentMoveIdx >= len(s.Moves) {
		return fmt.Errorf("no move to analyze")
	}
	
	move := s.Moves[s.CurrentMoveIdx]
	
	fmt.Printf("\n🎯 SCAR RANKINGS for move %d (Node %d → Community %d)\n", 
		s.CurrentMoveIdx, move.Node, move.ToComm)
	fmt.Println(strings.Repeat("=", 60))
	
	kValues := s.KValues
	if specificK > 0 {
		kValues = []int{specificK}
	}
	
	for _, k := range kValues {
		ranking := s.calculateScarRanking(k, move.Node, move.ToComm)
		if ranking == nil {
			fmt.Printf("k=%d: Node out of range\n", k)
			continue
		}
		
		status := "✅ PERFECT"
		if ranking.TargetRank > 1 {
			status = fmt.Sprintf("❌ RANK %d", ranking.TargetRank)
		}
		
		fmt.Printf("\nk=%d: %s\n", k, status)
		fmt.Printf("  Target Community %d ranked #%d out of %d\n", 
			move.ToComm, ranking.TargetRank, len(ranking.Communities))
		fmt.Printf("  Target Gain: %.6f\n", ranking.TargetGain)
		fmt.Printf("  Best Gain:   %.6f\n", ranking.BestGain)
		fmt.Printf("  Worst Gain:  %.6f\n", ranking.WorstGain)
		
		// Show top 5 communities
		fmt.Printf("  Top 5 communities: ")
		for i := 0; i < 5 && i < len(ranking.Communities); i++ {
			marker := ""
			if ranking.Communities[i] == move.ToComm {
				marker = "*"
			}
			fmt.Printf("%d%s(%.4f) ", ranking.Communities[i], marker, ranking.Gains[i])
		}
		fmt.Println()
	}
	
	return nil
}

func (s *DebugSession) checkSynchronization() error {
	fmt.Printf("\n🔄 SYNCHRONIZATION CHECK (Move %d)\n", s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 40))
	s.log(fmt.Sprintf("=== SYNCHRONIZATION CHECK (Move %d) ===", s.CurrentMoveIdx))
	
	allSynced := true
	for _, k := range s.KValues {
		synced := s.verifySynchronization(k)
		status := "✅ SYNCED"
		if !synced {
			status = "❌ DESYNCED"
			allSynced = false
		}
		fmt.Printf("k=%d: %s\n", k, status)
		s.log(fmt.Sprintf("k=%d: %s", k, status))
	}
	
	if allSynced {
		fmt.Println("\n🎉 All SCAR variants synchronized with Louvain")
		s.log("🎉 All SCAR variants synchronized with Louvain")
	} else {
		fmt.Println("\n⚠️  Some SCAR variants are out of sync!")
		s.log("⚠️  Some SCAR variants are out of sync!")
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

func (s *DebugSession) saveSnapshot() error {
	snapshot := &StateSnapshot{
		MoveIdx:         s.CurrentMoveIdx,
		CommunityStats:  s.calculateCommunityStats(s.LouvainComm),
	}
	
	s.StateSnapshots[s.CurrentMoveIdx] = snapshot
	fmt.Printf("📸 Snapshot saved at move %d\n", s.CurrentMoveIdx)
	
	return nil
}

// NEW ANALYSIS FUNCTIONS

func (s *DebugSession) deepInspectScarRanking(k int) error {
	if s.CurrentMoveIdx >= len(s.Moves) {
		return fmt.Errorf("no move to inspect")
	}
	
	scarGraph, exists := s.ScarGraphs[k]
	if !exists {
		return fmt.Errorf("k=%d not available in current session", k)
	}
	
	scarComm := s.ScarComms[k]
	move := s.Moves[s.CurrentMoveIdx]
	
	fmt.Printf("\n🔍 DEEP SCAR RANKING INSPECTION (k=%d, Move %d)\n", k, s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 70))
	
	s.log(fmt.Sprintf("=== DEEP SCAR RANKING INSPECTION (k=%d, Move %d) ===", k, s.CurrentMoveIdx))
	
	if move.Node >= scarGraph.NumNodes {
		fmt.Printf("❌ Node %d out of range for SCAR graph (max: %d)\n", move.Node, scarGraph.NumNodes-1)
		return nil
	}
	
	// Basic move context
	fmt.Printf("Move Details:\n")
	fmt.Printf("  Node: %d (original: %s)\n", move.Node, s.ReverseMapping[move.Node])
	fmt.Printf("  Louvain Target: Community %d\n", move.ToComm)
	fmt.Printf("  Louvain Gain: %.8f\n", move.Gain)
	fmt.Printf("  Node Degree: %.2f\n", s.MaterializedGraph.Degrees[move.Node])
	fmt.Printf("  SCAR Node Degree: %.2f\n", scarGraph.GetDegree(move.Node))
	
	// Get sketch manager for detailed analysis
	sketchManager := scarGraph.GetSketchManager()
	filledCount := scarGraph.GetFilledCount(int64(move.Node))
	
	fmt.Printf("\n🎯 NODE SKETCH STATUS:\n")
	if filledCount > 0 {
		fmt.Printf("  Node %d sketch: filled=%d\n", move.Node, filledCount)
	} else {
		fmt.Printf("  Node %d: No sketch found\n", move.Node)
	}
	
	// Calculate all community gains with detailed info
	type DetailedCommunityGain struct {
		CommunityID      int
		Size             int
		Gain             float64
		EdgeWeight       float64
		CommunityWeight  float64
		EstimationMethod string
		IsTarget         bool
		// Calculation components
		PenaltyTerm      float64
		NetGain          float64
		// EdgeWeight breakdown
		EdgeWeightComponents EdgeWeightBreakdown
	}
	
	totalGraphWeight := scarGraph.TotalWeight
	nodeDegree := scarGraph.GetDegree(move.Node)
	
	fmt.Printf("\n📊 MODULARITY GAIN FORMULA BREAKDOWN:\n")
	fmt.Printf("Formula: gain = edgeWeight - (nodeDegree × communityWeight) / (2 × totalGraphWeight)\n")
	fmt.Printf("Node Degree: %.6f\n", nodeDegree)
	fmt.Printf("Total Graph Weight: %.6f\n", totalGraphWeight)
	fmt.Printf("Denominator (2m): %.6f\n", 2*totalGraphWeight)
	
	// Focus on top communities and target for detailed breakdown
	fmt.Printf("\n🔬 DETAILED EDGE WEIGHT BREAKDOWN (Top 3 + Target):\n")
	fmt.Println(strings.Repeat("=", 100))
	
	// First pass: calculate basic info for all communities
	allCommunityGains := make([]DetailedCommunityGain, 0)
	
	for commID := 0; commID < scarComm.NumCommunities; commID++ {
		if len(scarComm.CommunityNodes[commID]) == 0 {
			continue
		}
		
		edgeWeight := scarGraph.EstimateEdgesToCommunity(move.Node, commID, scarComm)
		communityWeight := scarComm.CommunityWeights[commID]
		penaltyTerm := (nodeDegree * communityWeight) / (2.0 * totalGraphWeight)
		gain := edgeWeight - penaltyTerm
		
		// Basic estimation method check
		estimationMethod := "Exact"
		if filledCount > 0 {
			nodeSketch := sketchManager.GetVertexSketch(int64(move.Node))
			if nodeSketch != nil && nodeSketch.IsSketchFull() {
				estimationMethod = "Sketch"
			}
		}
		
		allCommunityGains = append(allCommunityGains, DetailedCommunityGain{
			CommunityID:      commID,
			Size:             len(scarComm.CommunityNodes[commID]),
			Gain:             gain,
			EdgeWeight:       edgeWeight,
			CommunityWeight:  communityWeight,
			EstimationMethod: estimationMethod,
			IsTarget:         (commID == move.ToComm),
			PenaltyTerm:      penaltyTerm,
			NetGain:          gain,
		})
	}
	
	// Sort by gain for ranking
	sort.Slice(allCommunityGains, func(i, j int) bool {
		if allCommunityGains[i].Gain == allCommunityGains[j].Gain {
			return allCommunityGains[i].CommunityID < allCommunityGains[j].CommunityID
		}
		return allCommunityGains[i].Gain > allCommunityGains[j].Gain
	})
	
	// Get detailed breakdown for top communities and target
	communitiesToAnalyze := make(map[int]bool)
	
	// Add top 3
	for i := 0; i < min(3, len(allCommunityGains)); i++ {
		communitiesToAnalyze[allCommunityGains[i].CommunityID] = true
	}
	
	// Add target community
	targetCommunityID := move.ToComm
	communitiesToAnalyze[targetCommunityID] = true
	
	// Now get detailed breakdown for selected communities
	for commID := range communitiesToAnalyze {
		if len(scarComm.CommunityNodes[commID]) == 0 {
			continue
		}
		
		fmt.Printf("\n🔍 COMMUNITY %d EDGE WEIGHT ANALYSIS:\n", commID)
		fmt.Printf("  Community Size: %d nodes\n", len(scarComm.CommunityNodes[commID]))
		fmt.Printf("  Community Weight: %.2f\n", scarComm.CommunityWeights[commID])
		
		// Get community sketch info using the debug method
		communitySketches := scarGraph.GetCommunitySketches()
		communitySketch, hasSketch := communitySketches[commID]
		fmt.Printf("  Community Sketch: ")
		if hasSketch && communitySketch != nil {
			communityFilledCount := scarGraph.GetFilledCount(int64(commID))
			fmt.Printf("filled=%d, full=%v\n", communityFilledCount, communitySketch.IsSketchFull())
		} else {
			fmt.Printf("No sketch\n")
		}
		
		// Detailed edge weight calculation
		s.analyzeEdgeWeightCalculation(scarGraph, scarComm, move.Node, commID)
		
		// Find this community in our gains list
		for i, cg := range allCommunityGains {
			if cg.CommunityID == commID {
				rank := i + 1
				marker := ""
				if cg.IsTarget {
					marker = " 🎯 TARGET"
				}
				if rank == 1 {
					marker += " 🥇 BEST"
				}
				
				fmt.Printf("  Final Rank: #%d%s\n", rank, marker)
				fmt.Printf("  Final Gain: %.8f\n", cg.Gain)
				break
			}
		}
	}
	
	// Show condensed ranking table
	fmt.Printf("\nCommunity Gain Summary:\n")
	fmt.Printf("%-4s %-6s %-12s %-12s %-12s %-12s %-10s %-8s %s\n", 
		"Rank", "CommID", "EdgeWeight", "CommWeight", "Penalty", "Gain", "Method", "Size", "Notes")
	fmt.Println(strings.Repeat("-", 100))
	
	targetRank := -1
	for rank, cg := range allCommunityGains {
		rankNum := rank + 1
		notes := ""
		if cg.IsTarget {
			targetRank = rankNum
			notes = "🎯 TARGET"
		}
		if rankNum == 1 {
			notes += " 🥇 BEST"
		}
		
		fmt.Printf("%-4d %-6d %-12.6f %-12.2f %-12.6f %-12.6f %-10s %-8d %s\n",
			rankNum, cg.CommunityID, cg.EdgeWeight, cg.CommunityWeight, 
			cg.PenaltyTerm, cg.Gain, cg.EstimationMethod, cg.Size, notes)
		
		// Log important ones
		if rankNum <= 5 || cg.IsTarget {
			s.log(fmt.Sprintf("Rank %d: Comm %d, Edge %.6f, CommWeight %.2f, Penalty %.6f, Gain %.6f %s",
				rankNum, cg.CommunityID, cg.EdgeWeight, cg.CommunityWeight, cg.PenaltyTerm, cg.Gain, notes))
		}
	}
	
	// Final analysis
	if targetRank > 0 {
		fmt.Printf("\n📊 FINAL ANALYSIS:\n")
		fmt.Printf("  🎯 Target Community %d ranked #%d out of %d\n", targetCommunityID, targetRank, len(allCommunityGains))
		
		if targetRank == 1 {
			fmt.Printf("  ✅ PERFECT MATCH!\n")
		} else {
			topChoice := allCommunityGains[0]
			fmt.Printf("  ❌ SUBOPTIMAL: SCAR prefers Community %d (gain: %.6f)\n", 
				topChoice.CommunityID, topChoice.Gain)
			fmt.Printf("  🔴 LIKELY BUG: EdgeWeight calculations appear inflated\n")
		}
	}
	
	return nil
}

// Analyze the edge weight calculation in detail
func (s *DebugSession) analyzeEdgeWeightCalculation(scarGraph *scar.SketchGraph, scarComm *scar.Community, node, commID int) {
	fmt.Printf("  📊 Edge Weight Calculation Method Detection:\n")
	
	// Check what method should be used
	filledCount := scarGraph.GetFilledCount(int64(node))
	shouldUseSketch := false
	if filledCount > 0 {
		sketchManager := scarGraph.GetSketchManager()
		nodeSketch := sketchManager.GetVertexSketch(int64(node))
		if nodeSketch != nil && nodeSketch.IsSketchFull() {
			shouldUseSketch = true
			fmt.Printf("    Node sketch is full → Should use SKETCH method\n")
		} else {
			fmt.Printf("    Node sketch not full → Should use EXACT method\n")
		}
	} else {
		fmt.Printf("    Node sketch empty → Should use EXACT method\n")
	}
	
	// Get the actual edge weight
	actualEdgeWeight := scarGraph.EstimateEdgesToCommunity(node, commID, scarComm)
	fmt.Printf("    Actual returned EdgeWeight: %.8f\n", actualEdgeWeight)
	
	if shouldUseSketch {
		// Analyze sketch-based calculation
		fmt.Printf("    🎨 SKETCH-BASED CALCULATION (Inclusion-Exclusion):\n")
		
		// Get components for inclusion-exclusion
		nodeDegree := scarGraph.GetDegree(node)  // This uses sketch if full
		communityDegree := scarGraph.EstimateCommunityCardinality(commID, scarComm)
		
		fmt.Printf("      Node degree estimate: %.6f\n", nodeDegree)
		fmt.Printf("      Community degree estimate: %.6f\n", communityDegree)
		
		// For union calculation, we'd need access to the union sketch
		// This is complex since we need to union the node and community sketches
		sketchManager := scarGraph.GetSketchManager()
		nodeSketch := sketchManager.GetVertexSketch(int64(node))
		communitySketches := scarGraph.GetCommunitySketches()
		communitySketch, hasSketch := communitySketches[commID]
		
		if hasSketch && communitySketch != nil && nodeSketch != nil {
			fmt.Printf("      🔬 Attempting inclusion-exclusion analysis...\n")
			
			// The formula should be: intersection = node_degree + community_degree - union_degree
			// But union calculation requires sketch operations we can't easily access
			unionEstimate := nodeDegree + communityDegree  // Simplified - actual union would be less
			intersection := nodeDegree + communityDegree - unionEstimate
			
			fmt.Printf("      Simplified intersection estimate: %.6f\n", intersection)
			fmt.Printf("      ⚠️  Actual union calculation requires internal sketch operations\n")
		} else {
			fmt.Printf("      ❌ Missing sketches for inclusion-exclusion\n")
		}
	} else {
		// Analyze exact calculation
		fmt.Printf("    🎯 EXACT CALCULATION (Direct Edge Counting):\n")
		
		// For exact calculation, it should count edges directly
		// We can simulate this by checking adjacency
		neighbors, weights := scarGraph.GetNeighbors(node)
		exactCount := 0.0
		
		fmt.Printf("      Node %d has %d neighbors\n", node, len(neighbors))
		
		for i, neighbor := range neighbors {
			// Check if this neighbor is in the target community
			if neighbor < len(scarComm.NodeToCommunity) && scarComm.NodeToCommunity[neighbor] == commID {
				exactCount += weights[i]
			}
		}
		
		fmt.Printf("      Calculated exact edge weight: %.8f\n", exactCount)
		fmt.Printf("      Returned edge weight: %.8f\n", actualEdgeWeight)
		
		if abs(exactCount - actualEdgeWeight) > 0.01 {
			fmt.Printf("      🔴 MAJOR DISCREPANCY: %.8f difference!\n", abs(exactCount - actualEdgeWeight))
			fmt.Printf("      🐛 This suggests a bug in EstimateEdgesToCommunity!\n")
		} else {
			fmt.Printf("      ✅ Exact calculation appears correct\n")
		}
	}
}

// Calculate Louvain's ranking for the same node and move (for cross-validation)
func (s *DebugSession) calculateLouvainRankingForComparison(node int) *CommunityRanking {
	type CommunityGain struct {
		CommunityID int
		Gain        float64
	}
	
	communityGains := make([]CommunityGain, 0)
	
	for commID := 0; commID < s.LouvainComm.NumCommunities; commID++ {
		if len(s.LouvainComm.CommunityNodes[commID]) > 0 {
			// Calculate modularity gain using Louvain's method
			edgeWeight := louvain.GetEdgeWeightToComm(s.MaterializedGraph, s.LouvainComm, node, commID)
			gain := louvain.CalculateModularityGain(s.MaterializedGraph, s.LouvainComm, node, commID, edgeWeight)
			
			communityGains = append(communityGains, CommunityGain{
				CommunityID: commID,
				Gain:        gain,
			})
		}
	}
	
	// Sort by gain (descending)
	sort.Slice(communityGains, func(i, j int) bool {
		if communityGains[i].Gain == communityGains[j].Gain {
			return communityGains[i].CommunityID < communityGains[j].CommunityID
		}
		return communityGains[i].Gain > communityGains[j].Gain
	})
	
	// Build ranking
	communities := make([]int, len(communityGains))
	gains := make([]float64, len(communityGains))
	
	for i, cg := range communityGains {
		communities[i] = cg.CommunityID
		gains[i] = cg.Gain
	}
	
	return &CommunityRanking{
		Communities: communities,
		Gains:      gains,
	}
}

func (s *DebugSession) analyzeCommunityHealth() error {
	fmt.Printf("\n🏥 COMMUNITY HEALTH ANALYSIS (Move %d)\n", s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 60))
	
	s.log(fmt.Sprintf("=== COMMUNITY HEALTH ANALYSIS (Move %d) ===", s.CurrentMoveIdx))
	
	// Analyze Louvain community structure
	louvainStats := s.calculateCommunityStats(s.LouvainComm)
	
	fmt.Printf("Louvain Community Structure:\n")
	fmt.Printf("  Active Communities: %d\n", louvainStats.NumActive)
	fmt.Printf("  Size Range: %d - %d nodes\n", louvainStats.SmallestSize, louvainStats.LargestSize)
	fmt.Printf("  Average Size: %.1f nodes\n", louvainStats.AverageSize)
	
	// Community size distribution analysis
	fmt.Printf("\nCommunity Size Health:\n")
	verySmall := 0  // Size 1-2
	small := 0      // Size 3-10
	medium := 0     // Size 11-50
	large := 0      // Size 51+
	
	for size, count := range louvainStats.SizeDistribution {
		if size <= 2 {
			verySmall += count
		} else if size <= 10 {
			small += count
		} else if size <= 50 {
			medium += count
		} else {
			large += count
		}
	}
	
	total := verySmall + small + medium + large
	fmt.Printf("  Very Small (1-2):  %d communities (%.1f%%)\n", verySmall, float64(verySmall)/float64(total)*100)
	fmt.Printf("  Small (3-10):      %d communities (%.1f%%)\n", small, float64(small)/float64(total)*100)
	fmt.Printf("  Medium (11-50):    %d communities (%.1f%%)\n", medium, float64(medium)/float64(total)*100)
	fmt.Printf("  Large (51+):       %d communities (%.1f%%)\n", large, float64(large)/float64(total)*100)
	
	// Health warnings
	fmt.Printf("\n⚠️  HEALTH WARNINGS:\n")
	warnings := 0
	
	if verySmall > total/4 {
		fmt.Printf("  🔴 HIGH FRAGMENTATION: %.1f%% communities are very small (≤2 nodes)\n", 
			float64(verySmall)/float64(total)*100)
		fmt.Printf("     → Small communities make sketch estimation unreliable\n")
		warnings++
	}
	
	if louvainStats.LargestSize > s.MaterializedGraph.NumNodes/4 {
		fmt.Printf("  🟡 GIANT COMPONENT: Largest community has %d nodes (%.1f%% of graph)\n", 
			louvainStats.LargestSize, float64(louvainStats.LargestSize)/float64(s.MaterializedGraph.NumNodes)*100)
		fmt.Printf("     → May indicate incomplete clustering\n")
		warnings++
	}
	
	if louvainStats.NumActive > s.MaterializedGraph.NumNodes/2 {
		fmt.Printf("  🟠 HIGH GRANULARITY: %d communities for %d nodes (ratio: %.2f)\n",
			louvainStats.NumActive, s.MaterializedGraph.NumNodes, 
			float64(louvainStats.NumActive)/float64(s.MaterializedGraph.NumNodes))
		fmt.Printf("     → Very fine-grained clustering may challenge sketch algorithms\n")
		warnings++
	}
	
	if warnings == 0 {
		fmt.Printf("  ✅ Community structure appears healthy\n")
	}
	
	// Compare with SCAR community structures
	fmt.Printf("\n🔄 SCAR vs Louvain Community Comparison:\n")
	for _, k := range s.KValues {
		scarComm := s.ScarComms[k]
		scarStats := s.calculateScarCommunityStats(scarComm)
		
		sizeDiff := scarStats.NumActive - louvainStats.NumActive
		avgSizeDiff := scarStats.AverageSize - louvainStats.AverageSize
		
		status := "✅"
		if absInt(sizeDiff) > louvainStats.NumActive/10 || abs(avgSizeDiff) > louvainStats.AverageSize*0.2 {
			status = "⚠️"
		}
		
		fmt.Printf("  k=%d: %s %d communities (Δ%+d), avg size %.1f (Δ%+.1f)\n",
			k, status, scarStats.NumActive, sizeDiff, scarStats.AverageSize, avgSizeDiff)
	}
	
	s.log(fmt.Sprintf("Community health: %d active, size range %d-%d, %d warnings", 
		louvainStats.NumActive, louvainStats.SmallestSize, louvainStats.LargestSize, warnings))
	
	return nil
}

func (s *DebugSession) exportSketchDebugInfo(k int) error {
	scarGraph, exists := s.ScarGraphs[k]
	if !exists {
		return fmt.Errorf("k=%d not available in current session", k)
	}
	
	fmt.Printf("\n📊 EXPORTING SCAR DEBUG INFO (k=%d, Move %d)\n", k, s.CurrentMoveIdx)
	fmt.Println("This will capture detailed sketch information...")
	
	// Create debug output file
	debugFilename := fmt.Sprintf("scar_debug_k%d_move%d.txt", k, s.CurrentMoveIdx)
	debugFile, err := os.Create(debugFilename)
	if err != nil {
		return fmt.Errorf("failed to create debug file: %w", err)
	}
	defer debugFile.Close()
	
	// Write header information
	fmt.Fprintf(debugFile, "SCAR SKETCH DEBUG INFORMATION\n")
	fmt.Fprintf(debugFile, "=============================\n")
	fmt.Fprintf(debugFile, "Move: %d\n", s.CurrentMoveIdx)
	fmt.Fprintf(debugFile, "K Value: %d\n", k)
	fmt.Fprintf(debugFile, "Timestamp: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Fprintf(debugFile, "\n")
	
	if s.CurrentMoveIdx < len(s.Moves) {
		move := s.Moves[s.CurrentMoveIdx]
		fmt.Fprintf(debugFile, "Current Move Context:\n")
		fmt.Fprintf(debugFile, "  Node: %d (original: %s)\n", move.Node, s.ReverseMapping[move.Node])
		fmt.Fprintf(debugFile, "  Target Community: %d\n", move.ToComm)
		fmt.Fprintf(debugFile, "  Louvain Gain: %.8f\n", move.Gain)
		fmt.Fprintf(debugFile, "\n")
	}
	
	fmt.Fprintf(debugFile, "SCAR SKETCH GRAPH DEBUG OUTPUT:\n")
	fmt.Fprintf(debugFile, "===============================\n")
	
	// Temporarily redirect stdout to capture PrintDebug output
	oldStdout := os.Stdout
	
	// Create a temporary file to capture the output
	tempFile, err := os.CreateTemp("", "debug_capture_*.txt")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()
	
	// Redirect stdout to temp file
	os.Stdout = tempFile
	
	// Call PrintDebug (this will write to temp file now)
	scarGraph.PrintDebug()
	
	// Restore stdout
	os.Stdout = oldStdout
	
	// Read from temp file and write to debug file
	tempFile.Seek(0, 0) // Reset to beginning of temp file
	
	buffer := make([]byte, 1024)
	for {
		n, err := tempFile.Read(buffer)
		if n > 0 {
			debugFile.Write(buffer[:n])
		}
		if err != nil {
			break
		}
	}
	
	// Add additional context about current move
	if s.CurrentMoveIdx < len(s.Moves) {
		move := s.Moves[s.CurrentMoveIdx]
		fmt.Fprintf(debugFile, "\n\nCURRENT MOVE ANALYSIS:\n")
		fmt.Fprintf(debugFile, "=====================\n")
		
		// Calculate and show the ranking for this move
		ranking := s.calculateScarRanking(k, move.Node, move.ToComm)
		if ranking != nil {
			fmt.Fprintf(debugFile, "Target Community Ranking:\n")
			fmt.Fprintf(debugFile, "  Target Community: %d\n", move.ToComm)
			fmt.Fprintf(debugFile, "  SCAR Rank: %d out of %d communities\n", 
				ranking.TargetRank, len(ranking.Communities))
			fmt.Fprintf(debugFile, "  Target Gain: %.8f\n", ranking.TargetGain)
			fmt.Fprintf(debugFile, "  Best Gain: %.8f\n", ranking.BestGain)
			fmt.Fprintf(debugFile, "  Gain Difference: %.8f\n", ranking.BestGain - ranking.TargetGain)
			
			fmt.Fprintf(debugFile, "\nTop 10 Community Rankings:\n")
			for i := 0; i < min(10, len(ranking.Communities)); i++ {
				commID := ranking.Communities[i]
				gain := ranking.Gains[i]
				marker := ""
				if commID == move.ToComm {
					marker = " <-- TARGET"
				}
				fmt.Fprintf(debugFile, "  %2d. Community %d: %.8f%s\n", 
					i+1, commID, gain, marker)
			}
		}
		
		// Add community size information
		scarComm := s.ScarComms[k]
		fmt.Fprintf(debugFile, "\nCommunity Size Analysis:\n")
		activeCommunities := 0
		smallCommunities := 0
		
		for commID := 0; commID < scarComm.NumCommunities; commID++ {
			size := len(scarComm.CommunityNodes[commID])
			if size > 0 {
				activeCommunities++
				if size <= 5 {
					smallCommunities++
				}
			}
		}
		
		fmt.Fprintf(debugFile, "  Active Communities: %d\n", activeCommunities)
		fmt.Fprintf(debugFile, "  Small Communities (≤5 nodes): %d (%.1f%%)\n", 
			smallCommunities, float64(smallCommunities)/float64(activeCommunities)*100)
		
		if smallCommunities > activeCommunities/2 {
			fmt.Fprintf(debugFile, "  ⚠️ WARNING: High proportion of small communities may affect sketch reliability\n")
		}
	}
	
	fmt.Printf("✅ Debug information exported to: %s\n", debugFilename)
	fmt.Printf("📄 File contains:\n")
	fmt.Printf("  • Detailed sketch states and hash mappings\n")
	fmt.Printf("  • Current move analysis and ranking\n")
	fmt.Printf("  • Community structure information\n")
	fmt.Printf("  • Sketch reliability indicators\n")
	
	s.log(fmt.Sprintf("Exported SCAR debug info k=%d to %s", k, debugFilename))
	
	return nil
}

// Helper function to get minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (s *DebugSession) compareGainCalculations(k int) error {
	if s.CurrentMoveIdx >= len(s.Moves) {
		return fmt.Errorf("no move to compare")
	}
	
	scarGraph, exists := s.ScarGraphs[k]
	if !exists {
		return fmt.Errorf("k=%d not available", k)
	}
	
	scarComm := s.ScarComms[k]
	move := s.Moves[s.CurrentMoveIdx]
	
	fmt.Printf("\n⚖️  GAIN CALCULATION COMPARISON (k=%d, Move %d)\n", k, s.CurrentMoveIdx)
	fmt.Println(strings.Repeat("=", 60))
	
	s.log(fmt.Sprintf("=== GAIN CALCULATION COMPARISON (k=%d, Move %d) ===", k, s.CurrentMoveIdx))
	
	if move.Node >= scarGraph.NumNodes {
		return fmt.Errorf("node %d out of range for SCAR graph", move.Node)
	}
	
	targetComm := move.ToComm
	
	fmt.Printf("Comparing gains for Node %d → Community %d:\n\n", move.Node, targetComm)
	
	// Louvain calculation
	louvainEdgeWeight := louvain.GetEdgeWeightToComm(s.MaterializedGraph, s.LouvainComm, move.Node, targetComm)
	louvainGain := louvain.CalculateModularityGain(s.MaterializedGraph, s.LouvainComm, move.Node, targetComm, louvainEdgeWeight)
	
	// SCAR calculation
	scarEdgeWeight := scarGraph.EstimateEdgesToCommunity(move.Node, targetComm, scarComm)
	scarGain := scar.CalculateModularityGain(scarGraph, scarComm, move.Node, targetComm, scarEdgeWeight)
	
	fmt.Printf("📊 CALCULATION BREAKDOWN:\n")
	fmt.Printf("%-15s %-15s %-15s %-15s\n", "Algorithm", "Edge Weight", "Node Degree", "Modularity Gain")
	fmt.Println(strings.Repeat("-", 65))
	
	louvainDegree := s.MaterializedGraph.Degrees[move.Node]
	scarDegree := scarGraph.GetDegree(move.Node)
	
	fmt.Printf("%-15s %-15.6f %-15.2f %-15.8f\n", "Louvain", louvainEdgeWeight, louvainDegree, louvainGain)
	fmt.Printf("%-15s %-15.6f %-15.2f %-15.8f\n", "SCAR", scarEdgeWeight, scarDegree, scarGain)
	
	// Calculate differences
	edgeWeightDiff := scarEdgeWeight - louvainEdgeWeight
	degreeDiff := scarDegree - louvainDegree
	gainDiff := scarGain - louvainGain
	
	fmt.Printf("%-15s %-15.6f %-15.2f %-15.8f\n", "Difference", edgeWeightDiff, degreeDiff, gainDiff)
	
	// Analysis
	fmt.Printf("\n🔍 ANALYSIS:\n")
	
	if abs(gainDiff) < 0.000001 {
		fmt.Printf("  ✅ EXCELLENT: Gains are virtually identical\n")
	} else if abs(gainDiff) < 0.0001 {
		fmt.Printf("  ✅ GOOD: Gains are very close (diff: %.8f)\n", gainDiff)
	} else if abs(gainDiff) < 0.001 {
		fmt.Printf("  ⚠️  MODERATE: Noticeable gain difference (%.8f)\n", gainDiff)
	} else {
		fmt.Printf("  🔴 SIGNIFICANT: Large gain difference (%.8f)\n", gainDiff)
	}
	
	// Identify sources of discrepancy
	if abs(edgeWeightDiff) > 0.01 {
		fmt.Printf("  📊 Edge weight estimation differs significantly: %.6f\n", edgeWeightDiff)
		
		filledCount := scarGraph.GetFilledCount(int64(move.Node))
		if filledCount > 0 {
			sketchManager := scarGraph.GetSketchManager()
			nodeSketch := sketchManager.GetVertexSketch(int64(move.Node))
			if nodeSketch != nil && nodeSketch.IsSketchFull() {
				fmt.Printf("     → Node sketch is full - using probabilistic estimation\n")
			} else {
				fmt.Printf("     → Node sketch not full - using exact calculation\n")
			}
		} else {
			fmt.Printf("     → Node sketch empty - using exact calculation\n")
		}
	}
	
	if abs(degreeDiff) > 0.1 {
		fmt.Printf("  📊 Node degree calculation differs: %.2f\n", degreeDiff)
		fmt.Printf("     → This suggests graph representation differences\n")
	}
	
	// Community context
	commSize := len(scarComm.CommunityNodes[targetComm])
	commWeight := scarComm.CommunityWeights[targetComm]
	
	fmt.Printf("\n📋 TARGET COMMUNITY CONTEXT:\n")
	fmt.Printf("  Community %d: %d nodes, total weight %.2f\n", targetComm, commSize, commWeight)
	
	if commSize < 5 {
		fmt.Printf("  ⚠️  Small community - sketch estimation may be unreliable\n")
	}
	
	s.log(fmt.Sprintf("Gain comparison k=%d: Louvain %.8f, SCAR %.8f, diff %.8f", 
		k, louvainGain, scarGain, gainDiff))
	
	return nil
}

// Helper function for absolute value (float64)
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// Helper function for absolute value (int)
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func (s *DebugSession) findProblemArea() error {
	fmt.Println("\n🔍 AUTO-DETECTING PROBLEM AREA...")
	fmt.Println("Testing every 10th move for ranking quality...")
	s.log("=== AUTO-DETECTING PROBLEM AREA ===")
	s.log("Testing every 10th move for ranking quality...")
	
	originalMove := s.CurrentMoveIdx
	defer s.gotoMove(originalMove) // Restore original position
	
	problemMoves := make([]int, 0)
	
	for moveIdx := 0; moveIdx < len(s.Moves); moveIdx += 10 {
		s.gotoMove(moveIdx)
		
		if moveIdx >= len(s.Moves) {
			break
		}
		
		move := s.Moves[moveIdx]
		badRankings := 0
		rankingDetails := make([]string, 0)
		
		for _, k := range s.KValues {
			ranking := s.calculateScarRanking(k, move.Node, move.ToComm)
			if ranking != nil && ranking.TargetRank > 3 {
				badRankings++
				rankingDetails = append(rankingDetails, fmt.Sprintf("k=%d:rank%d", k, ranking.TargetRank))
			}
		}
		
		if badRankings >= len(s.KValues)/2 {
			problemMoves = append(problemMoves, moveIdx)
			s.log(fmt.Sprintf("PROBLEM DETECTED at move %d: Node %d → %d, bad rankings: %s", 
				moveIdx, move.Node, move.ToComm, strings.Join(rankingDetails, ", ")))
		}
		
		if moveIdx%100 == 0 {
			fmt.Printf("  Tested move %d...\n", moveIdx)
		}
	}
	
	if len(problemMoves) == 0 {
		fmt.Println("✅ No significant problems detected")
		s.log("✅ No significant problems detected")
	} else {
		fmt.Printf("❌ Found %d problematic areas:\n", len(problemMoves))
		for _, moveIdx := range problemMoves {
			fmt.Printf("  Move %d\n", moveIdx)
		}
		s.log(fmt.Sprintf("❌ Found %d problematic areas: %v", len(problemMoves), problemMoves))
		
		if len(problemMoves) > 0 {
			fmt.Printf("\n💡 Jumping to first problem area (move %d)\n", problemMoves[0])
			s.log(fmt.Sprintf("💡 Jumping to first problem area (move %d)", problemMoves[0]))
			return s.gotoMove(problemMoves[0])
		}
	}
	
	return nil
}

// Helper functions implementation
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

	// Clean up temp file
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
			continue // Skip malformed lines
		}
		moves = append(moves, move)
	}

	return moves, scanner.Err()
}
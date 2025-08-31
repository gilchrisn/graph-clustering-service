package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"bufio"
	"math"
	"encoding/json"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/utils"
)

type Experiment struct {
	Name       string
	Algorithm  string  // "louvain" or "scar"
	K          int     // SCAR k parameter (ignored for Louvain)
	OutputFile string
}

// CommunityReplay manages replaying moves on a community structure
type CommunityReplay struct {
	NodeToCommunity map[int]int  // node -> community mapping
	NextCommID      int          // for creating new communities
	Graph           *MaterializedGraph
}

// MaterializedGraph wraps the materialized graph for replay
type MaterializedGraph struct {
	Nodes        map[string]bool
	Edges        map[EdgeKey]float64
	NodeMapping  map[string]int  // original ID -> replay ID
	ReverseMap   map[int]string  // replay ID -> original ID
	NumNodes     int
}

type EdgeKey struct {
	From, To string
}

// ReplayAnalysis stores detailed convergence tracking
type ReplayAnalysis struct {
	Algorithm    string
	K            int
	Moves        []utils.MoveEvent
	States       []map[int]int      // Community state after each move
	Convergence  []float64          // NMI with Louvain at each move
	StepMetrics  []StepMetric       // Detailed metrics per step
}

type StepMetric struct {
	MoveNumber     int     `json:"move_number"`
	NMI           float64 `json:"nmi"`
	ARI           float64 `json:"ari"`
	NumCommunities int     `json:"num_communities"`
	Modularity    float64 `json:"modularity"`
}

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("🔬 ENHANCED CONVERGENCE ANALYSIS EXPERIMENT (LEVEL 0 FOCUS)")
	fmt.Println("============================================================")

	// Define experiments
	experiments := []Experiment{
		{"louvain", "louvain", 0, "moves_louvain.jsonl"},
		// {"scar_k2", "scar", 2, "moves_scar_k2.jsonl"},
		// {"scar_k4", "scar", 4, "moves_scar_k4.jsonl"},
		// {"scar_k8", "scar", 8, "moves_scar_k8.jsonl"},
		// {"scar_k16", "scar", 16, "moves_scar_k16.jsonl"},
		// {"scar_k32", "scar", 32, "moves_scar_k32.jsonl"},
		{"scar_k64", "scar", 64, "moves_scar_k64.jsonl"},
		// {"scar_k128", "scar", 128, "moves_scar_k128.jsonl"},
		{"scar_k256", "scar", 256, "moves_scar_k256.jsonl"},
		// {"scar_k512", "scar", 512, "moves_scar_k512.jsonl"},
		// {"scar_k1024", "scar", 1024, "moves_scar_k1024.jsonl"},
	}
	// Run all experiments
	fmt.Printf("Running %d experiments...\n", len(experiments))
	for i, exp := range experiments {
		fmt.Printf("\n[%d/%d] Running %s...", i+1, len(experiments), exp.Name)
		runExperiment(exp, graphFile, propertiesFile, pathFile)
		fmt.Printf(" ✅")
	}

	// Enhanced convergence analysis
	fmt.Println("\n\n🔬 PERFORMING ENHANCED ANALYSIS...")
	analyzeConvergenceWithReplay(experiments, graphFile, propertiesFile, pathFile)
}

func runExperiment(exp Experiment, graphFile, propertiesFile, pathFile string) {
	ctx := context.Background()

	if exp.Algorithm == "louvain" {
		// Materialization + Louvain pipeline
		graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
		if err != nil {
			log.Fatalf("Parse failed: %v", err)
		}

		config := materialization.DefaultMaterializationConfig()
		config.Aggregation.Strategy = materialization.Average
		engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
		materializationResult, err := engine.Materialize()
		if err != nil {
			log.Fatalf("Materialization failed: %v", err)
		}

		louvainGraph := convertToLouvainGraph(materializationResult.HomogeneousGraph)

		// Configure with move tracking
		louvainConfig := louvain.NewConfig()
		louvainConfig.Set("algorithm.random_seed", int64(42))  // Same seed for all
		louvainConfig.Set("algorithm.max_iterations", 50)
		louvainConfig.Set("algorithm.min_modularity_gain", 1e-1)
		louvainConfig.Set("analysis.track_moves", true)
		louvainConfig.Set("analysis.output_file", exp.OutputFile)

		_, err = louvain.Run(louvainGraph, louvainConfig, ctx)
		if err != nil {
			log.Fatalf("Louvain failed: %v", err)
		}

	} else if exp.Algorithm == "scar" {
		// SCAR pipeline
		config := scar.NewConfig()
		config.Set("algorithm.random_seed", int64(42))  // Same seed for all
		config.Set("algorithm.max_iterations", 50)
		config.Set("algorithm.min_modularity_gain", 1e-1)
		config.Set("scar.k", int64(exp.K))
		config.Set("scar.nk", int64(1))
		config.Set("analysis.track_moves", true)
		config.Set("analysis.output_file", exp.OutputFile)

		_, err := scar.Run(graphFile, propertiesFile, pathFile, config, ctx)
		if err != nil {
			log.Fatalf("SCAR failed: %v", err)
		}
	}
}

func convertToLouvainGraph(hgraph *materialization.HomogeneousGraph) *louvain.Graph {
	if len(hgraph.Nodes) == 0 {
		log.Fatal("Empty homogeneous graph")
	}

	// Create ordered list of node IDs with intelligent sorting
	nodeList := make([]string, 0, len(hgraph.Nodes))
	for nodeID := range hgraph.Nodes {
		nodeList = append(nodeList, nodeID)
	}

	// Sort nodes intelligently (numeric if all are integers, lexicographic otherwise)
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

	// Create mapping from original IDs to normalized indices
	originalToNormalized := make(map[string]int)
	for i, originalID := range nodeList {
		originalToNormalized[originalID] = i
	}

	// Create Louvain graph
	graph := louvain.NewGraph(len(nodeList))

	// Add edges with deduplication
	processedEdges := make(map[string]bool)
	for edgeKey, weight := range hgraph.Edges {
		fromNormalized, fromExists := originalToNormalized[edgeKey.From]
		toNormalized, toExists := originalToNormalized[edgeKey.To]

		if !fromExists || !toExists {
			log.Printf("Warning: edge references unknown nodes: %s -> %s", edgeKey.From, edgeKey.To)
			continue
		}

		// Create canonical edge ID to avoid duplicates
		var canonicalID string
		if fromNormalized <= toNormalized {
			canonicalID = fmt.Sprintf("%d-%d", fromNormalized, toNormalized)
		} else {
			canonicalID = fmt.Sprintf("%d-%d", toNormalized, fromNormalized)
		}

		// Only process each undirected edge once
		if !processedEdges[canonicalID] {
			if err := graph.AddEdge(fromNormalized, toNormalized, weight); err != nil {
				log.Printf("Failed to add edge %d-%d: %v", fromNormalized, toNormalized, err)
				continue
			}
			processedEdges[canonicalID] = true
		}
	}

	return graph
}

// Enhanced analyzeConvergence with move replay (LEVEL 0 ANALYSIS ONLY)
func analyzeConvergenceWithReplay(experiments []Experiment, graphFile, propertiesFile, pathFile string) {
	fmt.Println("\n🔬 ENHANCED CONVERGENCE ANALYSIS WITH MOVE REPLAY (LEVEL 0 ONLY)")

	// Step 1: Materialize the graph once for consistent replay
	fmt.Println("Step 1: Materializing graph for replay...")
	materializedGraph, err := materializeGraphForReplay(graphFile, propertiesFile, pathFile)
	if err != nil {
		fmt.Printf("❌ Failed to materialize graph: %v\n", err)
		return
	}

	fmt.Printf("✅ Materialized graph: %d nodes\n", materializedGraph.NumNodes)

	// Step 2: Load all move logs
	fmt.Println("Step 2: Loading move logs...")
	replayAnalyses := make(map[string]*ReplayAnalysis)

	for _, exp := range experiments {
		moves, err := loadMoves(exp.OutputFile)
		if err != nil {
			fmt.Printf("⚠️  Couldn't load %s: %v\n", exp.OutputFile, err)
			continue
		}

		replayAnalyses[exp.Name] = &ReplayAnalysis{
			Algorithm:   exp.Algorithm,
			K:           exp.K,
			Moves:       moves,
			States:      make([]map[int]int, 0),
			StepMetrics: make([]StepMetric, 0),
		}

		fmt.Printf("   Loaded %s: %d total moves\n", exp.Name, len(moves))
	}

	// Step 3: Replay moves and compute convergence
	fmt.Println("Step 3: Replaying level-0 moves and computing convergence...")

	// Get Louvain as reference
	louvainAnalysis, exists := replayAnalyses["louvain"]
	if !exists {
		fmt.Println("❌ Louvain reference not found")
		return
	}

	// 🚨 DEBUG: Print Louvain info
	fmt.Printf("   Louvain: %d total moves\n", len(louvainAnalysis.Moves))

	// Replay Louvain moves
	fmt.Println("   Replaying Louvain level-0 moves...")
	err = replayMovesForAnalysis(louvainAnalysis, materializedGraph, nil) // No comparison for Louvain
	if err != nil {
		fmt.Printf("❌ Failed to replay Louvain: %v\n", err)
		return
	}
	
	// 🚨 DEBUG: Print Louvain states info
	fmt.Printf("   Louvain generated %d states\n", len(louvainAnalysis.States))

	// Replay SCAR variants and compare to Louvain
	for name, analysis := range replayAnalyses {
		if strings.HasPrefix(name, "scar_") {
			fmt.Printf("   Replaying %s level-0 moves...\n", name)
			
			// 🚨 DEBUG: Print SCAR info before replay
			fmt.Printf("     %s: %d total moves, k=%d\n", name, len(analysis.Moves), analysis.K)
			
			err = replayMovesForAnalysis(analysis, materializedGraph, louvainAnalysis)
			if err != nil {
				fmt.Printf("❌ Failed to replay %s: %v\n", name, err)
				continue
			}
			
			// 🚨 DEBUG: Print SCAR states info after replay
			fmt.Printf("     %s generated %d states, %d convergence points\n", 
				name, len(analysis.States), len(analysis.Convergence))
		}
	}

	// Step 4: Display enhanced results
	fmt.Println("Step 4: Displaying results...")
	displayEnhancedConvergenceResults(replayAnalyses, louvainAnalysis)
}

// materializeGraphForReplay creates a consistent graph for move replay
func materializeGraphForReplay(graphFile, propertiesFile, pathFile string) (*MaterializedGraph, error) {
	// Parse SCAR input
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
	if err != nil {
		return nil, fmt.Errorf("parse failed: %w", err)
	}

	// Run materialization
	config := materialization.DefaultMaterializationConfig()
	config.Aggregation.Strategy = materialization.Average
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()
	if err != nil {
		return nil, fmt.Errorf("materialization failed: %w", err)
	}

	hgraph := result.HomogeneousGraph

	// Create node mapping (same logic as convertToLouvainGraph)
	nodeList := make([]string, 0, len(hgraph.Nodes))
	for nodeID := range hgraph.Nodes {
		nodeList = append(nodeList, nodeID)
	}

	// Sort nodes intelligently
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

	// Build mappings
	nodeMapping := make(map[string]int)
	reverseMap := make(map[int]string)
	for i, originalID := range nodeList {
		nodeMapping[originalID] = i
		reverseMap[i] = originalID
	}

	// Convert types to match our MaterializedGraph structure
	nodes := make(map[string]bool)
	for nodeID := range hgraph.Nodes {
		nodes[nodeID] = true
	}

	edges := make(map[EdgeKey]float64)
	for edgeKey, weight := range hgraph.Edges {
		edges[EdgeKey{From: edgeKey.From, To: edgeKey.To}] = weight
	}

	return &MaterializedGraph{
		Nodes:       nodes,
		Edges:       edges,
		NodeMapping: nodeMapping,
		ReverseMap:  reverseMap,
		NumNodes:    len(nodeList),
	}, nil
}

// NewCommunityReplay creates a replay structure from materialized graph
func NewCommunityReplay(materializedGraph *MaterializedGraph) *CommunityReplay {
	replay := &CommunityReplay{
		NodeToCommunity: make(map[int]int),
		NextCommID:      0,
		Graph:          materializedGraph,
	}

	// Initialize: each node in its own community
	for i := 0; i < materializedGraph.NumNodes; i++ {
		replay.NodeToCommunity[i] = i
		replay.NextCommID = i + 1
	}

	return replay
}

// ApplyMove applies a single move to the community structure - WITH DEBUG
func (cr *CommunityReplay) ApplyMove(move utils.MoveEvent) error {
	// Convert original node ID to replay ID if needed
	nodeID := move.Node

	if nodeID < 0 || nodeID >= cr.Graph.NumNodes {
		return fmt.Errorf("invalid node ID: %d (max: %d)", nodeID, cr.Graph.NumNodes-1)
	}

	// Apply the move
	oldComm := cr.NodeToCommunity[nodeID]
	newComm := move.ToComm

	// 🚨 DEBUG: Validate move consistency
	if oldComm != move.FromComm {
		// This might be OK if moves are from different algorithms, but log it
		// fmt.Printf("DEBUG: Node %d in comm %d, move says from %d\n", nodeID, oldComm, move.FromComm)
	}

	// Ensure new community exists in our mapping
	if newComm >= cr.NextCommID {
		cr.NextCommID = newComm + 1
	}

	cr.NodeToCommunity[nodeID] = newComm
	
	// 🚨 ASSERTION: Move was applied
	if cr.NodeToCommunity[nodeID] != newComm {
		return fmt.Errorf("❌ ASSERTION FAILED: Move not applied correctly")
	}
	
	return nil
}

// GetCommunityState returns current community mapping
func (cr *CommunityReplay) GetCommunityState() map[int]int {
	state := make(map[int]int)
	for node, comm := range cr.NodeToCommunity {
		state[node] = comm
	}
	return state
}

// CountCommunities returns number of unique communities
func (cr *CommunityReplay) CountCommunities() int {
	communities := make(map[int]bool)
	for _, comm := range cr.NodeToCommunity {
		communities[comm] = true
	}
	return len(communities)
}

// replayMovesForAnalysis replays moves and computes metrics (LEVEL 0 ONLY) - WITH DEBUG
func replayMovesForAnalysis(analysis *ReplayAnalysis, graph *MaterializedGraph, reference *ReplayAnalysis) error {
	// Create replay structure
	replay := NewCommunityReplay(graph)

	// Filter moves to level 0 only (original graph, not super-graph)
	level0Moves := make([]utils.MoveEvent, 0)
	for _, move := range analysis.Moves {
		if move.Level == 0 {
			level0Moves = append(level0Moves, move)
		} else {
			// Stop when we hit level 1 (super-graph level)
			break
		}
	}

	fmt.Printf("      Filtered to %d level-0 moves (from %d total)\n", len(level0Moves), len(analysis.Moves))
	
	// 🚨 ASSERTION: Should have some level-0 moves
	if len(level0Moves) == 0 {
		return fmt.Errorf("❌ ASSERTION FAILED: No level-0 moves found for %s", analysis.Algorithm)
	}
	
	analysis.Moves = level0Moves // Update for consistent reporting

	// Track every move for accurate comparison (no sampling for now)
	trackEvery := 1  // Store every state for accurate comparison
	
	// Optional: Only sample if memory becomes an issue
	if len(level0Moves) > 5000 {
		trackEvery = len(level0Moves) / 1000 // Max 1000 points for very large runs
		fmt.Printf("      Large dataset detected, sampling every %d moves\n", trackEvery)
	}

	// Initial state
	initialState := replay.GetCommunityState()
	analysis.States = append(analysis.States, initialState)
	
	// 🚨 DEBUG: Print initial state summary
	fmt.Printf("      Initial state: %d nodes, %d communities\n", 
		len(initialState), replay.CountCommunities())

	if reference != nil {
		// 🚨 ASSERTION: Reference should have states
		if len(reference.States) == 0 {
			return fmt.Errorf("❌ ASSERTION FAILED: Reference has no states")
		}
		
		refInitial := reference.States[0]
		
		// 🚨 ASSERTION: Same number of nodes
		if len(initialState) != len(refInitial) {
			return fmt.Errorf("❌ ASSERTION FAILED: Node count mismatch - analysis: %d, reference: %d", 
				len(initialState), len(refInitial))
		}
		
		// 🚨 DEBUG: Check if initial states are identical (should be!)
		identical := true
		for node := range initialState {
			if initialState[node] != refInitial[node] {
				identical = false
				break
			}
		}
		fmt.Printf("      Initial states identical: %v\n", identical)
		
		initialNMI := calculateNMI(initialState, refInitial)
		fmt.Printf("      Initial NMI: %.6f (should be 1.0 if identical)\n", initialNMI)
		
		// 🚨 ASSERTION: Initial NMI should be 1.0
		if initialNMI < 0.999 {
			fmt.Printf("⚠️  WARNING: Initial NMI is %.6f, not 1.0 - states may not be identical\n", initialNMI)
		}
		
		analysis.Convergence = append(analysis.Convergence, initialNMI)

		analysis.StepMetrics = append(analysis.StepMetrics, StepMetric{
			MoveNumber:     0,
			NMI:           initialNMI,
			NumCommunities: replay.CountCommunities(),
		})
	}

	// 🚨 DEBUG: Sample first few moves
	fmt.Printf("      Sample moves:\n")
	for i := 0; i < min(3, len(level0Moves)); i++ {
		move := level0Moves[i]
		fmt.Printf("        Move %d: Node %d from comm %d to comm %d (gain: %.4f)\n", 
			i+1, move.Node, move.FromComm, move.ToComm, move.Gain)
	}

	// Replay level 0 moves only
	for i, move := range level0Moves {
		// 🚨 DEBUG: Validate move
		if move.Node < 0 || move.Node >= graph.NumNodes {
			return fmt.Errorf("❌ ASSERTION FAILED: Invalid node ID %d at move %d (max: %d)", 
				move.Node, i, graph.NumNodes-1)
		}
		
		oldComm := replay.NodeToCommunity[move.Node]
		
		// 🚨 ASSERTION: FromComm should match current community
		if oldComm != move.FromComm {
			fmt.Printf("⚠️  Move %d: Node %d current comm %d != logged FromComm %d\n", 
				i+1, move.Node, oldComm, move.FromComm)
		}

		err := replay.ApplyMove(move)
		if err != nil {
			return fmt.Errorf("failed to apply level-0 move %d: %w", i, err)
		}
		
		// 🚨 ASSERTION: Move was applied correctly
		newComm := replay.NodeToCommunity[move.Node]
		if newComm != move.ToComm {
			return fmt.Errorf("❌ ASSERTION FAILED: Move %d failed - node %d in comm %d, expected %d", 
				i+1, move.Node, newComm, move.ToComm)
		}

		// Track at intervals or important moves
		if i%trackEvery == 0 || i == len(level0Moves)-1 {
			state := replay.GetCommunityState()
			analysis.States = append(analysis.States, state)

			if reference != nil {
				// 🚨 FIX: Use proper state indexing
				// We need to match states by move number, not array index
				refStateIndex := i + 1 // +1 because states array includes initial state at index 0
				
				if refStateIndex < len(reference.States) {
					refState := reference.States[refStateIndex]
					nmi := calculateNMI(state, refState)
					analysis.Convergence = append(analysis.Convergence, nmi)
					
					// 🚨 DEBUG: Print occasional NMI values
					if i < 5 || i%50 == 0 {
						fmt.Printf("        Move %d NMI: %.6f (communities: %d vs ref: %d)\n", 
							i+1, nmi, replay.CountCommunities(), countCommunities(refState))
					}
				} else {
					// 🚨 Handle case where SCAR has more moves than Louvain
					fmt.Printf("⚠️  Move %d: Reference has only %d states, using last\n", 
						i+1, len(reference.States))
					lastRefState := reference.States[len(reference.States)-1]
					nmi := calculateNMI(state, lastRefState)
					analysis.Convergence = append(analysis.Convergence, nmi)
				}

				analysis.StepMetrics = append(analysis.StepMetrics, StepMetric{
					MoveNumber:     i + 1,
					NMI:           analysis.Convergence[len(analysis.Convergence)-1],
					NumCommunities: replay.CountCommunities(),
					Modularity:    move.Modularity, // From logged move
				})
			}
		}
	}
	
	// 🚨 FINAL DEBUG: Print final state summary
	// finalState := replay.GetCommunityState()
	fmt.Printf("      Final state: %d communities\n", replay.CountCommunities())
	
	if reference != nil && len(analysis.Convergence) > 0 {
		finalNMI := analysis.Convergence[len(analysis.Convergence)-1]
		fmt.Printf("      Final NMI: %.6f\n", finalNMI)
		
		// 🚨 ASSERTION: For k=512, final NMI should be very high
		if analysis.Algorithm == "scar" && len(analysis.Moves) > 0 {
			// 🚨 DEBUG: Print analysis details for debugging
			fmt.Printf("        Analysis K value: %d\n", analysis.K)
			
			if analysis.K == 512 && finalNMI < 0.95 {
				fmt.Printf("🚨 ASSERTION FAILED: k=512 should have high NMI (~1.0), got %.6f\n", finalNMI)
				fmt.Printf("    This suggests the replay or NMI calculation has a bug!\n")
			}
			if analysis.K == 128 && finalNMI > 0.99 {
				fmt.Printf("🚨 ASSERTION FAILED: k=128 should not have perfect NMI (%.6f), something's wrong\n", finalNMI)
			}
		}
	}

	return nil
}

// displayEnhancedConvergenceResults shows detailed analysis
func displayEnhancedConvergenceResults(analyses map[string]*ReplayAnalysis, louvainRef *ReplayAnalysis) {
	fmt.Println("\n📊 ENHANCED CONVERGENCE RESULTS (LEVEL 0 ANALYSIS)")

	// Sort SCAR results by K value
	scarNames := make([]string, 0)
	for name := range analyses {
		if strings.HasPrefix(name, "scar_") {
			scarNames = append(scarNames, name)
		}
	}
	sort.Slice(scarNames, func(i, j int) bool {
		return analyses[scarNames[i]].K < analyses[scarNames[j]].K
	})

	// Summary statistics
	fmt.Printf("\n🎯 FINAL CONVERGENCE (NMI at Level 0):\n")
	for _, name := range scarNames {
		analysis := analyses[name]
		if len(analysis.Convergence) > 0 {
			finalNMI := analysis.Convergence[len(analysis.Convergence)-1]
			fmt.Printf("   %s (k=%d): %.4f (from %d level-0 moves)\n", name, analysis.K, finalNMI, len(analysis.Moves))
			
			// 🚨 DEBUG: Print additional info
			if len(analysis.States) > 1 {
				finalComms := countCommunities(analysis.States[len(analysis.States)-1])
				fmt.Printf("       Final communities: %d\n", finalComms)
			}
		} else {
			fmt.Printf("   %s (k=%d): NO CONVERGENCE DATA\n", name, analysis.K)
		}
	}

	// Convergence trajectory (sample points)
	fmt.Printf("\n📈 CONVERGENCE TRAJECTORY (Level 0 moves only):\n")
	fmt.Printf("Move    ")
	for _, name := range scarNames {
		fmt.Printf("  k=%-3d", analyses[name].K)
	}
	fmt.Printf("\n")

	// Sample 10 points across the trajectory
	if len(louvainRef.States) > 0 {
		samplePoints := []int{0}
		
		// Add every 10th move
		for move := 10; move < len(louvainRef.States); move += 10 {
			samplePoints = append(samplePoints, move)
		}
		
		// Always include final move if not already included
		finalMove := len(louvainRef.States) - 1
		if len(samplePoints) == 0 || samplePoints[len(samplePoints)-1] != finalMove {
			samplePoints = append(samplePoints, finalMove)
		}

		for _, moveIdx := range samplePoints {
			fmt.Printf("%4d    ", moveIdx)

			for _, name := range scarNames {
				analysis := analyses[name]
				if moveIdx < len(analysis.Convergence) {
					fmt.Printf("  %.3f", analysis.Convergence[moveIdx])
				} else {
					fmt.Printf("   ---")
				}
			}
			fmt.Printf("\n")
		}
	}

	// Convergence speed analysis
	fmt.Printf("\n⚡ CONVERGENCE SPEED (Level 0):\n")
	for _, name := range scarNames {
		analysis := analyses[name]
		if len(analysis.Convergence) >= 20 {
			early := average(analysis.Convergence[:10])
			late := average(analysis.Convergence[len(analysis.Convergence)-10:])
			speed := (late - early) / float64(len(analysis.Convergence))
			fmt.Printf("   %s: %.4f → %.4f (speed: %.6f/move)\n",
				name, early, late, speed)
		}
	}

	// Community evolution
	fmt.Printf("\n🏘️  COMMUNITY EVOLUTION (Level 0):\n")
	fmt.Printf("Algorithm     Initial   Final   Reduction\n")
	fmt.Printf("Louvain       %5d     %5d     %.1f%%\n",
		len(louvainRef.States[0]),
		countCommunities(louvainRef.States[len(louvainRef.States)-1]),
		100.0*(1.0-float64(countCommunities(louvainRef.States[len(louvainRef.States)-1]))/float64(len(louvainRef.States[0]))))

	for _, name := range scarNames {
		analysis := analyses[name]
		if len(analysis.States) >= 2 {
			initial := len(analysis.States[0])
			final := countCommunities(analysis.States[len(analysis.States)-1])
			reduction := 100.0 * (1.0 - float64(final)/float64(initial))
			fmt.Printf("%-12s  %5d     %5d     %.1f%%\n",
				name, initial, final, reduction)
		}
	}

	fmt.Printf("\n🔬 INSIGHT: Higher k should converge faster to Louvain at level 0 (original graph)!\n")
}

// loadMoves loads move events from a JSONL file
func loadMoves(filename string) ([]utils.MoveEvent, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
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

// calculateNMI computes Normalized Mutual Information between two clusterings - WITH DEBUG
func calculateNMI(communities1, communities2 map[int]int) float64 {
	// Get all nodes
	allNodes := make(map[int]bool)
	for node := range communities1 {
		allNodes[node] = true
	}
	for node := range communities2 {
		allNodes[node] = true
	}

	if len(allNodes) == 0 {
		return 0.0
	}
	
	// 🚨 DEBUG: Check for node mismatches
	nodesMismatch := false
	for node := range allNodes {
		_, exists1 := communities1[node]
		_, exists2 := communities2[node]
		if !exists1 || !exists2 {
			if !nodesMismatch {
				fmt.Printf("⚠️  NMI: Node %d missing in one clustering\n", node)
				nodesMismatch = true
			}
		}
	}

	// Build contingency table
	contingency := make(map[int]map[int]int)

	for node := range allNodes {
		comm1, exists1 := communities1[node]
		comm2, exists2 := communities2[node]

		if !exists1 || !exists2 {
			continue // Skip nodes not in both clusterings
		}

		if contingency[comm1] == nil {
			contingency[comm1] = make(map[int]int)
		}
		contingency[comm1][comm2]++
	}

	// Calculate mutual information
	n := float64(len(allNodes))
	if n == 0 {
		return 0.0
	}

	// Calculate marginals
	marginal1 := make(map[int]int)
	marginal2 := make(map[int]int)

	for comm1, row := range contingency {
		for comm2, count := range row {
			marginal1[comm1] += count
			marginal2[comm2] += count
		}
	}

	// Calculate MI
	mi := 0.0
	for comm1, row := range contingency {
		for comm2, count := range row {
			if count == 0 {
				continue
			}

			pij := float64(count) / n
			pi := float64(marginal1[comm1]) / n
			pj := float64(marginal2[comm2]) / n

			if pij > 0 && pi > 0 && pj > 0 {
				mi += pij * math.Log2(pij/(pi*pj))
			}
		}
	}

	// Calculate entropies
	h1 := 0.0
	for _, count := range marginal1 {
		if count > 0 {
			p := float64(count) / n
			h1 -= p * math.Log2(p)
		}
	}

	h2 := 0.0
	for _, count := range marginal2 {
		if count > 0 {
			p := float64(count) / n
			h2 -= p * math.Log2(p)
		}
	}

	// Normalized MI
	if h1 == 0 && h2 == 0 {
		return 1.0
	}
	if h1 == 0 || h2 == 0 {
		return 0.0
	}

	nmi := 2 * mi / (h1 + h2)
	
	return nmi
}

// Helper function to count unique communities
func countCommunities(communities map[int]int) int {
	commSet := make(map[int]bool)
	for _, commID := range communities {
		commSet[commID] = true
	}
	return len(commSet)
}

// Helper function to calculate average
func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
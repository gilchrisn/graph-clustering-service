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

	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/utils"
)

// ScarDebugSession holds the complete state for interactive SCAR debugging
type ScarDebugSession struct {
	// Graph and algorithm state
	Graph        *scar.SketchGraph
	Community    *scar.Community
	NodeMapping  *scar.NodeMapping
	
	// Move sequence
	Moves        []utils.MoveEvent
	CurrentMove  int
	
	// Community tracking for plotting
	CommunityHistory []CommunitySnapshot
	
	// Configuration
	Config       *scar.Config
	
	// File logging
	LogFile      *os.File
	AnalysisFile *os.File
	
	// Original file paths
	GraphFile    string
	PropertiesFile string
	PathFile     string
}

// CommunitySnapshot captures state at each move for analysis
type CommunitySnapshot struct {
	MoveIndex      int                `json:"move_index"`
	NumCommunities int                `json:"num_communities"`
	Modularity     float64            `json:"modularity"`
	MoveDetails    *utils.MoveEvent   `json:"move_details,omitempty"`
	ActiveComms    []int              `json:"active_communities"`
	SketchStats    SketchStatistics   `json:"sketch_stats"`
}

// SketchStatistics for tracking sketch health
type SketchStatistics struct {
	TotalSketches    int     `json:"total_sketches"`
	FullSketches     int     `json:"full_sketches"`
	PartialSketches  int     `json:"partial_sketches"`
	SketchUtilization float64 `json:"sketch_utilization"`
}

// ModularityGain represents a potential move and its gain
type ModularityGain struct {
	Node             int     `json:"node"`
	FromComm         int     `json:"from_community"`
	ToComm           int     `json:"to_community"`
	Gain             float64 `json:"gain"`
	NodeDegree       float64 `json:"node_degree"`
	EdgeWeight       float64 `json:"edge_weight_to_community"`
	FromCommWeight   float64 `json:"from_community_weight"`
	ToCommWeight     float64 `json:"to_community_weight"`
	FromCommSize     int     `json:"from_community_size"`
	ToCommSize       int     `json:"to_community_size"`
	IsSketchBased    bool    `json:"is_sketch_based"`
	NeighborMethod   string  `json:"neighbor_method"`
	
	// Intersection details for sketch-based calculations
	NodeCardinality  float64 `json:"node_cardinality,omitempty"`
	CommCardinality  float64 `json:"comm_cardinality,omitempty"`
	UnionCardinality float64 `json:"union_cardinality,omitempty"`
	Intersection     float64 `json:"intersection,omitempty"`
	EstimationMethod string  `json:"estimation_method,omitempty"`
}
// DetailedAnalysis captures comprehensive state information
type DetailedAnalysis struct {
	MoveIndex        int              `json:"move_index"`
	AllPossibleGains []ModularityGain `json:"all_possible_gains"`
	TopGains         []ModularityGain `json:"top_gains"`
	ChosenMove       *ModularityGain  `json:"chosen_move"`
	CommunityStates  []CommunityState `json:"community_states"`
	SketchDetails    []SketchDetail   `json:"sketch_details"`
	GraphStats       GraphStatistics  `json:"graph_stats"`
	NeighborAnalysis NeighborAnalysis `json:"neighbor_analysis"`
}

// CommunityState represents the state of a single community
type CommunityState struct {
	CommunityID      int      `json:"community_id"`
	Size             int      `json:"size"`
	InternalWeight   float64  `json:"internal_weight"`
	TotalWeight      float64  `json:"total_weight"`
	EstimatedWeight  float64  `json:"estimated_weight"`
	Nodes            []int    `json:"nodes"`
	HasSketch        bool     `json:"has_sketch"`
	SketchFull       bool     `json:"sketch_full"`
	SketchFilledCount int64   `json:"sketch_filled_count"`
	IsSketchMethod   bool     `json:"is_sketch_method"`
}

// SketchDetail represents detailed sketch information
type SketchDetail struct {
	CommunityID     int       `json:"community_id"`
	HasSketch       bool      `json:"has_sketch"`
	IsFull          bool      `json:"is_full"`
	FilledCount     int64     `json:"filled_count"`
	K               int64     `json:"k"`
	NK              int64     `json:"nk"`
	LayerHashes     [][]int64 `json:"layer_hashes"`
	EstimatedWeight float64   `json:"estimated_weight"`
	ExactWeight     float64   `json:"exact_weight"`
	WeightMethod    string    `json:"weight_method"`
}

// NeighborAnalysis analyzes how neighbors are found
type NeighborAnalysis struct {
	NodesWithFullSketches    int                    `json:"nodes_with_full_sketches"`
	NodesWithPartialSketches int                    `json:"nodes_with_partial_sketches"`
	SketchBasedMoves         int                    `json:"sketch_based_moves"`
	ExactBasedMoves          int                    `json:"exact_based_moves"`
	NeighborMethods          map[string]int         `json:"neighbor_methods"`
	CommunityAccessPatterns  map[string]int         `json:"community_access_patterns"`
}

// GraphStatistics captures overall graph state
type GraphStatistics struct {
	NumNodes          int     `json:"num_nodes"`
	NumCommunities    int     `json:"num_communities"`
	TotalWeight       float64 `json:"total_weight"`
	Modularity        float64 `json:"modularity"`
	SketchUtilization float64 `json:"sketch_utilization"`
	SketchK           int64   `json:"sketch_k"`
	SketchNK          int64   `json:"sketch_nk"`
	HashMappings      int     `json:"hash_mappings"`
	VertexSketches    int     `json:"vertex_sketches"`
}

// PlotData for Python visualization
type PlotData struct {
	Moves            []int     `json:"moves"`
	NumCommunities   []int     `json:"num_communities"`
	Modularity       []float64 `json:"modularity"`
	SketchHealth     []float64 `json:"sketch_health"`
	Config           PlotConfig `json:"config"`
}

type PlotConfig struct {
	K            int64  `json:"k"`
	NK           int64  `json:"nk"`
	GraphFile    string `json:"graph_file"`
	TotalMoves   int    `json:"total_moves"`
}

func main() {
	if len(os.Args) != 4 {
		fmt.Println("SCAR Move Debugger - Integrated Generate & Debug")
		fmt.Println("===============================================")
		fmt.Println("Usage:")
		fmt.Println("  ./scar_debugger <graph_file> <properties_file> <path_file>")
		fmt.Println("")
		fmt.Println("This will:")
		fmt.Println("  1. Run SCAR to generate moves")
		fmt.Println("  2. Start interactive debugging immediately")
		fmt.Println("  3. Create detailed analysis files")
		fmt.Println("")
		fmt.Println("Example:")
		fmt.Println("  ./scar_debugger graph.txt properties.txt path.txt")
		return
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("🎯 SCAR INTEGRATED DEBUGGER")
	fmt.Println("===========================")
	fmt.Printf("Graph: %s\n", graphFile)
	fmt.Printf("Properties: %s\n", propertiesFile)
	fmt.Printf("Path: %s\n", pathFile)

	// Phase 1: Generate moves
	fmt.Println("\n🚀 PHASE 1: Generating SCAR moves...")
	movesFile, err := generateMoves(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to generate moves: %v", err)
	}

	// Phase 2: Start debugging immediately
	fmt.Println("\n🔍 PHASE 2: Starting interactive debugging...")
	session, err := NewScarDebugSession(graphFile, propertiesFile, pathFile, movesFile)
	if err != nil {
		log.Fatalf("Failed to initialize debug session: %v", err)
	}
	defer session.Close()

	fmt.Printf("✅ Ready: %d nodes, %d moves\n", session.Graph.NumNodes, len(session.Moves))
	fmt.Printf("📊 K=%d, NK=%d\n", session.Config.K(), session.Config.NK())
	fmt.Printf("📝 Logging to: scar_debugger.log\n")
	fmt.Printf("📈 Analysis to: scar_detailed_analysis.jsonl\n")

	// Start interactive session
	session.RunInteractiveSession()
}

func generateMoves(graphFile, propertiesFile, pathFile string) (string, error) {
	// Create configuration
	config := scar.NewConfig()
	config.Set("algorithm.max_iterations", 50)
	config.Set("algorithm.min_modularity_gain", -100)
	config.Set("logging.level", "info")
	config.Set("analysis.track_moves", true)
	config.Set("analysis.output_file", "scar_moves.jsonl")
	config.Set("scar.k", int64(256))    // Large K for exact computation
	config.Set("scar.nk", int64(1))
	config.Set("scar.threshold", 0.0)
	config.Set("algorithm.random_seed", int64(42))

	fmt.Printf("Configuration: K=%d, NK=%d, Seed=42\n", config.K(), config.NK())

	// Run SCAR to generate moves
	ctx := context.Background()
	result, err := scar.Run(graphFile, propertiesFile, pathFile, config, ctx)
	if err != nil {
		return "", fmt.Errorf("SCAR failed: %w", err)
	}

	fmt.Printf("✅ SCAR completed: %d levels, %.6f modularity, %d moves\n", 
		result.NumLevels, result.Modularity, result.Statistics.TotalMoves)

	// Verify moves file was created
	movesFile := config.OutputFile()
	if _, err := os.Stat(movesFile); err != nil {
		return "", fmt.Errorf("moves file not created: %w", err)
	}

	return movesFile, nil
}

func NewScarDebugSession(graphFile, propertiesFile, pathFile, movesFile string) (*ScarDebugSession, error) {
	session := &ScarDebugSession{
		GraphFile:      graphFile,
		PropertiesFile: propertiesFile,
		PathFile:       pathFile,
		CommunityHistory: make([]CommunitySnapshot, 0),
	}

	// Initialize log files
	logFile, err := os.Create("scar_debugger.log")
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %w", err)
	}
	session.LogFile = logFile

	analysisFile, err := os.Create("scar_detailed_analysis.jsonl")
	if err != nil {
		return nil, fmt.Errorf("failed to create analysis file: %w", err)
	}
	session.AnalysisFile = analysisFile

	session.log("=== SCAR INTEGRATED DEBUGGER SESSION STARTED ===")
	session.writeAnalysis(map[string]interface{}{
		"session_start": time.Now().Format("2006-01-02 15:04:05"),
		"graph_file": graphFile,
		"properties_file": propertiesFile,
		"path_file": pathFile,
	})

	// Load moves
	session.log("Loading moves from " + movesFile + "...")
	moves, err := loadMovesFromJSONL(movesFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load moves: %w", err)
	}
	session.Moves = moves
	session.log(fmt.Sprintf("✅ Loaded %d moves", len(moves)))

	// Initialize SCAR components
	session.log("Initializing SCAR components...")
	config := scar.NewConfig()
	config.Set("scar.k", int64(256))
	config.Set("scar.nk", int64(1))
	config.Set("algorithm.random_seed", int64(42))
	session.Config = config

	graph, nodeMapping, err := scar.BuildSketchGraph(graphFile, propertiesFile, pathFile, config, config.CreateLogger())
	if err != nil {
		return nil, fmt.Errorf("failed to build sketch graph: %w", err)
	}

	session.Graph = graph
	session.NodeMapping = nodeMapping
	session.Community = scar.NewCommunity(graph)
	session.log("✅ SCAR components initialized")

	// Take initial snapshot with detailed analysis
	session.takeSnapshot()
	session.performDetailedAnalysis()

	return session, nil
}

func (s *ScarDebugSession) log(message string) {
	fmt.Println(message)
	if s.LogFile != nil {
		timestamp := time.Now().Format("15:04:05")
		fmt.Fprintf(s.LogFile, "[%s] %s\n", timestamp, message)
	}
}

func (s *ScarDebugSession) writeAnalysis(data interface{}) {
	if s.AnalysisFile != nil {
		encoder := json.NewEncoder(s.AnalysisFile)
		encoder.Encode(data)
	}
}

func (s *ScarDebugSession) Close() {
	if s.LogFile != nil {
		s.log("=== SCAR INTEGRATED DEBUGGER SESSION ENDED ===")
		s.LogFile.Close()
	}
	if s.AnalysisFile != nil {
		s.AnalysisFile.Close()
	}
}

func (s *ScarDebugSession) RunInteractiveSession() {
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Println("\n🎮 INTERACTIVE SCAR DEBUGGER")
	fmt.Println("Commands:")
	fmt.Println("  step [n]          - Execute next n moves (default: 1)")
	fmt.Println("  goto <move>       - Jump to specific move number")
	fmt.Println("  analyze           - Perform detailed analysis of current state")
	fmt.Println("  gains             - Show all modularity gains for current state")
	fmt.Println("  status            - Show current state")
	fmt.Println("  communities       - Show community distribution")
	fmt.Println("  sketches          - Show sketch states")
	fmt.Println("  neighbors <node>  - Show neighbor discovery for specific node")
	fmt.Println("  export            - Export data for Python plotting")
	fmt.Println("  reset             - Reset to beginning")
	fmt.Println("  help              - Show this help")
	fmt.Println("  quit              - Exit")
	fmt.Println()

	for {
		// Show current status in prompt
		nextMoveInfo := ""
		if s.CurrentMove < len(s.Moves) {
			nextMove := s.Moves[s.CurrentMove]
			nextMoveInfo = fmt.Sprintf(" next:N%d→C%d", nextMove.Node, nextMove.ToComm)
		}

		fmt.Printf("scar[%d/%d%s]> ", s.CurrentMove, len(s.Moves), nextMoveInfo)

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

func (s *ScarDebugSession) executeCommand(command string) error {
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
		return s.performDetailedAnalysis()

	case "gains":
		return s.showAllModularityGains()

	case "status":
		return s.showStatus()

	case "communities":
		return s.showCommunities()

	case "sketches":
		return s.showSketches()

	case "neighbors":
		if len(parts) < 2 {
			return fmt.Errorf("usage: neighbors <node_id>")
		}
		nodeId, err := strconv.Atoi(parts[1])
		if err != nil {
			return fmt.Errorf("invalid node id: %s", parts[1])
		}
		return s.showNeighborDiscovery(nodeId)

	case "export":
		return s.exportPlotData()

	case "reset":
		return s.resetToBeginning()

	case "help":
		fmt.Println("\n🔍 SCAR INTEGRATED DEBUGGER COMMANDS")
		fmt.Println("====================================")
		fmt.Println("  step [n]          - Execute next n moves")
		fmt.Println("  goto <move>       - Jump to specific move")
		fmt.Println("  analyze           - Detailed state analysis (→ analysis file)")
		fmt.Println("  gains             - All modularity gains for current state")
		fmt.Println("  status            - Current state summary")
		fmt.Println("  communities       - Community distribution")
		fmt.Println("  sketches          - Sketch state details")
		fmt.Println("  neighbors <node>  - Neighbor discovery method for node")
		fmt.Println("  export            - Export plot data to JSON")
		fmt.Println("  reset             - Reset to beginning")
		fmt.Println("  help              - Show this help")
		fmt.Println("  quit/exit         - Exit debugger")
		fmt.Println("\n📁 Output Files:")
		fmt.Println("  scar_debugger.log           - Session log")
		fmt.Println("  scar_detailed_analysis.jsonl - Detailed analysis")
		fmt.Println("  scar_community_evolution.json - Plot data")
		return nil

	case "quit", "exit":
		fmt.Println("Goodbye!")
		os.Exit(0)

	default:
		return fmt.Errorf("unknown command: %s (type 'help' for available commands)", parts[0])
	}

	return nil
}

func (s *ScarDebugSession) executeSteps(steps int) error {
	for i := 0; i < steps && s.CurrentMove < len(s.Moves); i++ {
		move := s.Moves[s.CurrentMove]

		s.log(fmt.Sprintf("Executing move %d: Node %d → Community %d (gain: %.6f)",
			s.CurrentMove, move.Node, move.ToComm, move.Gain))

		// Apply move to SCAR
		if move.Node < s.Graph.NumNodes {
			currentComm := s.Community.NodeToCommunity[move.Node]
			scar.MoveNode(s.Graph, s.Community, move.Node, currentComm, move.ToComm)
		}

		s.CurrentMove++

		// Take snapshot after move
		s.takeSnapshot()

		if steps == 1 {
			fmt.Printf("✅ Executed move %d: Node %d → Community %d (gain: %.6f)\n",
				s.CurrentMove-1, move.Node, move.ToComm, move.Gain)
		}
	}

	if steps > 1 {
		fmt.Printf("✅ Executed %d moves (now at move %d)\n", steps, s.CurrentMove)
		s.log(fmt.Sprintf("✅ Executed %d moves (now at move %d)", steps, s.CurrentMove))
	}

	return nil
}

func (s *ScarDebugSession) gotoMove(targetMove int) error {
	if targetMove < 0 || targetMove > len(s.Moves) {
		return fmt.Errorf("move %d out of range [0, %d]", targetMove, len(s.Moves))
	}

	if targetMove < s.CurrentMove {
		// Need to restart from beginning
		fmt.Printf("⏪ Resetting to beginning and replaying to move %d...\n", targetMove)
		if err := s.resetToBeginning(); err != nil {
			return err
		}
	}

	// Execute moves to reach target
	remaining := targetMove - s.CurrentMove
	if remaining > 0 {
		fmt.Printf("⏩ Executing %d moves to reach move %d...\n", remaining, targetMove)
		return s.executeSteps(remaining)
	}

	fmt.Printf("📍 Now at move %d\n", s.CurrentMove)
	return nil
}

func (s *ScarDebugSession) resetToBeginning() error {
	s.CurrentMove = 0
	s.Community = scar.NewCommunity(s.Graph)
	s.CommunityHistory = make([]CommunitySnapshot, 0)
	s.takeSnapshot()
	s.performDetailedAnalysis()
	return nil
}

func (s *ScarDebugSession) performDetailedAnalysis() error {
	fmt.Printf("🔍 Performing detailed analysis (writing to file)...\n")
	
	analysis := DetailedAnalysis{
		MoveIndex:       s.CurrentMove,
		CommunityStates: s.getCommunityStates(),
		SketchDetails:   s.getSketchDetails(),
		GraphStats:      s.getGraphStatistics(),
		NeighborAnalysis: s.analyzeNeighborDiscovery(),
	}

	// Calculate all possible modularity gains
	allGains := s.calculateAllModularityGains()
	analysis.AllPossibleGains = allGains

	// Sort and get top gains
	sort.Slice(allGains, func(i, j int) bool {
		return allGains[i].Gain > allGains[j].Gain
	})

	// Top 20 gains
	topCount := min(20, len(allGains))
	analysis.TopGains = allGains[:topCount]

	// Find chosen move if we have one
	if s.CurrentMove < len(s.Moves) {
		chosenMove := s.Moves[s.CurrentMove]
		for _, gain := range allGains {
			if gain.Node == chosenMove.Node && gain.ToComm == chosenMove.ToComm {
				analysis.ChosenMove = &gain
				break
			}
		}
	}

	// Write to analysis file
	s.writeAnalysis(analysis)

	fmt.Printf("  📊 Calculated %d possible moves\n", len(allGains))
	if len(allGains) > 0 {
		fmt.Printf("  🏆 Top gain: %.6f\n", allGains[0].Gain)
	}
	fmt.Printf("  📝 Analysis written to scar_detailed_analysis.jsonl\n")

	return nil
}
func (s *ScarDebugSession) showAllModularityGains() error {
	fmt.Printf("\n📊 ALL MODULARITY GAINS (Move %d)\n", s.CurrentMove)
	fmt.Println(strings.Repeat("=", 140))

	allGains := s.calculateAllModularityGains()
	
	// Sort by gain (highest first)
	sort.Slice(allGains, func(i, j int) bool {
		return allGains[i].Gain > allGains[j].Gain
	})

	fmt.Printf("Total possible moves: %d\n", len(allGains))
	fmt.Printf("\n📈 TOP 15 MODULARITY GAINS:\n")
	fmt.Printf("%-6s %-8s %-8s %-12s %-8s %-10s %-10s %-10s %-8s %-8s %-10s %-8s %s\n", 
		"Node", "From", "To", "Gain", "Degree", "EdgeWt", "FromWt", "ToWt", "FromSz", "ToSz", "UnionCard", "Sketch", "Status")
	fmt.Println(strings.Repeat("-", 140))

	maxShow := min(15, len(allGains))
	for i := 0; i < maxShow; i++ {
		gain := allGains[i]
		status := ""
		
		// Check if this is the chosen move
		if s.CurrentMove < len(s.Moves) {
			chosenMove := s.Moves[s.CurrentMove]
			if gain.Node == chosenMove.Node && gain.ToComm == chosenMove.ToComm {
				status = "👑 CHOSEN"
			}
		}
		
		// Mark if staying in same community
		if gain.FromComm == gain.ToComm {
			if status == "" {
				status = "⚪ STAY"
			} else {
				status += " STAY"
			}
		}

		sketchFlag := "❌"
		if gain.IsSketchBased {
			sketchFlag = "✅"
		}

		// Show union cardinality for sketch-based moves
		unionCardStr := "n/a"
		if gain.EstimationMethod == "inclusion_exclusion" {
			unionCardStr = fmt.Sprintf("%.3f", gain.UnionCardinality)
		} else if gain.EstimationMethod == "exact" {
			unionCardStr = "exact"
		}

		fmt.Printf("%-6d %-8d %-8d %-12.6f %-8.3f %-10.3f %-10.3f %-10.3f %-8d %-8d %-10s %-8s %s\n",
			gain.Node, gain.FromComm, gain.ToComm, gain.Gain, 
			gain.NodeDegree, gain.EdgeWeight, gain.FromCommWeight, gain.ToCommWeight,
			gain.FromCommSize, gain.ToCommSize, unionCardStr, sketchFlag, status)
	}

	// Rest of the function remains the same...
	if len(allGains) > maxShow {
		fmt.Printf("... and %d more moves (written to analysis file)\n", len(allGains)-maxShow)
	}

	// Show detailed inclusion-exclusion breakdown for top 3 sketch-based moves
	fmt.Printf("\n🔍 INCLUSION-EXCLUSION BREAKDOWN (Top 3 sketch-based moves):\n")
	fmt.Println(strings.Repeat("-", 80))
	
	sketchCount := 0
	for i := 0; i < len(allGains) && sketchCount < 3; i++ {
		gain := allGains[i]
		if gain.EstimationMethod == "inclusion_exclusion" {
			fmt.Printf("Move %d: Node %d → Community %d (Gain: %.6f)\n", sketchCount+1, gain.Node, gain.ToComm, gain.Gain)
			fmt.Printf("  📊 Node Degree:          %.6f\n", gain.NodeDegree)
			fmt.Printf("  📊 Community Weight:     %.6f\n", gain.ToCommWeight)
			fmt.Printf("  📊 Union Cardinality:    %.6f  ← Key value!\n", gain.UnionCardinality)
			fmt.Printf("  🎯 Edge Weight:          %.6f = %.6f + %.6f - %.6f\n", 
				gain.EdgeWeight, gain.NodeDegree, gain.ToCommWeight, gain.UnionCardinality)
			fmt.Printf("  ⚡ Modularity Gain:      %.6f\n", gain.Gain)
			fmt.Println()
			sketchCount++
		}
	}

	// Show next chosen move if available
	if s.CurrentMove < len(s.Moves) {
		chosenMove := s.Moves[s.CurrentMove]
		fmt.Printf("\n👑 NEXT CHOSEN MOVE: Node %d → Community %d (gain: %.6f)\n",
			chosenMove.Node, chosenMove.ToComm, chosenMove.Gain)
	}

	return nil
}

func (s *ScarDebugSession) calculateAllModularityGains() []ModularityGain {
	var allGains []ModularityGain

	for node := 0; node < s.Graph.NumNodes; node++ {
		currentComm := s.Community.NodeToCommunity[node]
		nodeDegree := s.Graph.GetDegree(node)

		// Find neighboring communities using SCAR's method
		neighborComms := s.Graph.FindNeighboringCommunities(node, s.Community)
		
		// Determine if this node uses sketch-based neighbor discovery
		nodeSketch := s.Graph.GetSketchManager().GetVertexSketch(int64(node))
		isSketchBased := nodeSketch != nil && nodeSketch.IsSketchFull()
		
		method := "exact"
		if isSketchBased {
			method = "sketch"
		}

		// Get current community weight and size
		fromCommWeight := s.Community.CommunityWeights[currentComm]
		fromCommSize := len(s.Community.CommunityNodes[currentComm])

		for targetComm, edgeWeight := range neighborComms {
			if len(s.Community.CommunityNodes[targetComm]) > 0 {
				// Calculate modularity gain using SCAR's function
				gain := scar.CalculateModularityGain(s.Graph, s.Community, node, targetComm, edgeWeight)
				
				// Get target community weight and size
				toCommWeight := s.Community.CommunityWeights[targetComm]
				toCommSize := len(s.Community.CommunityNodes[targetComm])
				
				// Create basic gain record
				modGain := ModularityGain{
					Node:           node,
					FromComm:       currentComm,
					ToComm:         targetComm,
					Gain:           gain,
					NodeDegree:     nodeDegree,
					EdgeWeight:     edgeWeight,
					FromCommWeight: fromCommWeight,
					ToCommWeight:   toCommWeight,
					FromCommSize:   fromCommSize,
					ToCommSize:     toCommSize,
					IsSketchBased:  isSketchBased,
					NeighborMethod: method,
				}

				// Calculate intersection details if this was sketch-based
				if isSketchBased {
					s.calculateIntersectionDetails(node, targetComm, &modGain)
				} else {
					modGain.EstimationMethod = "exact"
				}
				
				allGains = append(allGains, modGain)
			}
		}
	}

	return allGains
}

// calculateIntersectionDetails calculates the intersection breakdown for sketch-based estimation
func (s *ScarDebugSession) calculateIntersectionDetails(node, targetComm int, gain *ModularityGain) {
	nodeSketch := s.Graph.GetSketchManager().GetVertexSketch(int64(node))
	if nodeSketch == nil || !nodeSketch.IsSketchFull() {
		gain.EstimationMethod = "exact"
		return
	}

	communitySketch := s.Community.GetCommunitySketch(targetComm)
	if communitySketch == nil {
		gain.EstimationMethod = "exact"
		return
	}

	// Check which estimation method was actually used
	currentComm := s.Community.NodeToCommunity[node]
	
	if targetComm == currentComm && len(s.Community.CommunityNodes[targetComm]) == 1 {
		// Special case: node alone in community
		gain.EstimationMethod = "alone_in_community"
		gain.Intersection = 0.0
		return
	}

	if !communitySketch.IsSketchFull() {
		// Hybrid method was used
		gain.EstimationMethod = "hybrid_exact"
		// For hybrid, intersection is calculated via exact adjacency
		return
	}

	// Full inclusion-exclusion was used
	gain.EstimationMethod = "inclusion_exclusion"
	
	// Recalculate the intersection components for debugging
	gain.NodeCardinality = nodeSketch.EstimateCardinality()
	gain.CommCardinality = s.Graph.EstimateCommunityCardinality(targetComm, s.Community)
	
	unionSketch := nodeSketch.UnionWith(communitySketch)
	if unionSketch != nil {
		gain.UnionCardinality = unionSketch.EstimateCardinality()
		gain.Intersection = gain.NodeCardinality + gain.CommCardinality - gain.UnionCardinality
		
		// Ensure non-negative
		if gain.Intersection < 0 {
			gain.Intersection = 0.0
		}
	} else {
		gain.EstimationMethod = "union_failed"
		gain.Intersection = 0.0
	}
}

func (s *ScarDebugSession) showNeighborDiscovery(nodeId int) error {
	if nodeId < 0 || nodeId >= s.Graph.NumNodes {
		return fmt.Errorf("node %d out of range [0, %d)", nodeId, s.Graph.NumNodes)
	}

	fmt.Printf("\n🔍 NEIGHBOR DISCOVERY ANALYSIS (Node %d)\n", nodeId)
	fmt.Println(strings.Repeat("=", 80))

	currentComm := s.Community.NodeToCommunity[nodeId]
	nodeDegree := s.Graph.GetDegree(nodeId)
	
	// Check sketch status
	nodeSketch := s.Graph.GetSketchManager().GetVertexSketch(int64(nodeId))
	hasSketch := nodeSketch != nil
	isSketchFull := hasSketch && nodeSketch.IsSketchFull()
	
	fmt.Printf("Node Information:\n")
	fmt.Printf("  Current Community: %d\n", currentComm)
	fmt.Printf("  Node Degree: %.3f\n", nodeDegree)
	fmt.Printf("  Has Sketch: %v\n", hasSketch)
	fmt.Printf("  Sketch Full: %v\n", isSketchFull)
	
	if hasSketch {
		fmt.Printf("  Sketch Filled Count: %d\n", nodeSketch.GetFilledCount())
		fmt.Printf("  Sketch K: %d\n", nodeSketch.GetK())
		fmt.Printf("  Sketch NK: %d\n", nodeSketch.GetNk())
	}

	// Get neighboring communities using SCAR's method
	neighborComms := s.Graph.FindNeighboringCommunities(nodeId, s.Community)
	
	fmt.Printf("\nNeighbor Discovery Method: %s\n", 
		map[bool]string{true: "SKETCH-BASED (all communities)", false: "EXACT (adjacency list)"}[isSketchFull])
	
	fmt.Printf("Found %d neighboring communities:\n", len(neighborComms))
	fmt.Printf("%-8s %-10s %-12s %-8s %-10s %-10s %s\n", 
		"CommID", "EdgeWeight", "ModGain", "Size", "CommWt", "UnionCard", "Method")
	fmt.Println(strings.Repeat("-", 75))

	// Calculate gains and prepare for sorting by gain
	type commGainInfo struct {
		commID     int
		weight     float64
		gain       float64
		size       int
		commWeight float64
		unionCard  string
		method     string
		status     string
	}
	
	var commGains []commGainInfo
	for commID, weight := range neighborComms {
		size := len(s.Community.CommunityNodes[commID])
		commWeight := s.Community.CommunityWeights[commID]
		
		// Calculate modularity gain
		gain := scar.CalculateModularityGain(s.Graph, s.Community, nodeId, commID, weight)
		
		// Determine how this edge weight was calculated and get union cardinality
		method := "exact"
		unionCard := "n/a"
		
		if isSketchFull {
			method = "sketch"
			// Calculate union cardinality for sketch-based moves
			if communitySketch := s.Community.GetCommunitySketch(commID); communitySketch != nil {
				if unionSketch := nodeSketch.UnionWith(communitySketch); unionSketch != nil {
					unionCard = fmt.Sprintf("%.3f", unionSketch.EstimateCardinality())
				}
			}
		}
		
		status := ""
		if commID == currentComm {
			status = "(current)"
		}

		commGains = append(commGains, commGainInfo{
			commID:     commID,
			weight:     weight,
			gain:       gain,
			size:       size,
			commWeight: commWeight,
			unionCard:  unionCard,
			method:     method,
			status:     status,
		})
	}

	// Sort by modularity gain (highest first)
	sort.Slice(commGains, func(i, j int) bool {
		return commGains[i].gain > commGains[j].gain
	})

	// Display sorted results
	for _, cg := range commGains {
		fmt.Printf("%-8d %-10.3f %-12.6f %-8d %-10.3f %-10s %s %s\n", 
			cg.commID, cg.weight, cg.gain, cg.size, cg.commWeight, cg.unionCard, cg.method, cg.status)
	}

	// Show detailed sketch analysis if available
	if hasSketch {
		fmt.Printf("\nSketch Details:\n")
		for layer := int64(0); layer < nodeSketch.GetNk(); layer++ {
			hashes := nodeSketch.GetLayerHashes(layer)
			fmt.Printf("  Layer %d: %d hashes\n", layer, len(hashes))
			
			if len(hashes) > 0 {
				// Show which nodes these hashes belong to
				hashOwners := make(map[int]int) // nodeId -> count
				for _, hash := range hashes {
					if ownerNode, exists := s.Graph.GetSketchManager().GetNodeFromHash(uint32(hash)); exists {
						hashOwners[int(ownerNode)]++
					}
				}
				
				fmt.Printf("    Hash owners: ")
				count := 0
				for nodeId, hashCount := range hashOwners {
					if count > 0 {
						fmt.Printf(", ")
					}
					fmt.Printf("N%d(%d)", nodeId, hashCount)
					count++
					if count >= 10 {
						fmt.Printf(", ...")
						break
					}
				}
				fmt.Printf("\n")
			}
		}
	}

	return nil
}

func (s *ScarDebugSession) analyzeNeighborDiscovery() NeighborAnalysis {
	analysis := NeighborAnalysis{
		NeighborMethods:         make(map[string]int),
		CommunityAccessPatterns: make(map[string]int),
	}

	for node := 0; node < s.Graph.NumNodes; node++ {
		nodeSketch := s.Graph.GetSketchManager().GetVertexSketch(int64(node))
		if nodeSketch != nil {
			if nodeSketch.IsSketchFull() {
				analysis.NodesWithFullSketches++
				analysis.NeighborMethods["sketch_based"]++
			} else {
				analysis.NodesWithPartialSketches++
				analysis.NeighborMethods["exact_based"]++
			}
		} else {
			analysis.NeighborMethods["no_sketch"]++
		}

		// Analyze neighbor discovery pattern
		neighborComms := s.Graph.FindNeighboringCommunities(node, s.Community)
		if nodeSketch != nil && nodeSketch.IsSketchFull() {
			analysis.SketchBasedMoves++
			analysis.CommunityAccessPatterns["all_communities"] += len(neighborComms)
		} else {
			analysis.ExactBasedMoves++
			analysis.CommunityAccessPatterns["neighbor_only"] += len(neighborComms)
		}
	}

	return analysis
}

func (s *ScarDebugSession) getCommunityStates() []CommunityState {
	var states []CommunityState
	
	activeCommunities := s.getActiveCommunities()
	for _, commID := range activeCommunities {
		nodes := s.Community.CommunityNodes[commID]
		hasSketch := s.Community.HasCommunitySketch(commID)
		sketchFull := false
		filledCount := int64(0)
		
		if hasSketch {
			sketch := s.Community.GetCommunitySketch(commID)
			if sketch != nil {
				sketchFull = sketch.IsSketchFull()
				filledCount = sketch.GetFilledCount()
			}
		}
		
		// Calculate weights
		internalWeight := s.Community.CommunityInternalWeights[commID]
		totalWeight := s.Community.CommunityWeights[commID]
		estimatedWeight := s.Graph.EstimateCommunityCardinality(commID, s.Community)
		
		// Determine if sketch method is used for weight calculation
		isSketchMethod := hasSketch && sketchFull
		
		states = append(states, CommunityState{
			CommunityID:       commID,
			Size:              len(nodes),
			InternalWeight:    internalWeight,
			TotalWeight:       totalWeight,
			EstimatedWeight:   estimatedWeight,
			Nodes:             nodes,
			HasSketch:         hasSketch,
			SketchFull:        sketchFull,
			SketchFilledCount: filledCount,
			IsSketchMethod:    isSketchMethod,
		})
	}
	
	return states
}

func (s *ScarDebugSession) getSketchDetails() []SketchDetail {
	var details []SketchDetail
	
	activeCommunities := s.getActiveCommunities()
	for _, commID := range activeCommunities {
		hasSketch := s.Community.HasCommunitySketch(commID)
		
		detail := SketchDetail{
			CommunityID: commID,
			HasSketch:   hasSketch,
		}
		
		if hasSketch {
			sketch := s.Community.GetCommunitySketch(commID)
			if sketch != nil {
				detail.IsFull = sketch.IsSketchFull()
				detail.FilledCount = sketch.GetFilledCount()
				detail.K = sketch.GetK()
				detail.NK = sketch.GetNk()
				detail.EstimatedWeight = s.Graph.EstimateCommunityCardinality(commID, s.Community)
				detail.ExactWeight = s.Community.CommunityWeights[commID]
				
				if detail.IsFull {
					detail.WeightMethod = "sketch_estimation"
				} else {
					detail.WeightMethod = "exact_sum"
				}
				
				// Get layer hashes
				for layer := int64(0); layer < sketch.GetNk(); layer++ {
					hashes := sketch.GetLayerHashes(layer)
					int64Hashes := make([]int64, len(hashes))
					for i, h := range hashes {
						int64Hashes[i] = int64(h)
					}
					detail.LayerHashes = append(detail.LayerHashes, int64Hashes)
				}
			}
		}
		
		details = append(details, detail)
	}
	
	return details
}

func (s *ScarDebugSession) getGraphStatistics() GraphStatistics {
	activeComms := s.getActiveCommunities()
	sketchStats := s.calculateSketchStats()
	
	sketchManager := s.Graph.GetSketchManager()
	hashMappings := 0
	vertexSketches := 0
	
	if sketchManager != nil {
		// Note: We can't directly access these private fields, so we'll estimate
		// Based on the number of nodes that have sketches
		for node := 0; node < s.Graph.NumNodes; node++ {
			if sketch := sketchManager.GetVertexSketch(int64(node)); sketch != nil {
				vertexSketches++
				// Each sketch contributes NK hash values
				hashMappings += int(s.Config.NK())
			}
		}
	}
	
	return GraphStatistics{
		NumNodes:          s.Graph.NumNodes,
		NumCommunities:    len(activeComms),
		TotalWeight:       s.Graph.TotalWeight,
		Modularity:        scar.CalculateModularity(s.Graph, s.Community),
		SketchUtilization: sketchStats.SketchUtilization,
		SketchK:           s.Config.K(),
		SketchNK:          s.Config.NK(),
		HashMappings:      hashMappings,
		VertexSketches:    vertexSketches,
	}
}

func (s *ScarDebugSession) showStatus() error {
	fmt.Printf("\n📋 SCAR DEBUGGER STATUS\n")
	fmt.Println(strings.Repeat("=", 40))
	fmt.Printf("Current Move: %d/%d\n", s.CurrentMove, len(s.Moves))
	fmt.Printf("Graph Nodes: %d\n", s.Graph.NumNodes)
	fmt.Printf("K Value: %d\n", s.Config.K())
	fmt.Printf("NK Value: %d\n", s.Config.NK())

	// Current community state
	activeComms := s.getActiveCommunities()
	fmt.Printf("Active Communities: %d\n", len(activeComms))

	// Current modularity
	modularity := scar.CalculateModularity(s.Graph, s.Community)
	fmt.Printf("Current Modularity: %.6f\n", modularity)

	// Sketch statistics
	sketchStats := s.calculateSketchStats()
	fmt.Printf("Sketch Health: %.1f%% (%d/%d full)\n", 
		sketchStats.SketchUtilization*100, sketchStats.FullSketches, sketchStats.TotalSketches)

	// Graph statistics
	graphStats := s.getGraphStatistics()
	fmt.Printf("Vertex Sketches: %d\n", graphStats.VertexSketches)
	fmt.Printf("Hash Mappings: %d\n", graphStats.HashMappings)

	// Next move info
	if s.CurrentMove < len(s.Moves) {
		nextMove := s.Moves[s.CurrentMove]
		fmt.Printf("\nNext Move: Node %d → Community %d (gain: %.6f)\n",
			nextMove.Node, nextMove.ToComm, nextMove.Gain)
	} else {
		fmt.Printf("\nNo more moves available\n")
	}

	return nil
}

func (s *ScarDebugSession) showCommunities() error {
	fmt.Printf("\n🏘️  COMMUNITY DISTRIBUTION\n")
	fmt.Println(strings.Repeat("=", 70))

	activeComms := s.getActiveCommunities()
	fmt.Printf("Total Active Communities: %d\n", len(activeComms))

	// Get community states
	commStates := s.getCommunityStates()
	
	// Sort by size
	sort.Slice(commStates, func(i, j int) bool {
		return commStates[i].Size > commStates[j].Size
	})

	fmt.Printf("\n📊 TOP COMMUNITIES BY SIZE:\n")
	fmt.Printf("%-8s %-6s %-12s %-12s %-12s %-12s %s\n", 
		"CommID", "Size", "Internal", "Total", "Estimated", "HasSketch", "Status")
	fmt.Println(strings.Repeat("-", 80))

	maxShow := min(15, len(commStates))
	for i := 0; i < maxShow; i++ {
		comm := commStates[i]
		
		status := "✅"
		if comm.HasSketch {
			if !comm.SketchFull {
				status = "⚠️ PARTIAL"
			}
		} else {
			status = "❌ NO SKETCH"
		}

		fmt.Printf("%-8d %-6d %-12.3f %-12.3f %-12.3f %-12v %s\n", 
			comm.CommunityID, comm.Size, comm.InternalWeight, comm.TotalWeight, 
			comm.EstimatedWeight, comm.HasSketch, status)
	}

	if len(commStates) > maxShow {
		fmt.Printf("... and %d more communities (see analysis file for details)\n", len(commStates)-maxShow)
	}

	return nil
}

func (s *ScarDebugSession) showSketches() error {
	fmt.Printf("\n🎨 SKETCH STATE ANALYSIS\n")
	fmt.Println(strings.Repeat("=", 70))

	stats := s.calculateSketchStats()
	fmt.Printf("Total Sketches: %d\n", stats.TotalSketches)
	fmt.Printf("Full Sketches: %d\n", stats.FullSketches)
	fmt.Printf("Partial Sketches: %d\n", stats.PartialSketches)
	fmt.Printf("Sketch Utilization: %.1f%%\n", stats.SketchUtilization*100)

	sketchDetails := s.getSketchDetails()
	
	fmt.Printf("\n📊 SKETCH DETAILS:\n")
	fmt.Printf("%-8s %-6s %-10s %-8s %-10s %-12s %-12s %s\n", 
		"CommID", "Size", "HasSketch", "IsFull", "Filled", "EstWeight", "ExactWeight", "Status")
	fmt.Println(strings.Repeat("-", 85))

	maxShow := min(10, len(sketchDetails))
	for i := 0; i < maxShow; i++ {
		detail := sketchDetails[i]
		size := len(s.Community.CommunityNodes[detail.CommunityID])
		
		status := "❌ NO SKETCH"
		if detail.HasSketch {
			if detail.IsFull {
				status = "✅ FULL"
			} else {
				status = "⚠️ PARTIAL"
			}
		}

		fmt.Printf("%-8d %-6d %-10v %-8v %-10d %-12.3f %-12.3f %s\n", 
			detail.CommunityID, size, detail.HasSketch, detail.IsFull, 
			detail.FilledCount, detail.EstimatedWeight, detail.ExactWeight, status)
	}

	fmt.Printf("\n📝 Detailed sketch information written to analysis file\n")
	return nil
}

func (s *ScarDebugSession) exportPlotData() error {
	if len(s.CommunityHistory) == 0 {
		return fmt.Errorf("no data to export")
	}

	plotData := PlotData{
		Config: PlotConfig{
			K:            s.Config.K(),
			NK:           s.Config.NK(),
			GraphFile:    s.GraphFile,
			TotalMoves:   len(s.Moves),
		},
	}

	// Extract data from history
	for _, snapshot := range s.CommunityHistory {
		plotData.Moves = append(plotData.Moves, snapshot.MoveIndex)
		plotData.NumCommunities = append(plotData.NumCommunities, snapshot.NumCommunities)
		plotData.Modularity = append(plotData.Modularity, snapshot.Modularity)
		plotData.SketchHealth = append(plotData.SketchHealth, snapshot.SketchStats.SketchUtilization)
	}

	filename := "scar_community_evolution.json"
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(plotData); err != nil {
		return fmt.Errorf("failed to encode data: %w", err)
	}

	fmt.Printf("✅ Plot data exported to: %s\n", filename)
	fmt.Println("Now you can run: python plot.py")
	s.log(fmt.Sprintf("Exported plot data to: %s", filename))

	return nil
}

// Helper functions
func (s *ScarDebugSession) takeSnapshot() {
	activeComms := s.getActiveCommunities()
	sketchStats := s.calculateSketchStats()
	modularity := scar.CalculateModularity(s.Graph, s.Community)

	var moveDetails *utils.MoveEvent
	if s.CurrentMove > 0 && s.CurrentMove <= len(s.Moves) {
		moveDetails = &s.Moves[s.CurrentMove-1]
	}

	snapshot := CommunitySnapshot{
		MoveIndex:      s.CurrentMove,
		NumCommunities: len(activeComms),
		Modularity:     modularity,
		MoveDetails:    moveDetails,
		ActiveComms:    activeComms,
		SketchStats:    sketchStats,
	}

	s.CommunityHistory = append(s.CommunityHistory, snapshot)
}

func (s *ScarDebugSession) getActiveCommunities() []int {
	active := make([]int, 0)
	for commID := 0; commID < s.Community.NumCommunities; commID++ {
		if len(s.Community.CommunityNodes[commID]) > 0 {
			active = append(active, commID)
		}
	}
	sort.Ints(active)
	return active
}

func (s *ScarDebugSession) calculateSketchStats() SketchStatistics {
	totalSketches := 0
	fullSketches := 0
	partialSketches := 0

	activeComms := s.getActiveCommunities()
	for _, commID := range activeComms {
		if s.Community.HasCommunitySketch(commID) {
			totalSketches++
			sketch := s.Community.GetCommunitySketch(commID)
			if sketch != nil && sketch.IsSketchFull() {
				fullSketches++
			} else {
				partialSketches++
			}
		}
	}

	utilization := 0.0
	if totalSketches > 0 {
		utilization = float64(fullSketches) / float64(totalSketches)
	}

	return SketchStatistics{
		TotalSketches:     totalSketches,
		FullSketches:      fullSketches,
		PartialSketches:   partialSketches,
		SketchUtilization: utilization,
	}
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
			continue // Skip invalid lines
		}
		moves = append(moves, move)
	}

	return moves, scanner.Err()
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
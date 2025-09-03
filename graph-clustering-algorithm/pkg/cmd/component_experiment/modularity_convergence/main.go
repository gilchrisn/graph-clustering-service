package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"bufio"
	// "time"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/utils"
)

type AlgorithmRun struct {
	Name       string
	Algorithm  string  // "louvain" or "scar"
	K          int     // SCAR k parameter (0 for Louvain)
	OutputFile string
	Moves      []utils.MoveEvent
}

// ModularityTable tracks modularity progression for each algorithm
type ModularityTable struct {
	MaxMoves   int
	Algorithms []AlgorithmRun
	Table      [][]float64  // [move][algorithm] = modularity
}



func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("🎯 MOVE-BY-MOVE MODULARITY COMPARISON")
	fmt.Println("====================================")
	fmt.Println("Tracking exact modularity progression for Louvain vs SCAR(k)")

	// Step 1: Materialize the reference graph
	fmt.Println("\nStep 1: Materializing reference graph...")
	materializedGraph, nodeMapping, err := materializeReferenceGraph(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to materialize graph: %v", err)
	}
	fmt.Printf("✅ Materialized graph: %d nodes, %.2f total weight\n", 
		materializedGraph.NumNodes, materializedGraph.TotalWeight)

	// Step 2: Define algorithm runs
	runs := []AlgorithmRun{
		{"Louvain", "louvain", 0, "moves_louvain.jsonl", nil},
		{"SCAR-K2", "scar", 2, "moves_scar_k2.jsonl", nil},
		// {"SCAR-K4", "scar", 4, "moves_scar_k4.jsonl", nil},
		{"SCAR-K8", "scar", 8, "moves_scar_k8.jsonl", nil},
		// {"SCAR-K16", "scar", 16, "moves_scar_k16.jsonl", nil},
		{"SCAR-K32", "scar", 32, "moves_scar_k32.jsonl", nil},
		// {"SCAR-K64", "scar", 64, "moves_scar_k64.jsonl", nil},
		{"SCAR-K128", "scar", 128, "moves_scar_k128.jsonl", nil},
		// {"SCAR-K256", "scar", 256, "moves_scar_k256.jsonl", nil},
		{"SCAR-K512", "scar", 512, "moves_scar_k512.jsonl", nil},
		// {"SCAR-K1024", "scar", 1024, "moves_scar_k1024.jsonl", nil},
		
	}

	// Step 3: Run algorithms with move tracking
	fmt.Println("\nStep 2: Running algorithms with move tracking...")
	for i, run := range runs {
		fmt.Printf("  [%d/%d] Running %s...", i+1, len(runs), run.Name)
		err := executeAlgorithmRun(&runs[i], graphFile, propertiesFile, pathFile, materializedGraph, nodeMapping)
		if err != nil {
			fmt.Printf(" ❌ Failed: %v\n", err)
			continue
		}
		fmt.Printf(" ✅ (%d moves)\n", len(runs[i].Moves))
	}

	// Step 4: Build modularity table
	fmt.Println("\nStep 3: Building move-by-move modularity table...")
	modularityTable := buildModularityTable(materializedGraph, runs)

	// Step 5: Display results
	fmt.Println("\nStep 4: Results")
	displayModularityTable(modularityTable)

	fmt.Println("\nStep 5: Exporting data for visualization...")
	if err := ExportModularityTable(modularityTable); err != nil {
		fmt.Printf("❌ Failed to export table: %v\n", err)
	}
}

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

func executeAlgorithmRun(run *AlgorithmRun, graphFile, propertiesFile, pathFile string, 
	materializedGraph *louvain.Graph, nodeMapping map[string]int) error {
	
	ctx := context.Background()

	if run.Algorithm == "louvain" {
		// Run Louvain on materialized graph
		config := louvain.NewConfig()
		config.Set("algorithm.random_seed", int64(42))
		config.Set("algorithm.max_iterations", 20)
		config.Set("algorithm.min_modularity_gain", -100.0)
		config.Set("algorithm.max_levels", 5)
		config.Set("analysis.track_moves", true)
		config.Set("analysis.output_file", run.OutputFile)

		_, err := louvain.Run(materializedGraph, config, ctx)
		if err != nil {
			return fmt.Errorf("Louvain failed: %w", err)
		}

	} else if run.Algorithm == "scar" {
		// Run SCAR
		config := scar.NewConfig()
		config.Set("algorithm.random_seed", int64(42))
		// config.Set("algorithm.random_seed", int64(time.Now().UnixNano())) // Use current time for randomness
		config.Set("algorithm.max_iterations", 20)
		config.Set("algorithm.min_modularity_gain", -100.0)
		config.Set("algorithm.max_levels", 5)
		config.Set("scar.k", int64(run.K))
		config.Set("scar.nk", int64(1))
		config.Set("analysis.track_moves", true)
		config.Set("analysis.output_file", run.OutputFile)

		scarResult, err := scar.Run(graphFile, propertiesFile, pathFile, config, ctx)
		if err != nil {
			return fmt.Errorf("SCAR failed: %w", err)
		}

		// Load and convert SCAR moves to materialized graph node IDs
		moves, err := loadMovesFromJSONL(run.OutputFile)
		if err != nil {
			return fmt.Errorf("failed to load SCAR moves: %w", err)
		}

		// Convert SCAR node IDs to materialized graph node IDs
		convertedMoves, err := convertSCARMoves(moves, scarResult.NodeMapping, nodeMapping)
		if err != nil {
			return fmt.Errorf("failed to convert SCAR moves: %w", err)
		}

		run.Moves = convertedMoves
		return nil
	}

	// Load moves for Louvain (no conversion needed)
	moves, err := loadMovesFromJSONL(run.OutputFile)
	if err != nil {
		return fmt.Errorf("failed to load moves: %w", err)
	}

	run.Moves = moves
	return nil
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

func convertSCARMoves(scarMoves []utils.MoveEvent, scarNodeMapping *scar.NodeMapping, 
	materializedNodeMapping map[string]int) ([]utils.MoveEvent, error) {
	
	convertedMoves := make([]utils.MoveEvent, 0, len(scarMoves))

	for _, move := range scarMoves {
		// Convert SCAR compressed node ID to original node ID
		if move.Node >= len(scarNodeMapping.CompressedToOriginal) {
			continue // Skip invalid moves
		}
		originalIntID := scarNodeMapping.CompressedToOriginal[move.Node]
		
		// Convert original int ID to string ID to materialized index
		originalStringID := strconv.Itoa(originalIntID)
		if matIndex, exists := materializedNodeMapping[originalStringID]; exists {
			convertedMove := move
			convertedMove.Node = matIndex
			convertedMoves = append(convertedMoves, convertedMove)
		}
	}

	return convertedMoves, nil
}

func buildModularityTable(graph *louvain.Graph, runs []AlgorithmRun) *ModularityTable {
	// Find maximum number of moves
	maxMoves := 0
	validRuns := make([]AlgorithmRun, 0)
	
	for _, run := range runs {
		if len(run.Moves) > 0 {
			validRuns = append(validRuns, run)
			if len(run.Moves) > maxMoves {
				maxMoves = len(run.Moves)
			}
		}
	}

	if len(validRuns) == 0 {
		return &ModularityTable{MaxMoves: 0, Algorithms: validRuns, Table: nil}
	}

	// Build table: rows = moves, cols = algorithms
	table := make([][]float64, maxMoves+1) // +1 for initial state
	for i := range table {
		table[i] = make([]float64, len(validRuns))
	}

	// For each algorithm, replay moves and calculate exact modularity
	for algIdx, run := range validRuns {
		fmt.Printf("    Replaying %s moves...\n", run.Name)
		
		// Create fresh community structure
		comm := louvain.NewCommunity(graph)
		
		// Initial modularity
		table[0][algIdx] = louvain.CalculateModularity(graph, comm)
		
		// Apply moves one by one and calculate exact modularity
		for moveIdx, move := range run.Moves {
			if move.Node >= graph.NumNodes {
				fmt.Printf("      Warning: Skipping invalid node %d\n", move.Node)
				continue
			}
			
			oldComm := comm.NodeToCommunity[move.Node]
			newComm := move.ToComm
			
			// Ensure new community exists
			if newComm >= len(comm.CommunityNodes) {
				// Extend community arrays if needed
				for len(comm.CommunityNodes) <= newComm {
					comm.CommunityNodes = append(comm.CommunityNodes, []int{})
					comm.CommunityWeights = append(comm.CommunityWeights, 0.0)
					comm.CommunityInternalWeights = append(comm.CommunityInternalWeights, 0.0)
				}
				if newComm >= comm.NumCommunities {
					comm.NumCommunities = newComm + 1
				}
			}
			
			// Apply move using exact Louvain logic
			louvain.MoveNode(graph, comm, move.Node, oldComm, newComm)
			
			// Calculate exact modularity on exact graph
			exactModularity := louvain.CalculateModularity(graph, comm)
			table[moveIdx+1][algIdx] = exactModularity
		}
		
		// Fill remaining cells with final modularity for shorter runs
		finalModularity := table[len(run.Moves)][algIdx]
		for i := len(run.Moves) + 1; i < len(table); i++ {
			table[i][algIdx] = finalModularity
		}

		activeCommunities := 0
		for commID := 0; commID < comm.NumCommunities; commID++ {
			if len(comm.CommunityNodes[commID]) > 0 {
				activeCommunities++
			}
		}
		fmt.Printf("        → Final communities: %d\n", activeCommunities)
	}

	return &ModularityTable{
		MaxMoves:   maxMoves,
		Algorithms: validRuns,
		Table:      table,
	}
}

func displayModularityTable(mt *ModularityTable) {
	if len(mt.Algorithms) == 0 {
		fmt.Println("❌ No valid algorithm runs to display")
		return
	}

	fmt.Println("\n📊 MOVE-BY-MOVE MODULARITY TABLE")
	fmt.Println("=================================")

	// Print header
	fmt.Printf("%-8s", "Move")
	for _, run := range mt.Algorithms {
		fmt.Printf(" %-12s", run.Name)
	}
	fmt.Println()
	
	fmt.Printf("%s", "────────")
	for range mt.Algorithms {
		fmt.Printf(" %s", "────────────")
	}
	fmt.Println()

	// Print initial state
	fmt.Printf("%-8s", "Initial")
	for algIdx := range mt.Algorithms {
		fmt.Printf(" %-12.6f", mt.Table[0][algIdx])
	}
	fmt.Println()

	// Print selected moves for readability
	samplePoints := []int{}
	
	// Add early moves
	for i := 1; i <= 10 && i < len(mt.Table); i++ {
		samplePoints = append(samplePoints, i)
	}
	
	// Add every 20th move
	for i := 20; i < len(mt.Table); i += 20 {
		samplePoints = append(samplePoints, i)
	}
	
	// Always add final move
	if len(mt.Table) > 1 {
		final := len(mt.Table) - 1
		if len(samplePoints) == 0 || samplePoints[len(samplePoints)-1] != final {
			samplePoints = append(samplePoints, final)
		}
	}

	for _, moveIdx := range samplePoints {
		fmt.Printf("%-8d", moveIdx)
		for algIdx := range mt.Algorithms {
			fmt.Printf(" %-12.6f", mt.Table[moveIdx][algIdx])
		}
		fmt.Println()
	}

	// Analysis
	fmt.Printf("\n🔍 CONVERGENCE ANALYSIS:\n")
	
	if len(mt.Table) == 0 {
		fmt.Printf("   No data to analyze\n")
		return
	}
	
	// Get final modularities
	finalRow := len(mt.Table) - 1
	louvainFinal := mt.Table[finalRow][0] // Assume Louvain is first
	fmt.Printf("   Louvain final modularity: %.6f\n", louvainFinal)
	
	// Compare SCAR variants
	for algIdx, run := range mt.Algorithms {
		if run.Algorithm == "scar" {
			finalMod := mt.Table[finalRow][algIdx]
			gap := finalMod - louvainFinal
			
			fmt.Printf("   %s final: %.6f (gap: %+.6f)", run.Name, finalMod, gap)
			if gap > -0.001 {
				fmt.Printf(" ✅ Excellent")
			} else if gap > -0.01 {
				fmt.Printf(" ⚠️  Good")
			} else if gap > -0.05 {
				fmt.Printf(" 📊 Moderate")
			} else {
				fmt.Printf(" ❌ Poor")
			}
			fmt.Println()
		}
	}

	// Convergence trend analysis
	fmt.Printf("\n💡 CONVERGENCE PATTERN:\n")
	scarRuns := []AlgorithmRun{}
	for _, run := range mt.Algorithms {
		if run.Algorithm == "scar" {
			scarRuns = append(scarRuns, run)
		}
	}
	
	if len(scarRuns) >= 2 {
		// Sort by k value
		sort.Slice(scarRuns, func(i, j int) bool {
			return scarRuns[i].K < scarRuns[j].K
		})
		
		fmt.Printf("   K values tested: ")
		for i, run := range scarRuns {
			if i > 0 { fmt.Printf(", ") }
			fmt.Printf("%d", run.K)
		}
		fmt.Println()
		
		// Find algorithm indices in the main table
		firstSCARIdx, lastSCARIdx := -1, -1
		for algIdx, run := range mt.Algorithms {
			if run.Name == scarRuns[0].Name {
				firstSCARIdx = algIdx
			}
			if run.Name == scarRuns[len(scarRuns)-1].Name {
				lastSCARIdx = algIdx
			}
		}
		
		if firstSCARIdx >= 0 && lastSCARIdx >= 0 {
			firstSCARFinal := mt.Table[finalRow][firstSCARIdx]
			lastSCARFinal := mt.Table[finalRow][lastSCARIdx]
			improvement := lastSCARFinal - firstSCARFinal
			
			fmt.Printf("   Improvement from k=%d to k=%d: %+.6f\n", 
				scarRuns[0].K, scarRuns[len(scarRuns)-1].K, improvement)
			
			if improvement > 0.01 {
				fmt.Printf("   → ✅ Strong convergence! Higher k significantly improves modularity\n")
			} else if improvement > 0.001 {
				fmt.Printf("   → 📈 Moderate convergence trend\n")
			} else if improvement > -0.001 {
				fmt.Printf("   → 📊 Minimal change - might already be converged\n")
			} else {
				fmt.Printf("   → ❌ Unexpected: higher k performing worse! Investigate bug\n")
			}
		}
	}

	fmt.Printf("\n📝 INTERPRETATION:\n")
	fmt.Printf("   • All algorithms use the SAME exact graph for modularity calculation\n")
	fmt.Printf("   • Differences are purely due to different move sequences\n")
	fmt.Printf("   • Expected: SCAR modularity → Louvain modularity as k → ∞\n")
	fmt.Printf("   • If convergence fails, SCAR's sketch approximation causes suboptimal moves\n")
}

// ExportableModularityTable for JSON export
type ExportableModularityTable struct {
	MaxMoves      int                    `json:"max_moves"`
	AlgorithmInfo []AlgorithmInfo        `json:"algorithms"`
	ModularityData [][]float64           `json:"modularity_data"`  // [move][algorithm]
}

type AlgorithmInfo struct {
	Name      string `json:"name"`
	Algorithm string `json:"algorithm"`
	K         int    `json:"k"`
	NumMoves  int    `json:"num_moves"`
}

// ExportModularityTable exports the table to a fixed filename
func ExportModularityTable(mt *ModularityTable) error {
	// Prepare algorithm info
	algInfo := make([]AlgorithmInfo, len(mt.Algorithms))
	for i, run := range mt.Algorithms {
		algInfo[i] = AlgorithmInfo{
			Name:      run.Name,
			Algorithm: run.Algorithm,
			K:         run.K,
			NumMoves:  len(run.Moves),
		}
	}
	
	// Create exportable structure
	exportTable := ExportableModularityTable{
		MaxMoves:       mt.MaxMoves,
		AlgorithmInfo:  algInfo,
		ModularityData: mt.Table,
	}
	
	// Write to fixed filename
	file, err := os.Create("modularity_table.json")
	if err != nil {
		return fmt.Errorf("failed to create modularity_table.json: %w", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(exportTable); err != nil {
		return fmt.Errorf("failed to encode modularity table: %w", err)
	}
	
	fmt.Println("✅ Modularity table exported to modularity_table.json")
	return nil
}
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
	"time"
	
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

// MoveRecord represents a logged move from SCAR
type MoveRecord struct {
	Iteration  int     `json:"iteration"`
	Node       int     `json:"node"`
	FromComm   int     `json:"from_comm"`
	ToComm     int     `json:"to_comm"`
	ModGain    float64 `json:"mod_gain"`
	Modularity float64 `json:"modularity"`
}

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("🔍 SCAR COMMUNITY SKETCH VERIFICATION")
	fmt.Println("====================================")
	fmt.Printf("Graph: %s\n", graphFile)
	fmt.Printf("Properties: %s\n", propertiesFile) 
	fmt.Printf("Path: %s\n", pathFile)

	// Step 1: Run SCAR once to get move sequence
	fmt.Println("\n=== STEP 1: GET MOVE SEQUENCE ===")
	
	config := scar.NewConfig()
	config.Set("scar.k", int64(256))        
	config.Set("scar.nk", int64(1))        
	config.Set("algorithm.random_seed", int64(43)) // Fixed seed
	config.Set("algorithm.max_levels", 1)
	config.Set("algorithm.max_iterations", 50)
	
	// Enable move tracking
	moveTrackingFile := fmt.Sprintf("moves_%d.jsonl", time.Now().Unix())
	config.Set("analysis.track_moves", true)
	config.Set("analysis.output_file", moveTrackingFile)
	defer os.Remove(moveTrackingFile)

	// Run SCAR to get moves
	_, err := scar.Run(graphFile, propertiesFile, pathFile, config, context.Background())
	if err != nil {
		log.Fatalf("SCAR run failed: %v", err)
	}

	// Parse moves
	moves, err := parseMoveLog(moveTrackingFile)
	if err != nil {
		log.Fatalf("Failed to parse moves: %v", err)
	}
	
	fmt.Printf("✅ Captured %d moves\n", len(moves))
	if len(moves) == 0 {
		fmt.Println("No moves to verify")
		return
	}

	// Step 2: Build sketch graph (used by both methods)
	fmt.Println("\n=== STEP 2: BUILD SKETCH GRAPH ===")
	
	sketchGraph, _, err := scar.BuildSketchGraph(graphFile, propertiesFile, pathFile, config, config.CreateLogger())
	if err != nil {
		log.Fatalf("Failed to build sketch graph: %v", err)
	}
	
	fmt.Printf("✅ Built sketch graph: %d nodes\n", sketchGraph.NumNodes)

	// Step 3: Initialize both community structures
	fmt.Println("\n=== STEP 3: INITIALIZE COMMUNITIES ===")
	
	// SCAR community (uses built-in sketch management)
	scarComm := scar.NewCommunity(sketchGraph)
	
	// Naive community (same initial state, but we'll track manually)
	naiveComm := scar.NewCommunity(sketchGraph)
	
	fmt.Printf("✅ Both communities initialized with %d nodes\n", sketchGraph.NumNodes)

	// Step 4: Apply moves and verify sketches
	fmt.Println("\n=== STEP 4: APPLY MOVES AND VERIFY ===")
	
	verifiedMoves := 0
	
	for i, move := range moves {
		fmt.Printf("\n--- Move %d/%d: Node %d (%d→%d) ---\n", 
			i+1, len(moves), move.Node, move.FromComm, move.ToComm)
		
		// Validate move
		if move.Node < 0 || move.Node >= sketchGraph.NumNodes {
			fmt.Printf("SKIP: Invalid node %d\n", move.Node)
			continue
		}
		
		if move.FromComm < 0 || move.FromComm >= scarComm.NumCommunities ||
		   move.ToComm < 0 || move.ToComm >= scarComm.NumCommunities {
			fmt.Printf("SKIP: Invalid communities (%d→%d)\n", move.FromComm, move.ToComm)
			continue
		}
		
		if scarComm.NodeToCommunity[move.Node] != move.FromComm {
			fmt.Printf("SKIP: Node %d in community %d, not %d\n", 
				move.Node, scarComm.NodeToCommunity[move.Node], move.FromComm)
			continue
		}
		
		if len(scarComm.CommunityNodes[move.FromComm]) == 0 {
			fmt.Printf("SKIP: Source community %d is empty\n", move.FromComm)
			continue
		}

		// Apply move to SCAR community
		fmt.Printf("  Applying to SCAR community...\n")
		scar.MoveNode(sketchGraph, scarComm, move.Node, move.FromComm, move.ToComm)
		
		// Apply same move to naive community  
		fmt.Printf("  Applying to naive community...\n")
		scar.MoveNode(sketchGraph, naiveComm, move.Node, move.FromComm, move.ToComm)
		
		// Compare community sketches for both affected communities
		fmt.Printf("  Verifying source community %d...\n", move.FromComm)
		if len(scarComm.CommunityNodes[move.FromComm]) > 0 { // Only if still exists
			if !compareCommunitySketch(sketchGraph, scarComm, naiveComm, move.FromComm, "SOURCE") {
				log.Fatalf("❌ Source community %d sketch mismatch at move %d", move.FromComm, i+1)
			}
		}
		
		fmt.Printf("  Verifying target community %d...\n", move.ToComm)
		if !compareCommunitySketch(sketchGraph, scarComm, naiveComm, move.ToComm, "TARGET") {
			log.Fatalf("❌ Target community %d sketch mismatch at move %d", move.ToComm, i+1)
		}
		
		fmt.Printf("  ✅ Move %d verified\n", i+1)
		verifiedMoves++
	}
	
	fmt.Printf("\n🎉 VERIFICATION COMPLETE!\n")
	fmt.Printf("✅ Successfully verified: %d moves\n", verifiedMoves)
	fmt.Printf("💡 Community sketch building is working correctly!\n")
}

// compareCommunitySketch compares SCAR vs naive community sketch
func compareCommunitySketch(sketchGraph *scar.SketchGraph, scarComm, naiveComm *scar.Community, commId int, label string) bool {
	// Get SCAR community sketch (built-in)
	scarSketch := scarComm.GetCommunitySketch(commId)
	
	// Get naive community sketch (manual aggregation)
	naiveSketch := buildNaiveCommunitySketch(sketchGraph, naiveComm, commId)
	
	// Compare
	if scarSketch == nil && naiveSketch == nil {
		fmt.Printf("    [%s] Both sketches nil ✓\n", label)
		return true
	}
	
	if scarSketch == nil {
		fmt.Printf("    [%s] ❌ SCAR nil, naive has %d layers\n", label, len(naiveSketch))
		return false
	}
	
	if naiveSketch == nil {
		fmt.Printf("    [%s] ❌ Naive nil, SCAR exists\n", label)
		return false
	}
	
	// Compare dimensions
	if scarSketch.GetNk() != int64(len(naiveSketch)) {
		fmt.Printf("    [%s] ❌ Layer mismatch: SCAR=%d, Naive=%d\n", 
			label, scarSketch.GetNk(), len(naiveSketch))
		return false
	}
	
	// Compare each layer
	for layer := int64(0); layer < scarSketch.GetNk(); layer++ {
		scarLayer := scarSketch.GetSketch(layer)
		naiveLayer := naiveSketch[layer]
		
		k := scarSketch.GetK()
		
		// Pad naive layer to k length with MaxUint32
		paddedNaive := make([]uint32, k)
		for i := int64(0); i < k; i++ {
			if i < int64(len(naiveLayer)) {
				paddedNaive[i] = naiveLayer[i]
			} else {
				paddedNaive[i] = math.MaxUint32
			}
		}
		
		// Compare values
		for i := int64(0); i < k; i++ {
			if scarLayer[i] != paddedNaive[i] {
				fmt.Printf("    [%s] ❌ Community %d Layer %d[%d]: SCAR=%d, Naive=%d\n", 
					label, commId, layer, i, scarLayer[i], paddedNaive[i])
				
				// Show community members for debugging
				fmt.Printf("    [%s] Community %d members: %v\n", 
					label, commId, scarComm.CommunityNodes[commId])
				
				// Show full sketches (first 10 elements)
				fmt.Printf("    [%s] SCAR:  %v\n", label, scarLayer[:min(int(k), 10)])
				fmt.Printf("    [%s] Naive: %v\n", label, paddedNaive[:min(int(k), 10)])
				
				return false
			}
		}
	}
	
	fmt.Printf("    [%s] Community %d sketch matches ✓\n", label, commId)
	return true
}

// buildNaiveCommunitySketch manually aggregates hashes from community members
func buildNaiveCommunitySketch(sketchGraph *scar.SketchGraph, community *scar.Community, commId int) [][]uint32 {
	memberNodes := community.CommunityNodes[commId]
	if len(memberNodes) == 0 {
		return nil
	}
	
	// Get dimensions from first member
	firstSketch := sketchGraph.GetSketchManager().GetVertexSketch(int64(memberNodes[0]))
	if firstSketch == nil {
		return nil
	}
	
	k := firstSketch.GetK()
	nk := firstSketch.GetNk()
	
	// Collect all hashes per layer from all members
	layerHashes := make([][]uint32, nk)
	
	for _, memberNode := range memberNodes {
		memberSketch := sketchGraph.GetSketchManager().GetVertexSketch(int64(memberNode))
		if memberSketch == nil {
			continue
		}
		
		for layer := int64(0); layer < nk; layer++ {
			hashes := memberSketch.GetLayerHashes(layer)
			layerHashes[layer] = append(layerHashes[layer], hashes...)
		}
	}
	
	// Sort and take bottom-k per layer
	result := make([][]uint32, nk)
	for layer := int64(0); layer < nk; layer++ {
		if len(layerHashes[layer]) == 0 {
			result[layer] = []uint32{}
			continue
		}
		
		// Sort all hashes
		sort.Slice(layerHashes[layer], func(i, j int) bool {
			return layerHashes[layer][i] < layerHashes[layer][j]
		})
		
		// Remove duplicates and take bottom-k
		unique := []uint32{}
		prev := uint32(math.MaxUint32)
		for _, hash := range layerHashes[layer] {
			if hash != prev && hash != math.MaxUint32 {
				unique = append(unique, hash)
				prev = hash
				if len(unique) >= int(k) {
					break
				}
			}
		}
		
		result[layer] = unique
	}
	
	return result
}

// parseMoveLog parses JSONL move log
func parseMoveLog(filename string) ([]MoveRecord, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var moves []MoveRecord
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		var move MoveRecord
		if err := json.Unmarshal(scanner.Bytes(), &move); err != nil {
			continue // Skip malformed lines
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
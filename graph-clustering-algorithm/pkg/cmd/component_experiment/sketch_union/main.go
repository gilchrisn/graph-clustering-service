package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

// ExperimentConfig holds all tunable parameters
type ExperimentConfig struct {
	NumNodes              int     // Number of nodes in the graph
	EdgeProbability       float64 // Erdos-Renyi edge probability
	SketchSize           int64   // Bottom-k sketch size
	NumLayers            int64   // Number of sketch layers (nk)
	CommunityProbability float64 // Probability of including node in community
	NumCommunities       int     // Number of communities to generate
	RandomSeed           int64   // For reproducibility
}

// DefaultExperimentConfig returns reasonable default parameters
func DefaultExperimentConfig() *ExperimentConfig {
	return &ExperimentConfig{
		NumNodes:              1000,
		EdgeProbability:       0.02,  // Sparse graph
		SketchSize:           20,
		NumLayers:            1,      // nk=1 as requested
		CommunityProbability: 0.3,   // 30% chance per node
		NumCommunities:       10,
		RandomSeed:           time.Now().UnixNano(),
	}
}

// ExperimentResults holds the comparison results
type ExperimentResults struct {
	TotalCommunities    int     `json:"total_communities"`
	AvgCommunitySize    float64 `json:"avg_community_size"`
	
	// Method 1: Sum of individual estimates
	Method1MAE          float64 `json:"method1_mae"`
	Method1RMSE         float64 `json:"method1_rmse"`
	Method1MeanError    float64 `json:"method1_mean_error"`
	Method1StdError     float64 `json:"method1_std_error"`
	
	// Method 2: Union sketch estimate
	Method2MAE          float64 `json:"method2_mae"`
	Method2RMSE         float64 `json:"method2_rmse"`
	Method2MeanError    float64 `json:"method2_mean_error"`
	Method2StdError     float64 `json:"method2_std_error"`
	
	// Comparison
	Method1WinCount     int     `json:"method1_win_count"`
	Method2WinCount     int     `json:"method2_win_count"`
	TieCount           int     `json:"tie_count"`
}

// CommunityData holds information about a single community
type CommunityData struct {
	Members       []int   `json:"members"`
	TrueDegree    float64 `json:"true_degree"`
	Method1Est    float64 `json:"method1_estimate"`
	Method2Est    float64 `json:"method2_estimate"`
	Method1Error  float64 `json:"method1_error"`
	Method2Error  float64 `json:"method2_error"`
}

// RawGraph represents the input graph structure (simplified from your code)
type RawGraph struct {
	NumNodes  int
	Adjacency [][]int64
}

// buildSketchGraphFromExperiment adapts your buildSketchGraphFromSketches for experiment use
func buildSketchGraphFromExperiment(rawGraph *RawGraph, sketches []uint32, nodeHashValues []uint32, config *ExperimentConfig) (*scar.SketchGraph, error) {
	n := rawGraph.NumNodes
	finalK := config.SketchSize
	nk := config.NumLayers
	
	// Create sketch graph (no compression - use all nodes)
	sketchGraph := scar.NewSketchGraph(n)
	sketchManager := scar.NewSketchManager(finalK, nk)
	sketchGraph.SetSketchManager(sketchManager)
	
	// Pre-build per-node self-hash set
	selfHashes := make([]map[uint32]struct{}, n)
	for i := int64(0); i < int64(n); i++ {
		m := make(map[uint32]struct{}, nk)
		for j := int64(0); j < nk; j++ {
			hv := nodeHashValues[i*nk+j]
			if hv != 0 {
				m[hv] = struct{}{}
			}
		}
		selfHashes[i] = m
	}
	
	// Convert sketches to VertexBottomKSketch objects - for ALL nodes (no compression)
	for nodeId := 0; nodeId < n; nodeId++ {
		if nodeHashValues[int64(nodeId)*nk] != 0 { // Node has sketches
			sketch := scar.NewVertexBottomKSketch(int64(nodeId), finalK, nk)
			
			// Fill sketch layers: extract neighbor hashes, drop self-hashes
			for layer := int64(0); layer < nk; layer++ {
				out := make([]uint32, 0, finalK)
				
				// Extract from sketch array (simplified - no propK, just use finalK)
				for ki := int64(0); ki < finalK; ki++ {
					idx := layer*int64(n)*finalK + int64(nodeId)*finalK + ki
					if int(idx) >= len(sketches) {
						break
					}
					
					v := sketches[idx]
					if v == math.MaxUint32 {
						break // remaining are empty
					}
					
					// Drop self-hash
					if _, isSelf := selfHashes[nodeId][v]; isSelf {
						continue
					}
					
					// Optional dedup within layer
					if len(out) > 0 && v == out[len(out)-1] {
						continue
					}
					
					out = append(out, v)
					if len(out) == int(finalK) {
						break
					}
				}
				
				// Pad if under-filled to finalK size
				for idx := int64(0); idx < finalK; idx++ {
					if idx < int64(len(out)) {
						sketch.SetSketchValue(layer, idx, out[idx])
					} else {
						sketch.SetSketchValue(layer, idx, math.MaxUint32)
					}
				}
			}
			
			sketch.UpdateFilledCount()
			
			// Store using original nodeId (no compression)
			sketchManager.SetVertexSketch(int64(nodeId), sketch)
			
			// Build hash-to-node mapping from node seeds
			for j := int64(0); j < nk; j++ {
				hashValue := nodeHashValues[int64(nodeId)*nk+j]
				if hashValue != 0 {
					sketchManager.SetHashToNode(hashValue, int64(nodeId))
				}
			}
		}
	}
	
	// Build adjacency list (simplified version of your buildAdjacencyList)
	totalWeight := 0.0
	for nodeId := 0; nodeId < n; nodeId++ {
		for _, neighbor := range rawGraph.Adjacency[nodeId] {
			sketchGraph.AddEdgeToAdjacencyList(nodeId, int(neighbor), 1.0)
		}
		totalWeight += sketchGraph.GetDegree(nodeId)
	}
	sketchGraph.SetTotalWeight(totalWeight / 2.0) // Undirected graph
	
	return sketchGraph, nil
}

// RunCommunityDegreeExperiment runs the complete experiment
func RunCommunityDegreeExperiment(config *ExperimentConfig) (*ExperimentResults, []CommunityData, error) {
	rng := rand.New(rand.NewSource(config.RandomSeed))
	
	fmt.Printf("Starting experiment with %d nodes, %.3f edge probability, k=%d\n", 
		config.NumNodes, config.EdgeProbability, config.SketchSize)
	
	// Step 1: Generate Erdos-Renyi graph
	fmt.Println("Generating Erdos-Renyi graph...")
	rawGraph := generateErdosRenyiGraph(config.NumNodes, config.EdgeProbability, rng)
	
	// Step 2: Assign random hashes and build sketches
	fmt.Println("Building sketches...")
	sketches, nodeHashValues := buildSketchesFromGraph(rawGraph, config, rng)
	
	// Step 3: Build sketch graph using your pattern
	fmt.Println("Building sketch graph...")
	sketchGraph, err := buildSketchGraphFromExperiment(rawGraph, sketches, nodeHashValues, config)
	if err != nil {
		return nil, nil, err
	}
	
	// Step 4: Generate random communities
	fmt.Println("Generating random communities...")
	communities := generateRandomCommunities(config.NumNodes, config.NumCommunities, 
		config.CommunityProbability, rng)
	
	// Step 5: Calculate true community degrees
	fmt.Println("Calculating true and estimated community degrees...")
	communityData := make([]CommunityData, len(communities))
	for i, members := range communities {
		communityData[i] = CommunityData{
			Members:    members,
			TrueDegree: calculateTrueCommunityDegree(members, rawGraph),
		}
		
		// Method 1: Sum of individual estimates
		method1Est := 0.0
		for _, nodeId := range members {
			method1Est += sketchGraph.GetDegree(nodeId)
		}
		communityData[i].Method1Est = method1Est
		
		// Method 2: Union sketch estimate
		unionSketch := createUnionSketch(sketchGraph, members)
		method2Est := 0.0
		if unionSketch != nil {
			if unionSketch.IsSketchFull() {
				method2Est = unionSketch.EstimateCardinality()
			} else {
				method2Est = method1Est // Use Method 1 if sketch is not full
			}
		}
		communityData[i].Method2Est = method2Est
		
		// Calculate errors
		communityData[i].Method1Error = method1Est - communityData[i].TrueDegree
		communityData[i].Method2Error = method2Est - communityData[i].TrueDegree
	}
	
	// Step 6: Calculate statistics
	fmt.Println("Calculating statistics...")
	results := calculateExperimentStatistics(communityData)
	
	fmt.Printf("Experiment completed! Processed %d communities.\n", len(communityData))
	return results, communityData, nil
}

// generateErdosRenyiGraph creates RawGraph for Erdos-Renyi graph
func generateErdosRenyiGraph(numNodes int, edgeProbability float64, rng *rand.Rand) *RawGraph {
	adjacency := make([][]int64, numNodes)
	
	edgeCount := 0
	for i := 0; i < numNodes; i++ {
		for j := i + 1; j < numNodes; j++ {
			if rng.Float64() < edgeProbability {
				adjacency[i] = append(adjacency[i], int64(j))
				adjacency[j] = append(adjacency[j], int64(i))
				edgeCount++
			}
		}
	}
	
	fmt.Printf("Generated graph with %d edges (density: %.4f)\n", 
		edgeCount, float64(edgeCount*2)/float64(numNodes*(numNodes-1)))
	
	return &RawGraph{
		NumNodes:  numNodes,
		Adjacency: adjacency,
	}
}

// buildSketchesFromGraph creates sketches by unioning neighbor hashes
func buildSketchesFromGraph(rawGraph *RawGraph, config *ExperimentConfig, rng *rand.Rand) ([]uint32, []uint32) {
	n := rawGraph.NumNodes
	finalK := config.SketchSize
	nk := config.NumLayers
	
	// Generate unique random hashes for each node
	nodeHashes := make([]uint32, n)
	usedHashes := make(map[uint32]bool)
	
	for i := 0; i < n; i++ {
		for {
			hash := rng.Uint32()
			if hash != math.MaxUint32 && !usedHashes[hash] {
				nodeHashes[i] = hash
				usedHashes[hash] = true
				break
			}
		}
	}
	
	// Build node hash values (self-hashes for each layer)
	nodeHashValues := make([]uint32, int64(n)*nk)
	for i := 0; i < n; i++ {
		for j := int64(0); j < nk; j++ {
			nodeHashValues[int64(i)*nk+j] = nodeHashes[i]
		}
	}
	
	// Build sketches by unioning neighbor hashes (including self)
	sketches := make([]uint32, int64(n)*nk*finalK)
	for i := range sketches {
		sketches[i] = math.MaxUint32
	}
	
	for nodeId := 0; nodeId < n; nodeId++ {
		// Collect neighbor hashes (including self)
		neighborHashes := []uint32{nodeHashes[nodeId]}
		for _, neighbor := range rawGraph.Adjacency[nodeId] {
			neighborHashes = append(neighborHashes, nodeHashes[neighbor])
		}
		
		// Sort to get bottom-k
		sort.Slice(neighborHashes, func(i, j int) bool {
			return neighborHashes[i] < neighborHashes[j]
		})
		
		// Fill sketch for each layer
		for layer := int64(0); layer < nk; layer++ {
			for ki := int64(0); ki < finalK && ki < int64(len(neighborHashes)); ki++ {
				idx := layer*int64(n)*finalK + int64(nodeId)*finalK + ki
				sketches[idx] = neighborHashes[ki]
			}
		}
	}
	
	return sketches, nodeHashValues
}

// generateRandomCommunities creates random communities
func generateRandomCommunities(numNodes, numCommunities int, probability float64, rng *rand.Rand) [][]int {
	communities := make([][]int, 0, numCommunities)
	
	for c := 0; c < numCommunities; c++ {
		community := make([]int, 0)
		for nodeId := 0; nodeId < numNodes; nodeId++ {
			if rng.Float64() < probability {
				community = append(community, nodeId)
			}
		}
		if len(community) > 0 {
			communities = append(communities, community)
		}
	}
	
	return communities
}

// calculateTrueCommunityDegree calculates the exact community degree
func calculateTrueCommunityDegree(members []int, rawGraph *RawGraph) float64 {
	totalDegree := 0.0
	for _, nodeId := range members {
		// Count all edges from this node
		totalDegree += float64(len(rawGraph.Adjacency[nodeId]))
	}
	return totalDegree
}

// createUnionSketch creates union of all member node sketches
func createUnionSketch(sketchGraph *scar.SketchGraph, members []int) *scar.VertexBottomKSketch {
	if len(members) == 0 {
		return nil
	}
	
	sketchManager := sketchGraph.GetSketchManager()
	firstSketch := sketchManager.GetVertexSketch(int64(members[0]))
	if firstSketch == nil {
		return nil
	}
	
	// Initialize union sketch
	unionSketch := scar.NewVertexBottomKSketch(-1, firstSketch.GetK(), firstSketch.GetNk())
	
	// Union all member sketches
	for _, nodeId := range members {
		nodeSketch := sketchManager.GetVertexSketch(int64(nodeId))
		if nodeSketch != nil {
			for layer := int64(0); layer < nodeSketch.GetNk(); layer++ {
				unionSketch.UnionWithLayer(layer, nodeSketch.GetSketch(layer))
			}
		}
	}
	
	return unionSketch
}

// calculateExperimentStatistics computes summary statistics
func calculateExperimentStatistics(communityData []CommunityData) *ExperimentResults {
	if len(communityData) == 0 {
		return &ExperimentResults{}
	}
	
	results := &ExperimentResults{TotalCommunities: len(communityData)}
	
	// Calculate community size statistics
	totalSize := 0
	for _, cd := range communityData {
		totalSize += len(cd.Members)
	}
	results.AvgCommunitySize = float64(totalSize) / float64(len(communityData))
	
	// Calculate error statistics
	method1Errors := make([]float64, len(communityData))
	method2Errors := make([]float64, len(communityData))
	method1AbsErrors := make([]float64, len(communityData))
	method2AbsErrors := make([]float64, len(communityData))
	
	for i, cd := range communityData {
		method1Errors[i] = cd.Method1Error
		method2Errors[i] = cd.Method2Error
		method1AbsErrors[i] = math.Abs(cd.Method1Error)
		method2AbsErrors[i] = math.Abs(cd.Method2Error)
		
		// Count wins
		if method1AbsErrors[i] < method2AbsErrors[i] {
			results.Method1WinCount++
		} else if method2AbsErrors[i] < method1AbsErrors[i] {
			results.Method2WinCount++
		} else {
			results.TieCount++
		}
	}
	
	// Method 1 statistics
	results.Method1MAE = calculateMean(method1AbsErrors)
	results.Method1RMSE = calculateRMSE(method1Errors)
	results.Method1MeanError = calculateMean(method1Errors)
	results.Method1StdError = calculateStdDev(method1Errors)
	
	// Method 2 statistics
	results.Method2MAE = calculateMean(method2AbsErrors)
	results.Method2RMSE = calculateRMSE(method2Errors)
	results.Method2MeanError = calculateMean(method2Errors)
	results.Method2StdError = calculateStdDev(method2Errors)
	
	return results
}

// Helper functions for statistics
func calculateMean(values []float64) float64 {
	if len(values) == 0 { return 0.0 }
	sum := 0.0
	for _, v := range values { sum += v }
	return sum / float64(len(values))
}

func calculateRMSE(values []float64) float64 {
	if len(values) == 0 { return 0.0 }
	sumSquares := 0.0
	for _, v := range values { sumSquares += v * v }
	return math.Sqrt(sumSquares / float64(len(values)))
}

func calculateStdDev(values []float64) float64 {
	if len(values) <= 1 { return 0.0 }
	mean := calculateMean(values)
	sumSquares := 0.0
	for _, v := range values { sumSquares += (v - mean) * (v - mean) }
	return math.Sqrt(sumSquares / float64(len(values)-1))
}

// PrintResults prints a formatted summary
func PrintResults(results *ExperimentResults) {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("COMMUNITY DEGREE ESTIMATION EXPERIMENT RESULTS")
	fmt.Println(strings.Repeat("=", 60))
	
	fmt.Printf("Total Communities: %d\n", results.TotalCommunities)
	fmt.Printf("Average Community Size: %.2f nodes\n", results.AvgCommunitySize)
	fmt.Println()
	
	fmt.Println("METHOD 1 (Sum of Individual Estimates):")
	fmt.Printf("  MAE:        %.4f\n", results.Method1MAE)
	fmt.Printf("  RMSE:       %.4f\n", results.Method1RMSE)
	fmt.Printf("  Mean Error: %.4f\n", results.Method1MeanError)
	fmt.Printf("  Std Error:  %.4f\n", results.Method1StdError)
	fmt.Println()
	
	fmt.Println("METHOD 2 (Union Sketch Estimate):")
	fmt.Printf("  MAE:        %.4f\n", results.Method2MAE)
	fmt.Printf("  RMSE:       %.4f\n", results.Method2RMSE)
	fmt.Printf("  Mean Error: %.4f\n", results.Method2MeanError)
	fmt.Printf("  Std Error:  %.4f\n", results.Method2StdError)
	fmt.Println()
	
	fmt.Println("COMPARISON:")
	fmt.Printf("  Method 1 Wins: %d (%.1f%%)\n", results.Method1WinCount,
		100.0*float64(results.Method1WinCount)/float64(results.TotalCommunities))
	fmt.Printf("  Method 2 Wins: %d (%.1f%%)\n", results.Method2WinCount,
		100.0*float64(results.Method2WinCount)/float64(results.TotalCommunities))
	fmt.Printf("  Ties:          %d (%.1f%%)\n", results.TieCount,
		100.0*float64(results.TieCount)/float64(results.TotalCommunities))
	
	winnerText := "TIE"
	if results.Method1MAE < results.Method2MAE {
		winnerText = "METHOD 1 (Sum of Individual)"
	} else if results.Method2MAE < results.Method1MAE {
		winnerText = "METHOD 2 (Union Sketch)"
	}
	fmt.Printf("\nOVERALL WINNER (by MAE): %s\n", winnerText)
	fmt.Println(strings.Repeat("=", 60))
}

// main function
func main() {
	fmt.Println("Community Degree Estimation Experiment")
	fmt.Println("======================================")
	
	config := DefaultExperimentConfig()
	
	// Customize parameters
	config.NumNodes = 10000
	config.SketchSize = 4096
	config.NumCommunities = 100
	config.EdgeProbability = 0.2
	config.CommunityProbability = 0.25
	
	fmt.Printf("Configuration:\n")
	fmt.Printf("  Nodes: %d\n", config.NumNodes)
	fmt.Printf("  Edge Probability: %.3f\n", config.EdgeProbability)
	fmt.Printf("  Sketch Size (k): %d\n", config.SketchSize)
	fmt.Printf("  Communities: %d\n", config.NumCommunities)
	fmt.Printf("  Community Inclusion Probability: %.2f\n", config.CommunityProbability)
	fmt.Println()
	
	results, _, err := RunCommunityDegreeExperiment(config)
	if err != nil {
		fmt.Printf("Experiment failed: %v\n", err)
		return
	}
	
	PrintResults(results)
}
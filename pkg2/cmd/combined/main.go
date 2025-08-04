package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	// "time"
	
	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg2/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg2/scar"
)

type ComparisonResult struct {
	LouvainResult *louvain.Result
	ScarResult    *scar.Result
	HMI           float64
	NMI           float64  // Final level comparison
	ARI           float64  // Final level comparison
	Metrics       ComparisonMetrics
}

type ComparisonMetrics struct {
	ModularityDiff       float64
	NumCommunitiesDiff   int
	RuntimeRatio        float64
	BestLevelMatch      BestLevelMatch
}

type BestLevelMatch struct {
	LouvainLevel int
	ScarLevel    int
	NMI          float64
}

func main() {
	fmt.Println("=== Louvain vs SCAR Algorithm Comparison ===")
	
	// Check command line arguments
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}
	
	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]
	
	fmt.Printf("Input files:\n")
	fmt.Printf("  Graph: %s\n", graphFile)
	fmt.Printf("  Properties: %s\n", propertiesFile)
	fmt.Printf("  Path: %s\n", pathFile)
	
	// Run both pipelines
	// fmt.Println("\n" + "="*60)
	louvainResult := runMaterializationLouvainPipeline(graphFile, propertiesFile, pathFile)
	
	// fmt.Println("\n" + "="*60)
	scarResult := runSCARPipeline(graphFile, propertiesFile, pathFile)
	
	// Compare results
	// fmt.Println("\n" + "="*60)
	fmt.Println("COMPARISON ANALYSIS")
	// fmt.Println("="*60)
	
	comparison := compareResults(louvainResult, scarResult)
	displayComparison(comparison)
}

func runMaterializationLouvainPipeline(graphFile, propertiesFile, pathFile string) *louvain.Result {
	fmt.Println("🔵 RUNNING MATERIALIZATION + LOUVAIN PIPELINE")
	
	// Step 1: Parse SCAR input for materialization
	fmt.Println("\nStep 1: Parsing input files...")
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to parse input: %v", err)
	}
	fmt.Printf("  Loaded graph with %d nodes\n", len(graph.Nodes))
	
	// Step 2: Run materialization
	fmt.Println("\nStep 2: Running materialization...")
	config := materialization.DefaultMaterializationConfig()
	config.Aggregation.Strategy = materialization.Average
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
	materializationResult, err := engine.Materialize()
	if err != nil {
		log.Fatalf("Materialization failed: %v", err)
	}
	
	materializedGraph := materializationResult.HomogeneousGraph
	fmt.Printf("  Materialized graph: %d nodes, %d edges\n", 
		len(materializedGraph.Nodes), len(materializedGraph.Edges))

	// Step 3: Convert to Louvain graph format
	fmt.Println("\nStep 3: Converting to Louvain format...")
	louvainGraph := convertToLouvainGraph(materializedGraph)
	fmt.Printf("  Louvain graph: %d nodes, total weight: %.1f\n", 
		louvainGraph.NumNodes, louvainGraph.TotalWeight)
	
	// Step 4: Run Louvain
	fmt.Println("\nStep 4: Running Louvain clustering...")
	louvainConfig := louvain.NewConfig()
	louvainConfig.Set("algorithm.max_iterations", 5)
	louvainConfig.Set("algorithm.min_modularity_gain", 1e-6)
	louvainConfig.Set("logging.level", "info")
	louvainConfig.Set("algorithm.random_seed", int64(42))
	
	ctx := context.Background()
	result, err := louvain.Run(louvainGraph, louvainConfig, ctx)
	if err != nil {
		log.Fatalf("Louvain failed: %v", err)
	}
	
	fmt.Printf("✅ Louvain completed: %d levels, %.6f modularity, %d ms\n", 
		result.NumLevels, result.Modularity, result.Statistics.RuntimeMS)
	
	return result
}

func runSCARPipeline(graphFile, propertiesFile, pathFile string) *scar.Result {
	fmt.Println("🔴 RUNNING SCAR PIPELINE")
	
	// Create configuration with LARGE K for exact computation
	config := scar.NewConfig()
	config.Set("algorithm.max_iterations", 5)
	config.Set("algorithm.min_modularity_gain", 1e-6)
	config.Set("logging.level", "info")
	
	// LARGE K ensures sketches are never full → exact computation (same as Louvain)
	config.Set("scar.k", 512)    // Large K for exact computation
	config.Set("scar.nk", 1)     // Multiple layers
	config.Set("scar.threshold", 0.0)
	config.Set("algorithm.random_seed", int64(42)) 
	
	fmt.Printf("\nSCAR Configuration:\n")
	fmt.Printf("  K: %d (large → exact computation)\n", config.K())
	fmt.Printf("  NK: %d\n", config.NK())
	fmt.Printf("  Max Iterations: %d\n", config.MaxIterations())
	
	// Run algorithm
	ctx := context.Background()
	result, err := scar.Run(graphFile, propertiesFile, pathFile, config, ctx)
	if err != nil {
		log.Fatalf("SCAR failed: %v", err)
	}
	
	fmt.Printf("✅ SCAR completed: %d levels, %.6f modularity, %d ms\n", 
		result.NumLevels, result.Modularity, result.Statistics.RuntimeMS)
	
	return result
}

func compareResults(louvainResult *louvain.Result, scarResult *scar.Result) *ComparisonResult {
	fmt.Println("📊 Computing comparison metrics...")
	
	// Calculate HMI
	hmi := calculateHierarchicalMutualInformation(louvainResult.Levels, scarResult.Levels)
	
	// Calculate final level NMI and ARI
	finalNMI := calculateNMI(louvainResult.FinalCommunities, scarResult.FinalCommunities)
	finalARI := calculateAdjustedRandIndex(louvainResult.FinalCommunities, scarResult.FinalCommunities)
	
	// Find best level match
	bestMatch := findBestLevelMatch(louvainResult.Levels, scarResult.Levels)
	
	// Calculate other metrics
	metrics := ComparisonMetrics{
		ModularityDiff:     louvainResult.Modularity - scarResult.Modularity,
		NumCommunitiesDiff: countFinalCommunities(louvainResult.FinalCommunities) - countFinalCommunities(scarResult.FinalCommunities),
		RuntimeRatio:       float64(louvainResult.Statistics.RuntimeMS) / float64(scarResult.Statistics.RuntimeMS),
		BestLevelMatch:     bestMatch,
	}
	
	return &ComparisonResult{
		LouvainResult: louvainResult,
		ScarResult:    scarResult,
		HMI:           hmi,
		NMI:           finalNMI,
		ARI:           finalARI,
		Metrics:       metrics,
	}
}

func calculateHierarchicalMutualInformation(louvainLevels []louvain.LevelInfo, scarLevels []scar.LevelInfo) float64 {
	// Simplified HMI implementation
	// In practice, you'd want a more sophisticated implementation
	
	totalMI := 0.0
	comparisons := 0
	
	// Compare each level combination and weight by level importance
	for i, louvainLevel := range louvainLevels {
		for j, scarLevel := range scarLevels {
			// Convert to comparable format
			louvainCommunities := convertLouvainLevelToCommunityMap(louvainLevel)
			scarCommunities := convertScarLevelToCommunityMap(scarLevel)
			
			mi := calculateNMI(louvainCommunities, scarCommunities)
			
			// Weight by inverse of level difference (prefer similar hierarchy levels)
			levelDiff := float64(abs(i - j))
			weight := 1.0 / (1.0 + levelDiff)
			
			totalMI += mi * weight
			comparisons++
		}
	}
	
	if comparisons == 0 {
		return 0.0
	}
	
	return totalMI / float64(comparisons)
}

func findBestLevelMatch(louvainLevels []louvain.LevelInfo, scarLevels []scar.LevelInfo) BestLevelMatch {
	bestNMI := -1.0
	bestLouvainLevel := -1
	bestScarLevel := -1
	
	for i, louvainLevel := range louvainLevels {
		for j, scarLevel := range scarLevels {
			louvainCommunities := convertLouvainLevelToCommunityMap(louvainLevel)
			scarCommunities := convertScarLevelToCommunityMap(scarLevel)
			
			nmi := calculateNMI(louvainCommunities, scarCommunities)
			
			if nmi > bestNMI {
				bestNMI = nmi
				bestLouvainLevel = i
				bestScarLevel = j
			}
		}
	}
	
	return BestLevelMatch{
		LouvainLevel: bestLouvainLevel,
		ScarLevel:    bestScarLevel,
		NMI:          bestNMI,
	}
}

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
	
	return 2 * mi / (h1 + h2)
}

func calculateAdjustedRandIndex(communities1, communities2 map[int]int) float64 {
	// Simplified ARI implementation
	// In practice, you'd want a more robust implementation
	
	// Build contingency table (same as NMI)
	allNodes := make(map[int]bool)
	for node := range communities1 {
		allNodes[node] = true
	}
	for node := range communities2 {
		allNodes[node] = true
	}
	
	contingency := make(map[int]map[int]int)
	for node := range allNodes {
		comm1, exists1 := communities1[node]
		comm2, exists2 := communities2[node]
		
		if !exists1 || !exists2 {
			continue
		}
		
		if contingency[comm1] == nil {
			contingency[comm1] = make(map[int]int)
		}
		contingency[comm1][comm2]++
	}
	
	// Calculate ARI components
	n := len(allNodes)
	if n <= 1 {
		return 1.0
	}
	
	// This is a simplified version - full ARI calculation is more complex
	agreements := 0
	total := 0
	
	for node1 := range allNodes {
		for node2 := range allNodes {
			if node1 >= node2 {
				continue
			}
			
			comm1_1, exists1_1 := communities1[node1]
			comm1_2, exists1_2 := communities1[node2]
			comm2_1, exists2_1 := communities2[node1]
			comm2_2, exists2_2 := communities2[node2]
			
			if !exists1_1 || !exists1_2 || !exists2_1 || !exists2_2 {
				continue
			}
			
			sameInFirst := (comm1_1 == comm1_2)
			sameInSecond := (comm2_1 == comm2_2)
			
			if sameInFirst == sameInSecond {
				agreements++
			}
			total++
		}
	}
	
	if total == 0 {
		return 1.0
	}
	
	return float64(agreements) / float64(total)
}

func displayComparison(comparison *ComparisonResult) {
	fmt.Printf("\n🎯 HIERARCHICAL MUTUAL INFORMATION: %.4f\n", comparison.HMI)
	fmt.Printf("📊 FINAL LEVEL COMPARISON:\n")
	fmt.Printf("   Normalized Mutual Information: %.4f\n", comparison.NMI)
	fmt.Printf("   Adjusted Rand Index: %.4f\n", comparison.ARI)
	
	fmt.Printf("\n📈 ALGORITHM PERFORMANCE:\n")
	fmt.Printf("   Modularity Difference (L-S): %+.6f\n", comparison.Metrics.ModularityDiff)
	fmt.Printf("   Community Count Difference: %+d\n", comparison.Metrics.NumCommunitiesDiff)
	fmt.Printf("   Runtime Ratio (L/S): %.2fx\n", comparison.Metrics.RuntimeRatio)
	
	fmt.Printf("\n🔍 BEST LEVEL MATCH:\n")
	fmt.Printf("   Louvain Level %d ↔ SCAR Level %d\n", 
		comparison.Metrics.BestLevelMatch.LouvainLevel, 
		comparison.Metrics.BestLevelMatch.ScarLevel)
	fmt.Printf("   NMI at best match: %.4f\n", comparison.Metrics.BestLevelMatch.NMI)
	
	fmt.Printf("\n📋 DETAILED RESULTS:\n")
	
	fmt.Printf("\n  🔵 LOUVAIN:\n")
	fmt.Printf("     Levels: %d\n", comparison.LouvainResult.NumLevels)
	fmt.Printf("     Final Modularity: %.6f\n", comparison.LouvainResult.Modularity)
	fmt.Printf("     Runtime: %d ms\n", comparison.LouvainResult.Statistics.RuntimeMS)
	fmt.Printf("     Total Moves: %d\n", comparison.LouvainResult.Statistics.TotalMoves)
	fmt.Printf("     Final Communities: %d\n", countFinalCommunities(comparison.LouvainResult.FinalCommunities))
	
	fmt.Printf("\n  🔴 SCAR:\n")
	fmt.Printf("     Levels: %d\n", comparison.ScarResult.NumLevels)
	fmt.Printf("     Final Modularity: %.6f\n", comparison.ScarResult.Modularity)
	fmt.Printf("     Runtime: %d ms\n", comparison.ScarResult.Statistics.RuntimeMS)
	fmt.Printf("     Total Moves: %d\n", comparison.ScarResult.Statistics.TotalMoves)
	fmt.Printf("     Final Communities: %d\n", countFinalCommunities(comparison.ScarResult.FinalCommunities))
	
	// Show level-by-level comparison matrix (if not too large)
	if len(comparison.LouvainResult.Levels) <= 5 && len(comparison.ScarResult.Levels) <= 5 {
		fmt.Printf("\n📊 LEVEL-BY-LEVEL NMI MATRIX:\n")
		fmt.Printf("        ")
		for j := range comparison.ScarResult.Levels {
			fmt.Printf("SCAR-L%d  ", j)
		}
		fmt.Printf("\n")
		
		for i, louvainLevel := range comparison.LouvainResult.Levels {
			fmt.Printf("Louv-L%d ", i)
			for _, scarLevel := range comparison.ScarResult.Levels {
				louvainCommunities := convertLouvainLevelToCommunityMap(louvainLevel)
				scarCommunities := convertScarLevelToCommunityMap(scarLevel)
				nmi := calculateNMI(louvainCommunities, scarCommunities)
				fmt.Printf("  %.3f  ", nmi)
			}
			fmt.Printf("\n")
		}
	}
}

// Helper functions

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

func convertLouvainLevelToCommunityMap(level louvain.LevelInfo) map[int]int {
	result := make(map[int]int)
	for commID, nodes := range level.Communities {
		for _, node := range nodes {
			result[node] = commID
		}
	}
	return result
}

func convertScarLevelToCommunityMap(level scar.LevelInfo) map[int]int {
	result := make(map[int]int)
	for commID, nodes := range level.Communities {
		for _, node := range nodes {
			result[node] = commID
		}
	}
	return result
}

func countFinalCommunities(communities map[int]int) int {
	communitySet := make(map[int]bool)
	for _, commID := range communities {
		communitySet[commID] = true
	}
	return len(communitySet)
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
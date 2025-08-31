package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg2/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg2/louvain"
)

type GraphStats struct {
	Name            string
	K               int     // For sketch graphs
	NumNodes        int
	NumEdges        int
	AvgDegree       float64
	StdDevDegree    float64
	MinDegree       float64
	MaxDegree       float64
	MedianDegree    float64
	P25Degree       float64 // 25th percentile
	P75Degree       float64 // 75th percentile
	P90Degree       float64 // 90th percentile
	P95Degree       float64 // 95th percentile
	Density         float64
	MaxPossibleEdges int64
}

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("📊 GRAPH STATISTICS COMPARISON")
	fmt.Println("==============================")

	// Step 1: Materialize the reference graph
	fmt.Println("📈 Computing reference graph statistics...")
	materializedGraph, _, err := materializeReferenceGraph(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to materialize graph: %v", err)
	}

	refStats := computeGraphStats(materializedGraph, "Reference", 0)
	fmt.Printf("✅ Reference graph: %d nodes, %d edges\n", refStats.NumNodes, refStats.NumEdges)

	// Step 2: Compute sketch graph statistics for different k values
	kValues := []int{2, 16, 32, 64, 128, 256, 512, 1024}
	allStats := []GraphStats{refStats}

	fmt.Println("🎯 Computing SCAR sketch graph statistics...")
	for _, k := range kValues {
		fmt.Printf("  Computing k=%d...", k)
		
		sketchStats, err := computeSCARStats(graphFile, propertiesFile, pathFile, k)
		if err != nil {
			fmt.Printf(" ❌ Failed: %v\n", err)
			continue
		}
		
		allStats = append(allStats, sketchStats)
		fmt.Printf(" ✅ %d nodes, %d edges\n", sketchStats.NumNodes, sketchStats.NumEdges)
	}

	// Step 3: Display comprehensive comparison table
	fmt.Println("\n📋 COMPREHENSIVE STATISTICS COMPARISON")
	displayStatsTable(allStats)
	
	// Step 4: Display analysis
	fmt.Println("\n🔍 COMPARATIVE ANALYSIS")
	displayAnalysis(allStats)
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

	// Convert to Louvain graph
	hgraph := result.HomogeneousGraph
	nodeList := make([]string, 0, len(hgraph.Nodes))
	for nodeID := range hgraph.Nodes {
		nodeList = append(nodeList, nodeID)
	}

	// Intelligent sorting
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

func computeSCARStats(graphFile, propertiesFile, pathFile string, k int) (GraphStats, error) {
	// Build SCAR sketch graph
	config := scar.NewConfig()
	config.Set("scar.k", int64(k))
	config.Set("scar.nk", int64(1))
	config.Set("algorithm.random_seed", int64(42))

	logger := config.CreateLogger()
	sketchGraph, _, err := scar.BuildSketchGraph(graphFile, propertiesFile, pathFile, config, logger)
	if err != nil {
		return GraphStats{}, fmt.Errorf("sketch graph build failed: %w", err)
	}

	return computeGraphStats(sketchGraph, fmt.Sprintf("SCAR-k%d", k), k), nil
}

func computeGraphStats(graph interface{}, name string, k int) GraphStats {
	var numNodes, numEdges int
	var degrees []float64
	var totalWeight float64

	// Handle different graph types explicitly
	switch g := graph.(type) {
	case *louvain.Graph:
		numNodes = g.NumNodes
		totalWeight = g.TotalWeight
		numEdges = int(totalWeight) // Since TotalWeight is sum of edge weights, and each edge weight is typically 1
		degrees = make([]float64, len(g.Degrees))
		copy(degrees, g.Degrees)
		
	case *scar.SketchGraph:
		numNodes = g.NumNodes
		totalWeight = g.TotalWeight
		numEdges = int(totalWeight) // Approximation
		degrees = make([]float64, numNodes)
		for i := 0; i < numNodes; i++ {
			degrees[i] = g.GetDegree(i)
		}
		
	default:
		// Unknown graph type
		return GraphStats{Name: name, K: k}
	}

	// Sort degrees for percentile calculations
	sortedDegrees := make([]float64, len(degrees))
	copy(sortedDegrees, degrees)
	sort.Float64s(sortedDegrees)

	// Calculate statistics
	stats := GraphStats{
		Name:     name,
		K:        k,
		NumNodes: numNodes,
		NumEdges: numEdges,
	}

	if len(degrees) > 0 {
		// Basic stats
		sum := 0.0
		for _, d := range degrees {
			sum += d
		}
		stats.AvgDegree = sum / float64(len(degrees))

		// Standard deviation
		sumSq := 0.0
		for _, d := range degrees {
			diff := d - stats.AvgDegree
			sumSq += diff * diff
		}
		stats.StdDevDegree = math.Sqrt(sumSq / float64(len(degrees)))

		// Min/Max
		stats.MinDegree = sortedDegrees[0]
		stats.MaxDegree = sortedDegrees[len(sortedDegrees)-1]

		// Percentiles
		stats.P25Degree = percentile(sortedDegrees, 0.25)
		stats.MedianDegree = percentile(sortedDegrees, 0.50)
		stats.P75Degree = percentile(sortedDegrees, 0.75)
		stats.P90Degree = percentile(sortedDegrees, 0.90)
		stats.P95Degree = percentile(sortedDegrees, 0.95)
	}

	// Graph density
	stats.MaxPossibleEdges = int64(numNodes) * int64(numNodes-1) / 2
	if stats.MaxPossibleEdges > 0 {
		stats.Density = float64(numEdges) / float64(stats.MaxPossibleEdges)
	}

	return stats
}

func percentile(sortedData []float64, p float64) float64 {
	if len(sortedData) == 0 {
		return 0
	}
	index := p * float64(len(sortedData)-1)
	lower := int(index)
	upper := lower + 1
	
	if upper >= len(sortedData) {
		return sortedData[len(sortedData)-1]
	}
	
	weight := index - float64(lower)
	return sortedData[lower]*(1-weight) + sortedData[upper]*weight
}

func displayStatsTable(allStats []GraphStats) {
	fmt.Println("=" + strings.Repeat("=", 120))
	
	// Header
	fmt.Printf("%-12s %-6s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-10s\n",
		"Graph", "K", "Nodes", "Edges", "AvgDeg", "StdDev", "MinDeg", "MaxDeg", "P25", "Median", "P75", "P95", "Density")
	fmt.Println("-" + strings.Repeat("-", 120))

	// Data rows
	for _, stats := range allStats {
		kStr := "-"
		if stats.K > 0 {
			kStr = fmt.Sprintf("%d", stats.K)
		}
		
		fmt.Printf("%-12s %-6s %-8d %-8d %-8.2f %-8.2f %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f %-10.6f\n",
			truncateString(stats.Name, 12),
			kStr,
			stats.NumNodes,
			stats.NumEdges,
			stats.AvgDegree,
			stats.StdDevDegree,
			stats.MinDegree,
			stats.MaxDegree,
			stats.P25Degree,
			stats.MedianDegree,
			stats.P75Degree,
			stats.P95Degree,
			stats.Density)
	}
	fmt.Println("=" + strings.Repeat("=", 120))
}

func displayAnalysis(allStats []GraphStats) {
	if len(allStats) < 2 {
		return
	}

	refStats := allStats[0]
	fmt.Printf("🎯 Reference Graph Summary:\n")
	fmt.Printf("   • %d nodes, %d edges (density: %.6f)\n", 
		refStats.NumNodes, refStats.NumEdges, refStats.Density)
	fmt.Printf("   • Average degree: %.2f ± %.2f\n", 
		refStats.AvgDegree, refStats.StdDevDegree)
	fmt.Printf("   • Degree range: [%.1f, %.1f]\n", 
		refStats.MinDegree, refStats.MaxDegree)

	fmt.Printf("\n📈 SCAR Approximation Quality:\n")
	fmt.Printf("%-6s %-12s %-12s %-12s %-12s %-15s\n",
		"K", "Node Ratio", "Edge Ratio", "Avg Deg Ratio", "Density Ratio", "Degree Preserve")
	fmt.Printf("%s\n", strings.Repeat("-", 75))

	for _, stats := range allStats[1:] {
		nodeRatio := float64(stats.NumNodes) / float64(refStats.NumNodes)
		edgeRatio := float64(stats.NumEdges) / float64(refStats.NumEdges)
		avgDegRatio := stats.AvgDegree / refStats.AvgDegree
		densityRatio := stats.Density / refStats.Density
		
		// Simple degree preservation metric
		degreePreserve := 1.0 - math.Abs(avgDegRatio-1.0)
		
		fmt.Printf("%-6d %-12.3f %-12.3f %-12.3f %-12.3f %-15.3f\n",
			stats.K, nodeRatio, edgeRatio, avgDegRatio, densityRatio, degreePreserve)
	}

	fmt.Printf("\n💡 Key Insights:\n")
	fmt.Printf("   • Node Ratio: Fraction of original nodes retained\n")
	fmt.Printf("   • Edge Ratio: Fraction of original edges retained\n")
	fmt.Printf("   • Avg Deg Ratio: How well average degree is preserved (1.0 = perfect)\n")
	fmt.Printf("   • Density Ratio: How graph density changes\n")
	fmt.Printf("   • Degree Preserve: Overall degree distribution preservation (1.0 = perfect)\n")
	
	// Find best k value
	bestK := 0
	bestScore := -1.0
	for _, stats := range allStats[1:] {
		avgDegRatio := stats.AvgDegree / refStats.AvgDegree
		score := 1.0 - math.Abs(avgDegRatio-1.0)
		if score > bestScore {
			bestScore = score
			bestK = stats.K
		}
	}
	
	if bestK > 0 {
		fmt.Printf("\n🏆 Best k value for degree preservation: k=%d (score: %.3f)\n", bestK, bestScore)
	}
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
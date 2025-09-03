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
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
)

type DegreeDistribution struct {
	Degrees []float64
	Mean    float64
	StdDev  float64
	Min     float64
	Max     float64
	Median  float64
}

type SimpleComparison struct {
	RMSE        float64  // Root Mean Square Error
	MAE         float64  // Mean Absolute Error
	MAPE        float64  // Mean Absolute Percentage Error
	PearsonCorr float64  // Pearson correlation
	R2Score     float64  // R-squared score
}

type ExperimentResult struct {
	K                    int
	MaterializedStats    DegreeDistribution
	SketchStats          DegreeDistribution
	Comparison           SimpleComparison
	NumNodes             int
	NumTargetNodes       int
}

type NodeSketchDebug struct {
	OriginalID      int
	CompressedID    int
	IsSketchFull    bool
	FilledCount     int64
	EstimatedDegree float64
	ExactDegree     float64
	SketchLayers    [][]uint32
	OwnHashes       []uint32
}

type SketchDebugReport struct {
	K                int64
	NK               int64
	TotalNodes       int
	TargetNodes      int
	NodeDetails      []NodeSketchDebug
	HashMappings     map[uint32]int64
	GlobalStats      struct {
		FullSketches    int
		NonFullSketches int
		EmptySketches   int
		AvgFilledCount  float64
	}
}

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("🔬 SCAR SKETCH DEBUGGING WITH FILE OUTPUT")
	fmt.Println("=========================================")

	// Step 1: Materialize the reference graph
	fmt.Println("Step 1: Materializing reference graph...")
	materializedGraph, nodeMapping, err := materializeReferenceGraph(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to materialize graph: %v", err)
	}

	refDegrees := computeLouvainDegrees(materializedGraph)
	refStats := computeDegreeStats(refDegrees)

	fmt.Printf("📊 Reference: %d nodes, mean degree=%.2f±%.2f, range=[%.2f, %.2f]\n",
		len(refDegrees), refStats.Mean, refStats.StdDev, refStats.Min, refStats.Max)

	// Step 2: Test with detailed debugging for specific k values
	debugKValues := []int{4, 16, 64, 256, 1024}
	results := make([]ExperimentResult, 0)

	fmt.Println("\nStep 2: Testing SCAR with file output...")
	
	for _, k := range debugKValues {
		fmt.Printf("Testing k=%d... ", k)
		
		debugReport, result, err := debugSCARSketchGraph(graphFile, propertiesFile, pathFile, k, refStats, nodeMapping)
		if err != nil {
			fmt.Printf("❌ Failed: %v\n", err)
			continue
		}

		// Write detailed debug info to files
		err = writeDebugReportToFiles(debugReport, result, k)
		if err != nil {
			fmt.Printf("⚠️  Warning: Failed to write debug files: %v\n", err)
		}

		results = append(results, result)
		fmt.Printf("✅ RMSE=%.2f, MAE=%.2f, Corr=%.3f\n", 
			result.Comparison.RMSE, result.Comparison.MAE, result.Comparison.PearsonCorr)
	}

	// Step 3: Display summary results
	fmt.Println("\nStep 3: Summary Results")
	displaySummaryResults(results)
	
	fmt.Println("\n📁 Detailed debug files written:")
	for _, k := range debugKValues {
		fmt.Printf("   • sketch_debug_k%d.txt - Full sketch details\n", k)
		fmt.Printf("   • degree_comparison_k%d.csv - Node-by-node degree comparison\n", k)
		fmt.Printf("   • hash_mappings_k%d.txt - Hash to node mappings\n", k)
	}
}

func debugSCARSketchGraph(graphFile, propertiesFile, pathFile string, k int, refStats DegreeDistribution, materializedNodeMapping map[string]int) (*SketchDebugReport, ExperimentResult, error) {
	config := scar.NewConfig()
	config.Set("scar.k", int64(k))
	config.Set("scar.nk", int64(1))
	config.Set("algorithm.random_seed", int64(42))

	logger := config.CreateLogger()

	sketchGraph, nodeMapping, err := scar.BuildSketchGraph(graphFile, propertiesFile, pathFile, config, logger)
	if err != nil {
		return nil, ExperimentResult{}, fmt.Errorf("sketch graph build failed: %w", err)
	}

	debugReport := createSketchDebugReport(sketchGraph, nodeMapping, config)

	sketchDegrees := make([]float64, sketchGraph.NumNodes)
	for i := 0; i < sketchGraph.NumNodes; i++ {
		sketchDegrees[i] = sketchGraph.GetDegree(i)
	}

	sketchStats := computeDegreeStats(sketchDegrees)

	matchedRefDegrees, matchedSketchDegrees := matchNodeDegrees(materializedNodeMapping, nodeMapping, refStats.Degrees, sketchDegrees)

	comparison := computeSimpleComparison(matchedRefDegrees, matchedSketchDegrees)

	result := ExperimentResult{
		K:                k,
		MaterializedStats: refStats,
		SketchStats:      sketchStats,
		Comparison:       comparison,
		NumNodes:         len(refStats.Degrees),
		NumTargetNodes:   len(sketchDegrees),
	}

	return debugReport, result, nil
}

func createSketchDebugReport(sketchGraph *scar.SketchGraph, nodeMapping *scar.NodeMapping, config *scar.Config) *SketchDebugReport {
	report := &SketchDebugReport{
		K:           config.K(),
		NK:          config.NK(),
		TotalNodes:  len(nodeMapping.CompressedToOriginal),
		TargetNodes: sketchGraph.NumNodes,
		NodeDetails: make([]NodeSketchDebug, 0, sketchGraph.NumNodes),
		HashMappings: make(map[uint32]int64),
	}

	sketchManager := sketchGraph.GetSketchManager()
	if sketchManager == nil {
		return report
	}

	fullSketches := 0
	nonFullSketches := 0
	emptySketches := 0
	totalFilledCount := int64(0)

	for compressedID := 0; compressedID < sketchGraph.NumNodes; compressedID++ {
		originalID := nodeMapping.CompressedToOriginal[compressedID]
		
		nodeDebug := NodeSketchDebug{
			OriginalID:   originalID,
			CompressedID: compressedID,
		}

		sketch := sketchManager.GetVertexSketch(int64(compressedID))
		if sketch != nil {
			nodeDebug.IsSketchFull = sketch.IsSketchFull()
			nodeDebug.FilledCount = sketch.GetFilledCount()
			nodeDebug.EstimatedDegree = sketchGraph.GetDegree(compressedID)
			
			// Extract raw sketch data
			nodeDebug.SketchLayers = make([][]uint32, sketch.GetNk())
			for layer := int64(0); layer < sketch.GetNk(); layer++ {
				layerData := sketch.GetSketch(layer)
				if layerData != nil {
					nodeDebug.SketchLayers[layer] = make([]uint32, len(layerData))
					copy(nodeDebug.SketchLayers[layer], layerData)
				}
			}

			// Extract own hashes
			nodeDebug.OwnHashes = make([]uint32, 0)
			for layer := int64(0); layer < sketch.GetNk(); layer++ {
				for _, hash := range nodeDebug.SketchLayers[layer] {
					if hash != math.MaxUint32 {
						if nodeId, exists := sketchManager.GetNodeFromHash(hash); exists {
							report.HashMappings[hash] = nodeId
							if nodeId == int64(compressedID) {
								nodeDebug.OwnHashes = append(nodeDebug.OwnHashes, hash)
							}
						}
					}
				}
			}

			if nodeDebug.FilledCount == 0 {
				emptySketches++
			} else if nodeDebug.IsSketchFull {
				fullSketches++
			} else {
				nonFullSketches++
			}
			totalFilledCount += nodeDebug.FilledCount
		} else {
			emptySketches++
		}

		// Get exact degree from adjacency list
		_, weights := sketchGraph.GetNeighbors(compressedID)
		exactDegree := 0.0
		for _, weight := range weights {
			exactDegree += weight
		}
		nodeDebug.ExactDegree = exactDegree

		report.NodeDetails = append(report.NodeDetails, nodeDebug)
	}

	report.GlobalStats.FullSketches = fullSketches
	report.GlobalStats.NonFullSketches = nonFullSketches
	report.GlobalStats.EmptySketches = emptySketches
	if sketchGraph.NumNodes > 0 {
		report.GlobalStats.AvgFilledCount = float64(totalFilledCount) / float64(sketchGraph.NumNodes)
	}

	return report
}

func writeDebugReportToFiles(report *SketchDebugReport, result ExperimentResult, k int) error {
	// 1. Write full sketch debug report
	debugFile, err := os.Create(fmt.Sprintf("sketch_debug_k%d.txt", k))
	if err != nil {
		return err
	}
	defer debugFile.Close()

	writeSketchDebugFile(debugFile, report, result)

	// 2. Write degree comparison CSV
	csvFile, err := os.Create(fmt.Sprintf("degree_comparison_k%d.csv", k))
	if err != nil {
		return err
	}
	defer csvFile.Close()

	writeDegreeComparisonCSV(csvFile, report)

	// 3. Write hash mappings
	hashFile, err := os.Create(fmt.Sprintf("hash_mappings_k%d.txt", k))
	if err != nil {
		return err
	}
	defer hashFile.Close()

	writeHashMappingsFile(hashFile, report)

	return nil
}

func writeSketchDebugFile(file *os.File, report *SketchDebugReport, result ExperimentResult) {
	fmt.Fprintf(file, "SCAR SKETCH DEBUG REPORT (k=%d)\n", report.K)
	fmt.Fprintf(file, "================================\n\n")
	
	fmt.Fprintf(file, "CONFIGURATION:\n")
	fmt.Fprintf(file, "  k = %d\n", report.K)
	fmt.Fprintf(file, "  nk = %d\n", report.NK)
	fmt.Fprintf(file, "  Total original nodes = %d\n", report.TotalNodes)
	fmt.Fprintf(file, "  Target nodes (filtered) = %d\n", report.TargetNodes)
	fmt.Fprintf(file, "  Compression ratio = %.2f%%\n\n", 100.0*float64(report.TargetNodes)/float64(report.TotalNodes))

	fmt.Fprintf(file, "GLOBAL STATISTICS:\n")
	fmt.Fprintf(file, "  Full sketches: %d (%.1f%%)\n", 
		report.GlobalStats.FullSketches, 
		100.0*float64(report.GlobalStats.FullSketches)/float64(report.TargetNodes))
	fmt.Fprintf(file, "  Non-full sketches: %d (%.1f%%)\n", 
		report.GlobalStats.NonFullSketches,
		100.0*float64(report.GlobalStats.NonFullSketches)/float64(report.TargetNodes))
	fmt.Fprintf(file, "  Empty sketches: %d (%.1f%%)\n", 
		report.GlobalStats.EmptySketches,
		100.0*float64(report.GlobalStats.EmptySketches)/float64(report.TargetNodes))
	fmt.Fprintf(file, "  Average filled count: %.2f\n\n", report.GlobalStats.AvgFilledCount)

	fmt.Fprintf(file, "DEGREE ACCURACY:\n")
	fmt.Fprintf(file, "  RMSE: %.4f\n", result.Comparison.RMSE)
	fmt.Fprintf(file, "  MAE: %.4f\n", result.Comparison.MAE)
	fmt.Fprintf(file, "  MAPE: %.2f%%\n", result.Comparison.MAPE)
	fmt.Fprintf(file, "  Pearson Correlation: %.4f\n", result.Comparison.PearsonCorr)
	fmt.Fprintf(file, "  R² Score: %.4f\n\n", result.Comparison.R2Score)

	// Sort nodes by original ID
	sortedNodes := make([]NodeSketchDebug, len(report.NodeDetails))
	copy(sortedNodes, report.NodeDetails)
	sort.Slice(sortedNodes, func(i, j int) bool {
		return sortedNodes[i].OriginalID < sortedNodes[j].OriginalID
	})

	fmt.Fprintf(file, "DETAILED NODE ANALYSIS:\n")
	fmt.Fprintf(file, "%-8s %-8s %-6s %-8s %-10s %-10s %-8s %-10s\n",
		"OrigID", "CompID", "Full?", "Filled", "Estimated", "Exact", "Error", "Error%")
	fmt.Fprintf(file, "%s\n", strings.Repeat("-", 80))

	for _, node := range sortedNodes {
		fullStr := "No"
		if node.IsSketchFull {
			fullStr = "Yes"
		}

		error := node.EstimatedDegree - node.ExactDegree
		errorPct := 0.0
		if node.ExactDegree > 0 {
			errorPct = 100.0 * math.Abs(error) / node.ExactDegree
		}

		fmt.Fprintf(file, "%-8d %-8d %-6s %-8d %-10.2f %-10.2f %-8.2f %-10.1f\n",
			node.OriginalID, node.CompressedID, fullStr, node.FilledCount,
			node.EstimatedDegree, node.ExactDegree, error, errorPct)
	}

	fmt.Fprintf(file, "\nDETAILED SKETCH CONTENTS:\n")
	fmt.Fprintf(file, "%s\n", strings.Repeat("-", 80))

	for i, node := range sortedNodes {
		if i >= 20 { // Limit to first 20 nodes
			fmt.Fprintf(file, "... (showing first 20 nodes only)\n")
			break
		}

		fmt.Fprintf(file, "\nNode %d (compressed %d):\n", node.OriginalID, node.CompressedID)
		fmt.Fprintf(file, "  Status: %s, Filled: %d, Est: %.2f, Exact: %.2f\n",
			map[bool]string{true: "FULL", false: "NON-FULL"}[node.IsSketchFull],
			node.FilledCount, node.EstimatedDegree, node.ExactDegree)

		for layer, layerData := range node.SketchLayers {
			fmt.Fprintf(file, "  Layer %d: [", layer)
			validHashes := 0
			for _, hash := range layerData {
				if hash == math.MaxUint32 {
					break
				}
				if validHashes > 0 {
					fmt.Fprintf(file, ", ")
				}
				if ownerNode, exists := report.HashMappings[hash]; exists {
					fmt.Fprintf(file, "%d→N%d", hash, ownerNode)
				} else {
					fmt.Fprintf(file, "%d→?", hash)
				}
				validHashes++
				if validHashes >= 10 { // Limit display
					fmt.Fprintf(file, "...")
					break
				}
			}
			fmt.Fprintf(file, "]\n")
		}

		if len(node.OwnHashes) > 0 {
			fmt.Fprintf(file, "  Own hashes: %v\n", node.OwnHashes)
		}
	}
}

func writeDegreeComparisonCSV(file *os.File, report *SketchDebugReport) {
	fmt.Fprintf(file, "OriginalID,CompressedID,SketchFull,FilledCount,EstimatedDegree,ExactDegree,AbsoluteError,RelativeError\n")
	
	// Sort nodes by original ID
	sortedNodes := make([]NodeSketchDebug, len(report.NodeDetails))
	copy(sortedNodes, report.NodeDetails)
	sort.Slice(sortedNodes, func(i, j int) bool {
		return sortedNodes[i].OriginalID < sortedNodes[j].OriginalID
	})

	for _, node := range sortedNodes {
		absError := math.Abs(node.EstimatedDegree - node.ExactDegree)
		relError := 0.0
		if node.ExactDegree > 0 {
			relError = absError / node.ExactDegree
		}

		fmt.Fprintf(file, "%d,%d,%t,%d,%.6f,%.6f,%.6f,%.6f\n",
			node.OriginalID, node.CompressedID, node.IsSketchFull, node.FilledCount,
			node.EstimatedDegree, node.ExactDegree, absError, relError)
	}
}

func writeHashMappingsFile(file *os.File, report *SketchDebugReport) {
	fmt.Fprintf(file, "HASH TO NODE MAPPINGS (k=%d)\n", report.K)
	fmt.Fprintf(file, "============================\n\n")
	fmt.Fprintf(file, "Total mappings: %d\n\n", len(report.HashMappings))
	fmt.Fprintf(file, "%-12s %-8s\n", "Hash", "NodeID")
	fmt.Fprintf(file, "%s\n", strings.Repeat("-", 20))

	// Sort hashes for consistent output
	hashes := make([]uint32, 0, len(report.HashMappings))
	for hash := range report.HashMappings {
		hashes = append(hashes, hash)
	}
	sort.Slice(hashes, func(i, j int) bool { return hashes[i] < hashes[j] })

	for _, hash := range hashes {
		nodeId := report.HashMappings[hash]
		fmt.Fprintf(file, "%-12d %-8d\n", hash, nodeId)
	}
}

func computeSimpleComparison(ref, sketch []float64) SimpleComparison {
	if len(ref) == 0 || len(sketch) == 0 || len(ref) != len(sketch) {
		return SimpleComparison{}
	}

	n := float64(len(ref))
	
	// Calculate means
	meanRef, meanSketch := 0.0, 0.0
	for i := range ref {
		meanRef += ref[i]
		meanSketch += sketch[i]
	}
	meanRef /= n
	meanSketch /= n

	// Calculate metrics
	sumSquaredError := 0.0
	sumAbsError := 0.0
	sumAbsPercentError := 0.0
	sumRefVariance := 0.0
	sumCrossProduct := 0.0
	sumSketchVariance := 0.0

	for i := range ref {
		error := sketch[i] - ref[i]
		sumSquaredError += error * error
		sumAbsError += math.Abs(error)
		
		if ref[i] != 0 {
			sumAbsPercentError += math.Abs(error) / math.Abs(ref[i])
		}

		refDev := ref[i] - meanRef
		sketchDev := sketch[i] - meanSketch
		sumRefVariance += refDev * refDev
		sumSketchVariance += sketchDev * sketchDev
		sumCrossProduct += refDev * sketchDev
	}

	// RMSE
	rmse := math.Sqrt(sumSquaredError / n)
	
	// MAE
	mae := sumAbsError / n
	
	// MAPE (as percentage)
	mape := 100.0 * sumAbsPercentError / n
	
	// Pearson correlation
	pearsonCorr := 0.0
	denominator := math.Sqrt(sumRefVariance * sumSketchVariance)
	if denominator > 0 {
		pearsonCorr = sumCrossProduct / denominator
	}

	// R² score
	r2Score := 0.0
	if sumRefVariance > 0 {
		r2Score = 1.0 - sumSquaredError/sumRefVariance
	}

	return SimpleComparison{
		RMSE:        rmse,
		MAE:         mae,
		MAPE:        mape,
		PearsonCorr: pearsonCorr,
		R2Score:     r2Score,
	}
}

func displaySummaryResults(results []ExperimentResult) {
	fmt.Printf("\n📊 SUMMARY COMPARISON RESULTS\n")
	fmt.Printf("════════════════════════════════════════════════════════════════\n")
	fmt.Printf("%-6s %-8s %-8s %-8s %-8s %-8s %-8s\n",
		"K", "Nodes", "RMSE", "MAE", "MAPE%", "Corr", "R²")
	fmt.Printf("%-6s %-8s %-8s %-8s %-8s %-8s %-8s\n",
		"", "", "", "", "", "", "")
	fmt.Printf("%s\n", strings.Repeat("-", 64))

	for _, result := range results {
		fmt.Printf("%-6d %-8d %-8.2f %-8.2f %-8.1f %-8.3f %-8.3f\n",
			result.K,
			result.NumTargetNodes,
			result.Comparison.RMSE,
			result.Comparison.MAE,
			result.Comparison.MAPE,
			result.Comparison.PearsonCorr,
			result.Comparison.R2Score)
	}

	fmt.Printf("\n💡 INTERPRETATION:\n")
	fmt.Printf("   • RMSE: Root Mean Square Error (lower is better)\n")
	fmt.Printf("   • MAE: Mean Absolute Error (lower is better)\n")
	fmt.Printf("   • MAPE: Mean Absolute Percentage Error (lower is better)\n")
	fmt.Printf("   • Corr: Pearson correlation (closer to 1.0 is better)\n")
	fmt.Printf("   • R²: R-squared score (closer to 1.0 is better)\n")
	
	if len(results) > 1 {
		fmt.Printf("\n🔍 TREND ANALYSIS:\n")
		
		// Find best performing k
		bestRMSE := math.Inf(1)
		bestK := 0
		for _, result := range results {
			if result.Comparison.RMSE < bestRMSE {
				bestRMSE = result.Comparison.RMSE
				bestK = result.K
			}
		}
		fmt.Printf("   • Best RMSE: k=%d (RMSE=%.2f)\n", bestK, bestRMSE)
		
		// Check if performance improves with k
		firstRMSE := results[0].Comparison.RMSE
		lastRMSE := results[len(results)-1].Comparison.RMSE
		if lastRMSE < firstRMSE {
			fmt.Printf("   • ✅ Performance improves with larger k\n")
		} else {
			fmt.Printf("   • ⚠️  Performance does not consistently improve with k\n")
		}
	}
}

// Rest of helper functions (materializeReferenceGraph, computeLouvainDegrees, etc.) remain the same...
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

func computeLouvainDegrees(graph *louvain.Graph) []float64 {
	degrees := make([]float64, graph.NumNodes)
	copy(degrees, graph.Degrees)
	return degrees
}

func computeDegreeStats(degrees []float64) DegreeDistribution {
	if len(degrees) == 0 {
		return DegreeDistribution{}
	}

	sorted := make([]float64, len(degrees))
	copy(sorted, degrees)
	sort.Float64s(sorted)

	sum := 0.0
	for _, d := range degrees {
		sum += d
	}
	mean := sum / float64(len(degrees))

	sumSq := 0.0
	for _, d := range degrees {
		diff := d - mean
		sumSq += diff * diff
	}
	stdDev := math.Sqrt(sumSq / float64(len(degrees)))

	median := sorted[len(sorted)/2]

	return DegreeDistribution{
		Degrees: degrees,
		Mean:    mean,
		StdDev:  stdDev,
		Min:     sorted[0],
		Max:     sorted[len(sorted)-1],
		Median:  median,
	}
}

func matchNodeDegrees(materializedMapping map[string]int, sketchMapping *scar.NodeMapping, materializedDegrees, sketchDegrees []float64) ([]float64, []float64) {
	matchedRef := make([]float64, 0)
	matchedSketch := make([]float64, 0)

	for originalID, materializedIdx := range materializedMapping {
		if materializedIdx >= len(materializedDegrees) {
			continue
		}

		originalIntID, err := strconv.Atoi(originalID)
		if err != nil {
			continue
		}

		if sketchIdx, exists := sketchMapping.OriginalToCompressed[originalIntID]; exists {
			if sketchIdx < len(sketchDegrees) {
				matchedRef = append(matchedRef, materializedDegrees[materializedIdx])
				matchedSketch = append(matchedSketch, sketchDegrees[sketchIdx])
			}
		}
	}

	return matchedRef, matchedSketch
}
package main

import (
	// "context"
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

type DegreeDistribution struct {
	Degrees []float64  // Raw degree values
	Mean    float64
	StdDev  float64
	Min     float64
	Max     float64
	Median  float64
	P90     float64    // 90th percentile
	P95     float64    // 95th percentile
}

type DistributionComparison struct {
	KSDistance        float64  // Kolmogorov-Smirnov distance [0,1]
	JSDivergence      float64  // Jensen-Shannon divergence [0,1]
	WassersteinDist   float64  // Earth Mover's distance
	PearsonCorr       float64  // Pearson correlation (if nodes can be matched)
	MeanAbsError      float64  // Mean absolute error between degrees
	RelativeMSE       float64  // Relative mean squared error
}

type ExperimentResult struct {
	K                    int
	MaterializedStats    DegreeDistribution
	SketchStats          DegreeDistribution
	Comparison           DistributionComparison
	NumNodes             int
	NumTargetNodes       int  // Nodes in sketch graph after filtering
}

func main() {
	if len(os.Args) != 4 {
		log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]

	fmt.Println("🔬 DEGREE DISTRIBUTION COMPARISON EXPERIMENT")
	fmt.Println("=============================================")

	// Step 1: Materialize the reference graph
	fmt.Println("Step 1: Materializing reference graph...")
	materializedGraph, nodeMapping, err := materializeReferenceGraph(graphFile, propertiesFile, pathFile)
	if err != nil {
		log.Fatalf("Failed to materialize graph: %v", err)
	}

	// Compute reference degree distribution
	refDegrees := computeLouvainDegrees(materializedGraph)
	refStats := computeDegreeStats(refDegrees)

	fmt.Printf("📊 Reference degree stats: mean=%.2f, std=%.2f, range=[%.2f, %.2f]\n",
		refStats.Mean, refStats.StdDev, refStats.Min, refStats.Max)

	// Step 2: Test different k values
	kValues := []int{2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
	results := make([]ExperimentResult, 0, len(kValues))

	fmt.Println("\nStep 2: Testing SCAR sketch graphs...")
	for _, k := range kValues {
		fmt.Printf("  Testing k=%d...", k)
		
		result, err := testSCARDegrees(graphFile, propertiesFile, pathFile, k, refStats, nodeMapping)
		if err != nil {
			fmt.Printf(" ❌ Failed: %v\n", err)
			continue
		}
		
		results = append(results, result)
		fmt.Printf(" ✅ KS=%.4f, Corr=%.4f\n", 
			result.Comparison.KSDistance, result.Comparison.PearsonCorr)
	}

	// Step 3: Display comprehensive results
	fmt.Println("\nStep 3: Analysis Results")
	displayDegreeComparisonResults(results, refStats)
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

	// Convert to Louvain graph (same logic as your converter)
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

func testSCARDegrees(graphFile, propertiesFile, pathFile string, k int, refStats DegreeDistribution, materializedNodeMapping map[string]int) (ExperimentResult, error) {
	// Build SCAR sketch graph
	config := scar.NewConfig()
	config.Set("scar.k", int64(k))
	config.Set("scar.nk", int64(1))  // Standard nk value
	config.Set("algorithm.random_seed", int64(42))  // Consistent seed

	// ctx := context.Background()
	logger := config.CreateLogger()

	// Build sketch graph (this is the core SCAR preprocessing)
	sketchGraph, nodeMapping, err := scar.BuildSketchGraph(graphFile, propertiesFile, pathFile, config, logger)

	if err != nil {
		return ExperimentResult{}, fmt.Errorf("sketch graph build failed: %w", err)
	}

	// Extract degrees from sketch graph
	sketchDegrees := make([]float64, sketchGraph.NumNodes)
	for i := 0; i < sketchGraph.NumNodes; i++ {
		sketchDegrees[i] = sketchGraph.GetDegree(i)
	}

	sketchStats := computeDegreeStats(sketchDegrees)

	// Attempt to match nodes for correlation analysis
	matchedRefDegrees, matchedSketchDegrees := matchNodeDegrees(materializedNodeMapping, nodeMapping, refStats.Degrees, sketchDegrees)

	// Compute comprehensive comparison metrics
	comparison := computeDistributionComparison(refStats.Degrees, sketchDegrees, matchedRefDegrees, matchedSketchDegrees)

	return ExperimentResult{
		K:                k,
		MaterializedStats: refStats,
		SketchStats:      sketchStats,
		Comparison:       comparison,
		NumNodes:         len(refStats.Degrees),
		NumTargetNodes:   len(sketchDegrees),
	}, nil
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

	// Basic stats
	sum := 0.0
	for _, d := range degrees {
		sum += d
	}
	mean := sum / float64(len(degrees))

	// Standard deviation
	sumSq := 0.0
	for _, d := range degrees {
		diff := d - mean
		sumSq += diff * diff
	}
	stdDev := math.Sqrt(sumSq / float64(len(degrees)))

	// Percentiles
	median := sorted[len(sorted)/2]
	p90 := sorted[int(0.90*float64(len(sorted)))]
	p95 := sorted[int(0.95*float64(len(sorted)))]

	return DegreeDistribution{
		Degrees: degrees,
		Mean:    mean,
		StdDev:  stdDev,
		Min:     sorted[0],
		Max:     sorted[len(sorted)-1],
		Median:  median,
		P90:     p90,
		P95:     p95,
	}
}

func matchNodeDegrees(materializedMapping map[string]int, sketchMapping *scar.NodeMapping, materializedDegrees, sketchDegrees []float64) ([]float64, []float64) {
	matchedRef := make([]float64, 0)
	matchedSketch := make([]float64, 0)

	// Try to match nodes by original ID
	for originalID, materializedIdx := range materializedMapping {
		if materializedIdx >= len(materializedDegrees) {
			continue
		}

		// Convert string ID to int for sketch mapping lookup
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

func computeDistributionComparison(refDegrees, sketchDegrees, matchedRef, matchedSketch []float64) DistributionComparison {
	// Kolmogorov-Smirnov distance
	ksDistance := computeKSDistance(refDegrees, sketchDegrees)

	// Jensen-Shannon divergence (on binned distributions)
	jsDiv := computeJSDivergence(refDegrees, sketchDegrees)

	// Wasserstein distance (simplified 1D version)
	wassersteinDist := computeWassersteinDistance(refDegrees, sketchDegrees)

	// Pearson correlation (only if we have matched nodes)
	var pearsonCorr float64
	if len(matchedRef) > 0 && len(matchedSketch) > 0 {
		pearsonCorr = computePearsonCorrelation(matchedRef, matchedSketch)
	}

	// Mean absolute error (for matched nodes)
	var meanAbsError float64
	if len(matchedRef) > 0 {
		sumAbsError := 0.0
		for i := 0; i < len(matchedRef); i++ {
			sumAbsError += math.Abs(matchedRef[i] - matchedSketch[i])
		}
		meanAbsError = sumAbsError / float64(len(matchedRef))
	}

	// Relative MSE
	var relativeMSE float64
	if len(matchedRef) > 0 {
		sumSqError := 0.0
		sumSqRef := 0.0
		for i := 0; i < len(matchedRef); i++ {
			diff := matchedRef[i] - matchedSketch[i]
			sumSqError += diff * diff
			sumSqRef += matchedRef[i] * matchedRef[i]
		}
		if sumSqRef > 0 {
			relativeMSE = sumSqError / sumSqRef
		}
	}

	return DistributionComparison{
		KSDistance:      ksDistance,
		JSDivergence:    jsDiv,
		WassersteinDist: wassersteinDist,
		PearsonCorr:     pearsonCorr,
		MeanAbsError:    meanAbsError,
		RelativeMSE:     relativeMSE,
	}
}

func computeKSDistance(dist1, dist2 []float64) float64 {
	// Empirical CDF comparison
	sorted1 := make([]float64, len(dist1))
	sorted2 := make([]float64, len(dist2))
	copy(sorted1, dist1)
	copy(sorted2, dist2)
	sort.Float64s(sorted1)
	sort.Float64s(sorted2)

	// Find common range
	minVal := math.Min(sorted1[0], sorted2[0])
	maxVal := math.Max(sorted1[len(sorted1)-1], sorted2[len(sorted2)-1])

	// Sample points for CDF comparison
	numPoints := 1000
	maxDiff := 0.0

	for i := 0; i <= numPoints; i++ {
		x := minVal + float64(i)*(maxVal-minVal)/float64(numPoints)

		cdf1 := empiricalCDF(sorted1, x)
		cdf2 := empiricalCDF(sorted2, x)

		diff := math.Abs(cdf1 - cdf2)
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	return maxDiff
}

func empiricalCDF(sortedData []float64, x float64) float64 {
	count := 0
	for _, val := range sortedData {
		if val <= x {
			count++
		} else {
			break
		}
	}
	return float64(count) / float64(len(sortedData))
}

func computeJSDivergence(dist1, dist2 []float64) float64 {
	// Bin the distributions
	numBins := 50
	minVal := math.Min(sliceMin(dist1), sliceMin(dist2))
	maxVal := math.Max(sliceMax(dist1), sliceMax(dist2))

	hist1 := histogram(dist1, numBins, minVal, maxVal)
	hist2 := histogram(dist2, numBins, minVal, maxVal)

	// Normalize to probabilities
	normalize(hist1)
	normalize(hist2)

	// Compute JS divergence
	return jensenShannonDivergence(hist1, hist2)
}

func histogram(data []float64, numBins int, minVal, maxVal float64) []float64 {
	bins := make([]float64, numBins)
	binWidth := (maxVal - minVal) / float64(numBins)

	for _, val := range data {
		binIdx := int((val - minVal) / binWidth)
		if binIdx >= numBins {
			binIdx = numBins - 1
		}
		if binIdx < 0 {
			binIdx = 0
		}
		bins[binIdx]++
	}

	return bins
}

func normalize(hist []float64) {
	sum := 0.0
	for _, count := range hist {
		sum += count
	}
	if sum > 0 {
		for i := range hist {
			hist[i] /= sum
		}
	}
}

func jensenShannonDivergence(p, q []float64) float64 {
	if len(p) != len(q) {
		return 0.0
	}

	// M = (P + Q) / 2
	m := make([]float64, len(p))
	for i := range p {
		m[i] = (p[i] + q[i]) / 2.0
	}

	// JS = (KL(P||M) + KL(Q||M)) / 2
	klPM := klDivergence(p, m)
	klQM := klDivergence(q, m)

	return (klPM + klQM) / 2.0
}

func klDivergence(p, q []float64) float64 {
	kl := 0.0
	for i := range p {
		if p[i] > 0 && q[i] > 0 {
			kl += p[i] * math.Log2(p[i]/q[i])
		}
	}
	return kl
}

func computeWassersteinDistance(dist1, dist2 []float64) float64 {
	// Simplified 1D Wasserstein distance
	sorted1 := make([]float64, len(dist1))
	sorted2 := make([]float64, len(dist2))
	copy(sorted1, dist1)
	copy(sorted2, dist2)
	sort.Float64s(sorted1)
	sort.Float64s(sorted2)

	// Interpolate to same length for comparison
	n := int(math.Min(float64(len(sorted1)), float64(len(sorted2))))
	sum := 0.0

	for i := 0; i < n; i++ {
		idx1 := int(float64(i) * float64(len(sorted1)) / float64(n))
		idx2 := int(float64(i) * float64(len(sorted2)) / float64(n))
		
		if idx1 >= len(sorted1) {
			idx1 = len(sorted1) - 1
		}
		if idx2 >= len(sorted2) {
			idx2 = len(sorted2) - 1
		}
		
		sum += math.Abs(sorted1[idx1] - sorted2[idx2])
	}

	return sum / float64(n)
}

func computePearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}

	// Calculate means
	meanX, meanY := 0.0, 0.0
	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= float64(len(x))
	meanY /= float64(len(y))

	// Calculate correlation
	numerator, sumSqX, sumSqY := 0.0, 0.0, 0.0
	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY
		numerator += dx * dy
		sumSqX += dx * dx
		sumSqY += dy * dy
	}

	denominator := math.Sqrt(sumSqX * sumSqY)
	if denominator == 0 {
		return 0.0
	}

	return numerator / denominator
}

func displayDegreeComparisonResults(results []ExperimentResult, refStats DegreeDistribution) {
	fmt.Println("\n📊 DEGREE DISTRIBUTION COMPARISON RESULTS")
	fmt.Println("==========================================")

	fmt.Printf("\n🎯 REFERENCE (Materialized Graph):\n")
	fmt.Printf("   Nodes: %d\n", len(refStats.Degrees))
	fmt.Printf("   Mean degree: %.2f ± %.2f\n", refStats.Mean, refStats.StdDev)
	fmt.Printf("   Range: [%.2f, %.2f]\n", refStats.Min, refStats.Max)
	fmt.Printf("   Percentiles: P50=%.2f, P90=%.2f, P95=%.2f\n", 
		refStats.Median, refStats.P90, refStats.P95)

	fmt.Printf("\n📈 COMPARISON BY K VALUE:\n")
	fmt.Printf("%-6s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n",
		"K", "Nodes", "KS-Dist", "JS-Div", "Wasserst", "Pearson", "MAE", "RelMSE")
	fmt.Printf("%s\n", strings.Repeat("-", 70))

	for _, result := range results {
		fmt.Printf("%-6d %-8d %-8.4f %-8.4f %-8.2f %-8.4f %-8.2f %-8.4f\n",
			result.K,
			result.NumTargetNodes,
			result.Comparison.KSDistance,
			result.Comparison.JSDivergence,
			result.Comparison.WassersteinDist,
			result.Comparison.PearsonCorr,
			result.Comparison.MeanAbsError,
			result.Comparison.RelativeMSE)
	}

	fmt.Printf("\n🔍 DETAILED ANALYSIS:\n")
	for _, result := range results {
		fmt.Printf("\nk=%d (Target nodes: %d/%d = %.1f%%):\n",
			result.K, result.NumTargetNodes, result.NumNodes,
			100.0*float64(result.NumTargetNodes)/float64(result.NumNodes))
		fmt.Printf("   Sketch stats: mean=%.2f±%.2f, range=[%.2f,%.2f]\n",
			result.SketchStats.Mean, result.SketchStats.StdDev,
			result.SketchStats.Min, result.SketchStats.Max)
		fmt.Printf("   Distribution similarity: KS=%.4f (lower=better)\n", 
			result.Comparison.KSDistance)
		fmt.Printf("   Node-level correlation: r=%.4f (higher=better)\n", 
			result.Comparison.PearsonCorr)
		
		// Interpretation
		if result.Comparison.KSDistance < 0.1 {
			fmt.Printf("   → Excellent distribution match! 🎯\n")
		} else if result.Comparison.KSDistance < 0.2 {
			fmt.Printf("   → Good distribution match ✅\n")
		} else if result.Comparison.KSDistance < 0.4 {
			fmt.Printf("   → Moderate distribution match ⚠️\n")
		} else {
			fmt.Printf("   → Poor distribution match ❌\n")
		}
	}

	fmt.Printf("\n💡 INSIGHTS:\n")
	fmt.Printf("   • KS Distance: Measures maximum difference in cumulative distributions\n")
	fmt.Printf("   • Pearson Correlation: Measures linear relationship between matched node degrees\n")
	fmt.Printf("   • Wasserstein Distance: Measures cost to transform one distribution to another\n")
	fmt.Printf("   • Lower KS/Wasserstein and higher Pearson indicate better approximation\n")
	fmt.Printf("   • Expected: Higher k should yield better degree approximation\n")
}

// Helper functions
func sliceMin(slice []float64) float64 {
	if len(slice) == 0 {
		return 0
	}
	min := slice[0]
	for _, val := range slice[1:] {
		if val < min {
			min = val
		}
	}
	return min
}

func sliceMax(slice []float64) float64 {
	if len(slice) == 0 {
		return 0
	}
	max := slice[0]
	for _, val := range slice[1:] {
		if val > max {
			max = val
		}
	}
	return max
}
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	// "math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/graph/network"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/mds"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
	"github.com/gilchrisn/graph-clustering-service/pkg/parser"
)

// Simple data structures
type NodeViz struct {
	ID       string  `json:"id"`
	PageRank float64 `json:"pagerank"`
	X        float64 `json:"x"`
	Y        float64 `json:"y"`
	Radius   float64 `json:"radius"`
	Label    string  `json:"label"`
}

type LevelViz struct {
	Level int       `json:"level"`
	Nodes []NodeViz `json:"nodes"`
}

func main() {
	if len(os.Args) < 5 {
		fmt.Println("Usage: pipeline <graph> <properties> <path> <output_dir>")
		os.Exit(1)
	}

	graphFile := os.Args[1]
	propertiesFile := os.Args[2]
	pathFile := os.Args[3]
	outputDir := os.Args[4]

	fmt.Println("🚀 Simple Graph Pipeline")
	fmt.Printf("Input: %s %s %s\n", graphFile, propertiesFile, pathFile)
	fmt.Printf("Output: %s\n", outputDir)

	// Always overwrite - keep it simple
	os.RemoveAll(outputDir)
	
	// Run both algorithms
	err := runMaterializationPipeline(graphFile, propertiesFile, pathFile, outputDir)
	if err != nil {
		fmt.Printf("❌ Materialization pipeline failed: %v\n", err)
		os.Exit(1)
	}

	err = runScarPipeline(graphFile, propertiesFile, pathFile, outputDir)
	if err != nil {
		fmt.Printf("❌ SCAR pipeline failed: %v\n", err)
		os.Exit(1)
	}

	matLvl0 := filepath.Join(outputDir, "materialization", "hierarchy", "level_0.edgelist")
	scarLvl0 := filepath.Join(outputDir, "scar",          "hierarchy", "level_0.edgelist")
	jaccard, err := computeJaccardSimilarity(matLvl0, scarLvl0)
	if err != nil {
		fmt.Printf("⚠️  failed to compare lowest‐level graphs: %v\n", err)
	} else {
		fmt.Printf("🧮 Jaccard similarity (level 0): %.4f\n", jaccard)
	}
	
	fmt.Println("✅ Pipeline completed!")

}

// ===== MATERIALIZATION + LOUVAIN PIPELINE =====

func runMaterializationPipeline(graphFile, propertiesFile, pathFile, baseOutputDir string) error {
	algorithm := "materialization"
	fmt.Printf("\n=== %s Pipeline ===\n", algorithm)

	// Step 1: Run materialization + Louvain
	clusteringDir := filepath.Join(baseOutputDir, algorithm, "clustering")
	err := os.MkdirAll(clusteringDir, 0755)
	if err != nil {
		return err
	}

	fmt.Println("1. Running materialization + Louvain...")
	err = runMaterializationLouvain(graphFile, propertiesFile, pathFile, clusteringDir)
	if err != nil {
		return fmt.Errorf("clustering failed: %w", err)
	}

	// Step 2: Parse hierarchy to level files
	hierarchyDir := filepath.Join(baseOutputDir, algorithm, "hierarchy")
	err = os.MkdirAll(hierarchyDir, 0755)
	if err != nil {
		return err
	}

	fmt.Println("2. Parsing hierarchy...")
	maxLevel, err := parseMaterializationHierarchy(clusteringDir, hierarchyDir)
	if err != nil {
		return fmt.Errorf("hierarchy parsing failed: %w", err)
	}

	// Step 3: Generate PageRank + MDS for each level
	vizDir := filepath.Join(baseOutputDir, algorithm, "visualization")
	err = os.MkdirAll(vizDir, 0755)
	if err != nil {
		return err
	}

	fmt.Println("3. Computing PageRank + MDS...")
	err = generateVisualization(hierarchyDir, vizDir, algorithm, maxLevel)
	if err != nil {
		return fmt.Errorf("visualization failed: %w", err)
	}

	fmt.Printf("✅ %s pipeline completed\n", algorithm)
	return nil
}

// ===== SCAR PIPELINE =====

func runScarPipeline(graphFile, propertiesFile, pathFile, baseOutputDir string) error {
	algorithm := "scar"
	fmt.Printf("\n=== %s Pipeline ===\n", algorithm)

	// Step 1: Run SCAR
	clusteringDir := filepath.Join(baseOutputDir, algorithm, "clustering")
	err := os.MkdirAll(clusteringDir, 0755)
	if err != nil {
		return err
	}

	fmt.Println("1. Running SCAR...")
	err = runScar(graphFile, propertiesFile, pathFile, clusteringDir)
	if err != nil {
		return fmt.Errorf("SCAR failed: %w", err)
	}

	// Step 2: Parse hierarchy to level files
	hierarchyDir := filepath.Join(baseOutputDir, algorithm, "hierarchy")
	err = os.MkdirAll(hierarchyDir, 0755)
	if err != nil {
		return err
	}

	fmt.Println("2. Parsing hierarchy...")
	maxLevel, err := parseScarHierarchy(clusteringDir, hierarchyDir)
	if err != nil {
		return fmt.Errorf("hierarchy parsing failed: %w", err)
	}

	// Step 3: Generate PageRank + MDS for each level
	vizDir := filepath.Join(baseOutputDir, algorithm, "visualization")
	err = os.MkdirAll(vizDir, 0755)
	if err != nil {
		return err
	}

	fmt.Println("3. Computing PageRank + MDS...")
	err = generateVisualization(hierarchyDir, vizDir, algorithm, maxLevel)
	if err != nil {
		return fmt.Errorf("visualization failed: %w", err)
	}

	fmt.Printf("✅ %s pipeline completed\n", algorithm)
	return nil
}

// ===== CLUSTERING ALGORITHM CALLS =====

// NewPipelineConfig creates default configuration for both pipelines  
func NewPipelineConfig() *PipelineConfig {
	return &PipelineConfig{
		Verbose:      true,
		OutputDir:    "pipeline_output",
		OutputPrefix: "communities",
		
		// Default materialization + Louvain
		MaterializationConfig: materialization.DefaultMaterializationConfig(),
		LouvainConfig:        louvain.DefaultLouvainConfig(),
		
		// Default SCAR config
		SCARConfig: scar.SCARConfig{
			K:           256,
			NK:          1,
			Threshold:   0.5,
			UseLouvain:  true,
			SketchOutput: true, // For hierarchy output compatible with PPRViz
			WriteSketchGraph: true, // Write sketch graph files
			SketchGraphWeights: false, // Use weights in sketch graph files
		},
	}
}

func runMaterializationLouvain(graphFile, propertiesFile, pathFile, outputDir string) error {
	// Create default configuration exactly like your original code
	config := NewPipelineConfig()
	config.OutputDir = outputDir
	config.OutputPrefix = "communities"
	config.Verbose = true
	
	// Configure materialization + Louvain with same settings as your original
	config.MaterializationConfig.Aggregation.Strategy = materialization.Average
	config.MaterializationConfig.Aggregation.Symmetric = true
	config.MaterializationConfig.Traversal.MaxInstances = 1000000
	
	config.LouvainConfig.MaxIterations = 10
	config.LouvainConfig.MinModularity = 0.001
	config.LouvainConfig.RandomSeed = 42

	fmt.Println("   🔄 Running materialization...")
	fmt.Printf("   Materialization config: max_instances=%d, symmetric=%t\n", 
		config.MaterializationConfig.Traversal.MaxInstances, 
		config.MaterializationConfig.Aggregation.Symmetric)
	fmt.Printf("   Louvain config: max_iter=%d, min_mod=%.6f, seed=%d\n", 
		config.LouvainConfig.MaxIterations, 
		config.LouvainConfig.MinModularity, 
		config.LouvainConfig.RandomSeed)
	
	// Now call your exact RunMaterializationLouvain function
	result, err := RunMaterializationLouvain(graphFile, propertiesFile, pathFile, config)
	if err != nil {
		return err
	}
	
	fmt.Printf("   ✅ Materialization + Louvain completed in %d ms\n", result.TotalRuntimeMS)
	fmt.Printf("   Final modularity: %.6f\n", result.LouvainResult.Modularity)
	if len(result.LouvainResult.Levels) > 0 {
		finalLevel := result.LouvainResult.Levels[len(result.LouvainResult.Levels)-1]
		fmt.Printf("   Number of communities: %d, Hierarchy levels: %d\n", 
			finalLevel.NumCommunities, result.LouvainResult.NumLevels)
	}
	
	return nil
}

func runScar(graphFile, propertiesFile, pathFile, outputDir string) error {
	// Create default configuration exactly like your original code
	config := NewPipelineConfig()
	
	config.OutputDir = outputDir
	config.OutputPrefix = "communities"
	config.Verbose = true
	
	// // Configure SCAR with same settings as your original
	// config.SCARConfig.K = 25
	// config.SCARConfig.NK = 1
	// config.SCARConfig.Threshold = 0.5
	// config.SCARConfig.UseLouvain = true
	// config.SCARConfig.SketchOutput = true

	fmt.Println("   🔄 Running SCAR...")
	fmt.Printf("   SCAR config: k=%d, nk=%d, threshold=%.3f, sketch_output=%t\n", 
		config.SCARConfig.K, config.SCARConfig.NK, config.SCARConfig.Threshold, config.SCARConfig.SketchOutput)
	
	// Now call your exact RunSketchLouvain function  
	result, err := RunSketchLouvain(graphFile, propertiesFile, pathFile, config)
	if err != nil {
		return err
	}
	
	fmt.Printf("   ✅ SCAR completed in %d ms\n", result.TotalRuntimeMS)
	return nil
}

// ===== HIERARCHY PARSING =====

func parseMaterializationHierarchy(clusteringDir, hierarchyDir string) (int, error) {
	// Build file paths - materialization uses standard names
	edgelistFile := filepath.Join(clusteringDir, "materialized_graph.edgelist")
	mappingFile := filepath.Join(clusteringDir, "communities.mapping")
	hierarchyFile := filepath.Join(clusteringDir, "communities.hierarchy") 
	rootFile := filepath.Join(clusteringDir, "communities.root")
	
	// Check if files exist
	if err := checkRequiredFiles(edgelistFile, mappingFile, hierarchyFile, rootFile); err != nil {
		return 0, fmt.Errorf("required files missing: %w", err)
	}
	
	// Call parser - it creates files as {outputPrefix}_level_{N}.txt
	outputPrefix := filepath.Join(hierarchyDir, "hierarchy")
	err := parser.ParseLouvainHierarchy(edgelistFile, mappingFile, hierarchyFile, rootFile, outputPrefix)
	if err != nil {
		return 0, fmt.Errorf("ParseLouvainHierarchy failed: %w", err)
	}
	
	// Rename .txt files to .edgelist and count levels
	maxLevel := 0
	for {
		sourceFile := fmt.Sprintf("%s_level_%d.txt", outputPrefix, maxLevel)
		targetFile := filepath.Join(hierarchyDir, fmt.Sprintf("level_%d.edgelist", maxLevel))
		
		if _, err := os.Stat(sourceFile); os.IsNotExist(err) {
			break
		}
		
		err := os.Rename(sourceFile, targetFile)
		if err != nil {
			return 0, fmt.Errorf("failed to rename %s to %s: %w", sourceFile, targetFile, err)
		}
		maxLevel++
	}
	
	if maxLevel == 0 {
		return 0, fmt.Errorf("no level files were created")
	}
	
	return maxLevel - 1, nil
}

func parseScarHierarchy(clusteringDir, hierarchyDir string) (int, error) {
	// Build file paths - SCAR uses .dat extensions
	sketchFile := filepath.Join(clusteringDir, "communities.sketch")
	mappingFile := filepath.Join(clusteringDir, "communities_mapping.dat")  // Note: different name
	hierarchyFile := filepath.Join(clusteringDir, "communities_hierarchy.dat")  // Note: different name
	rootFile := filepath.Join(clusteringDir, "communities_root.dat")  // Note: different name
	
	// Check if files exist
	if err := checkRequiredFiles(sketchFile, mappingFile, hierarchyFile, rootFile); err != nil {
		return 0, fmt.Errorf("required files missing: %w", err)
	}
	
	// Call parser - it creates files as {outputPrefix}_level_{N}.txt  
	outputPrefix := filepath.Join(hierarchyDir, "sketch_hierarchy")
	err := parser.ParseSketchLouvainHierarchy(sketchFile, mappingFile, hierarchyFile, rootFile, outputPrefix)
	if err != nil {
		return 0, fmt.Errorf("ParseSketchLouvainHierarchy failed: %w", err)
	}
	
	// Rename .txt files to .edgelist and count levels
	maxLevel := 0
	for {
		sourceFile := fmt.Sprintf("%s_level_%d.txt", outputPrefix, maxLevel)
		targetFile := filepath.Join(hierarchyDir, fmt.Sprintf("level_%d.edgelist", maxLevel))
		
		if _, err := os.Stat(sourceFile); os.IsNotExist(err) {
			break
		}
		
		err := os.Rename(sourceFile, targetFile)
		if err != nil {
			return 0, fmt.Errorf("failed to rename %s to %s: %w", sourceFile, targetFile, err)
		}
		maxLevel++
	}
	
	if maxLevel == 0 {
		return 0, fmt.Errorf("no level files were created")
	}
	
	return maxLevel - 1, nil
}

// Helper function to check if required files exist
func checkRequiredFiles(filenames ...string) error {
	for _, filename := range filenames {
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			return fmt.Errorf("file does not exist: %s", filename)
		}
	}
	return nil
}

// ===== VISUALIZATION GENERATION =====

func generateVisualization(hierarchyDir, vizDir, algorithm string, maxLevel int) error {
	var allLevels []LevelViz

	// Process each level
	for level := 0; level <= maxLevel; level++ {
		fmt.Printf("   Level %d...", level)
		
		// Load level graph
		levelPath := filepath.Join(hierarchyDir, fmt.Sprintf("level_%d.edgelist", level))
		graph, nodeMapping, err := loadGraph(levelPath)
		if err != nil {
			fmt.Printf(" ❌ failed to load: %v\n", err)
			continue
		}

		if graph.Nodes().Len() == 0 {
			fmt.Printf(" ⏭️ empty graph\n")
			continue
		}

		// Compute PageRank
		pageRankScores := network.PageRank(graph, 0.85, 1e-6)

		// Compute MDS
		distMatrix := createDistanceMatrix(graph)
		coords := applyMDS(distMatrix)

		// Create visualization data
		levelViz := createLevelVisualization(level, pageRankScores, coords, nodeMapping, algorithm)
		allLevels = append(allLevels, levelViz)

		fmt.Printf(" ✅ %d nodes\n", len(levelViz.Nodes))
	}

	// Save all levels to single file
	vizPath := filepath.Join(vizDir, "levels.json")
	err := saveJSON(vizPath, allLevels)
	if err != nil {
		return err
	}

	fmt.Printf("   Saved visualization: %d levels\n", len(allLevels))
	return nil
}

func createLevelVisualization(level int, pageRankScores map[int64]float64, coords *mat.Dense, nodeMapping map[int64]string, algorithm string) LevelViz {
	var nodes []NodeViz

	// Find min/max PageRank for scaling
	var minPR, maxPR float64
	first := true
	for _, score := range pageRankScores {
		if first {
			minPR, maxPR = score, score
			first = false
		} else {
			if score < minPR {
				minPR = score
			}
			if score > maxPR {
				maxPR = score
			}
		}
	}
	if maxPR == minPR {
		maxPR = minPR + 0.001
	}

	// Create node visualizations
	for nodeID, score := range pageRankScores {
		// Get coordinates
		x := coords.At(int(nodeID), 0)
		y := coords.At(int(nodeID), 1)

		// Scale radius based on PageRank
		normalizedPR := (score - minPR) / (maxPR - minPR)
		radius := 3.0 + normalizedPR*15.0

		// Create label
		originalID := nodeMapping[nodeID]
		label := fmt.Sprintf("%s_%s_L%d (%.4f)", algorithm, originalID, level, score)

		node := NodeViz{
			ID:       originalID,
			PageRank: score,
			X:        x,
			Y:        y,
			Radius:   radius,
			Label:    label,
		}
		nodes = append(nodes, node)
	}

	return LevelViz{
		Level: level,
		Nodes: nodes,
	}
}

// ===== UTILITY FUNCTIONS =====

type Edge struct {
	From   string
	To     string
	Weight float64
}

func parseMapping(filename string) (map[string][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	mapping := make(map[string][]string)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		// Read community ID
		communityID := strings.TrimSpace(scanner.Text())
		if communityID == "" {
			continue
		}

		// Read node count
		if !scanner.Scan() {
			break
		}
		countStr := strings.TrimSpace(scanner.Text())
		count, err := strconv.Atoi(countStr)
		if err != nil {
			continue
		}

		// Read nodes
		nodes := make([]string, 0, count)
		for i := 0; i < count && scanner.Scan(); i++ {
			node := strings.TrimSpace(scanner.Text())
			nodes = append(nodes, node)
		}

		mapping[communityID] = nodes
	}

	return mapping, scanner.Err()
}

func extractLevel(communityID string) int {
	// Extract level from community ID (e.g., "c0_l1_0" -> 1)
	parts := strings.Split(communityID, "_")
	if len(parts) >= 2 && strings.HasPrefix(parts[1], "l") {
		if level, err := strconv.Atoi(parts[1][1:]); err == nil {
			return level
		}
	}
	return 0
}

func loadEdges(filename string) ([]Edge, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var edges []Edge
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}

		from := parts[0]
		to := parts[1]
		weight := 1.0

		if len(parts) >= 3 {
			if w, err := strconv.ParseFloat(parts[2], 64); err == nil {
				weight = w
			}
		}

		edges = append(edges, Edge{From: from, To: to, Weight: weight})
	}

	return edges, scanner.Err()
}

func saveEdges(edges []Edge, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for _, edge := range edges {
		fmt.Fprintf(file, "%s %s %.6f\n", edge.From, edge.To, edge.Weight)
	}

	return nil
}

func aggregateEdges(originalEdges []Edge, mapping map[string][]string, targetLevel int) []Edge {
	// Build node -> community mapping for target level
	nodeToComm := make(map[string]string)
	for communityID, nodes := range mapping {
		level := extractLevel(communityID)
		if level == targetLevel {
			for _, node := range nodes {
				nodeToComm[node] = communityID
			}
		}
	}

	// Aggregate edges between communities
	edgeWeights := make(map[string]float64)
	for _, edge := range originalEdges {
		fromComm, fromExists := nodeToComm[edge.From]
		toComm, toExists := nodeToComm[edge.To]

		if !fromExists || !toExists {
			continue
		}

		// Create edge key
		var edgeKey string
		if fromComm <= toComm {
			edgeKey = fromComm + "|" + toComm
		} else {
			edgeKey = toComm + "|" + fromComm
		}

		edgeWeights[edgeKey] += edge.Weight
	}

	// Convert to edge list
	var result []Edge
	for edgeKey, weight := range edgeWeights {
		parts := strings.Split(edgeKey, "|")
		if len(parts) != 2 {
			continue
		}

		from := parts[0]
		to := parts[1]

		// Add both directions for undirected graph
		result = append(result, Edge{From: from, To: to, Weight: weight})
		if from != to {
			result = append(result, Edge{From: to, To: from, Weight: weight})
		}
	}

	return result
}

func loadGraph(filename string) (*simple.DirectedGraph, map[int64]string, error) {
	edges, err := loadEdges(filename)
	if err != nil {
		return nil, nil, err
	}

	graph := simple.NewDirectedGraph()
	nodeMapping := make(map[int64]string)
	stringToInt := make(map[string]int64)
	nextID := int64(0)

	for _, edge := range edges {
		// Skip self-loops for now
		if edge.From == edge.To {
			continue
		}

		// Map string IDs to integers
		fromID, exists := stringToInt[edge.From]
		if !exists {
			fromID = nextID
			stringToInt[edge.From] = fromID
			nodeMapping[fromID] = edge.From
			graph.AddNode(simple.Node(fromID))
			nextID++
		}

		toID, exists := stringToInt[edge.To]
		if !exists {
			toID = nextID
			stringToInt[edge.To] = toID
			nodeMapping[toID] = edge.To
			graph.AddNode(simple.Node(toID))
			nextID++
		}

		// Add edge if it doesn't exist
		if !graph.HasEdgeFromTo(fromID, toID) {
			graph.SetEdge(simple.Edge{
				F: simple.Node(fromID),
				T: simple.Node(toID),
			})
		}
	}

	return graph, nodeMapping, nil
}

func createDistanceMatrix(g *simple.DirectedGraph) *mat.SymDense {
	nodes := g.Nodes()
	nodeList := make([]int64, 0)
	for nodes.Next() {
		nodeList = append(nodeList, nodes.Node().ID())
	}
	n := len(nodeList)

	if n == 0 {
		return mat.NewSymDense(0, nil)
	}

	distMatrix := mat.NewSymDense(n, nil)

	for i, nodeI := range nodeList {
		distances := bfsDistances(g, nodeI)
		for j, nodeJ := range nodeList {
			dist := distances[nodeJ]
			if dist == -1 {
				dist = float64(n) // Use graph diameter as max distance
			}
			distMatrix.SetSym(i, j, dist)
		}
	}

	return distMatrix
}

func bfsDistances(g *simple.DirectedGraph, source int64) map[int64]float64 {
	distances := make(map[int64]float64)
	visited := make(map[int64]bool)
	queue := []int64{source}
	
	distances[source] = 0
	visited[source] = true

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		neighbors := g.From(current)
		for neighbors.Next() {
			neighbor := neighbors.Node().ID()
			if !visited[neighbor] {
				visited[neighbor] = true
				distances[neighbor] = distances[current] + 1
				queue = append(queue, neighbor)
			}
		}
	}

	nodes := g.Nodes()
	for nodes.Next() {
		nodeID := nodes.Node().ID()
		if _, exists := distances[nodeID]; !exists {
			distances[nodeID] = -1
		}
	}

	return distances
}

func applyMDS(distMatrix *mat.SymDense) *mat.Dense {
	var coords mat.Dense
	var eigenvals []float64

	k, _ := mds.TorgersonScaling(&coords, eigenvals, distMatrix)
	
	fmt.Printf("   MDS: %d positive eigenvalues\n", k)

	return &coords
}

func saveJSON(path string, data interface{}) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

// ===== MISSING TYPES AND FUNCTIONS =====

// PipelineConfig holds configuration for both pipeline types
type PipelineConfig struct {
	// Common options
	Verbose      bool
	OutputDir    string
	OutputPrefix string
	
	// Materialization + Louvain config
	MaterializationConfig materialization.MaterializationConfig
	LouvainConfig        louvain.LouvainConfig
	
	// SCAR config
	SCARConfig scar.SCARConfig
}

// convertHomogeneousToNormalized converts materialization output to Louvain input format
func convertHomogeneousToNormalized(hgraph *materialization.HomogeneousGraph) (*louvain.NormalizedGraph, *louvain.GraphParser, error) {
	if len(hgraph.Nodes) == 0 {
		return nil, nil, fmt.Errorf("empty homogeneous graph")
	}
	
	parser := louvain.NewGraphParser()
	
	// Create ordered list of node IDs
	nodeList := make([]string, 0, len(hgraph.Nodes))
	for nodeID := range hgraph.Nodes {
		nodeList = append(nodeList, nodeID)
	}

	// Use intelligent sorting
	allIntegers := allNodesAreIntegers(nodeList)
	if allIntegers {
		// Sort numerically: 1 < 2 < 5 < 10
		sort.Slice(nodeList, func(i, j int) bool {
			a, _ := strconv.Atoi(nodeList[i])
			b, _ := strconv.Atoi(nodeList[j])
			return a < b
		})
	} else {
		// Sort lexicographically: "1" < "10" < "2" < "5"
		sort.Strings(nodeList)
	}
	
	// Create normalized graph
	normalizedGraph := louvain.NewNormalizedGraph(len(nodeList))
	
	// Build node ID mappings and set weights
	for i, originalID := range nodeList {
		parser.OriginalToNormalized[originalID] = i
		parser.NormalizedToOriginal[i] = originalID
		
		// Set node weights (using default for now)
		if _, exists := hgraph.Nodes[originalID]; exists {
			normalizedGraph.Weights[i] = 1.0 // Default weight
		}
	}
	parser.NumNodes = len(nodeList)
	
	// Convert edges with deduplication to prevent double counting
	processedEdges := make(map[string]bool)
	edgeCount := 0
	
	for edgeKey, weight := range hgraph.Edges {
		fromNormalized, fromExists := parser.OriginalToNormalized[edgeKey.From]
		toNormalized, toExists := parser.OriginalToNormalized[edgeKey.To]
		
		if !fromExists || !toExists {
			return nil, nil, fmt.Errorf("edge references unknown nodes: %s -> %s", edgeKey.From, edgeKey.To)
		}
		
		// Create canonical edge ID (smaller index first) to avoid duplicates
		var canonicalID string
		if fromNormalized <= toNormalized {
			canonicalID = fmt.Sprintf("%d-%d", fromNormalized, toNormalized)
		} else {
			canonicalID = fmt.Sprintf("%d-%d", toNormalized, fromNormalized)
		}
		
		// Only process each undirected edge once
		if !processedEdges[canonicalID] {
			normalizedGraph.AddEdge(fromNormalized, toNormalized, weight)
			processedEdges[canonicalID] = true
			edgeCount++
		}
	}
	
	// Validate the converted graph
	if err := normalizedGraph.Validate(); err != nil {
		return nil, nil, fmt.Errorf("converted graph validation failed: %w", err)
	}

	return normalizedGraph, parser, nil
}

// writeLouvainOutputs generates Louvain output files
func writeLouvainOutputs(result *louvain.LouvainResult, parser *louvain.GraphParser, materializedGraph *materialization.HomogeneousGraph, config *PipelineConfig) error {
	// Create output directory
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Write materialized graph files
	if materializedGraph != nil {
		// Write edge list (simple format for other tools)
		edgeListPath := filepath.Join(config.OutputDir, "materialized_graph.edgelist")
		if err := materialization.SaveAsSimpleEdgeList(materializedGraph, edgeListPath); err != nil {
			return fmt.Errorf("failed to write materialized edgelist: %w", err)
		}
	}
	
	// Write Louvain results
	writer := louvain.NewFileWriter()
	if err := writer.WriteAll(result, parser, config.OutputDir, config.OutputPrefix); err != nil {
		return fmt.Errorf("failed to write Louvain results: %w", err)
	}
	
	return nil
}

// allNodesAreIntegers checks if all node IDs are integers
func allNodesAreIntegers(nodes []string) bool {
	for _, node := range nodes {
		if _, err := strconv.Atoi(node); err != nil {
			return false
		}
	}
	return true
}

// RunMaterializationLouvain executes the materialization + Louvain pipeline (from your original code)
func RunMaterializationLouvain(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*PipelineResult, error) {
	startTime := time.Now()
	
	if config.Verbose {
		fmt.Println("=== Running Materialization + Louvain Pipeline ===")
	}
	
	// Step 1: Parse SCAR input for materialization
	if config.Verbose {
		fmt.Println("Step 1: Parsing input files for materialization...")
	}
	
	graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
	if err != nil {
		return nil, fmt.Errorf("failed to parse SCAR input: %w", err)
	}
	
	if config.Verbose {
		fmt.Printf("  Loaded graph with %d nodes\n", len(graph.Nodes))
	}
	
	// Step 2: Run materialization
	if config.Verbose {
		fmt.Println("Step 2: Running graph materialization...")
	}
	
	materializationStart := time.Now()
	
	// Setup progress callback for materialization
	var materializationProgressCb func(int, int, string)
	if config.Verbose {
		materializationProgressCb = func(current, total int, message string) {
			fmt.Printf("  Materialization progress: %d/%d - %s\n", current, total, message)
		}
	}
	
	engine := materialization.NewMaterializationEngine(graph, metaPath, config.MaterializationConfig, materializationProgressCb)
	materializationResult, err := engine.Materialize()
	if err != nil {
		return nil, fmt.Errorf("materialization failed: %w", err)
	}
	
	materializationTime := time.Since(materializationStart)
	
	if config.Verbose {
		fmt.Printf("  Materialization completed in %v\n", materializationTime)
		fmt.Printf("  Materialized graph has %d nodes and %d edges\n", 
			len(materializationResult.HomogeneousGraph.Nodes),
			len(materializationResult.HomogeneousGraph.Edges))
	}
	
	// Step 3: Convert HomogeneousGraph to NormalizedGraph for Louvain
	if config.Verbose {
		fmt.Println("Step 3: Converting graph format for Louvain...")
	}
	
	normalizedGraph, graphParser, err := convertHomogeneousToNormalized(materializationResult.HomogeneousGraph)
	if err != nil {
		return nil, fmt.Errorf("graph conversion failed: %w", err)
	}
	
	if config.Verbose {
		fmt.Printf("  Converted to normalized graph with %d nodes\n", normalizedGraph.NumNodes)
		fmt.Printf("  Total edge weight: %.2f\n", normalizedGraph.TotalWeight)
	}
	
	// Step 4: Run Louvain clustering
	if config.Verbose {
		fmt.Println("Step 4: Running Louvain community detection...")
	}
	
	louvainStart := time.Now()
	
	// Setup progress callback for Louvain
	if config.Verbose {
		config.LouvainConfig.Verbose = true
		config.LouvainConfig.ProgressCallback = func(level, iteration int, message string) {
			fmt.Printf("  Louvain [L%d I%d]: %s\n", level, iteration, message)
		}
	}
	
	louvainResult, err := louvain.RunLouvain(normalizedGraph, config.LouvainConfig)
	if err != nil {
		return nil, fmt.Errorf("Louvain clustering failed: %w", err)
	}
	
	louvainTime := time.Since(louvainStart)
	louvainResult.Parser = graphParser // Attach parser for output generation
	
	if config.Verbose {
		fmt.Printf("  Louvain completed in %v\n", louvainTime)
		fmt.Printf("  Final modularity: %.6f\n", louvainResult.Modularity)
		finalLevel := louvainResult.Levels[len(louvainResult.Levels)-1]
		fmt.Printf("  Number of communities: %d\n", finalLevel.NumCommunities)
		fmt.Printf("  Hierarchy levels: %d\n", louvainResult.NumLevels)
	}
	
	// Step 5: Generate output files
	if config.Verbose {
		fmt.Println("Step 5: Writing output files...")
	}
	
	if err := writeLouvainOutputs(louvainResult, graphParser, materializationResult.HomogeneousGraph, config); err != nil {
		return nil, fmt.Errorf("output generation failed: %w", err)
	}
	
	totalTime := time.Since(startTime)
	
	// Create final result
	result := &PipelineResult{
		PipelineType:      MaterializationLouvain,
		MaterializedGraph: materializationResult.HomogeneousGraph,
		LouvainResult:     louvainResult,
		TotalRuntimeMS:    totalTime.Milliseconds(),
	}
	
	if config.Verbose {
		fmt.Println("=== Materialization + Louvain Pipeline Complete ===")
		fmt.Printf("Total runtime: %v\n", totalTime)
		fmt.Printf("Materialization: %v, Louvain: %v\n", materializationTime, louvainTime)
		fmt.Printf("Final modularity: %.6f\n", result.LouvainResult.Modularity)
	}
	
	return result, nil
}

// RunSketchLouvain executes the SCAR sketch-based Louvain pipeline (from your original code)
func RunSketchLouvain(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*PipelineResult, error) {
	startTime := time.Now()
	
	if config.Verbose {
		fmt.Println("=== Running SCAR Sketch-based Louvain Pipeline ===")
	}
	
	// Step 1: Configure SCAR with input files
	if config.Verbose {
		fmt.Println("Step 1: Configuring SCAR engine...")
	}
	
	// Create a copy of SCAR config and set file paths
	scarConfig := config.SCARConfig
	scarConfig.GraphFile = graphFile
	scarConfig.PropertyFile = propertiesFile
	scarConfig.PathFile = pathFile
	scarConfig.Prefix = filepath.Join(config.OutputDir, config.OutputPrefix)
	scarConfig.NumWorkers = 1
	if config.Verbose {
		fmt.Printf("  Graph file: %s\n", graphFile)
		fmt.Printf("  Properties file: %s\n", propertiesFile)
		fmt.Printf("  Path file: %s\n", pathFile)
		fmt.Printf("  SCAR parameters: k=%d, nk=%d, threshold=%.3f\n", 
			scarConfig.K, scarConfig.NK, scarConfig.Threshold)
		fmt.Printf("  Sketch output: %t\n", scarConfig.SketchOutput)
	}
	
	// Step 2: Create output directory
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Step 3: Run SCAR engine
	if config.Verbose {
		fmt.Println("Step 2: Running SCAR sketch-based Louvain...")
	}
	
	scarStart := time.Now()
	
	engine := scar.NewSketchLouvainEngine(scarConfig)
	err := engine.RunLouvain()
	if err != nil {
		return nil, fmt.Errorf("SCAR sketch Louvain failed: %w", err)
	}
	
	scarTime := time.Since(scarStart)
	totalTime := time.Since(startTime)
	
	if config.Verbose {
		fmt.Printf("  SCAR completed in %v\n", scarTime)
		fmt.Println("Step 3: Writing SCAR summary...")
	}
	
	// Create final result
	result := &PipelineResult{
		PipelineType:   SketchLouvain,
		TotalRuntimeMS: totalTime.Milliseconds(),
		SCARSuccess:    true,
		SCARConfig:     &scarConfig,
	}
	
	if config.Verbose {
		fmt.Println("=== SCAR Sketch Louvain Pipeline Complete ===")
		fmt.Printf("Total runtime: %v\n", totalTime)
		fmt.Printf("SCAR execution: %v\n", scarTime)
		if scarConfig.SketchOutput {
			fmt.Println("Generated SCAR hierarchy files for PPRViz integration")
		}
	}
	
	return result, nil
}


func computeJaccardSimilarity(fileA, fileB string) (float64, error) {
    edgesA, err := loadEdges(fileA)
    if err != nil {
        return 0, fmt.Errorf("loading %s: %w", fileA, err)
    }
    edgesB, err := loadEdges(fileB)
    if err != nil {
        return 0, fmt.Errorf("loading %s: %w", fileB, err)
    }

    // Build sets of canonical undirected edges
    setA := make(map[string]struct{})
    for _, e := range edgesA {
        u, v := e.From, e.To
        if u > v { u, v = v, u }
        setA[u+"|"+v] = struct{}{}
    }
    setB := make(map[string]struct{})
    for _, e := range edgesB {
        u, v := e.From, e.To
        if u > v { u, v = v, u }
        setB[u+"|"+v] = struct{}{}
    }

    // Count intersection and union
    intersection := 0
    for k := range setA {
        if _, ok := setB[k]; ok {
            intersection++
        }
    }
    union := len(setA) + len(setB) - intersection
    if union == 0 {
        return 0, nil
    }
    return float64(intersection) / float64(union), nil
}

// PipelineType defines which pipeline to run
type PipelineType int

const (
	MaterializationLouvain PipelineType = iota
	SketchLouvain
	Comparison
)

// PipelineResult contains results from either pipeline
type PipelineResult struct {
	PipelineType    PipelineType
	TotalRuntimeMS  int64
	
	// Materialization + Louvain results (nil if SketchLouvain was used)
	MaterializedGraph *materialization.HomogeneousGraph
	LouvainResult     *louvain.LouvainResult
	
	// SCAR results (basic info - actual files written to disk)
	SCARSuccess bool
	SCARConfig  *scar.SCARConfig
}
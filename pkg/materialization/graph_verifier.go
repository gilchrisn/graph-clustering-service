package materialization

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	// "strings"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// VerificationResult contains the results of verification
type VerificationResult struct {
	Passed           bool                   `json:"passed"`
	TotalTests       int                    `json:"total_tests"`
	PassedTests      int                    `json:"passed_tests"`
	FailedTests      int                    `json:"failed_tests"`
	TestResults      []TestResult           `json:"test_results"`
	GraphStats       GraphVerificationStats `json:"graph_stats"`
	RecommendedFixes []string               `json:"recommended_fixes"`
}

// TestResult represents a single test result
type TestResult struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Passed      bool   `json:"passed"`
	Expected    string `json:"expected"`
	Actual      string `json:"actual"`
	ErrorMsg    string `json:"error_msg,omitempty"`
	Severity    string `json:"severity"` // "critical", "warning", "info"
}

// GraphVerificationStats contains statistics for verification
type GraphVerificationStats struct {
	OriginalGraph    GraphStats `json:"original_graph"`
	MaterializedGraph GraphStats `json:"materialized_graph"`
	MetaPath         MetaPathStats `json:"meta_path"`
}

// GraphStats contains basic graph statistics
type GraphStats struct {
	NodeCount      int                `json:"node_count"`
	EdgeCount      int                `json:"edge_count"`
	NodeTypes      map[string]int     `json:"node_types"`      // type -> count
	EdgeTypes      map[string]int     `json:"edge_types"`      // type -> count
	AvgDegree      float64            `json:"avg_degree"`
	MaxDegree      int                `json:"max_degree"`
	ConnectedComponents int           `json:"connected_components"`
}

// MetaPathStats contains meta path statistics
type MetaPathStats struct {
	Length           int      `json:"length"`
	IsSymmetric      bool     `json:"is_symmetric"`
	StartNodeType    string   `json:"start_node_type"`
	EndNodeType      string   `json:"end_node_type"`
	NodeTypes        []string `json:"node_types"`
	EdgeTypes        []string `json:"edge_types"`
	EstimatedInstances int    `json:"estimated_instances"`
}

// GraphVerifier handles verification of graph materialization
type GraphVerifier struct {
	originalGraph      *models.HeterogeneousGraph
	metaPath          *models.MetaPath
	materializedResult *MaterializationResult
	config            MaterializationConfig
	results           []TestResult
}

// NewGraphVerifier creates a new graph verifier
func NewGraphVerifier() *GraphVerifier {
	return &GraphVerifier{
		results: make([]TestResult, 0),
	}
}

// LoadFromFiles loads graph and meta path from files
func (gv *GraphVerifier) LoadFromFiles(graphFile, metaPathFile string) error {
	// Load heterogeneous graph
	graph, err := gv.loadGraphFromFile(graphFile)
	if err != nil {
		return fmt.Errorf("failed to load graph: %w", err)
	}
	gv.originalGraph = graph

	// Load meta path
	metaPath, err := gv.loadMetaPathFromFile(metaPathFile)
	if err != nil {
		return fmt.Errorf("failed to load meta path: %w", err)
	}
	gv.metaPath = metaPath

	return nil
}

// LoadFromObjects loads graph and meta path from objects
func (gv *GraphVerifier) LoadFromObjects(graph *models.HeterogeneousGraph, metaPath *models.MetaPath) {
	gv.originalGraph = graph
	gv.metaPath = metaPath
}

// VerifyMaterialization performs the materialization and verifies the results
func (gv *GraphVerifier) VerifyMaterialization(config MaterializationConfig) (*VerificationResult, error) {
	gv.config = config
	gv.results = make([]TestResult, 0)

	// Perform materialization
	engine := NewMaterializationEngine(gv.originalGraph, gv.metaPath, config, nil)
	result, err := engine.Materialize()
	if err != nil {
		return nil, fmt.Errorf("materialization failed: %w", err)
	}

	gv.materializedResult = result

	// Run all verification tests
	gv.runAllTests()

	// Compile results
	return gv.compileResults(), nil
}

// runAllTests executes all verification tests
func (gv *GraphVerifier) runAllTests() {
	// Critical tests
	gv.testBasicValidation()
	gv.testNodeCountConsistency()
	gv.testMetaPathTraversability()
	gv.testEdgeTypeConsistency()
	gv.testNodeTypeConsistency()
	
	// Quality tests
	gv.testPathInstanceValidity()
	gv.testWeightConsistency()
	gv.testSymmetryProperties()
	gv.testConnectivityPreservation()
	gv.testEdgeCountReasonableness()
	
	// Performance tests
	gv.testMemoryUsage()
	gv.testProcessingTime()
	
	// Edge case tests
	gv.testIsolatedNodes()
	gv.testSelfLoops()
	gv.testDuplicateEdges()
}

// testBasicValidation performs basic validation checks
func (gv *GraphVerifier) testBasicValidation() {
	// Test 1: Materialization succeeded
	gv.addTestResult("materialization_success", 
		"Materialization process completed successfully",
		gv.materializedResult.Success,
		"true", fmt.Sprintf("%t", gv.materializedResult.Success),
		gv.materializedResult.Error, "critical")

	// Test 2: Result graph is not nil
	gv.addTestResult("result_graph_exists",
		"Materialized graph exists",
		gv.materializedResult.HomogeneousGraph != nil,
		"non-nil graph", 
		fmt.Sprintf("graph exists: %t", gv.materializedResult.HomogeneousGraph != nil),
		"", "critical")

	// Test 3: Statistics are populated
	gv.addTestResult("statistics_populated",
		"Processing statistics are populated",
		gv.materializedResult.Statistics.RuntimeMS > 0,
		"positive runtime", 
		fmt.Sprintf("%d ms", gv.materializedResult.Statistics.RuntimeMS),
		"", "warning")
}

// testNodeCountConsistency verifies node count consistency
func (gv *GraphVerifier) testNodeCountConsistency() {
	if gv.materializedResult.HomogeneousGraph == nil {
		return
	}

	startNodeType := gv.metaPath.NodeSequence[0]
	endNodeType := gv.metaPath.NodeSequence[len(gv.metaPath.NodeSequence)-1]

	// Count original nodes of start type
	originalStartCount := 0
	for _, node := range gv.originalGraph.Nodes {
		if node.Type == startNodeType {
			originalStartCount++
		}
	}

	// Count materialized nodes
	materializedCount := len(gv.materializedResult.HomogeneousGraph.Nodes)

	// For symmetric paths, materialized count should equal original start count
	if gv.metaPath.IsSymmetric() {
		gv.addTestResult("symmetric_node_count",
			fmt.Sprintf("Node count matches for symmetric path (type: %s)", startNodeType),
			materializedCount == originalStartCount,
			fmt.Sprintf("%d nodes", originalStartCount),
			fmt.Sprintf("%d nodes", materializedCount),
			"", "critical")
	} else {
		// For non-symmetric paths, check that materialized count is reasonable
		// (should be <= original start count + original end count)
		originalEndCount := 0
		if startNodeType != endNodeType {
			for _, node := range gv.originalGraph.Nodes {
				if node.Type == endNodeType {
					originalEndCount++
				}
			}
		}
		
		maxExpected := originalStartCount + originalEndCount
		gv.addTestResult("asymmetric_node_count",
			"Node count is reasonable for asymmetric path",
			materializedCount <= maxExpected,
			fmt.Sprintf("<= %d nodes", maxExpected),
			fmt.Sprintf("%d nodes", materializedCount),
			"", "warning")
	}

	// Test that all materialized nodes existed in original graph
	allNodesExisted := true
	for nodeID := range gv.materializedResult.HomogeneousGraph.Nodes {
		if _, exists := gv.originalGraph.Nodes[nodeID]; !exists {
			allNodesExisted = false
			break
		}
	}

	gv.addTestResult("nodes_exist_in_original",
		"All materialized nodes existed in original graph",
		allNodesExisted,
		"all nodes from original",
		fmt.Sprintf("valid nodes: %t", allNodesExisted),
		"", "critical")
}

// testMetaPathTraversability checks if meta path can be traversed
func (gv *GraphVerifier) testMetaPathTraversability() {
	traversable := true
	errorMsg := ""

	for step := 0; step < len(gv.metaPath.EdgeSequence); step++ {
		fromNodeType := gv.metaPath.NodeSequence[step]
		toNodeType := gv.metaPath.NodeSequence[step+1]
		edgeType := gv.metaPath.EdgeSequence[step]

		// Check if this transition exists in the graph
		transitionExists := false
		for _, edge := range gv.originalGraph.Edges {
			if edge.Type == edgeType {
				fromNode, fromExists := gv.originalGraph.Nodes[edge.From]
				toNode, toExists := gv.originalGraph.Nodes[edge.To]

				if fromExists && toExists &&
					fromNode.Type == fromNodeType &&
					toNode.Type == toNodeType {
					transitionExists = true
					break
				}
			}
		}

		if !transitionExists {
			traversable = false
			errorMsg = fmt.Sprintf("Step %d not traversable: %s -[%s]-> %s", 
				step, fromNodeType, edgeType, toNodeType)
			break
		}
	}

	gv.addTestResult("meta_path_traversable",
		"Meta path is traversable in the graph",
		traversable,
		"all steps traversable",
		fmt.Sprintf("traversable: %t", traversable),
		errorMsg, "critical")
}

// testEdgeTypeConsistency checks edge type consistency
func (gv *GraphVerifier) testEdgeTypeConsistency() {
	// Check that all edge types in meta path exist in graph
	graphEdgeTypes := make(map[string]bool)
	for _, edge := range gv.originalGraph.Edges {
		graphEdgeTypes[edge.Type] = true
	}

	allEdgeTypesExist := true
	missingTypes := []string{}

	for _, edgeType := range gv.metaPath.EdgeSequence {
		if !graphEdgeTypes[edgeType] {
			allEdgeTypesExist = false
			missingTypes = append(missingTypes, edgeType)
		}
	}

	gv.addTestResult("edge_types_exist",
		"All meta path edge types exist in graph",
		allEdgeTypesExist,
		"all edge types present",
		fmt.Sprintf("missing types: %v", missingTypes),
		"", "critical")
}

// testNodeTypeConsistency checks node type consistency
func (gv *GraphVerifier) testNodeTypeConsistency() {
	// Check that all node types in meta path exist in graph
	graphNodeTypes := make(map[string]bool)
	for _, node := range gv.originalGraph.Nodes {
		graphNodeTypes[node.Type] = true
	}

	allNodeTypesExist := true
	missingTypes := []string{}

	for _, nodeType := range gv.metaPath.NodeSequence {
		if !graphNodeTypes[nodeType] {
			allNodeTypesExist = false
			missingTypes = append(missingTypes, nodeType)
		}
	}

	gv.addTestResult("node_types_exist",
		"All meta path node types exist in graph",
		allNodeTypesExist,
		"all node types present",
		fmt.Sprintf("missing types: %v", missingTypes),
		"", "critical")
}

// testPathInstanceValidity checks path instance validity
func (gv *GraphVerifier) testPathInstanceValidity() {
	// This requires running the instance generator to check path instances
	config := DefaultTraversalConfig()
	config.MaxInstances = 1000 // Limit for testing
	
	generator := NewInstanceGenerator(gv.originalGraph, gv.metaPath, config)
	instances, _, err := generator.FindAllInstances(nil)

	if err != nil {
		gv.addTestResult("path_instances_generated",
			"Path instances can be generated",
			false, "no errors", fmt.Sprintf("error: %v", err),
			err.Error(), "warning")
		return
	}

	// Check that instances match expected count
	expectedCount := gv.materializedResult.Statistics.InstancesGenerated
	actualCount := len(instances)

	gv.addTestResult("path_instance_count",
		"Path instance count matches statistics",
		actualCount == expectedCount,
		fmt.Sprintf("%d instances", expectedCount),
		fmt.Sprintf("%d instances", actualCount),
		"", "warning")

	// Check instance validity
	validInstances := 0
	for _, instance := range instances {
		if instance.IsValid() {
			validInstances++
		}
	}

	gv.addTestResult("path_instances_valid",
		"All generated path instances are valid",
		validInstances == len(instances),
		fmt.Sprintf("%d valid instances", len(instances)),
		fmt.Sprintf("%d valid instances", validInstances),
		"", "warning")
}

// testWeightConsistency checks weight consistency
func (gv *GraphVerifier) testWeightConsistency() {
	if gv.materializedResult.HomogeneousGraph == nil || len(gv.materializedResult.HomogeneousGraph.Edges) == 0 {
		return
	}

	// Check for invalid weights
	validWeights := true
	invalidWeightCount := 0

	for _, weight := range gv.materializedResult.HomogeneousGraph.Edges {
		if math.IsNaN(weight) || math.IsInf(weight, 0) || weight < 0 {
			validWeights = false
			invalidWeightCount++
		}
	}

	gv.addTestResult("weights_valid",
		"All edge weights are valid (not NaN, infinite, or negative)",
		validWeights,
		"all weights valid",
		fmt.Sprintf("%d invalid weights", invalidWeightCount),
		"", "critical")

	// Check weight distribution
	totalEdges := len(gv.materializedResult.HomogeneousGraph.Edges)
	if totalEdges > 0 {
		stats := gv.materializedResult.HomogeneousGraph.Statistics
		reasonableWeights := stats.MinWeight >= 0 && stats.MaxWeight < math.Inf(1)

		gv.addTestResult("weight_distribution",
			"Weight distribution is reasonable",
			reasonableWeights,
			"min >= 0, max < infinity",
			fmt.Sprintf("min: %.2f, max: %.2f", stats.MinWeight, stats.MaxWeight),
			"", "warning")
	}
}

// testSymmetryProperties checks symmetry properties for symmetric paths
func (gv *GraphVerifier) testSymmetryProperties() {
	if !gv.metaPath.IsSymmetric() || !gv.config.Aggregation.Symmetric {
		return // Skip if not symmetric
	}

	if gv.materializedResult.HomogeneousGraph == nil {
		return
	}

	// Check that for every edge (A,B), there exists edge (B,A) with same weight
	symmetryViolations := 0
	checkedEdges := make(map[EdgeKey]bool)

	for edgeKey, weight := range gv.materializedResult.HomogeneousGraph.Edges {
		if checkedEdges[edgeKey] {
			continue
		}

		reverseKey := edgeKey.Reverse()
		reverseWeight, reverseExists := gv.materializedResult.HomogeneousGraph.Edges[reverseKey]

		if !reverseExists || math.Abs(weight-reverseWeight) > 0.001 {
			symmetryViolations++
		}

		checkedEdges[edgeKey] = true
		checkedEdges[reverseKey] = true
	}

	gv.addTestResult("graph_symmetry",
		"Graph maintains symmetry for symmetric meta path",
		symmetryViolations == 0,
		"0 symmetry violations",
		fmt.Sprintf("%d violations", symmetryViolations),
		"", "warning")
}

// testConnectivityPreservation checks connectivity preservation
func (gv *GraphVerifier) testConnectivityPreservation() {
	if gv.materializedResult.HomogeneousGraph == nil {
		return
	}

	// Calculate connected components in materialized graph
	components := gv.calculateConnectedComponents(gv.materializedResult.HomogeneousGraph)
	componentCount := len(components)

	// For most cases, we expect the graph to be reasonably connected
	// This is a soft check - too many components might indicate issues
	nodeCount := len(gv.materializedResult.HomogeneousGraph.Nodes)
	maxReasonableComponents := int(math.Sqrt(float64(nodeCount))) + 1

	reasonableConnectivity := componentCount <= maxReasonableComponents

	gv.addTestResult("connectivity_preserved",
		"Graph maintains reasonable connectivity",
		reasonableConnectivity,
		fmt.Sprintf("<= %d components", maxReasonableComponents),
		fmt.Sprintf("%d components", componentCount),
		"", "info")
}

// testEdgeCountReasonableness checks if edge count is reasonable
func (gv *GraphVerifier) testEdgeCountReasonableness() {
	if gv.materializedResult.HomogeneousGraph == nil {
		return
	}

	nodeCount := len(gv.materializedResult.HomogeneousGraph.Nodes)
	edgeCount := len(gv.materializedResult.HomogeneousGraph.Edges)

	if nodeCount == 0 {
		return
	}

	// Maximum possible edges in undirected graph: n*(n-1)/2
	// Maximum possible edges in directed graph: n*(n-1)
	maxPossibleEdges := nodeCount * (nodeCount - 1)
	if gv.config.Aggregation.Symmetric {
		maxPossibleEdges = maxPossibleEdges / 2
	}

	reasonableEdgeCount := edgeCount <= maxPossibleEdges

	gv.addTestResult("edge_count_reasonable",
		"Edge count is within reasonable bounds",
		reasonableEdgeCount,
		fmt.Sprintf("<= %d edges", maxPossibleEdges),
		fmt.Sprintf("%d edges", edgeCount),
		"", "warning")

	// Check density
	density := float64(edgeCount) / float64(maxPossibleEdges)
	reasonableDensity := density <= 1.0

	gv.addTestResult("graph_density",
		"Graph density is reasonable",
		reasonableDensity,
		"<= 1.0",
		fmt.Sprintf("%.3f", density),
		"", "info")
}

// testMemoryUsage checks memory usage
func (gv *GraphVerifier) testMemoryUsage() {
	memoryMB := gv.materializedResult.Statistics.MemoryPeakMB
	
	// Check if memory usage is reported
	memoryReported := memoryMB > 0
	gv.addTestResult("memory_reported",
		"Memory usage is reported",
		memoryReported,
		"> 0 MB",
		fmt.Sprintf("%d MB", memoryMB),
		"", "info")

	// Check if memory usage is reasonable (< 10GB for most cases)
	reasonableMemory := memoryMB < 10240 // 10GB
	gv.addTestResult("memory_reasonable",
		"Memory usage is reasonable",
		reasonableMemory,
		"< 10GB",
		fmt.Sprintf("%d MB", memoryMB),
		"", "warning")
}

// testProcessingTime checks processing time
func (gv *GraphVerifier) testProcessingTime() {
	runtimeMS := gv.materializedResult.Statistics.RuntimeMS

	// Check if runtime is reasonable (< 5 minutes for most cases)
	reasonableTime := runtimeMS < 300000 // 5 minutes
	gv.addTestResult("processing_time",
		"Processing time is reasonable",
		reasonableTime,
		"< 5 minutes",
		fmt.Sprintf("%.2f seconds", float64(runtimeMS)/1000),
		"", "info")
}

// testIsolatedNodes checks for isolated nodes
func (gv *GraphVerifier) testIsolatedNodes() {
	if gv.materializedResult.HomogeneousGraph == nil {
		return
	}

	isolatedCount := 0
	for _, node := range gv.materializedResult.HomogeneousGraph.Nodes {
		if node.Degree == 0 {
			isolatedCount++
		}
	}

	noIsolatedNodes := isolatedCount == 0
	gv.addTestResult("no_isolated_nodes",
		"No isolated nodes in result (unless filtered)",
		noIsolatedNodes,
		"0 isolated nodes",
		fmt.Sprintf("%d isolated nodes", isolatedCount),
		"", "warning")
}

// testSelfLoops checks for self-loops
func (gv *GraphVerifier) testSelfLoops() {
	if gv.materializedResult.HomogeneousGraph == nil {
		return
	}

	selfLoopCount := 0
	for edgeKey := range gv.materializedResult.HomogeneousGraph.Edges {
		if edgeKey.From == edgeKey.To {
			selfLoopCount++
		}
	}

	// Self-loops are typically removed, so we expect 0
	noSelfLoops := selfLoopCount == 0
	gv.addTestResult("no_self_loops",
		"No self-loops in result (typically removed)",
		noSelfLoops,
		"0 self-loops",
		fmt.Sprintf("%d self-loops", selfLoopCount),
		"", "info")
}

// testDuplicateEdges checks for duplicate edges
func (gv *GraphVerifier) testDuplicateEdges() {
	if gv.materializedResult.HomogeneousGraph == nil {
		return
	}

	// In a map-based representation, duplicates are automatically handled
	// This is more of a sanity check
	edgeCount := len(gv.materializedResult.HomogeneousGraph.Edges)
	uniqueEdges := make(map[string]bool)

	for edgeKey := range gv.materializedResult.HomogeneousGraph.Edges {
		keyStr := fmt.Sprintf("%s->%s", edgeKey.From, edgeKey.To)
		uniqueEdges[keyStr] = true
	}

	noDuplicates := len(uniqueEdges) == edgeCount
	gv.addTestResult("no_duplicate_edges",
		"No duplicate edges in result",
		noDuplicates,
		fmt.Sprintf("%d unique edges", edgeCount),
		fmt.Sprintf("%d unique edges", len(uniqueEdges)),
		"", "info")
}

// Helper methods

// addTestResult adds a test result to the results list
func (gv *GraphVerifier) addTestResult(name, description string, passed bool, expected, actual, errorMsg, severity string) {
	result := TestResult{
		Name:        name,
		Description: description,
		Passed:      passed,
		Expected:    expected,
		Actual:      actual,
		ErrorMsg:    errorMsg,
		Severity:    severity,
	}
	gv.results = append(gv.results, result)
}

// compileResults compiles all test results into a verification result
func (gv *GraphVerifier) compileResults() *VerificationResult {
	totalTests := len(gv.results)
	passedTests := 0
	failedTests := 0

	for _, result := range gv.results {
		if result.Passed {
			passedTests++
		} else {
			failedTests++
		}
	}

	// Generate recommended fixes
	fixes := gv.generateRecommendedFixes()

	// Generate statistics
	stats := gv.generateVerificationStats()

	return &VerificationResult{
		Passed:           failedTests == 0,
		TotalTests:       totalTests,
		PassedTests:      passedTests,
		FailedTests:      failedTests,
		TestResults:      gv.results,
		GraphStats:       stats,
		RecommendedFixes: fixes,
	}
}

// generateRecommendedFixes generates recommended fixes based on failed tests
func (gv *GraphVerifier) generateRecommendedFixes() []string {
	fixes := []string{}

	for _, result := range gv.results {
		if !result.Passed && result.Severity == "critical" {
			switch result.Name {
			case "materialization_success":
				fixes = append(fixes, "Check input data validity and meta path correctness")
			case "symmetric_node_count":
				fixes = append(fixes, "Verify meta path traversability and node connectivity")
			case "meta_path_traversable":
				fixes = append(fixes, "Ensure all edge types and node types in meta path exist in graph")
			case "weights_valid":
				fixes = append(fixes, "Check aggregation strategy and weight calculation logic")
			case "nodes_exist_in_original":
				fixes = append(fixes, "Verify node ID consistency between input and output")
			}
		}
	}

	if len(fixes) == 0 {
		fixes = append(fixes, "All critical tests passed - no immediate fixes needed")
	}

	return fixes
}

// generateVerificationStats generates statistics for verification
func (gv *GraphVerifier) generateVerificationStats() GraphVerificationStats {
	originalStats := gv.calculateGraphStats(gv.originalGraph)
	
	var materializedStats GraphStats
	if gv.materializedResult != nil && gv.materializedResult.HomogeneousGraph != nil {
		materializedStats = gv.calculateHomogeneousGraphStats(gv.materializedResult.HomogeneousGraph)
	}

	metaPathStats := gv.calculateMetaPathStats()

	return GraphVerificationStats{
		OriginalGraph:     originalStats,
		MaterializedGraph: materializedStats,
		MetaPath:          metaPathStats,
	}
}

// calculateGraphStats calculates statistics for heterogeneous graph
func (gv *GraphVerifier) calculateGraphStats(graph *models.HeterogeneousGraph) GraphStats {
	nodeTypes := make(map[string]int)
	edgeTypes := make(map[string]int)
	
	for _, node := range graph.Nodes {
		nodeTypes[node.Type]++
	}
	
	for _, edge := range graph.Edges {
		edgeTypes[edge.Type]++
	}

	// Calculate average degree
	nodeDegrees := make(map[string]int)
	for _, edge := range graph.Edges {
		nodeDegrees[edge.From]++
		nodeDegrees[edge.To]++
	}

	totalDegree := 0
	maxDegree := 0
	for _, degree := range nodeDegrees {
		totalDegree += degree
		if degree > maxDegree {
			maxDegree = degree
		}
	}

	avgDegree := 0.0
	if len(graph.Nodes) > 0 {
		avgDegree = float64(totalDegree) / float64(len(graph.Nodes))
	}

	return GraphStats{
		NodeCount:           len(graph.Nodes),
		EdgeCount:           len(graph.Edges),
		NodeTypes:           nodeTypes,
		EdgeTypes:           edgeTypes,
		AvgDegree:           avgDegree,
		MaxDegree:           maxDegree,
		ConnectedComponents: gv.calculateHeterogeneousConnectedComponents(graph),
	}
}

// InvestigateMissingNodes provides detailed analysis of why nodes are missing
func (gv *GraphVerifier) InvestigateMissingNodes() map[string]interface{} {
	if gv.materializedResult == nil || gv.materializedResult.HomogeneousGraph == nil {
		return map[string]interface{}{"error": "No materialization result available"}
	}

	startNodeType := gv.metaPath.NodeSequence[0]
	
	// Find missing nodes
	originalNodes := []string{}
	materializedNodes := make(map[string]bool)
	
	for nodeID, node := range gv.originalGraph.Nodes {
		if node.Type == startNodeType {
			originalNodes = append(originalNodes, nodeID)
		}
	}
	
	for nodeID := range gv.materializedResult.HomogeneousGraph.Nodes {
		materializedNodes[nodeID] = true
	}
	
	missingNodes := []string{}
	for _, nodeID := range originalNodes {
		if !materializedNodes[nodeID] {
			missingNodes = append(missingNodes, nodeID)
		}
	}

	// Analyze each missing node
	analysis := make(map[string]map[string]interface{})
	
	for _, nodeID := range missingNodes {
		nodeAnalysis := map[string]interface{}{
			"node_id": nodeID,
			"outgoing_edges": 0,
			"valid_first_step": false,
			"edge_types": []string{},
		}
		
		// Check outgoing edges
		for _, edge := range gv.originalGraph.Edges {
			if edge.From == nodeID {
				nodeAnalysis["outgoing_edges"] = nodeAnalysis["outgoing_edges"].(int) + 1
				nodeAnalysis["edge_types"] = append(nodeAnalysis["edge_types"].([]string), edge.Type)
				
				// Check if this edge matches the first step of meta path
				if len(gv.metaPath.EdgeSequence) > 0 && edge.Type == gv.metaPath.EdgeSequence[0] {
					nodeAnalysis["valid_first_step"] = true
				}
			}
		}
		
		analysis[nodeID] = nodeAnalysis
	}

	// Summary categories
	isolated := 0
	noValidFirstStep := 0
	hasEdgesButNoPath := 0
	
	for _, nodeAnalysis := range analysis {
		outgoingEdges := nodeAnalysis["outgoing_edges"].(int)
		validFirstStep := nodeAnalysis["valid_first_step"].(bool)
		
		if outgoingEdges == 0 {
			isolated++
		} else if !validFirstStep {
			noValidFirstStep++
		} else {
			hasEdgesButNoPath++
		}
	}

	return map[string]interface{}{
		"total_original": len(originalNodes),
		"total_materialized": len(materializedNodes),
		"missing_count": len(missingNodes),
		"missing_nodes": missingNodes,
		"detailed_analysis": analysis,
		"summary": map[string]int{
			"isolated_nodes": isolated,
			"no_valid_first_step": noValidFirstStep,
			"has_edges_but_no_complete_path": hasEdgesButNoPath,
		},
	}
}

// PrintMissingNodesReport prints a detailed report of missing nodes
func PrintMissingNodesReport(investigation map[string]interface{}) {
	fmt.Printf("\n=== MISSING NODES INVESTIGATION ===\n")
	
	if errMsg, hasError := investigation["error"]; hasError {
		fmt.Printf("Error: %s\n", errMsg)
		return
	}
	
	totalOriginal := investigation["total_original"].(int)
	totalMaterialized := investigation["total_materialized"].(int)
	missingCount := investigation["missing_count"].(int)
	
	fmt.Printf("Original nodes: %d\n", totalOriginal)
	fmt.Printf("Materialized nodes: %d\n", totalMaterialized)
	fmt.Printf("Missing nodes: %d\n", missingCount)
	
	if missingCount == 0 {
		fmt.Printf("✅ No missing nodes!\n")
		return
	}
	
	summary := investigation["summary"].(map[string]int)
	fmt.Printf("\n=== MISSING NODES BREAKDOWN ===\n")
	fmt.Printf("🔌 Isolated nodes (no outgoing edges): %d\n", summary["isolated_nodes"])
	fmt.Printf("🚫 No valid first step (wrong edge types): %d\n", summary["no_valid_first_step"])  
	fmt.Printf("❓ Has edges but no complete path: %d\n", summary["has_edges_but_no_complete_path"])
	
	// Show sample missing nodes
	missingNodes := investigation["missing_nodes"].([]string)
	if len(missingNodes) > 0 {
		fmt.Printf("\n=== SAMPLE MISSING NODES ===\n")
		showCount := 3
		if len(missingNodes) < showCount {
			showCount = len(missingNodes)
		}
		
		detailedAnalysis := investigation["detailed_analysis"].(map[string]map[string]interface{})
		for i := 0; i < showCount; i++ {
			nodeID := missingNodes[i]
			analysis := detailedAnalysis[nodeID]
			
			fmt.Printf("Node: %s\n", nodeID)
			fmt.Printf("  - Outgoing edges: %d\n", analysis["outgoing_edges"])
			fmt.Printf("  - Valid first step: %t\n", analysis["valid_first_step"])
			fmt.Printf("  - Edge types: %v\n", analysis["edge_types"])
			fmt.Println()
		}
		
		if len(missingNodes) > showCount {
			fmt.Printf("... and %d more missing nodes\n", len(missingNodes)-showCount)
		}
	}
}

// calculateHomogeneousGraphStats calculates statistics for homogeneous graph
func (gv *GraphVerifier) calculateHomogeneousGraphStats(graph *HomogeneousGraph) GraphStats {
	nodeTypes := make(map[string]int)
	edgeTypes := make(map[string]int)
	
	for _, node := range graph.Nodes {
		nodeTypes[node.Type]++
	}
	
	// For homogeneous graphs, there's typically one edge type
	edgeTypes["materialized"] = len(graph.Edges)

	totalDegree := 0
	maxDegree := 0
	for _, node := range graph.Nodes {
		totalDegree += node.Degree
		if node.Degree > maxDegree {
			maxDegree = node.Degree
		}
	}

	avgDegree := 0.0
	if len(graph.Nodes) > 0 {
		avgDegree = float64(totalDegree) / float64(len(graph.Nodes))
	}

	return GraphStats{
		NodeCount:           len(graph.Nodes),
		EdgeCount:           len(graph.Edges),
		NodeTypes:           nodeTypes,
		EdgeTypes:           edgeTypes,
		AvgDegree:           avgDegree,
		MaxDegree:           maxDegree,
		ConnectedComponents: len(gv.calculateConnectedComponents(graph)),
	}
}

// calculateMetaPathStats calculates meta path statistics
func (gv *GraphVerifier) calculateMetaPathStats() MetaPathStats {
	estimatedInstances := 0
	if gv.originalGraph != nil && gv.metaPath != nil {
		engine := NewMaterializationEngine(gv.originalGraph, gv.metaPath, gv.config, nil)
		estimated, _ := engine.EstimateComplexity()
		estimatedInstances = estimated
	}

	return MetaPathStats{
		Length:             len(gv.metaPath.EdgeSequence),
		IsSymmetric:        gv.metaPath.IsSymmetric(),
		StartNodeType:      gv.metaPath.NodeSequence[0],
		EndNodeType:        gv.metaPath.NodeSequence[len(gv.metaPath.NodeSequence)-1],
		NodeTypes:          gv.metaPath.NodeSequence,
		EdgeTypes:          gv.metaPath.EdgeSequence,
		EstimatedInstances: estimatedInstances,
	}
}

// calculateConnectedComponents calculates connected components for homogeneous graph
func (gv *GraphVerifier) calculateConnectedComponents(graph *HomogeneousGraph) map[string][]string {
	visited := make(map[string]bool)
	components := make(map[string][]string)
	componentID := 0

	var dfs func(string, string)
	dfs = func(nodeID, compID string) {
		visited[nodeID] = true
		components[compID] = append(components[compID], nodeID)

		// Visit neighbors
		for edgeKey := range graph.Edges {
			if edgeKey.From == nodeID && !visited[edgeKey.To] {
				dfs(edgeKey.To, compID)
			}
			if edgeKey.To == nodeID && !visited[edgeKey.From] {
				dfs(edgeKey.From, compID)
			}
		}
	}

	for nodeID := range graph.Nodes {
		if !visited[nodeID] {
			compID := fmt.Sprintf("component_%d", componentID)
			dfs(nodeID, compID)
			componentID++
		}
	}

	return components
}

// calculateHeterogeneousConnectedComponents calculates connected components for heterogeneous graph
func (gv *GraphVerifier) calculateHeterogeneousConnectedComponents(graph *models.HeterogeneousGraph) int {
	visited := make(map[string]bool)
	componentCount := 0

	var dfs func(string)
	dfs = func(nodeID string) {
		visited[nodeID] = true

		// Visit neighbors
		for _, edge := range graph.Edges {
			if edge.From == nodeID && !visited[edge.To] {
				dfs(edge.To)
			}
			if edge.To == nodeID && !visited[edge.From] {
				dfs(edge.From)
			}
		}
	}

	for nodeID := range graph.Nodes {
		if !visited[nodeID] {
			dfs(nodeID)
			componentCount++
		}
	}

	return componentCount
}

// File loading methods

// loadGraphFromFile loads a heterogeneous graph from a file
func (gv *GraphVerifier) loadGraphFromFile(filename string) (*models.HeterogeneousGraph, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var graph models.HeterogeneousGraph
	if err := json.Unmarshal(data, &graph); err != nil {
		return nil, err
	}

	return &graph, nil
}

// loadMetaPathFromFile loads a meta path from a file
func (gv *GraphVerifier) loadMetaPathFromFile(filename string) (*models.MetaPath, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var metaPath models.MetaPath
	if err := json.Unmarshal(data, &metaPath); err != nil {
		return nil, err
	}

	return &metaPath, nil
}

// Utility functions

// SaveVerificationResult saves verification results to a file
func SaveVerificationResult(result *VerificationResult, outputPath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(outputPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal result: %w", err)
	}

	// Write to file
	if err := ioutil.WriteFile(outputPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// PrintVerificationSummary prints a summary of verification results
func PrintVerificationSummary(result *VerificationResult) {
	fmt.Printf("\n=== GRAPH MATERIALIZATION VERIFICATION SUMMARY ===\n")
	fmt.Printf("Overall Result: %s\n", map[bool]string{true: "PASSED", false: "FAILED"}[result.Passed])
	fmt.Printf("Tests: %d total, %d passed, %d failed\n", 
		result.TotalTests, result.PassedTests, result.FailedTests)
	
	if result.FailedTests > 0 {
		fmt.Printf("\n=== FAILED TESTS ===\n")
		for _, test := range result.TestResults {
			if !test.Passed {
				fmt.Printf("❌ %s [%s]: %s\n", test.Name, test.Severity, test.Description)
				fmt.Printf("   Expected: %s, Got: %s\n", test.Expected, test.Actual)
				if test.ErrorMsg != "" {
					fmt.Printf("   Error: %s\n", test.ErrorMsg)
				}
			}
		}
	}

	fmt.Printf("\n=== GRAPH STATISTICS ===\n")
	fmt.Printf("Original Graph: %d nodes, %d edges\n", 
		result.GraphStats.OriginalGraph.NodeCount, result.GraphStats.OriginalGraph.EdgeCount)
	fmt.Printf("Materialized Graph: %d nodes, %d edges\n", 
		result.GraphStats.MaterializedGraph.NodeCount, result.GraphStats.MaterializedGraph.EdgeCount)
	fmt.Printf("Meta Path: %s -> %s (length: %d)\n",
		result.GraphStats.MetaPath.StartNodeType, result.GraphStats.MetaPath.EndNodeType,
		result.GraphStats.MetaPath.Length)

	if len(result.RecommendedFixes) > 0 {
		fmt.Printf("\n=== RECOMMENDED FIXES ===\n")
		for i, fix := range result.RecommendedFixes {
			fmt.Printf("%d. %s\n", i+1, fix)
		}
	}

	fmt.Printf("\n=== END SUMMARY ===\n")
}

// QuickVerify performs a quick verification with default settings
func QuickVerify(graphFile, metaPathFile string) (*VerificationResult, error) {
	verifier := NewGraphVerifier()
	
	if err := verifier.LoadFromFiles(graphFile, metaPathFile); err != nil {
		return nil, err
	}

	config := DefaultMaterializationConfig()
	return verifier.VerifyMaterialization(config)
}
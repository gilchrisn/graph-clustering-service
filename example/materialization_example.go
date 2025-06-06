package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/models"
	"github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

func main() {
	fmt.Println("Graph Materialization Example")
	fmt.Println("=============================")

	if len(os.Args) < 3 {
		fmt.Println("Usage: go run materialization_example.go <graph_file> <metapath_file>")
		fmt.Println("Example: go run materialization_example.go ../data/graph_input.json ../data/meta_path.json")
		os.Exit(1)
	}

	graphFile := os.Args[1]
	metaPathFile := os.Args[2]

	// Run the complete materialization example
	if err := runMaterializationExample(graphFile, metaPathFile); err != nil {
		log.Fatalf("Example failed: %v", err)
	}
}

func runMaterializationExample(graphFile, metaPathFile string) error {
	// Step 1: Load and validate input data
	fmt.Println("📋 Step 1: Loading and validating input data...")
	
	graph, err := validation.LoadAndValidateGraph(graphFile)
	if err != nil {
		return fmt.Errorf("failed to load graph: %w", err)
	}
	
	metaPath, err := validation.LoadAndValidateMetaPath(metaPathFile)
	if err != nil {
		return fmt.Errorf("failed to load meta path: %w", err)
	}
	
	// Validate compatibility
	if err := validation.ValidateMetaPathAgainstGraph(metaPath, graph); err != nil {
		return fmt.Errorf("meta path incompatible with graph: %w", err)
	}
	
	fmt.Printf("✅ Input validation successful!\n")
	fmt.Printf("   Graph: %d nodes, %d edges\n", len(graph.Nodes), len(graph.Edges))
	fmt.Printf("   Meta path: %s\n", metaPath.String())
	
	// Step 2: Configure materialization
	fmt.Println("\n⚙️  Step 2: Configuring materialization...")
	
	config := materialization.DefaultMaterializationConfig()
	
	// Customize configuration for this example
	config.Traversal.MaxInstances = 100000    // Limit instances for demo
	config.Aggregation.Strategy = materialization.Count  // Count instances as edge weights
	config.Aggregation.Symmetric = true       // Create symmetric edges
	config.Aggregation.MinWeight = 1.0        // Require at least 1 instance
	config.Progress.EnableProgress = true      // Show progress
	
	fmt.Printf("✅ Configuration set:\n")
	fmt.Printf("   Strategy: BFS traversal\n")
	fmt.Printf("   Aggregation: %s\n", getAggregationName(config.Aggregation.Strategy))
	fmt.Printf("   Max instances: %d\n", config.Traversal.MaxInstances)
	
	// Step 3: Check feasibility
	fmt.Println("\n🧮 Step 3: Checking materialization feasibility...")
	
	engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
	
	canMaterialize, reason, err := engine.CanMaterialize(1000) // 1GB memory limit
	if err != nil {
		return fmt.Errorf("failed to check feasibility: %w", err)
	}
	
	fmt.Printf("   Feasibility: %v\n", canMaterialize)
	fmt.Printf("   Reason: %s\n", reason)
	
	if !canMaterialize {
		fmt.Println("❌ Materialization not feasible with current settings")
		return nil
	}
	
	// Step 4: Estimate complexity
	complexity, err := engine.EstimateComplexity()
	if err != nil {
		return fmt.Errorf("failed to estimate complexity: %w", err)
	}
	
	memoryEstimate, err := engine.GetMemoryEstimate()
	if err != nil {
		return fmt.Errorf("failed to estimate memory: %w", err)
	}
	
	fmt.Printf("   Estimated instances: %d\n", complexity)
	fmt.Printf("   Estimated memory: %d MB\n", memoryEstimate)
	
	// Step 5: Perform materialization
	fmt.Println("\n🔄 Step 5: Performing materialization...")
	
	// Create progress callback
	progressCallback := func(current, total int, message string) {
		if total > 0 {
			percentage := float64(current) / float64(total) * 100
			fmt.Printf("\r   Progress: %.1f%% (%d/%d) - %s", percentage, current, total, message)
		} else {
			fmt.Printf("\r   %s", message)
		}
	}
	
	// Create engine with progress callback
	engine = materialization.NewMaterializationEngine(graph, metaPath, config, progressCallback)
	
	// Perform materialization
	result, err := engine.Materialize()
	if err != nil {
		return fmt.Errorf("materialization failed: %w", err)
	}
	
	fmt.Println() // New line after progress
	
	if !result.Success {
		return fmt.Errorf("materialization unsuccessful: %s", result.Error)
	}
	
	// Step 6: Analyze results
	fmt.Println("\n📊 Step 6: Analyzing materialization results...")
	
	homogGraph := result.HomogeneousGraph
	stats := result.Statistics
	
	fmt.Printf("✅ Materialization completed successfully!\n")
	fmt.Printf("\n📈 Homogeneous Graph Statistics:\n")
	fmt.Printf("   Node type: %s\n", homogGraph.NodeType)
	fmt.Printf("   Nodes: %d\n", len(homogGraph.Nodes))
	fmt.Printf("   Edges: %d\n", len(homogGraph.Edges))
	fmt.Printf("   Density: %.4f\n", homogGraph.Statistics.Density)
	fmt.Printf("   Average weight: %.2f\n", homogGraph.Statistics.AverageWeight)
	fmt.Printf("   Weight range: %.2f - %.2f\n", homogGraph.Statistics.MinWeight, homogGraph.Statistics.MaxWeight)
	
	fmt.Printf("\n⏱️  Performance Statistics:\n")
	fmt.Printf("   Runtime: %d ms\n", stats.RuntimeMS)
	fmt.Printf("   Peak memory: %d MB\n", stats.MemoryPeakMB)
	fmt.Printf("   Instances generated: %d\n", stats.InstancesGenerated)
	fmt.Printf("   Instances filtered: %d\n", stats.InstancesFiltered)
	fmt.Printf("   Nodes visited: %d\n", stats.TraversalStatistics.NodesVisited)
	fmt.Printf("   Edges traversed: %d\n", stats.TraversalStatistics.EdgesTraversed)
	
	// Step 7: Show sample connections
	fmt.Println("\n🔗 Step 7: Sample connections in materialized graph...")
	
	showSampleConnections(homogGraph, 5)
	
	// Step 8: Demonstrate different configurations
	fmt.Println("\n🧪 Step 8: Comparing different aggregation strategies...")
	
	if err := compareAggregationStrategies(graph, metaPath); err != nil {
		return fmt.Errorf("failed to compare strategies: %w", err)
	}
	
	fmt.Println("\n🎉 Materialization example completed successfully!")
	
	return nil
}

func showSampleConnections(graph *materialization.HomogeneousGraph, maxSamples int) {
	count := 0
	for edgeKey, weight := range graph.Edges {
		if count >= maxSamples {
			break
		}
		
		fromNode := graph.Nodes[edgeKey.From]
		toNode := graph.Nodes[edgeKey.To]
		
		fmt.Printf("   %s ↔ %s (weight: %.2f)\n", 
			fromNode.ID, toNode.ID, weight)
		
		count++
	}
	
	if len(graph.Edges) > maxSamples {
		fmt.Printf("   ... and %d more connections\n", len(graph.Edges)-maxSamples)
	}
}

func compareAggregationStrategies(graph *models.HeterogeneousGraph, metaPath *models.MetaPath) error {
	strategies := []materialization.AggregationStrategy{
		materialization.Count,
		materialization.Sum,
		materialization.Average,
		materialization.Maximum,
	}
	
	strategyNames := []string{"Count", "Sum", "Average", "Maximum"}
	
	fmt.Printf("   Comparing strategies for same graph/meta-path:\n")
	
	for i, strategy := range strategies {
		config := materialization.DefaultMaterializationConfig()
		config.Aggregation.Strategy = strategy
		config.Progress.EnableProgress = false // Disable progress for comparison
		
		engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
		result, err := engine.Materialize()
		if err != nil {
			fmt.Printf("   ❌ %s: failed (%v)\n", strategyNames[i], err)
			continue
		}
		
		homogGraph := result.HomogeneousGraph
		fmt.Printf("   📊 %s: %d edges, avg weight: %.2f, range: %.2f-%.2f\n",
			strategyNames[i],
			len(homogGraph.Edges),
			homogGraph.Statistics.AverageWeight,
			homogGraph.Statistics.MinWeight,
			homogGraph.Statistics.MaxWeight)
	}
	
	return nil
}

func getAggregationName(strategy materialization.AggregationStrategy) string {
	switch strategy {
	case materialization.Count:
		return "Count instances"
	case materialization.Sum:
		return "Sum weights"
	case materialization.Average:
		return "Average weights"
	case materialization.Maximum:
		return "Maximum weight"
	case materialization.Minimum:
		return "Minimum weight"
	default:
		return "Unknown"
	}
}

func TestMeetingBasedImplementation(t *testing.T) {
    // Create test graph where meeting-based differs from direct
    graph := &models.HeterogeneousGraph{
        Nodes: map[string]models.Node{
            "a1": {Type: "Author"},
            "a2": {Type: "Author"},
            "p1": {Type: "Paper"},
            "p2": {Type: "Paper"},
            "v1": {Type: "Venue"},
        },
        Edges: []models.Edge{
            {From: "a1", To: "p1", Type: "writes", Weight: 1.0},
            {From: "a2", To: "p2", Type: "writes", Weight: 1.0},
            {From: "p1", To: "v1", Type: "published_in", Weight: 1.0},
            {From: "p2", To: "v1", Type: "published_in", Weight: 1.0},
        },
    }
    graph.PopulateTypeMaps()
    
    metaPath := &models.MetaPath{
        ID:           "author_venue",
        NodeSequence: []string{"Author", "Paper", "Venue"},
        EdgeSequence: []string{"writes", "published_in"},
        Description:  "Authors to venues through papers",
    }
    
    // Test Direct Traversal
    configDirect := DefaultMaterializationConfig()
    configDirect.Aggregation.Interpretation = DirectTraversal
    engineDirect := NewMaterializationEngine(graph, metaPath, configDirect, nil)
    resultDirect, _ := engineDirect.Materialize()
    
    // Test Meeting-Based
    configMeeting := DefaultMaterializationConfig()
    configMeeting.Aggregation.Interpretation = MeetingBased
    engineMeeting := NewMaterializationEngine(graph, metaPath, configMeeting, nil)
    resultMeeting, _ := engineMeeting.Materialize()
    
    // For Author->Paper->Venue:
    // Direct should create: a1->v1, a2->v1
    // Meeting-based should create: Nothing! (Papers don't connect multiple authors or venues)
    
    if len(resultDirect.HomogeneousGraph.Edges) == len(resultMeeting.HomogeneousGraph.Edges) {
        t.Error("Meeting-based should produce different results than direct traversal")
    }
}
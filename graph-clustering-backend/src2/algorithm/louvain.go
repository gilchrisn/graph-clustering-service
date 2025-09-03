package algorithm

import (
	"context"
	"fmt"
	"time"
	"strconv"
	"sort"

	"github.com/rs/zerolog/log"
	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"

	"graph-viz-backend/algorithm/coordinates"
	"graph-viz-backend/models"
)

// LouvainAdapter adapts the new Louvain API to our algorithm interface
type LouvainAdapter struct{
	coordGenerator *coordinates.Generator
}

// NewLouvainAdapter creates a new Louvain algorithm adapter
func NewLouvainAdapter() Algorithm {
	return &LouvainAdapter{
		coordGenerator: coordinates.NewGenerator(),
	}
}

func (l *LouvainAdapter) Name() models.AlgorithmType {
	return models.AlgorithmLouvain
}

func (l *LouvainAdapter) ValidateParameters(params models.JobParameters) error {
	// Set defaults if not provided
	if params.MaxLevels == nil {
		defaultVal := 10
		params.MaxLevels = &defaultVal
	}
	
	if params.MaxIterations == nil {
		defaultVal := 100
		params.MaxIterations = &defaultVal
	}
	
	if params.MinModularityGain == nil {
		defaultVal := 1e-6
		params.MinModularityGain = &defaultVal
	}
	
	// Validate ranges
	if *params.MaxLevels < 1 || *params.MaxLevels > 50 {
		return fmt.Errorf("maxLevels must be between 1 and 50, got %d", *params.MaxLevels)
	}
	
	if *params.MaxIterations < 1 || *params.MaxIterations > 1000 {
		return fmt.Errorf("maxIterations must be between 1 and 1000, got %d", *params.MaxIterations)
	}
	
	return nil
}

func (l *LouvainAdapter) Run(ctx context.Context, dataset *models.Dataset, params models.JobParameters, progressCb ProgressCallback) (*AlgorithmResult, error) {
	startTime := time.Now()
	
	log.Info().
		Str("dataset_id", dataset.ID).
		Str("algorithm", "louvain").
		Msg("Starting Louvain clustering")
	
	progressCb(10, "Parsing input files for materialization...")
	
	// Step 1: Parse heterogeneous graph input
	graph, metaPath, err := materialization.ParseSCARInput(
		dataset.Files.GraphFile,
		dataset.Files.PropertiesFile,
		dataset.Files.PathFile,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to parse input files: %w", err)
	}
	
	log.Debug().
		Int("nodes", len(graph.Nodes)).
		Msg("Parsed heterogeneous graph")
	
	progressCb(30, "Running graph materialization...")
	
	// Step 2: Materialize heterogeneous graph to homogeneous
	materializationConfig := materialization.DefaultMaterializationConfig()
	materializationConfig.Aggregation.Strategy = materialization.Average
	materializationConfig.Aggregation.Symmetric = true
	
	engine := materialization.NewMaterializationEngine(graph, metaPath, materializationConfig, nil)
	materializationResult, err := engine.Materialize()
	if err != nil {
		return nil, fmt.Errorf("materialization failed: %w", err)
	}
	
	log.Debug().
		Int("materialized_nodes", len(materializationResult.HomogeneousGraph.Nodes)).
		Int("materialized_edges", len(materializationResult.HomogeneousGraph.Edges)).
		Msg("Graph materialization complete")
	
	progressCb(50, "Converting to Louvain graph format...")
	
	// Step 3: Convert to Louvain graph format
	louvainGraph, err := l.convertToLouvainGraph(materializationResult.HomogeneousGraph)
	if err != nil {
		return nil, fmt.Errorf("graph conversion failed: %w", err)
	}
	
	progressCb(70, "Running Louvain clustering...")
	
	// Step 4: Configure and run Louvain with graph storage enabled
	config := louvain.NewConfig()
	config.Set("algorithm.max_levels", *params.MaxLevels)
	config.Set("algorithm.max_iterations", *params.MaxIterations)
	config.Set("algorithm.min_modularity_gain", *params.MinModularityGain)
	config.Set("algorithm.random_seed", int64(42))
	config.Set("logging.level", "info")
	
	// Enable graph storage for coordinate generation
	config.Set("output.store_graphs_at_each_level", true)
	
	result, err := louvain.Run(louvainGraph, config, ctx)
	fmt.Printf("Louvain result: modularity=%.6f, levels=%d\n",
		result.Modularity, result.NumLevels)
	if err != nil {
		return nil, fmt.Errorf("Louvain clustering failed: %w", err)
	}
	
	processingTime := time.Since(startTime)
	
	log.Info().
		Str("dataset_id", dataset.ID).
		Float64("modularity", result.Modularity).
		Int("levels", result.NumLevels).
		Int64("processing_time_ms", processingTime.Milliseconds()).
		Msg("Louvain clustering complete")
	
	progressCb(90, "Building hierarchy with coordinates...")
	
	// Step 5: Convert result to our hierarchy format with coordinates
	hierarchy, err := l.convertToHierarchy(result, dataset.ID, "")
	if err != nil {
		return nil, fmt.Errorf("hierarchy conversion failed: %w", err)
	}
	
	progressCb(100, "Complete")
	
	return &AlgorithmResult{
		RawResult:        result,
		Hierarchy:        hierarchy,
		ProcessingTimeMS: processingTime.Milliseconds(),
		Modularity:       result.Modularity,
		NumLevels:        result.NumLevels,
		NumCommunities:   l.countFinalCommunities(result.FinalCommunities),
		Statistics:       l.buildStatistics(result),
	}, nil
}

func (l *LouvainAdapter) convertToLouvainGraph(hgraph *materialization.HomogeneousGraph) (*louvain.Graph, error) {
	if len(hgraph.Nodes) == 0 {
		return nil, fmt.Errorf("empty homogeneous graph")
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
	nodeMapping := make(map[string]int)
	for i, originalID := range nodeList {
		nodeMapping[originalID] = i
	}
	
	// Create Louvain graph
	graph := louvain.NewGraph(len(nodeList))
	
	// Add edges with deduplication
	processedEdges := make(map[string]bool)
	for edgeKey, weight := range hgraph.Edges {
		fromNormalized, fromExists := nodeMapping[edgeKey.From]
		toNormalized, toExists := nodeMapping[edgeKey.To]
		
		if !fromExists || !toExists {
			log.Warn().
				Str("from", edgeKey.From).
				Str("to", edgeKey.To).
				Msg("Edge references unknown nodes")
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
			if fromNormalized != toNormalized { // Skip self-loops
				err := graph.AddEdge(fromNormalized, toNormalized, weight)
				if err != nil {
					log.Warn().
						Str("from", edgeKey.From).
						Str("to", edgeKey.To).
						Err(err).
						Msg("Failed to add edge")
				} else {
					processedEdges[canonicalID] = true
				}
			}
		}
	}
	
	log.Debug().
		Int("nodes", graph.NumNodes).
		Float64("total_weight", graph.TotalWeight).
		Int("edges_processed", len(processedEdges)).
		Msg("Louvain graph created with deterministic ordering")
	
	return graph, nil
}

func (l *LouvainAdapter) convertToHierarchy(result *louvain.Result, datasetID, jobID string) (*models.Hierarchy, error) {
	hierarchy := &models.Hierarchy{
		DatasetID:   datasetID,
		JobID:       jobID,
		Algorithm:   models.AlgorithmLouvain,
		Levels:      make([]models.HierarchyLevel, len(result.Levels)),
		Coordinates: make(map[string]models.NodePosition),
	}
	
	// Convert each level
	for i, level := range result.Levels {
		hierarchyLevel := models.HierarchyLevel{
			Level:       i,
			Communities: make(map[string][]string),
			ParentMap:   make(map[string]string),
		}
		
		// Convert communities
		for communityID, nodes := range level.Communities {
			communityKey := fmt.Sprintf("c%d_l%d_%d", 0, i, communityID)
			nodeStrings := make([]string, len(nodes))
			for j, node := range nodes {
				nodeStrings[j] = fmt.Sprintf("%d", node)
			}
			hierarchyLevel.Communities[communityKey] = nodeStrings
		}
		
		// Build parent mappings from hierarchy tracking
		if len(level.CommunityToSuperNode) > 0 {
			for childCommunity, parentSupernode := range level.CommunityToSuperNode {
				childKey := fmt.Sprintf("c%d_l%d_%d", 0, i, childCommunity)
				parentKey := fmt.Sprintf("c%d_l%d_%d", 0, i+1, parentSupernode)
				hierarchyLevel.ParentMap[childKey] = parentKey
			}
		}
		
		hierarchy.Levels[i] = hierarchyLevel
	}
	
	// Set root node (highest level community)
	if len(result.Levels) > 0 {
		finalLevel := result.Levels[len(result.Levels)-1]
		if len(finalLevel.Communities) > 0 {
			// Find first community in final level
			for communityID := range finalLevel.Communities {
				hierarchy.RootNode = fmt.Sprintf("c%d_l%d_%d", 0, len(result.Levels)-1, communityID)
				break
			}
		}
	}
	
	// Phase 2: Generate coordinates for all levels
	log.Info().Msg("Generating coordinates using PageRank + MDS")
	
	coordinates, err := l.coordGenerator.GenerateFromLouvainResult(result)
	if err != nil {
		log.Error().Err(err).Msg("Failed to generate coordinates")
		// Don't fail the entire job if coordinate generation fails
		coordinates = make(map[string]models.NodePosition)
	} else {
		log.Info().
			Int("coordinates_count", len(coordinates)).
			Msg("Coordinates generated successfully")
	}
	
	hierarchy.Coordinates = coordinates
	
	return hierarchy, nil
}

func (l *LouvainAdapter) countFinalCommunities(finalCommunities map[int]int) int {
	communitySet := make(map[int]bool)
	for _, communityID := range finalCommunities {
		communitySet[communityID] = true
	}
	return len(communitySet)
}

func (l *LouvainAdapter) buildStatistics(result *louvain.Result) map[string]interface{} {
	stats := make(map[string]interface{})
	
	stats["total_moves"] = result.Statistics.TotalMoves
	stats["runtime_ms"] = result.Statistics.RuntimeMS
	stats["final_communities"] = l.countFinalCommunities(result.FinalCommunities)
	
	// Level-wise statistics
	levelStats := make([]map[string]interface{}, len(result.Levels))
	for i, level := range result.Levels {
		levelStats[i] = map[string]interface{}{
			"level":           level.Level,
			"modularity":      level.Modularity,
			"num_communities": level.NumCommunities,
			"num_moves":       level.NumMoves,
			"runtime_ms":      level.RuntimeMS,
		}
	}
	stats["levels"] = levelStats
	
	return stats
}
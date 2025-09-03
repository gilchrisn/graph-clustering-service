package algorithm

import (
	"context"
	"fmt"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"

	"graph-viz-backend/algorithm/coordinates"
	"graph-viz-backend/models"
)

// SCARAdapter adapts the new SCAR API to our algorithm interface
type SCARAdapter struct{
	coordGenerator *coordinates.Generator
}

// NewSCARAdapter creates a new SCAR algorithm adapter
func NewSCARAdapter() Algorithm {
	return &SCARAdapter{
		coordGenerator: coordinates.NewGenerator(),
	}
}

func (s *SCARAdapter) Name() models.AlgorithmType {
	return models.AlgorithmSCAR
}

func (s *SCARAdapter) ValidateParameters(params models.JobParameters) error {
	// Set defaults if not provided
	if params.K == nil {
		defaultVal := int64(64)
		params.K = &defaultVal
	}
	
	if params.NK == nil {
		defaultVal := int64(4)
		params.NK = &defaultVal
	}
	
	if params.Threshold == nil {
		defaultVal := 0.5
		params.Threshold = &defaultVal
	}
	
	// Validate ranges
	if *params.K < 1 || *params.K > 10000 {
		return fmt.Errorf("k must be between 1 and 10000, got %d", *params.K)
	}
	
	if *params.NK < 1 || *params.NK > 20 {
		return fmt.Errorf("nk must be between 1 and 20, got %d", *params.NK)
	}
	
	if *params.Threshold < 0 || *params.Threshold > 1 {
		return fmt.Errorf("threshold must be between 0 and 1, got %f", *params.Threshold)
	}


	// Validate reconstruction threshold
	if params.ReconstructionThreshold == nil {
		// Set default value
		defaultVal := 0.0
		params.ReconstructionThreshold = &defaultVal
	}
	
	// Validate reconstruction mode
	if params.ReconstructionMode != nil {
		validModes := map[string]bool{
			"inclusion_exclusion": true,
			"full":               true,
		}
		if !validModes[*params.ReconstructionMode] {
			return fmt.Errorf("invalid reconstructionMode: %s, must be 'inclusion_exclusion' or 'full'", *params.ReconstructionMode)
		}
	} else {
		// Set default value
		defaultVal := "inclusion_exclusion"
		params.ReconstructionMode = &defaultVal
	}
	
	// Validate edge weight normalization
	if params.EdgeWeightNormalization == nil {
		// Set default value
		defaultVal := true
		params.EdgeWeightNormalization = &defaultVal
	}
	
	return nil
}

func (s *SCARAdapter) Run(ctx context.Context, dataset *models.Dataset, params models.JobParameters, progressCb ProgressCallback) (*AlgorithmResult, error) {
	startTime := time.Now()
	
	log.Info().
		Str("dataset_id", dataset.ID).
		Str("algorithm", "scar").
		Int64("k", *params.K).
		Int64("nk", *params.NK).
		Float64("threshold", *params.Threshold).
		Msg("Starting SCAR clustering")
	
	progressCb(10, "Configuring SCAR parameters...")
	
	// Configure SCAR with graph storage enabled
	config := scar.NewConfig()
	
	// SCAR-specific parameters
	config.Set("scar.k", *params.K)
	config.Set("scar.nk", *params.NK)
	config.Set("scar.threshold", *params.Threshold)
	
	// Standard Louvain parameters (SCAR uses Louvain internally)
	config.Set("algorithm.max_levels", *params.MaxLevels)
	config.Set("algorithm.max_iterations", *params.MaxIterations)
	config.Set("algorithm.min_modularity_gain", *params.MinModularityGain)
	config.Set("algorithm.random_seed", int64(42))
	config.Set("logging.level", "info")
	
	// Enable graph storage for coordinate generation
	config.Set("output.store_graphs_at_each_level", true)
	
	progressCb(30, "Running SCAR sketch computation...")
	
	// Run SCAR directly with file paths
	result, err := scar.Run(
		dataset.Files.GraphFile,
		dataset.Files.PropertiesFile,
		dataset.Files.PathFile,
		config,
		ctx,
	)
	if err != nil {
		return nil, fmt.Errorf("SCAR clustering failed: %w", err)
	}
	
	processingTime := time.Since(startTime)
	
	log.Info().
		Str("dataset_id", dataset.ID).
		Float64("modularity", result.Modularity).
		Int("levels", result.NumLevels).
		Int("target_nodes", result.NodeMapping.NumTargetNodes).
		Int64("processing_time_ms", processingTime.Milliseconds()).
		Msg("SCAR clustering complete")
	
	progressCb(90, "Building hierarchy with coordinates...")
	
	// Convert result to our hierarchy format with coordinates
	hierarchy, err := s.convertToHierarchy(result, dataset.ID, "", &params)
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
		NumCommunities:   s.countFinalCommunities(result.FinalCommunities),
		Statistics:       s.buildStatistics(result),
	}, nil
}

func (s *SCARAdapter) convertToHierarchy(result *scar.Result, datasetID, jobID string, params *models.JobParameters) (*models.Hierarchy, error) {
	hierarchy := &models.Hierarchy{
		DatasetID:   datasetID,
		JobID:       jobID,
		Algorithm:   models.AlgorithmSCAR,
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
				// Map back from compressed to original node IDs if needed
				if result.NodeMapping != nil && node < len(result.NodeMapping.CompressedToOriginal) {
					nodeStrings[j] = fmt.Sprintf("%d", result.NodeMapping.CompressedToOriginal[node])
				} else {
					nodeStrings[j] = fmt.Sprintf("%d", node)
				}
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

	reconstructionThreshold := params.ReconstructionThreshold
	coordinates, err := s.coordGenerator.GenerateFromSCARResult(result, reconstructionThreshold)
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

func (s *SCARAdapter) countFinalCommunities(finalCommunities map[int]int) int {
	communitySet := make(map[int]bool)
	for _, communityID := range finalCommunities {
		communitySet[communityID] = true
	}
	return len(communitySet)
}

func (s *SCARAdapter) buildStatistics(result *scar.Result) map[string]interface{} {
	stats := make(map[string]interface{})
	
	stats["total_moves"] = result.Statistics.TotalMoves
	stats["runtime_ms"] = result.Statistics.RuntimeMS
	stats["final_communities"] = s.countFinalCommunities(result.FinalCommunities)
	
	// SCAR-specific statistics
	if result.NodeMapping != nil {
		stats["num_target_nodes"] = result.NodeMapping.NumTargetNodes
		stats["compression_ratio"] = float64(len(result.NodeMapping.CompressedToOriginal)) / float64(len(result.NodeMapping.OriginalToCompressed))
	}
	
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
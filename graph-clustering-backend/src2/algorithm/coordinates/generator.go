package coordinates

import (
	"fmt"

	"github.com/rs/zerolog/log"
	"gonum.org/v1/gonum/graph/simple"
	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"

	"graph-viz-backend/models"
)

// Generator orchestrates coordinate generation using PageRank and MDS
type Generator struct {
	pageRankCalc *PageRankCalculator
	mdsCalc      *MDSCalculator
	graphAdapter *GraphAdapter
}

// NewGenerator creates a new coordinate generator
func NewGenerator() *Generator {
	return &Generator{
		pageRankCalc: NewPageRankCalculator(),
		mdsCalc:      NewMDSCalculator(),
		graphAdapter: NewGraphAdapter(),
	}
}

// GenerateFromLouvainResult generates coordinates for all hierarchy levels from Louvain result
func (g *Generator) GenerateFromLouvainResult(result *louvain.Result) (map[string]models.NodePosition, error) {
	if result == nil {
		return nil, fmt.Errorf("louvain result is nil")
	}

	coordinates := make(map[string]models.NodePosition)

	log.Info().Int("num_levels", len(result.Levels)).Msg("Generating coordinates for all hierarchy levels")

	// Process each level
	for levelIdx, levelInfo := range result.Levels {
		if levelInfo.Graph == nil {
			log.Warn().Int("level", levelIdx).Msg("No graph stored for level, skipping coordinate generation")
			continue
		}

		log.Debug().
			Int("level", levelIdx).
			Int("communities", len(levelInfo.Communities)).
			Msg("Processing level")

		// Generate coordinates for this level
		levelCoords, err := g.generateForLouvainLevel(&levelInfo, levelIdx)
		if err != nil {
			log.Error().
				Int("level", levelIdx).
				Err(err).
				Msg("Failed to generate coordinates for level")
			continue
		}

		// Merge into result
		for nodeID, position := range levelCoords {
			coordinates[nodeID] = position
		}

		log.Debug().
			Int("level", levelIdx).
			Int("coordinates_generated", len(levelCoords)).
			Msg("Level coordinates generated")
	}

	log.Info().
		Int("total_coordinates", len(coordinates)).
		Msg("Coordinate generation complete")

	return coordinates, nil
}

// GenerateFromSCARResult generates coordinates for all hierarchy levels from SCAR result
func (g *Generator) GenerateFromSCARResult(result *scar.Result, reconstructionThreshold *float64) (map[string]models.NodePosition, error) {
	if result == nil {
		return nil, fmt.Errorf("SCAR result is nil")
	}

	coordinates := make(map[string]models.NodePosition)

	log.Info().Int("num_levels", len(result.Levels)).Msg("Generating coordinates for all hierarchy levels")

	// Process each level
	for levelIdx, levelInfo := range result.Levels {
		if levelInfo.SketchGraph == nil {
			log.Warn().Int("level", levelIdx).Msg("No sketch graph stored for level, skipping coordinate generation")
			continue
		}

		log.Debug().
			Int("level", levelIdx).
			Int("communities", len(levelInfo.Communities)).
			Msg("Processing level")

		// Generate coordinates for this level
		levelCoords, err := g.generateForSCARLevel(&levelInfo, levelIdx, result.NodeMapping, reconstructionThreshold)
		if err != nil {
			log.Error().
				Int("level", levelIdx).
				Float64("reconstruction_threshold", *reconstructionThreshold).
				Err(err).
				Msg("Failed to generate coordinates for level")
			continue
		}

		// Merge into result
		for nodeID, position := range levelCoords {
			coordinates[nodeID] = position
		}

		log.Debug().
			Int("level", levelIdx).
			Int("coordinates_generated", len(levelCoords)).
			Msg("Level coordinates generated")
	}

	log.Info().
		Int("total_coordinates", len(coordinates)).
		Msg("Coordinate generation complete")

	return coordinates, nil
}

// generateForLouvainLevel generates coordinates for a single Louvain level
func (g *Generator) generateForLouvainLevel(levelInfo *louvain.LevelInfo, levelIdx int) (map[string]models.NodePosition, error) {
	// Build node list from communities
	nodeList := g.graphAdapter.BuildNodeList(levelInfo.Communities)
	if len(nodeList) == 0 {
		return make(map[string]models.NodePosition), nil
	}

	// Convert to gonum graph
	gnumGraph, nodeMapping, err := g.graphAdapter.ConvertLouvainGraph(levelInfo.Graph, nodeList)
	if err != nil {
		return nil, fmt.Errorf("failed to convert Louvain graph: %w", err)
	}

	// Generate coordinates
	return g.generateCoordinatesForLevel(gnumGraph, nodeMapping, levelIdx)
}

// generateForSCARLevel generates coordinates for a single SCAR level
func (g *Generator) generateForSCARLevel(levelInfo *scar.LevelInfo, levelIdx int, nodeMapping *scar.NodeMapping, reconstructionThreshold *float64) (map[string]models.NodePosition, error) {
	// Build node list from communities
	nodeList := g.graphAdapter.BuildNodeList(levelInfo.Communities)
	if len(nodeList) == 0 {
		return make(map[string]models.NodePosition), nil
	}

	// Convert to gonum graph
	// gnumGraph, gnumNodeMapping, err := g.graphAdapter.ConvertSCARGraph(levelInfo.SketchGraph, nodeList)
	gnumGraph, gnumNodeMapping, err := g.graphAdapter.ConvertSCARGraphWithInclusionExclusion(levelInfo.SketchGraph, nodeList, *reconstructionThreshold)

	if err != nil {
		return nil, fmt.Errorf("failed to convert SCAR graph: %w", err)
	}

	// Generate coordinates
	return g.generateCoordinatesForLevel(gnumGraph, gnumNodeMapping, levelIdx)
}

// generateCoordinatesForLevel generates coordinates for a level using PageRank + MDS
func (g *Generator) generateCoordinatesForLevel(gnumGraph *simple.WeightedUndirectedGraph, nodeMapping map[int64]int, levelIdx int) (map[string]models.NodePosition, error) {
	if gnumGraph.Nodes().Len() == 0 {
		return make(map[string]models.NodePosition), nil
	}

	log.Debug().
		Int("level", levelIdx).
		Int("nodes", gnumGraph.Nodes().Len()).
		Msg("Computing PageRank and MDS")

	// Compute PageRank scores
	pageRankResult, err := g.pageRankCalc.Calculate(gnumGraph)
	if err != nil {
		return nil, fmt.Errorf("PageRank calculation failed: %w", err)
	}

	// Compute MDS coordinates
	mdsResult, err := g.mdsCalc.Calculate(gnumGraph, nodeMapping)
	if err != nil {
		return nil, fmt.Errorf("MDS calculation failed: %w", err)
	}

	// Combine PageRank and MDS results
	coordinates := make(map[string]models.NodePosition)

	for gnumNodeID, originalNodeID := range nodeMapping {
		// Get normalized position from MDS
		position := mdsResult.GetScaledPosition(gnumNodeID, -100, 100) // Scale to [-100, 100] range

		// Get radius from PageRank
		radius := pageRankResult.GetRadiusFromScore(gnumNodeID, 3.0, 20.0) // Radius between 3-20

		// Create community-style node ID for hierarchy levels
		var nodeKey string
		if levelIdx == 0 {
			// Leaf nodes use original ID
			nodeKey = fmt.Sprintf("%d", originalNodeID)
		} else {
			// Supernode format: c0_l{level}_{nodeID}
			nodeKey = fmt.Sprintf("c0_l%d_%d", levelIdx, originalNodeID)
		}

		coordinates[nodeKey] = models.NodePosition{
			X:      position.X,
			Y:      position.Y,
			Radius: radius,
		}
	}

	return coordinates, nil
}


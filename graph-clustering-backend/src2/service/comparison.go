package service

import (
	"fmt"
	"math"
	"sync"
	"time"
	"strings"
	"sort"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"

	"graph-viz-backend/models"
)

// ComparisonService handles experiment comparisons
type ComparisonService struct {
	comparisons       map[string]*models.Comparison
	clusteringService *ClusteringService
	mutex             sync.RWMutex
}

// NewComparisonService creates a new comparison service
func NewComparisonService(clusteringService *ClusteringService) *ComparisonService {
	return &ComparisonService{
		comparisons:       make(map[string]*models.Comparison),
		clusteringService: clusteringService,
	}
}

// Create starts a new comparison between two experiments
func (s *ComparisonService) Create(name string, expA, expB models.ExperimentRef, metrics []string, options models.ComparisonOptions) (*models.Comparison, error) {
	// Validate inputs
	if err := s.validateComparison(expA, expB, metrics); err != nil {
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	// Generate unique comparison ID
	comparisonID := uuid.New().String()

	log.Info().
		Str("comparison_id", comparisonID).
		Str("experiment_a", fmt.Sprintf("%s/%s", expA.DatasetID, expA.JobID)).
		Str("experiment_b", fmt.Sprintf("%s/%s", expB.DatasetID, expB.JobID)).
		Strs("metrics", metrics).
		Msg("Starting comparison")

	// Create comparison record
	comparison := &models.Comparison{
		ID:          comparisonID,
		Name:        name,
		ExperimentA: expA,
		ExperimentB: expB,
		Metrics:     metrics,
		Options:     options,
		Status:      models.ComparisonStatusRunning,
		CreatedAt:   time.Now(),
	}

	// Store comparison
	s.mutex.Lock()
	s.comparisons[comparisonID] = comparison
	s.mutex.Unlock()

	// Compute comparison synchronously (fast operation)
	go s.computeComparison(comparisonID)

	return comparison, nil
}

// Get retrieves a comparison by ID
func (s *ComparisonService) Get(comparisonID string) (*models.Comparison, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	comparison, exists := s.comparisons[comparisonID]
	if !exists {
		return nil, fmt.Errorf("comparison not found: %s", comparisonID)
	}

	return comparison, nil
}

// List returns all comparisons (optional for frontend)
func (s *ComparisonService) List() []*models.Comparison {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	comparisons := make([]*models.Comparison, 0, len(s.comparisons))
	for _, comparison := range s.comparisons {
		comparisons = append(comparisons, comparison)
	}

	return comparisons
}

// Delete removes a comparison
func (s *ComparisonService) Delete(comparisonID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if _, exists := s.comparisons[comparisonID]; !exists {
		return fmt.Errorf("comparison not found: %s", comparisonID)
	}

	delete(s.comparisons, comparisonID)

	log.Info().
		Str("comparison_id", comparisonID).
		Msg("Comparison deleted")

	return nil
}

// computeComparison performs the actual comparison computation
func (s *ComparisonService) computeComparison(comparisonID string) {
	s.mutex.RLock()
	comparison := s.comparisons[comparisonID]
	s.mutex.RUnlock()

	if comparison == nil {
		return
	}

	log.Debug().
		Str("comparison_id", comparisonID).
		Msg("Computing comparison metrics")

	// Get hierarchies for both experiments
	hierA, err := s.clusteringService.GetHierarchy(comparison.ExperimentA.DatasetID, comparison.ExperimentA.JobID)
	if err != nil {
		s.failComparison(comparisonID, fmt.Errorf("failed to get hierarchy A: %w", err))
		return
	}

	hierB, err := s.clusteringService.GetHierarchy(comparison.ExperimentB.DatasetID, comparison.ExperimentB.JobID)
	if err != nil {
		s.failComparison(comparisonID, fmt.Errorf("failed to get hierarchy B: %w", err))
		return
	}

	// Compute requested metrics
	result := &models.ComparisonResult{}

	for _, metric := range comparison.Metrics {
		switch metric {
		case "agds":
			if agds, err := s.computeAGDS(hierA, hierB); err == nil {
				result.AGDS = &agds
			} else {
				log.Warn().Str("comparison_id", comparisonID).Err(err).Msg("AGDS computation failed")
			}
		case "jaccard":
			if jaccard, err := s.computeJaccard(hierA, hierB); err == nil {
				result.Jaccard = &jaccard
			} else {
				log.Warn().Str("comparison_id", comparisonID).Err(err).Msg("Jaccard computation failed")
			}
		case "hmi":
			if hmi, err := s.computeHMI(hierA, hierB); err == nil {
				result.HMI = &hmi
			} else {
				log.Warn().Str("comparison_id", comparisonID).Err(err).Msg("HMI computation failed")
			}
		case "ari":
			if ari, err := s.computeARI(hierA, hierB); err == nil {
				result.ARI = &ari
			} else {
				log.Warn().Str("comparison_id", comparisonID).Err(err).Msg("ARI computation failed")
			}
		}
	}

	// Compute level-wise metrics if requested
	if comparison.Options.LevelWise {
		result.LevelWise = s.computeLevelWiseMetrics(hierA, hierB)
	}

	// Generate summary
	result.Summary = s.generateSummary(result)

	// Update comparison with results
	s.completeComparison(comparisonID, result)
}

// validateComparison validates the comparison request
func (s *ComparisonService) validateComparison(expA, expB models.ExperimentRef, metrics []string) error {
	// Validate experiments exist and are completed
	jobA, err := s.clusteringService.GetJobStatus(expA.JobID)
	if err != nil {
		return fmt.Errorf("experiment A not found: %w", err)
	}
	if jobA.Status != models.JobStatusCompleted {
		return fmt.Errorf("experiment A not completed, status: %s", jobA.Status)
	}
	if jobA.DatasetID != expA.DatasetID {
		return fmt.Errorf("experiment A dataset mismatch")
	}

	jobB, err := s.clusteringService.GetJobStatus(expB.JobID)
	if err != nil {
		return fmt.Errorf("experiment B not found: %w", err)
	}
	if jobB.Status != models.JobStatusCompleted {
		return fmt.Errorf("experiment B not completed, status: %s", jobB.Status)
	}
	if jobB.DatasetID != expB.DatasetID {
		return fmt.Errorf("experiment B dataset mismatch")
	}

	// Validate metrics
	validMetrics := map[string]bool{
		"agds":    true,
		"hmi":     true,
		"jaccard": true,
		"ari":     true,
	}

	for _, metric := range metrics {
		if !validMetrics[metric] {
			return fmt.Errorf("invalid metric: %s", metric)
		}
	}

	if len(metrics) == 0 {
		return fmt.Errorf("at least one metric must be specified")
	}

	return nil
}

// Metric computation methods
func (s *ComparisonService) computeAGDS(hierA, hierB *models.Hierarchy) (float64, error) {
	// AGDS: Average Geometric Distance Similarity between node positions
	if len(hierA.Coordinates) == 0 || len(hierB.Coordinates) == 0 {
		return 0, fmt.Errorf("missing coordinates for AGDS computation")
	}

	var totalDistance float64
	var count int

	// Compare positions of common nodes
	for nodeID, posA := range hierA.Coordinates {
		if posB, exists := hierB.Coordinates[nodeID]; exists {
			distance := math.Sqrt(math.Pow(posA.X-posB.X, 2) + math.Pow(posA.Y-posB.Y, 2))
			totalDistance += distance
			count++
		}
	}

	if count == 0 {
		return 0, fmt.Errorf("no common nodes for AGDS computation")
	}

	avgDistance := totalDistance / float64(count)
	// Convert to similarity (inverse relationship, normalized to 0-1)
	similarity := 1.0 / (1.0 + avgDistance/100.0) // Assuming coordinate range is ~200
	return similarity, nil
}

func (s *ComparisonService) computeJaccard(hierA, hierB *models.Hierarchy) (float64, error) {
	// Jaccard: Community overlap similarity
	if len(hierA.Levels) == 0 || len(hierB.Levels) == 0 {
		return 0, fmt.Errorf("empty hierarchies for Jaccard computation")
	}

	// Compare communities at the finest level (level 0)
	communitiesA := hierA.Levels[0].Communities
	communitiesB := hierB.Levels[0].Communities

	// Build node-to-community mappings
	nodeCommA := make(map[string]string)
	nodeCommB := make(map[string]string)

	for commID, nodes := range communitiesA {
		for _, nodeID := range nodes {
			nodeCommA[nodeID] = commID
		}
	}

	for commID, nodes := range communitiesB {
		for _, nodeID := range nodes {
			nodeCommB[nodeID] = commID
		}
	}

	// Get all unique nodes
	allNodes := make(map[string]bool)
	for nodeID := range nodeCommA {
		allNodes[nodeID] = true
	}
	for nodeID := range nodeCommB {
		allNodes[nodeID] = true
	}

	// Count agreements and total pairs
	var agreements, total int

	nodes := make([]string, 0, len(allNodes))
	for nodeID := range allNodes {
		nodes = append(nodes, nodeID)
	}

	// Compare all pairs of nodes
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			nodeI, nodeJ := nodes[i], nodes[j]

			// Check if both nodes exist in both clusterings
			commAI, existsAI := nodeCommA[nodeI]
			commAJ, existsAJ := nodeCommA[nodeJ]
			commBI, existsBI := nodeCommB[nodeI]
			commBJ, existsBJ := nodeCommB[nodeJ]

			if existsAI && existsAJ && existsBI && existsBJ {
				total++
				// Agreement: both in same community in A and same in B, or both in different communities in A and different in B
				sameInA := commAI == commAJ
				sameInB := commBI == commBJ
				if sameInA == sameInB {
					agreements++
				}
			}
		}
	}

	if total == 0 {
		return 0, fmt.Errorf("no comparable node pairs for Jaccard computation")
	}

	return float64(agreements) / float64(total), nil
}


func (s *ComparisonService) computeARI(hierA, hierB *models.Hierarchy) (float64, error) {
	// ARI: Adjusted Rand Index (simplified implementation)
	// Similar to Jaccard but with different normalization
	
	jaccard, err := s.computeJaccard(hierA, hierB)
	if err != nil {
		return 0, fmt.Errorf("ARI computation failed: %w", err)
	}

	// Convert Jaccard to ARI-like metric (simplified)
	// ARI = 2 * Jaccard - 1 (maps [0,1] to [-1,1])
	ari := 2*jaccard - 1
	if ari < 0 {
		ari = 0 // Ensure non-negative
	}

	return ari, nil
}

func (s *ComparisonService) computeLevelWiseMetrics(hierA, hierB *models.Hierarchy) []models.ComparisonLevelResult {
	var levelResults []models.ComparisonLevelResult

	maxLevels := len(hierA.Levels)
	if len(hierB.Levels) < maxLevels {
		maxLevels = len(hierB.Levels)
	}

	for level := 0; level < maxLevels; level++ {
		tempHierA := &models.Hierarchy{Levels: []models.HierarchyLevel{hierA.Levels[level]}}
		tempHierB := &models.Hierarchy{Levels: []models.HierarchyLevel{hierB.Levels[level]}}

		result := models.ComparisonLevelResult{Level: level}

		if jaccard, err := s.computeJaccard(tempHierA, tempHierB); err == nil {
			result.Jaccard = jaccard
		}

		if ari, err := s.computeARI(tempHierA, tempHierB); err == nil {
			result.ARI = ari
		}

		levelResults = append(levelResults, result)
	}

	return levelResults
}

func (s *ComparisonService) generateSummary(result *models.ComparisonResult) models.ComparisonSummary {
	summary := models.ComparisonSummary{
		KeyDifferences: []string{},
	}

	// Determine overall similarity based on available metrics
	var avgSimilarity float64
	var metricCount int

	if result.AGDS != nil {
		avgSimilarity += *result.AGDS
		metricCount++
	}
	if result.Jaccard != nil {
		avgSimilarity += *result.Jaccard
		metricCount++
	}
	if result.HMI != nil {
		avgSimilarity += *result.HMI
		metricCount++
	}
	if result.ARI != nil {
		avgSimilarity += *result.ARI
		metricCount++
	}

	if metricCount > 0 {
		avgSimilarity /= float64(metricCount)
	}

	// Categorize similarity
	if avgSimilarity >= 0.8 {
		summary.OverallSimilarity = "High"
		summary.Recommendation = "Experiments show very similar clustering patterns. Consider using either approach."
	} else if avgSimilarity >= 0.5 {
		summary.OverallSimilarity = "Medium"
		summary.Recommendation = "Experiments show moderate similarity. Analyze specific differences to choose best approach."
	} else {
		summary.OverallSimilarity = "Low"
		summary.Recommendation = "Experiments show significant differences. Review parameters and data to understand variations."
	}

	// Add specific insights
	if result.AGDS != nil && *result.AGDS < 0.5 {
		summary.KeyDifferences = append(summary.KeyDifferences, "Node positions differ significantly between experiments")
	}
	if result.Jaccard != nil && *result.Jaccard < 0.5 {
		summary.KeyDifferences = append(summary.KeyDifferences, "Community structures show substantial differences")
	}

	return summary
}

func (s *ComparisonService) completeComparison(comparisonID string, result *models.ComparisonResult) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	comparison := s.comparisons[comparisonID]
	if comparison == nil {
		return
	}

	comparison.Status = models.ComparisonStatusCompleted
	comparison.Result = result
	now := time.Now()
	comparison.CompletedAt = &now

	log.Info().
		Str("comparison_id", comparisonID).
		Str("overall_similarity", result.Summary.OverallSimilarity).
		Msg("Comparison completed successfully")
}

func (s *ComparisonService) failComparison(comparisonID string, err error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	comparison := s.comparisons[comparisonID]
	if comparison == nil {
		return
	}

	comparison.Status = models.ComparisonStatusFailed
	comparison.Error = err.Error()
	now := time.Now()
	comparison.CompletedAt = &now

	log.Error().
		Str("comparison_id", comparisonID).
		Err(err).
		Msg("Comparison failed")
}

// CreateMultiComparison creates a comparison between multiple experiments and a user-specified baseline
func (s *ComparisonService) CreateMultiComparison(req models.MultiComparisonRequest) (*models.Comparison, error) {
	// Validate request
	if len(req.SelectedExperiments) < 2 {
		return nil, fmt.Errorf("at least 2 experiments required for multi-comparison")
	}
	
	if len(req.SelectedExperiments) > 10 {
		return nil, fmt.Errorf("maximum 10 experiments allowed")
	}
	
	// Validate all experiments exist and are completed
	for _, exp := range req.SelectedExperiments {
		if err := s.validateExperiment(exp); err != nil {
			return nil, fmt.Errorf("invalid experiment %s/%s: %w", exp.DatasetID, exp.JobID, err)
		}
	}

	// Validate baseline experiment specifically
	if err := s.validateExperiment(req.BaselineExperiment); err != nil {
		return nil, fmt.Errorf("invalid baseline experiment %s/%s: %w", req.BaselineExperiment.DatasetID, req.BaselineExperiment.JobID, err)
	}
	
	// Get dataset ID from baseline experiment
	datasetID := req.BaselineExperiment.DatasetID
	
	// Ensure all experiments belong to same dataset
	for _, exp := range req.SelectedExperiments {
		if exp.DatasetID != datasetID {
			return nil, fmt.Errorf("all experiments must belong to the same dataset")
		}
	}

	// Use the explicitly provided baseline (no more auto-creation)
	baselineJobID := req.BaselineExperiment.JobID
	
	// Generate comparison ID
	comparisonID := uuid.New().String()
	
	log.Info().
		Str("comparison_id", comparisonID).
		Int("experiment_count", len(req.SelectedExperiments)).
		Str("baseline_job", baselineJobID).
		Strs("metrics", req.Metrics).
		Msg("Starting multi-experiment comparison with explicit baseline")
	
	// Create comparison record (store as regular Comparison with special handling)
	comparison := &models.Comparison{
		ID:        comparisonID,
		Name:      req.Name,
		// Store baseline as experiment A for compatibility
		ExperimentA: req.BaselineExperiment,
		ExperimentB: req.SelectedExperiments[0], // First non-baseline experiment
		Metrics:     req.Metrics,
		Options:     req.Options,
		Status:      models.ComparisonStatusRunning,
		CreatedAt:   time.Now(),
	}
	
	// Store comparison with multi-experiment metadata
	s.mutex.Lock()
	s.comparisons[comparisonID] = comparison
	s.storeMultiComparisonMetadata(comparisonID, req.SelectedExperiments, baselineJobID)
	s.mutex.Unlock()
	
	// Start processing in background
	go s.processMultiComparison(comparisonID, req.SelectedExperiments, baselineJobID)
	
	return comparison, nil
}

// processMultiComparison - simplified without auto-baseline creation
func (s *ComparisonService) processMultiComparison(comparisonID string, experiments []models.ExperimentRef, baselineJobID string) {
	log.Debug().
		Str("comparison_id", comparisonID).
		Int("experiment_count", len(experiments)).
		Str("baseline_job_id", baselineJobID).
		Msg("Processing multi-experiment comparison")
	
	// Get baseline hierarchy (no longer auto-creating, just using provided one)
	var baselineDatasetID string
	for _, exp := range experiments {
		if exp.JobID == baselineJobID {
			baselineDatasetID = exp.DatasetID
			break
		}
	}
	
	baselineHier, err := s.getHierarchyForExperiment(models.ExperimentRef{
		DatasetID: baselineDatasetID,
		JobID:     baselineJobID,
	})
	if err != nil {
		s.failComparison(comparisonID, fmt.Errorf("failed to get baseline hierarchy: %w", err))
		return
	}
	
	// Get baseline configuration for transparency
	baselineConfig, err := s.getBaselineConfiguration(baselineJobID)
	if err != nil {
		log.Warn().
			Str("comparison_id", comparisonID).
			Str("baseline_job_id", baselineJobID).
			Err(err).
			Msg("Could not get baseline configuration, using generic info")
		// Use generic baseline info if we can't get specific details
		baselineConfig = models.BaselineInfo{
			Algorithm:   "unknown",
			Parameters:  map[string]interface{}{},
			Description: fmt.Sprintf("User-selected baseline (Job ID: %s)", baselineJobID),
			IsStandard:  false,
		}
	}
	
	// Process each experiment vs baseline
	var experimentResults []models.ExperimentResult
	
	for _, exp := range experiments {
		// Skip the baseline experiment itself
		if exp.JobID == baselineJobID {
			continue
		}
		
		expHier, err := s.getHierarchyForExperiment(exp)
		if err != nil {
			log.Error().
				Str("comparison_id", comparisonID).
				Str("job_id", exp.JobID).
				Err(err).
				Msg("Failed to get experiment hierarchy, skipping")
			continue
		}
		
		// Generate experiment label
		label := s.generateExperimentLabel(exp)
		
		// Compute metrics for this experiment vs baseline
		metrics, err := s.computeMetricsForExperiment(expHier, baselineHier)
		if err != nil {
			log.Error().
				Str("comparison_id", comparisonID).
				Str("job_id", exp.JobID).
				Err(err).
				Msg("Failed to compute metrics, skipping")
			continue
		}
		
		experimentResults = append(experimentResults, models.ExperimentResult{
			JobID:   exp.JobID,
			Label:   label,
			Metrics: metrics,
		})
	}
	
	// Create multi-comparison result
	result := &models.MultiComparisonResult{
		BaselineJobID:  baselineJobID,
		BaselineConfig: baselineConfig,
		Experiments:    experimentResults,
	}
	
	// Store result
	s.completeMultiComparison(comparisonID, result)
}


// Get baseline configuration from job details
func (s *ComparisonService) getBaselineConfiguration(jobID string) (models.BaselineInfo, error) {
	job, err := s.clusteringService.GetJobStatus(jobID)
	if err != nil {
		return models.BaselineInfo{}, fmt.Errorf("failed to get job details: %w", err)
	}
	
	// Build parameter map
	paramMap := make(map[string]interface{})
	
	switch job.Algorithm {
	case models.AlgorithmLouvain:
		if job.Parameters.MaxLevels != nil {
			paramMap["maxLevels"] = *job.Parameters.MaxLevels
		}
		if job.Parameters.MaxIterations != nil {
			paramMap["maxIterations"] = *job.Parameters.MaxIterations
		}
		if job.Parameters.MinModularityGain != nil {
			paramMap["minModularityGain"] = *job.Parameters.MinModularityGain
		}
		
	case models.AlgorithmSCAR:
		if job.Parameters.K != nil {
			paramMap["k"] = *job.Parameters.K
		}
		if job.Parameters.NK != nil {
			paramMap["nk"] = *job.Parameters.NK
		}
		if job.Parameters.Threshold != nil {
			paramMap["threshold"] = *job.Parameters.Threshold
		}
		if job.Parameters.ReconstructionThreshold != nil {
			paramMap["reconstructionThreshold"] = *job.Parameters.ReconstructionThreshold
		}
		if job.Parameters.ReconstructionMode != nil {
			paramMap["reconstructionMode"] = *job.Parameters.ReconstructionMode
		}
		if job.Parameters.EdgeWeightNormalization != nil {
			paramMap["edgeWeightNormalization"] = *job.Parameters.EdgeWeightNormalization
		}
	}
	
	// Check if this is a "standard" baseline (standard Louvain parameters)
	isStandard := job.Algorithm == models.AlgorithmLouvain &&
		job.Parameters.MaxLevels != nil && *job.Parameters.MaxLevels == 10 &&
		job.Parameters.MaxIterations != nil && *job.Parameters.MaxIterations == 100 &&
		job.Parameters.MinModularityGain != nil && *job.Parameters.MinModularityGain == 1e-6
	
	var description string
	if isStandard {
		description = "Standard Louvain Baseline (industry default parameters)"
	} else {
		description = fmt.Sprintf("User-selected %s baseline", strings.Title(string(job.Algorithm)))
	}
	
	return models.BaselineInfo{
		Algorithm:   string(job.Algorithm),
		Parameters:  paramMap,
		Description: description,
		IsStandard:  isStandard,
	}, nil
}

// computeCustomLeafMetric computes visualization metric for entire leaf level
func (s *ComparisonService) computeCustomLeafMetric(hierA, hierB *models.Hierarchy) (float64, error) {
	// Get all leaf nodes from both hierarchies
	nodesA := s.getAllLeafNodes(hierA)
	nodesB := s.getAllLeafNodes(hierB)
	fmt.Printf("Leaf nodes A: %d, Leaf nodes B: %d\n", len(nodesA), len(nodesB))
	
	return s.computeVisualizationMetric(hierA, hierB, nodesA, nodesB)
}

// computeCustomDisplayedMetric computes visualization metric for largest community comparison
func (s *ComparisonService) computeCustomDisplayedMetric(hierA, hierB *models.Hierarchy) (float64, error) {
	// Get largest community nodes from both hierarchies
	nodesA := s.getLargestCommunityNodes(hierA)
	nodesB := s.getLargestCommunityNodes(hierB)
	
	return s.computeVisualizationMetric(hierA, hierB, nodesA, nodesB)
}

// computeVisualizationMetric computes the composite visualization metric
func (s *ComparisonService) computeVisualizationMetric(hierA, hierB *models.Hierarchy, nodesA, nodesB []string) (float64, error) {
	if len(nodesA) == 0 || len(nodesB) == 0 {
		return 0, fmt.Errorf("empty node sets for visualization metric")
	}
	
	// Calculate Jaccard similarity of node sets
	jaccard := s.calculateNodeSetJaccard(nodesA, nodesB)
	
	// Find matching nodes
	matchingNodes := s.findMatchingNodes(nodesA, nodesB)
	if len(matchingNodes) == 0 {
		return jaccard, nil // Return just Jaccard if no matching nodes
	}

	// Calculate cosine similarities
	var coordsA, coordsB []models.NodePosition
	for _, nodeID := range matchingNodes {
		posA, existsA := hierA.Coordinates[nodeID]
		posB, existsB := hierB.Coordinates[nodeID]

		if !existsA || !existsB {
			continue
		}
		coordsA = append(coordsA, posA)
		coordsB = append(coordsB, posB)
	}

	if len(coordsA) == 0 {
		return jaccard, nil
	}

	// Cosine similarity of flattened coordinates
	cosSim, err := cosineSimilarity(coordsA, coordsB)
	if err != nil {
		return 0, err
	}

	// Composite score: jaccard × cosine similarity
	compositeScore := jaccard * cosSim
	fmt.Printf("Debug: Matching=%d, Jaccard=%.4f, Cosine=%.4f, Composite=%.4f\n",
		len(matchingNodes), jaccard, cosSim, compositeScore)
	return compositeScore, nil
}

// cosineSimilarity computes cosine similarity between two lists of coordinates
func cosineSimilarity(coordsA, coordsB []models.NodePosition) (float64, error) {
	if len(coordsA) != len(coordsB) {
		return 0, fmt.Errorf("coordinate list length mismatch: %d vs %d", len(coordsA), len(coordsB))
	}

	var dot, normA, normB float64
	for i := 0; i < len(coordsA); i++ {
		a := coordsA[i]
		b := coordsB[i]

		// Flatten as [x,y]
		dot += a.X*b.X + a.Y*b.Y
		normA += a.X*a.X + a.Y*a.Y
		normB += b.X*b.X + b.Y*b.Y
	}

	if normA == 0 || normB == 0 {
		return 0, fmt.Errorf("zero-length vector (one shape has no magnitude)")
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB)), nil
}


// getLargestCommunityNodes returns nodes from the largest community in leaf level
func (s *ComparisonService) getLargestCommunityNodes(hier *models.Hierarchy) []string {
	if len(hier.Levels) == 0 {
		return []string{}
	}
	
	leafLevel := hier.Levels[0] // Level 0 is leaf level
	var largestCommunity []string
	
	// Find community with most nodes
	for _, nodes := range leafLevel.Communities {
		if len(nodes) > len(largestCommunity) {
			largestCommunity = make([]string, len(nodes))
			copy(largestCommunity, nodes)
		}
	}
	
	return largestCommunity
}

// getAllLeafNodes returns all nodes from leaf level
func (s *ComparisonService) getAllLeafNodes(hier *models.Hierarchy) []string {
	if len(hier.Levels) == 0 {
		return []string{}
	}
	
	leafLevel := hier.Levels[0]
	var allNodes []string
	
	for _, nodes := range leafLevel.Communities {
		allNodes = append(allNodes, nodes...)
	}
	
	return allNodes
}

// calculateNodeSetJaccard calculates Jaccard similarity between two node sets
func (s *ComparisonService) calculateNodeSetJaccard(nodesA, nodesB []string) float64 {
	if len(nodesA) == 0 && len(nodesB) == 0 {
		return 1.0 // Both empty sets are identical
	}
	
	if len(nodesA) == 0 || len(nodesB) == 0 {
		return 0.0 // One empty, one non-empty
	}
	
	// Convert to sets for efficient operations
	setA := make(map[string]bool)
	setB := make(map[string]bool)
	
	for _, node := range nodesA {
		setA[node] = true
	}
	
	for _, node := range nodesB {
		setB[node] = true
	}
	
	// Calculate intersection and union
	intersection := 0
	union := make(map[string]bool)
	
	// Add all nodes from A to union
	for node := range setA {
		union[node] = true
		if setB[node] {
			intersection++
		}
	}
	
	// Add all nodes from B to union
	for node := range setB {
		union[node] = true
	}
	
	if len(union) == 0 {
		return 1.0 // Shouldn't happen, but handle edge case
	}
	
	return float64(intersection) / float64(len(union))
}

// findMatchingNodes returns nodes that exist in both node sets
func (s *ComparisonService) findMatchingNodes(nodesA, nodesB []string) []string {
	setB := make(map[string]bool)
	for _, node := range nodesB {
		setB[node] = true
	}
	
	var matching []string
	for _, node := range nodesA {
		if setB[node] {
			matching = append(matching, node)
		}
	}
	
	return matching
}

// Helper methods for multi-comparison processing

func (s *ComparisonService) validateExperiment(exp models.ExperimentRef) error {
	job, err := s.clusteringService.GetJobStatus(exp.JobID)
	if err != nil {
		return fmt.Errorf("experiment not found: %w", err)
	}
	
	if job.DatasetID != exp.DatasetID {
		return fmt.Errorf("experiment dataset mismatch")
	}
	
	if job.Status != models.JobStatusCompleted {
		return fmt.Errorf("experiment not completed, status: %s", job.Status)
	}
	
	return nil
}

func (s *ComparisonService) getHierarchyForExperiment(exp models.ExperimentRef) (*models.Hierarchy, error) {
	return s.clusteringService.GetHierarchy(exp.DatasetID, exp.JobID)
}

func (s *ComparisonService) generateExperimentLabel(exp models.ExperimentRef) string {
	// Get job details to create descriptive label
	job, err := s.clusteringService.GetJobStatus(exp.JobID)
	if err != nil {
		return fmt.Sprintf("Job %s", exp.JobID[:8]) // Fallback to short job ID
	}
	
	switch job.Algorithm {
	case models.AlgorithmLouvain:
		return "Louvain"
	case models.AlgorithmSCAR:
		if job.Parameters.K != nil {
			return fmt.Sprintf("SCAR k=%d", *job.Parameters.K)
		}
		return "SCAR"
	default:
		return string(job.Algorithm)
	}
}

func (s *ComparisonService) computeMetricsForExperiment(expHier, baselineHier *models.Hierarchy) (map[string]float64, error) {
	metrics := make(map[string]float64)
	
	fmt.Printf("\n\n\n")
	fmt.Printf("Computing Metrics for experiment algorithm: %s vs baseline algorithm: %s\n", expHier.Algorithm, baselineHier.Algorithm)
	// Compute HMI (reuse existing method)
	if hmi, err := s.computeHMI(expHier, baselineHier); err == nil {
		metrics["hmi"] = hmi
	}
	
	fmt.Printf("Debug: Computing custom metrics for experiment hierarchy id=%p vs baseline id=%p\n", expHier, baselineHier)
	// Compute custom leaf metric
	if leafMetric, err := s.computeCustomLeafMetric(expHier, baselineHier); err == nil {
		metrics["custom_leaf_metric"] = leafMetric
	}
	
	fmt.Printf("Debug: Computing custom displayed metric for experiment hierarchy id=%p vs baseline id=%p\n", expHier, baselineHier)
	// Compute custom displayed metric
	if displayedMetric, err := s.computeCustomDisplayedMetric(expHier, baselineHier); err == nil {
		metrics["custom_displayed_metric"] = displayedMetric
	}
	
	return metrics, nil
}


// Multi-comparison metadata storage
type MultiComparisonMetadata struct {
	Experiments   []models.ExperimentRef `json:"experiments"`
	BaselineJobID string                 `json:"baselineJobId"`
	CreatedAt     time.Time              `json:"createdAt"`
}

var multiComparisonMetadata = make(map[string]*MultiComparisonMetadata)
var multiMetadataMutex sync.RWMutex

func (s *ComparisonService) storeMultiComparisonMetadata(comparisonID string, experiments []models.ExperimentRef, baselineJobID string) {
	multiMetadataMutex.Lock()
	defer multiMetadataMutex.Unlock()
	
	multiComparisonMetadata[comparisonID] = &MultiComparisonMetadata{
		Experiments:   experiments,
		BaselineJobID: baselineJobID,
		CreatedAt:     time.Now(),
	}
	
	log.Debug().
		Str("comparison_id", comparisonID).
		Int("experiment_count", len(experiments)).
		Str("baseline_job_id", baselineJobID).
		Msg("Multi-comparison metadata stored")
}

var multiComparisonResults = make(map[string]*models.MultiComparisonResult)
var multiComparisonMutex sync.RWMutex

func (s *ComparisonService) completeMultiComparison(comparisonID string, result *models.MultiComparisonResult) {
    s.mutex.Lock()
    comparison := s.comparisons[comparisonID]
    s.mutex.Unlock()
    
    if comparison == nil {
        return
    }

    // Store the multi-comparison result separately (not in legacy format)
    multiComparisonMutex.Lock()
    multiComparisonResults[comparisonID] = result
    multiComparisonMutex.Unlock()

    // Update comparison status only
    s.mutex.Lock()
    comparison.Status = models.ComparisonStatusCompleted
    now := time.Now()
    comparison.CompletedAt = &now
    s.mutex.Unlock()

    log.Info().
        Str("comparison_id", comparisonID).
        Int("experiment_count", len(result.Experiments)).
        Msg("Multi-comparison completed successfully")
}

func (s *ComparisonService) GetMultiComparisonResult(comparisonID string) (*models.MultiComparisonResult, error) {
    multiComparisonMutex.RLock()
    defer multiComparisonMutex.RUnlock()

    result, exists := multiComparisonResults[comparisonID]
    if !exists {
        return nil, fmt.Errorf("multi-comparison result not found: %s", comparisonID)
    }

    return result, nil
}

func (s *ComparisonService) IsMultiComparison(comparisonID string) bool {
	multiMetadataMutex.RLock()
	defer multiMetadataMutex.RUnlock()
	
	_, exists := multiComparisonMetadata[comparisonID]
	return exists
}



// HMI IMPLEMENTATION - MOVE LATER	

// computeHMI computes Hierarchical Mutual Information between two hierarchies
func (s *ComparisonService) computeHMI(hierA, hierB *models.Hierarchy) (float64, error) {
	if len(hierA.Levels) == 0 || len(hierB.Levels) == 0 {
		return 0, fmt.Errorf("empty hierarchies for HMI computation")
	}

	// Find common leaf nodes
	commonNodes := s.findCommonLeafNodes(hierA, hierB)
	if len(commonNodes) < 5 { // Need at least 5 common nodes
		return 0, fmt.Errorf("insufficient common nodes: %d", len(commonNodes))
	}

	// Compute weighted mutual information across levels
	totalWeightedMI := 0.0
	totalWeight := 0.0

	maxLevels := len(hierA.Levels)
	if len(hierB.Levels) < maxLevels {
		maxLevels = len(hierB.Levels)
	}

	// Compare each level with inverse weighting (finer levels get more weight)
	for level := 0; level < maxLevels; level++ {
		levelMI := s.computeLevelMutualInfo(&hierA.Levels[level], &hierB.Levels[level], commonNodes)
		weight := 1.0 / float64(level+1) // Give more weight to finer-grained levels
		
		totalWeightedMI += weight * levelMI
		totalWeight += weight
	}

	if totalWeight == 0 {
		return 0, fmt.Errorf("no comparable levels")
	}

	// Compute hierarchical entropies for normalization
	entropyA := s.computeHierarchicalEntropy(hierA, commonNodes)
	entropyB := s.computeHierarchicalEntropy(hierB, commonNodes)

	if entropyA == 0 && entropyB == 0 {
		return 1.0, nil // Both hierarchies are identical/trivial
	}
	if entropyA == 0 || entropyB == 0 {
		return 0.0, nil // One hierarchy is trivial
	}

	// Normalize using geometric mean (like NMI)
	avgWeightedMI := totalWeightedMI / totalWeight
	normalizedHMI := avgWeightedMI / math.Sqrt(entropyA*entropyB)

	// Clamp to [0, 1] range
	if normalizedHMI > 1.0 {
		normalizedHMI = 1.0
	}
	if normalizedHMI < 0.0 {
		normalizedHMI = 0.0
	}

	return normalizedHMI, nil
}

// Helper methods to add to the ComparisonService:

// findCommonLeafNodes extracts nodes present in both hierarchies
func (s *ComparisonService) findCommonLeafNodes(hierA, hierB *models.Hierarchy) []string {
	// Get all leaf nodes from both hierarchies (level 0)
	nodesA := make(map[string]bool)
	nodesB := make(map[string]bool)

	if len(hierA.Levels) > 0 {
		for _, nodes := range hierA.Levels[0].Communities {
			for _, nodeID := range nodes {
				nodesA[nodeID] = true
			}
		}
	}

	if len(hierB.Levels) > 0 {
		for _, nodes := range hierB.Levels[0].Communities {
			for _, nodeID := range nodes {
				nodesB[nodeID] = true
			}
		}
	}

	// Find intersection
	var commonNodes []string
	for nodeID := range nodesA {
		if nodesB[nodeID] {
			commonNodes = append(commonNodes, nodeID)
		}
	}

	return commonNodes
}

// computeLevelMutualInfo computes mutual information between partitions at a specific level
func (s *ComparisonService) computeLevelMutualInfo(levelA, levelB *models.HierarchyLevel, commonNodes []string) float64 {
	if len(levelA.Communities) <= 1 || len(levelB.Communities) <= 1 {
		return 0.0 // No information in single partition
	}

	// Build node-to-partition mappings
	partMapA := s.buildNodeToPartitionMap(levelA)
	partMapB := s.buildNodeToPartitionMap(levelB)

	// Build contingency table
	contingency, rowTotals, colTotals, total := s.buildContingencyTable(partMapA, partMapB, commonNodes)

	if total == 0 {
		return 0.0
	}

	// Compute mutual information
	return s.mutualInformation(contingency, rowTotals, colTotals, total)
}

// buildNodeToPartitionMap creates mapping from node to partition index
func (s *ComparisonService) buildNodeToPartitionMap(level *models.HierarchyLevel) map[string]int {
	nodeToPartition := make(map[string]int)
	partitionIdx := 0

	// Sort community keys for deterministic results
	var communityKeys []string
	for key := range level.Communities {
		communityKeys = append(communityKeys, key)
	}
	sort.Strings(communityKeys)

	for _, communityKey := range communityKeys {
		nodes := level.Communities[communityKey]
		for _, nodeID := range nodes {
			nodeToPartition[nodeID] = partitionIdx
		}
		partitionIdx++
	}

	return nodeToPartition
}

// buildContingencyTable creates contingency table for MI calculation
func (s *ComparisonService) buildContingencyTable(partMapA, partMapB map[string]int, commonNodes []string) ([][]int, []int, []int, int) {
	// Find max partition indices
	maxPartA, maxPartB := 0, 0
	for _, partA := range partMapA {
		if partA > maxPartA {
			maxPartA = partA
		}
	}
	for _, partB := range partMapB {
		if partB > maxPartB {
			maxPartB = partB
		}
	}

	numPartA, numPartB := maxPartA+1, maxPartB+1

	// Initialize tables
	contingency := make([][]int, numPartA)
	for i := range contingency {
		contingency[i] = make([]int, numPartB)
	}
	rowTotals := make([]int, numPartA)
	colTotals := make([]int, numPartB)
	total := 0

	// Fill contingency table using only common nodes
	for _, nodeID := range commonNodes {
		partA, existsA := partMapA[nodeID]
		partB, existsB := partMapB[nodeID]

		if existsA && existsB {
			contingency[partA][partB]++
			rowTotals[partA]++
			colTotals[partB]++
			total++
		}
	}

	return contingency, rowTotals, colTotals, total
}

// mutualInformation computes MI from contingency table
func (s *ComparisonService) mutualInformation(contingency [][]int, rowTotals, colTotals []int, total int) float64 {
	if total == 0 {
		return 0.0
	}

	mi := 0.0
	totalFloat := float64(total)

	for i := range contingency {
		for j := range contingency[i] {
			if contingency[i][j] > 0 && rowTotals[i] > 0 && colTotals[j] > 0 {
				pij := float64(contingency[i][j]) / totalFloat
				pi := float64(rowTotals[i]) / totalFloat
				pj := float64(colTotals[j]) / totalFloat

				mi += pij * math.Log(pij/(pi*pj))
			}
		}
	}

	return mi
}

// computeHierarchicalEntropy computes entropy of a hierarchy
func (s *ComparisonService) computeHierarchicalEntropy(hier *models.Hierarchy, commonNodes []string) float64 {
	totalWeightedEntropy := 0.0
	totalWeight := 0.0

	for level, hierarchyLevel := range hier.Levels {
		if len(hierarchyLevel.Communities) <= 1 {
			continue // Skip trivial levels
		}

		levelEntropy := s.computeLevelEntropy(&hierarchyLevel, commonNodes)
		weight := 1.0 / float64(level+1) // Same weighting as MI

		totalWeightedEntropy += weight * levelEntropy
		totalWeight += weight
	}

	if totalWeight == 0 {
		return 0.0
	}

	return totalWeightedEntropy / totalWeight
}

// computeLevelEntropy computes entropy for a single level
func (s *ComparisonService) computeLevelEntropy(level *models.HierarchyLevel, commonNodes []string) float64 {
	partitionMap := s.buildNodeToPartitionMap(level)

	// Count nodes in each partition
	partitionCounts := make(map[int]int)
	totalNodes := 0

	for _, nodeID := range commonNodes {
		if partIdx, exists := partitionMap[nodeID]; exists {
			partitionCounts[partIdx]++
			totalNodes++
		}
	}

	if totalNodes == 0 {
		return 0.0
	}

	// Compute Shannon entropy
	entropy := 0.0
	for _, count := range partitionCounts {
		if count > 0 {
			p := float64(count) / float64(totalNodes)
			entropy -= p * math.Log(p)
		}
	}

	return entropy
}
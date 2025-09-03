package service

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"

	// "graph-viz-backend/algorithm"
	"graph-viz-backend/models"


	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

// ClusteringService orchestrates clustering operations
type ClusteringService struct {
	datasetService *DatasetService
	jobService     *JobService
}

// NewClusteringService creates a new clustering service
func NewClusteringService(datasetService *DatasetService, jobService *JobService) *ClusteringService {
	return &ClusteringService{
		datasetService: datasetService,
		jobService:     jobService,
	}
}

// StartClustering starts a clustering job for a dataset
func (s *ClusteringService) StartClustering(datasetID string, algorithmType models.AlgorithmType, params models.JobParameters) (*models.Job, error) {
    log.Info().
        Str("dataset_id", datasetID).
        Str("algorithm", string(algorithmType)).
        Msg("Starting clustering job")

    dataset, err := s.datasetService.Get(datasetID)
    if err != nil {
        return nil, fmt.Errorf("dataset not found: %w", err)
    }

    if dataset.Status != models.DatasetStatusReady {
        return nil, fmt.Errorf("dataset not available: %s", dataset.Status)
    }

    activeJobs := s.jobService.GetActiveJobsForDataset(datasetID)
    if len(activeJobs) >= 3 { // Configurable limit
        return nil, fmt.Errorf("maximum concurrent experiments (3) reached for dataset")
    }
    
    // Submit job (completely independent lifecycle)
    job, err := s.jobService.Submit(datasetID, algorithmType, params)
    if err != nil {
        return nil, fmt.Errorf("failed to submit job: %w", err)
    }

    log.Info().
        Str("job_id", job.ID).
        Str("dataset_id", datasetID).
        Str("algorithm", string(algorithmType)).
        Msg("Clustering job started successfully")

    return job, nil
}

// GetJobStatus retrieves the status of a clustering job
func (s *ClusteringService) GetJobStatus(jobID string) (*models.Job, error) {
    job, err := s.jobService.Get(jobID)
    if err != nil {
        return nil, err
    }

    return job, nil
}
// GetHierarchy retrieves the full hierarchy for a completed job
func (s *ClusteringService) GetHierarchy(datasetID, jobID string) (*models.Hierarchy, error) {
	// Validate job exists and is completed
	job, err := s.jobService.Get(jobID)
	if err != nil {
		return nil, fmt.Errorf("job not found: %w", err)
	}

	if job.DatasetID != datasetID {
		return nil, fmt.Errorf("job does not belong to dataset %s", datasetID)
	}

	if job.Status != models.JobStatusCompleted {
		return nil, fmt.Errorf("job not completed, status: %s", job.Status)
	}

	// Get algorithm result
	result, err := s.jobService.GetResult(jobID)
	if err != nil {
		return nil, fmt.Errorf("result not found: %w", err)
	}

	if result.Hierarchy == nil {
		return nil, fmt.Errorf("hierarchy not available")
	}

	// Set job ID in hierarchy
	hierarchy := *result.Hierarchy
	hierarchy.JobID = jobID

	return &hierarchy, nil
}

// GetHierarchyLevel retrieves a specific level of the hierarchy
func (s *ClusteringService) GetHierarchyLevel(datasetID, jobID string, level int) (*models.HierarchyLevel, error) {
	hierarchy, err := s.GetHierarchy(datasetID, jobID)
	if err != nil {
		return nil, err
	}

	if level < 0 || level >= len(hierarchy.Levels) {
		return nil, fmt.Errorf("invalid level %d, hierarchy has %d levels", level, len(hierarchy.Levels))
	}

	return &hierarchy.Levels[level], nil
}

// GetClusterNodes retrieves nodes for a specific cluster
func (s *ClusteringService) GetClusterNodes(datasetID, jobID, clusterID string) (*models.ClusterResponse, error) {
	hierarchy, err := s.GetHierarchy(datasetID, jobID)
	if err != nil {
		return nil, err
	}

	// Find cluster in hierarchy levels
	for levelIdx, level := range hierarchy.Levels {
		if nodes, exists := level.Communities[clusterID]; exists {
			// Build response
			nodeInfos := make([]models.NodeInfo, len(nodes))
			for i, nodeID := range nodes {
				nodeInfo := models.NodeInfo{
					ID:    nodeID,
					Label: nodeID,
					Type:  "leaf",
					Metadata: map[string]interface{}{
						"level": levelIdx,
					},
				}

				// Add coordinates if available
				if pos, hasCoords := hierarchy.Coordinates[nodeID]; hasCoords {
					nodeInfo.Position = pos
				}

				// Determine node type based on level
				if levelIdx > 0 {
					nodeInfo.Type = "supernode"
				}

				nodeInfos[i] = nodeInfo
			}

			// Get edges for this cluster (simplified - would need actual edge data)
			edges := s.getClusterEdges(hierarchy, clusterID, levelIdx)

			return &models.ClusterResponse{
				ClusterID: clusterID,
				Level:     levelIdx,
				Nodes:     nodeInfos,
				Edges:     edges,
			}, nil
		}
	}

	return nil, fmt.Errorf("cluster not found: %s", clusterID)
}

// CancelJob cancels a running clustering job
func (s *ClusteringService) CancelJob(jobID string) error {
    err := s.jobService.Cancel(jobID)
    if err != nil {
        return err
    }
    
    log.Info().
        Str("job_id", jobID).
        Msg("Clustering job cancelled")

    return nil
}


// GetJobHistory returns all jobs for a dataset
func (s *ClusteringService) GetJobHistory(datasetID string) ([]*models.Job, error) {
	// Validate dataset exists
	_, err := s.datasetService.Get(datasetID)
	if err != nil {
		return nil, fmt.Errorf("dataset not found: %w", err)
	}

	return s.jobService.List(datasetID), nil
}

// getClusterEdges extracts edges for a specific cluster at a given level
func (s *ClusteringService) getClusterEdges(hierarchy *models.Hierarchy, clusterID string, level int) []models.EdgeInfo {
    result, err := s.jobService.GetResult(hierarchy.JobID)
    if err != nil {
        return []models.EdgeInfo{}
    }
    
    // Extract from stored graph data
    switch typedResult := result.RawResult.(type) {
    case *louvain.Result:
        if level < len(typedResult.Levels) && typedResult.Levels[level].Graph != nil {
            return s.extractEdges(typedResult.Levels[level].Graph, typedResult.Levels[level].Communities, clusterID, level)
        }
    case *scar.Result:
        if level < len(typedResult.Levels) && typedResult.Levels[level].SketchGraph != nil {
            return s.extractEdges(typedResult.Levels[level].SketchGraph, typedResult.Levels[level].Communities, clusterID, level)
        }
    }
    return []models.EdgeInfo{}
}

// In service/clustering.go
func (s *ClusteringService) extractEdges(graph interface{ GetNeighbors(int) ([]int, []float64) }, communities map[int][]int, clusterID string, level int) []models.EdgeInfo {
    // Parse cluster ID to get community ID
    communityID := s.parseClusterID(clusterID)
    if communityID == -1 {
        log.Debug().Str("cluster_id", clusterID).Msg("Invalid cluster ID format")
        return []models.EdgeInfo{}
    }
    
    // Get nodes in this cluster
    clusterNodes, exists := communities[communityID]
    if !exists {
        log.Debug().Int("community_id", communityID).Msg("Community not found")
        return []models.EdgeInfo{}
    }
    
    if len(clusterNodes) == 0 {
        return []models.EdgeInfo{}
    }
    
    edges := make([]models.EdgeInfo, 0)
    nodeSet := make(map[int]bool)
    
    // Build node set for O(1) lookup
    for _, nodeID := range clusterNodes {
        nodeSet[nodeID] = true
    }
    
    // Extract edges from each node in the cluster
    for _, sourceNode := range clusterNodes {
        neighbors, weights := graph.GetNeighbors(sourceNode)
        
        for i, targetNode := range neighbors {
            // Only include edges within this cluster
            if !nodeSet[targetNode] {
                continue
            }
            
            // Avoid duplicate edges (since graph is undirected)
            // Only add edge if source <= target
            if sourceNode <= targetNode {
                edges = append(edges, models.EdgeInfo{
                    Source: s.formatNodeID(sourceNode, level),
                    Target: s.formatNodeID(targetNode, level),
                    Weight: weights[i],
                })
            }
        }
    }
    
    log.Debug().
        Str("cluster_id", clusterID).
        Int("level", level).
        Int("nodes", len(clusterNodes)).
        Int("edges", len(edges)).
        Msg("Extracted cluster edges")
    
    return edges
}

// Helper to parse cluster ID from formats like "c0_l1_5" or just "5"
func (s *ClusteringService) parseClusterID(clusterID string) int {
    // Try parsing hierarchical format: "c0_l1_5" -> 5
    if strings.Contains(clusterID, "_") {
        parts := strings.Split(clusterID, "_")
        if len(parts) >= 3 {
            if id, err := strconv.Atoi(parts[len(parts)-1]); err == nil {
                return id
            }
        }
    }
    
    // Try direct integer parsing: "5" -> 5
    if id, err := strconv.Atoi(clusterID); err == nil {
        return id
    }
    
    return -1
}

// Helper to format node IDs consistently with hierarchy format
func (s *ClusteringService) formatNodeID(nodeID, level int) string {
    if level == 0 {
        // Leaf nodes: use original ID
        return fmt.Sprintf("%d", nodeID)
    } else {
        // Super-nodes: use hierarchy format
        return fmt.Sprintf("c0_l%d_%d", level, nodeID)
    }
}
package api

import (
	"encoding/json"
	"mime/multipart"
	"net/http"
	"strconv"
	"fmt"
	"time"

	"github.com/gorilla/mux"
	"github.com/rs/zerolog/log"

	"graph-viz-backend/models"
	"graph-viz-backend/service"
	"graph-viz-backend/utils"
)

// Handlers contains HTTP request handlers
type Handlers struct {
	datasetService    *service.DatasetService
	clusteringService *service.ClusteringService
	jobService        *service.JobService
	comparisonService  *service.ComparisonService
}

// NewHandlers creates new API handlers
func NewHandlers(datasetService *service.DatasetService, clusteringService *service.ClusteringService, jobService *service.JobService, comparisonService *service.ComparisonService) *Handlers {
	return &Handlers{
		datasetService:    datasetService,
		clusteringService: clusteringService,
		jobService:        jobService,
		comparisonService: comparisonService,
	}
}

// UploadDataset handles dataset upload
func (h *Handlers) UploadDataset(w http.ResponseWriter, r *http.Request) {
	log.Info().Msg("Dataset upload request received")

	// Parse multipart form
	err := r.ParseMultipartForm(100 << 20) // 100MB max
	if err != nil {
		log.Error().Err(err).Msg("Failed to parse multipart form")
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Invalid multipart form", err)
		return
	}

	// Get dataset name from form
	name := r.FormValue("name")
	if name == "" {
		name = "Unnamed Dataset"
	}

	// Extract files
	files := make(map[string]*multipart.FileHeader)
	requiredFiles := []string{"graphFile", "propertiesFile", "pathFile"}
	
	for _, fieldName := range requiredFiles {
		file, header, err := r.FormFile(fieldName)
		if err != nil {
			log.Error().
				Str("field", fieldName).
				Err(err).
				Msg("Missing required file")
			utils.WriteErrorResponse(w, http.StatusBadRequest, "Missing required file: "+fieldName, err)
			return
		}
		file.Close() // Close immediately, we'll reopen in service
		files[fieldName] = header
	}

	// Upload dataset
	dataset, err := h.datasetService.Upload(name, files)
	if err != nil {
		log.Error().Err(err).Msg("Dataset upload failed")
		utils.WriteErrorResponse(w, http.StatusInternalServerError, "Dataset upload failed", err)
		return
	}

	log.Info().
		Str("dataset_id", dataset.ID).
		Str("name", dataset.Name).
		Msg("Dataset uploaded successfully")

	// Return success response
	response := models.UploadResponse{
		DatasetID: dataset.ID,
		Dataset:   *dataset,
	}
	utils.WriteSuccessResponse(w, "Dataset uploaded successfully", response)
}

// ListDatasets lists all datasets
func (h *Handlers) ListDatasets(w http.ResponseWriter, r *http.Request) {
	datasets := h.datasetService.List()
	utils.WriteSuccessResponse(w, "Datasets retrieved successfully", datasets)
}

// GetDataset retrieves a specific dataset
func (h *Handlers) GetDataset(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]

	dataset, err := h.datasetService.Get(datasetID)
	if err != nil {
		log.Error().
			Str("dataset_id", datasetID).
			Err(err).
			Msg("Dataset not found")
		utils.WriteErrorResponse(w, http.StatusNotFound, "Dataset not found", err)
		return
	}

	utils.WriteSuccessResponse(w, "Dataset retrieved successfully", dataset)
}

// DeleteDataset deletes a dataset
func (h *Handlers) DeleteDataset(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]

	err := h.datasetService.Delete(datasetID)
	if err != nil {
		log.Error().
			Str("dataset_id", datasetID).
			Err(err).
			Msg("Dataset deletion failed")
		utils.WriteErrorResponse(w, http.StatusInternalServerError, "Dataset deletion failed", err)
		return
	}

	utils.WriteSuccessResponse(w, "Dataset deleted successfully", nil)
}

// StartClustering starts a clustering job
func (h *Handlers) StartClustering(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]

	// Parse request body
	var req struct {
		Algorithm  models.AlgorithmType  `json:"algorithm"`
		Parameters models.JobParameters `json:"parameters"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Error().Err(err).Msg("Invalid request body")
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	// Validate algorithm
	if req.Algorithm != models.AlgorithmLouvain && req.Algorithm != models.AlgorithmSCAR {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Invalid algorithm, must be 'louvain' or 'scar'", nil)
		return
	}

	log.Info().
		Str("dataset_id", datasetID).
		Str("algorithm", string(req.Algorithm)).
		Msg("Starting clustering job")

	// Start clustering
	job, err := h.clusteringService.StartClustering(datasetID, req.Algorithm, req.Parameters)
	if err != nil {
		log.Error().
			Str("dataset_id", datasetID).
			Err(err).
			Msg("Failed to start clustering job")
		utils.WriteErrorResponse(w, http.StatusInternalServerError, "Failed to start clustering", err)
		return
	}

	log.Info().
		Str("job_id", job.ID).
		Str("dataset_id", datasetID).
		Str("algorithm", string(req.Algorithm)).
		Msg("Clustering job started successfully")

	// Return job info
	response := models.ClusteringResponse{
		JobID: job.ID,
		Job:   *job,
	}
	utils.WriteSuccessResponse(w, "Clustering job started", response)
}

// GetClusteringJob gets clustering job status
func (h *Handlers) GetClusteringJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]
	jobID := vars["jobId"]

	job, err := h.clusteringService.GetJobStatus(jobID)
	if err != nil {
		log.Error().
			Str("job_id", jobID).
			Err(err).
			Msg("Job not found")
		utils.WriteErrorResponse(w, http.StatusNotFound, "Job not found", err)
		return
	}

	if job.DatasetID != datasetID {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Job does not belong to this dataset", nil)
		return
	}

	utils.WriteSuccessResponse(w, "Job status retrieved", job)
}

// CancelClusteringJob cancels a clustering job
func (h *Handlers) CancelClusteringJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["jobId"]

	err := h.clusteringService.CancelJob(jobID)
	if err != nil {
		log.Error().
			Str("job_id", jobID).
			Err(err).
			Msg("Failed to cancel job")
		utils.WriteErrorResponse(w, http.StatusInternalServerError, "Failed to cancel job", err)
		return
	}

	utils.WriteSuccessResponse(w, "Job cancelled successfully", nil)
}

// GetFullHierarchy retrieves the complete hierarchy
func (h *Handlers) GetFullHierarchy(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]
	jobID := r.URL.Query().Get("jobId")

	if jobID == "" {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Missing jobId query parameter", nil)
		return
	}

	hierarchy, err := h.clusteringService.GetHierarchy(datasetID, jobID)
	if err != nil {
		log.Error().
			Str("dataset_id", datasetID).
			Str("job_id", jobID).
			Err(err).
			Msg("Failed to get hierarchy")
		utils.WriteErrorResponse(w, http.StatusNotFound, "Hierarchy not found", err)
		return
	}

	response := models.HierarchyResponse{
		Hierarchy: *hierarchy,
	}
	utils.WriteSuccessResponse(w, "Hierarchy retrieved successfully", response)
}

// GetHierarchyLevel retrieves a specific hierarchy level
func (h *Handlers) GetHierarchyLevel(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]
	levelStr := vars["level"]
	jobID := r.URL.Query().Get("jobId")

	if jobID == "" {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Missing jobId query parameter", nil)
		return
	}

	level, err := strconv.Atoi(levelStr)
	if err != nil {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Invalid level parameter", err)
		return
	}

	hierarchyLevel, err := h.clusteringService.GetHierarchyLevel(datasetID, jobID, level)
	if err != nil {
		log.Error().
			Str("dataset_id", datasetID).
			Str("job_id", jobID).
			Int("level", level).
			Err(err).
			Msg("Failed to get hierarchy level")
		utils.WriteErrorResponse(w, http.StatusNotFound, "Hierarchy level not found", err)
		return
	}

	utils.WriteSuccessResponse(w, "Hierarchy level retrieved successfully", hierarchyLevel)
}

// GetClusterNodes retrieves nodes for a specific cluster
func (h *Handlers) GetClusterNodes(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]
	clusterID := vars["clusterId"]
	jobID := r.URL.Query().Get("jobId")

	if jobID == "" {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Missing jobId query parameter", nil)
		return
	}

	cluster, err := h.clusteringService.GetClusterNodes(datasetID, jobID, clusterID)
	if err != nil {
		log.Error().
			Str("dataset_id", datasetID).
			Str("job_id", jobID).
			Str("cluster_id", clusterID).
			Err(err).
			Msg("Failed to get cluster nodes")
		utils.WriteErrorResponse(w, http.StatusNotFound, "Cluster not found", err)
		return
	}

	utils.WriteSuccessResponse(w, "Cluster nodes retrieved successfully", cluster)
}

// GetJob retrieves a job by ID
func (h *Handlers) GetJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["jobId"]

	job, err := h.jobService.Get(jobID)
	if err != nil {
		utils.WriteErrorResponse(w, http.StatusNotFound, "Job not found", err)
		return
	}

	utils.WriteSuccessResponse(w, "Job retrieved successfully", job)
}

// CancelJob cancels a job
func (h *Handlers) CancelJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["jobId"]

	err := h.jobService.Cancel(jobID)
	if err != nil {
		utils.WriteErrorResponse(w, http.StatusInternalServerError, "Failed to cancel job", err)
		return
	}

	utils.WriteSuccessResponse(w, "Job cancelled successfully", nil)
}

// HealthCheck returns server health status
func (h *Handlers) HealthCheck(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().Format(time.RFC3339),
		"version":   "2.0.0",
	}
	utils.WriteSuccessResponse(w, "Service is healthy", health)
}

// ListAlgorithms lists available algorithms
func (h *Handlers) ListAlgorithms(w http.ResponseWriter, r *http.Request) {
	algorithms := []map[string]interface{}{
		{
			"name":        "louvain",
			"description": "Louvain algorithm with materialization for heterogeneous graphs",
			"parameters": []map[string]interface{}{
				{"name": "maxLevels", "type": "integer", "default": 10, "description": "Maximum hierarchy levels"},
				{"name": "maxIterations", "type": "integer", "default": 100, "description": "Maximum iterations per level"},
				{"name": "minModularityGain", "type": "number", "default": 0.000001, "description": "Minimum modularity gain threshold"},
			},
		},
		{
			"name":        "scar",
			"description": "SCAR (Sketch-based Community detection with Approximate Refinement)",
			"parameters": []map[string]interface{}{
				{"name": "k", "type": "integer", "default": 64, "description": "Bottom-K sketch size"},
				{"name": "nk", "type": "integer", "default": 4, "description": "Number of sketch layers"},
				{"name": "threshold", "type": "number", "default": 0.5, "description": "Sketch fullness threshold"},
			},
		},
	}

	utils.WriteSuccessResponse(w, "Algorithms retrieved successfully", algorithms)
}



// CreateComparison starts a new comparison between experiments
func (h *Handlers) CreateComparison(w http.ResponseWriter, r *http.Request) {
	log.Info().Msg("Comparison creation request received")

	// Parse request body
	var req models.CreateComparisonRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Error().Err(err).Msg("Invalid request body")
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	// Validate required fields
	if req.Name == "" {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Comparison name is required", nil)
		return
	}

	if req.ExperimentA.DatasetID == "" || req.ExperimentA.JobID == "" {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Experiment A references are required", nil)
		return
	}

	if req.ExperimentB.DatasetID == "" || req.ExperimentB.JobID == "" {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Experiment B references are required", nil)
		return
	}

	if len(req.Metrics) == 0 {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "At least one metric must be specified", nil)
		return
	}

	log.Info().
		Str("name", req.Name).
		Str("experiment_a", fmt.Sprintf("%s/%s", req.ExperimentA.DatasetID, req.ExperimentA.JobID)).
		Str("experiment_b", fmt.Sprintf("%s/%s", req.ExperimentB.DatasetID, req.ExperimentB.JobID)).
		Strs("metrics", req.Metrics).
		Msg("Creating comparison")

	// Create comparison
	comparison, err := h.comparisonService.Create(req.Name, req.ExperimentA, req.ExperimentB, req.Metrics, req.Options)
	if err != nil {
		log.Error().Err(err).Msg("Comparison creation failed")
		utils.WriteErrorResponse(w, http.StatusInternalServerError, "Comparison creation failed", err)
		return
	}

	log.Info().
		Str("comparison_id", comparison.ID).
		Str("name", req.Name).
		Msg("Comparison created successfully")

	// Return success response
	response := models.ComparisonResponse{
		ComparisonID: comparison.ID,
		Comparison:   *comparison,
	}
	utils.WriteSuccessResponse(w, "Comparison started successfully", response)
}

// GetComparison retrieves a comparison by ID (handles both legacy and multi-format)
func (h *Handlers) GetComparison(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    comparisonID := vars["comparisonId"]

    // Get basic comparison info
    comparison, err := h.comparisonService.Get(comparisonID)
    if err != nil {
        log.Error().
            Str("comparison_id", comparisonID).
            Err(err).
            Msg("Comparison not found")
        utils.WriteErrorResponse(w, http.StatusNotFound, "Comparison not found", err)
        return
    }

    // Check if this is a multi-comparison
    if h.comparisonService.IsMultiComparison(comparisonID) {
        // Return multi-comparison format
        multiResult, err := h.comparisonService.GetMultiComparisonResult(comparisonID)
        if err != nil {
            log.Error().
                Str("comparison_id", comparisonID).
                Err(err).
                Msg("Failed to get multi-comparison result")
            utils.WriteErrorResponse(w, http.StatusInternalServerError, "Failed to get multi-comparison result", err)
            return
        }

        // ✅ CORRECT: Multi-comparison response format
        response := map[string]interface{}{
            "id":          comparison.ID,
            "name":        comparison.Name,
            "status":      comparison.Status,
            "createdAt":   comparison.CreatedAt,
            "completedAt": comparison.CompletedAt,
            "result":      multiResult, // Contains baselineConfig and experiments
        }

        utils.WriteSuccessResponse(w, "Multi-comparison retrieved successfully", response)
        return
    }

    // ✅ UNCHANGED: Legacy two-experiment comparison format
    utils.WriteSuccessResponse(w, "Comparison retrieved successfully", comparison)
}

// DeleteComparison removes a comparison
func (h *Handlers) DeleteComparison(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	comparisonID := vars["comparisonId"]

	err := h.comparisonService.Delete(comparisonID)
	if err != nil {
		log.Error().
			Str("comparison_id", comparisonID).
			Err(err).
			Msg("Comparison deletion failed")
		utils.WriteErrorResponse(w, http.StatusInternalServerError, "Comparison deletion failed", err)
		return
	}

	log.Info().
		Str("comparison_id", comparisonID).
		Msg("Comparison deleted successfully")

	utils.WriteSuccessResponse(w, "Comparison deleted successfully", nil)
}

// ListComparisons returns all comparisons (optional)
func (h *Handlers) ListComparisons(w http.ResponseWriter, r *http.Request) {
	comparisons := h.comparisonService.List()
	utils.WriteSuccessResponse(w, "Comparisons retrieved successfully", comparisons)
}

func (h *Handlers) CreateMultiComparison(w http.ResponseWriter, r *http.Request) {
	log.Info().Msg("Multi-comparison request received")

	// Parse request body
	var req models.MultiComparisonRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Error().Err(err).Msg("Invalid multi-comparison request body")
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Invalid request body", err)
		return
	}

	// Validate required fields
	if req.Name == "" {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Comparison name is required", nil)
		return
	}

	if len(req.SelectedExperiments) < 2 {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "At least 2 experiments required", nil)
		return
	}

	if len(req.SelectedExperiments) > 10 {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "Maximum 10 experiments allowed", nil)
		return
	}

	if len(req.Metrics) == 0 {
		utils.WriteErrorResponse(w, http.StatusBadRequest, "At least one metric must be specified", nil)
		return
	}

	// Validate metric types
	validMetrics := map[string]bool{
		"hmi":                    true, // Hierarchical Mutual Information
		"custom_leaf_metric":     true, // jaccard × AM/QM_displacement × AM/QM_radius (entire leaf level)
		"custom_displayed_metric": true, // same formula but for largest community comparison
	}

	for _, metric := range req.Metrics {
		if !validMetrics[metric] {
			utils.WriteErrorResponse(w, http.StatusBadRequest, fmt.Sprintf("Invalid metric: %s", metric), nil)
			return
		}
	}

	log.Info().
		Str("name", req.Name).
		Str("selected_experiments", fmt.Sprintf("%v", req.SelectedExperiments)).
		Str("baseline_experiment", fmt.Sprintf("%v", req.BaselineExperiment)).
		Strs("metrics", req.Metrics).

		Msg("Multi-comparison details")

	// Create multi-comparison
	comparison, err := h.comparisonService.CreateMultiComparison(req)
	if err != nil {
		log.Error().Err(err).Msg("Multi-comparison creation failed")
		utils.WriteErrorResponse(w, http.StatusInternalServerError, "Multi-comparison creation failed", err)
		return
	}

	log.Info().
		Str("comparison_id", comparison.ID).
		Str("name", req.Name).
		Str("baseline_experiment", fmt.Sprintf("%v", req.BaselineExperiment)).
		Msg("Multi-comparison created successfully")

	// Return success response with clear explanation
	response := models.ComparisonResponse{
		ComparisonID: comparison.ID,
		Comparison:   *comparison,
	}
	
	message := "Multi-comparison started successfully"
	
	utils.WriteSuccessResponse(w, message, response)
}

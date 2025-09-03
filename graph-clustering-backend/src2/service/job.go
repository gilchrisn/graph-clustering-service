package service

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"

	"graph-viz-backend/algorithm"
	"graph-viz-backend/models"
)

// JobService handles background job processing
type JobService struct {
	jobs            map[string]*models.Job
	results         map[string]*algorithm.AlgorithmResult
	workers         chan struct{}
	registry        *algorithm.Registry
	datasetService  *DatasetService
	mutex           sync.RWMutex
	jobTTL          time.Duration
	cleanupInterval time.Duration
}

// NewJobService creates a new job service
func NewJobService(datasetService *DatasetService) *JobService {
	service := &JobService{
		jobs:            make(map[string]*models.Job),
		results:         make(map[string]*algorithm.AlgorithmResult),
		workers:         make(chan struct{}, 4), // Default to 4 concurrent jobs
		registry:        algorithm.NewRegistry(),
		datasetService:  datasetService,
		jobTTL:          time.Hour,
		cleanupInterval: 5 * time.Minute,
	}

	// Start cleanup goroutine
	go service.cleanupLoop()

	return service
}

// Submit creates and queues a new clustering job
func (s *JobService) Submit(datasetID string, algorithmType models.AlgorithmType, params models.JobParameters) (*models.Job, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	// Generate unique job ID
	jobID := uuid.New().String()

	// Get algorithm
	algorithm, exists := s.registry.Get(algorithmType)
	if !exists {
		return nil, fmt.Errorf("unknown algorithm: %s", algorithmType)
	}

	// Validate parameters
	if err := algorithm.ValidateParameters(params); err != nil {
		return nil, fmt.Errorf("invalid parameters: %w", err)
	}

	// Create job
	now := time.Now()
	job := &models.Job{
		ID:         jobID,
		DatasetID:  datasetID,
		Algorithm:  algorithmType,
		Parameters: params,
		Status:     models.JobStatusQueued,
		Progress: models.JobProgress{
			Percentage: 0,
			Message:    "Queued",
		},
		CreatedAt: now,
		UpdatedAt: now,
	}

	// Store job
	s.jobs[jobID] = job

	log.Info().
		Str("job_id", jobID).
		Str("dataset_id", datasetID).
		Str("algorithm", string(algorithmType)).
		Msg("Job submitted")

	// Start processing in background
	go s.processJob(jobID)

	return job, nil
}

// Get retrieves a job by ID
func (s *JobService) Get(jobID string) (*models.Job, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	job, exists := s.jobs[jobID]
	if !exists {
		return nil, fmt.Errorf("job not found: %s", jobID)
	}

	return job, nil
}

// GetResult retrieves the algorithm result for a completed job
func (s *JobService) GetResult(jobID string) (*algorithm.AlgorithmResult, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	result, exists := s.results[jobID]
	if !exists {
		return nil, fmt.Errorf("result not found for job: %s", jobID)
	}

	return result, nil
}

// List returns all jobs for a dataset
func (s *JobService) List(datasetID string) []*models.Job {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	var jobs []*models.Job
	for _, job := range s.jobs {
		if job.DatasetID == datasetID {
			jobs = append(jobs, job)
		}
	}

	return jobs
}

// Cancel cancels a running job
func (s *JobService) Cancel(jobID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	job, exists := s.jobs[jobID]
	if !exists {
		return fmt.Errorf("job not found: %s", jobID)
	}

	if job.Status == models.JobStatusRunning {
		job.Status = models.JobStatusCancelled
		job.Progress.Message = "Cancelled"
		now := time.Now()
		job.CompletedAt = &now

		log.Info().
			Str("job_id", jobID).
			Msg("Job cancelled")
	}

	return nil
}

// processJob processes a job in the background
func (s *JobService) processJob(jobID string) {
	// Acquire worker slot
	s.workers <- struct{}{}
	defer func() { <-s.workers }()

	// Get job
	s.mutex.RLock()
	job, exists := s.jobs[jobID]
	s.mutex.RUnlock()

	if !exists {
		log.Error().Str("job_id", jobID).Msg("Job not found during processing")
		return
	}

	// Update job status to running and set start time
	startTime := time.Now()
	s.updateJobStatusWithStartTime(jobID, models.JobStatusRunning, 0, "Starting...", &startTime)

	log.Info().
		Str("job_id", jobID).
		Str("dataset_id", job.DatasetID).
		Str("algorithm", string(job.Algorithm)).
		Msg("Job processing started")

	// Get algorithm
	alg, exists := s.registry.Get(job.Algorithm)
	if !exists {
		s.failJob(jobID, fmt.Errorf("unknown algorithm: %s", job.Algorithm))
		return
	}

	// Get dataset (this would normally come from dataset service)
	dataset, err := s.getDatasetForJob(job.DatasetID)
	if err != nil {
		s.failJob(jobID, fmt.Errorf("failed to get dataset: %w", err))
		return
	}

	// Create progress callback
	progressCallback := func(percentage int, message string) {
		s.updateJobStatus(jobID, models.JobStatusRunning, percentage, message)
	}

	// Run algorithm with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	result, err := alg.Run(ctx, dataset, job.Parameters, progressCallback)
	if err != nil {
		s.failJob(jobID, fmt.Errorf("algorithm execution failed: %w", err))
		return
	}

	// Complete job
	s.completeJob(jobID, result)
}

// GetActiveJobsForDataset retrieves all active jobs for a dataset
func (s *JobService) GetActiveJobsForDataset(datasetID string) []*models.Job {
    s.mutex.RLock()
    defer s.mutex.RUnlock()

    var activeJobs []*models.Job
    for _, job := range s.jobs {
        if job.DatasetID == datasetID && 
           (job.Status == models.JobStatusQueued || job.Status == models.JobStatusRunning) {
            activeJobs = append(activeJobs, job)
        }
    }
    return activeJobs
}

// updateJobStatus updates job progress
func (s *JobService) updateJobStatus(jobID string, status models.JobStatus, percentage int, message string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	job, exists := s.jobs[jobID]
	if !exists {
		return
	}

	job.Status = status
	job.Progress.Percentage = percentage
	job.Progress.Message = message
	job.UpdatedAt = time.Now()

	log.Debug().
		Str("job_id", jobID).
		Str("status", string(status)).
		Int("percentage", percentage).
		Str("message", message).
		Msg("Job status updated")
}

// updateJobStatusWithStartTime updates job progress and sets start time
func (s *JobService) updateJobStatusWithStartTime(jobID string, status models.JobStatus, percentage int, message string, startTime *time.Time) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	job, exists := s.jobs[jobID]
	if !exists {
		return
	}

	job.Status = status
	job.Progress.Percentage = percentage
	job.Progress.Message = message
	job.UpdatedAt = time.Now()
	if startTime != nil {
		job.StartedAt = startTime
	}

	log.Debug().
		Str("job_id", jobID).
		Str("status", string(status)).
		Int("percentage", percentage).
		Str("message", message).
		Msg("Job status updated with start time")
}

// completeJob marks a job as completed with results
func (s *JobService) completeJob(jobID string, result *algorithm.AlgorithmResult) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	job, exists := s.jobs[jobID]
	if !exists {
		return
	}

	// Update job
	job.Status = models.JobStatusCompleted
	job.Progress.Percentage = 100
	job.Progress.Message = "Complete"
	now := time.Now()
	job.CompletedAt = &now
	job.UpdatedAt = now

	// Set job result summary
	job.Result = &models.JobResult{
		Modularity:       result.Modularity,
		NumLevels:        result.NumLevels,
		NumCommunities:   result.NumCommunities,
		ProcessingTimeMS: result.ProcessingTimeMS,
		Statistics:       result.Statistics,
	}

	// Store full algorithm result
	s.results[jobID] = result

	log.Info().
		Str("job_id", jobID).
		Float64("modularity", result.Modularity).
		Int("levels", result.NumLevels).
		Int64("processing_time_ms", result.ProcessingTimeMS).
		Msg("Job completed successfully")
}

// failJob marks a job as failed
func (s *JobService) failJob(jobID string, err error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	job, exists := s.jobs[jobID]
	if !exists {
		return
	}

	job.Status = models.JobStatusFailed
	job.Error = err.Error()
	job.Progress.Message = "Failed"
	now := time.Now()
	job.CompletedAt = &now
	job.UpdatedAt = now

	log.Error().
		Str("job_id", jobID).
		Err(err).
		Msg("Job failed")
}

// cleanupLoop periodically cleans up old jobs and results
func (s *JobService) cleanupLoop() {
	ticker := time.NewTicker(s.cleanupInterval)
	defer ticker.Stop()

	for range ticker.C {
		s.cleanup()
	}
}

// cleanup removes old jobs and results
func (s *JobService) cleanup() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	cutoff := time.Now().Add(-s.jobTTL)
	cleaned := 0

	for jobID, job := range s.jobs {
		if job.UpdatedAt.Before(cutoff) {
			delete(s.jobs, jobID)
			delete(s.results, jobID)
			cleaned++
		}
	}

	if cleaned > 0 {
		log.Info().
			Int("cleaned_jobs", cleaned).
			Msg("Job cleanup completed")
	}
}

// getDatasetForJob retrieves dataset info needed for job processing
func (s *JobService) getDatasetForJob(datasetID string) (*models.Dataset, error) {
	return s.datasetService.Get(datasetID)
}
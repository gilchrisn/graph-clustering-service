package service

import (
	"bufio"
	"fmt"
	"io"
	"mime/multipart"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/rs/zerolog/log"

	"graph-viz-backend/models"
)

// DatasetService handles dataset operations
type DatasetService struct {
	datasets map[string]*models.Dataset
	mutex    sync.RWMutex
}

// NewDatasetService creates a new dataset service
func NewDatasetService() *DatasetService {
	return &DatasetService{
		datasets: make(map[string]*models.Dataset),
	}
}

// Upload creates a new dataset from uploaded files
func (s *DatasetService) Upload(name string, files map[string]*multipart.FileHeader) (*models.Dataset, error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	// Generate unique dataset ID
	datasetID := uuid.New().String()
	
	log.Info().
		Str("dataset_id", datasetID).
		Str("name", name).
		Msg("Starting dataset upload")

	// Validate required files
	requiredFiles := []string{"graphFile", "propertiesFile", "pathFile"}
	for _, required := range requiredFiles {
		if _, exists := files[required]; !exists {
			return nil, fmt.Errorf("missing required file: %s", required)
		}
	}

	// Create upload directory
	uploadDir := filepath.Join("uploads", datasetID)
	if err := os.MkdirAll(uploadDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create upload directory: %w", err)
	}

	dataset := &models.Dataset{
		ID:        datasetID,
		Name:      name,
		Status:    models.DatasetStatusReady,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Save each file
	var totalSize int64
	for fileType, fileHeader := range files {
		destPath, err := s.saveUploadedFile(fileHeader, uploadDir, fileType)
		if err != nil {
			// Cleanup on failure
			os.RemoveAll(uploadDir)
			return nil, fmt.Errorf("failed to save %s: %w", fileType, err)
		}

		// Store file paths
		switch fileType {
		case "graphFile":
			dataset.Files.GraphFile = destPath
		case "propertiesFile":
			dataset.Files.PropertiesFile = destPath
		case "pathFile":
			dataset.Files.PathFile = destPath
		}

		// Accumulate file size
		if stat, err := os.Stat(destPath); err == nil {
			totalSize += stat.Size()
		}
	}

	// Analyze uploaded files
	metadata, err := s.analyzeFiles(dataset.Files)
	if err != nil {
		log.Warn().
			Str("dataset_id", datasetID).
			Err(err).
			Msg("Failed to analyze files, using defaults")
		metadata = models.DatasetMetadata{
			NodeCount: -1,
			EdgeCount: -1,
			FileSize:  totalSize,
		}
	}

	dataset.Metadata = metadata

	// Store dataset
	s.datasets[datasetID] = dataset

	log.Info().
		Str("dataset_id", datasetID).
		Int("nodes", metadata.NodeCount).
		Int("edges", metadata.EdgeCount).
		Int64("size_bytes", totalSize).
		Msg("Dataset upload complete")

	return dataset, nil
}

// Get retrieves a dataset by ID
func (s *DatasetService) Get(datasetID string) (*models.Dataset, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	dataset, exists := s.datasets[datasetID]
	if !exists {
		return nil, fmt.Errorf("dataset not found: %s", datasetID)
	}

	return dataset, nil
}

// List returns all datasets
func (s *DatasetService) List() []*models.Dataset {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	datasets := make([]*models.Dataset, 0, len(s.datasets))
	for _, dataset := range s.datasets {
		datasets = append(datasets, dataset)
	}

	return datasets
}

// UpdateStatus updates dataset status
func (s *DatasetService) UpdateStatus(datasetID string, status models.DatasetStatus) error {
    s.mutex.Lock()
    defer s.mutex.Unlock()

    dataset, exists := s.datasets[datasetID]
    if !exists {
        return fmt.Errorf("dataset not found: %s", datasetID)
    }

    switch status {
    case models.DatasetStatusReady, models.DatasetStatusCorrupted, models.DatasetStatusDeleted:
    default:
        return fmt.Errorf("invalid dataset status: %s (only ready/corrupted/deleted allowed)", status)
    }

    dataset.Status = status
    dataset.UpdatedAt = time.Now()

    log.Debug().
        Str("dataset_id", datasetID).
        Str("status", string(status)).
        Msg("Dataset status updated")

    return nil
}

// Delete removes a dataset and its files
func (s *DatasetService) Delete(datasetID string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	dataset, exists := s.datasets[datasetID]
	if !exists {
		return fmt.Errorf("dataset not found: %s", datasetID)
	}

	// Remove files
	uploadDir := filepath.Dir(dataset.Files.GraphFile)
	if err := os.RemoveAll(uploadDir); err != nil {
		log.Warn().
			Str("dataset_id", datasetID).
			Err(err).
			Msg("Failed to remove dataset files")
	}

	// Remove from memory
	delete(s.datasets, datasetID)

	log.Info().
		Str("dataset_id", datasetID).
		Msg("Dataset deleted")

	return nil
}

// saveUploadedFile saves an uploaded file to the destination
func (s *DatasetService) saveUploadedFile(fileHeader *multipart.FileHeader, uploadDir, fileType string) (string, error) {
	// Open uploaded file
	file, err := fileHeader.Open()
	if err != nil {
		return "", fmt.Errorf("failed to open uploaded file: %w", err)
	}
	defer file.Close()

	// Determine file extension based on type
	var extension string
	switch fileType {
	case "graphFile":
		extension = ".txt"
	case "propertiesFile":
		extension = "_properties.txt"
	case "pathFile":
		extension = "_path.txt"
	default:
		extension = ".txt"
	}

	// Create destination file
	destPath := filepath.Join(uploadDir, fileType+extension)
	destFile, err := os.Create(destPath)
	if err != nil {
		return "", fmt.Errorf("failed to create destination file: %w", err)
	}
	defer destFile.Close()

	// Copy file contents
	_, err = io.Copy(destFile, file)
	if err != nil {
		return "", fmt.Errorf("failed to copy file contents: %w", err)
	}

	return destPath, nil
}

// analyzeFiles analyzes uploaded files to extract metadata
func (s *DatasetService) analyzeFiles(files models.DatasetFiles) (models.DatasetMetadata, error) {
	metadata := models.DatasetMetadata{}

	// Count nodes from properties file
	nodeCount, err := s.countLinesInFile(files.PropertiesFile)
	if err != nil {
		log.Warn().Err(err).Msg("Failed to count nodes from properties file")
	} else {
		metadata.NodeCount = nodeCount
	}

	// Count edges from graph file
	edgeCount, err := s.countLinesInFile(files.GraphFile)
	if err != nil {
		log.Warn().Err(err).Msg("Failed to count edges from graph file")
	} else {
		metadata.EdgeCount = edgeCount
	}

	// Calculate total file size
	totalSize := int64(0)
	for _, filePath := range []string{files.GraphFile, files.PropertiesFile, files.PathFile} {
		if stat, err := os.Stat(filePath); err == nil {
			totalSize += stat.Size()
		}
	}
	metadata.FileSize = totalSize

	return metadata, nil
}

// countLinesInFile counts non-empty, non-comment lines in a file
func (s *DatasetService) countLinesInFile(filePath string) (int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	count := 0
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			count++
		}
	}

	return count, scanner.Err()
}
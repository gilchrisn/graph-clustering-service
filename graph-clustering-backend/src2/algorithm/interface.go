package algorithm

import (
	"context"

	"graph-viz-backend/models"
)

// Algorithm defines the interface for clustering algorithms
type Algorithm interface {
	// Name returns the algorithm name
	Name() models.AlgorithmType
	
	// ValidateParameters validates the job parameters for this algorithm
	ValidateParameters(params models.JobParameters) error
	
	// Run executes the algorithm with given parameters
	Run(ctx context.Context, dataset *models.Dataset, params models.JobParameters, progressCb ProgressCallback) (*AlgorithmResult, error)
}

// ProgressCallback is called during algorithm execution to report progress
type ProgressCallback func(percentage int, message string)

// AlgorithmResult contains the raw result from algorithm execution
type AlgorithmResult struct {
	// Raw result from the algorithm API (*louvain.Result or *scar.Result)
	RawResult interface{}
	
	// Processed hierarchy data for frontend
	Hierarchy *models.Hierarchy
	
	// Algorithm execution metadata
	ProcessingTimeMS int64
	Modularity       float64
	NumLevels        int
	NumCommunities   int
	
	// Additional algorithm-specific statistics
	Statistics map[string]interface{}
}

// Registry manages available algorithms
type Registry struct {
	algorithms map[models.AlgorithmType]Algorithm
}

// NewRegistry creates a new algorithm registry
func NewRegistry() *Registry {
	registry := &Registry{
		algorithms: make(map[models.AlgorithmType]Algorithm),
	}
	
	// Register available algorithms
	registry.Register(NewLouvainAdapter())
	registry.Register(NewSCARAdapter())
	
	return registry
}

// Register adds an algorithm to the registry
func (r *Registry) Register(alg Algorithm) {
	r.algorithms[alg.Name()] = alg
}

// Get retrieves an algorithm by name
func (r *Registry) Get(name models.AlgorithmType) (Algorithm, bool) {
	alg, exists := r.algorithms[name]
	return alg, exists
}

// List returns all available algorithm names
func (r *Registry) List() []models.AlgorithmType {
	names := make([]models.AlgorithmType, 0, len(r.algorithms))
	for name := range r.algorithms {
		names = append(names, name)
	}
	return names
}
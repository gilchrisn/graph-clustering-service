package models

import (
	"time"
)

// Dataset represents an uploaded heterogeneous graph dataset
type Dataset struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Status      DatasetStatus     `json:"status"`
	Files       DatasetFiles      `json:"files"`
	Metadata    DatasetMetadata   `json:"metadata"`
	CreatedAt   time.Time         `json:"createdAt"`
	UpdatedAt   time.Time         `json:"updatedAt"`
}

type DatasetStatus string

const (
    // Resource integrity states (immutable once uploaded)
    DatasetStatusReady      DatasetStatus = "ready"      // ✅ Available for experiments
    DatasetStatusCorrupted  DatasetStatus = "corrupted"  // ❌ Data integrity issue
    DatasetStatusDeleted    DatasetStatus = "deleted"    // 🗑️ Marked for deletion
)
type DatasetFiles struct {
	GraphFile      string `json:"graphFile"`
	PropertiesFile string `json:"propertiesFile"`
	PathFile       string `json:"pathFile"`
}

type DatasetMetadata struct {
	NodeCount int   `json:"nodeCount"`
	EdgeCount int   `json:"edgeCount"`
	FileSize  int64 `json:"fileSize"`
}

// Job represents a clustering job
type Job struct {
	ID           string         `json:"id"`
	DatasetID    string         `json:"datasetId"`
	Algorithm    AlgorithmType  `json:"algorithm"`
	Parameters   JobParameters  `json:"parameters"`
	Status       JobStatus      `json:"status"`
	Progress     JobProgress    `json:"progress"`
	Result       *JobResult     `json:"result,omitempty"`
	Error        string         `json:"error,omitempty"`
	CreatedAt    time.Time      `json:"createdAt"`
	UpdatedAt    time.Time      `json:"updatedAt"`
	StartedAt    *time.Time     `json:"startedAt,omitempty"`
	CompletedAt  *time.Time     `json:"completedAt,omitempty"`
}

type AlgorithmType string

const (
	AlgorithmLouvain AlgorithmType = "louvain"
	AlgorithmSCAR    AlgorithmType = "scar"
)

type JobParameters struct {
	// Louvain parameters
	MaxLevels         *int     `json:"maxLevels,omitempty"`
	MaxIterations     *int     `json:"maxIterations,omitempty"`
	MinModularityGain *float64 `json:"minModularityGain,omitempty"`
	
	// SCAR parameters
	K         *int64   `json:"k,omitempty"`
	NK        *int64   `json:"nk,omitempty"`
	Threshold *float64 `json:"threshold,omitempty"`

	ReconstructionThreshold *float64 `json:"reconstructionThreshold,omitempty"` // 0.0-1.0, controls edge inclusion
	ReconstructionMode      *string  `json:"reconstructionMode,omitempty"`      // "inclusion_exclusion" or "full"
	EdgeWeightNormalization *bool    `json:"edgeWeightNormalization,omitempty"` // normalize weights during reconstruction
}

type JobStatus string

const (
	JobStatusQueued    JobStatus = "queued"
	JobStatusRunning   JobStatus = "running"
	JobStatusCompleted JobStatus = "completed"
	JobStatusFailed    JobStatus = "failed"
	JobStatusCancelled JobStatus = "cancelled"
)

type JobProgress struct {
	Percentage int    `json:"percentage"`
	Message    string `json:"message"`
}

type JobResult struct {
	Modularity       float64               `json:"modularity"`
	NumLevels        int                   `json:"numLevels"`
	NumCommunities   int                   `json:"numCommunities"`
	ProcessingTimeMS int64                 `json:"processingTimeMS"`
	Statistics       map[string]interface{} `json:"statistics,omitempty"`
}

// Hierarchy represents the hierarchical clustering result
type Hierarchy struct {
	DatasetID   string                    `json:"datasetId"`
	JobID       string                    `json:"jobId"`
	Algorithm   AlgorithmType             `json:"algorithm"`
	Levels      []HierarchyLevel         `json:"levels"`
	RootNode    string                   `json:"rootNode"`
	Coordinates map[string]NodePosition  `json:"coordinates"`
}

type HierarchyLevel struct {
	Level       int                      `json:"level"`
	Communities map[string][]string      `json:"communities"` // communityID -> nodeIDs
	ParentMap   map[string]string        `json:"parentMap"`   // childID -> parentID
}

type NodePosition struct {
	X      float64 `json:"x"`
	Y      float64 `json:"y"`
	Radius float64 `json:"radius"`
}

// API Response types
type APIResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

type UploadResponse struct {
	DatasetID string `json:"datasetId"`
	Dataset   Dataset `json:"dataset"`
}

type ClusteringResponse struct {
	JobID string `json:"jobId"`
	Job   Job    `json:"job"`
}

type HierarchyResponse struct {
	Hierarchy Hierarchy `json:"hierarchy"`
}

type ClusterResponse struct {
	ClusterID string              `json:"clusterId"`
	Level     int                 `json:"level"`
	Nodes     []NodeInfo          `json:"nodes"`
	Edges     []EdgeInfo          `json:"edges"`
}

type NodeInfo struct {
	ID       string      `json:"id"`
	Label    string      `json:"label"`
	Position NodePosition `json:"position"`
	Type     string      `json:"type"` // "leaf" or "supernode"
	Metadata map[string]interface{} `json:"metadata"`
}

type EdgeInfo struct {
	Source string  `json:"source"`
	Target string  `json:"target"`
	Weight float64 `json:"weight"`
}

// WebSocket message types for real-time updates
type WSMessage struct {
	Type string      `json:"type"`
	Data interface{} `json:"data"`
}

type ProgressUpdate struct {
	JobID      string `json:"jobId"`
	Percentage int    `json:"percentage"`
	Message    string `json:"message"`
}

type ParameterChange struct {
	JobID      string        `json:"jobId"`
	Parameters JobParameters `json:"parameters"`
}

type CompletionNotification struct {
	JobID  string     `json:"jobId"`
	Result *JobResult `json:"result"`
}

// Comparison represents a comparison between two experiments
type Comparison struct {
	ID           string              `json:"id"`
	Name         string              `json:"name"`
	ExperimentA  ExperimentRef       `json:"experimentA"`
	ExperimentB  ExperimentRef       `json:"experimentB"`
	Metrics      []string            `json:"metrics"`
	Options      ComparisonOptions   `json:"options"`
	Status       ComparisonStatus    `json:"status"`
	Result       *ComparisonResult   `json:"result,omitempty"`
	Error        string              `json:"error,omitempty"`
	CreatedAt    time.Time           `json:"createdAt"`
	CompletedAt  *time.Time          `json:"completedAt,omitempty"`
}

// ExperimentRef references a completed clustering experiment
type ExperimentRef struct {
	DatasetID string `json:"datasetId"`
	JobID     string `json:"jobId"`
}

// ComparisonOptions configures how comparison is performed
type ComparisonOptions struct {
	LevelWise              bool `json:"levelWise"`
	IncludeVisualization  bool `json:"includeVisualization"`
}

// ComparisonStatus represents the status of a comparison
type ComparisonStatus string

const (
	ComparisonStatusRunning   ComparisonStatus = "running"
	ComparisonStatusCompleted ComparisonStatus = "completed"
	ComparisonStatusFailed    ComparisonStatus = "failed"
)

// ComparisonResult contains the computed metrics
type ComparisonResult struct {
	AGDS     *float64                `json:"agds,omitempty"`     // Position similarity
	HMI      *float64                `json:"hmi,omitempty"`      // Hierarchical mutual information
	Jaccard  *float64                `json:"jaccard,omitempty"`  // Community overlap
	ARI      *float64                `json:"ari,omitempty"`      // Adjusted rand index
	LevelWise []ComparisonLevelResult `json:"levelWise,omitempty"` // Per-level metrics
	Summary   ComparisonSummary       `json:"summary"`
}

// ComparisonLevelResult contains metrics for a specific hierarchy level
type ComparisonLevelResult struct {
	Level   int     `json:"level"`
	Jaccard float64 `json:"jaccard"`
	ARI     float64 `json:"ari"`
}

// ComparisonSummary provides human-readable analysis
type ComparisonSummary struct {
	OverallSimilarity string   `json:"overallSimilarity"` // "High", "Medium", "Low"
	KeyDifferences    []string `json:"keyDifferences"`
	Recommendation    string   `json:"recommendation"`
}

// API Request/Response types for comparisons
type CreateComparisonRequest struct {
	Name        string            `json:"name"`
	ExperimentA ExperimentRef     `json:"experimentA"`
	ExperimentB ExperimentRef     `json:"experimentB"`
	Metrics     []string          `json:"metrics"`
	Options     ComparisonOptions `json:"options"`
}

type ComparisonResponse struct {
	ComparisonID string     `json:"comparisonId"`
	Comparison   Comparison `json:"comparison"`
}

// MultiComparisonRequest represents a request to compare multiple experiments
type MultiComparisonRequest struct {
	Name                string            `json:"name"`
	SelectedExperiments []ExperimentRef   `json:"selectedExperiments"`
	BaselineExperiment  ExperimentRef     `json:"baselineExperiment"`  
	Metrics             []string          `json:"metrics"`
	Options             ComparisonOptions `json:"options"`
}

// MultiComparisonResult contains results from comparing multiple experiments
type MultiComparisonResult struct {
	BaselineJobID   string              `json:"baselineJobId"`
	BaselineConfig  BaselineInfo        `json:"baselineConfig"`  // Transparency: show what baseline was used
	Experiments     []ExperimentResult  `json:"experiments"`
}

// BaselineInfo provides transparency about the baseline used
type BaselineInfo struct {
	Algorithm    string             `json:"algorithm"`     // Always "louvain" 
	Parameters   map[string]interface{} `json:"parameters"`    // Standard params used
	Description  string             `json:"description"`   // "Standard Louvain Baseline"
	IsStandard   bool              `json:"isStandard"`    // Always true for our baselines
}

// ExperimentResult contains metrics for a single experiment in multi-comparison
type ExperimentResult struct {
	JobID   string             `json:"jobId"`
	Label   string             `json:"label"` // "SCAR k=64", "Louvain default", etc.
	Metrics map[string]float64 `json:"metrics"`
}

// VisualizationMetric contains custom visualization similarity metrics
type VisualizationMetric struct {
	JaccardSimilarity float64 `json:"jaccardSimilarity"` // Node set similarity
	AMQMDisplacement  float64 `json:"amqmDisplacement"`  // AM/QM ratio for position displacements
	AMQMRadius        float64 `json:"amqmRadius"`        // AM/QM ratio for radius differences
	CompositeScore    float64 `json:"compositeScore"`    // jaccard × displacement × radius
}
package services

import (
    "path/filepath"
    "github.com/gilchrisn/graph-clustering-service/pkg/materialization"
    "github.com/gilchrisn/graph-clustering-service/pkg/louvain"
    "github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

// ===== PIPELINE CONFIGURATION AND PATHS =====

// Pipeline configuration and paths
type PipelinePaths struct {
    BaseDir          string
    ClusteringDir    string  
    HierarchyDir     string
    VisualizationDir string
    Algorithm        string
}

// Get paths for specific algorithm
func GetPipelineOutputPaths(datasetId, algorithm string) PipelinePaths {
    baseDir := filepath.Join("pipeline_output", datasetId)
    return PipelinePaths{
        BaseDir:          baseDir,
        ClusteringDir:    filepath.Join(baseDir, algorithm, "clustering"),
        HierarchyDir:     filepath.Join(baseDir, algorithm, "hierarchy"),
        VisualizationDir: filepath.Join(baseDir, algorithm, "visualization"),
        Algorithm:        algorithm,
    }
}

// Algorithm-specific file patterns
func (p PipelinePaths) GetClusteringFiles() ClusteringFiles {
    if p.Algorithm == "materialization" {
        return ClusteringFiles{
            EdgeFile:     filepath.Join(p.ClusteringDir, "materialized_graph.edgelist"),
            MappingFile:  filepath.Join(p.ClusteringDir, "communities.mapping"),
            HierarchyFile: filepath.Join(p.ClusteringDir, "communities.hierarchy"),
            RootFile:     filepath.Join(p.ClusteringDir, "communities.root"),
        }
    } else { // scar
        return ClusteringFiles{
            EdgeFile:     filepath.Join(p.ClusteringDir, "communities.sketch"),
            MappingFile:  filepath.Join(p.ClusteringDir, "communities_mapping.dat"),
            HierarchyFile: filepath.Join(p.ClusteringDir, "communities_hierarchy.dat"),
            RootFile:     filepath.Join(p.ClusteringDir, "communities_root.dat"),
        }
    }
}

func (p PipelinePaths) GetVisualizationFile() string {
    return filepath.Join(p.VisualizationDir, "levels.json")
}

// File structure for clustering outputs
type ClusteringFiles struct {
    EdgeFile      string  // .edgelist for materialization, .sketch for SCAR
    MappingFile   string  // communities.mapping or communities_mapping.dat
    HierarchyFile string  // communities.hierarchy or communities_hierarchy.dat  
    RootFile      string  // communities.root or communities_root.dat
}

// ===== PIPELINE CONFIGURATION FROM YOUR CODE =====

// PipelineConfig holds configuration for both pipeline types
type PipelineConfig struct {
    // Common options
    Verbose      bool
    OutputDir    string
    OutputPrefix string
    
    // Materialization + Louvain config
    MaterializationConfig materialization.MaterializationConfig
    LouvainConfig        louvain.LouvainConfig
    
    // SCAR config
    SCARConfig scar.SCARConfig
}

func NewPipelineConfig() *PipelineConfig {
    return &PipelineConfig{
        Verbose:      true,
        OutputPrefix: "communities",
        
        // Default materialization + Louvain
        MaterializationConfig: materialization.DefaultMaterializationConfig(),
        LouvainConfig:        louvain.DefaultLouvainConfig(),
        
        // Default SCAR config
        SCARConfig: scar.SCARConfig{
            K:           1024,
            NK:          4,
            Threshold:   0.5,
            UseLouvain:  true,
            SketchOutput: true,
            WriteSketchGraph: true,
            SketchGraphWeights: false,
        },
    }
}

// ===== PIPELINE RESULT TYPES FROM YOUR CODE =====

// PipelineType defines which pipeline to run
type PipelineType int

const (
    MaterializationLouvain PipelineType = iota
    SketchLouvain
    Comparison
)

// PipelineResult contains results from either pipeline
type PipelineResult struct {
    PipelineType    PipelineType
    TotalRuntimeMS  int64
    
    // Materialization + Louvain results (nil if SketchLouvain was used)
    MaterializedGraph *materialization.HomogeneousGraph
    LouvainResult     *louvain.LouvainResult
    
    // SCAR results (basic info - actual files written to disk)
    SCARSuccess bool
    SCARConfig  *scar.SCARConfig
}

// ===== VISUALIZATION TYPES FROM YOUR CODE =====

// NodeViz from your pipeline code
type NodeViz struct {
    ID       string  `json:"id"`
    PageRank float64 `json:"pagerank"`
    X        float64 `json:"x"`
    Y        float64 `json:"y"`
    Radius   float64 `json:"radius"`
    Label    string  `json:"label"`
}

// LevelViz from your pipeline code
type LevelViz struct {
    Level int       `json:"level"`
    Nodes []NodeViz `json:"nodes"`
}

// Edge from your pipeline code
type Edge struct {
    From   string
    To     string
    Weight float64
}

// ===== BACKEND API RESPONSE TYPES (KEEP EXISTING) =====

// Existing types - keep for API compatibility
type ProcessResult struct {
    Success        bool                   `json:"success"`
    Message        string                 `json:"message"`
    DatasetId      string                 `json:"datasetId"`
    RootNode       string                 `json:"rootNode"`        // ← ADD THIS
    AlgorithmId    string                 `json:"algorithmId"`     // ← ADD THIS
    Parameters     map[string]interface{} `json:"parameters"`      // ← ADD THIS
    NodeCount      int                    `json:"nodeCount"`       // ← ADD THIS
    EdgeCount      int                    `json:"edgeCount"`       // ← ADD THIS
    ProcessingTime float64                `json:"processingTime"`  // ← ADD THIS
    K              int                    `json:"k,omitempty"`
    FilePath       string                 `json:"filePath,omitempty"`
    ProcessingType string                 `json:"processingType,omitempty"`
}

// Response types that controllers expect
type ProcessResponse struct {
    Success bool   `json:"success"`
    Message string `json:"message"`
    Error   string `json:"error,omitempty"`
    Result  *ProcessResult `json:"result,omitempty"`
}

type HierarchyResponse struct {
    Success   bool                   `json:"success,omitempty"`
    Message   string                 `json:"message,omitempty"`
    Error     string                 `json:"error,omitempty"`
    Hierarchy map[string]interface{} `json:"hierarchy,omitempty"`
    Mapping   map[string]interface{} `json:"mapping,omitempty"`
}

type CoordinatesResponse struct {
    Success bool                     `json:"success,omitempty"`
    Message string                   `json:"message,omitempty"`
    Error   string                   `json:"error,omitempty"`
    Nodes   []interface{}            `json:"nodes,omitempty"`
    Edges   []interface{}            `json:"edges,omitempty"`
	}

type StatisticsResponse struct {
    Success    bool                   `json:"success,omitempty"`
    Message    string                 `json:"message,omitempty"`
    Error      string                 `json:"error,omitempty"`
    Statistics map[string]interface{} `json:"statistics,omitempty"`
}

type ComparisonResponse struct {
    Success    bool               `json:"success"`
    Message    string             `json:"message"`
    Error      string             `json:"error,omitempty"`
    Comparison *ComparisonResult  `json:"comparison,omitempty"`
}

// Internal data types for pipeline processing
type HierarchyData struct {
    Communities map[string][]string `json:"communities"`
    Levels      map[int][]string    `json:"levels"`
    RootNode    string              `json:"rootNode"`
}

type MappingData struct {
    NodeToCluster map[string]string `json:"nodeToCluster"`
    ClusterToNodes map[string][]string `json:"clusterToNodes"`
}

type CoordinatesData struct {
    Coordinates map[string][2]float64 `json:"coordinates"`
    Metadata    map[string]interface{} `json:"metadata"`
}

// ===== COMPARISON RESULT TYPES =====

// Comparison results for the /compare endpoint
type ComparisonResult struct {
    Heterogeneous ComparisonAlgorithmResult `json:"heterogeneous"`
    SCAR         ComparisonAlgorithmResult `json:"scar"`
    Metrics      ComparisonMetrics         `json:"metrics"`
}

type ComparisonAlgorithmResult struct {
    DatasetId     string                 `json:"datasetId"`
    HierarchyData HierarchyData         `json:"hierarchyData"`
    MappingData   MappingData           `json:"mappingData"`
    RootNode      string                `json:"rootNode"`
    Parameters    map[string]interface{} `json:"parameters"`
}

type ComparisonMetrics struct {
    NMI           float64                `json:"nmi"`
    ClusterCounts map[string]int         `json:"clusterCounts"`
    Similarity    string                 `json:"similarity"`
    Details       map[string]interface{} `json:"details"`
}
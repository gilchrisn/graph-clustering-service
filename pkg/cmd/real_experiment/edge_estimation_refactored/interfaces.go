package main

import "math/rand"

// ============================================================================
// COMMON INTERFACES
// ============================================================================

// Graph represents a graph structure
type Graph struct {
    NumNodes    int
    Edges       map[int]map[int]bool
    Communities [][]int
}

// SketchEstimator defines the interface for all estimation methods
type SketchEstimator interface {
    // BuildGraph creates the sketch representation of the graph
    BuildGraph(graph *Graph, rng *rand.Rand) error
    
    // EstimateEdges estimates the number of edges between two communities
    EstimateEdges(commA, commB []int) float64
    
    // GetMethodName returns a descriptive name for this method
    GetMethodName() string
    
    // GetParameters returns a map of parameter names to values
    GetParameters() map[string]interface{}
}

// ExperimentConfig holds configuration for running experiments
type ExperimentConfig struct {
    // Graph parameters
    NumNodes       int     `json:"num_nodes"`
    EdgeProb       float64 `json:"edge_prob,omitempty"`
    NumCommunities int     `json:"num_communities"`
    
    // For planted partition
    PIntra         float64 `json:"p_intra,omitempty"`
    PInter         float64 `json:"p_inter,omitempty"`
    
    // Experiment parameters
    GraphType      string  `json:"graph_type"`
    Repetitions    int     `json:"repetitions"`
    Seed           int64   `json:"seed"`
    
    // Method-specific parameters will be handled by individual estimators
}

// ComparisonResult holds the results of comparing methods on a community pair
type ComparisonResult struct {
    CommunityA    []int              `json:"community_a"`
    CommunityB    []int              `json:"community_b"`
    TrueWeight    float64            `json:"true_weight"`
    Estimates     map[string]float64 `json:"estimates"`     // method_name -> estimate
    Errors        map[string]float64 `json:"errors"`        // method_name -> absolute error
    
    // Context for analysis
    CommunityASize   int     `json:"community_a_size"`
    CommunityBSize   int     `json:"community_b_size"`
    EdgeMultiplicity float64 `json:"edge_multiplicity"`
    LocalDensity     float64 `json:"local_density"`
}

// ExperimentResult holds results for a single experiment configuration
type ExperimentResult struct {
    Config      ExperimentConfig    `json:"config"`
    Methods     []string           `json:"methods"`        // List of method names tested
    Comparisons []ComparisonResult `json:"comparisons"`
    Summary     Summary            `json:"summary"`
    RuntimeMS   int64              `json:"runtime_ms"`
}

// Summary holds aggregate statistics across all comparisons
type Summary struct {
    MethodMAEs           map[string]float64 `json:"method_maes"`           // method_name -> MAE
    MethodMaxErrors      map[string]float64 `json:"method_max_errors"`     // method_name -> max error
    DensityCorrelations  map[string]float64 `json:"density_correlations"`  // method_name -> correlation with density
    TotalComparisons     int                `json:"total_comparisons"`
    
    // Cross-method correlations
    MethodCorrelations   map[string]map[string]float64 `json:"method_correlations"` // method1 -> method2 -> correlation
}
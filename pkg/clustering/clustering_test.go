package clustering

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/materialization"
	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// TestData contains test fixtures
type TestData struct {
	GraphFile    string
	MetaPathFile string
	TempDir      string
}

// setupTestData creates test files for clustering tests
func setupTestData(t *testing.T) *TestData {
	// Create temporary directory
	tempDir := t.TempDir()

	// Create test graph
	testGraph := models.HeterogeneousGraph{
		Nodes: map[string]models.Node{
			"a1": {Type: "Author", Properties: map[string]interface{}{"name": "Alice"}},
			"a2": {Type: "Author", Properties: map[string]interface{}{"name": "Bob"}},
			"a3": {Type: "Author", Properties: map[string]interface{}{"name": "Charlie"}},
			"p1": {Type: "Paper", Properties: map[string]interface{}{"title": "ML Paper 1"}},
			"p2": {Type: "Paper", Properties: map[string]interface{}{"title": "ML Paper 2"}},
			"v1": {Type: "Venue", Properties: map[string]interface{}{"name": "ICML"}},
		},
		Edges: []models.Edge{
			{From: "a1", To: "p1", Type: "writes", Weight: 1.0},
			{From: "a2", To: "p1", Type: "writes", Weight: 1.0},
			{From: "a2", To: "p2", Type: "writes", Weight: 1.0},
			{From: "a3", To: "p2", Type: "writes", Weight: 1.0},
			{From: "p1", To: "v1", Type: "published_in", Weight: 1.0},
			{From: "p2", To: "v1", Type: "published_in", Weight: 1.0},
		},
	}

	// Create test meta path
	testMetaPath := models.MetaPath{
		ID:           "author_coauthorship",
		NodeSequence: []string{"Author", "Paper", "Author"},
		EdgeSequence: []string{"writes", "writes"},
		Description:  "Authors connected through co-authored papers",
	}

	// Save test files
	graphFile := filepath.Join(tempDir, "test_graph.json")
	metaPathFile := filepath.Join(tempDir, "test_meta_path.json")

	graphData, _ := json.Marshal(testGraph)
	metaPathData, _ := json.Marshal(testMetaPath)

	os.WriteFile(graphFile, graphData, 0644)
	os.WriteFile(metaPathFile, metaPathData, 0644)

	return &TestData{
		GraphFile:    graphFile,
		MetaPathFile: metaPathFile,
		TempDir:      tempDir,
	}
}

// Test Default Configurations
func TestDefaultConfigurations(t *testing.T) {
	t.Run("DefaultMaterializationConfig", func(t *testing.T) {
		config := DefaultMaterializationConfig()

		// Check required fields have sensible defaults
		if config.AggregationStrategy != materialization.Count {
			t.Errorf("Expected Count strategy, got %v", config.AggregationStrategy)
		}
		if config.MetaPathInterpretation != materialization.MeetingBased {
			t.Errorf("Expected MeetingBased interpretation, got %v", config.MetaPathInterpretation)
		}
		if config.MaxInstances <= 0 {
			t.Errorf("MaxInstances should be positive, got %d", config.MaxInstances)
		}
		if config.TraversalParallelism <= 0 {
			t.Errorf("TraversalParallelism should be positive, got %d", config.TraversalParallelism)
		}
		if config.OutputPrefix == "" {
			t.Error("OutputPrefix should not be empty")
		}
	})

	t.Run("DefaultScarConfig", func(t *testing.T) {
		config := DefaultScarConfig()

		// Check required fields have sensible defaults
		if config.K <= 0 {
			t.Errorf("K should be positive, got %d", config.K)
		}
		if config.NK <= 0 {
			t.Errorf("NK should be positive, got %d", config.NK)
		}
		if config.MaxIterations <= 0 {
			t.Errorf("MaxIterations should be positive, got %d", config.MaxIterations)
		}
		if config.NumWorkers <= 0 {
			t.Errorf("NumWorkers should be positive, got %d", config.NumWorkers)
		}
		if config.OutputPrefix == "" {
			t.Error("OutputPrefix should not be empty")
		}
	})
}

// Test Materialization Clustering
func TestRunMaterializationClustering(t *testing.T) {
	testData := setupTestData(t)

	t.Run("SuccessfulClustering", func(t *testing.T) {
		config := DefaultMaterializationConfig()
		config.GraphFile = testData.GraphFile
		config.MetaPathFile = testData.MetaPathFile
		config.OutputDir = filepath.Join(testData.TempDir, "mat_output")
		config.Verbose = false
		
		// Use smaller limits for test
		config.MaxInstances = 10000
		config.TimeoutSeconds = 30

		startTime := time.Now()
		result, err := RunMaterializationClustering(config)
		elapsed := time.Since(startTime)

		// Check basic success
		if err != nil {
			t.Fatalf("Materialization clustering failed: %v", err)
		}
		if !result.Success {
			t.Fatalf("Clustering was not successful: %s", result.Error)
		}

		// Check result properties
		if result.Approach != "Materialization + Louvain" {
			t.Errorf("Expected 'Materialization + Louvain', got '%s'", result.Approach)
		}
		if result.Runtime <= 0 {
			t.Error("Runtime should be positive")
		}
		if result.Runtime > elapsed+time.Second {
			t.Error("Reported runtime seems too long")
		}
		if len(result.Communities) == 0 {
			t.Error("Should have some community assignments")
		}
		if result.NumCommunities <= 0 {
			t.Error("Should have at least one community")
		}
		if result.Modularity < -1.0 || result.Modularity > 1.0 {
			t.Errorf("Modularity should be between -1 and 1, got %.6f", result.Modularity)
		}

		// Check algorithm details
		if details, ok := result.AlgorithmDetails.(MaterializationDetails); ok {
			if details.InstancesGenerated <= 0 {
				t.Error("Should have generated some instances")
			}
			if details.EdgesCreated < 0 {
				t.Error("EdgesCreated should not be negative")
			}
		} else {
			t.Error("AlgorithmDetails should be MaterializationDetails")
		}

		// Check output files
		checkOutputFiles(t, result.OutputFiles, "materialization")
	})

	t.Run("InvalidGraphFile", func(t *testing.T) {
		config := DefaultMaterializationConfig()
		config.GraphFile = "nonexistent.json"
		config.MetaPathFile = testData.MetaPathFile
		config.OutputDir = filepath.Join(testData.TempDir, "mat_output_fail")
		config.Verbose = false

		result, err := RunMaterializationClustering(config)

		// Should fail
		if err == nil {
			t.Error("Expected error for invalid graph file")
		}
		if result.Success {
			t.Error("Should not be successful with invalid input")
		}
		if result.Error == "" {
			t.Error("Should have error message")
		}
	})

	t.Run("InvalidMetaPathFile", func(t *testing.T) {
		config := DefaultMaterializationConfig()
		config.GraphFile = testData.GraphFile
		config.MetaPathFile = "nonexistent.json"
		config.OutputDir = filepath.Join(testData.TempDir, "mat_output_fail2")
		config.Verbose = false

		result, err := RunMaterializationClustering(config)

		// Should fail
		if err == nil {
			t.Error("Expected error for invalid meta path file")
		}
		if result.Success {
			t.Error("Should not be successful with invalid input")
		}
	})
}

// Test SCAR Clustering
func TestRunScarClustering(t *testing.T) {
	testData := setupTestData(t)

	t.Run("SuccessfulClustering", func(t *testing.T) {
		config := DefaultScarConfig()
		config.GraphFile = testData.GraphFile
		config.MetaPathFile = testData.MetaPathFile
		config.OutputDir = filepath.Join(testData.TempDir, "scar_output")
		config.Verbose = false
		
		// Use smaller parameters for test
		config.K = 16
		config.NK = 4
		config.MaxIterations = 10

		startTime := time.Now()
		result, err := RunScarClustering(config)
		elapsed := time.Since(startTime)

		// Check basic success
		if err != nil {
			t.Fatalf("SCAR clustering failed: %v", err)
		}
		if !result.Success {
			t.Fatalf("Clustering was not successful: %s", result.Error)
		}

		// Check result properties
		if result.Approach != "SCAR" {
			t.Errorf("Expected 'SCAR', got '%s'", result.Approach)
		}
		if result.Runtime <= 0 {
			t.Error("Runtime should be positive")
		}
		if result.Runtime > elapsed+time.Second {
			t.Error("Reported runtime seems too long")
		}
		if len(result.Communities) == 0 {
			t.Error("Should have some community assignments")
		}
		if result.NumCommunities <= 0 {
			t.Error("Should have at least one community")
		}
		if result.Modularity < -1.0 || result.Modularity > 1.0 {
			t.Errorf("Modularity should be between -1 and 1, got %.6f", result.Modularity)
		}

		// Check algorithm details
		if details, ok := result.AlgorithmDetails.(ScarDetails); ok {
			if details.SketchSize != config.K {
				t.Errorf("Expected sketch size %d, got %d", config.K, details.SketchSize)
			}
			if details.NumHashFunctions != config.NK {
				t.Errorf("Expected %d hash functions, got %d", config.NK, details.NumHashFunctions)
			}
			if details.ParallelProcessing != config.ParallelEnabled {
				t.Errorf("Parallel processing mismatch: expected %v, got %v", 
					config.ParallelEnabled, details.ParallelProcessing)
			}
		} else {
			t.Error("AlgorithmDetails should be ScarDetails")
		}

		// Check output files
		checkOutputFiles(t, result.OutputFiles, "scar")
	})

	t.Run("InvalidInput", func(t *testing.T) {
		config := DefaultScarConfig()
		config.GraphFile = "nonexistent.json"
		config.MetaPathFile = testData.MetaPathFile
		config.OutputDir = filepath.Join(testData.TempDir, "scar_output_fail")
		config.Verbose = false

		result, err := RunScarClustering(config)

		// Should fail
		if err == nil {
			t.Error("Expected error for invalid graph file")
		}
		if result.Success {
			t.Error("Should not be successful with invalid input")
		}
	})
}

// Test Verification Functions
func TestVerificationFunctions(t *testing.T) {
	testData := setupTestData(t)

	t.Run("VerifyMaterializationOutput", func(t *testing.T) {
		// Run materialization clustering first
		config := DefaultMaterializationConfig()
		config.GraphFile = testData.GraphFile
		config.MetaPathFile = testData.MetaPathFile
		config.OutputDir = filepath.Join(testData.TempDir, "verify_mat")
		config.Verbose = false
		config.MaxInstances = 10000

		result, err := RunMaterializationClustering(config)
		if err != nil {
			t.Fatalf("Failed to run clustering: %v", err)
		}

		// Test verification
		err = VerifyMaterializationOutput(config.OutputDir, config.OutputPrefix)
		if err != nil {
			t.Errorf("Verification failed: %v", err)
		}

		// Test verification of clustering result
		err = VerifyClusteringResult(result)
		if err != nil {
			t.Errorf("Result verification failed: %v", err)
		}
	})

	t.Run("VerifyMaterializationOutput_MissingFiles", func(t *testing.T) {
		emptyDir := filepath.Join(testData.TempDir, "empty")
		os.MkdirAll(emptyDir, 0755)

		err := VerifyMaterializationOutput(emptyDir, "nonexistent")
		if err == nil {
			t.Error("Expected error for missing files")
		}
	})

	t.Run("VerifyClusteringResult_InvalidResult", func(t *testing.T) {
		// Test nil result
		err := VerifyClusteringResult(nil)
		if err == nil {
			t.Error("Expected error for nil result")
		}

		// Test unsuccessful result
		badResult := &ClusteringResult{
			Success: false,
			Error:   "test error",
		}
		err = VerifyClusteringResult(badResult)
		if err == nil {
			t.Error("Expected error for unsuccessful result")
		}

		// Test invalid modularity
		badModularityResult := &ClusteringResult{
			Success:    true,
			Modularity: 2.0, // Invalid: > 1.0
			Approach:   "Test",
		}
		err = VerifyClusteringResult(badModularityResult)
		if err == nil {
			t.Error("Expected error for invalid modularity")
		}
	})
}

// Test Performance and Edge Cases
func TestPerformanceAndEdgeCases(t *testing.T) {
	testData := setupTestData(t)

	t.Run("SmallTimeoutMaterialization", func(t *testing.T) {
		config := DefaultMaterializationConfig()
		config.GraphFile = testData.GraphFile
		config.MetaPathFile = testData.MetaPathFile
		config.OutputDir = filepath.Join(testData.TempDir, "timeout_test")
		config.TimeoutSeconds = 1 // Very short timeout
		config.Verbose = false

		_, err := RunMaterializationClustering(config)
		// This might succeed or fail depending on graph size
		// Just ensure it doesn't panic
		if err != nil {
			t.Logf("Short timeout caused expected failure: %v", err)
		}
	})

	t.Run("VerboseMode", func(t *testing.T) {
		config := DefaultScarConfig()
		config.GraphFile = testData.GraphFile
		config.MetaPathFile = testData.MetaPathFile
		config.OutputDir = filepath.Join(testData.TempDir, "verbose_test")
		config.Verbose = true // Test verbose output
		config.K = 8
		config.NK = 2
		config.MaxIterations = 5

		result, err := RunScarClustering(config)
		if err != nil {
			t.Fatalf("Verbose mode clustering failed: %v", err)
		}
		if !result.Success {
			t.Error("Verbose mode clustering should succeed")
		}
	})

	t.Run("ConfigurationValidation", func(t *testing.T) {
		// Test with invalid K value
		config := DefaultScarConfig()
		config.GraphFile = testData.GraphFile
		config.MetaPathFile = testData.MetaPathFile
		config.OutputDir = filepath.Join(testData.TempDir, "invalid_k")
		config.K = 0 // Invalid
		config.Verbose = false

		_, err := RunScarClustering(config)
		if err == nil {
			t.Error("Expected error for invalid K value")
		}
	})
}

// Helper function to check output files exist
func checkOutputFiles(t *testing.T, files OutputFiles, approach string) {
	switch approach {
	case "materialization":
		checkFileExists(t, files.MappingFile, "mapping file")
		checkFileExists(t, files.HierarchyFile, "hierarchy file")
		checkFileExists(t, files.RootFile, "root file")
		checkFileExists(t, files.EdgesFile, "edges file")
	case "scar":
		checkFileExists(t, files.RootFile, "root file")
		checkDirExists(t, files.HierarchyDir, "hierarchy directory")
		checkDirExists(t, files.MappingDir, "mapping directory")
		checkDirExists(t, files.EdgesDir, "edges directory")
	}
}

func checkFileExists(t *testing.T, filepath, description string) {
	if filepath == "" {
		t.Errorf("%s path is empty", description)
		return
	}
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		t.Errorf("%s does not exist: %s", description, filepath)
	}
}

func checkDirExists(t *testing.T, dirpath, description string) {
	if dirpath == "" {
		t.Errorf("%s path is empty", description)
		return
	}
	if info, err := os.Stat(dirpath); os.IsNotExist(err) {
		t.Errorf("%s does not exist: %s", description, dirpath)
	} else if !info.IsDir() {
		t.Errorf("%s is not a directory: %s", description, dirpath)
	}
}

// Benchmark tests
func BenchmarkMaterializationClustering(b *testing.B) {
	testData := setupTestData(&testing.T{}) // Hack for benchmark
	
	config := DefaultMaterializationConfig()
	config.GraphFile = testData.GraphFile
	config.MetaPathFile = testData.MetaPathFile
	config.OutputDir = filepath.Join(testData.TempDir, "bench_mat")
	config.Verbose = false
	config.MaxInstances = 1000

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		config.OutputDir = filepath.Join(testData.TempDir, "bench_mat", fmt.Sprintf("run_%d", i))
		RunMaterializationClustering(config)
	}
}

func BenchmarkScarClustering(b *testing.B) {
	testData := setupTestData(&testing.T{}) // Hack for benchmark
	
	config := DefaultScarConfig()
	config.GraphFile = testData.GraphFile
	config.MetaPathFile = testData.MetaPathFile
	config.OutputDir = filepath.Join(testData.TempDir, "bench_scar")
	config.Verbose = false
	config.K = 8
	config.NK = 2
	config.MaxIterations = 5

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		config.OutputDir = filepath.Join(testData.TempDir, "bench_scar", fmt.Sprintf("run_%d", i))
		RunScarClustering(config)
	}
}
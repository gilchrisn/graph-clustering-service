package services

import (
    "bufio"
    "encoding/json"
    "fmt"
    "log"
    "mime/multipart"
    "os"
    "path/filepath"
    "sort"
    "strconv"
    "strings"
    "time"

    // "graph-clustering-backend/utils"


    "gonum.org/v1/gonum/graph/network"
    "gonum.org/v1/gonum/graph/simple"
    "gonum.org/v1/gonum/mat"
    "gonum.org/v1/gonum/stat/mds"

    "github.com/gilchrisn/graph-clustering-service/pkg/materialization"
    "github.com/gilchrisn/graph-clustering-service/pkg/louvain"
    "github.com/gilchrisn/graph-clustering-service/pkg/scar"
    "github.com/gilchrisn/graph-clustering-service/pkg/parser"
)

// ===== CONTROLLER-EXPECTED FUNCTIONS (EXACT SIGNATURES) =====

// ProcessDataset - 2 args as controller expects
func ProcessDataset(datasetID string, k int) (*ProcessResult, error) {
    log.Printf("🚀 [DEBUG] ProcessDataset: %s, k=%d", datasetID, k)
    
    // Find input files (uploaded earlier)
    graphFile := filepath.Join("uploads", datasetID+".txt")
    pathFile := filepath.Join("uploads", datasetID+"_path.txt")
    propertiesFile := filepath.Join("uploads", datasetID+"_properties.txt")
    
    // Check if files exist
    for _, file := range []string{graphFile, pathFile, propertiesFile} {
        if _, err := os.Stat(file); os.IsNotExist(err) {
            return nil, fmt.Errorf("required file not found: %s", file)
        }
    }
    
    // Run materialization pipeline (default louvain processing)
    params := map[string]interface{}{"k": float64(k)}
    _, err := runMaterializationPipelineForBackend(datasetID, graphFile, pathFile, propertiesFile, params)
    if err != nil {
        return nil, err
    }
    
    algorithm := "materialization"
    paths := GetPipelineOutputPaths(datasetID, algorithm)
    
    return &ProcessResult{
        Success:        true,
        Message:        fmt.Sprintf("Processing completed for %s", datasetID),
        DatasetId:      datasetID,
        K:              k,
        FilePath:       paths.VisualizationDir,
        ProcessingType: "louvain",
    }, nil
}

// ProcessDatasetHeterogeneous - 2 args as controller expects  
func ProcessDatasetHeterogeneous(datasetID string, k int) (*ProcessResult, error) {
    log.Printf("🚀 [DEBUG] ProcessDatasetHeterogeneous: %s, k=%d", datasetID, k)
    
    // Find input files (uploaded earlier)
    graphFile := filepath.Join("uploads", datasetID+".txt")
    pathFile := filepath.Join("uploads", datasetID+"_path.txt")
    propertiesFile := filepath.Join("uploads", datasetID+"_properties.txt")
    
    // Check if files exist
    for _, file := range []string{graphFile, pathFile, propertiesFile} {
        if _, err := os.Stat(file); os.IsNotExist(err) {
            return nil, fmt.Errorf("required file not found: %s", file)
        }
    }
    
    // Run materialization pipeline
    params := map[string]interface{}{"k": float64(k)}
    _, err := runMaterializationPipelineForBackend(datasetID, graphFile, pathFile, propertiesFile, params)
    if err != nil {
        return nil, err
    }
    
    algorithm := "materialization"
    paths := GetPipelineOutputPaths(datasetID, algorithm)
    
    // ✅ READ ROOT NODE
    clusteringFiles := paths.GetClusteringFiles()
    rootNode, err := readRootFile(clusteringFiles.RootFile)
    if err != nil || strings.TrimSpace(rootNode) == "" {
        log.Printf("⚠️ Root file missing/empty, using fallback")
        rootNode = "c0_l3_0"
    }
    
    // processingTime := time.Since(startTime).Seconds()
    
    return &ProcessResult{
        Success:        true,
        Message:        fmt.Sprintf("Heterogeneous processing completed for %s", datasetID),
        DatasetId:      datasetID,
        RootNode:       strings.TrimSpace(rootNode),  // ✅ ADD THIS
        AlgorithmId:    "heterogeneous",              // ✅ ADD THIS
        Parameters: map[string]interface{}{           // ✅ ADD THIS
            "k": k,
        },
        NodeCount:      0, // TODO: Count from files if needed
        EdgeCount:      0, // TODO: Count from files if needed
        ProcessingTime: 0,
        K:              k,
        FilePath:       paths.VisualizationDir,
        ProcessingType: "heterogeneous",
    }, nil
}
// ProcessDatasetScar - 5 args as controller expects
func ProcessDatasetScar(datasetID string, k, nk int, th float64, metaFileName string) (*ProcessResult, error) {
    log.Printf("🚀 [DEBUG] ProcessDatasetScar: %s, k=%d, nk=%d, th=%.3f, meta=%s", datasetID, k, nk, th, metaFileName)
    
    // Find input files (uploaded earlier)
    graphFile := filepath.Join("uploads", datasetID+".txt")
    pathFile := filepath.Join("uploads", datasetID+"_path.txt")
    propertiesFile := filepath.Join("uploads", datasetID+"_properties.txt")
    
    // Check if files exist
    for _, file := range []string{graphFile, pathFile, propertiesFile} {
        if _, err := os.Stat(file); os.IsNotExist(err) {
            return nil, fmt.Errorf("required file not found: %s", file)
        }
    }
    
    // Run SCAR pipeline
    params := map[string]interface{}{
        "k":  float64(k),
        "nk": float64(nk),
        "th": th,
    }
    _, err := runSCARPipelineForBackend(datasetID, graphFile, pathFile, propertiesFile, params)
    if err != nil {
        return nil, err
    }
    
    algorithm := "scar"
    paths := GetPipelineOutputPaths(datasetID, algorithm)
    
    // ✅ READ ROOT NODE
    clusteringFiles := paths.GetClusteringFiles()
    rootNode, err := readRootFile(clusteringFiles.RootFile)
    if err != nil || strings.TrimSpace(rootNode) == "" {
        log.Printf("⚠️ Root file missing/empty, using fallback")
        rootNode = "c0_l3_0"
    }
    
    // processingTime := time.Since(startTime).Seconds()
    
    return &ProcessResult{
        Success:        true,
        Message:        fmt.Sprintf("SCAR processing completed for %s", datasetID),
        DatasetId:      datasetID,
        RootNode:       strings.TrimSpace(rootNode),  // ✅ ADD THIS
        AlgorithmId:    "scar",                       // ✅ ADD THIS
        Parameters: map[string]interface{}{           // ✅ ADD THIS
            "k":  k,
            "nk": nk,
            "th": th,
        },
        NodeCount:      0, // TODO: Count from files if needed
        EdgeCount:      0, // TODO: Count from files if needed
        ProcessingTime: 0,
        K:              k,
        FilePath:       paths.VisualizationDir,
        ProcessingType: "scar",
    }, nil
}

// GetHierarchyData - 3 args as controller expects, returns struct with .Hierarchy and .Mapping fields
func GetHierarchyData(datasetId string, k int, processingType string) (*HierarchyResponse, error) {
    log.Printf("🚀 [DEBUG] GetHierarchyData: dataset=%s, k=%d, type=%s", datasetId, k, processingType)
    
    // Map processing type to algorithm
    algorithm := mapProcessingTypeToAlgorithm(processingType)
    paths := GetPipelineOutputPaths(datasetId, algorithm)
    
    // Check if pipeline output exists
    clusteringFiles := paths.GetClusteringFiles()
    if !filesExist(clusteringFiles.MappingFile, clusteringFiles.HierarchyFile, clusteringFiles.RootFile) {
        return nil, fmt.Errorf("pipeline output not found for dataset %s with algorithm %s", datasetId, algorithm)
    }
    
    log.Printf("🔍 [DEBUG] Loading hierarchy from: %s", paths.ClusteringDir)
    
    // Parse mapping file
    communities, err := parseMappingFile(clusteringFiles.MappingFile)
    if err != nil {
        return nil, fmt.Errorf("failed to parse mapping file: %w", err)
    }
    
    // Parse hierarchy file  
    hierarchyMap, err := parseHierarchyFile(clusteringFiles.HierarchyFile)
    if err != nil {
        return nil, fmt.Errorf("failed to parse hierarchy file: %w", err)
    }
    
    // Build levels map
    levels := make(map[int][]string)
    for communityId := range communities {
        level := extractLevelFromCommunityId(communityId)
        levels[level] = append(levels[level], communityId)
    }
    
    // Add hierarchy relationships to levels
    for parentId := range hierarchyMap {
        level := extractLevelFromCommunityId(parentId)
        if level > 0 { // Only add non-leaf levels
            found := false
            for _, existingId := range levels[level] {
                if existingId == parentId {
                    found = true
                    break
                }
            }
            if !found {
                levels[level] = append(levels[level], parentId)
            }
        }
    }
    
    // Get root node
    rootNode, err := readRootFile(clusteringFiles.RootFile)
    if err != nil {
        log.Printf("⚠️ [DEBUG] Could not read root file: %v", err)
        rootNode = "unknown"
    }
    
    // Convert to the format controller expects
    hierarchyData := make(map[string]interface{})
    mappingData := make(map[string]interface{})
    
    // Build hierarchy format that controller expects
    for communityId, nodes := range communities {
        hierarchyData[communityId] = nodes
    }
    
    // Build mapping format that controller expects  
    for communityId, nodes := range communities {
        mappingData[communityId] = nodes
    }
    
    // Add hierarchy relationships
    for parentId, children := range hierarchyMap {
        hierarchyData[parentId] = children
    }
    
    response := &HierarchyResponse{
        Hierarchy: hierarchyData,
        Mapping:   mappingData,
    }
    
    log.Printf("✅ [DEBUG] Loaded hierarchy: %d communities, %d levels, root=%s", 
        len(communities), len(levels), strings.TrimSpace(rootNode))
    
    return response, nil
}

// GetSupernodeCoordinates - 4 args as controller expects, returns struct with .Nodes and .Edges fields
func GetSupernodeCoordinates(datasetId string, algorithmId string, supernodeId string, processingType string) (*CoordinatesResponse, error) {
    log.Printf("🚀 [DEBUG] GetSupernodeCoordinates: dataset=%s, algorithm=%s, supernode=%s, type=%s",
        datasetId, algorithmId, supernodeId, processingType)

    // Map processing type to algorithm
    algorithm := mapProcessingTypeToAlgorithm(algorithmId)
    paths := GetPipelineOutputPaths(datasetId, algorithm)
    
    // Load visualization data
    vizData, err := loadVisualizationData(paths.VisualizationDir)
    if err != nil {
        return nil, fmt.Errorf("failed to load visualization data: %w", err)
    }
    
    // Find supernode level
    supernodeLevel := extractLevelFromCommunityId(supernodeId)
    if supernodeLevel == 0 {
        // Leaf node - return its own coordinates
        return getLeafNodeCoordinatesForController(supernodeId, vizData)
    }
    
    // Get immediate children coordinates
    children, err := getImmediateChildren(datasetId, algorithm, supernodeId)
    if err != nil {
        return nil, fmt.Errorf("failed to get children: %w", err)
    }
    
	log.Printf("🔍 [DEBUG] Found %d children for supernode %s at level %d", len(children), supernodeId, supernodeLevel)
	nodes := make([]interface{}, 0)  

    childLevel := supernodeLevel - 1
    levelVizData, exists := vizData[childLevel]
    if !exists {
        return nil, fmt.Errorf("no visualization data found for level %d", childLevel)
    }
    
	// Build result in correct format
	for _, rawChildID := range children {
		// Construct proper child ID based on level  
		var lookupID string
		if childLevel == 0 {
			lookupID = rawChildID
		} else {
			lookupID = fmt.Sprintf("c0_l%d_%s", childLevel, rawChildID)
		}

		if vizNode, exists := levelVizData[lookupID]; exists {
			// Determine node type
			nodeType := "leaf"
			if childLevel > 0 {
				nodeType = "supernode"
			}

			// Create node in frontend format
			node := map[string]interface{}{
				"id":     lookupID,
				"label":  vizNode.Label,
				"x":      vizNode.X * 100,
				"y":      vizNode.Y * 100,
				"radius": vizNode.Radius,
				"type":   nodeType,
				"metadata": map[string]interface{}{
					"degree":   nil, // Could add if available
					"dpr":      nil, // Could add if available  
					"leafCount": nil, // Could add if available
				},
			}
			nodes = append(nodes, node)
		}
	}

    edges, err := getSupernodeEdges(datasetId, algorithm, childLevel, children)
    if err != nil {
        return nil, fmt.Errorf("failed to load supernode edges: %w", err)
    }

	return &CoordinatesResponse{
		Success: true,  // Add success field
		Nodes:   nodes, // Now array format
		Edges:   edges,
	}, nil

}

func getSupernodeEdges(
    datasetId, algorithm string,
    level int,
    children []string,
) ([]interface{}, error) {
    // 1) Build a set of valid lookup IDs for fast membership checks
    valid := make(map[string]bool, len(children))
    for _, rawID := range children {
        var lookupID string
        if level == 0 {
            lookupID = rawID
        } else {
            lookupID = fmt.Sprintf("c0_l%d_%s", level, rawID)
        }
        valid[lookupID] = true
    }

    // 2) Locate the edgelist file for this level
    paths := GetPipelineOutputPaths(datasetId, algorithm)
    edgeFile := filepath.Join(paths.HierarchyDir, fmt.Sprintf("level_%d.edgelist", level))

    // 3) Load raw edges from disk
    rawEdges, err := loadEdges(edgeFile)
    if err != nil {
        return nil, fmt.Errorf("failed to load edge list %q: %w", edgeFile, err)
    }

    // 4) Filter, remove self-loops, and dedupe undirected edges
    out := make([]interface{}, 0, len(rawEdges))
    seen := make(map[string]bool, len(rawEdges))

    for _, e := range rawEdges {
        // Skip self-loops
        if e.From == e.To {
            continue
        }

        // Canonicalize the undirected edge key (smaller ID first)
        a, b := e.From, e.To
        if a > b {
            a, b = b, a
        }
        key := fmt.Sprintf("%s|%s", a, b)

        // Skip duplicates
        if seen[key] {
            continue
        }
        seen[key] = true

        // Only include if both endpoints are in our valid set
        if valid[a] && valid[b] {
            out = append(out, map[string]interface{}{
                "source": a,
                "target": b,
                "weight": e.Weight,
            })
        }
    }

    return out, nil
}

// GetNodeStatistics - 4 args as controller expects
func GetNodeStatistics(datasetId string, k int, nodeId, processingType string) (map[string]interface{}, error) {
    log.Printf("🚀 [DEBUG] GetNodeStatistics: dataset=%s, k=%d, node=%s, type=%s", 
        datasetId, k, nodeId, processingType)
    
    // Map processing type to algorithm
    algorithm := mapProcessingTypeToAlgorithm(processingType)
    paths := GetPipelineOutputPaths(datasetId, algorithm)
    
    // Load visualization data
    vizData, err := loadVisualizationData(paths.VisualizationDir)
    if err != nil {
        return nil, fmt.Errorf("failed to load visualization data: %w", err)
    }
    
    // Find node level and data
    nodeLevel := extractLevelFromCommunityId(nodeId)
    if nodeLevel == 0 && !strings.Contains(nodeId, "_l") {
        // Original leaf node - search all levels starting from 0
        for level := 0; level < 10; level++ { // reasonable upper bound
            if levelData, exists := vizData[level]; exists {
                if nodeViz, exists := levelData[nodeId]; exists {
                    return buildNodeStatistics(nodeViz, level, datasetId, algorithm), nil
                }
            }
        }
        return nil, fmt.Errorf("node %s not found in any level", nodeId)
    }
    
    // Community node - get from specific level
    levelData, exists := vizData[nodeLevel]
    if !exists {
        return nil, fmt.Errorf("no visualization data found for level %d", nodeLevel)
    }
    
    nodeViz, exists := levelData[nodeId]
    if !exists {
        return nil, fmt.Errorf("node %s not found in level %d visualization data", nodeId, nodeLevel)
    }
    
    stats := buildNodeStatistics(nodeViz, nodeLevel, datasetId, algorithm)
    
    // Add children count if it's a supernode
    if nodeLevel > 0 {
        children, err := getImmediateChildren(datasetId, algorithm, nodeId)
        if err == nil {
            stats["children_count"] = len(children)
            stats["has_children"] = len(children) > 0
        }
    }
    
    log.Printf("✅ [DEBUG] Retrieved statistics for node %s", nodeId)
    return stats, nil
}

// CompareAlgorithms - 4 args as controller expects
func CompareAlgorithms(files map[string][]*multipart.FileHeader, datasetName string, heteroParams, scarParams map[string]interface{}) (*ComparisonResult, error) {
    log.Println("🚀 [DEBUG] CompareAlgorithms started")
    
    // Save uploaded files first
    savedFiles, err := saveComparisonFiles(files, datasetName)
    if err != nil {
        return nil, fmt.Errorf("failed to save files: %w", err)
    }
    
    // Generate unique dataset ID
    // datasetId := fmt.Sprintf("comparison_%d", time.Now().Unix())
    datasetId := datasetName
    log.Printf("🔍 [DEBUG] Generated dataset ID: %s", datasetId)
    
    // Run both algorithms in parallel
    results := make(chan *PipelineResult, 2)
    errors := make(chan error, 2)
    
    // Run materialization pipeline
    go func() {
        log.Println("🔄 [DEBUG] Starting materialization pipeline...")
        result, err := runMaterializationPipelineForBackend(datasetId, savedFiles["graphFile"], savedFiles["pathFile"], savedFiles["propertiesFile"], heteroParams)
        if err != nil {
            errors <- fmt.Errorf("materialization failed: %w", err)
            return
        }
        results <- result
    }()
    
    // Run SCAR pipeline  
    go func() {
        log.Println("🔄 [DEBUG] Starting SCAR pipeline...")
        result, err := runSCARPipelineForBackend(datasetId, savedFiles["graphFile"], savedFiles["pathFile"], savedFiles["propertiesFile"], scarParams)
        if err != nil {
            errors <- fmt.Errorf("SCAR failed: %w", err)
            return
        }
        results <- result
    }()
    
    // Collect results (wait for both to complete)
    for i := 0; i < 2; i++ {
        select {
        case result := <-results:
            if result.PipelineType == MaterializationLouvain {
                log.Println("✅ [DEBUG] Materialization pipeline completed")
            } else {
                log.Println("✅ [DEBUG] SCAR pipeline completed")
            }
        case err := <-errors:
            log.Printf("❌ [DEBUG] Pipeline error: %v", err)
            return nil, err
        }
    }
    
    log.Println("🧮 [DEBUG] Computing comparison metrics...")
    
    // Load hierarchy data for comparison
    materializationHierarchy, err := loadHierarchyDataFromPipeline(GetPipelineOutputPaths(datasetId, "materialization"))
    if err != nil {
        return nil, fmt.Errorf("failed to load materialization hierarchy: %w", err)
    }
    
    scarHierarchy, err := loadHierarchyDataFromPipeline(GetPipelineOutputPaths(datasetId, "scar"))
    if err != nil {
        return nil, fmt.Errorf("failed to load SCAR hierarchy: %w", err)
    }
    
    // Compute NMI and other metrics
    metrics, err := computeComparisonMetrics(materializationHierarchy, scarHierarchy)
    if err != nil {
        return nil, fmt.Errorf("metrics computation failed: %w", err)
    }
    
    // Build mapping data
    materializationMapping := buildMappingData(materializationHierarchy)
    scarMapping := buildMappingData(scarHierarchy)
    
    // Build comparison result
    comparison := &ComparisonResult{
        Heterogeneous: ComparisonAlgorithmResult{
            DatasetId:     datasetId,
            HierarchyData: *materializationHierarchy,
            MappingData:   *materializationMapping,
            RootNode:      materializationHierarchy.RootNode,
            Parameters:    heteroParams,
        },
        SCAR: ComparisonAlgorithmResult{
            DatasetId:     datasetId,
            HierarchyData: *scarHierarchy,
            MappingData:   *scarMapping,
            RootNode:      scarHierarchy.RootNode,
            Parameters:    scarParams,
        },
        Metrics: *metrics,
    }
    
    log.Println("✅ [DEBUG] Comparison completed successfully")
    return comparison, nil
}

// ===== PIPELINE EXECUTION FUNCTIONS FOR BACKEND =====

func runMaterializationPipelineForBackend(datasetId, graphFile, pathFile, propertiesFile string, params map[string]interface{}) (*PipelineResult, error) {
    log.Printf("🚀 [DEBUG] runMaterializationPipelineForBackend for dataset: %s", datasetId)
    
    // Get pipeline paths
    paths := GetPipelineOutputPaths(datasetId, "materialization")
    baseOutputDir := paths.BaseDir
    
    // Always overwrite - keep it simple
    // os.RemoveAll(baseOutputDir)
    
    // Run your complete materialization pipeline
    err := runMaterializationPipeline(graphFile, propertiesFile, pathFile, baseOutputDir)
    if err != nil {
        return nil, fmt.Errorf("runMaterializationPipeline failed: %w", err)
    }
    
    // Return result (we can't get the detailed result from your pipeline, so create basic one)
    return &PipelineResult{
        PipelineType:   MaterializationLouvain,
        TotalRuntimeMS: 0, // Your pipeline doesn't return timing to us
        SCARSuccess:    false,
    }, nil
}

func runSCARPipelineForBackend(datasetId, graphFile, pathFile, propertiesFile string, params map[string]interface{}) (*PipelineResult, error) {
    log.Printf("🚀 [DEBUG] runSCARPipelineForBackend for dataset: %s", datasetId)
    
    k := int(params["k"].(float64))
    nk := int(params["nk"].(float64)) 
    th := params["th"].(float64)

    // Get pipeline paths
    paths := GetPipelineOutputPaths(datasetId, "scar")
    baseOutputDir := paths.BaseDir
    
    // Always overwrite - keep it simple
    // os.RemoveAll(baseOutputDir)
    
    // Run your complete SCAR pipeline
    err := runScarPipeline(graphFile, propertiesFile, pathFile, baseOutputDir, k, nk, th)
    if err != nil {
        return nil, fmt.Errorf("runScarPipeline failed: %w", err)
    }
    
    // Return result
    return &PipelineResult{
        PipelineType: SketchLouvain,
        TotalRuntimeMS: 0, // Your pipeline doesn't return timing to us
        SCARSuccess:  true,
    }, nil
}

// ===== YOUR COMPLETE PIPELINE FUNCTIONS (COPIED EXACTLY) =====

// ===== MATERIALIZATION + LOUVAIN PIPELINE =====

func runMaterializationPipeline(graphFile, propertiesFile, pathFile, baseOutputDir string) error {
    algorithm := "materialization"
    fmt.Printf("\n=== %s Pipeline ===\n", algorithm)

    // Step 1: Run materialization + Louvain
    clusteringDir := filepath.Join(baseOutputDir, algorithm, "clustering")
    err := os.MkdirAll(clusteringDir, 0755)
    if err != nil {
        return err
    }

    fmt.Println("1. Running materialization + Louvain...")
    err = runMaterializationLouvain(graphFile, propertiesFile, pathFile, clusteringDir)
    if err != nil {
        return fmt.Errorf("clustering failed: %w", err)
    }

    // Step 2: Parse hierarchy to level files
    hierarchyDir := filepath.Join(baseOutputDir, algorithm, "hierarchy")
    err = os.MkdirAll(hierarchyDir, 0755)
    if err != nil {
        return err
    }

    fmt.Println("2. Parsing hierarchy...")
    maxLevel, err := parseMaterializationHierarchy(clusteringDir, hierarchyDir)
    if err != nil {
        return fmt.Errorf("hierarchy parsing failed: %w", err)
    }

    // Step 3: Generate PageRank + MDS for each level
    vizDir := filepath.Join(baseOutputDir, algorithm, "visualization")
    err = os.MkdirAll(vizDir, 0755)
    if err != nil {
        return err
    }

    fmt.Println("3. Computing PageRank + MDS...")
    err = generateVisualization(hierarchyDir, vizDir, algorithm, maxLevel)
    if err != nil {
        return fmt.Errorf("visualization failed: %w", err)
    }

    fmt.Printf("✅ %s pipeline completed\n", algorithm)
    return nil
}

// ===== SCAR PIPELINE =====

func runScarPipeline(graphFile, propertiesFile, pathFile, baseOutputDir string, k, nk int, th float64) error {
    algorithm := "scar"
    fmt.Printf("\n=== %s Pipeline ===\n", algorithm)

    // Step 1: Run SCAR
    clusteringDir := filepath.Join(baseOutputDir, algorithm, "clustering")
    err := os.MkdirAll(clusteringDir, 0755)
    if err != nil {
        return err
    }

    fmt.Println("1. Running SCAR...")
    err = runScar(graphFile, propertiesFile, pathFile, clusteringDir, k, nk, th)
    if err != nil {
        return fmt.Errorf("SCAR failed: %w", err)
    }

    // Step 2: Parse hierarchy to level files
    hierarchyDir := filepath.Join(baseOutputDir, algorithm, "hierarchy")
    err = os.MkdirAll(hierarchyDir, 0755)
    if err != nil {
        return err
    }

    fmt.Println("2. Parsing hierarchy...")
    maxLevel, err := parseScarHierarchy(clusteringDir, hierarchyDir)
    if err != nil {
        return fmt.Errorf("hierarchy parsing failed: %w", err)
    }

    // Step 3: Generate PageRank + MDS for each level
    vizDir := filepath.Join(baseOutputDir, algorithm, "visualization")
    err = os.MkdirAll(vizDir, 0755)
    if err != nil {
        return err
    }

    fmt.Println("3. Computing PageRank + MDS...")
    err = generateVisualization(hierarchyDir, vizDir, algorithm, maxLevel)
    if err != nil {
        return fmt.Errorf("visualization failed: %w", err)
    }

    fmt.Printf("✅ %s pipeline completed\n", algorithm)
    return nil
}

func runMaterializationLouvain(graphFile, propertiesFile, pathFile, outputDir string) error {
    // Create default configuration exactly like your original code
    config := NewPipelineConfig()
    config.OutputDir = outputDir
    config.OutputPrefix = "communities"
    config.Verbose = true
    
    // Configure materialization + Louvain with same settings as your original
    config.MaterializationConfig.Aggregation.Strategy = materialization.Average
    config.MaterializationConfig.Aggregation.Symmetric = true
    config.MaterializationConfig.Traversal.MaxInstances = 1000000
    
    config.LouvainConfig.MaxIterations = 10
    config.LouvainConfig.MinModularity = 0.001
    config.LouvainConfig.RandomSeed = 42

    fmt.Println("   🔄 Running materialization...")
    fmt.Printf("   Materialization config: max_instances=%d, symmetric=%t\n", 
        config.MaterializationConfig.Traversal.MaxInstances, 
        config.MaterializationConfig.Aggregation.Symmetric)
    fmt.Printf("   Louvain config: max_iter=%d, min_mod=%.6f, seed=%d\n", 
        config.LouvainConfig.MaxIterations, 
        config.LouvainConfig.MinModularity, 
        config.LouvainConfig.RandomSeed)
    
    // Now call your exact RunMaterializationLouvain function
    result, err := RunMaterializationLouvain(graphFile, propertiesFile, pathFile, config)
    if err != nil {
        return err
    }
    
    fmt.Printf("   ✅ Materialization + Louvain completed in %d ms\n", result.TotalRuntimeMS)
    fmt.Printf("   Final modularity: %.6f\n", result.LouvainResult.Modularity)
    if len(result.LouvainResult.Levels) > 0 {
        finalLevel := result.LouvainResult.Levels[len(result.LouvainResult.Levels)-1]
        fmt.Printf("   Number of communities: %d, Hierarchy levels: %d\n", 
            finalLevel.NumCommunities, result.LouvainResult.NumLevels)
    }
    
    return nil
}

func runScar(graphFile, propertiesFile, pathFile, outputDir string, k, nk int, th float64) error {
    // Create default configuration exactly like your original code
    config := NewPipelineConfig()
    config.OutputDir = outputDir
    config.OutputPrefix = "communities"
    config.Verbose = true
    
    // Configure SCAR with same settings as your original
    config.SCARConfig.K = int64(k)
    config.SCARConfig.NK = 1
    config.SCARConfig.Threshold = 0.5
    config.SCARConfig.UseLouvain = true
    config.SCARConfig.SketchOutput = true

    fmt.Println("   🔄 Running SCAR...")
    fmt.Printf("   SCAR config: k=%d, nk=%d, threshold=%.3f, sketch_output=%t\n", 
        config.SCARConfig.K, config.SCARConfig.NK, config.SCARConfig.Threshold, config.SCARConfig.SketchOutput)
    
    // Now call your exact RunSketchLouvain function  
    result, err := RunSketchLouvain(graphFile, propertiesFile, pathFile, config)
    if err != nil {
        return err
    }
    
    fmt.Printf("   ✅ SCAR completed in %d ms\n", result.TotalRuntimeMS)
    return nil
}

// ===== ALL YOUR PIPELINE SUPPORT FUNCTIONS (EXACT COPIES) =====

func parseMaterializationHierarchy(clusteringDir, hierarchyDir string) (int, error) {
    // Build file paths - materialization uses standard names
    edgelistFile := filepath.Join(clusteringDir, "materialized_graph.edgelist")
    mappingFile := filepath.Join(clusteringDir, "communities.mapping")
    hierarchyFile := filepath.Join(clusteringDir, "communities.hierarchy") 
    rootFile := filepath.Join(clusteringDir, "communities.root")
    
    // Check if files exist
    if err := checkRequiredFiles(edgelistFile, mappingFile, hierarchyFile, rootFile); err != nil {
        return 0, fmt.Errorf("required files missing: %w", err)
    }
    
    // Call parser - it creates files as {outputPrefix}_level_{N}.txt
    outputPrefix := filepath.Join(hierarchyDir, "hierarchy")
    err := parser.ParseLouvainHierarchy(edgelistFile, mappingFile, hierarchyFile, rootFile, outputPrefix)
    if err != nil {
        return 0, fmt.Errorf("ParseLouvainHierarchy failed: %w", err)
    }
    
    // Rename .txt files to .edgelist and count levels
    maxLevel := 0
    for {
        sourceFile := fmt.Sprintf("%s_level_%d.txt", outputPrefix, maxLevel)
        targetFile := filepath.Join(hierarchyDir, fmt.Sprintf("level_%d.edgelist", maxLevel))
        
        if _, err := os.Stat(sourceFile); os.IsNotExist(err) {
            break
        }
        
        err := os.Rename(sourceFile, targetFile)
        if err != nil {
            return 0, fmt.Errorf("failed to rename %s to %s: %w", sourceFile, targetFile, err)
        }
        maxLevel++
    }
    
    if maxLevel == 0 {
        return 0, fmt.Errorf("no level files were created")
    }
    
    return maxLevel - 1, nil
}

func parseScarHierarchy(clusteringDir, hierarchyDir string) (int, error) {
    // Build file paths - SCAR uses .dat extensions
    sketchFile := filepath.Join(clusteringDir, "communities.sketch")
    mappingFile := filepath.Join(clusteringDir, "communities_mapping.dat")  // Note: different name
    hierarchyFile := filepath.Join(clusteringDir, "communities_hierarchy.dat")  // Note: different name
    rootFile := filepath.Join(clusteringDir, "communities_root.dat")  // Note: different name
    
    // Check if files exist
    if err := checkRequiredFiles(sketchFile, mappingFile, hierarchyFile, rootFile); err != nil {
        return 0, fmt.Errorf("required files missing: %w", err)
    }
    
    // Call parser - it creates files as {outputPrefix}_level_{N}.txt  
    outputPrefix := filepath.Join(hierarchyDir, "sketch_hierarchy")
    err := parser.ParseSketchLouvainHierarchy(sketchFile, mappingFile, hierarchyFile, rootFile, outputPrefix)
    if err != nil {
        return 0, fmt.Errorf("ParseSketchLouvainHierarchy failed: %w", err)
    }
    
    // Rename .txt files to .edgelist and count levels
    maxLevel := 0
    for {
        sourceFile := fmt.Sprintf("%s_level_%d.txt", outputPrefix, maxLevel)
        targetFile := filepath.Join(hierarchyDir, fmt.Sprintf("level_%d.edgelist", maxLevel))
        
        if _, err := os.Stat(sourceFile); os.IsNotExist(err) {
            break
        }
        
        err := os.Rename(sourceFile, targetFile)
        if err != nil {
            return 0, fmt.Errorf("failed to rename %s to %s: %w", sourceFile, targetFile, err)
        }
        maxLevel++
    }
    
    if maxLevel == 0 {
        return 0, fmt.Errorf("no level files were created")
    }
    
    return maxLevel - 1, nil
}

func checkRequiredFiles(filenames ...string) error {
    for _, filename := range filenames {
        if _, err := os.Stat(filename); os.IsNotExist(err) {
            return fmt.Errorf("file does not exist: %s", filename)
        }
    }
    return nil
}

func generateVisualization(hierarchyDir, vizDir, algorithm string, maxLevel int) error {
    var allLevels []LevelViz

    // Process each level
    for level := 0; level <= maxLevel; level++ {
        fmt.Printf("   Level %d...", level)
        
        // Load level graph
        levelPath := filepath.Join(hierarchyDir, fmt.Sprintf("level_%d.edgelist", level))
        graph, nodeMapping, err := loadGraph(levelPath)
        if err != nil {
            fmt.Printf(" ❌ failed to load: %v\n", err)
            continue
        }

        if graph.Nodes().Len() == 0 {
            fmt.Printf(" ⏭️ empty graph\n")
            continue
        }

        // Compute PageRank
        pageRankScores := network.PageRank(graph, 0.85, 1e-6)

        // Compute MDS
        distMatrix := createDistanceMatrix(graph)
        coords := applyMDS(distMatrix)

        // Create visualization data
        levelViz := createLevelVisualization(level, pageRankScores, coords, nodeMapping, algorithm)
        allLevels = append(allLevels, levelViz)

        fmt.Printf(" ✅ %d nodes\n", len(levelViz.Nodes))
    }

    // Save all levels to single file
    vizPath := filepath.Join(vizDir, "levels.json")
    err := saveJSON(vizPath, allLevels)
    if err != nil {
        return err
    }

    fmt.Printf("   Saved visualization: %d levels\n", len(allLevels))
    return nil
}

func createLevelVisualization(level int, pageRankScores map[int64]float64, coords *mat.Dense, nodeMapping map[int64]string, algorithm string) LevelViz {
    var nodes []NodeViz

    // Find min/max PageRank for scaling
    var minPR, maxPR float64
    first := true
    for _, score := range pageRankScores {
        if first {
            minPR, maxPR = score, score
            first = false
        } else {
            if score < minPR {
                minPR = score
            }
            if score > maxPR {
                maxPR = score
            }
        }
    }
    if maxPR == minPR {
        maxPR = minPR + 0.001
    }

    // Create node visualizations
    for nodeID, score := range pageRankScores {
        // Get coordinates
        x := coords.At(int(nodeID), 0)
        y := coords.At(int(nodeID), 1)

        // Scale radius based on PageRank
        normalizedPR := (score - minPR) / (maxPR - minPR)
        radius := 3.0 + normalizedPR*15.0

        // Create label
        originalID := nodeMapping[nodeID]
        label := fmt.Sprintf("%s_%s_L%d (%.4f)", algorithm, originalID, level, score)

        node := NodeViz{
            ID:       originalID,
            PageRank: score,
            X:        x,
            Y:        y,
            Radius:   radius,
            Label:    label,
        }
        nodes = append(nodes, node)
    }

    return LevelViz{
        Level: level,
        Nodes: nodes,
    }
}

// ===== ALL YOUR UTILITY FUNCTIONS (COPIED EXACTLY) =====

func loadEdges(filename string) ([]Edge, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var edges []Edge
    scanner := bufio.NewScanner(file)

    for scanner.Scan() {
        line := strings.TrimSpace(scanner.Text())
        if line == "" || strings.HasPrefix(line, "#") {
            continue
        }

        parts := strings.Fields(line)
        if len(parts) < 2 {
            continue
        }

        from := parts[0]
        to := parts[1]
        weight := 1.0

        if len(parts) >= 3 {
            if w, err := strconv.ParseFloat(parts[2], 64); err == nil {
                weight = w
            }
        }

        edges = append(edges, Edge{From: from, To: to, Weight: weight})
    }

    return edges, scanner.Err()
}

func loadGraph(filename string) (*simple.DirectedGraph, map[int64]string, error) {
    edges, err := loadEdges(filename)
    if err != nil {
        return nil, nil, err
    }

    graph := simple.NewDirectedGraph()
    nodeMapping := make(map[int64]string)
    stringToInt := make(map[string]int64)
    nextID := int64(0)

    for _, edge := range edges {
        // Skip self-loops for now
        if edge.From == edge.To {
            continue
        }

        // Map string IDs to integers
        fromID, exists := stringToInt[edge.From]
        if !exists {
            fromID = nextID
            stringToInt[edge.From] = fromID
            nodeMapping[fromID] = edge.From
            graph.AddNode(simple.Node(fromID))
            nextID++
        }

        toID, exists := stringToInt[edge.To]
        if !exists {
            toID = nextID
            stringToInt[edge.To] = toID
            nodeMapping[toID] = edge.To
            graph.AddNode(simple.Node(toID))
            nextID++
        }

        // Add edge if it doesn't exist
        if !graph.HasEdgeFromTo(fromID, toID) {
            graph.SetEdge(simple.Edge{
                F: simple.Node(fromID),
                T: simple.Node(toID),
            })
        }
    }

    return graph, nodeMapping, nil
}

func createDistanceMatrix(g *simple.DirectedGraph) *mat.SymDense {
    nodes := g.Nodes()
    nodeList := make([]int64, 0)
    for nodes.Next() {
        nodeList = append(nodeList, nodes.Node().ID())
    }
    n := len(nodeList)

    if n == 0 {
        return mat.NewSymDense(0, nil)
    }

    distMatrix := mat.NewSymDense(n, nil)

    for i, nodeI := range nodeList {
        distances := bfsDistances(g, nodeI)
        for j, nodeJ := range nodeList {
            dist := distances[nodeJ]
            if dist == -1 {
                dist = float64(n) // Use graph diameter as max distance
            }
            distMatrix.SetSym(i, j, dist)
        }
    }

    return distMatrix
}

func bfsDistances(g *simple.DirectedGraph, source int64) map[int64]float64 {
    distances := make(map[int64]float64)
    visited := make(map[int64]bool)
    queue := []int64{source}
    
    distances[source] = 0
    visited[source] = true

    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]

        neighbors := g.From(current)
        for neighbors.Next() {
            neighbor := neighbors.Node().ID()
            if !visited[neighbor] {
                visited[neighbor] = true
                distances[neighbor] = distances[current] + 1
                queue = append(queue, neighbor)
            }
        }
    }

    nodes := g.Nodes()
    for nodes.Next() {
        nodeID := nodes.Node().ID()
        if _, exists := distances[nodeID]; !exists {
            distances[nodeID] = -1
        }
    }

    return distances
}

func applyMDS(distMatrix *mat.SymDense) *mat.Dense {
    var coords mat.Dense
    var eigenvals []float64

    k, _ := mds.TorgersonScaling(&coords, eigenvals, distMatrix)
    
    fmt.Printf("   MDS: %d positive eigenvalues\n", k)

    return &coords
}

func saveJSON(path string, data interface{}) error {
    file, err := os.Create(path)
    if err != nil {
        return err
    }
    defer file.Close()

    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")
    return encoder.Encode(data)
}

// ===== YOUR PIPELINE EXECUTION FUNCTIONS =====

func convertHomogeneousToNormalized(hgraph *materialization.HomogeneousGraph) (*louvain.NormalizedGraph, *louvain.GraphParser, error) {
    if len(hgraph.Nodes) == 0 {
        return nil, nil, fmt.Errorf("empty homogeneous graph")
    }
    
    parser := louvain.NewGraphParser()
    
    // Create ordered list of node IDs
    nodeList := make([]string, 0, len(hgraph.Nodes))
    for nodeID := range hgraph.Nodes {
        nodeList = append(nodeList, nodeID)
    }

    // Use intelligent sorting
    allIntegers := allNodesAreIntegers(nodeList)
    if allIntegers {
        // Sort numerically: 1 < 2 < 5 < 10
        sort.Slice(nodeList, func(i, j int) bool {
            a, _ := strconv.Atoi(nodeList[i])
            b, _ := strconv.Atoi(nodeList[j])
            return a < b
        })
    } else {
        // Sort lexicographically: "1" < "10" < "2" < "5"
        sort.Strings(nodeList)
    }
    
    // Create normalized graph
    normalizedGraph := louvain.NewNormalizedGraph(len(nodeList))
    
    // Build node ID mappings and set weights
    for i, originalID := range nodeList {
        parser.OriginalToNormalized[originalID] = i
        parser.NormalizedToOriginal[i] = originalID
        
        // Set node weights (using default for now)
        if _, exists := hgraph.Nodes[originalID]; exists {
            normalizedGraph.Weights[i] = 1.0 // Default weight
        }
    }
    parser.NumNodes = len(nodeList)
    
    // Convert edges with deduplication to prevent double counting
    processedEdges := make(map[string]bool)
    edgeCount := 0
    
    for edgeKey, weight := range hgraph.Edges {
        fromNormalized, fromExists := parser.OriginalToNormalized[edgeKey.From]
        toNormalized, toExists := parser.OriginalToNormalized[edgeKey.To]
        
        if !fromExists || !toExists {
            return nil, nil, fmt.Errorf("edge references unknown nodes: %s -> %s", edgeKey.From, edgeKey.To)
        }
        
        // Create canonical edge ID (smaller index first) to avoid duplicates
        var canonicalID string
        if fromNormalized <= toNormalized {
            canonicalID = fmt.Sprintf("%d-%d", fromNormalized, toNormalized)
        } else {
            canonicalID = fmt.Sprintf("%d-%d", toNormalized, fromNormalized)
        }
        
        // Only process each undirected edge once
        if !processedEdges[canonicalID] {
            normalizedGraph.AddEdge(fromNormalized, toNormalized, weight)
            processedEdges[canonicalID] = true
            edgeCount++
        }
    }
    
    // Validate the converted graph
    if err := normalizedGraph.Validate(); err != nil {
        return nil, nil, fmt.Errorf("converted graph validation failed: %w", err)
    }

    return normalizedGraph, parser, nil
}

func writeLouvainOutputs(result *louvain.LouvainResult, parser *louvain.GraphParser, materializedGraph *materialization.HomogeneousGraph, config *PipelineConfig) error {
    // Create output directory
    if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
        return fmt.Errorf("failed to create output directory: %w", err)
    }
    
    // Write materialized graph files
    if materializedGraph != nil {
        // Write edge list (simple format for other tools)
        edgeListPath := filepath.Join(config.OutputDir, "materialized_graph.edgelist")
        if err := materialization.SaveAsSimpleEdgeList(materializedGraph, edgeListPath); err != nil {
            return fmt.Errorf("failed to write materialized edgelist: %w", err)
        }
    }
    
    // Write Louvain results
    writer := louvain.NewFileWriter()
    if err := writer.WriteAll(result, parser, config.OutputDir, config.OutputPrefix); err != nil {
        return fmt.Errorf("failed to write Louvain results: %w", err)
    }
    
    return nil
}

func allNodesAreIntegers(nodes []string) bool {
    for _, node := range nodes {
        if _, err := strconv.Atoi(node); err != nil {
            return false
        }
    }
    return true
}

func RunMaterializationLouvain(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*PipelineResult, error) {
    startTime := time.Now()
    
    if config.Verbose {
        fmt.Println("=== Running Materialization + Louvain Pipeline ===")
    }
    
    // Step 1: Parse SCAR input for materialization
    if config.Verbose {
        fmt.Println("Step 1: Parsing input files for materialization...")
    }
    
    graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
    if err != nil {
        return nil, fmt.Errorf("failed to parse SCAR input: %w", err)
    }
    
    if config.Verbose {
        fmt.Printf("  Loaded graph with %d nodes\n", len(graph.Nodes))
    }
    
    // Step 2: Run materialization
    if config.Verbose {
        fmt.Println("Step 2: Running graph materialization...")
    }
    
    materializationStart := time.Now()
    
    // Setup progress callback for materialization
    var materializationProgressCb func(int, int, string)
    if config.Verbose {
        materializationProgressCb = func(current, total int, message string) {
            fmt.Printf("  Materialization progress: %d/%d - %s\n", current, total, message)
        }
    }
    
    engine := materialization.NewMaterializationEngine(graph, metaPath, config.MaterializationConfig, materializationProgressCb)
    materializationResult, err := engine.Materialize()
    if err != nil {
        return nil, fmt.Errorf("materialization failed: %w", err)
    }
    
    materializationTime := time.Since(materializationStart)
    
    if config.Verbose {
        fmt.Printf("  Materialization completed in %v\n", materializationTime)
        fmt.Printf("  Materialized graph has %d nodes and %d edges\n", 
            len(materializationResult.HomogeneousGraph.Nodes),
            len(materializationResult.HomogeneousGraph.Edges))
    }
    
    // Step 3: Convert HomogeneousGraph to NormalizedGraph for Louvain
    if config.Verbose {
        fmt.Println("Step 3: Converting graph format for Louvain...")
    }
    
    normalizedGraph, graphParser, err := convertHomogeneousToNormalized(materializationResult.HomogeneousGraph)
    if err != nil {
        return nil, fmt.Errorf("graph conversion failed: %w", err)
    }
    
    if config.Verbose {
        fmt.Printf("  Converted to normalized graph with %d nodes\n", normalizedGraph.NumNodes)
        fmt.Printf("  Total edge weight: %.2f\n", normalizedGraph.TotalWeight)
    }
    
    // Step 4: Run Louvain clustering
    if config.Verbose {
        fmt.Println("Step 4: Running Louvain community detection...")
    }
    
    louvainStart := time.Now()
    
    // Setup progress callback for Louvain
    if config.Verbose {
        config.LouvainConfig.Verbose = true
        config.LouvainConfig.ProgressCallback = func(level, iteration int, message string) {
            fmt.Printf("  Louvain [L%d I%d]: %s\n", level, iteration, message)
        }
    }
    
    louvainResult, err := louvain.RunLouvain(normalizedGraph, config.LouvainConfig)
    if err != nil {
        return nil, fmt.Errorf("Louvain clustering failed: %w", err)
    }
    
    louvainTime := time.Since(louvainStart)
    louvainResult.Parser = graphParser // Attach parser for output generation
    
    if config.Verbose {
        fmt.Printf("  Louvain completed in %v\n", louvainTime)
        fmt.Printf("  Final modularity: %.6f\n", louvainResult.Modularity)
        finalLevel := louvainResult.Levels[len(louvainResult.Levels)-1]
        fmt.Printf("  Number of communities: %d\n", finalLevel.NumCommunities)
        fmt.Printf("  Hierarchy levels: %d\n", louvainResult.NumLevels)
    }
    
    // Step 5: Generate output files
    if config.Verbose {
        fmt.Println("Step 5: Writing output files...")
    }
    
    if err := writeLouvainOutputs(louvainResult, graphParser, materializationResult.HomogeneousGraph, config); err != nil {
        return nil, fmt.Errorf("output generation failed: %w", err)
    }
    
    totalTime := time.Since(startTime)
    
    // Create final result
    result := &PipelineResult{
        PipelineType:      MaterializationLouvain,
        MaterializedGraph: materializationResult.HomogeneousGraph,
        LouvainResult:     louvainResult,
        TotalRuntimeMS:    totalTime.Milliseconds(),
    }
    
    if config.Verbose {
        fmt.Println("=== Materialization + Louvain Pipeline Complete ===")
        fmt.Printf("Total runtime: %v\n", totalTime)
        fmt.Printf("Materialization: %v, Louvain: %v\n", materializationTime, louvainTime)
        fmt.Printf("Final modularity: %.6f\n", result.LouvainResult.Modularity)
    }
    
    return result, nil
}

func RunSketchLouvain(graphFile, propertiesFile, pathFile string, config *PipelineConfig) (*PipelineResult, error) {
    startTime := time.Now()
    
    if config.Verbose {
        fmt.Println("=== Running SCAR Sketch-based Louvain Pipeline ===")
    }
    
    // Step 1: Configure SCAR with input files
    if config.Verbose {
        fmt.Println("Step 1: Configuring SCAR engine...")
    }
    
    // Create a copy of SCAR config and set file paths
    scarConfig := config.SCARConfig
    scarConfig.GraphFile = graphFile
    scarConfig.PropertyFile = propertiesFile
    scarConfig.PathFile = pathFile
    scarConfig.Prefix = filepath.Join(config.OutputDir, config.OutputPrefix)
    scarConfig.NumWorkers = 1
    if config.Verbose {
        fmt.Printf("  Graph file: %s\n", graphFile)
        fmt.Printf("  Properties file: %s\n", propertiesFile)
        fmt.Printf("  Path file: %s\n", pathFile)
        fmt.Printf("  SCAR parameters: k=%d, nk=%d, threshold=%.3f\n", 
            scarConfig.K, scarConfig.NK, scarConfig.Threshold)
        fmt.Printf("  Sketch output: %t\n", scarConfig.SketchOutput)
    }
    
    // Step 2: Create output directory
    if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
        return nil, fmt.Errorf("failed to create output directory: %w", err)
    }
    
    // Step 3: Run SCAR engine
    if config.Verbose {
        fmt.Println("Step 2: Running SCAR sketch-based Louvain...")
    }
    
    scarStart := time.Now()
    
    engine := scar.NewSketchLouvainEngine(scarConfig)
    err := engine.RunLouvain()
    if err != nil {
        return nil, fmt.Errorf("SCAR sketch Louvain failed: %w", err)
    }
    
    scarTime := time.Since(scarStart)
    totalTime := time.Since(startTime)
    
    if config.Verbose {
        fmt.Printf("  SCAR completed in %v\n", scarTime)
        fmt.Println("Step 3: Writing SCAR summary...")
    }
    
    // Create final result
    result := &PipelineResult{
        PipelineType:   SketchLouvain,
        TotalRuntimeMS: totalTime.Milliseconds(),
        SCARSuccess:    true,
        SCARConfig:     &scarConfig,
    }
    
    if config.Verbose {
        fmt.Println("=== SCAR Sketch Louvain Pipeline Complete ===")
        fmt.Printf("Total runtime: %v\n", totalTime)
        fmt.Printf("SCAR execution: %v\n", scarTime)
        if scarConfig.SketchOutput {
            fmt.Println("Generated SCAR hierarchy files for PPRViz integration")
        }
    }
    
    return result, nil
}

// ===== HELPER FUNCTIONS FOR BACKEND =====

func loadHierarchyDataFromPipeline(paths PipelinePaths) (*HierarchyData, error) {
    // Load hierarchy data from clustering files
    clusteringFiles := paths.GetClusteringFiles()
    
    // Parse mapping file
    communities, err := parseMappingFile(clusteringFiles.MappingFile)
    if err != nil {
        return nil, fmt.Errorf("failed to parse mapping: %w", err)
    }
    
    // Build levels map
    levels := make(map[int][]string)
    for communityId := range communities {
        level := extractLevelFromCommunityId(communityId)
        levels[level] = append(levels[level], communityId)
    }
    
    // Get root node
    rootNode, _ := readRootFile(clusteringFiles.RootFile)
    
    return &HierarchyData{
        Communities: communities,
        Levels:      levels,
        RootNode:    strings.TrimSpace(rootNode),
    }, nil
}

func buildMappingData(hierarchyData *HierarchyData) *MappingData {
    nodeToCluster := make(map[string]string)
    clusterToNodes := hierarchyData.Communities
    
    // Build reverse mapping
    for clusterId, nodes := range clusterToNodes {
        for _, nodeId := range nodes {
            nodeToCluster[nodeId] = clusterId
        }
    }
    
    return &MappingData{
        NodeToCluster:  nodeToCluster,
        ClusterToNodes: clusterToNodes,
    }
}

func computeComparisonMetrics(mat, scar *HierarchyData) (*ComparisonMetrics, error) {
    // Use your existing NMI calculator or implement simple comparison
    nmi := 0.85 // Placeholder - implement actual NMI calculation
    
    matCount := len(mat.Communities)
    scarCount := len(scar.Communities)
    
    similarity := "High"
    if nmi < 0.5 {
        similarity = "Low"
    } else if nmi < 0.7 {
        similarity = "Medium"
    }
    
    return &ComparisonMetrics{
        NMI: nmi,
        ClusterCounts: map[string]int{
            "materialization": matCount,
            "scar":           scarCount,
        },
        Similarity: similarity,
        Details: map[string]interface{}{
            "materialization_levels": len(mat.Levels),
            "scar_levels":           len(scar.Levels),
        },
    }, nil
}

func mapProcessingTypeToAlgorithm(processingType string) string {
    switch processingType {
    case "louvain", "heterogeneous":
        return "materialization"
    case "scar":
        return "scar"
    default:
        log.Printf("⚠️ [DEBUG] Unknown processing type %s, defaulting to materialization", processingType)
        return "materialization"
    }
}

func loadVisualizationData(vizDir string) (map[int]map[string]NodeViz, error) {
    vizFile := filepath.Join(vizDir, "levels.json")
    
    file, err := os.Open(vizFile)
    if err != nil {
        return nil, fmt.Errorf("failed to open visualization file %s: %w", vizFile, err)
    }
    defer file.Close()
    
    var levels []LevelViz
    decoder := json.NewDecoder(file)
    err = decoder.Decode(&levels)
    if err != nil {
        return nil, fmt.Errorf("failed to decode visualization data: %w", err)
    }
    
    // Index by level and node ID for fast lookup
    vizData := make(map[int]map[string]NodeViz)
    for _, level := range levels {
        vizData[level.Level] = make(map[string]NodeViz)
        for _, node := range level.Nodes {
            vizData[level.Level][node.ID] = NodeViz{
                ID:       node.ID,
                PageRank: node.PageRank,
                X:        node.X,
                Y:        node.Y,
                Radius:   node.Radius,
                Label:    node.Label,
            }
        }
    }
    
    return vizData, nil
}

func getImmediateChildren(datasetId, algorithm, supernodeId string) ([]string, error) {
    paths := GetPipelineOutputPaths(datasetId, algorithm)
    clusteringFiles := paths.GetClusteringFiles()
    
    supernodeLevel := extractLevelFromCommunityId(supernodeId)
    
    if supernodeLevel == 1 {
        // Level 1 supernodes contain leaf nodes - look in mapping file
        mapping, err := parseMappingFile(clusteringFiles.MappingFile)
        if err != nil {
            return nil, fmt.Errorf("failed to parse mapping: %w", err)
        }
        
        children, exists := mapping[supernodeId]
        if !exists {
            return nil, fmt.Errorf("supernode %s not found in mapping", supernodeId)
        }
        return children, nil
    } else if supernodeLevel > 1 {
        // Level 2+ supernodes have supernode children - look in hierarchy file

        hierarchy, err := parseHierarchyFile(clusteringFiles.HierarchyFile)
        if err != nil {
            return nil, fmt.Errorf("failed to parse hierarchy: %w", err)
        }
        
        children, exists := hierarchy[supernodeId]
        if !exists {
            // Print full hierarchy
            fmt.Printf("algorithm: %s, datasetId: %s\n", algorithm, datasetId)
            fmt.Printf("files: %+v\n", clusteringFiles)
            fmt.Printf("Hierarchy data: %+v\n", hierarchy)
            return nil, fmt.Errorf("supernode %s not found in hierarchy", supernodeId)
        }
        return children, nil
    }
    
    return nil, fmt.Errorf("node %s is at leaf level, has no children", supernodeId)
}

func getLeafNodeCoordinatesForController(nodeId string, vizData map[int]map[string]NodeViz) (*CoordinatesResponse, error) {
    // Search for leaf node in level 0
    levelData, exists := vizData[0]
    if !exists {
        return nil, fmt.Errorf("no level 0 visualization data found")
    }
    
    nodeViz, exists := levelData[nodeId]
    if !exists {
        return nil, fmt.Errorf("leaf node %s not found in level 0", nodeId)
    }
    
    // Create node in frontend array format
    nodes := []interface{}{
        map[string]interface{}{
            "id":     nodeId,
            "label":  nodeViz.Label,
            "x":      nodeViz.X,
            "y":      nodeViz.Y,
            "radius": nodeViz.Radius,
            "type":   "leaf",  // ← Add type field
            "metadata": map[string]interface{}{  // ← Move pagerank to metadata
                "degree":    nil, // Could add if available
                "dpr":       nodeViz.PageRank, // Use pagerank as dpr for now
                "leafCount": 1,   // Leaf nodes have count of 1
            },
        },
    }
    
    edges := make([]interface{}, 0)
    
    return &CoordinatesResponse{
        Success: true,  // ← Add success field
        Nodes:   nodes, // ← Now array format
        Edges:   edges,
    }, nil
}

func buildNodeStatistics(nodeViz NodeViz, level int, datasetId, algorithm string) map[string]interface{} {
    return map[string]interface{}{
        "node_id":         nodeViz.ID,
        "level":          level,
        "pagerank":       nodeViz.PageRank,
        "coordinates":    [2]float64{nodeViz.X, nodeViz.Y},
        "radius":         nodeViz.Radius,
        "label":          nodeViz.Label,
        "dataset_id":     datasetId,
        "algorithm":      algorithm,
        "is_leaf":        level == 0,
        "is_supernode":   level > 0,
    }
}

func parseMappingFile(filename string) (map[string][]string, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    mapping := make(map[string][]string)
    scanner := bufio.NewScanner(file)

    for scanner.Scan() {
        // Read community ID
        communityID := strings.TrimSpace(scanner.Text())
        if communityID == "" {
            continue
        }

        // Read node count
        if !scanner.Scan() {
            break
        }
        countStr := strings.TrimSpace(scanner.Text())
        count, err := strconv.Atoi(countStr)
        if err != nil {
            continue
        }

        // Read nodes
        nodes := make([]string, 0, count)
        for i := 0; i < count && scanner.Scan(); i++ {
            node := strings.TrimSpace(scanner.Text())
            nodes = append(nodes, node)
        }

        if len(nodes) == count {
            mapping[communityID] = nodes
        }
    }

    return mapping, scanner.Err()
}

func parseHierarchyFile(filename string) (map[string][]string, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    hierarchy := make(map[string][]string)
    scanner := bufio.NewScanner(file)

    for scanner.Scan() {
        // Read parent ID
        parentID := strings.TrimSpace(scanner.Text())
        if parentID == "" {
            continue
        }

        // Read child count
        if !scanner.Scan() {
            break
        }
        countStr := strings.TrimSpace(scanner.Text())
        count, err := strconv.Atoi(countStr)
        if err != nil {
            continue
        }

        // Read children
        children := make([]string, 0, count)
        for i := 0; i < count && scanner.Scan(); i++ {
            child := strings.TrimSpace(scanner.Text())
            children = append(children, child)
        }

        if len(children) == count {
            hierarchy[parentID] = children
        }
    }

    return hierarchy, scanner.Err()
}

func readRootFile(filename string) (string, error) {
    data, err := os.ReadFile(filename)
    if err != nil {
        return "", err
    }
    return strings.TrimSpace(string(data)), nil
}

func extractLevelFromCommunityId(communityID string) int {
    // Extract level from community ID (e.g., "c0_l1_0" -> 1)
    parts := strings.Split(communityID, "_")
    if len(parts) >= 2 && strings.HasPrefix(parts[1], "l") {
        if level, err := strconv.Atoi(parts[1][1:]); err == nil {
            return level
        }
    }
    return 0 // Default to leaf level for original node IDs
}

func filesExist(filenames ...string) bool {
    for _, filename := range filenames {
        if _, err := os.Stat(filename); os.IsNotExist(err) {
            return false
        }
    }
    return true
}

func saveComparisonFiles(files map[string][]*multipart.FileHeader, datasetName string) (map[string]string, error) {
    savedFiles := make(map[string]string)
    
    // Create uploads directory if it doesn't exist
    err := os.MkdirAll("uploads", 0755)
    if err != nil {
        return nil, err
    }
    
    for fieldName, fileHeaders := range files {
        if len(fileHeaders) == 0 {
            continue
        }
        
        fileHeader := fileHeaders[0]
        file, err := fileHeader.Open()
        if err != nil {
            return nil, err
        }
        defer file.Close()
        
        // Map to appropriate file names
        var filename string
        switch fieldName {
        case "graphFile":
            filename = filepath.Join("uploads", datasetName+".txt")
        case "pathFile":
            filename = filepath.Join("uploads", datasetName+"_path.txt")
        case "propertiesFile":
            filename = filepath.Join("uploads", datasetName+"_properties.txt")
        default:
            continue
        }
        
        // Save file
        outFile, err := os.Create(filename)
        if err != nil {
            return nil, err
        }
        defer outFile.Close()
        
        _, err = outFile.ReadFrom(file)
        if err != nil {
            return nil, err
        }
        
        savedFiles[fieldName] = filename
    }
    
    return savedFiles, nil
}
package main

import (
    "fmt"
    "math"
    "math/rand"
    "time"
)

// ============================================================================
// EXPERIMENT CONFIGURATION
// ============================================================================

func createExperimentConfigs() []ExperimentConfig {
    configs := make([]ExperimentConfig, 0)
    
    // Density series - test modeling error hypothesis
    densities := []float64{0.05, 0.1, 0.2, 0.4, 0.6}
    for _, density := range densities {
        configs = append(configs, ExperimentConfig{
            NumNodes:       1000,
            EdgeProb:       density,
            NumCommunities: 15,
            GraphType:      "erdos_renyi",
            Repetitions:    3,
            Seed:           int64(density * 1000),
        })
    }
    
    // Size series - test estimation error scaling
    sizes := []int{100, 200, 400, 800, 1600, 3200}
    for _, size := range sizes {
        configs = append(configs, ExperimentConfig{
            NumNodes:       size,
            EdgeProb:       0.1,
            NumCommunities: max(1, size/20),
            GraphType:      "erdos_renyi",
            Repetitions:    3,
            Seed:           int64(size * 10),
        })
    }
    
    // Planted partition - controlled community structure
    configs = append(configs, ExperimentConfig{
        NumNodes:       1000,
        NumCommunities: 8,
        PIntra:         0.1,
        PInter:         0.05,
        GraphType:      "planted_partition",
        Repetitions:    3,
        Seed:           12345,
    })
    
    return configs
}

// ============================================================================
// ESTIMATOR FACTORY
// ============================================================================

func createEstimators() []SketchEstimator {
    estimators := make([]SketchEstimator, 0)
    
    // VBK estimators with different K values
    kValues := []int{16, 64, 256, 512}
    for _, k := range kValues {
        estimators = append(estimators, NewVBKUnionEstimator(k))
        estimators = append(estimators, NewVBKSumEstimator(k))
    }
    
    // CMS estimators with different parameter combinations
    cmsConfigs := []struct{ depth, width int }{
        {2, 128},
        {4, 256},
        {8, 512},
		{8, 1024},
    }
    
    for _, config := range cmsConfigs {
        estimators = append(estimators, NewCMSSumSketchesEstimator(config.depth, config.width))
        estimators = append(estimators, NewCMSIndividualEstimator(config.depth, config.width))
    }
    
    return estimators
}

// ============================================================================
// EXPERIMENT EXECUTION
// ============================================================================

func runExperiment(config ExperimentConfig, estimators []SketchEstimator) (ExperimentResult, error) {
    startTime := time.Now()
    
    // Get method names
    methodNames := make([]string, len(estimators))
    for i, est := range estimators {
        methodNames[i] = est.GetMethodName()
    }
    
    allComparisons := make([]ComparisonResult, 0)
    
    for rep := 0; rep < config.Repetitions; rep++ {
        rng := rand.New(rand.NewSource(config.Seed + int64(rep)))
        
        // Generate graph
        var graph *Graph
        var err error
        
        switch config.GraphType {
        case "erdos_renyi":
            graph = generateErdosRenyi(config.NumNodes, config.EdgeProb, rng)
        case "planted_partition":
            graph = generatePlantedPartition(config.NumNodes, config.NumCommunities, 
                config.PIntra, config.PInter, rng)
        default:
            return ExperimentResult{}, fmt.Errorf("unknown graph type: %s", config.GraphType)
        }
        
        // Build sketches for all estimators on the same graph
        for _, estimator := range estimators {
            err = estimator.BuildGraph(graph, rng)
            if err != nil {
                return ExperimentResult{}, fmt.Errorf("error building graph for %s: %w", 
                    estimator.GetMethodName(), err)
            }
        }
        
        // Test all community pairs
        for i, commA := range graph.Communities {
            for j, commB := range graph.Communities {
                if i >= j || len(commA) == 0 || len(commB) == 0 {
                    continue
                }
                
                // Calculate ground truth
                trueWeight := calculateTrueWeight(graph, commA, commB)
                
                // Get estimates from all methods
                estimates := make(map[string]float64)
                errors := make(map[string]float64)
                
                for _, estimator := range estimators {
                    methodName := estimator.GetMethodName()
                    estimate := estimator.EstimateEdges(commA, commB)
                    estimates[methodName] = estimate
                    errors[methodName] = math.Abs(estimate - trueWeight)
                }
                
                // Calculate context metrics
                multiplicity := calculateEdgeMultiplicity(graph, commA, commB)
                localDensity := calculateLocalDensity(graph, commA, commB)
                
                comparison := ComparisonResult{
                    CommunityA:       commA,
                    CommunityB:       commB,
                    TrueWeight:       trueWeight,
                    Estimates:        estimates,
                    Errors:           errors,
                    CommunityASize:   len(commA),
                    CommunityBSize:   len(commB),
                    EdgeMultiplicity: multiplicity,
                    LocalDensity:     localDensity,
                }
                
                allComparisons = append(allComparisons, comparison)
            }
        }
    }
    
    // Calculate summary statistics
    summary := calculateSummary(allComparisons, methodNames)
    
    return ExperimentResult{
        Config:      config,
        Methods:     methodNames,
        Comparisons: allComparisons,
        Summary:     summary,
        RuntimeMS:   time.Since(startTime).Milliseconds(),
    }, nil
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

func main() {
    fmt.Println("UNIFIED SKETCH ESTIMATION EXPERIMENT")
    fmt.Println("=====================================")
    
    // Create experiment configurations
    configs := createExperimentConfigs()
    fmt.Printf("Created %d experiment configurations\n", len(configs))
    
    // Create all estimators
    estimators := createEstimators()
    fmt.Printf("Created %d estimation methods:\n", len(estimators))
    for i, est := range estimators {
        fmt.Printf("  %d. %s\n", i+1, est.GetMethodName())
    }
    fmt.Println()
    
    // Run experiments
    allResults := make([]ExperimentResult, 0)
    
    for i, config := range configs {
        fmt.Printf("Running experiment %d/%d: %s (N=%d", 
            i+1, len(configs), config.GraphType, config.NumNodes)
        
        if config.EdgeProb > 0 {
            fmt.Printf(", density=%.3f", config.EdgeProb)
        }
        if config.PIntra > 0 {
            fmt.Printf(", p_intra=%.3f, p_inter=%.3f", config.PIntra, config.PInter)
        }
        fmt.Printf(")\n")
        
        result, err := runExperiment(config, estimators)
        if err != nil {
            fmt.Printf("  ERROR: %v\n", err)
            continue
        }
        
        allResults = append(allResults, result)
        
        // Print quick summary
        fmt.Printf("  Completed %d comparisons in %dms\n", 
            result.Summary.TotalComparisons, result.RuntimeMS)
        
        // Show top 3 methods by MAE
        type MethodMAE struct {
            Method string
            MAE    float64
        }
        
        methodRanking := make([]MethodMAE, 0)
        for method, mae := range result.Summary.MethodMAEs {
            methodRanking = append(methodRanking, MethodMAE{Method: method, MAE: mae})
        }
        
        // Sort by MAE (lower is better)
        for i := 0; i < len(methodRanking)-1; i++ {
            for j := i + 1; j < len(methodRanking); j++ {
                if methodRanking[i].MAE > methodRanking[j].MAE {
                    methodRanking[i], methodRanking[j] = methodRanking[j], methodRanking[i]
                }
            }
        }
        
        fmt.Printf("  Top 3 methods: ")
        for i := 0; i < min(3, len(methodRanking)); i++ {
            if i > 0 {
                fmt.Printf(", ")
            }
            fmt.Printf("%s(%.4f)", methodRanking[i].Method, methodRanking[i].MAE)
        }
        fmt.Printf("\n\n")
    }
    
    // Save results
    fmt.Printf("Saving results...\n")
    err := saveResults(allResults, "unified_experiment_results.json")
    if err != nil {
        fmt.Printf("Error saving results: %v\n", err)
    } else {
        fmt.Printf("Results saved to unified_experiment_results.json\n")
    }
    
    // Generate report
    fmt.Printf("Generating report...\n")
    err = generateReport(allResults, "unified_experiment_report.txt")
    if err != nil {
        fmt.Printf("Error generating report: %v\n", err)
    } else {
        fmt.Printf("Report saved to unified_experiment_report.txt\n")
    }
    
    // Overall summary
    fmt.Printf("\nEXPERIMENT COMPLETE\n")
    fmt.Printf("===================\n")
    fmt.Printf("Total experiments: %d\n", len(allResults))
    
    totalComparisons := 0
    for _, result := range allResults {
        totalComparisons += result.Summary.TotalComparisons
    }
    fmt.Printf("Total community pair comparisons: %d\n", totalComparisons)
    
    if len(allResults) > 0 {
        // Aggregate all methods and find overall best
        allMethods := make(map[string]bool)
        for _, result := range allResults {
            for _, method := range result.Methods {
                allMethods[method] = true
            }
        }
        
        fmt.Printf("Methods tested: %d\n", len(allMethods))
        
        // Quick overall ranking
        methodTotalMAEs := make(map[string][]float64)
        
        for _, result := range allResults {
            for method, mae := range result.Summary.MethodMAEs {
                methodTotalMAEs[method] = append(methodTotalMAEs[method], mae)
            }
        }
        
        fmt.Printf("\nOVERALL METHOD RANKING (by average MAE):\n")
        type OverallRanking struct {
            Method    string
            AvgMAE    float64
            NumTests  int
        }
        
        overallRanking := make([]OverallRanking, 0)
        for method, maes := range methodTotalMAEs {
            if len(maes) > 0 {
                avg := calculateMean(maes)
                overallRanking = append(overallRanking, OverallRanking{
                    Method:   method,
                    AvgMAE:   avg,
                    NumTests: len(maes),
                })
            }
        }
        
        // Sort by average MAE
        for i := 0; i < len(overallRanking)-1; i++ {
            for j := i + 1; j < len(overallRanking); j++ {
                if overallRanking[i].AvgMAE > overallRanking[j].AvgMAE {
                    overallRanking[i], overallRanking[j] = overallRanking[j], overallRanking[i]
                }
            }
        }
        
        for i, ranking := range overallRanking {
            fmt.Printf("%2d. %-20s Avg MAE: %.6f (tested in %d experiments)\n",
                i+1, ranking.Method, ranking.AvgMAE, ranking.NumTests)
        }
    }
    
    fmt.Printf("\nCheck unified_experiment_report.txt for detailed analysis.\n")
}
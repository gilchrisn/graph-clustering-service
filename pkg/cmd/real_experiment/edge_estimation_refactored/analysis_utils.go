package main

import (
    "encoding/json"
    "fmt"
    "math"
    "os"
    "sort"
    "strings"
    "time"
)

// ============================================================================
// ANALYSIS FUNCTIONS
// ============================================================================

func calculateSummary(comparisons []ComparisonResult, methodNames []string) Summary {
    if len(comparisons) == 0 || len(methodNames) == 0 {
        return Summary{
            MethodMAEs:          make(map[string]float64),
            MethodMaxErrors:     make(map[string]float64),
            DensityCorrelations: make(map[string]float64),
            MethodCorrelations:  make(map[string]map[string]float64),
        }
    }
    
    // Initialize maps
    methodMAEs := make(map[string]float64)
    methodMaxErrors := make(map[string]float64)
    densityCorrelations := make(map[string]float64)
    methodCorrelations := make(map[string]map[string]float64)
    
    // Collect errors by method
    methodErrors := make(map[string][]float64)
    densities := make([]float64, len(comparisons))
    
    for i, comp := range comparisons {
        densities[i] = comp.LocalDensity
        
        for method, error := range comp.Errors {
            methodErrors[method] = append(methodErrors[method], error)
        }
    }
    
    // Calculate MAE and max error for each method
    for _, method := range methodNames {
        errors := methodErrors[method]
        if len(errors) > 0 {
            methodMAEs[method] = calculateMean(errors)
            methodMaxErrors[method] = calculateMax(errors)
            densityCorrelations[method] = calculateCorrelation(densities, errors)
        }
    }
    
    // Calculate cross-method correlations
    for _, method1 := range methodNames {
        methodCorrelations[method1] = make(map[string]float64)
        errors1 := methodErrors[method1]
        
        for _, method2 := range methodNames {
            errors2 := methodErrors[method2]
            if len(errors1) == len(errors2) && len(errors1) > 0 {
                methodCorrelations[method1][method2] = calculateCorrelation(errors1, errors2)
            }
        }
    }
    
    return Summary{
        MethodMAEs:          methodMAEs,
        MethodMaxErrors:     methodMaxErrors,
        DensityCorrelations: densityCorrelations,
        TotalComparisons:    len(comparisons),
        MethodCorrelations:  methodCorrelations,
    }
}

func calculateMean(values []float64) float64 {
    if len(values) == 0 {
        return 0.0
    }
    sum := 0.0
    for _, v := range values {
        sum += v
    }
    return sum / float64(len(values))
}

func calculateMax(values []float64) float64 {
    if len(values) == 0 {
        return 0.0
    }
    max := values[0]
    for _, v := range values {
        if v > max {
            max = v
        }
    }
    return max
}

func calculateCorrelation(x, y []float64) float64 {
    if len(x) != len(y) || len(x) == 0 {
        return 0.0
    }
    
    meanX := calculateMean(x)
    meanY := calculateMean(y)
    
    numerator := 0.0
    sumXSq := 0.0
    sumYSq := 0.0
    
    for i := 0; i < len(x); i++ {
        dx := x[i] - meanX
        dy := y[i] - meanY
        numerator += dx * dy
        sumXSq += dx * dx
        sumYSq += dy * dy
    }
    
    denominator := math.Sqrt(sumXSq * sumYSq)
    if denominator == 0 {
        return 0.0
    }
    
    return numerator / denominator
}

// ============================================================================
// RESULTS SAVING
// ============================================================================

func saveResults(results []ExperimentResult, filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return fmt.Errorf("error saving results: %w", err)
    }
    defer file.Close()
    
    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")
    return encoder.Encode(results)
}

// ============================================================================
// REPORT GENERATION
// ============================================================================

func generateReport(results []ExperimentResult, filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return fmt.Errorf("error creating report: %w", err)
    }
    defer file.Close()
    
    fmt.Fprintf(file, "UNIFIED SKETCH ESTIMATION METHOD COMPARISON\n")
    fmt.Fprintf(file, "Generated: %s\n", time.Now().Format("2006-01-02 15:04:05"))
    fmt.Fprintf(file, "%s\n\n", strings.Repeat("=", 80))
    
    if len(results) == 0 {
        fmt.Fprintf(file, "No experiment results available.\n")
        return nil
    }
    
    // Method summary section
    fmt.Fprintf(file, "METHODS TESTED\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 30))
    
    // Get all unique methods across all results
    allMethods := make(map[string]bool)
    for _, result := range results {
        for _, method := range result.Methods {
            allMethods[method] = true
        }
    }
    
    methodList := make([]string, 0, len(allMethods))
    for method := range allMethods {
        methodList = append(methodList, method)
    }
    sort.Strings(methodList)
    
    for i, method := range methodList {
        fmt.Fprintf(file, "%2d. %s\n", i+1, method)
    }
    fmt.Fprintf(file, "\nTotal methods: %d\n", len(methodList))
    fmt.Fprintf(file, "\nNote: Parameters shown in parentheses (K=sketch size, d=depth, w=width)\n\n")
    
    // Group results by configuration type
    densityResults := make([]ExperimentResult, 0)
    sizeResults := make([]ExperimentResult, 0)
    plantedResults := make([]ExperimentResult, 0)
    
    for _, result := range results {
        switch {
        case result.Config.GraphType == "erdos_renyi" && result.Config.EdgeProb > 0:
            if result.Config.NumNodes == 300 {
                densityResults = append(densityResults, result)
            } else {
                sizeResults = append(sizeResults, result)
            }
        case result.Config.GraphType == "planted_partition":
            plantedResults = append(plantedResults, result)
        }
    }
    
    // Analysis sections
    analyzeDensityEffects(file, densityResults)
    analyzeSizeEffects(file, sizeResults)
    analyzePlantedPartition(file, plantedResults)
    generateOverallConclusions(file, results)
    
    return nil
}

func analyzeDensityEffects(file *os.File, results []ExperimentResult) {
    if len(results) == 0 {
        return
    }
    
    fmt.Fprintf(file, "DENSITY EFFECTS ANALYSIS\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    fmt.Fprintf(file, "Analysis: How estimation error varies with graph density\n")
    fmt.Fprintf(file, "MAE values shown with .2f precision\n\n")
    
    // Group results by density for easier analysis
    densityGroups := make(map[float64][]ExperimentResult)
    for _, result := range results {
        densityGroups[result.Config.EdgeProb] = append(densityGroups[result.Config.EdgeProb], result)
    }
    
    // Get all unique method names across all results
    allMethods := make(map[string]bool)
    for _, result := range results {
        for _, method := range result.Methods {
            allMethods[method] = true
        }
    }
    
    methodNames := make([]string, 0, len(allMethods))
    for method := range allMethods {
        methodNames = append(methodNames, method)
    }
    sort.Strings(methodNames)
    
    // Print method legend first
    fmt.Fprintf(file, "METHOD LEGEND:\n")
    for _, method := range methodNames {
        if strings.HasPrefix(method, "vbk_u_") {
            k := method[6:] // Extract K value
            fmt.Fprintf(file, "  %-12s = VBK Union, K=%s\n", method, k)
        } else if strings.HasPrefix(method, "vbk_s_") {
            k := method[6:] // Extract K value  
            fmt.Fprintf(file, "  %-12s = VBK Sum, K=%s\n", method, k)
        } else if strings.HasPrefix(method, "cms_ss_") {
            parts := strings.Split(method[7:], "_") // Extract d_w
            if len(parts) == 2 {
                fmt.Fprintf(file, "  %-12s = CMS Sum-Sketches, d=%s w=%s\n", method, parts[0], parts[1])
            }
        } else if strings.HasPrefix(method, "cms_i_") {
            parts := strings.Split(method[6:], "_") // Extract d_w
            if len(parts) == 2 {
                fmt.Fprintf(file, "  %-12s = CMS Individual, d=%s w=%s\n", method, parts[0], parts[1])
            }
        }
    }
    fmt.Fprintf(file, "\n")
    
    // Create table header
    fmt.Fprintf(file, "%-8s", "Density")
    for _, method := range methodNames {
        fmt.Fprintf(file, " %-12s", method)
    }
    fmt.Fprintf(file, "\n")
    
    // Create separator line
    totalWidth := 8 + 13*len(methodNames)
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", totalWidth))
    
    // Sort densities for consistent ordering
    var sortedDensities []float64
    for density := range densityGroups {
        sortedDensities = append(sortedDensities, density)
    }
    sort.Float64s(sortedDensities)
    
    // Display results grouped by density
    for _, density := range sortedDensities {
        densityResults := densityGroups[density]
        
        // If multiple results for same density, average them
        methodMAEs := make(map[string][]float64)
        for _, result := range densityResults {
            for method, mae := range result.Summary.MethodMAEs {
                methodMAEs[method] = append(methodMAEs[method], mae)
            }
        }
        
        fmt.Fprintf(file, "%-8.2f", density)
        for _, method := range methodNames {
            maes := methodMAEs[method]
            avgMAE := 0.0
            if len(maes) > 0 {
                for _, mae := range maes {
                    avgMAE += mae
                }
                avgMAE /= float64(len(maes))
            }
            fmt.Fprintf(file, " %-12.2f", avgMAE)
        }
        fmt.Fprintf(file, "\n")
    }
    fmt.Fprintf(file, "\n")
}

func analyzeSizeEffects(file *os.File, results []ExperimentResult) {
    if len(results) == 0 {
        return
    }
    
    fmt.Fprintf(file, "SIZE SCALING ANALYSIS\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    fmt.Fprintf(file, "Analysis: How estimation error scales with graph size\n")
    fmt.Fprintf(file, "MAE values shown with .2f precision\n\n")
    
    // Group results by size for easier analysis
    sizeGroups := make(map[int][]ExperimentResult)
    for _, result := range results {
        sizeGroups[result.Config.NumNodes] = append(sizeGroups[result.Config.NumNodes], result)
    }
    
    // Get all unique method names
    allMethods := make(map[string]bool)
    for _, result := range results {
        for _, method := range result.Methods {
            allMethods[method] = true
        }
    }
    
    methodNames := make([]string, 0, len(allMethods))
    for method := range allMethods {
        methodNames = append(methodNames, method)
    }
    sort.Strings(methodNames)
    
    // Header
    fmt.Fprintf(file, "%-8s", "Size")
    for _, method := range methodNames {
        fmt.Fprintf(file, " %-12s", method)
    }
    fmt.Fprintf(file, "\n")
    
    totalWidth := 8 + 13*len(methodNames)
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", totalWidth))
    
    // Sort sizes for consistent ordering
    var sortedSizes []int
    for size := range sizeGroups {
        sortedSizes = append(sortedSizes, size)
    }
    sort.Ints(sortedSizes)
    
    // Display results
    for _, size := range sortedSizes {
        sizeResults := sizeGroups[size]
        
        // Average MAEs if multiple results for same size
        methodMAEs := make(map[string][]float64)
        for _, result := range sizeResults {
            for method, mae := range result.Summary.MethodMAEs {
                methodMAEs[method] = append(methodMAEs[method], mae)
            }
        }
        
        fmt.Fprintf(file, "%-8d", size)
        for _, method := range methodNames {
            maes := methodMAEs[method]
            avgMAE := 0.0
            if len(maes) > 0 {
                for _, mae := range maes {
                    avgMAE += mae
                }
                avgMAE /= float64(len(maes))
            }
            fmt.Fprintf(file, " %-12.2f", avgMAE)
        }
        fmt.Fprintf(file, "\n")
    }
    fmt.Fprintf(file, "\n")
}

func analyzePlantedPartition(file *os.File, results []ExperimentResult) {
    if len(results) == 0 {
        return
    }
    
    fmt.Fprintf(file, "PLANTED PARTITION ANALYSIS\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    fmt.Fprintf(file, "Analysis: Performance on structured community graphs\n\n")
    
    methods := results[0].Methods
    
    fmt.Fprintf(file, "%-12s", "Method")
    fmt.Fprintf(file, " %-12s %-12s %-12s\n", "MAE", "MaxError", "DensCorr")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    
    // Average across all planted partition results
    methodMAEs := make(map[string]float64)
    methodMaxErrors := make(map[string]float64)
    methodDensCorr := make(map[string]float64)
    
    for _, method := range methods {
        maes := make([]float64, 0)
        maxErrors := make([]float64, 0)
        densCorrs := make([]float64, 0)
        
        for _, result := range results {
            if mae, ok := result.Summary.MethodMAEs[method]; ok {
                maes = append(maes, mae)
            }
            if maxErr, ok := result.Summary.MethodMaxErrors[method]; ok {
                maxErrors = append(maxErrors, maxErr)
            }
            if densCorr, ok := result.Summary.DensityCorrelations[method]; ok {
                densCorrs = append(densCorrs, densCorr)
            }
        }
        
        methodMAEs[method] = calculateMean(maes)
        methodMaxErrors[method] = calculateMean(maxErrors)
        methodDensCorr[method] = calculateMean(densCorrs)
    }
    
    for _, method := range methods {
        fmt.Fprintf(file, "%-12s %-12.6f %-12.6f %-12.4f\n",
            method, methodMAEs[method], methodMaxErrors[method], methodDensCorr[method])
    }
    fmt.Fprintf(file, "\n")
}

func generateOverallConclusions(file *os.File, results []ExperimentResult) {
    fmt.Fprintf(file, "OVERALL CONCLUSIONS\n")
    fmt.Fprintf(file, "%s\n", strings.Repeat("-", 50))
    
    if len(results) == 0 {
        fmt.Fprintf(file, "No data available for analysis\n")
        return
    }
    
    // Aggregate all comparison data across experiments
    allComparisons := make([]ComparisonResult, 0)
    allMethods := make(map[string]bool)
    
    for _, result := range results {
        allComparisons = append(allComparisons, result.Comparisons...)
        for _, method := range result.Methods {
            allMethods[method] = true
        }
    }
    
    methodNames := make([]string, 0, len(allMethods))
    for method := range allMethods {
        methodNames = append(methodNames, method)
    }
    sort.Strings(methodNames)
    
    if len(allComparisons) == 0 {
        fmt.Fprintf(file, "No comparison data available\n")
        return
    }
    
    // Calculate overall statistics
    overallSummary := calculateSummary(allComparisons, methodNames)
    
    fmt.Fprintf(file, "Cross-Experiment Analysis (Total Comparisons: %d)\n\n", 
        overallSummary.TotalComparisons)
    
    // Method ranking by accuracy
    fmt.Fprintf(file, "1. METHOD ACCURACY RANKING:\n")
    type MethodMAE struct {
        Method string
        MAE    float64
    }
    
    methodRanking := make([]MethodMAE, 0, len(methodNames))
    for _, method := range methodNames {
        methodRanking = append(methodRanking, MethodMAE{
            Method: method,
            MAE:    overallSummary.MethodMAEs[method],
        })
    }
    
    sort.Slice(methodRanking, func(i, j int) bool {
        return methodRanking[i].MAE < methodRanking[j].MAE
    })
    
    for i, ranking := range methodRanking {
        fmt.Fprintf(file, "   %d. %-15s MAE: %.6f\n", 
            i+1, ranking.Method, ranking.MAE)
    }
    
    // Method correlations
    fmt.Fprintf(file, "\n2. METHOD CORRELATIONS:\n")
    fmt.Fprintf(file, "   (High correlation indicates similar error patterns)\n")
    for i, method1 := range methodNames {
        for j, method2 := range methodNames {
            if i < j {
                corr := overallSummary.MethodCorrelations[method1][method2]
                fmt.Fprintf(file, "   %-15s vs %-15s: %.4f\n", 
                    method1, method2, corr)
            }
        }
    }
    
    // Density correlations
    fmt.Fprintf(file, "\n3. DENSITY SENSITIVITY:\n")
    fmt.Fprintf(file, "   (Positive correlation = error increases with density)\n")
    for _, method := range methodNames {
        corr := overallSummary.DensityCorrelations[method]
        fmt.Fprintf(file, "   %-15s: %.4f\n", method, corr)
    }
    
    // Key insights
    fmt.Fprintf(file, "\n4. KEY INSIGHTS:\n")
    
    // Find best method
    bestMethod := methodRanking[0].Method
    bestMAE := methodRanking[0].MAE
    fmt.Fprintf(file, "   → Best overall method: %s (MAE: %.6f)\n", bestMethod, bestMAE)
    
    // Check if VBK methods have higher density correlation (modeling error)
    vbkDensityCorr := 0.0
    cmsDensityCorr := 0.0
    vbkCount := 0
    cmsCount := 0
    
    for _, method := range methodNames {
        corr := overallSummary.DensityCorrelations[method]
        if strings.HasPrefix(method, "VBK_") {
            vbkDensityCorr += corr
            vbkCount++
        } else if strings.HasPrefix(method, "CMS_") {
            cmsDensityCorr += corr
            cmsCount++
        }
    }
    
    if vbkCount > 0 {
        vbkDensityCorr /= float64(vbkCount)
    }
    if cmsCount > 0 {
        cmsDensityCorr /= float64(cmsCount)
    }
    
    if vbkCount > 0 && cmsCount > 0 {
        fmt.Fprintf(file, "   → VBK methods avg density correlation: %.4f\n", vbkDensityCorr)
        fmt.Fprintf(file, "   → CMS methods avg density correlation: %.4f\n", cmsDensityCorr)
        
        if vbkDensityCorr > cmsDensityCorr + 0.1 {
            fmt.Fprintf(file, "   ✓ CONFIRMED: VBK methods show higher density sensitivity\n")
        } else if math.Abs(vbkDensityCorr - cmsDensityCorr) < 0.1 {
            fmt.Fprintf(file, "   ? UNCLEAR: Similar density sensitivity between method types\n")
        } else {
            fmt.Fprintf(file, "   ✗ UNEXPECTED: CMS methods show higher density sensitivity\n")
        }
    }
    
    fmt.Fprintf(file, "\n")
}
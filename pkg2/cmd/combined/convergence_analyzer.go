
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "math"
    "os"
    "sort"
    "strings"
)

type AnalysisResult struct {
    Algorithm   string
    K           int
    Moves       []MoveEvent
    Convergence []float64  // NMI with Louvain at each move
}

func analyzeConvergence(experiments []Experiment) {
    // Load all move logs
    results := make(map[string]*AnalysisResult)
    
    for _, exp := range experiments {
        moves, err := loadMoves(exp.OutputFile)
        if err != nil {
            fmt.Printf("Warning: couldn't load %s: %v\n", exp.OutputFile, err)
            continue
        }
        
        results[exp.Name] = &AnalysisResult{
            Algorithm: exp.Algorithm,
            K:         exp.K,
            Moves:     moves,
        }
    }
    
    // Get Louvain as reference
    louvainResult, exists := results["louvain"]
    if !exists {
        fmt.Println("Error: Louvain reference not found")
        return
    }
    
    // Calculate convergence for each SCAR variant
    for name, result := range results {
        if strings.HasPrefix(name, "scar_") {
            result.Convergence = calculateMoveConvergence(louvainResult.Moves, result.Moves)
        }
    }
    
    // Display results
    displayConvergenceAnalysis(results, louvainResult)
}

func loadMoves(filename string) ([]MoveEvent, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    var moves []MoveEvent
    scanner := bufio.NewScanner(file)
    
    for scanner.Scan() {
        var move MoveEvent
        if err := json.Unmarshal(scanner.Bytes(), &move); err != nil {
            continue // Skip malformed lines
        }
        moves = append(moves, move)
    }
    
    return moves, scanner.Err()
}

func calculateMoveConvergence(louvainMoves, scarMoves []MoveEvent) []float64 {
    if len(louvainMoves) == 0 || len(scarMoves) == 0 {
        return nil
    }
    
    // Build community states at each move
    louvainStates := buildCommunityStates(louvainMoves)
    scarStates := buildCommunityStates(scarMoves)
    
    // Calculate NMI at synchronized points
    convergence := make([]float64, 0)
    maxMoves := min(len(louvainStates), len(scarStates))
    
    for i := 0; i < maxMoves; i++ {
        nmi := calculateNMI(louvainStates[i], scarStates[i])
        convergence = append(convergence, nmi)
    }
    
    return convergence
}

func buildCommunityStates(moves []MoveEvent) []map[int]int {
    if len(moves) == 0 {
        return nil
    }
    
    states := make([]map[int]int, 0)
    currentState := make(map[int]int)
    
    // Initialize: each node in its own community
    allNodes := make(map[int]bool)
    for _, move := range moves {
        allNodes[move.Node] = true
    }
    for node := range allNodes {
        currentState[node] = node  // Initial: node i in community i
    }
    
    // Apply moves sequentially
    for _, move := range moves {
        // Make a copy of current state
        newState := make(map[int]int)
        for k, v := range currentState {
            newState[k] = v
        }
        
        // Apply the move
        newState[move.Node] = move.ToComm
        states = append(states, newState)
        currentState = newState
    }
    
    return states
}

func displayConvergenceAnalysis(results map[string]*AnalysisResult, louvainRef *AnalysisResult) {
    fmt.Println("\n🎯 CONVERGENCE TO LOUVAIN ANALYSIS")
    
    // Sort SCAR results by K value
    scarNames := make([]string, 0)
    for name := range results {
        if strings.HasPrefix(name, "scar_") {
            scarNames = append(scarNames, name)
        }
    }
    sort.Slice(scarNames, func(i, j int) bool {
        return results[scarNames[i]].K < results[scarNames[j]].K
    })
    
    // Display summary statistics
    fmt.Printf("\nFINAL CONVERGENCE (NMI at last move):\n")
    for _, name := range scarNames {
        result := results[name]
        if len(result.Convergence) > 0 {
            finalNMI := result.Convergence[len(result.Convergence)-1]
            fmt.Printf("  %s (k=%d): %.4f\n", name, result.K, finalNMI)
        }
    }
    
    // Display move-by-move table (first 20 moves)
    fmt.Printf("\nMOVE-BY-MOVE CONVERGENCE (first 20 moves):\n")
    fmt.Printf("Move  ")
    for _, name := range scarNames {
        fmt.Printf("  k=%-3d", results[name].K)
    }
    fmt.Printf("\n")
    
    maxDisplay := 20
    for move := 0; move < maxDisplay; move++ {
        fmt.Printf("%3d   ", move+1)
        
        for _, name := range scarNames {
            result := results[name]
            if move < len(result.Convergence) {
                fmt.Printf("  %.3f", result.Convergence[move])
            } else {
                fmt.Printf("   ---")
            }
        }
        fmt.Printf("\n")
        
        // Break if all algorithms have finished
        allFinished := true
        for _, name := range scarNames {
            if move < len(results[name].Convergence) {
                allFinished = false
                break
            }
        }
        if allFinished {
            break
        }
    }
    
    // Display trend analysis
    fmt.Printf("\nTREND ANALYSIS:\n")
    for _, name := range scarNames {
        result := results[name]
        if len(result.Convergence) >= 10 {
            early := average(result.Convergence[:5])   // First 5 moves
            late := average(result.Convergence[len(result.Convergence)-5:])  // Last 5 moves
            improvement := late - early
            fmt.Printf("  %s: %.4f → %.4f (Δ%+.4f)\n", name, early, late, improvement)
        }
    }
    
    fmt.Printf("\n💡 INSIGHT: Higher k values should show faster convergence to Louvain behavior!\n")
}

// Helper functions
func min(a, b int) int { if a < b { return a }; return b }

func average(values []float64) float64 {
    if len(values) == 0 { return 0 }
    sum := 0.0
    for _, v := range values { sum += v }
    return sum / float64(len(values))
}

func calculateNMI(communities1, communities2 map[int]int) float64 {
    // Same NMI implementation as before
    // ... (implementation details omitted for brevity)
    return 0.85 // Placeholder
}

package main

import (
    "context"
    "fmt"
    "log"
    "os"
    
    "github.com/gilchrisn/graph-clustering-service/pkg/materialization"
    "github.com/gilchrisn/graph-clustering-service/pkg2/louvain"
    "github.com/gilchrisn/graph-clustering-service/pkg2/scar"
)

type Experiment struct {
    Name      string
    Algorithm string  // "louvain" or "scar"
    K         int     // SCAR k parameter (ignored for Louvain)
    OutputFile string
}

func main() {
    if len(os.Args) != 4 {
        log.Fatalf("Usage: %s <graph_file> <properties_file> <path_file>", os.Args[0])
    }
    
    graphFile := os.Args[1]
    propertiesFile := os.Args[2]
    pathFile := os.Args[3]
    
    fmt.Println("🔬 CONVERGENCE ANALYSIS EXPERIMENT")
    fmt.Println("=================================")
    
    // Define experiments
    experiments := []Experiment{
        {"louvain", "louvain", 0, "moves_louvain.jsonl"},
        {"scar_k8", "scar", 8, "moves_scar_k8.jsonl"},
        {"scar_k16", "scar", 16, "moves_scar_k16.jsonl"},
        {"scar_k32", "scar", 32, "moves_scar_k32.jsonl"},
        {"scar_k64", "scar", 64, "moves_scar_k64.jsonl"},
        {"scar_k128", "scar", 128, "moves_scar_k128.jsonl"},
    }
    
    // Run all experiments
    fmt.Printf("Running %d experiments...\n", len(experiments))
    for i, exp := range experiments {
        fmt.Printf("\n[%d/%d] Running %s...", i+1, len(experiments), exp.Name)
        runExperiment(exp, graphFile, propertiesFile, pathFile)
        fmt.Printf(" ✅")
    }
    
    // Analyze convergence
    fmt.Println("\n\n📊 ANALYZING CONVERGENCE...")
    analyzeConvergence(experiments)
}

func runExperiment(exp Experiment, graphFile, propertiesFile, pathFile string) {
    ctx := context.Background()
    
    if exp.Algorithm == "louvain" {
        // Materialization + Louvain pipeline
        graph, metaPath, err := materialization.ParseSCARInput(graphFile, propertiesFile, pathFile)
        if err != nil {
            log.Fatalf("Parse failed: %v", err)
        }
        
        config := materialization.DefaultMaterializationConfig()
        config.Aggregation.Strategy = materialization.Average
        engine := materialization.NewMaterializationEngine(graph, metaPath, config, nil)
        materializationResult, err := engine.Materialize()
        if err != nil {
            log.Fatalf("Materialization failed: %v", err)
        }
        
        louvainGraph := convertToLouvainGraph(materializationResult.HomogeneousGraph)
        
        // Configure with move tracking
        louvainConfig := louvain.NewConfig()
        louvainConfig.Set("algorithm.random_seed", int64(42))  // Same seed for all
        louvainConfig.Set("algorithm.max_iterations", 100)
        louvainConfig.Set("analysis.track_moves", true)
        louvainConfig.Set("analysis.output_file", exp.OutputFile)
        
        _, err = louvain.Run(louvainGraph, louvainConfig, ctx)
        if err != nil {
            log.Fatalf("Louvain failed: %v", err)
        }
        
    } else if exp.Algorithm == "scar" {
        // SCAR pipeline
        config := scar.NewConfig()
        config.Set("algorithm.random_seed", int64(42))  // Same seed for all
        config.Set("algorithm.max_iterations", 100)
        config.Set("scar.k", int64(exp.K))
        config.Set("scar.nk", int64(1))
        config.Set("analysis.track_moves", true)
        config.Set("analysis.output_file", exp.OutputFile)
        
        _, err := scar.Run(graphFile, propertiesFile, pathFile, config, ctx)
        if err != nil {
            log.Fatalf("SCAR failed: %v", err)
        }
    }
}

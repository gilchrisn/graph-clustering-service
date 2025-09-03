package main

import (
	"os"
	"fmt"
	"strconv"
	"strings"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

func main() {
	RunSCAR(os.Args[1:])
}


// RunSCAR executes the SCAR community detection algorithm
func RunSCAR(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: RunSCAR([]string{\"<inFile>\", \"[options]\"})")
		return
	}
	
	config := parseArguments(args)
	
	// Check if user wants Louvain version
	if config.UseLouvain {
		fmt.Println("Running Louvain-SCAR version...")
		engine := scar.NewSketchLouvainEngine(config)
		if err := engine.RunLouvain(); err != nil {
			fmt.Printf("Error running Louvain-SCAR: %v\n", err)
		}
		return
	} else {
		fmt.Println("SCAR not supported")
	}
	return
}

func parseArguments(args []string) scar.SCARConfig {
	config := scar.SCARConfig{
		GraphFile:    args[0],
		OutputFile:   "output.txt",
		EdgesFile:    "",
		PropertyFile: "",
		PathFile:     "",
		Prefix:       "", // Will be set from GraphFile name
		K:            10,
		NK:           4,
		Threshold:    0.5,
		UseLouvain:   false,
		SketchOutput: false, // Add this line
		WriteSketchGraph: true, // Whether to write sketch graph files
		SketchGraphWeights: false, // Whether to use weights in sketch graph files
		NumWorkers:   1, // Default number of workers
	}
	
	// Set default prefix from graph file name
	if config.GraphFile != "" {
		// Extract base name without extension
		baseName := config.GraphFile
		if idx := strings.LastIndex(baseName, "."); idx != -1 {
			baseName = baseName[:idx]
		}
		if idx := strings.LastIndex(baseName, "/"); idx != -1 {
			baseName = baseName[idx+1:]
		}
		if idx := strings.LastIndex(baseName, "\\"); idx != -1 {
			baseName = baseName[idx+1:]
		}
		config.Prefix = baseName
	}
	
	fmt.Printf("=== PARSING ARGUMENTS ===\n")
	fmt.Printf("Raw args: %v\n", args)
	
	// Parse additional arguments
	for i := 1; i < len(args); i++ {
		arg := args[i]
		fmt.Printf("Processing arg %d: '%s'\n", i, arg)
		
		if strings.HasPrefix(arg, "-o=") {
			config.OutputFile = strings.TrimPrefix(arg, "-o=")
			fmt.Printf("  OutputFile = '%s'\n", config.OutputFile)
		} else if strings.HasPrefix(arg, "-edges=") {
			config.EdgesFile = strings.TrimPrefix(arg, "-edges=")
			fmt.Printf("  EdgesFile = '%s'\n", config.EdgesFile)
		} else if strings.HasPrefix(arg, "-pro=") {
			config.PropertyFile = strings.TrimPrefix(arg, "-pro=")
			fmt.Printf("  PropertyFile = '%s'\n", config.PropertyFile)
		} else if strings.HasPrefix(arg, "-path=") {
			config.PathFile = strings.TrimPrefix(arg, "-path=")
			fmt.Printf("  PathFile = '%s'\n", config.PathFile)
		} else if strings.HasPrefix(arg, "-prefix=") {
			config.Prefix = strings.TrimPrefix(arg, "-prefix=")
			fmt.Printf("  Prefix = '%s'\n", config.Prefix)
		} else if strings.HasPrefix(arg, "-k=") {
			if val, err := strconv.ParseInt(strings.TrimPrefix(arg, "-k="), 10, 64); err == nil {
				config.K = val
				fmt.Printf("  K = %d\n", config.K)
			}
		} else if strings.HasPrefix(arg, "-nk=") {
			if val, err := strconv.ParseInt(strings.TrimPrefix(arg, "-nk="), 10, 64); err == nil {
				config.NK = val
				fmt.Printf("  NK = %d\n", config.NK)
			}
		} else if strings.HasPrefix(arg, "-th=") {
			if val, err := strconv.ParseFloat(strings.TrimPrefix(arg, "-th="), 64); err == nil {
				config.Threshold = val
				fmt.Printf("  Threshold = %f\n", config.Threshold)
			}
		} else if arg == "-louvain" {
			config.UseLouvain = true
			fmt.Printf("  UseLouvain = true\n")
		} else if arg == "-sketch-output" {  // Add this block
			config.SketchOutput = true
			fmt.Printf("  SketchOutput = true\n")
		} else if strings.HasPrefix(arg, "-workers=") {
			if val, err := strconv.ParseInt(strings.TrimPrefix(arg, "-workers="), 10, 32); err == nil {
				config.NumWorkers = int(val)
				fmt.Printf("  NumWorkers = %d\n", config.NumWorkers)
			}
		} else if arg == "-no-parallel" {
			config.NumWorkers = 1
			fmt.Printf("  NumWorkers = 1 (sequential)\n")
		} else {
			fmt.Printf("  Unknown argument: '%s'\n", arg)
		}
	}
	
	fmt.Printf("Final config:\n")
	fmt.Printf("  GraphFile: '%s'\n", config.GraphFile)
	fmt.Printf("  PropertyFile: '%s'\n", config.PropertyFile)
	fmt.Printf("  PathFile: '%s'\n", config.PathFile)
	fmt.Printf("  OutputFile: '%s'\n", config.OutputFile)
	fmt.Printf("  Prefix: '%s'\n", config.Prefix)
	fmt.Printf("  UseLouvain: %t\n", config.UseLouvain)
	fmt.Printf("  SketchOutput: %t\n", config.SketchOutput)
	fmt.Println()
	
	return config
}

// To run: go build -o scar.exe && .\scar.exe "test_graph.txt" -pro="properties.txt" -path="path.txt" -k=6 -nk=1 -o="communities.txt" -louvain
package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/gilchrisn/graph-clustering-service/pkg/louvain"

)


func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <edgelist_file> [output_dir] [output_prefix]")
		fmt.Println("Example: go run main.go output/graph.edgelist ./results graph")
		os.Exit(1)
	}

	edgelistFile := os.Args[1]
	outputDir := "./results"
	outputPrefix := "communities"

	if len(os.Args) > 2 {
		outputDir = os.Args[2]
	}
	if len(os.Args) > 3 {
		outputPrefix = os.Args[3]
	}

	fmt.Printf("Reading graph from: %s\n", edgelistFile)
	fmt.Printf("Output directory: %s\n", outputDir)
	fmt.Printf("Output prefix: %s\n", outputPrefix)

	// Read the graph from edge list file
	graph, err := readEdgeListFile(edgelistFile)
	if err != nil {
		fmt.Printf("Error reading edge list: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Graph loaded: %d nodes, %d edges\n", 
		len(graph.Nodes), len(graph.Edges)/2) // Divide by 2 because edges are bidirectional

	// Validate the graph
	if err := graph.Validate(); err != nil {
		fmt.Printf("Graph validation failed: %v\n", err)
		os.Exit(1)
	}

	// Configure Louvain algorithm
	config := DefaultLouvainConfig()
	config.Verbose = true
	config.RandomSeed = 42 // For reproducible results
	config.ProgressCallback = func(level, iteration int, message string) {
		if level >= 0 {
			fmt.Printf("Level %d: %s\n", level, message)
		} else {
			fmt.Printf("  %s\n", message)
		}
	}

	fmt.Println("\nRunning Louvain algorithm...")
	
	// Run Louvain algorithm
	result, err := RunLouvain(graph, config)
	if err != nil {
		fmt.Printf("Louvain algorithm failed: %v\n", err)
		os.Exit(1)
	}

	// Print results summary
	fmt.Printf("\n=== Louvain Results ===\n")
	fmt.Printf("Final modularity: %.6f\n", result.Modularity)
	fmt.Printf("Number of levels: %d\n", result.NumLevels)
	fmt.Printf("Runtime: %d ms\n", result.Statistics.RuntimeMS)
	fmt.Printf("Total iterations: %d\n", result.Statistics.TotalIterations)
	fmt.Printf("Total moves: %d\n", result.Statistics.TotalMoves)

	// Print community information for each level
	for i, level := range result.Levels {
		fmt.Printf("\nLevel %d:\n", level.Level)
		fmt.Printf("  Communities: %d\n", level.NumCommunities)
		fmt.Printf("  Modularity: %.6f\n", level.Modularity)
		fmt.Printf("  Moves: %d\n", level.NumMoves)
		
		// Print actual communities
		fmt.Printf("  Community assignments:\n")
		for commID, nodes := range level.Communities {
			fmt.Printf("    Community %d: %v\n", commID, nodes)
		}
	}

	// Write output files
	fmt.Printf("\nWriting output files to %s...\n", outputDir)
	
	writer := NewFileWriter()
	err = writer.WriteAll(result, graph, outputDir, outputPrefix)
	if err != nil {
		fmt.Printf("Error writing output files: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\nOutput files generated successfully:")
	fmt.Printf("  %s/%s.mapping   - Community to nodes mapping\n", outputDir, outputPrefix)
	fmt.Printf("  %s/%s.hierarchy - Hierarchical community structure\n", outputDir, outputPrefix)
	fmt.Printf("  %s/%s.root      - Top-level communities\n", outputDir, outputPrefix)
	fmt.Printf("  %s/%s.edges     - Edges between communities\n", outputDir, outputPrefix)
}

// readEdgeListFile reads a graph from an edge list file
func readEdgeListFile(filename string) (*HomogeneousGraph, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	graph := NewHomogeneousGraph()
	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		
		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		
		// First line should contain number of nodes and edges
		if lineNum == 1 {
			if len(parts) != 2 {
				return nil, fmt.Errorf("invalid header format on line %d: expected 'num_nodes num_edges'", lineNum)
			}
			
			numNodes, err := strconv.Atoi(parts[0])
			if err != nil {
				return nil, fmt.Errorf("invalid number of nodes on line %d: %v", lineNum, err)
			}
			
			numEdges, err := strconv.Atoi(parts[1])
			if err != nil {
				return nil, fmt.Errorf("invalid number of edges on line %d: %v", lineNum, err)
			}
			
			fmt.Printf("Expected %d nodes and %d edges\n", numNodes, numEdges)
			continue
		}

		// Parse edge lines
		if len(parts) < 2 {
			return nil, fmt.Errorf("invalid edge format on line %d: expected at least 'from to [weight]'", lineNum)
		}

		from := parts[0]
		to := parts[1]
		
		// Default weight is 1.0
		weight := 1.0
		if len(parts) >= 3 {
			weight, err = strconv.ParseFloat(parts[2], 64)
			if err != nil {
				return nil, fmt.Errorf("invalid weight on line %d: %v", lineNum, err)
			}
		}

		// Add edge to graph (this will also add nodes if they don't exist)
		graph.AddEdge(from, to, weight)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return graph, nil
}
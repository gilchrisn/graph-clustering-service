package scar

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// SCAR represents the main SCAR algorithm orchestrator
type SCAR struct {
	graphReader       *GraphReader
	fileReader        *FileReader
	sketchComputer    *SketchComputer
	communityDetector *CommunityDetector
	outputWriter      *OutputWriter
	modCalculator     *ModularityCalculator
}

func NewSCAR() *SCAR {
	return &SCAR{
		graphReader:       NewGraphReader(),
		fileReader:        NewFileReader(),
		sketchComputer:    NewSketchComputer(),
		communityDetector: NewCommunityDetector(10, 4, 0.5),
		outputWriter:      NewOutputWriter(),
		modCalculator:     NewModularityCalculator(),
	}
}

func (s *SCAR) RunWithConfig(config SCARConfig) error {
	fmt.Println("Start reading graph")
	
	// Read graph
	graph, err := s.graphReader.ReadFromFile(config.GraphFile)
	if err != nil {
		return fmt.Errorf("failed to read graph: %v", err)
	}
	
	n := graph.n
	pathLength := int64(10)
	
	startTime := time.Now()
	
	// Initialize arrays
	oldSketches := make([]uint32, pathLength*n*config.K*config.NK)
	for i := range oldSketches {
		oldSketches[i] = math.MaxUint32
	}
	
	nodeHashValue := make([]uint32, n*config.NK)
	
	// Read properties and path
	vertexProperties, err := s.fileReader.ReadProperties(config.PropertyFile, n)
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
	}
	
	path, actualPathLength, err := s.fileReader.ReadPath(config.PathFile)
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
	}
	pathLength = actualPathLength
	
	fmt.Println("Start get graph parameters")
	
	// Compute sketch
	s.sketchComputer.ComputeForGraph(graph, oldSketches, path, pathLength, vertexProperties, nodeHashValue, config.K, config.NK)
	
	sketches := oldSketches[(pathLength-1)*n*config.K*config.NK:]
	
	// Add 1 to all sketches
	for i := range sketches {
		if sketches[i] != math.MaxUint32 {
			sketches[i]++
		}
	}
	
	// Initialize community detection
	data := NewCommunityData(n)
	s.communityDetector = NewCommunityDetector(config.K, config.NK, config.Threshold)
	s.communityDetector.InitializeCommunities(n, config.K, config.NK, sketches, nodeHashValue, data)
	
	// Reconstruct edges if needed
	if config.EdgesFile != "" {
		err := s.outputWriter.ReconstructEdges(config.EdgesFile, data.NodesInCommunity, data.CommunitySketches, data.HashToNodeMap)
		if err != nil {
			fmt.Printf("Warning: failed to reconstruct edges: %v\n", err)
		}
	}
	
	fmt.Println("Finish calculate sketches")
	
	// Calculate whole weight
	wholeWeight := s.calculateWholeWeight(data.CommunityDegreeSketches)
	fmt.Printf("Whole weight: %f\n", wholeWeight)
	
	fmt.Println("Finish calculate sketches and mappings")
	fmt.Println("Finish Initialization of the community")
	
	// Run community detection iterations
	s.runCommunityDetection(data, config, wholeWeight)
	
	fmt.Printf("Community detection time: %v\n", time.Since(startTime))
	
	// Write output
	err = s.outputWriter.WriteResults(config.OutputFile, data.Community, n)
	if err != nil {
		return fmt.Errorf("failed to write output: %v", err)
	}
	
	// Calculate and print modularity
	modularity := s.modCalculator.Calculate(
		data.CommunitySketches,
		data.NodesInCommunities,
		data.DegreeSketches,
		data.Community,
		data.HashToNodeMap,
		sketches,
		config.K,
		wholeWeight,
	)
	fmt.Printf("Final Modularity: %f\n", modularity)
	
	return nil
}

func (s *SCAR) calculateWholeWeight(communityDegreeSketches []uint32) float64 {
	wholeWeight := 0.0
	for _, degree := range communityDegreeSketches {
		wholeWeight += float64(degree)
	}
	return wholeWeight / 2.0
}

func (s *SCAR) runCommunityDetection(data *CommunityData, config SCARConfig, wholeWeight float64) {
	for iter := 0; iter < 20; iter++ {
		start := time.Now()
		fmt.Printf("Number of communities: %d\n", len(data.CommunitySketches))
		
		var newCommunities [][]int64
		
		if iter < 1 {
			algorithm := NewInitialMergeAlgorithm(config.K, config.NK)
			newCommunities = algorithm.FindMerges(
				data.CommunityEdgeTable,
				data.CommunitySketches,
				data.Community,
				data.NodesInCommunities,
				data.CommunityDegreeSketches,
				wholeWeight,
			)
		} else if iter < 2 {
			algorithm := NewQuickMergeAlgorithm(config.K, config.NK)
			newCommunities = algorithm.FindMerges(
				data.CommunityEdgeTable,
				data.CommunitySketches,
				data.Community,
				data.NodesInCommunities,
				data.CommunityDegreeSketches,
				wholeWeight,
			)
		} else {
			adjustedThreshold := config.Threshold
			for i := 1; i < iter; i++ {
				adjustedThreshold *= config.Threshold
			}
			algorithm := NewAdvancedMergeAlgorithm(config.K, config.NK, adjustedThreshold)
			newCommunities = algorithm.FindMerges(
				data.CommunityEdgeTable,
				data.CommunitySketches,
				data.Community,
				data.NodesInCommunities,
				data.CommunityDegreeSketches,
				wholeWeight,
			)
		}
		
		if len(newCommunities) == 0 {
			break
		}
		fmt.Printf("New communities: %d\n", len(newCommunities))
		
		// Merge communities
		s.communityDetector.MergeCommunities(data, newCommunities)
		
		// Merge edge tables
		merger := NewEdgeTableMerger()
		data.CommunityEdgeTable = merger.MergeEdgeTables(data.CommunityEdgeTable, newCommunities)
		
		// Recalculate community degree sketches
		s.recalculateCommunityDegrees(data, config.K)
		
		fmt.Printf("Merge time: %v\n", time.Since(start))
	}
}

func (s *SCAR) recalculateCommunityDegrees(data *CommunityData, k int64) {
	data.CommunityDegreeSketches = data.CommunityDegreeSketches[:0]
	for i := int64(0); i < int64(len(data.CommunitySketches)); i++ {
		currentSketch := uint32(0)
		count := uint32(0)
		
		if sketches, exists := data.CommunitySketches[i]; exists {
			for _, sketch := range sketches {
				count++
				if sketch > currentSketch {
					currentSketch = sketch
				}
			}
		}
		
		if currentSketch != 0 {
			if count >= uint32(k-1) {
				degree := uint32(float64(math.MaxUint32)/float64(currentSketch) * float64(k-1))
				data.CommunityDegreeSketches = append(data.CommunityDegreeSketches, degree)
			} else {
				data.CommunityDegreeSketches = append(data.CommunityDegreeSketches, count)
			}
		} else {
			data.CommunityDegreeSketches = append(data.CommunityDegreeSketches, 1)
		}
	}
}

// RunSCAR executes the SCAR community detection algorithm (legacy function)
func RunSCAR(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: RunSCAR([]string{\"<inFile>\", \"[options]\"})")
		return
	}
	
	config := parseArguments(args)
	
	// Check if user wants Louvain version
	if config.UseLouvain {
		fmt.Println("Running Louvain-SCAR version...")
		louvain := NewLouvainSCAR()
		if err := louvain.RunWithConfig(config); err != nil {
			fmt.Printf("Error running Louvain-SCAR: %v\n", err)
		}
		return
	}
	
	// Original SCAR algorithm
	fmt.Println("Running original SCAR version...")
	scar := NewSCAR()
	if err := scar.RunWithConfig(config); err != nil {
		fmt.Printf("Error running SCAR: %v\n", err)
	}
}

// RunSCARWithConfig runs SCAR with configuration struct (legacy function)
func RunSCARWithConfig(config SCARConfig) error {
	scar := NewSCAR()
	return scar.RunWithConfig(config)
}

func parseArguments(args []string) SCARConfig {
	config := SCARConfig{
		GraphFile:    args[0],
		OutputFile:   "output.txt",
		EdgesFile:    "",
		PropertyFile: "",
		PathFile:     "",
		K:            10,
		NK:           4,
		Threshold:    0.5,
		UseLouvain:   false,
	}
	
	// Parse additional arguments
	for i := 1; i < len(args); i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "-o=") {
			config.OutputFile = strings.TrimPrefix(arg, "-o=")
		} else if strings.HasPrefix(arg, "-edges=") {
			config.EdgesFile = strings.TrimPrefix(arg, "-edges=")
		} else if strings.HasPrefix(arg, "-pro=") {
			config.PropertyFile = strings.TrimPrefix(arg, "-pro=")
		} else if strings.HasPrefix(arg, "-path=") {
			config.PathFile = strings.TrimPrefix(arg, "-path=")
		} else if strings.HasPrefix(arg, "-k=") {
			if val, err := strconv.ParseInt(strings.TrimPrefix(arg, "-k="), 10, 64); err == nil {
				config.K = val
			}
		} else if strings.HasPrefix(arg, "-nk=") {
			if val, err := strconv.ParseInt(strings.TrimPrefix(arg, "-nk="), 10, 64); err == nil {
				config.NK = val
			}
		} else if strings.HasPrefix(arg, "-th=") {
			if val, err := strconv.ParseFloat(strings.TrimPrefix(arg, "-th="), 64); err == nil {
				config.Threshold = val
			}
		} else if arg == "-louvain" {
			config.UseLouvain = true
		}
	}
	
	return config
}
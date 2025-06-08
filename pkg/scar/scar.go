package scar

import (
	"fmt"
	// "math"
	"math/rand"
	"time"
)

// RunScar executes the complete SCAR algorithm with all fixes
func RunScar(graph *HeterogeneousGraph, config ScarConfig) (*ScarResult, error) {
	if !config.MetaPath.IsValid() {
		return nil, fmt.Errorf("invalid meta-path: %v", config.MetaPath)
	}

	// Initialize random seed
	rand.Seed(config.RandomSeed)

	startTime := time.Now()
	
	// Initialize state with enhanced structures
	state := &ScarState{
		Graph:             graph,
		Config:            config,
		N2C:               make(map[string]int),
		C2N:               make(map[int][]string),
		Sketches:          make(map[string]*VertexBottomKSketch),
		CommunitySketches: make(map[int]*VertexBottomKSketch),
		CommunityCounter:  0,
		Iteration:         0,
		HashToNodeMap:     NewHashToNodeMap(),           // Added hash-to-node mapping
		CommunityDegrees:  make(map[int]*DegreeEstimate), // Added degree tracking
		NodeDegrees:       make(map[string]*DegreeEstimate),
		CurrentLevel:      0,
		NodeToOriginal:    make(map[string][]string),
		MergePhase:        0, // Added three-phase merging
	}

	// Initialize node to original mapping
	for _, nodeID := range graph.NodeList {
		state.NodeToOriginal[nodeID] = []string{nodeID}
	}

	if config.Verbose {
		fmt.Printf("Starting SCAR algorithm with K=%d, NK=%d, meta-path: %s\n", 
			config.K, config.NK, config.MetaPath.String())
	}

	result := &ScarResult{
		Levels:          make([]LevelInfo, 0),
		HierarchyLevels: make([]map[string][]string, 0),
		MappingLevels:   make([]map[string][]string, 0),
	}

	totalIterations := 0
	
	// Multi-level community detection
	for level := 0; level < 1; level++ { // Just one level for now. 
		if config.Verbose {
			fmt.Printf("\n=== Level %d ===\n", level)
		}

		levelStart := time.Now()
		state.CurrentLevel = level

		// Iterative sketch construction with path-length iterations
		if err := state.constructSketchesIteratively(); err != nil {
			return nil, fmt.Errorf("failed to construct sketches at level %d: %v", level, err)
		}

		// Initialize communities (each node in its own community)
		state.initializeCommunities()

		// Calculate initial modularity
		initialModularity := state.calculateModularity()
		
		if config.Verbose {
			fmt.Printf("Initial modularity: %.6f\n", initialModularity)
		}

		// Three-phase iterative optimization
		levelIterations := 0
		improved := true
		lastModularity := initialModularity

		for improved && levelIterations < config.MaxIterations {
			levelIterations++
			totalIterations++
			
			// Set merge phase based on iteration
			if levelIterations < 1 {
				state.MergePhase = 0 // Initial best merge (quality optimization)
			} else if levelIterations < 2 {
				state.MergePhase = 1 // Quick best merge (greedy degree-based)
			} else {
				state.MergePhase = 2 // Sophisticated merge (E-function based)
			}
			
			iterationImproved, err := state.executeOneIteration()
			if err != nil {
				return nil, fmt.Errorf("error in iteration %d at level %d: %v", 
					levelIterations, level, err)
			}

			currentModularity := state.calculateModularity()
			improvement := currentModularity - lastModularity

			if config.Verbose {
				fmt.Printf("Iteration %d (phase %d): modularity=%.6f, improvement=%.6f\n", 
					levelIterations, state.MergePhase, currentModularity, improvement)
			}

			if config.ProgressCallback != nil {
				config.ProgressCallback(level, levelIterations, currentModularity, len(state.Graph.NodeList))
			}

			// Check for convergence
			if !iterationImproved || improvement < config.MinModularity {
				improved = false
			}

			lastModularity = currentModularity
		}

		// Record level information
		levelInfo := LevelInfo{
			Level:       level,
			Nodes:       len(state.Graph.NodeList),
			Communities: len(state.C2N),
			Modularity:  lastModularity,
			Improvement: lastModularity - initialModularity,
			Iterations:  levelIterations,
			Duration:    time.Since(levelStart),
			N2C:         make(map[string]int),
			C2N:         make(map[int][]string),
			NodeMapping: make(map[string][]string),
		}

		// Copy current state
		for node, comm := range state.N2C {
			levelInfo.N2C[node] = comm
		}
		for comm, nodes := range state.C2N {
			levelInfo.C2N[comm] = make([]string, len(nodes))
			copy(levelInfo.C2N[comm], nodes)
		}
		for node, originals := range state.NodeToOriginal {
			levelInfo.NodeMapping[node] = make([]string, len(originals))
			copy(levelInfo.NodeMapping[node], originals)
		}

		result.Levels = append(result.Levels, levelInfo)

		if config.Verbose {
			fmt.Printf("Level %d completed: %d nodes -> %d communities, modularity=%.6f\n",
				level, levelInfo.Nodes, levelInfo.Communities, levelInfo.Modularity)
		}

		// Check if we should continue to next level
		if len(state.C2N) == len(state.Graph.NodeList) || len(state.C2N) <= 1 {
			break
		}

		// Aggregate communities for next level
		// if err := state.aggregateCommunities(); err != nil {
		// 	return nil, fmt.Errorf("failed to aggregate communities at level %d: %v", level, err)
		// }
	}

	// Build final result
	result.NumLevels = len(result.Levels)
	if result.NumLevels > 0 {
		result.Modularity = result.Levels[result.NumLevels-1].Modularity
	}

	// Build final community assignments for original nodes
	result.FinalCommunities = state.buildFinalCommunities()

	// Build statistics
	result.Statistics = ScarStats{
		TotalLevels:     result.NumLevels,
		TotalIterations: totalIterations,
		TotalDuration:   time.Since(startTime),
		FinalModularity: result.Modularity,
		InitialNodes:    len(graph.NodeList),
		InitialEdges:    len(graph.Edges),
	}

	if result.NumLevels > 0 {
		result.Statistics.FinalNodes = result.Levels[result.NumLevels-1].Nodes
	}

	// Build hierarchy and mapping levels for output compatibility
	state.buildHierarchyAndMapping(result)

	if config.Verbose {
		fmt.Printf("\nSCAR completed: %d levels, final modularity=%.6f\n", 
			result.NumLevels, result.Modularity)
	}

	return result, nil
}

// Iterative sketch construction following the original C++ algorithm
func (s *ScarState) constructSketchesIteratively() error {
	fmt.Println("Checkpoint 1: Starting sketch construction...")
	pathLength := len(s.Config.MetaPath.EdgeTypes) + 1
	
	// Clear existing sketches
	s.Sketches = make(map[string]*VertexBottomKSketch)
	s.HashToNodeMap = NewHashToNodeMap()

	// Initialize sketches for all nodes at all path positions
	for _, nodeID := range s.Graph.NodeList {
		s.Sketches[nodeID] = NewVertexBottomKSketch(s.Config.K, s.Config.NK, 0)
	}

	fmt.Println("Checkpoint 2: Sketches initialized for all nodes.")
	// Get source type nodes (first node type in meta-path)
	sourceType := s.Config.MetaPath.NodeTypes[0]
	sourceNodes := s.Graph.GetNodesByType(sourceType)

	if len(sourceNodes) == 0 {
		return fmt.Errorf("no nodes found for source type: %s", sourceType)
	}

	// Assign hash values to source nodes across all nK hash functions
	for _, nodeID := range sourceNodes {
		for hashFunc := 0; hashFunc < s.Config.NK; hashFunc++ {
			hashValue := GenerateIndependentHashValue(nodeID, hashFunc, s.Config.RandomSeed)
			s.Sketches[nodeID].AddValue(hashFunc, hashValue)
			// Add to hash-to-node mapping
			s.HashToNodeMap.AddMapping(hashValue, nodeID)
		}
	}

	fmt.Println("Checkpoint 3: Source nodes sketches and hash mappings created.")
	// Path-length iterations for sketch propagation
	for iter := 1; iter < pathLength; iter++ {
		fmt.Printf("Checkpoint 4: Starting sketch propagation iteration %d/%d\n", iter, pathLength-1)
		if s.Config.Verbose {
			fmt.Printf("  Sketch construction iteration %d/%d\n", iter, pathLength-1)
		}

		// Create new sketches for this iteration
		newSketches := make(map[string]*VertexBottomKSketch)
		for _, nodeID := range s.Graph.NodeList {
			newSketches[nodeID] = NewVertexBottomKSketch(s.Config.K, s.Config.NK, iter)
		}

		// Propagate sketches along meta-path
		currentNodeType := s.Config.MetaPath.NodeTypes[iter-1]
		nextNodeType := s.Config.MetaPath.NodeTypes[iter]
		edgeType := s.Config.MetaPath.EdgeTypes[iter-1]

		for _, nodeID := range s.Graph.NodeList {
			// Progress check
			fmt.Printf("  Processing node %s (%d/%d)\n", nodeID, iter, len(s.Graph.NodeList))
			if s.Graph.NodeTypes[nodeID] != currentNodeType {
				continue
			}

			currentSketch := s.Sketches[nodeID]
			if currentSketch.IsEmpty() {
				continue
			}

			// Find neighbors of correct type
			neighbors := s.Graph.GetNeighbors(nodeID, edgeType)
			
			for _, neighbor := range neighbors {
				if s.Graph.NodeTypes[neighbor] == nextNodeType {
					// Merge sketches across all nK hash functions
					for hashFunc := 0; hashFunc < s.Config.NK; hashFunc++ {
						for _, value := range currentSketch.Sketches[hashFunc] {
							newSketches[neighbor].AddValue(hashFunc, value)
						}
					}
				}
			}
		}

		// Update sketches
		s.Sketches = newSketches
	}

	// Calculate node degrees
	s.NodeDegrees = make(map[string]*DegreeEstimate)
	for nodeID, sketch := range s.Sketches {
		s.NodeDegrees[nodeID] = sketch.EstimateDegree()
	}

	return nil
}

// Three-phase iteration execution
func (s *ScarState) executeOneIteration() (bool, error) {
	improved := false
	
	// Shuffle nodes for better convergence
	nodes := make([]string, len(s.Graph.NodeList))
	copy(nodes, s.Graph.NodeList)
	rand.Shuffle(len(nodes), func(i, j int) {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	})

	for _, nodeID := range nodes {
		currentCommunity := s.N2C[nodeID]
		var bestCommunity int
		var bestGain float64

		// Use different merge strategies based on phase
		switch s.MergePhase {
		case 0:
			// Initial best merge - quality optimization
			bestCommunity, bestGain = s.initialBestMerge(nodeID)
		case 1:
			// Quick best merge - greedy degree-based
			bestCommunity, bestGain = s.quickBestMerge(nodeID)
		case 2:
			// Sophisticated merge - E-function based
			bestCommunity, bestGain = s.calculateBestMerge(nodeID)
		default:
			bestCommunity, bestGain = s.calculateBestMerge(nodeID)
		}

		// Move node if beneficial
		if bestCommunity != currentCommunity && bestGain > 0 {
			s.moveNodeToCommunity(nodeID, bestCommunity)
			improved = true
		}
	}

	return improved, nil
}

// Initial best merge - quality optimization
func (s *ScarState) initialBestMerge(nodeID string) (int, float64) {
	currentCommunity := s.N2C[nodeID]
	bestCommunity := currentCommunity
	bestGain := 0.0

	// Find communities through sketch-based adjacency discovery
	neighborCommunities := s.findCommunitiesThroughSketches(nodeID)
	
	for _, targetCommunity := range neighborCommunities {
		if targetCommunity == currentCommunity {
			continue
		}

		// Use simple modularity gain for initial phase
		gain := s.estimateModularityGain(nodeID, targetCommunity)
		if gain > bestGain {
			bestGain = gain
			bestCommunity = targetCommunity
		}
	}

	return bestCommunity, bestGain
}

// Quick best merge - greedy degree-based
func (s *ScarState) quickBestMerge(nodeID string) (int, float64) {
	currentCommunity := s.N2C[nodeID]
	bestCommunity := currentCommunity
	bestGain := 0.0

	neighborCommunities := s.findCommunitiesThroughSketches(nodeID)
	nodeDegree := s.NodeDegrees[nodeID]
	
	for _, targetCommunity := range neighborCommunities {
		if targetCommunity == currentCommunity {
			continue
		}

		// Quick degree-based estimation
		targetDegree := s.CommunityDegrees[targetCommunity]
		if targetDegree == nil {
			continue
		}

		// Simple degree-based gain
		gain := nodeDegree.Value * targetDegree.Value / float64(len(s.Graph.Edges))
		if gain > bestGain {
			bestGain = gain
			bestCommunity = targetCommunity
		}
	}

	return bestCommunity, bestGain
}

// Sophisticated merge using E-function
func (s *ScarState) calculateBestMerge(nodeID string) (int, float64) {
	currentCommunity := s.N2C[nodeID]
	bestCommunity := currentCommunity
	bestGain := 0.0

	neighborCommunities := s.findCommunitiesThroughSketches(nodeID)
	
	for _, targetCommunity := range neighborCommunities {
		if targetCommunity == currentCommunity {
			continue
		}

		// Calculate E-function
		eResult := s.calculateEFunction(nodeID, targetCommunity)
		if eResult.Value > bestGain {
			bestGain = eResult.Value
			bestCommunity = targetCommunity
		}
	}

	return bestCommunity, bestGain
}

// E-function calculation following original algorithm
func (s *ScarState) calculateEFunction(nodeID string, targetCommunity int) *EFunctionResult {
	nodeSketch := s.Sketches[nodeID]
	communitySketch := s.CommunitySketches[targetCommunity]
	
	if nodeSketch == nil || communitySketch == nil {
		return &EFunctionResult{Value: 0}
	}

	// Calculate community sizes and degrees
	c1Size := s.NodeDegrees[nodeID].Value
	c2Size := s.CommunityDegrees[targetCommunity].Value
	
	// Calculate intersection using sophisticated method
	intersectK := nodeSketch.EstimateIntersectionWith(communitySketch)
	
	// Calculate total weight (total edges)
	wholeWeight := float64(len(s.Graph.Edges) * 2) // Undirected edges counted twice
	
	// Calculate expected edges
	n1 := 1.0 // Single node
	n2 := float64(len(s.C2N[targetCommunity]))
	expectedEdges := (n1 * n2) / (2.0 * wholeWeight)
	
	// E-function formula: c1Size + C2 - intersectK - (n1×n2)/(2×wholeWeight)
	eValue := c1Size + c2Size - intersectK - expectedEdges
	
	return &EFunctionResult{
		Value:         eValue,
		C1Size:        c1Size,
		C2Size:        c2Size,
		IntersectK:    intersectK,
		ExpectedEdges: expectedEdges,
		ActualEdges:   intersectK, // Approximation
	}
}

// Find communities through sketch-based adjacency discovery
func (s *ScarState) findCommunitiesThroughSketches(nodeID string) []int {
	communities := make(map[int]bool)
	nodeSketch := s.Sketches[nodeID]
	
	if nodeSketch == nil {
		return []int{s.N2C[nodeID]} // Return current community if no sketch
	}

	// Get all hash values from the node's sketch
	allHashValues := nodeSketch.GetAllHashValues()
	
	// For each hash value, find corresponding nodes through hash-to-node mapping
	for _, hashValue := range allHashValues {
		if adjacentNodeID, exists := s.HashToNodeMap.GetNode(hashValue); exists {
			if adjacentNodeID != nodeID { // Don't include self
				if adjacentCommunity, exists := s.N2C[adjacentNodeID]; exists {
					communities[adjacentCommunity] = true
				}
			}
		}
	}
	
	// Also check communities of nodes that have overlapping sketches
	for otherNodeID, otherSketch := range s.Sketches {
		if otherNodeID != nodeID && otherSketch != nil {
			// Check for sketch overlap
			if HasSketchOverlap(nodeSketch, otherSketch, 0.1) { // 10% overlap threshold
				if otherCommunity, exists := s.N2C[otherNodeID]; exists {
					communities[otherCommunity] = true
				}
			}
		}
	}
	
	// Add current community
	communities[s.N2C[nodeID]] = true
	
	result := make([]int, 0, len(communities))
	for community := range communities {
		result = append(result, community)
	}
	
	return result
}

// Enhanced community initialization with degree calculation
func (s *ScarState) initializeCommunities() {
	s.N2C = make(map[string]int)
	s.C2N = make(map[int][]string)
	s.CommunitySketches = make(map[int]*VertexBottomKSketch)
	s.CommunityDegrees = make(map[int]*DegreeEstimate)
	s.CommunityCounter = 0

	for _, nodeID := range s.Graph.NodeList {
		communityID := s.CommunityCounter
		s.N2C[nodeID] = communityID
		s.C2N[communityID] = []string{nodeID}
		
		// Initialize community sketch with node's sketch
		nodeSketch := s.Sketches[nodeID]
		if nodeSketch != nil {
			s.CommunitySketches[communityID] = nodeSketch.Clone()
			s.CommunityDegrees[communityID] = nodeSketch.EstimateDegree()
		}
		
		s.CommunityCounter++
	}
}

// Enhanced modularity calculation using sophisticated degree estimation
func (s *ScarState) calculateModularity() float64 {
	if len(s.Graph.Edges) == 0 {
		return 0
	}
	
	totalEdges := float64(len(s.Graph.Edges))
	modularity := 0.0
	
	for communityID, nodes := range s.C2N {
		if len(nodes) == 0 {
			continue
		}
		
		communitySketch := s.CommunitySketches[communityID]
		if communitySketch == nil {
			continue
		}
		
		// Use sophisticated degree estimation
		communityDegree := s.CommunityDegrees[communityID]
		if communityDegree == nil {
			continue
		}
		
		// Estimate internal edges using sketch intersections
		internalEdges := 0.0
		for i, node1 := range nodes {
			for j := i + 1; j < len(nodes); j++ {
				node2 := nodes[j]
				sketch1 := s.Sketches[node1]
				sketch2 := s.Sketches[node2]
				if sketch1 != nil && sketch2 != nil {
					intersection := sketch1.EstimateIntersectionWith(sketch2)
					internalEdges += intersection / float64(s.Config.NK) // Normalize by number of hash functions
				}
			}
		}
		
		// Expected edges calculation
		expectedEdges := (communityDegree.Value * communityDegree.Value) / (4.0 * totalEdges)
		
		// Modularity contribution
		modularity += (internalEdges - expectedEdges) / totalEdges
	}
	
	return modularity
}

// Enhanced move node with proper sketch and degree updates
func (s *ScarState) moveNodeToCommunity(nodeID string, newCommunity int) {
	oldCommunity := s.N2C[nodeID]
	
	// Remove from old community
	oldNodes := s.C2N[oldCommunity]
	for i, node := range oldNodes {
		if node == nodeID {
			s.C2N[oldCommunity] = append(oldNodes[:i], oldNodes[i+1:]...)
			break
		}
	}
	
	// Add to new community
	s.N2C[nodeID] = newCommunity
	s.C2N[newCommunity] = append(s.C2N[newCommunity], nodeID)
	
	// Update community sketches and degrees
	s.updateCommunitySketchAndDegree(oldCommunity)
	s.updateCommunitySketchAndDegree(newCommunity)
}

// Update community sketch and degree estimation
func (s *ScarState) updateCommunitySketchAndDegree(communityID int) {
	communityNodes := s.C2N[communityID]
	if len(communityNodes) == 0 {
		// Empty community
		delete(s.CommunitySketches, communityID)
		delete(s.CommunityDegrees, communityID)
		delete(s.C2N, communityID)
		return
	}
	
	// Collect sketches from community members
	memberSketches := make([]*VertexBottomKSketch, 0, len(communityNodes))
	for _, nodeID := range communityNodes {
		if sketch := s.Sketches[nodeID]; sketch != nil {
			memberSketches = append(memberSketches, sketch)
		}
	}
	
	// Merge sketches
	if len(memberSketches) > 0 {
		s.CommunitySketches[communityID] = MergeSketches(memberSketches)
		s.CommunityDegrees[communityID] = s.CommunitySketches[communityID].EstimateDegree()
	}
}

// Simple modularity gain estimation (used in initial phase)
func (s *ScarState) estimateModularityGain(nodeID string, targetCommunity int) float64 {
	nodeSketch := s.Sketches[nodeID]
	communitySketch := s.CommunitySketches[targetCommunity]
	
	if nodeSketch == nil || communitySketch == nil {
		return 0.0
	}
	
	nodeDegree := s.NodeDegrees[nodeID].Value
	communityDegree := s.CommunityDegrees[targetCommunity].Value
	
	// Estimate edges between node and community
	edgesToTarget := nodeSketch.EstimateIntersectionWith(communitySketch)
	
	totalEdges := float64(len(s.Graph.Edges))
	if totalEdges == 0 {
		return 0
	}

	// Simplified modularity gain
	gain := (edgesToTarget / totalEdges) - (nodeDegree * communityDegree) / (4 * totalEdges * totalEdges)
	
	return gain
}


// aggregateCommunities creates a new graph where communities become nodes
func (s *ScarState) aggregateCommunities() error {
	newGraph := NewHeterogeneousGraph()
	newNodeToOriginal := make(map[string][]string)
	
	// Create supernode for each community
	communityToSupernode := make(map[int]string)
	supernodeCounter := 0
	
	for communityID, nodes := range s.C2N {
		if len(nodes) == 0 {
			continue
		}
		
		supernodeID := fmt.Sprintf("c%d_l%d_%d", communityID, s.CurrentLevel, supernodeCounter)
		supernodeCounter++
		
		// Create supernode (use first node's type as representative)
		if len(nodes) > 0 {
			firstNodeType := s.Graph.NodeTypes[nodes[0]]
			supernode := HeteroNode{
				ID:   supernodeID,
				Type: firstNodeType,
				Properties: map[string]interface{}{
					"level":     s.CurrentLevel + 1,
					"size":      len(nodes),
					"community": communityID,
				},
			}
			newGraph.AddNode(supernode)
		}
		
		communityToSupernode[communityID] = supernodeID
		
		// Map supernode to original nodes
		originalNodes := make([]string, 0)
		for _, nodeID := range nodes {
			if originals, exists := s.NodeToOriginal[nodeID]; exists {
				originalNodes = append(originalNodes, originals...)
			} else {
				originalNodes = append(originalNodes, nodeID)
			}
		}
		newNodeToOriginal[supernodeID] = originalNodes
	}
	
	// Create edges between supernodes based on inter-community sketch overlaps
	interCommunityEdges := make(map[EdgeKey]float64)
	
	// Use sketch-based edge detection between communities
	for comm1, sketch1 := range s.CommunitySketches {
		for comm2, sketch2 := range s.CommunitySketches {
			if comm1 >= comm2 { // Avoid duplicates and self-loops
				continue
			}
			
			supernode1 := communityToSupernode[comm1]
			supernode2 := communityToSupernode[comm2]
			
			if supernode1 != "" && supernode2 != "" {
				// Calculate inter-community connection strength using sketch intersection
				connectionStrength := sketch1.EstimateIntersectionWith(sketch2)
				
				// Only create edge if connection strength is significant
				if connectionStrength > 0.1 { // Threshold for significant connection
					edgeKey := EdgeKey{From: supernode1, To: supernode2}
					interCommunityEdges[edgeKey] = connectionStrength
				}
			}
		}
	}
	
	// Add inter-community edges to new graph
	for edgeKey, weight := range interCommunityEdges {
		edge := HeteroEdge{
			From:   edgeKey.From,
			To:     edgeKey.To,
			Type:   "aggregated",
			Weight: weight,
		}
		newGraph.AddEdge(edge)
	}
	
	// Update state
	s.Graph = newGraph
	s.NodeToOriginal = newNodeToOriginal
	
	return nil
}

// buildFinalCommunities maps original nodes to final communities
func (s *ScarState) buildFinalCommunities() map[string]int {
	finalCommunities := make(map[string]int)
	
	// Map through all levels
	for communityID, nodes := range s.C2N {
		for _, nodeID := range nodes {
			if originals, exists := s.NodeToOriginal[nodeID]; exists {
				for _, originalNode := range originals {
					finalCommunities[originalNode] = communityID
				}
			}
		}
	}
	
	return finalCommunities
}

// buildHierarchyAndMapping builds hierarchy and mapping structures for output
func (s *ScarState) buildHierarchyAndMapping(result *ScarResult) {
	result.HierarchyLevels = make([]map[string][]string, len(result.Levels))
	result.MappingLevels = make([]map[string][]string, len(result.Levels))
	
	for i, level := range result.Levels {
		hierarchy := make(map[string][]string)
		mapping := make(map[string][]string)
		
		for communityID, nodes := range level.C2N {
			supernodeID := fmt.Sprintf("c%d_l%d_%d", communityID, level.Level, communityID)
			
			// Hierarchy: supernode -> direct children
			hierarchy[supernodeID] = make([]string, len(nodes))
			copy(hierarchy[supernodeID], nodes)
			
			// Mapping: supernode -> all original leaf nodes
			originalNodes := make([]string, 0)
			for _, nodeID := range nodes {
				if originals, exists := level.NodeMapping[nodeID]; exists {
					originalNodes = append(originalNodes, originals...)
				} else {
					originalNodes = append(originalNodes, nodeID)
				}
			}
			mapping[supernodeID] = originalNodes
		}
		
		result.HierarchyLevels[i] = hierarchy
		result.MappingLevels[i] = mapping
	}
}

// Enhanced validation functions

// ValidateState validates the internal state of SCAR algorithm
func (s *ScarState) ValidateState() error {
	// Validate sketches
	for nodeID, sketch := range s.Sketches {
		if sketch == nil {
			return fmt.Errorf("nil sketch for node %s", nodeID)
		}
		
		if err := sketch.ValidateSketch(); err != nil {
			return fmt.Errorf("invalid sketch for node %s: %v", nodeID, err)
		}
	}
	
	// Validate community sketches
	for commID, sketch := range s.CommunitySketches {
		if sketch == nil {
			return fmt.Errorf("nil community sketch for community %d", commID)
		}
		
		if err := sketch.ValidateSketch(); err != nil {
			return fmt.Errorf("invalid community sketch for community %d: %v", commID, err)
		}
	}
	
	// Validate node-to-community mapping
	for nodeID, commID := range s.N2C {
		if _, exists := s.C2N[commID]; !exists {
			return fmt.Errorf("node %s assigned to non-existent community %d", nodeID, commID)
		}
		
		// Check if node is in the community's node list
		found := false
		for _, node := range s.C2N[commID] {
			if node == nodeID {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("node %s not found in community %d node list", nodeID, commID)
		}
	}
	
	// Validate community-to-nodes mapping
	for commID, nodes := range s.C2N {
		for _, nodeID := range nodes {
			if assignedComm, exists := s.N2C[nodeID]; !exists {
				return fmt.Errorf("node %s in community %d but not in N2C mapping", nodeID, commID)
			} else if assignedComm != commID {
				return fmt.Errorf("node %s in community %d but assigned to community %d", nodeID, commID, assignedComm)
			}
		}
	}
	
	return nil
}

// Debug functions for development and testing

// PrintSketchStatistics prints detailed statistics about sketches
func (s *ScarState) PrintSketchStatistics() {
	fmt.Println("=== Sketch Statistics ===")
	
	totalSize := 0
	saturatedCount := 0
	undersaturatedCount := 0
	
	for nodeID, sketch := range s.Sketches {
		stats := sketch.GetStatistics()
		totalSize += stats["total_size"].(int)
		
		if stats["is_majority_saturated"].(bool) {
			saturatedCount++
		} else {
			undersaturatedCount++
		}
		
		if s.Config.Verbose {
			fmt.Printf("Node %s: size=%d, degree=%.2f, saturated=%v\n", 
				nodeID, stats["total_size"].(int), stats["estimated_degree"].(float64), 
				stats["is_majority_saturated"].(bool))
		}
	}
	
	fmt.Printf("Total sketches: %d\n", len(s.Sketches))
	fmt.Printf("Average sketch size: %.2f\n", float64(totalSize)/float64(len(s.Sketches)))
	fmt.Printf("Saturated sketches: %d (%.1f%%)\n", saturatedCount, 
		100.0*float64(saturatedCount)/float64(len(s.Sketches)))
	fmt.Printf("Undersaturated sketches: %d (%.1f%%)\n", undersaturatedCount,
		100.0*float64(undersaturatedCount)/float64(len(s.Sketches)))
}

// PrintCommunityStatistics prints detailed statistics about communities
func (s *ScarState) PrintCommunityStatistics() {
	fmt.Println("=== Community Statistics ===")
	
	totalNodes := 0
	
	for commID, nodes := range s.C2N {
		totalNodes += len(nodes)
		degree := s.CommunityDegrees[commID]
		
		fmt.Printf("Community %d: %d nodes, degree=%.2f, saturated=%v\n", 
			commID, len(nodes), degree.Value, degree.IsSaturated)
		
		if s.Config.Verbose {
			fmt.Printf("  Nodes: %v\n", nodes)
		}
	}
	
	fmt.Printf("Total communities: %d\n", len(s.C2N))
	fmt.Printf("Average community size: %.2f\n", float64(totalNodes)/float64(len(s.C2N)))
}

// Performance monitoring functions

// GetPerformanceMetrics returns detailed performance metrics
func (s *ScarState) GetPerformanceMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})
	
	// Sketch metrics
	totalSketchSize := 0
	saturatedSketches := 0
	
	for _, sketch := range s.Sketches {
		totalSketchSize += sketch.Size()
		degree := sketch.EstimateDegree()
		if degree.IsSaturated {
			saturatedSketches++
		}
	}
	
	metrics["total_sketch_size"] = totalSketchSize
	metrics["average_sketch_size"] = float64(totalSketchSize) / float64(len(s.Sketches))
	metrics["saturation_ratio"] = float64(saturatedSketches) / float64(len(s.Sketches))
	
	// Community metrics
	totalCommunitySize := 0
	for _, nodes := range s.C2N {
		totalCommunitySize += len(nodes)
	}
	
	metrics["num_communities"] = len(s.C2N)
	metrics["average_community_size"] = float64(totalCommunitySize) / float64(len(s.C2N))
	
	// Hash-to-node mapping metrics
	metrics["hash_mapping_size"] = len(s.HashToNodeMap.Mapping)
	
	// Graph metrics
	metrics["num_nodes"] = len(s.Graph.NodeList)
	metrics["num_edges"] = len(s.Graph.Edges)
	
	return metrics
}

// Utility functions for testing and debugging

// CompareWithGroundTruth compares results with known ground truth (for testing)
func CompareWithGroundTruth(result *ScarResult, groundTruth map[string]int) map[string]float64 {
	comparison := make(map[string]float64)
	
	if len(result.FinalCommunities) == 0 || len(groundTruth) == 0 {
		return comparison
	}
	
	// Calculate various similarity metrics
	correctAssignments := 0
	totalAssignments := 0
	
	for nodeID, predictedComm := range result.FinalCommunities {
		if trueComm, exists := groundTruth[nodeID]; exists {
			totalAssignments++
			if predictedComm == trueComm {
				correctAssignments++
			}
		}
	}
	
	if totalAssignments > 0 {
		comparison["accuracy"] = float64(correctAssignments) / float64(totalAssignments)
	}
	
	comparison["predicted_communities"] = float64(len(getUniqueCommunities(result.FinalCommunities)))
	comparison["true_communities"] = float64(len(getUniqueCommunities(groundTruth)))
	
	return comparison
}

// Helper function to get unique communities
func getUniqueCommunities(communities map[string]int) map[int]bool {
	unique := make(map[int]bool)
	for _, comm := range communities {
		unique[comm] = true
	}
	return unique
}

// Memory management functions

// CleanupState cleans up large data structures to free memory
func (s *ScarState) CleanupState() {
	// Clear large maps while preserving essential data
	s.Sketches = nil
	s.CommunitySketches = nil
	s.HashToNodeMap = nil
	s.CommunityDegrees = nil
	s.NodeDegrees = nil
}

// EstimateMemoryUsage estimates the memory usage of the algorithm state
func (s *ScarState) EstimateMemoryUsage() map[string]int64 {
	usage := make(map[string]int64)
	
	// Estimate sketch memory usage
	sketchMemory := int64(0)
	for _, sketch := range s.Sketches {
		// Each uint64 is 8 bytes, plus overhead for slice structures
		sketchMemory += int64(sketch.Size() * 8)
		sketchMemory += int64(sketch.NK * 24) // Slice overhead
	}
	usage["sketches"] = sketchMemory
	
	// Estimate community sketch memory
	communitySketchMemory := int64(0)
	for _, sketch := range s.CommunitySketches {
		communitySketchMemory += int64(sketch.Size() * 8)
		communitySketchMemory += int64(sketch.NK * 24)
	}
	usage["community_sketches"] = communitySketchMemory
	
	// Estimate hash-to-node mapping memory
	hashMapMemory := int64(len(s.HashToNodeMap.Mapping) * (8 + 16)) // uint64 + string overhead
	usage["hash_mapping"] = hashMapMemory
	
	// Estimate other structures
	usage["node_community_mapping"] = int64(len(s.N2C) * 24) // string + int + overhead
	usage["community_nodes_mapping"] = int64(len(s.C2N) * 100) // Estimated slice overhead
	
	total := int64(0)
	for _, mem := range usage {
		total += mem
	}
	usage["total"] = total
	
	return usage
}
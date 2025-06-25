package louvain

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"
)

// RunLouvain executes the complete Louvain algorithm on normalized graph
func RunLouvain(graph *NormalizedGraph, config LouvainConfig) (*LouvainResult, error) {
	startTime := time.Now()
	
	// Validate input
	if err := graph.Validate(); err != nil {
		return nil, fmt.Errorf("invalid graph: %w", err)
	}
	
	// // Initialize random seed
	// if config.RandomSeed < 0 {
	// 	config.RandomSeed = time.Now().UnixNano()
	// }
	config.RandomSeed = 42 // For reproducibility in tests
	rand.Seed(config.RandomSeed)
	
	// Initialize result
	result := &LouvainResult{
		Levels:     []LevelInfo{},
		Statistics: LouvainStats{},
	}
	
	// Initialize state
	state := NewLouvainState(graph, config)

	
	level := 0
	improvement := true
	
	// Main loop
	for improvement && level < config.MaxIterations {
		levelStart := time.Now()
		
		if config.ProgressCallback != nil {
			config.ProgressCallback(level, 0, fmt.Sprintf("Starting level %d with %d nodes", level, graph.NumNodes))
		}
		
		// Execute one level
		improvement, err := state.ExecuteOneLevel()
		if err != nil {
			return nil, fmt.Errorf("error at level %d: %w", level, err)
		}

		// Record level info
		levelInfo := state.GetLevelInfo(level)
		result.Levels = append(result.Levels, levelInfo)
		
		// Update statistics
		levelStats := LevelStats{
			Level:             level,
			Iterations:        state.Iteration,
			Moves:             levelInfo.NumMoves,
			InitialModularity: levelInfo.Modularity,
			FinalModularity:   state.GetModularity(),
			RuntimeMS:         time.Since(levelStart).Milliseconds(),
		}
		result.Statistics.LevelStats = append(result.Statistics.LevelStats, levelStats)
		result.Statistics.TotalIterations += levelStats.Iterations
		result.Statistics.TotalMoves += levelStats.Moves
		
		if !improvement {
			break
		}
		
		// Create super graph for next level
		superGraph, communityMap, err := state.CreateSuperGraph()
		if err != nil {
			return nil, fmt.Errorf("error creating super graph at level %d: %w", level, err)
		}
		
		// Check if we've converged
		if superGraph.NumNodes >= state.Graph.NumNodes {
			fmt.Printf("No compression at level %d (%d -> %d nodes), stopping\n", 
				level, state.Graph.NumNodes, superGraph.NumNodes)
			break
		}
		
		// Update state for next level
		state = NewLouvainStateFromCommunities(superGraph, config, communityMap)
		level++
	}
	
	// Set final results
	result.NumLevels = len(result.Levels)
	result.Modularity = state.GetModularity()
	result.FinalCommunities = state.GetFinalCommunities()
	if result.FinalCommunities == nil {
		return nil, fmt.Errorf("failed to generate final communities")
	}
	result.Statistics.RuntimeMS = time.Since(startTime).Milliseconds()
	result.Statistics.MemoryPeakMB = getMemoryUsage()

	return result, nil
}

// NewLouvainState creates a new Louvain state with initial communities
func NewLouvainState(graph *NormalizedGraph, config LouvainConfig) *LouvainState {
	state := &LouvainState{
		Graph:            graph,
		Config:           config,
		N2C:              make([]int, graph.NumNodes),
		C2N:              make(map[int][]int),
		In:               make(map[int]float64),
		Tot:              make(map[int]float64),
		CommunityCounter: 0,
		Iteration:        0,
	}
	
	// Initialize each node in its own community
	for i := 0; i < graph.NumNodes; i++ {
		commID := state.CommunityCounter
		state.CommunityCounter++
		
		state.N2C[i] = commID
		state.C2N[commID] = []int{i}
		
		// Calculate self-loops and degree
		selfLoop := graph.GetEdgeWeight(i, i)
		degree := graph.GetNodeDegree(i)
		
		state.In[commID] = selfLoop
		state.Tot[commID] = degree
	}
	
	return state
}

// NewLouvainStateFromCommunities creates a state where nodes are communities from previous level
func NewLouvainStateFromCommunities(graph *NormalizedGraph, config LouvainConfig, communityMap map[int][]int) *LouvainState {
	state := NewLouvainState(graph, config)
	state.CommunityCounter = len(communityMap)
	return state
}

// ExecuteOneLevel performs one level of the Louvain algorithm
func (s *LouvainState) ExecuteOneLevel() (bool, error) {
	improvement := false 
	nbMoves := 0
	s.Iteration = 0

	fmt.Printf("Starting Louvain level with %d nodes, initial modularity: %.4f\n",
		s.Graph.NumNodes, s.GetModularity())

	// ==============================================================================================

	for {
		s.Iteration++
		iterMoves := 0
		
		// Create random order for nodes
		nodeOrder := make([]int, s.Graph.NumNodes)
		for i := 0; i < s.Graph.NumNodes; i++ {
			nodeOrder[i] = i
		}
		// Shuffle nodes to randomize processing order (Disabled for reproducibility)
		// rand.Shuffle(len(nodeOrder), func(i, j int) {
		// 	nodeOrder[i], nodeOrder[j] = nodeOrder[j], nodeOrder[i]
		// })
		
		// Process nodes in chunks
		chunkSize := s.Config.ChunkSize
		if chunkSize <= 0 {
			chunkSize = 32
		}
		
		for i := 0; i < len(nodeOrder); i += chunkSize {
			end := i + chunkSize
			if end > len(nodeOrder) {
				end = len(nodeOrder)
			}
			
			moves := s.processNodeChunk(nodeOrder[i:end])
			iterMoves += moves
		}
		
		nbMoves += iterMoves
		
		if iterMoves == 0 {
			break
		}
		
		improvement = true
		
		if s.Config.ProgressCallback != nil {
			s.Config.ProgressCallback(-1, s.Iteration, 
				fmt.Sprintf("Iteration %d: %d moves", s.Iteration, iterMoves))
		}
	}

	finalModularity := s.GetModularity()
	fmt.Printf("Final modularity after level %d: %.4f, moves: %d\n",
		s.Iteration, finalModularity, nbMoves)

	return improvement && nbMoves > 0, nil
}

// processNodeChunk processes a chunk of nodes
func (s *LouvainState) processNodeChunk(nodes []int) int {
	moves := 0

	// Validate state before processing
	if err := s.ValidateState(); err != nil {
		fmt.Printf("State validation failed before processing chunk: %v\n", err)
		return 0
	}

	for _, node := range nodes {

		fmt.Printf("\n\n\nInitial state of community before processing node %d: %v\n",
			node, s.C2N)
		fmt.Printf("Initial community assignment %d\n\n\n",
			s.N2C)
		fmt.Printf("Processing node %d in community %d\n", node, s.N2C[node])
		oldComm := s.N2C[node]
		
		// Get neighbor communities
		neighborComms := s.getNeighborCommunities(node)
		
		// Find best community
		bestComm := oldComm
		bestGain := 0.0
		
		for _, nc := range neighborComms {
			if nc.Community == oldComm {
				continue
			}

			// Check size constraint
			if s.Config.MaxCommunitySize > 0 {
				currentSize := len(s.C2N[nc.Community])
				if nc.Community != oldComm && currentSize >= s.Config.MaxCommunitySize {
					continue
				}
			}

			gain := s.modularityGain(node, nc.Community, nc.Weight)
			fmt.Printf("If node %d moves to community %d, gain: %.4f\n",
				node, nc.Community, gain)

			if gain > bestGain {
				bestComm = nc.Community
				bestGain = gain
			}
		}

		if bestComm != oldComm && bestGain > s.Config.MinModularity {
			s.removeNodeFromCommunity(node, oldComm)
			s.insertNodeIntoCommunity(node, bestComm)
			moves++
		}

		fmt.Printf("Node %d moved from community %d to %d (gain: %.4f)\n",
			node, oldComm, bestComm, bestGain)
		actualModularity := s.GetModularity()
		fmt.Printf("Current modularity after move: %.4f\n", actualModularity)
		// Print community assignment after move for all communities
		fmt.Printf("Current community assignments: %v\n", s.N2C)
		fmt.Printf("Current community sizes: %v\n", s.C2N)

		// Validate state after each move
		if err := s.ValidateState(); err != nil {
			fmt.Printf("State validation failed after moving node %d: %v\n", node, err)
			return moves
		}
	}
	
	return moves
}

// getNeighborCommunities returns the communities of node's neighbors with weights
func (s *LouvainState) getNeighborCommunities(node int) []NeighborWeight {
	commWeights := make(map[int]float64)
	
	// Add current community
	currentComm := s.N2C[node]
	commWeights[currentComm] = 0
	
	// Get all neighbors
	neighbors := s.Graph.GetNeighbors(node)
	
	for neighbor, weight := range neighbors {
		if neighbor != node {
			comm := s.N2C[neighbor]
			commWeights[comm] += weight
		}
	}
	
	// Convert to slice
	result := make([]NeighborWeight, 0, len(commWeights))
	for comm, weight := range commWeights {
		result = append(result, NeighborWeight{
			Community: comm,
			Weight:    weight,
		})
	}
	
	return result
}

// removeNodeFromCommunity removes a node from its community
func (s *LouvainState) removeNodeFromCommunity(node int, comm int) {
	// Validate inputs
	if node < 0 || node >= s.Graph.NumNodes {
		fmt.Printf("ERROR: Attempting to remove invalid node %d\n", node)
		return
	}
	
	if s.N2C[node] != comm {
		fmt.Printf("ERROR: Node %d is in community %d, not %d\n", node, s.N2C[node], comm)
		return
	}

	// Update weights
	degree := s.Graph.GetNodeDegree(node)
	
	// Calculate weight to community
	weightToComm := 0.0
	for _, member := range s.C2N[comm] {
		weightToComm += s.Graph.GetEdgeWeight(node, member)
	}
	
	s.Tot[comm] -= degree
	s.In[comm] -= (2 * weightToComm)

	// Update community node list
	nodes := s.C2N[comm]
	nodeFound := false
	for i, n := range nodes {
		if n == node {
			s.C2N[comm] = append(nodes[:i], nodes[i+1:]...)
			nodeFound = true
			break
		}
	}
	
	if !nodeFound {
		fmt.Printf("ERROR: Node %d not found in C2N[%d]\n", node, comm)
		return
	}
	
	if len(s.C2N[comm]) == 0 {
		// If community is empty, remove it
		delete(s.C2N, comm)
		delete(s.In, comm)
		delete(s.Tot, comm)
	}
	
	s.N2C[node] = -1 // Mark as unassigned
}

// insertNodeIntoCommunity inserts a node into a community
func (s *LouvainState) insertNodeIntoCommunity(node int, comm int) {
	// Validate inputs
	if node < 0 || node >= s.Graph.NumNodes {
		fmt.Printf("ERROR: Attempting to insert invalid node %d\n", node)
		return
	}
	
	if s.N2C[node] != -1 {
		fmt.Printf("ERROR: Node %d already assigned to community %d\n", node, s.N2C[node])
		return
	}

	// Update mapping
	s.N2C[node] = comm
	s.C2N[comm] = append(s.C2N[comm], node)
	
	// Update weights
	degree := s.Graph.GetNodeDegree(node)
	
	// Calculate weight to community
	weightToComm := 0.0
	for _, member := range s.C2N[comm] {
		weightToComm += s.Graph.GetEdgeWeight(node, member)
	}

	s.Tot[comm] += degree
	s.In[comm] += 2*weightToComm 
}

// Temporary modularityGain function
func (s *LouvainState) modularityGain(node int, targetComm int, k_i_in float64) float64 {
	currentComm := s.N2C[node]
	if currentComm == targetComm {
		return 0.0
	}
	// k_i_in 

	oldModularity := s.GetModularity()

	s.removeNodeFromCommunity(node, currentComm)
	s.insertNodeIntoCommunity(node, targetComm)

	newModularity := s.GetModularity()
	s.removeNodeFromCommunity(node, targetComm)
	s.insertNodeIntoCommunity(node, currentComm)
	
	return newModularity - oldModularity
}

// CreateSuperGraph creates a new graph where nodes are communities
func (s *LouvainState) CreateSuperGraph() (*NormalizedGraph, map[int][]int, error) {
	communityMap := make(map[int][]int)
	
	// Count non-empty communities
	numCommunities := 0
	commToNewIndex := make(map[int]int)
	
	for comm, nodes := range s.C2N {
		if len(nodes) > 0 {
			commToNewIndex[comm] = numCommunities
			communityMap[numCommunities] = make([]int, len(nodes))
			copy(communityMap[numCommunities], nodes)
			numCommunities++
		}
	}
	
	// Create super graph
	superGraph := NewNormalizedGraph(numCommunities)
	
	// Calculate community weights
	for newIdx, originalComm := range commToNewIndex {
		communityWeight := 0.0
		for _, node := range s.C2N[originalComm] {
			communityWeight += s.Graph.Weights[node]
		}
		superGraph.Weights[newIdx] = communityWeight
	}
	
	// Add edges between communities
	communityEdges := make(map[string]float64)
	
	for i := 0; i < s.Graph.NumNodes; i++ {
		comm1 := commToNewIndex[s.N2C[i]]
		
		neighbors := s.Graph.GetNeighbors(i)
		for neighbor, weight := range neighbors {
			comm2 := commToNewIndex[s.N2C[neighbor]]
			
			// Create edge key with consistent ordering
			var key string
			if comm1 <= comm2 {
				key = fmt.Sprintf("%d-%d", comm1, comm2)
			} else {
				key = fmt.Sprintf("%d-%d", comm2, comm1)
			}
			
			communityEdges[key] += weight
		}
	}
	
	// Add aggregated edges to super graph
	for key, weight := range communityEdges {
		var from, to int
		fmt.Sscanf(key, "%d-%d", &from, &to)
		superGraph.AddEdge(from, to, weight)
	}

	return superGraph, communityMap, nil
}

// GetLevelInfo returns information about the current level
func (s *LouvainState) GetLevelInfo(level int) LevelInfo {
	communities := make(map[int][]int)
	communityMap := make(map[int]int)
	
	// Build community structures
	numComm := 0
	for comm, nodes := range s.C2N {
		if len(nodes) > 0 {
			communities[comm] = nodes
			for _, node := range nodes {
				communityMap[node] = comm
			}
			numComm++
		}
	}
	
	// Count moves (nodes not in their original community)
	moves := 0
	for i := 0; i < s.Graph.NumNodes; i++ {
		if s.N2C[i] != i {
			moves++
		}
	}
	
	return LevelInfo{
		Level:          level,
		Communities:    communities,
		CommunityMap:   communityMap,
		Graph:          s.Graph.Clone(),
		Modularity:     s.GetModularity(),
		NumCommunities: numComm,
		NumMoves:       moves,
	}
}

// GetFinalCommunities returns the final community assignment for nodes
func (s *LouvainState) GetFinalCommunities() map[int]int {
	result := make(map[int]int)
	
	if s.Graph.NumNodes != len(s.N2C) {
		fmt.Printf("ERROR: Node count mismatch in GetFinalCommunities\n")
		return nil
	}
	
	// Ensure all nodes are included
	for i := 0; i < s.Graph.NumNodes; i++ {
		if s.N2C[i] >= 0 {
			result[i] = s.N2C[i]
		} else {
			fmt.Printf("WARNING: Node %d not assigned to any community\n", i)
			return nil
		}
	}
	
	// Additional validation: check C2N consistency
	totalNodesInC2N := 0
	for comm, nodes := range s.C2N {
		totalNodesInC2N += len(nodes)
		for _, node := range nodes {
			if s.N2C[node] != comm {
				fmt.Printf("ERROR: Inconsistency - node %d in C2N[%d] but N2C[%d]=%d\n", 
					node, comm, node, s.N2C[node])
			}
		}
	}
	if totalNodesInC2N != s.Graph.NumNodes {
		fmt.Printf("ERROR: Total nodes in C2N (%d) does not match graph node count (%d)\n",
			totalNodesInC2N, s.Graph.NumNodes)
		return nil
	}
	
	return result
}

// getMemoryUsage returns current memory usage in MB
func getMemoryUsage() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc / 1024 / 1024)
}

func (s *LouvainState) ValidateState() error {
	// Check N2C and C2N consistency
	for node := 0; node < s.Graph.NumNodes; node++ {
		comm := s.N2C[node]
		if comm < 0 {
			continue // Unassigned node
		}
		
		found := false
		for _, n := range s.C2N[comm] {
			if n == node {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("node %d in N2C[%d] but not in C2N[%d]", node, comm, comm)
		}
	}
	
	for comm, nodes := range s.C2N {
		for _, node := range nodes {
			if s.N2C[node] != comm {
				return fmt.Errorf("node %d in C2N[%d] but N2C[%d]=%d", node, comm, node, s.N2C[node])
			}
		}
	}
	
	return nil
}

func (s *LouvainState) cleanupEmptyCommunities() {
    for comm, nodes := range s.C2N {
        if len(nodes) == 0 {
            delete(s.C2N, comm)
            delete(s.In, comm)
            delete(s.Tot, comm)
        }
    }
}
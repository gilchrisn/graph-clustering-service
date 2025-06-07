package louvain

import (
	"fmt"
	"math/rand"
	"runtime"
	// "sort"
	// "sync"
	"time"
)

// RunLouvain executes the complete Louvain algorithm
func RunLouvain(graph *HomogeneousGraph, config LouvainConfig) (*LouvainResult, error) {
	startTime := time.Now()
	
	// Validate input
	if err := graph.Validate(); err != nil {
		return nil, fmt.Errorf("invalid graph: %w", err)
	}
	
	// Initialize random seed
	if config.RandomSeed < 0 {
		config.RandomSeed = time.Now().UnixNano()
	}
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
			config.ProgressCallback(level, 0, fmt.Sprintf("Starting level %d with %d nodes", level, len(state.Graph.Nodes)))
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
		if len(superGraph.Nodes) >= len(state.Graph.Nodes) {
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
	result.Statistics.RuntimeMS = time.Since(startTime).Milliseconds()
	result.Statistics.MemoryPeakMB = getMemoryUsage()

	return result, nil
}

// NewLouvainState creates a new Louvain state with initial communities
func NewLouvainState(graph *HomogeneousGraph, config LouvainConfig) *LouvainState {
	state := &LouvainState{
		Graph:            graph,
		Config:           config,
		N2C:              make(map[string]int),
		C2N:              make(map[int][]string),
		In:               make(map[int]float64),
		Tot:              make(map[int]float64),
		CommunityCounter: 0,
		Iteration:        0,
	}
	
	// Initialize each node in its own community
	for _, nodeID := range graph.NodeList {
		commID := state.CommunityCounter
		state.CommunityCounter++
		
		state.N2C[nodeID] = commID
		state.C2N[commID] = []string{nodeID}
		
		// Calculate self-loops and degree
		selfLoop := graph.GetEdgeWeight(nodeID, nodeID)
		degree := graph.GetNodeDegree(nodeID)
		
		state.In[commID] = selfLoop
		state.Tot[commID] = degree
	}
	
	return state
}

// NewLouvainStateFromCommunities creates a state where nodes are communities from previous level
func NewLouvainStateFromCommunities(graph *HomogeneousGraph, config LouvainConfig, communityMap map[string][]string) *LouvainState {
	state := NewLouvainState(graph, config)
	state.CommunityCounter = len(communityMap)
	return state
}

// ExecuteOneLevel performs one level of the Louvain algorithm
func (s *LouvainState) ExecuteOneLevel() (bool, error) {
	improvement := false
	nbMoves := 0
	s.Iteration = 0
	
	for {
		s.Iteration++
		iterMoves := 0
		
		// Create random order for nodes
		nodeOrder := make([]string, len(s.Graph.NodeList))
		copy(nodeOrder, s.Graph.NodeList)
		rand.Shuffle(len(nodeOrder), func(i, j int) {
			nodeOrder[i], nodeOrder[j] = nodeOrder[j], nodeOrder[i]
		})
		
		// Process nodes in chunks for better cache locality
		chunkSize := s.Config.ChunkSize
		if chunkSize <= 0 {
			chunkSize = 32
		}
		
		for i := 0; i < len(nodeOrder); i += chunkSize {
			end := i + chunkSize
			if end > len(nodeOrder) {
				end = len(nodeOrder)
			}
			
			// Process chunk
			moves := s.processNodeChunk(nodeOrder[i:end])
			iterMoves += moves
		}
		
		nbMoves += iterMoves
		
		// Check convergence
		if iterMoves == 0 {
			break
		}
		
		improvement = true
		
		// Progress callback
		if s.Config.ProgressCallback != nil {
			s.Config.ProgressCallback(-1, s.Iteration, 
				fmt.Sprintf("Iteration %d: %d moves", s.Iteration, iterMoves))
		}
	}
	
	return improvement && nbMoves > 0, nil
}

// processNodeChunk processes a chunk of nodes
func (s *LouvainState) processNodeChunk(nodes []string) int {
	moves := 0
	
	for _, node := range nodes {
		oldComm := s.N2C[node]
		
		// Get neighbor communities
		neighborComms := s.getNeighborCommunities(node)
		
		// Remove node from its current community
		s.removeNodeFromCommunity(node, oldComm)
		
		// Find best community
		bestComm := oldComm
		bestGain := 0.0
		
		for _, nc := range neighborComms {
			// Check size constraint
			if s.Config.MaxCommunitySize > 0 {
				currentSize := len(s.C2N[nc.Community])
				if nc.Community != oldComm && currentSize >= s.Config.MaxCommunitySize {
					continue
				}
			}
			
			gain := s.modularityGain(node, nc.Community, nc.Weight)
			if gain > bestGain {
				bestComm = nc.Community
				bestGain = gain
			}
		}
		
		// Insert node into best community
		s.insertNodeIntoCommunity(node, bestComm)
		
		if bestComm != oldComm && bestGain > s.Config.MinModularity {
			moves++
		}
	}
	
	return moves
}

// getNeighborCommunities returns the communities of node's neighbors with weights
func (s *LouvainState) getNeighborCommunities(node string) []NeighborWeight {
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
func (s *LouvainState) removeNodeFromCommunity(node string, comm int) {
	// Update community node list
	nodes := s.C2N[comm]
	for i, n := range nodes {
		if n == node {
			s.C2N[comm] = append(nodes[:i], nodes[i+1:]...)
			break
		}
	}
	
	// Update weights
	degree := s.Graph.GetNodeDegree(node)
	selfLoop := s.Graph.GetEdgeWeight(node, node)
	
	// Calculate weight to community
	weightToComm := 0.0
	for _, member := range s.C2N[comm] {
		weightToComm += s.Graph.GetEdgeWeight(node, member)
	}
	
	s.Tot[comm] -= degree
	s.In[comm] -= 2*weightToComm + selfLoop

    if len(s.C2N[comm]) == 0 {
        delete(s.C2N, comm)
        delete(s.In, comm) 
        delete(s.Tot, comm)
    }
}

// insertNodeIntoCommunity inserts a node into a community
func (s *LouvainState) insertNodeIntoCommunity(node string, comm int) {
	// Update mapping
	s.N2C[node] = comm
	s.C2N[comm] = append(s.C2N[comm], node)
	
	// Update weights
	degree := s.Graph.GetNodeDegree(node)
	selfLoop := s.Graph.GetEdgeWeight(node, node)
	
	// Calculate weight to community
	weightToComm := 0.0
	for _, member := range s.C2N[comm] {
		if member != node {
			weightToComm += s.Graph.GetEdgeWeight(node, member)
		}
	}
	
	s.Tot[comm] += degree
	s.In[comm] += 2*weightToComm + selfLoop
}

// modularityGain calculates the modularity gain from moving a node to a community
func (s *LouvainState) modularityGain(node string, comm int, dnodecomm float64) float64 {
	totc := s.Tot[comm]
	degc := s.Graph.GetNodeDegree(node)
	m2 := 2 * s.Graph.TotalWeight

	if m2 == 0 {
		return 0
	}
	
	return dnodecomm/m2 - (totc*degc)/(m2*m2)
}

// CreateSuperGraph creates a new graph where nodes are communities
func (s *LouvainState) CreateSuperGraph() (*HomogeneousGraph, map[string][]string, error) {
	// Renumber communities to be consecutive
	communityRenumber := make(map[int]int)
	communityMap := make(map[string][]string)
	finalComm := 0
	
	for comm, nodes := range s.C2N {
		if len(nodes) > 0 {
			communityRenumber[comm] = finalComm
			commID := fmt.Sprintf("c%d", finalComm)
			communityMap[commID] = nodes
			finalComm++
		}
	}
	
	// Create new graph
	superGraph := NewHomogeneousGraph()
	
	// Add nodes (communities become nodes)
	for commID := range communityMap {
		superGraph.AddNode(commID, 1.0)
	}
	
	// Add edges between communities
	communityEdges := make(map[EdgeKey]float64)
	
	for _, nodeID := range s.Graph.NodeList {
		comm1 := s.N2C[nodeID]
		newComm1 := communityRenumber[comm1]
		
		neighbors := s.Graph.GetNeighbors(nodeID)
		for neighbor, weight := range neighbors {
			comm2 := s.N2C[neighbor]
			newComm2 := communityRenumber[comm2]
			
			if newComm1 <= newComm2 { // Avoid double counting
				from := fmt.Sprintf("c%d", newComm1)
				to := fmt.Sprintf("c%d", newComm2)
				key := EdgeKey{From: from, To: to}
				communityEdges[key] += weight
			}
		}
	}
	
	// Add aggregated edges to super graph
	for edge, weight := range communityEdges {
		superGraph.AddEdge(edge.From, edge.To, weight)
	}
	
	return superGraph, communityMap, nil
}

// GetLevelInfo returns information about the current level
func (s *LouvainState) GetLevelInfo(level int) LevelInfo {
	communities := make(map[int][]string)
	communityMap := make(map[string]int)
	
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
	for i, nodeID := range s.Graph.NodeList {
		if s.N2C[nodeID] != i {
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

// GetFinalCommunities returns the final community assignment for original nodes
func (s *LouvainState) GetFinalCommunities() map[string]int {
	result := make(map[string]int)
	for node, comm := range s.N2C {
		result[node] = comm
	}
	return result
}

// getMemoryUsage returns current memory usage in MB
func getMemoryUsage() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc / 1024 / 1024)
}
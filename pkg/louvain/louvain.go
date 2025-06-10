package louvain

import (
	"fmt"
	"math/rand"
	"runtime"
	// "sort"
	// "sync"
	"time"
	"strings"
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
			fmt.Printf("No compression at level %d (%d -> %d nodes), stopping\n", 
				level, len(state.Graph.Nodes), len(superGraph.Nodes))
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
	nbMoves := 0 // Total moves in this level
	s.Iteration = 0 // Reset iteration count for this level

	// Print initial modularity
	fmt.Printf("Starting Louvain level with %d nodes, initial modularity: %.4f\n",
		len(s.Graph.Nodes), s.GetModularity())
	
	for {
		s.Iteration++
		iterMoves := 0 // Moves in this iteration
		
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
		// fmt.Printf("Iteration %d completed with %d moves\n", s.Iteration, iterMoves)
	}

	// Print final modularity for this level
	finalModularity := s.GetModularity()
	fmt.Printf("Final modularity after level %d: %.4f, moves: %d\n",
		s.Iteration, finalModularity, nbMoves)

	return improvement && nbMoves > 0, nil
}

// processNodeChunk processes a chunk of nodes
func (s *LouvainState) processNodeChunk(nodes []string) int {
	moves := 0

	// Validate state before processing
	if err := s.ValidateState(); err != nil {
		fmt.Printf("State validation failed before processing chunk: %v\n", err)
		return 0
	}
	

	for _, node := range nodes {
		oldComm := s.N2C[node]
		
		// Get neighbor communities
		neighborComms := s.getNeighborCommunities(node)
		
		// Find best community
		bestComm := oldComm
		bestGain := 0.0

		// get the current modularity
		
		for _, nc := range neighborComms {
			// Skip if community is the same as old community
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

		// Validate state after each move
		if err := s.ValidateState(); err != nil {
			fmt.Printf("State validation failed after moving node %s: %v\n", node, err)
			// Try to recover or return error
			return moves
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
	// Validate inputs
	if _, nodeExists := s.Graph.Nodes[node]; !nodeExists {
		fmt.Printf("ERROR: Attempting to remove non-existent node %s\n", node)
		return
	}
	
	if currentComm, exists := s.N2C[node]; !exists {
		fmt.Printf("ERROR: Node %s not in N2C mapping\n", node)
		return
	} else if currentComm != comm {
		fmt.Printf("ERROR: Node %s is in community %d, not %d\n", node, currentComm, comm)
		return
	}

	// Update weights
	degree := s.Graph.GetNodeDegree(node)
	// selfLoop := s.Graph.GetEdgeWeight(node, node)
	
	// Calculate weight to community
	weightToComm := 0.0
	for _, member := range s.C2N[comm] {
		weightToComm += s.Graph.GetEdgeWeight(node, member) 
	}
	
	s.Tot[comm] -= degree
	s.In[comm] -= (2 * weightToComm)

	// fmt.Printf("\n\n\n\nRemoving node %s from community %d\n", node, comm)
	// // Show current state before removal
	// fmt.Printf("Current communities: %v\n", s.C2N)
	// fmt.Printf("Current N2C: %v\n", s.N2C)
	// fmt.Printf("Current In: %v\n", s.In)
	// fmt.Printf("Current Tot: %v\n\n", s.Tot)


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
		fmt.Printf("ERROR: Node %s not found in C2N[%d]\n", node, comm)
		return
	}
	
	// // Check if in or tot < 0.
	// if s.In[comm] < 0 || s.Tot[comm] < 0 {
	// 	fmt.Printf("ERROR: Negative weights after removing node %s from community %d\n", node, comm)
	// 	fmt.Printf("In[%d] = %.4f, Tot[%d] = %.4f\n", comm, s.In[comm], comm, s.Tot[comm])
	// 	//print state
	// 	fmt.Printf("Current communities: %v\n", s.C2N)
	// 	fmt.Printf("Current N2C: %v\n", s.N2C)
	// 	fmt.Printf("Current In: %v\n", s.In)
	// 	fmt.Printf("Current Tot: %v\n", s.Tot)
	// 	// values updated
	// 	fmt.Printf("Node %s has degree %.4f\n", node, degree)
	// 	// panic
	// 	panic(fmt.Sprintf("Negative weights after removing node %s from community %d", node, comm))
	// }
	if len(s.C2N[comm]) == 0 {

		// verify if tot = in = 0. and verify if the node id starts with c
		if !(s.Tot[comm] == 0 && s.In[comm] == 0) && !strings.HasPrefix(node, "c") {
			fmt.Printf("Ermmmm... what the sigma?\n")
			// Print current state
			//print degree and selfLoop
			fmt.Printf("Node %s has degree %.4f\n", node, degree)
			fmt.Printf("Current communities: %v\n", s.C2N)
			fmt.Printf("Current N2C: %v\n", s.N2C)
			fmt.Printf("Current In: %v\n", s.In)
			fmt.Printf("Current Tot: %v\n", s.Tot)
			fmt.Printf("Tot[%d] = %.4f, In[%d] = %.4f\n", comm, s.Tot[comm], comm, s.In[comm])
		}
		// If community is empty, remove it
		delete(s.C2N, comm)
		delete(s.In, comm)
		delete(s.Tot, comm)
	} 

	
	delete(s.N2C, node)
}

// insertNodeIntoCommunity inserts a node into a community
func (s *LouvainState) insertNodeIntoCommunity(node string, comm int) {
	// Validate inputs
	if _, nodeExists := s.Graph.Nodes[node]; !nodeExists {
		fmt.Printf("ERROR: Attempting to insert non-existent node %s\n", node)
		return
	}
	
	if _, exists := s.N2C[node]; exists {
		fmt.Printf("ERROR: Node %s already in N2C mapping\n", node)
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
		weightToComm += s.Graph.GetEdgeWeight(node, member) // self loops are already counted
	}

	s.Tot[comm] += degree
	s.In[comm] += 2*weightToComm 
}

// // modularityGain calculates the NET modularity gain from moving a node between communities
// func (s *LouvainState) modularityGain(node string, targetComm int, k_i_in float64) float64 {
//     currentComm := s.N2C[node]
    
//     // If moving to same community, no gain
//     if currentComm == targetComm {
//         return 0.0
//     }
    
//     k_i := s.Graph.GetNodeDegree(node)
//     M := s.Graph.TotalWeight
    
//     if M == 0 {
//         return 0.0
//     }

// 	s_tot := s.Tot[targetComm]

// 	gain := k_i_in / (2 * M) - (s_tot * k_i) / (2 * M * M)

// 	// Debug
// 	// fmt.Printf("Modularity gain for node %s moving from c%d to c%d: %.4f (k_i_in=%.4f, s_tot=%.4f, M=%.4f, k_i=%.4f)\n",
// 	// 	node, currentComm, targetComm, gain, k_i_in, s_tot, M, k_i)
	
// 	return gain
// }

// Temporary modularityGain function. This function is horrible
func (s *LouvainState) modularityGain(node string, targetComm int, k_i_in float64) float64 {
	// Calculate current modularity before move
	currentComm := s.N2C[node]
	if currentComm == targetComm {
		return 0.0 // No gain if moving to same community
	}

	oldModularity := s.GetModularity()

	s.removeNodeFromCommunity(node, currentComm)
	s.insertNodeIntoCommunity(node, targetComm)

	newModularity := s.GetModularity()
	s.removeNodeFromCommunity(node, targetComm)
	s.insertNodeIntoCommunity(node, currentComm)

	
	return newModularity - oldModularity
}

// CreateSuperGraph creates a new graph where nodes are communities
func (s *LouvainState) CreateSuperGraph() (*HomogeneousGraph, map[string][]string, error) {
	communityMap := make(map[string][]string)
	
	// Create super nodes using ORIGINAL community IDs (no renumbering!)
	superGraph := NewHomogeneousGraph()
	
	totalNodes := 0
	for comm, nodes := range s.C2N {
		if len(nodes) == 0 {
			continue
		}
		totalNodes += len(nodes)
		
		commID := fmt.Sprintf("c%d", comm)
		communityMap[commID] = make([]string, len(nodes))
		copy(communityMap[commID], nodes)
		
		// Calculate community weight (sum of node weights)
		communityWeight := 0.0
		for _, node := range nodes {
			if nodeData, exists := s.Graph.Nodes[node]; exists {
				communityWeight += nodeData.Weight
			}
		}
		
		superGraph.AddNode(commID, communityWeight)
	}
	
	if totalNodes != len(s.Graph.NodeList) {
		return nil, nil, fmt.Errorf("node count mismatch: communities have %d nodes, graph has %d", 
			totalNodes, len(s.Graph.NodeList))
	}
	
	// Add edges between communities
	communityEdges := make(map[EdgeKey]float64)
	
	for _, nodeID := range s.Graph.NodeList {
		comm1 := s.N2C[nodeID]
		
		neighbors := s.Graph.GetNeighbors(nodeID)
		for neighbor, weight := range neighbors {
			comm2 := s.N2C[neighbor]
			
			// Create edge key with consistent ordering
			var from, to string
			if comm1 <= comm2 {
				from = fmt.Sprintf("c%d", comm1)
				to = fmt.Sprintf("c%d", comm2)
			} else {
				from = fmt.Sprintf("c%d", comm2)
				to = fmt.Sprintf("c%d", comm1)
			}
			
			key := EdgeKey{From: from, To: to}
			communityEdges[key] += weight
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
	
	if (len(s.Graph.NodeList) != len(s.N2C)) {
		fmt.Printf("ERROR: Node count mismatch in GetFinalCommunities\n")
		return nil // This will help us catch the bug
	}
	
	// Ensure all nodes from the original graph are included
	for _, nodeID := range s.Graph.NodeList {
		if comm, exists := s.N2C[nodeID]; exists {
			result[nodeID] = comm
		} else {
			fmt.Printf("WARNING: Node %s missing from N2C mapping\n", nodeID)
			// Assign to a default community or throw error
			return nil // This will help us catch the bug
		}
	}
	
	// Additional validation: check C2N consistency
	totalNodesInC2N := 0
	for comm, nodes := range s.C2N {
		totalNodesInC2N += len(nodes)
		for _, node := range nodes {
			if s.N2C[node] != comm {
				fmt.Printf("ERROR: Inconsistency - node %s in C2N[%d] but N2C[%s]=%d\n", 
					node, comm, node, s.N2C[node])
			}
		}
	}
	if totalNodesInC2N != len(s.Graph.NodeList) {
		fmt.Printf("ERROR: Total nodes in C2N (%d) does not match graph node count (%d)\n",
			totalNodesInC2N, len(s.Graph.NodeList))
		return nil // This will help us catch the bug
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
	for node, comm := range s.N2C {
		found := false
		for _, n := range s.C2N[comm] {
			if n == node {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("node %s in N2C[%d] but not in C2N[%d]", node, comm, comm)
		}
	}
	
	for comm, nodes := range s.C2N {
		for _, node := range nodes {
			if s.N2C[node] != comm {
				return fmt.Errorf("node %s in C2N[%d] but N2C[%s]=%d", node, comm, node, s.N2C[node])
			}
		}
	}
	
	// Check all graph nodes are assigned
	for _, node := range s.Graph.NodeList {
		if _, exists := s.N2C[node]; !exists {
			return fmt.Errorf("graph node %s not in N2C", node)
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


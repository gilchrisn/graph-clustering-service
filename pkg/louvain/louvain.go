package louvain

import (
	"fmt"
	"math/rand"
	"runtime"
	"time"
	"strings"
	"sort"
	// "math"
)

// RunLouvain executes the complete Louvain algorithm on normalized graph
func RunLouvain(graph *NormalizedGraph, config LouvainConfig) (*LouvainResult, error) {
	startTime := time.Now()
	
	// //Print full graph structure
	// fmt.Println("Graph structure (edges):")
	// for i := 0; i < graph.NumNodes; i++ {
	// 	neighbors := graph.GetNeighbors(i)
	// 	fmt.Printf("Node %d: ", i)
	// 	for neighbor, weight := range neighbors {
	// 		fmt.Printf("(%d, %.2f) ", neighbor, weight)
	// 	}
	// 	fmt.Println()
	// }

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
	for improvement && level < 3 {
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
		
		// if there is only 1 community, and that community only has 1 node, we can stop and not include this level
		if len(levelInfo.Communities) == 1 {
			// Get the single community (regardless of its ID)
			var singleCommunityNodes []int
			for _, nodes := range levelInfo.Communities {
				singleCommunityNodes = nodes
				break // There's only one, so we can break after the first iteration
			}
			
			// If the single community has only 1 node, this level is degenerate
			if len(singleCommunityNodes) == 1 {
				fmt.Printf("Stopping at level %d with only 1 community of 1 node (degenerate case)\n", level)
				// Remove this degenerate level from results
				result.Levels = result.Levels[:len(result.Levels)-1]
				break
			}
			
			// If the single community has all nodes, no further improvement is possible
			if len(singleCommunityNodes) == state.Graph.NumNodes {
				fmt.Printf("Stopping at level %d: all nodes in single community, no further clustering possible\n", level)
				break
			}
		}

		if !improvement {
			break
		}

			
		// Create super graph for next level
		superGraph, communityMap, superNodeToCommMap, err := state.CreateSuperGraph()
		if err != nil {
			return nil, fmt.Errorf("error creating super graph at level %d: %w", level, err)
		}

		result.Levels[level].SuperNodeToCommMap = superNodeToCommMap
		
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


// This creates a fresh LouvainState for the super-graph where each super-node
// starts in its own community (ready for the next level of optimization)

func NewLouvainStateFromCommunities(superGraph *NormalizedGraph, config LouvainConfig, communityMap map[int][]int) *LouvainState {
	// Create the basic state structure
	state := &LouvainState{
		Graph:            superGraph,
		Config:           config,
		N2C:              make([]int, superGraph.NumNodes),      // super-node -> community mapping
		C2N:              make(map[int][]int),                   // community -> super-nodes mapping
		In:               make(map[int]float64),                 // internal weights of communities
		Tot:              make(map[int]float64),                 // total weights of communities
		CommunityCounter: 0,
		Iteration:        0,
	}
	
	// =============================================================================
	// INITIALIZE EACH SUPER-NODE IN ITS OWN COMMUNITY
	// =============================================================================
	// At the start of each level, every super-node is in its own community
	// This is the same as the original initialization, but for super-nodes
	
	for superNodeIdx := 0; superNodeIdx < superGraph.NumNodes; superNodeIdx++ {
		// Each super-node gets its own unique community ID
		communityID := state.CommunityCounter
		state.CommunityCounter++
		
		// Set up the mappings
		state.N2C[superNodeIdx] = communityID                    // super-node -> community
		state.C2N[communityID] = []int{superNodeIdx}             // community -> [super-node]
		
		// =============================================================================
		// INITIALIZE COMMUNITY WEIGHTS (In and Tot)
		// =============================================================================
		
		// Calculate self-loop weight (edges within this super-node's original community)
		selfLoopWeight := superGraph.GetEdgeWeight(superNodeIdx, superNodeIdx)
		
		// Calculate total degree (sum of all edges from this super-node)
		totalDegree := superGraph.GetNodeDegree(superNodeIdx)
		
		// Set community statistics
		state.In[communityID] = 2 * selfLoopWeight    // Internal weight = self-loops
		state.Tot[communityID] = totalDegree      // Total weight = total degree
	}
	
	return state
}

// ExecuteOneLevel performs one level of the Louvain algorithm
func (s *LouvainState) ExecuteOneLevel() (bool, error) {
	improvement := false 
	nbMoves := 0
	s.Iteration = 0

	// Print initial state
	// s.PrintState(fmt.Sprintf("LEVEL START - Level with %d nodes", s.Graph.NumNodes), "ALL")

	fmt.Printf("\n\n\nStarting Louvain level with %d nodes, initial modularity: %.4f\n",
		s.Graph.NumNodes, s.GetModularity())

	// ==============================================================================================

	for s.Iteration < s.Config.MaxIterations {
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

			if moves > 0 {
				improvement = true
			}
			iterMoves += moves
		}
		
		nbMoves += iterMoves
		
		if s.Config.ProgressCallback != nil {
			s.Config.ProgressCallback(-1, s.Iteration, 
				fmt.Sprintf("Iteration %d: %d moves", s.Iteration, iterMoves))
		}
	}

	finalModularity := s.GetModularity()
	fmt.Printf("Final modularity after iteration %d: %.4f, moves: %d\n",
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
		oldComm := s.N2C[node]
		
		// Get neighbor communities
		neighborComms := s.getNeighborCommunities(node)
		
		// Find best community
		bestComm := oldComm
		
		edgeToFrom := 0.0 
		for _, member := range s.C2N[oldComm] {
			edgeToFrom += s.Graph.GetEdgeWeight(node, member)
		}

		bestGain := s.modularityGain(node, oldComm, edgeToFrom)

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

			if gain > bestGain {
				// fmt.Printf("Node %d: found better community %d (gain: %.4f)\n", node, nc.Community, gain)
				bestComm = nc.Community
				bestGain = gain
			}
		}

		// if bestComm != oldComm && bestGain > s.Config.MinModularity {
		if bestComm != oldComm {
			// fmt.Printf("Node %d: moving from community %d to %d (gain: %.4f)\n",
			// 	node, oldComm, bestComm, bestGain)

			if err := s.moveNode(node, oldComm, bestComm); err != nil {
				fmt.Printf("Error moving node %d: %v\n", node, err)
				continue
			}
			moves++
		}

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
		comm := s.N2C[neighbor]
		commWeights[comm] += weight
	}
	
	// Convert to slice
	result := make([]NeighborWeight, 0, len(commWeights))
	for comm, weight := range commWeights {
		result = append(result, NeighborWeight{
			Community: comm,
			Weight:    weight,
		})
	}
	
						// ADDED FOR REPRODUCIBILITY
						sort.Slice(result, func(i, j int) bool {
							return result[i].Community < result[j].Community
						})
						// END OF ADDED FOR REPRODUCIBILITY

	return result
}


// moveNode atomically moves a node from one community to another
func (s *LouvainState) moveNode(node int, fromComm int, toComm int) error {
	// Validate inputs
	if node < 0 || node >= s.Graph.NumNodes {
		return fmt.Errorf("invalid node %d", node)
	}
	
	if s.N2C[node] != fromComm {
		return fmt.Errorf("node %d is in community %d, not %d", node, s.N2C[node], fromComm)
	}
	
	if fromComm == toComm {
		return nil // No-op
	}
	
	// Calculate weights to both communities 
	weightToFrom := 0.0
	for _, member := range s.C2N[fromComm] {
		weightToFrom += s.Graph.GetEdgeWeight(node, member)
	}
	
	weightToTo := 0.0
	for _, member := range s.C2N[toComm] {
		weightToTo += s.Graph.GetEdgeWeight(node, member)
	}
	
	nodeDegree := s.Graph.GetNodeDegree(node)
	
	// Update FROM community
	s.Tot[fromComm] -= nodeDegree
	s.In[fromComm] -= (2 * weightToFrom)
	
	// Remove node from fromComm
	nodes := s.C2N[fromComm]
	for i, n := range nodes {
		if n == node {
			s.C2N[fromComm] = append(nodes[:i], nodes[i+1:]...)
			break
		}
	}

												// // Print everything for debugging
												// fmt.Printf("Moving node %d from community %d to %d: edgesToFrom: %.4f, edgesToTo: %.4f, "+
												// 	"nodeDegree: %.4f, fromCommDegree: %.4f, toCommDegree: %.4f, wholeWeight: %.4f\n",
												// 	node, fromComm, toComm, weightToFrom, weightToTo, nodeDegree,
												// 	s.Tot[fromComm], s.Tot[toComm], s.Graph.TotalWeight)

												// // for each node in the old comm, print each node's degree and the weight of the edge to the moved node
												// for _, member := range s.C2N[fromComm] {
												// 	edgeWeight := s.Graph.GetEdgeWeight(node, member)
												// 	memberDegree := s.Graph.GetNodeDegree(member)
													// fmt.Printf("  Node %d -> in old community %d: degree=%.4f, edgeWeight=%.4f\n",
												// 		member, fromComm, memberDegree, edgeWeight)
												// }
												// fmt.Printf(" Node %d's self loop weight: %.4f\n", node, s.Graph.GetEdgeWeight(node, node))

	// Clean up empty community
	if len(s.C2N[fromComm]) == 0 {
		delete(s.C2N, fromComm)
		delete(s.In, fromComm)
		delete(s.Tot, fromComm)
	}
	
	// Update TO community
	s.Tot[toComm] += nodeDegree
	s.In[toComm] += 2 * (weightToTo + s.Graph.GetEdgeWeight(node, node)) // Add self-loop weight
	
	// Add node to toComm
	s.N2C[node] = toComm
	s.C2N[toComm] = append(s.C2N[toComm], node)
	
	return nil
}

func (s *LouvainState) modularityGain(node int, targetComm int, k_i_in float64) float64 {
	// Calculate edgesToTo (edges from node to target community)
	edgesToTo := k_i_in  // This is passed as parameter
	
	// Get other variables
	nodeDegree := s.Graph.GetNodeDegree(node)
	toCommDegree := 0.0
	if targetComm >= 0 {
		if tot, exists := s.Tot[targetComm]; exists {
			toCommDegree = tot
		}
	}
	wholeWeight := s.Graph.TotalWeight
	
	// fromCommDegree := 0.0
	// if currentComm >= 0 {
		// 	if tot, exists := s.Tot[currentComm]; exists {
			// 		fromCommDegree = tot
			// 	}
			// }
			
			// edgesToFrom := 0.0
			// for _, member := range s.C2N[currentComm] {
				// 	edgesToFrom += s.Graph.GetEdgeWeight(node, member)
				// }
				
				// // Apply your formula
				// gain := edgesToTo - edgesToFrom + nodeDegree * (fromCommDegree - toCommDegree - nodeDegree) / (2 * wholeWeight)
				
				// 														// // 	// Print all component for debugging
				// 														fmt.Printf("LOUVAIN: Moving node %d to community %d: edgesToTo: %.4f, edgesToFrom: %.4f, nodeDegree: %.4f, "+
				// 																" fromCommDegree: %.4f, toCommDegree: %.4f, wholeWeight: %.4f, gain: %.4f\n",
				// 																node, targetComm, edgesToTo, edgesToFrom, nodeDegree, fromCommDegree, toCommDegree, wholeWeight, gain)
				
				gain := edgesToTo - nodeDegree * toCommDegree / (2 * wholeWeight)
				
				// if true {
				// 	fmt.Printf("LOUVAIN: Moving node %d to community %d: edgesToTo: %.4f, nodeDegree: %.4f, "+
				// 	"toCommDegree: %.4f, wholeWeight: %.4f, gain: %.4f\n",
				// 	node, targetComm, edgesToTo, nodeDegree, toCommDegree, wholeWeight, gain)
				// }
				
	

	return gain
}

// CreateSuperGraph creates a new graph where each community becomes a single super-node
// This prepares the data structures for the next level of Louvain optimization
func (s *LouvainState) CreateSuperGraph() (*NormalizedGraph, map[int][]int, map[int]int, error) {
	// Clean up any empty communities first
	s.cleanupEmptyCommunities()
	
	// Validate current state
	if err := s.ValidateState(); err != nil {
		return nil, nil, nil, fmt.Errorf("invalid state before creating super graph: %w", err)
	}
	
	// =============================================================================
	// STEP 1: AGGREGATE COMMUNITIES INTO SUPER-NODES
	// =============================================================================
	// We need to map old community IDs to new super-node indices (0, 1, 2, ...)
	
	communityMap := make(map[int][]int)     // Maps super-node index -> original nodes it contains
	commToNewIndex := make(map[int]int)     // Maps old community ID -> new super-node index
	

							// // Count non-empty communities and create the mapping
							// numSuperNodes := 0
							// for oldCommID, nodes := range s.C2N {
							// 	if len(nodes) > 0 {
							// 		// This community becomes super-node with index numSuperNodes
							// 		commToNewIndex[oldCommID] = numSuperNodes
									
							// 		// Store which original nodes this super-node represents
							// 		communityMap[numSuperNodes] = make([]int, len(nodes))
							// 		copy(communityMap[numSuperNodes], nodes)
									
							// 		numSuperNodes++
							// 	}
							// }
					// DETERMINISTICALLY SORTED COMMUNITIES
					communityIDs := make([]int, 0, len(s.C2N))
					for oldCommID, nodes := range s.C2N {
						if len(nodes) > 0 {
							communityIDs = append(communityIDs, oldCommID)
						}
					}
					sort.Ints(communityIDs)

					// Process communities in sorted order
					numSuperNodes := 0
					for _, oldCommID := range communityIDs {
						nodes := s.C2N[oldCommID]
						// This community becomes super-node with index numSuperNodes
						commToNewIndex[oldCommID] = numSuperNodes
						
						// Store which original nodes this super-node represents
						communityMap[numSuperNodes] = make([]int, len(nodes))
						copy(communityMap[numSuperNodes], nodes)
						
						numSuperNodes++
					}
					// END OF DETERMINISTICALLY SORTED COMMUNITIES
	
	if numSuperNodes == 0 {
		return nil, nil, nil, fmt.Errorf("no valid communities found")
	}
	
	// Verify all nodes have valid community assignments
	for i := 0; i < s.Graph.NumNodes; i++ {
		nodeComm := s.N2C[i]
		if _, exists := commToNewIndex[nodeComm]; !exists {
			return nil, nil, nil, fmt.Errorf("node %d references invalid community %d", i, nodeComm)
		}
	}
	
	// =============================================================================
	// STEP 2: CREATE NEW SUPER-GRAPH STRUCTURE
	// =============================================================================
	// Create a brand new graph with numSuperNodes nodes
	
	superGraph := NewNormalizedGraph(numSuperNodes)
	
	// =============================================================================
	// STEP 3: CALCULATE SUPER-NODE WEIGHTS
	// =============================================================================
	// Each super-node's weight = sum of weights of all original nodes it contains
	
	for superNodeIdx := 0; superNodeIdx < numSuperNodes; superNodeIdx++ {
		originalNodes := communityMap[superNodeIdx]
		totalWeight := 0.0
		
		for _, originalNode := range originalNodes {
			if originalNode < 0 || originalNode >= len(s.Graph.Weights) {
				return nil, nil, nil, fmt.Errorf("invalid original node index %d", originalNode)
			}
			totalWeight += s.Graph.Weights[originalNode]
		}
		
		superGraph.Weights[superNodeIdx] = totalWeight
	}
	
	// =============================================================================
	// STEP 4: AGGREGATE EDGE WEIGHTS BETWEEN COMMUNITIES
	// =============================================================================
	// For every edge in the original graph, determine which super-nodes it connects
	// and accumulate the edge weights
	
	// Use string keys to track edges between super-nodes (e.g., "0-1", "2-2")
	superEdgeWeights := make(map[string]float64)
	
	// Process every edge in the original graph
	for nodeI := 0; nodeI < s.Graph.NumNodes; nodeI++ {
		// Find which super-node contains nodeI
		commI := s.N2C[nodeI]
		superNodeI, exists := commToNewIndex[commI]
		if !exists {
			return nil, nil, nil, fmt.Errorf("node %d community %d not found in mapping", nodeI, commI)
		}
		
		// Get all neighbors of nodeI
		neighbors := s.Graph.GetNeighbors(nodeI)
		
		for nodeJ, edgeWeight := range neighbors {
			// Find which super-node contains nodeJ
			commJ := s.N2C[nodeJ]
			superNodeJ, exists := commToNewIndex[commJ]
			if !exists {
				return nil, nil, nil, fmt.Errorf("neighbor %d community %d not found in mapping", nodeJ, commJ)
			}
			
			// Create consistent edge key (smaller index first)
			var edgeKey string
			if superNodeI <= superNodeJ {
				edgeKey = fmt.Sprintf("%d-%d", superNodeI, superNodeJ)
			} else {
				edgeKey = fmt.Sprintf("%d-%d", superNodeJ, superNodeI)
			}
			
			// Accumulate the edge weight
			if (nodeJ == nodeI) {
				superEdgeWeights[edgeKey] += 2 * edgeWeight // Self-loops count double
			} else {
				superEdgeWeights[edgeKey] += edgeWeight // Regular edges
			}
		}
	}
	
	// =============================================================================
	// STEP 5: ADD EDGES TO SUPER-GRAPH
	// =============================================================================
	// Convert the accumulated edge weights into actual edges in the super-graph
	
								// edgeCount := 0
								// for edgeKey, totalWeight := range superEdgeWeights {
								// 	// Parse the edge key to get super-node indices
								// 	var fromSuperNode, toSuperNode int
								// 	n, err := fmt.Sscanf(edgeKey, "%d-%d", &fromSuperNode, &toSuperNode)
								// 	if n != 2 || err != nil {
								// 		return nil, nil, nil, fmt.Errorf("failed to parse edge key %s: %w", edgeKey, err)
								// 	}
									
								// 	// Validate super-node indices
								// 	if fromSuperNode < 0 || fromSuperNode >= numSuperNodes || 
								// 	   toSuperNode < 0 || toSuperNode >= numSuperNodes {
								// 		return nil, nil, nil, fmt.Errorf("invalid super-node indices: %d-%d (max: %d)", 
								// 			fromSuperNode, toSuperNode, numSuperNodes-1)
								// 	}
									
								// 	// Add the edge to the super-graph
								// 	// Undirected graph: need to divide weight by 2 because of double counting
								// 	superGraph.AddEdge(fromSuperNode, toSuperNode, totalWeight / 2)
								// 	edgeCount++
								// }
						// DETERMINISTICALLY SORTED EDGES
						edgeKeys := make([]string, 0, len(superEdgeWeights))
						for edgeKey := range superEdgeWeights {
							edgeKeys = append(edgeKeys, edgeKey)
						}
						sort.Strings(edgeKeys)

						// Process edges in sorted order
						edgeCount := 0
						for _, edgeKey := range edgeKeys {
							totalWeight := superEdgeWeights[edgeKey]
							// Parse the edge key to get super-node indices
							var fromSuperNode, toSuperNode int
							n, err := fmt.Sscanf(edgeKey, "%d-%d", &fromSuperNode, &toSuperNode)
							if n != 2 || err != nil {
								return nil, nil, nil, fmt.Errorf("failed to parse edge key %s: %w", edgeKey, err)
							}
							
							// Validate super-node indices
							if fromSuperNode < 0 || fromSuperNode >= numSuperNodes || 
							toSuperNode < 0 || toSuperNode >= numSuperNodes {
								return nil, nil, nil, fmt.Errorf("invalid super-node indices: %d-%d (max: %d)", 
									fromSuperNode, toSuperNode, numSuperNodes-1)
							}
							
							// Add the edge to the super-graph
							// Undirected graph: need to divide weight by 2 because of double counting
							superGraph.AddEdge(fromSuperNode, toSuperNode, totalWeight / 2)
							edgeCount++
						}
						// END OF DETERMINISTICALLY SORTED EDGES

	// Create reverse mapping: super-node index → community ID
    superNodeToComm := make(map[int]int)
    for oldCommID, superNodeIdx := range commToNewIndex {
        superNodeToComm[superNodeIdx] = oldCommID
    }
	
	// =============================================================================
	// STEP 6: VALIDATE SUPER-GRAPH
	// =============================================================================
	// Ensure the super-graph is valid before returning
	
	if err := superGraph.Validate(); err != nil {
		return nil, nil, nil, fmt.Errorf("created invalid super-graph: %w", err)
	}
	
	return superGraph, communityMap, superNodeToComm, nil
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



// =================================================================================
// DEBUGGING AND PRINTING FUNCTIONS
// =================================================================================


// PrintState prints the state of the Louvain algorithm execution
// mode can be "ALL", "COMMUNITIES", or "GRAPH"
func (s *LouvainState) PrintState(label string, mode string) {
	fmt.Printf("\n" + strings.Repeat("=", 80) + "\n")
	fmt.Printf("STATE: %s (Mode: %s)\n", label, mode)
	fmt.Printf(strings.Repeat("=", 80) + "\n")
	
	if mode == "GRAPH" || mode == "ALL" {
		s.printGraph()
	}
	
	if mode == "COMMUNITIES" || mode == "ALL" {
		s.printCommunities()
	}
	
	fmt.Printf(strings.Repeat("=", 80) + "\n")
}

// printGraph prints the graph structure
func (s *LouvainState) printGraph() {
	fmt.Printf("GRAPH STRUCTURE:\n")
	fmt.Printf("  Nodes: %d\n", s.Graph.NumNodes)
	fmt.Printf("  Total weight: %.4f\n", s.Graph.TotalWeight)
	fmt.Printf("\n")
	
	for i := 0; i < s.Graph.NumNodes; i++ {
		neighbors := s.Graph.GetNeighbors(i)
		degree := s.Graph.GetNodeDegree(i)
		weight := s.Graph.Weights[i]
		
		fmt.Printf("  Node %d: degree=%.2f, weight=%.2f, neighbors=[", i, degree, weight)
		neighborList := []string{}
		for neighbor, edgeWeight := range neighbors {
			neighborList = append(neighborList, fmt.Sprintf("%d(%.2f)", neighbor, edgeWeight))
		}
		sort.Strings(neighborList)
		fmt.Printf("%s]\n", strings.Join(neighborList, ", "))
	}
	fmt.Printf("\n")
}

// printCommunities prints the community structure
func (s *LouvainState) printCommunities() {
	fmt.Printf("COMMUNITY STRUCTURE:\n")
	fmt.Printf("  Active communities: %d\n", len(s.C2N))
	fmt.Printf("  Current modularity: %.6f\n", s.GetModularity())
	fmt.Printf("\n")
	
	// Print N2C (node -> community)
	fmt.Printf("N2C (Node -> Community):\n")
	for i := 0; i < len(s.N2C); i++ {
		fmt.Printf("  Node %d -> Community %d\n", i, s.N2C[i])
	}
	fmt.Printf("\n")
	
	// Print C2N (community -> nodes)
	fmt.Printf("C2N (Community -> Nodes):\n")
	communityIDs := make([]int, 0, len(s.C2N))
	for commID := range s.C2N {
		communityIDs = append(communityIDs, commID)
	}
	sort.Ints(communityIDs)
	
	for _, commID := range communityIDs {
		nodes := s.C2N[commID]
		nodesCopy := make([]int, len(nodes))
		copy(nodesCopy, nodes)
		sort.Ints(nodesCopy)
		fmt.Printf("  Community %d -> Nodes %v (count: %d)\n", commID, nodesCopy, len(nodes))
	}
	fmt.Printf("\n")
	
	// Print In (internal weights)
	fmt.Printf("In (Internal Weights):\n")
	for _, commID := range communityIDs {
		fmt.Printf("  Community %d -> In: %.4f\n", commID, s.In[commID])
	}
	fmt.Printf("\n")
	
	// Print Tot (total weights)
	fmt.Printf("Tot (Total Weights):\n")
	totalSum := 0.0
	for _, commID := range communityIDs {
		tot := s.Tot[commID]
		fmt.Printf("  Community %d -> Tot: %.4f\n", commID, tot)
		totalSum += tot
	}
	fmt.Printf("  TOTAL Sum: %.4f\n", totalSum)
	fmt.Printf("\n")
	
	// Summary table
	fmt.Printf("COMMUNITY SUMMARY:\n")
	fmt.Printf("  %-10s %-10s %-10s %-10s %-10s\n", "Community", "Nodes", "In", "Tot", "Modularity")
	fmt.Printf("  %s\n", strings.Repeat("-", 55))
	
	for _, commID := range communityIDs {
		nodeCount := len(s.C2N[commID])
		in := s.In[commID]
		tot := s.Tot[commID]
		
		// Calculate individual community contribution to modularity
		m2 := 2 * s.Graph.TotalWeight
		commModularity := 0.0
		if m2 > 0 {
			commModularity = in/m2 - (tot/m2)*(tot/m2)
		}
		
		fmt.Printf("  %-10d %-10d %-10.4f %-10.4f %-10.6f\n", 
			commID, nodeCount, in, tot, commModularity)
	}
	fmt.Printf("\n")
}
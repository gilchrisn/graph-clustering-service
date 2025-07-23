
package louvain

import (
	"context"
	"fmt"
	// "math"
	"math/rand"
	"runtime"
	// "sort"
	"time"
	
	"github.com/rs/zerolog"
)

// Result represents the algorithm output
type Result struct {
	Levels           []LevelInfo          `json:"levels"`
	FinalCommunities map[int]int          `json:"final_communities"`
	Modularity       float64              `json:"modularity"`
	NumLevels        int                  `json:"num_levels"`
	Statistics       Statistics           `json:"statistics"`
}

// LevelInfo contains information about each hierarchical level
type LevelInfo struct {
	Level          int             `json:"level"`
	Communities    map[int][]int   `json:"communities"`
	Modularity     float64         `json:"modularity"`
	NumCommunities int             `json:"num_communities"`
	NumMoves       int             `json:"num_moves"`
	RuntimeMS      int64           `json:"runtime_ms"`
}

// Statistics contains algorithm performance metrics
type Statistics struct {
	TotalIterations int           `json:"total_iterations"`
	TotalMoves      int           `json:"total_moves"`
	RuntimeMS       int64         `json:"runtime_ms"`
	MemoryPeakMB    int64         `json:"memory_peak_mb"`
	LevelStats      []LevelStats  `json:"level_stats"`
}

// LevelStats contains per-level statistics
type LevelStats struct {
	Level             int     `json:"level"`
	Iterations        int     `json:"iterations"`
	Moves             int     `json:"moves"`
	InitialModularity float64 `json:"initial_modularity"`
	FinalModularity   float64 `json:"final_modularity"`
	RuntimeMS         int64   `json:"runtime_ms"`
}

// Community represents the state of communities (simple arrays, NetworkX style)
type Community struct {
	NodeToCommunity []int     // nodeToComm[i] = community ID of node i
	CommunityNodes  [][]int   // commNodes[c] = list of nodes in community c
	CommunityWeights []float64 // commWeights[c] = total weight of community c
	CommunityInternalWeights []float64 // commInternal[c] = internal weight of community c
	NumCommunities  int       // number of communities
}

// NewCommunity initializes each node in its own community
func NewCommunity(graph *Graph) *Community {
	n := graph.NumNodes
	comm := &Community{
		NodeToCommunity:          make([]int, n),
		CommunityNodes:           make([][]int, n),
		CommunityWeights:         make([]float64, n),
		CommunityInternalWeights: make([]float64, n),
		NumCommunities:          n,
	}
	
	// Initialize each node in its own community
	for i := 0; i < n; i++ {
		comm.NodeToCommunity[i] = i
		comm.CommunityNodes[i] = []int{i}
		comm.CommunityWeights[i] = graph.Degrees[i]
		comm.CommunityInternalWeights[i] = graph.GetEdgeWeight(i, i) * 2 // self-loops count double
	}
	
	return comm
}

// CalculateModularity computes Newman's modularity
func CalculateModularity(graph *Graph, comm *Community) float64 {
	if graph.TotalWeight == 0 {
		return 0.0
	}
	
	modularity := 0.0
	m2 := 2.0 * graph.TotalWeight
	
	for c := 0; c < comm.NumCommunities; c++ {
		if len(comm.CommunityNodes[c]) == 0 {
			continue
		}
		
		internal := comm.CommunityInternalWeights[c]
		total := comm.CommunityWeights[c]
		
		modularity += internal/m2 - (total/m2)*(total/m2)
	}
	
	return modularity
}

// CalculateModularityGain computes the modularity gain from moving a node
func CalculateModularityGain(graph *Graph, comm *Community, node, targetComm int, edgeWeight float64) float64 {
	nodeDegree := graph.Degrees[node]
	commTotal := comm.CommunityWeights[targetComm]
	m2 := 2.0 * graph.TotalWeight
	
	return edgeWeight/graph.TotalWeight - (nodeDegree*commTotal)/m2
}

// GetEdgeWeightToComm calculates total edge weight from node to community
func GetEdgeWeightToComm(graph *Graph, comm *Community, node, targetComm int) float64 {
	weight := 0.0
	neighbors, weights := graph.GetNeighbors(node)
	
	for i, neighbor := range neighbors {
		if comm.NodeToCommunity[neighbor] == targetComm {
			weight += weights[i]
		}
	}
	
	return weight
}

// MoveNode moves a node to a different community
func MoveNode(graph *Graph, comm *Community, node, oldComm, newComm int) {
	if oldComm == newComm {
		return
	}
	
	nodeDegree := graph.Degrees[node]
	
	// Remove from old community
	oldNodes := comm.CommunityNodes[oldComm]
	for i, n := range oldNodes {
		if n == node {
			comm.CommunityNodes[oldComm] = append(oldNodes[:i], oldNodes[i+1:]...)
			break
		}
	}
	
	// Update old community weights
	oldWeight := GetEdgeWeightToComm(graph, comm, node, oldComm)
	comm.CommunityWeights[oldComm] -= nodeDegree
	comm.CommunityInternalWeights[oldComm] -= 2 * oldWeight
	
	// Add to new community
	comm.CommunityNodes[newComm] = append(comm.CommunityNodes[newComm], node)
	comm.NodeToCommunity[node] = newComm
	
	// Update new community weights
	newWeight := GetEdgeWeightToComm(graph, comm, node, newComm)
	selfLoop := graph.GetEdgeWeight(node, node)
	comm.CommunityWeights[newComm] += nodeDegree
	comm.CommunityInternalWeights[newComm] += 2 * (newWeight + selfLoop)
}

// OneLevel performs one level of local optimization
func OneLevel(graph *Graph, comm *Community, config *Config, logger zerolog.Logger) (bool, int, error) {
	improvement := false
	totalMoves := 0
	rng := rand.New(rand.NewSource(config.RandomSeed()))
	
	// Create node processing order
	nodes := make([]int, graph.NumNodes)
	for i := 0; i < graph.NumNodes; i++ {
		nodes[i] = i
	}
	
	for iteration := 0; iteration < config.MaxIterations(); iteration++ {
		iterationMoves := 0
		
		// Shuffle nodes for better convergence
		rng.Shuffle(len(nodes), func(i, j int) { nodes[i], nodes[j] = nodes[j], nodes[i] })
		
		// Process each node
		for _, node := range nodes {
			oldComm := comm.NodeToCommunity[node]
			bestComm := oldComm
			bestGain := 0.0
			
			// Find neighbor communities
			neighborComms := make(map[int]float64)
			neighbors, weights := graph.GetNeighbors(node)
			
			for i, neighbor := range neighbors {
				nComm := comm.NodeToCommunity[neighbor]
				neighborComms[nComm] += weights[i]
			}
			
			// Include current community
			if _, exists := neighborComms[oldComm]; !exists {
				neighborComms[oldComm] = 0.0
			}
			
			// Find best community
			for targetComm, edgeWeight := range neighborComms {
				if len(comm.CommunityNodes[targetComm]) == 0 {
					continue
				}
				
				gain := CalculateModularityGain(graph, comm, node, targetComm, edgeWeight)
				if gain > bestGain {
					bestComm = targetComm
					bestGain = gain
				}
			}
			
			// Move node if beneficial
			if bestComm != oldComm && bestGain > config.MinModularityGain() {
				MoveNode(graph, comm, node, oldComm, bestComm)
				iterationMoves++
				improvement = true
			}
		}
		
		totalMoves += iterationMoves
		
		// Log progress
		if config.EnableProgress() && iteration%10 == 0 {
			logger.Info().
				Int("iteration", iteration+1).
				Int("moves", iterationMoves).
				Float64("modularity", CalculateModularity(graph, comm)).
				Msg("Local optimization progress")
		}
		
		// Early termination if no moves
		if iterationMoves == 0 {
			logger.Debug().Int("iteration", iteration+1).Msg("Converged: no moves")
			break
		}
	}
	
	return improvement, totalMoves, nil
}

// AggregateGraph creates a super-graph from communities
func AggregateGraph(graph *Graph, comm *Community, logger zerolog.Logger) (*Graph, [][]int, error) {
	// Find non-empty communities
	validComms := make([]int, 0)
	for c := 0; c < comm.NumCommunities; c++ {
		if len(comm.CommunityNodes[c]) > 0 {
			validComms = append(validComms, c)
		}
	}
	
	numSuperNodes := len(validComms)
	if numSuperNodes == 0 {
		return nil, nil, fmt.Errorf("no valid communities found")
	}
	
	// Create community mapping
	commToSuper := make(map[int]int)
	communityMapping := make([][]int, numSuperNodes)
	
	for i, commID := range validComms {
		commToSuper[commID] = i
		communityMapping[i] = make([]int, len(comm.CommunityNodes[commID]))
		copy(communityMapping[i], comm.CommunityNodes[commID])
	}
	
	// Create super-graph
	superGraph := NewGraph(numSuperNodes)
	
	// Calculate super-edge weights
	superEdges := make(map[[2]int]float64)
	
	for node := 0; node < graph.NumNodes; node++ {
		nodeComm := comm.NodeToCommunity[node]
		superI, exists := commToSuper[nodeComm]
		if !exists {
			continue
		}
		
		neighbors, weights := graph.GetNeighbors(node)
		for i, neighbor := range neighbors {
			neighborComm := comm.NodeToCommunity[neighbor]
			superJ, exists := commToSuper[neighborComm]
			if !exists {
				continue
			}
			
			// Create edge key (sorted for undirected graph)
			var edge [2]int
			if superI <= superJ {
				edge = [2]int{superI, superJ}
			} else {
				edge = [2]int{superJ, superI}
			}
			
			superEdges[edge] += weights[i]
		}
	}
	
	// Add edges to super-graph
	for edge, weight := range superEdges {
		if weight > 0 {
			superGraph.AddEdge(edge[0], edge[1], weight/2) // Divide by 2 for undirected
		}
	}
	
	logger.Info().
		Int("original_nodes", graph.NumNodes).
		Int("super_nodes", numSuperNodes).
		Float64("compression_ratio", float64(numSuperNodes)/float64(graph.NumNodes)).
		Msg("Graph aggregation completed")
	
	return superGraph, communityMapping, nil
}

// Run executes the complete Louvain algorithm
func Run(graph *Graph, config *Config, ctx context.Context) (*Result, error) {
	startTime := time.Now()
	logger := config.CreateLogger()
	
	logger.Info().
		Int("nodes", graph.NumNodes).
		Float64("total_weight", graph.TotalWeight).
		Msg("Starting Louvain algorithm")
	
	// Validate input graph
	if err := graph.Validate(); err != nil {
		return nil, fmt.Errorf("invalid graph: %w", err)
	}
	
	result := &Result{
		Levels:     make([]LevelInfo, 0),
		Statistics: Statistics{LevelStats: make([]LevelStats, 0)},
	}
	
	// Initialize community structure
	comm := NewCommunity(graph)
	currentGraph := graph.Clone()
	
	// Main hierarchical loop
	for level := 0; level < config.MaxLevels(); level++ {
		levelStart := time.Now()
		initialMod := CalculateModularity(currentGraph, comm)
		
		logger.Info().
			Int("level", level).
			Int("nodes", currentGraph.NumNodes).
			Float64("initial_modularity", initialMod).
			Msg("Starting level")
		
		// Phase 1: Local optimization
		improvement, moves, err := OneLevel(currentGraph, comm, config, logger)
		if err != nil {
			return nil, fmt.Errorf("local optimization failed at level %d: %w", level, err)
		}
		
		finalMod := CalculateModularity(currentGraph, comm)
		levelTime := time.Since(levelStart)
		
		// Record level information
		levelInfo := LevelInfo{
			Level:          level,
			Communities:    make(map[int][]int),
			Modularity:     finalMod,
			NumCommunities: 0,
			NumMoves:       moves,
			RuntimeMS:      levelTime.Milliseconds(),
		}
		
		// Build communities map
		commID := 0
		for c := 0; c < comm.NumCommunities; c++ {
			if len(comm.CommunityNodes[c]) > 0 {
				levelInfo.Communities[commID] = make([]int, len(comm.CommunityNodes[c]))
				copy(levelInfo.Communities[commID], comm.CommunityNodes[c])
				commID++
			}
		}
		levelInfo.NumCommunities = commID
		
		result.Levels = append(result.Levels, levelInfo)
		result.Statistics.TotalMoves += moves
		
		// Record level statistics
		levelStats := LevelStats{
			Level:             level,
			Moves:             moves,
			InitialModularity: initialMod,
			FinalModularity:   finalMod,
			RuntimeMS:         levelTime.Milliseconds(),
		}
		result.Statistics.LevelStats = append(result.Statistics.LevelStats, levelStats)
		
		// Check termination conditions
		if !improvement {
			logger.Info().Int("level", level).Msg("No improvement, stopping")
			break
		}
		
		if levelInfo.NumCommunities == 1 {
			logger.Info().Int("level", level).Msg("Single community remaining, stopping")
			break
		}
		
		// Phase 2: Create super-graph
		superGraph, communityMapping, err := AggregateGraph(currentGraph, comm, logger)
		if err != nil {
			return nil, fmt.Errorf("aggregation failed at level %d: %w", level, err)
		}
		
		// Check if compression occurred
		if superGraph.NumNodes >= currentGraph.NumNodes {
			logger.Info().Msg("No compression achieved, stopping")
			break
		}
		
		// Prepare for next level
		currentGraph = superGraph
		comm = NewCommunity(currentGraph)
		
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	
	// Finalize results
	result.NumLevels = len(result.Levels)
	result.Modularity = CalculateModularity(currentGraph, comm)
	result.Statistics.RuntimeMS = time.Since(startTime).Milliseconds()
	result.Statistics.MemoryPeakMB = getMemoryUsage()
	
	// Build final community assignments
	result.FinalCommunities = make(map[int]int)
	for i := 0; i < currentGraph.NumNodes; i++ {
		result.FinalCommunities[i] = comm.NodeToCommunity[i]
	}
	
	logger.Info().
		Int("levels", result.NumLevels).
		Float64("final_modularity", result.Modularity).
		Int64("runtime_ms", result.Statistics.RuntimeMS).
		Msg("Louvain algorithm completed")
	
	return result, nil
}

// getMemoryUsage returns current memory usage in MB
func getMemoryUsage() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc / 1024 / 1024)
}

package scar

import (
	"context"
	"fmt"
	"strings"
	"sort"
	// "math"
	"math/rand"
	"runtime"
	"time"

	"github.com/rs/zerolog"

	"github.com/gilchrisn/graph-clustering-service/pkg/utils"
)

// Result represents the algorithm output
type Result struct {
	Levels           []LevelInfo          `json:"levels"`
	FinalCommunities map[int]int          `json:"final_communities"`
	Modularity       float64              `json:"modularity"`
	NumLevels        int                  `json:"num_levels"`
	Statistics       Statistics           `json:"statistics"`

    NodeMapping      *NodeMapping         `json:"node_mapping,omitempty"`
}

// LevelInfo contains information about each hierarchical level 
type LevelInfo struct {
	Level          int             `json:"level"`
	Communities    map[int][]int   `json:"communities"`
	Modularity     float64         `json:"modularity"`
	NumCommunities int             `json:"num_communities"`
	NumMoves       int             `json:"num_moves"`
	RuntimeMS      int64           `json:"runtime_ms"`

	CommunityToSuperNode map[int]int `json:"community_to_supernode,omitempty"` // community ID -> super-node ID at next level
	SuperNodeToCommunity map[int]int `json:"supernode_to_community,omitempty"` // super-node ID -> community ID from this level

	SketchGraph *SketchGraph `json:"-"`
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

// Community represents the state of communities (IDENTICAL to Louvain + sketch management)
type Community struct {
	NodeToCommunity []int     // nodeToComm[i] = community ID of node i
	CommunityNodes  [][]int   // commNodes[c] = list of nodes in community c
	CommunityWeights []float64 // commWeights[c] = total weight of community c
	CommunityInternalWeights []float64 // commInternal[c] = internal weight of community c
	NumCommunities  int       // number of communities
	
	// SCAR-specific: community sketches
	communitySketches map[int]*VertexBottomKSketch
}

// NewCommunity initializes each node in its own community (MODIFIED for sketches)
func NewCommunity(graph *SketchGraph) *Community {
	n := graph.NumNodes
	comm := &Community{
		NodeToCommunity:          make([]int, n),
		CommunityNodes:           make([][]int, n),
		CommunityWeights:         make([]float64, n),
		CommunityInternalWeights: make([]float64, n),
		NumCommunities:          n,
		communitySketches:        make(map[int]*VertexBottomKSketch),
	}
	// Initialize each node in its own community
	for i := 0; i < n; i++ {
		comm.NodeToCommunity[i] = i
		comm.CommunityNodes[i] = []int{i}
		comm.CommunityWeights[i] = graph.GetDegree(i)                    
		comm.CommunityInternalWeights[i] = graph.GetEdgeWeight(i, i) * 2 // self-loops count double
		
		comm.communitySketches[i] = graph.sketchManager.nodeToHashMap[int64(i)]
	}
	
	return comm
}

// CalculateModularity computes Newman's modularity 
func CalculateModularity(graph *SketchGraph, comm *Community) float64 {
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
func CalculateModularityGain(graph *SketchGraph, comm *Community, node, targetComm int, edgeWeight float64) float64 {
	nodeDegree := graph.GetDegree(node)                    
	commTotal := comm.CommunityWeights[targetComm]
	m2 := 2.0 * graph.TotalWeight


	// gain := edgeWeight - (nodeDegree*commTotal)/m2
	// if (true) {
	// 	fmt.Printf("Calculating modularity gain for node %d to community %d: edgeWeight=%.6f, nodeDegree=%.6f, commTotal=%.6f, m2=%.6f, gain=%.6f\n", node, targetComm, edgeWeight, nodeDegree, commTotal, m2, gain)

	// }

	return edgeWeight - (nodeDegree*commTotal)/m2
}

// GetEdgeWeightToComm calculates total edge weight from node to community (SKETCH-BASED)
func GetEdgeWeightToComm(graph *SketchGraph, comm *Community, node, targetComm int) float64 {
	return graph.EstimateEdgesToCommunity(node, targetComm, comm)
}

// MoveNode moves a node to a different community 
func MoveNode(graph *SketchGraph, comm *Community, node, oldComm, newComm int) {
	if oldComm == newComm {
		return
	}
	
	nodeDegree := graph.GetDegree(node)                   
	
	// Remove from old community
	oldNodes := comm.CommunityNodes[oldComm]
	for i, n := range oldNodes {
		if n == node {
			comm.CommunityNodes[oldComm] = append(oldNodes[:i], oldNodes[i+1:]...)
			break
		}
	}

	// Update old community sketch
	if len(comm.CommunityNodes[oldComm]) > 0 {
		graph.removeNodeHashFromCommunitySketch(node, oldComm, comm)
	} else {
		delete(comm.communitySketches, oldComm)
	}
	
	// Update old community weights
	comm.CommunityWeights[oldComm] -= nodeDegree	

	oldWeight := GetEdgeWeightToComm(graph, comm, node, oldComm)
	comm.CommunityInternalWeights[oldComm] -= 2 * oldWeight
	
	// Add to new community
	comm.CommunityNodes[newComm] = append(comm.CommunityNodes[newComm], node)
	comm.NodeToCommunity[node] = newComm
	
	// Update new community sketch
	graph.addNodeHashToCommunitySketch(node, newComm, comm)

	// Update new community weights
	comm.CommunityWeights[newComm] += nodeDegree

	newWeight := GetEdgeWeightToComm(graph, comm, node, newComm)
	selfLoop := graph.GetEdgeWeight(node, node)           
	comm.CommunityInternalWeights[newComm] += 2 * (newWeight + selfLoop)
}

// OneLevel performs one level of local optimization (IDENTICAL structure, sketch data)
func OneLevel(graph *SketchGraph, comm *Community, config *Config, logger zerolog.Logger, moveTracker *utils.MoveTracker) (bool, int, error) {
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
			bestGain := -100.0 // Start with zero

			// Find neighbor communities (SKETCH-BASED)
			neighborComms := graph.FindNeighboringCommunities(node, comm)
			
			
			// Find best community
			for targetComm, edgeWeight := range neighborComms {

				if len(comm.CommunityNodes[targetComm]) == 0 {
					continue
				}

				gain := CalculateModularityGain(graph, comm, node, targetComm, edgeWeight)
				if gain > bestGain || (gain == bestGain && targetComm < bestComm) { // tiebreaking by community ID
					bestComm = targetComm
					bestGain = gain
				}
			}
			
			// Move node if beneficial
			if bestComm != oldComm && bestGain > config.MinModularityGain() {
				MoveNode(graph, comm, node, oldComm, bestComm)
				iterationMoves++
				improvement = true

                if moveTracker != nil {
                    moveTracker.LogMove(totalMoves + iterationMoves, node, oldComm, bestComm, bestGain, 
                                      CalculateModularity(graph, comm))
                }
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

// AggregateGraph creates a super-graph from communities (MODIFIED for sketches)
func AggregateGraph(graph *SketchGraph, comm *Community, logger zerolog.Logger) (*SketchGraph, [][]int, map[int]int, map[int]int, error) {
	// Find non-empty communities
	validComms := make([]int, 0)
	for c := 0; c < comm.NumCommunities; c++ {
		if len(comm.CommunityNodes[c]) > 0 {
			validComms = append(validComms, c)
		}
	}
	
	numSuperNodes := len(validComms)
	if numSuperNodes == 0 {
		return nil, nil, nil, nil, fmt.Errorf("no valid communities found")
	}
	
	// Create community mapping AND hierarchy tracking
	commToSuper := make(map[int]int)      // community ID -> super-node ID (for aggregation)
	superToComm := make(map[int]int)      // super-node ID -> community ID (for hierarchy)
	communityMapping := make([][]int, numSuperNodes)
	
	for i, commID := range validComms {
		commToSuper[commID] = i           // Reindex: community 15 -> super-node 1
		superToComm[i] = commID           // Reverse: super-node 1 -> community 15
		communityMapping[i] = make([]int, len(comm.CommunityNodes[commID]))
		copy(communityMapping[i], comm.CommunityNodes[commID])
	}

	// Create super-graph with sketch aggregation
	superGraph := NewSketchGraph(numSuperNodes)
	
	// Aggregate sketches for super-nodes
	err := superGraph.AggregateFromPreviousLevel(graph, commToSuper, comm)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("sketch aggregation failed: %w", err)
	}
	
	logger.Info().
		Int("original_nodes", graph.NumNodes).
		Int("super_nodes", numSuperNodes).
		Float64("compression_ratio", float64(numSuperNodes)/float64(graph.NumNodes)).
		Msg("Graph aggregation completed")
	
	return superGraph, communityMapping, superToComm, commToSuper, nil
}

// Run executes the complete SCAR algorithm (IDENTICAL structure + sketch preprocessing)
func Run(graphFile, propertiesFile, pathFile string, config *Config, ctx context.Context) (*Result, error) {
	startTime := time.Now()
	logger := config.CreateLogger()
	
    var moveTracker *utils.MoveTracker
    if config.TrackMoves() {
        moveTracker = utils.NewMoveTracker(config.OutputFile(), "scar")
        defer moveTracker.Close()
    }

    storeGraphs := config.StoreGraphsAtEachLevel()

    logger.Info().
        Str("graph_file", graphFile).
        Str("properties_file", propertiesFile).
        Str("path_file", pathFile).
        Bool("store_graphs", storeGraphs).  // Log the setting, random seed
		Int("max_iterations", config.MaxIterations()).
        Msg("Starting SCAR algorithm")

	logger.Info().
		Int("max_levels", config.MaxLevels()).
		Float64("min_modularity_gain", config.MinModularityGain()).
		Int64("random_seed", config.RandomSeed()).
		Msg("Configuration parameters")
	
	// Sketch preprocessing
    graph, nodeMapping, err := BuildSketchGraph(graphFile, propertiesFile, pathFile, config, logger)
    if err != nil {
        return nil, fmt.Errorf("sketch preprocessing failed: %w", err)
    }

	logger.Info().
		Int("nodes", graph.NumNodes).
		Float64("total_weight", graph.TotalWeight).
		Msg("Sketch preprocessing completed, starting Louvain")
	
	// Validate input graph
	if err := graph.Validate(); err != nil {
		return nil, fmt.Errorf("invalid graph: %w", err)
	}
	
	result := &Result{
		Levels:     make([]LevelInfo, 0),
		Statistics: Statistics{LevelStats: make([]LevelStats, 0)},
	}
	
	// Initialize community structure and mapping tracking 
	comm := NewCommunity(graph)
	currentGraph := graph
	
	// Track mapping from current level nodes back to original nodes
	nodeToOriginal := make([][]int, graph.NumNodes)
	for i := 0; i < graph.NumNodes; i++ {
		nodeToOriginal[i] = []int{i} // Initially, each node maps to itself
	}
	
	// Main hierarchical loop 
	for level := 0; level < config.MaxLevels(); level++ {
		levelStart := time.Now()
		initialMod := CalculateModularity(currentGraph, comm)
		
		logger.Info().
			Int("level", level).
			Int("nodes", currentGraph.NumNodes).
			Float64("initial_modularity", initialMod).
			Msg("Starting level")
		
		// currentGraph.PrintDebug()
		// Phase 1: Local optimization
		improvement, moves, err := OneLevel(currentGraph, comm, config, logger, moveTracker)
		if err != nil {
			return nil, fmt.Errorf("local optimization failed at level %d: %w", level, err)
		}
		
		finalMod := CalculateModularity(currentGraph, comm)
		levelTime := time.Since(levelStart)
		
		// Record level information with ORIGINAL node IDs 
		levelInfo := LevelInfo{
			Level:          level,
			Communities:    make(map[int][]int),
			Modularity:     finalMod,
			NumCommunities: 0,
			NumMoves:       moves,
			RuntimeMS:      levelTime.Milliseconds(),

			CommunityToSuperNode: make(map[int]int),
			SuperNodeToCommunity: make(map[int]int),
		}
		
        // We store a reference since sketch state is immutable at this point
        if storeGraphs {
            levelInfo.SketchGraph = currentGraph
            logger.Debug().
                Int("level", level).
                Int("stored_nodes", levelInfo.SketchGraph.NumNodes).
                Float64("stored_weight", levelInfo.SketchGraph.TotalWeight).
                Msg("Stored level sketch graph")
        }

		// Build communities map using CURRENT LEVEL node IDs
		for c := 0; c < comm.NumCommunities; c++ {
			if len(comm.CommunityNodes[c]) > 0 {
				levelInfo.Communities[c] = make([]int, len(comm.CommunityNodes[c]))
				copy(levelInfo.Communities[c], comm.CommunityNodes[c])
			}
		}
		
		levelInfo.NumCommunities = len(levelInfo.Communities)
		
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
		superGraph, communityMapping, superToComm, commToSuper, err := AggregateGraph(currentGraph, comm, logger)
		if err != nil {
			return nil, fmt.Errorf("aggregation failed at level %d: %w", level, err)
		}
		
		// Populate hierarchy tracking in the level info 
		currentLevelIndex := len(result.Levels) - 1  // Get the level we just added
		
		// Build community-to-supernode mapping (only for communities that have nodes)
		for commID := range result.Levels[currentLevelIndex].Communities {
			if superNodeID, exists := commToSuper[commID]; exists { 
				result.Levels[currentLevelIndex].CommunityToSuperNode[commID] = superNodeID
			}
		}
		
		// Build supernode-to-community mapping (reverse of above)
		for superNodeID, commID := range superToComm {
			if _, exists := result.Levels[currentLevelIndex].Communities[commID]; exists {
				result.Levels[currentLevelIndex].SuperNodeToCommunity[superNodeID] = commID
			}
		}

		// Check if compression occurred
		if superGraph.NumNodes >= currentGraph.NumNodes {
			logger.Info().Msg("No compression achieved, stopping")
			break
		}
		
		// Update node-to-original mapping for next level 
		newNodeToOriginal := make([][]int, superGraph.NumNodes)
		for superNodeID, originalNodesList := range communityMapping {
			newNodeToOriginal[superNodeID] = make([]int, 0)
			for _, currentLevelNode := range originalNodesList {
				newNodeToOriginal[superNodeID] = append(newNodeToOriginal[superNodeID], nodeToOriginal[currentLevelNode]...)
			}
		}
		nodeToOriginal = newNodeToOriginal
		
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
	
	// Build final community assignments using ORIGINAL node IDs
    result.FinalCommunities = make(map[int]int)
    for compressedNode := 0; compressedNode < currentGraph.NumNodes; compressedNode++ {
        finalCommID := comm.NodeToCommunity[compressedNode]
        originalNode := nodeMapping.CompressedToOriginal[compressedNode]
        result.FinalCommunities[originalNode] = finalCommID
    }
    result.NodeMapping = nodeMapping
	
    if storeGraphs {
        logger.Info().
            Int("levels_stored", len(result.Levels)).
            Msg("Sketch graph storage completed")
    }
	
	logger.Info().
		Int("levels", result.NumLevels).
		Float64("final_modularity", result.Modularity).
		Int64("runtime_ms", result.Statistics.RuntimeMS).
		Msg("SCAR algorithm completed")
	
	return result, nil
}

// getMemoryUsage returns current memory usage in MB 
func getMemoryUsage() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc / 1024 / 1024)
}

// GetHierarchyPath returns the hierarchy path for an original node
// Returns slice where [0] is the original node, [1] is super-node at level 1, etc.
func (r *Result) GetHierarchyPath(originalNodeID int) []int {
	path := []int{originalNodeID}
	currentNodeID := originalNodeID
	
	for level := 0; level < len(r.Levels)-1; level++ { // -1 because last level has no next level
		// Find which community this node belongs to at this level
		var communityID int = -1
		
		// Look through all communities at this level
		for commID, nodes := range r.Levels[level].Communities {
			for _, nodeID := range nodes {
				if nodeID == currentNodeID {
					communityID = commID
					break
				}
			}
			if communityID != -1 {
				break
			}
		}
		
		if communityID != -1 {
			// Find the super-node ID for this community at the next level
			if superNodeID, exists := r.Levels[level].CommunityToSuperNode[communityID]; exists {
				path = append(path, superNodeID)
				currentNodeID = superNodeID // Track this for next level
			} else {
				break // No mapping found, hierarchy ends here
			}
		} else {
			break // Node not found at this level
		}
	}
	
	return path
}

// GetCommunityHierarchy returns which community a node belongs to at each level
func (r *Result) GetCommunityHierarchy(originalNodeID int) []int {
	communities := make([]int, 0)
	currentNodeID := originalNodeID
	
	for level := 0; level < len(r.Levels); level++ {
		// Find which community this node belongs to at this level
		var communityID int = -1
		
		for commID, nodes := range r.Levels[level].Communities {
			for _, nodeID := range nodes {
				if nodeID == currentNodeID {
					communityID = commID
					break
				}
			}
			if communityID != -1 {
				break
			}
		}
		
		if communityID != -1 {
			communities = append(communities, communityID)
			
			// For next level, find what super-node this community became
			if level < len(r.Levels)-1 {
				if superNodeID, exists := r.Levels[level].CommunityToSuperNode[communityID]; exists {
					currentNodeID = superNodeID
				} else {
					break
				}
			}
		} else {
			break
		}
	}
	
	return communities
}

// GetAllHierarchyPaths returns hierarchy paths for all original nodes
func (r *Result) GetAllHierarchyPaths() map[int][]int {
	paths := make(map[int][]int)
	
	// Get all original nodes from the final communities
	if len(r.Levels) > 0 {
		for _, nodes := range r.Levels[0].Communities {
			for _, nodeID := range nodes {
				paths[nodeID] = r.GetHierarchyPath(nodeID)
			}
		}
	}
	
	return paths
}

// DEBUGGING FUNCTIONS


// HasCommunitySketch checks if a community has a sketch
func (c *Community) HasCommunitySketch(commID int) bool {
	return c.communitySketches != nil && c.communitySketches[commID] != nil
}

// GetCommunitySketch returns the sketch for a specific community (can be nil)
func (c *Community) GetCommunitySketch(commID int) *VertexBottomKSketch {
	if c.communitySketches == nil {
		return nil
	}
	return c.communitySketches[commID]
}

// GetCommunitySketchCount returns the total number of communities with sketches
func (c *Community) GetCommunitySketchCount() int {
	if c.communitySketches == nil {
		return 0
	}
	return len(c.communitySketches)
}

// GetAllCommunitySketchIDs returns a slice of all community IDs that have sketches
func (c *Community) GetAllCommunitySketchIDs() []int {
	if c.communitySketches == nil {
		return nil
	}
	
	ids := make([]int, 0, len(c.communitySketches))
	for commID := range c.communitySketches {
		ids = append(ids, commID)
	}
	return ids
}



// PrintFullResult prints the complete Result structure in a nicely formatted way for debugging
func (r *Result) PrintFullResult() {
	fmt.Println("=" + strings.Repeat("=", 78) + "=")
	fmt.Println("                           SCAR ALGORITHM RESULT")
	fmt.Println("=" + strings.Repeat("=", 78) + "=")
	
	// High-level summary
	r.printSummary()
	
	// Detailed level information
	r.printLevels()
	
	// Final communities
	r.printFinalCommunities()
	
	// Statistics
	r.printStatistics()
	
	// Node mapping
	r.printNodeMapping()
	
	fmt.Println("=" + strings.Repeat("=", 78) + "=")
	fmt.Println("                            END OF RESULT")
	fmt.Println("=" + strings.Repeat("=", 78) + "=")
}

// printSummary prints high-level summary information
func (r *Result) printSummary() {
	fmt.Println("\n📊 SUMMARY")
	fmt.Println(strings.Repeat("-", 50))
	fmt.Printf("  Number of Levels:     %d\n", r.NumLevels)
	fmt.Printf("  Final Modularity:     %.6f\n", r.Modularity)
	fmt.Printf("  Total Runtime:        %d ms (%.2f seconds)\n", r.Statistics.RuntimeMS, float64(r.Statistics.RuntimeMS)/1000.0)
	fmt.Printf("  Total Moves:          %d\n", r.Statistics.TotalMoves)
	fmt.Printf("  Total Iterations:     %d\n", r.Statistics.TotalIterations)
	fmt.Printf("  Peak Memory Usage:    %d MB\n", r.Statistics.MemoryPeakMB)
	fmt.Printf("  Final Communities:    %d\n", len(r.FinalCommunities))
	
	if r.NodeMapping != nil {
		fmt.Printf("  Target Nodes:         %d\n", r.NodeMapping.NumTargetNodes)
		fmt.Printf("  Original Nodes:       %d\n", len(r.NodeMapping.CompressedToOriginal))
	}
}

// printLevels prints detailed information for each level
func (r *Result) printLevels() {
	fmt.Println("\n🏗️  HIERARCHICAL LEVELS")
	fmt.Println(strings.Repeat("-", 50))
	
	if len(r.Levels) == 0 {
		fmt.Println("  No levels found!")
		return
	}
	
	for i, level := range r.Levels {
		fmt.Printf("\n  📍 LEVEL %d\n", level.Level)
		fmt.Printf("    ├─ Communities:       %d\n", level.NumCommunities)
		fmt.Printf("    ├─ Modularity:        %.6f\n", level.Modularity)
		fmt.Printf("    ├─ Moves:             %d\n", level.NumMoves)
		fmt.Printf("    └─ Runtime:           %d ms\n", level.RuntimeMS)
		
		// Print communities with their nodes
		if len(level.Communities) > 0 {
			fmt.Printf("    \n    🏘️  Communities Detail:\n")
			
			// Sort community IDs for consistent output
			commIDs := make([]int, 0, len(level.Communities))
			for commID := range level.Communities {
				commIDs = append(commIDs, commID)
			}
			sort.Ints(commIDs)
			
			for _, commID := range commIDs {
				nodes := level.Communities[commID]
				fmt.Printf("      Community %d (%d nodes): ", commID, len(nodes))
				
				// Print first 10 nodes, then summarize if more
				if len(nodes) <= 10 {
					fmt.Printf("%v\n", nodes)
				} else {
					fmt.Printf("%v ... (+%d more)\n", nodes[:10], len(nodes)-10)
				}
			}
		}
		
		// Print hierarchy mappings if available
		if len(level.CommunityToSuperNode) > 0 {
			fmt.Printf("    \n    🔗 Community → Super-node Mapping:\n")
			commIDs := make([]int, 0, len(level.CommunityToSuperNode))
			for commID := range level.CommunityToSuperNode {
				commIDs = append(commIDs, commID)
			}
			sort.Ints(commIDs)
			
			for _, commID := range commIDs {
				superNodeID := level.CommunityToSuperNode[commID]
				fmt.Printf("      Community %d → Super-node %d\n", commID, superNodeID)
			}
		}
		
		if len(level.SuperNodeToCommunity) > 0 {
			fmt.Printf("    \n    🔗 Super-node → Community Mapping:\n")
			superNodeIDs := make([]int, 0, len(level.SuperNodeToCommunity))
			for superNodeID := range level.SuperNodeToCommunity {
				superNodeIDs = append(superNodeIDs, superNodeID)
			}
			sort.Ints(superNodeIDs)
			
			for _, superNodeID := range superNodeIDs {
				commID := level.SuperNodeToCommunity[superNodeID]
				fmt.Printf("      Super-node %d → Community %d\n", superNodeID, commID)
			}
		}
		
		// Print sketch graph info if available
		if level.SketchGraph != nil {
			fmt.Printf("    \n    📈 Sketch Graph Info:\n")
			fmt.Printf("      ├─ Nodes:             %d\n", level.SketchGraph.NumNodes)
			fmt.Printf("      ├─ Total Weight:      %.2f\n", level.SketchGraph.TotalWeight)
			
			if level.SketchGraph.sketchManager != nil {
				fmt.Printf("      ├─ Sketch K:          %d\n", level.SketchGraph.sketchManager.k)
				fmt.Printf("      ├─ Sketch NK:         %d\n", level.SketchGraph.sketchManager.nk)
				fmt.Printf("      ├─ Vertex Sketches:   %d\n", len(level.SketchGraph.sketchManager.vertexSketches))
				fmt.Printf("      └─ Hash Mappings:     %d\n", len(level.SketchGraph.sketchManager.hashToNodeMap))
			}
		}
		
		// Add separator between levels
		if i < len(r.Levels)-1 {
			fmt.Println("    " + strings.Repeat("─", 40))
		}
	}
}

// printFinalCommunities prints the final community assignments
func (r *Result) printFinalCommunities() {
	fmt.Println("\n🎯 FINAL COMMUNITIES")
	fmt.Println(strings.Repeat("-", 50))
	
	if len(r.FinalCommunities) == 0 {
		fmt.Println("  No final communities found!")
		return
	}
	
	// Group nodes by community
	commToNodes := make(map[int][]int)
	for nodeID, commID := range r.FinalCommunities {
		commToNodes[commID] = append(commToNodes[commID], nodeID)
	}
	
	// Sort communities by ID
	commIDs := make([]int, 0, len(commToNodes))
	for commID := range commToNodes {
		commIDs = append(commIDs, commID)
	}
	sort.Ints(commIDs)
	
	fmt.Printf("  Total Communities: %d\n", len(commToNodes))
	fmt.Printf("  Total Nodes:       %d\n\n", len(r.FinalCommunities))
	
	for _, commID := range commIDs {
		nodes := commToNodes[commID]
		sort.Ints(nodes) // Sort nodes within community
		
		fmt.Printf("  Community %d (%d nodes): ", commID, len(nodes))
		if len(nodes) <= 15 {
			fmt.Printf("%v\n", nodes)
		} else {
			fmt.Printf("%v ... (+%d more)\n", nodes[:15], len(nodes)-15)
		}
	}
}

// printStatistics prints detailed statistics
func (r *Result) printStatistics() {
	fmt.Println("\n📈 STATISTICS")
	fmt.Println(strings.Repeat("-", 50))
	
	stats := &r.Statistics
	fmt.Printf("  Overall Performance:\n")
	fmt.Printf("    ├─ Total Runtime:     %d ms (%.2f seconds)\n", stats.RuntimeMS, float64(stats.RuntimeMS)/1000.0)
	fmt.Printf("    ├─ Total Iterations:  %d\n", stats.TotalIterations)
	fmt.Printf("    ├─ Total Moves:       %d\n", stats.TotalMoves)
	fmt.Printf("    └─ Peak Memory:       %d MB\n", stats.MemoryPeakMB)
	
	if len(stats.LevelStats) > 0 {
		fmt.Printf("\n  Per-Level Statistics:\n")
		fmt.Printf("    %-5s %-6s %-8s %-12s %-12s %-10s\n", "Level", "Iter", "Moves", "Init Mod", "Final Mod", "Time(ms)")
		fmt.Printf("    %s\n", strings.Repeat("─", 65))
		
		for _, levelStat := range stats.LevelStats {
			fmt.Printf("    %-5d %-6d %-8d %-12.6f %-12.6f %-10d\n",
				levelStat.Level,
				levelStat.Iterations,
				levelStat.Moves,
				levelStat.InitialModularity,
				levelStat.FinalModularity,
				levelStat.RuntimeMS)
		}
		
		// Calculate some derived statistics
		fmt.Printf("\n  Derived Statistics:\n")
		
		totalLevelTime := int64(0)
		maxModularityGain := 0.0
		for _, levelStat := range stats.LevelStats {
			totalLevelTime += levelStat.RuntimeMS
			gain := levelStat.FinalModularity - levelStat.InitialModularity
			if gain > maxModularityGain {
				maxModularityGain = gain
			}
		}
		
		fmt.Printf("    ├─ Avg time per level:    %.1f ms\n", float64(totalLevelTime)/float64(len(stats.LevelStats)))
		fmt.Printf("    ├─ Max modularity gain:   %.6f\n", maxModularityGain)
		fmt.Printf("    └─ Avg moves per level:   %.1f\n", float64(stats.TotalMoves)/float64(len(stats.LevelStats)))
	}
}

// printNodeMapping prints node mapping information
func (r *Result) printNodeMapping() {
	fmt.Println("\n🗂️  NODE MAPPING")
	fmt.Println(strings.Repeat("-", 50))
	
	if r.NodeMapping == nil {
		fmt.Println("  No node mapping available!")
		return
	}
	
	mapping := r.NodeMapping
	fmt.Printf("  Target Nodes:         %d\n", mapping.NumTargetNodes)
	fmt.Printf("  Mapped Nodes:         %d\n", len(mapping.OriginalToCompressed))
	fmt.Printf("  Compression Ratio:    %.2f%%\n", 
		float64(mapping.NumTargetNodes)/float64(len(mapping.OriginalToCompressed))*100.0)
	
	// Show sample mappings
	if len(mapping.OriginalToCompressed) > 0 {
		fmt.Printf("\n  Sample Original → Compressed Mappings:\n")
		
		// Get first 10 mappings in sorted order
		originalIDs := make([]int, 0, len(mapping.OriginalToCompressed))
		for originalID := range mapping.OriginalToCompressed {
			originalIDs = append(originalIDs, originalID)
		}
		sort.Ints(originalIDs)
		
		count := 0
		for _, originalID := range originalIDs {
			if count >= 10 {
				fmt.Printf("    ... (+%d more mappings)\n", len(originalIDs)-10)
				break
			}
			compressedID := mapping.OriginalToCompressed[originalID]
			fmt.Printf("    %d → %d\n", originalID, compressedID)
			count++
		}
	}
	
	// Show sample reverse mappings
	if len(mapping.CompressedToOriginal) > 0 {
		fmt.Printf("\n  Sample Compressed → Original Mappings:\n")
		
		maxShow := 10
		if len(mapping.CompressedToOriginal) < maxShow {
			maxShow = len(mapping.CompressedToOriginal)
		}
		
		for i := 0; i < maxShow; i++ {
			fmt.Printf("    %d → %d\n", i, mapping.CompressedToOriginal[i])
		}
		
		if len(mapping.CompressedToOriginal) > 10 {
			fmt.Printf("    ... (+%d more mappings)\n", len(mapping.CompressedToOriginal)-10)
		}
	}
}

// PrintResultSummary prints a condensed summary (useful for quick checks)
func (r *Result) PrintResultSummary() {
	fmt.Println("🔍 SCAR Result Summary:")
	fmt.Printf("   Levels: %d | Final Modularity: %.6f | Runtime: %dms | Communities: %d\n",
		r.NumLevels, r.Modularity, r.Statistics.RuntimeMS, len(r.FinalCommunities))
}

// PrintHierarchyPathsForNodes prints hierarchy paths for specific nodes (useful for debugging)
func (r *Result) PrintHierarchyPathsForNodes(nodeIDs []int) {
	fmt.Println("\n🛤️  HIERARCHY PATHS")
	fmt.Println(strings.Repeat("-", 50))
	
	for _, nodeID := range nodeIDs {
		path := r.GetHierarchyPath(nodeID)
		communities := r.GetCommunityHierarchy(nodeID)
		
		fmt.Printf("  Node %d:\n", nodeID)
		fmt.Printf("    Path:        %v\n", path)
		fmt.Printf("    Communities: %v\n", communities)
	}
}
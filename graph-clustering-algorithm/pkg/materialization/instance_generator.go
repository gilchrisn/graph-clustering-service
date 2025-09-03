package materialization

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// InstanceGenerator finds all instances of a meta path in a heterogeneous graph using BFS
type InstanceGenerator struct {
	graph    *models.HeterogeneousGraph
	metaPath *models.MetaPath
	config   TraversalConfig
	
	// Precomputed data for efficiency
	nodesByType map[string][]string              // node_type -> []node_ids
	edgesByFrom map[string][]models.Edge         // from_node -> []edges
	
	// Statistics tracking
	stats TraversalStats
	mu    sync.RWMutex
}

// PathState represents the current state during BFS traversal
type PathState struct {
	Nodes       []string          // Current path of nodes
	Edges       []string          // Current path of edge types
	Weight      float64           // Accumulated weight
	Step        int               // Current step in meta path (0 to len(edges))
	Visited     map[string]bool   // Visited nodes (for cycle detection)
}

// NewInstanceGenerator creates a new instance generator
func NewInstanceGenerator(graph *models.HeterogeneousGraph, metaPath *models.MetaPath, config TraversalConfig) *InstanceGenerator {
	ig := &InstanceGenerator{
		graph:    graph,
		metaPath: metaPath,
		config:   config,
		stats:    TraversalStats{WorkerUtilization: make(map[int]int)},
	}
	
	// Precompute indices for efficient traversal
	ig.buildIndices()
	
	return ig
}

// buildIndices creates lookup tables for efficient graph traversal
func (ig *InstanceGenerator) buildIndices() {
	ig.nodesByType = make(map[string][]string)
	ig.edgesByFrom = make(map[string][]models.Edge)
	
	// Index nodes by type
	for nodeID, node := range ig.graph.Nodes {
		ig.nodesByType[node.Type] = append(ig.nodesByType[node.Type], nodeID)
	}
	
	// Index edges by source node
	for _, edge := range ig.graph.Edges {
		ig.edgesByFrom[edge.From] = append(ig.edgesByFrom[edge.From], edge)
	}
	
	ig.stats.StartingNodes = len(ig.nodesByType[ig.metaPath.NodeSequence[0]])
}

// FindAllInstances finds all instances of the meta path using BFS
func (ig *InstanceGenerator) FindAllInstances(progressCb func(int, string)) ([]PathInstance, TraversalStats, error) {
	// Set up timeout context
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(ig.config.TimeoutSeconds)*time.Second)
	defer cancel()
	
	startTime := time.Now()
	var allInstances []PathInstance
	var mu sync.Mutex
	instanceCount := 0
	
	// Get starting nodes
	startNodeType := ig.metaPath.NodeSequence[0]
	startNodes := ig.nodesByType[startNodeType]
	
	if len(startNodes) == 0 {
		return nil, ig.stats, fmt.Errorf("no starting nodes of type '%s' found", startNodeType)
	}
	
	// Progress reporting
	processedNodes := 0
	reportProgress := func(message string) {
		if progressCb != nil && processedNodes%ig.config.Parallelism == 0 {
			progressCb(instanceCount, fmt.Sprintf("%s (processed %d/%d start nodes)", message, processedNodes, len(startNodes)))
		}
	}
	
	// Use workers for parallel processing
	numWorkers := ig.config.Parallelism
	if numWorkers <= 0 {
		numWorkers = 1
	}
	
	nodeChannel := make(chan string, len(startNodes))
	instanceChannel := make(chan []PathInstance, numWorkers)
	var wg sync.WaitGroup
	
	// Start workers
	for workerID := 0; workerID < numWorkers; workerID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			var workerInstances []PathInstance
			nodesProcessed := 0
			
			for {
				select {
				case <-ctx.Done():
					ig.mu.Lock()
					ig.stats.TimeoutOccurred = true
					ig.mu.Unlock()
					instanceChannel <- workerInstances
					return
				case startNode, ok := <-nodeChannel:
					if !ok {
						instanceChannel <- workerInstances
						return
					}
					
					// Find instances starting from this node
					instances := ig.findInstancesFromNode(ctx, startNode)
					workerInstances = append(workerInstances, instances...)
					nodesProcessed++
					
					// Update statistics
					ig.mu.Lock()
					ig.stats.WorkerUtilization[id] = nodesProcessed
					ig.mu.Unlock()
					
					// Check memory limits
					if len(allInstances) > ig.config.MaxInstances {
						select {
						case instanceChannel <- workerInstances:
						case <-ctx.Done():
						}
						return
					}
				}
			}
		}(workerID)
	}
	
	// Send start nodes to workers
	go func() {
		defer close(nodeChannel)
		for _, startNode := range startNodes {
			select {
			case nodeChannel <- startNode:
				processedNodes++
				reportProgress("Finding instances")
			case <-ctx.Done():
				return
			}
		}
	}()
	
	// Collect results from workers
	go func() {
		wg.Wait()
		close(instanceChannel)
	}()
	
	// Gather all instances
	for workerInstances := range instanceChannel {
		mu.Lock()
		allInstances = append(allInstances, workerInstances...)
		instanceCount = len(allInstances)
		mu.Unlock()
		
		reportProgress("Collecting results")
		
		// Check limits
		if len(allInstances) > ig.config.MaxInstances {
			cancel() // Stop all workers
			break
		}
	}
	
	// Update final statistics
	ig.stats.RuntimeMS = time.Since(startTime).Milliseconds()
	
	// Filter instances if needed
	if ig.config.MaxInstances > 0 && len(allInstances) > ig.config.MaxInstances {
		allInstances = allInstances[:ig.config.MaxInstances]
	}
	
	return allInstances, ig.stats, nil
}

// findInstancesFromNode finds all path instances starting from a specific node using BFS
func (ig *InstanceGenerator) findInstancesFromNode(ctx context.Context, startNode string) []PathInstance {

	var instances []PathInstance
	
	// Initialize BFS queue with starting state
	queue := []PathState{{
		Nodes:   []string{startNode},
		Edges:   []string{},
		Weight:  1.0,
		Step:    0,
		Visited: map[string]bool{startNode: true},
	}}
	
	for len(queue) > 0 {
		// Check for timeout
		select {
		case <-ctx.Done():
			return instances
		default:
		}
		
		// Dequeue current state
		current := queue[0]
		queue = queue[1:]
		
		ig.mu.Lock()
		ig.stats.PathsExplored++
		ig.mu.Unlock()
		
		// Check if we've completed the meta path
		if current.Step >= len(ig.metaPath.EdgeSequence) {
			// We have a complete path instance
			instance := PathInstance{
				Nodes:  make([]string, len(current.Nodes)),
				Edges:  make([]string, len(current.Edges)),
				Weight: current.Weight,
			}
			copy(instance.Nodes, current.Nodes)
			copy(instance.Edges, current.Edges)
			
			instances = append(instances, instance)
			continue
		}
		
		// Get current node and required next step
		currentNodeID := current.Nodes[len(current.Nodes)-1]
		requiredEdgeType := ig.metaPath.EdgeSequence[current.Step]
		requiredNextNodeType := ig.metaPath.NodeSequence[current.Step+1]
		
		// Find all valid next nodes
		for _, edge := range ig.edgesByFrom[currentNodeID] {
			// Check if edge type matches meta path requirement
			if edge.Type != requiredEdgeType {
				continue
			}
			
			// Check if target node type matches meta path requirement
			targetNode, exists := ig.graph.Nodes[edge.To]
			if !exists || targetNode.Type != requiredNextNodeType {
				continue
			}
			
			// Check for cycles if not allowed
			if !ig.config.AllowCycles && current.Visited[edge.To] {
				ig.mu.Lock()
				ig.stats.CyclesDetected++
				ig.mu.Unlock()
				continue
			}
			
			// Check path length limit
			if len(current.Nodes) >= ig.config.MaxPathLength {
				continue
			}
			
			// Create new state for this path extension
			newVisited := make(map[string]bool)
			if !ig.config.AllowCycles {
				for k, v := range current.Visited {
					newVisited[k] = v
				}
				newVisited[edge.To] = true
			}
			
			newState := PathState{
				Nodes:   append(current.Nodes, edge.To),
				Edges:   append(current.Edges, edge.Type),
				Weight:  current.Weight * edge.Weight, // Multiply weights along path
				Step:    current.Step + 1,
				Visited: newVisited,
			}
			
			queue = append(queue, newState)
			
			ig.mu.Lock()
			ig.stats.EdgesTraversed++
			ig.mu.Unlock()
		}
		
		ig.mu.Lock()
		ig.stats.NodesVisited++
		ig.mu.Unlock()
	}
	
	return instances
}

// EstimateInstanceCount provides a rough estimate of the number of path instances
func (ig *InstanceGenerator) EstimateInstanceCount() (int, error) {
	// Simple estimation based on node/edge counts and meta path length
	startNodeType := ig.metaPath.NodeSequence[0]
	startNodes := ig.nodesByType[startNodeType]
	
	if len(startNodes) == 0 {
		return 0, fmt.Errorf("no starting nodes of type '%s'", startNodeType)
	}
	
	// Estimate branching factor at each step
	totalEstimate := len(startNodes)
	
	for step := 0; step < len(ig.metaPath.EdgeSequence); step++ {
		edgeType := ig.metaPath.EdgeSequence[step]
		nextNodeType := ig.metaPath.NodeSequence[step+1]
		
		// Count edges of this type
		edgeCount := 0
		for _, edge := range ig.graph.Edges {
			if edge.Type == edgeType {
				if targetNode, exists := ig.graph.Nodes[edge.To]; exists && targetNode.Type == nextNodeType {
					edgeCount++
				}
			}
		}
		
		if edgeCount == 0 {
			return 0, nil // No possible paths
		}
		
		// Rough branching factor (edges per node)
		avgBranching := float64(edgeCount) / float64(len(ig.graph.Nodes))
		if avgBranching < 1 {
			avgBranching = 1
		}
		
		totalEstimate = int(float64(totalEstimate) * avgBranching)
		
		// Cap the estimate to prevent overflow
		if totalEstimate > 10000000 { // 10M max estimate
			return 10000000, nil
		}
	}
	
	return totalEstimate, nil
}

// GetTraversalStatistics returns current traversal statistics
func (ig *InstanceGenerator) GetTraversalStatistics() TraversalStats {
	ig.mu.RLock()
	defer ig.mu.RUnlock()
	
	// Create a copy to avoid race conditions
	stats := ig.stats
	stats.WorkerUtilization = make(map[int]int)
	for k, v := range ig.stats.WorkerUtilization {
		stats.WorkerUtilization[k] = v
	}
	
	return stats
}

// ValidateMetaPathTraversability checks if the meta path can be traversed in the graph
func (ig *InstanceGenerator) ValidateMetaPathTraversability() error {
	for step := 0; step < len(ig.metaPath.EdgeSequence); step++ {
		fromNodeType := ig.metaPath.NodeSequence[step]
		toNodeType := ig.metaPath.NodeSequence[step+1]
		edgeType := ig.metaPath.EdgeSequence[step]
		
		// Check if this transition exists
		transitionExists := false
		for _, edge := range ig.graph.Edges {
			if edge.Type == edgeType {
				fromNode, fromExists := ig.graph.Nodes[edge.From]
				toNode, toExists := ig.graph.Nodes[edge.To]
				
				if fromExists && toExists && 
					fromNode.Type == fromNodeType && 
					toNode.Type == toNodeType {
					transitionExists = true
					break
				}
			}
		}
		
		if !transitionExists {
			return fmt.Errorf("meta path step %d not traversable: %s -[%s]-> %s", 
				step, fromNodeType, edgeType, toNodeType)
		}
	}
	
	return nil
}

// GetConnectedComponents analyzes the connectivity of nodes relevant to the meta path
func (ig *InstanceGenerator) GetConnectedComponents() map[string][]string {
	components := make(map[string][]string)
	visited := make(map[string]bool)
	componentID := 0
	
	// Only consider nodes that are relevant to the meta path
	relevantNodes := make(map[string]bool)
	for _, nodeType := range ig.metaPath.NodeSequence {
		for _, nodeID := range ig.nodesByType[nodeType] {
			relevantNodes[nodeID] = true
		}
	}
	
	// Find connected components using DFS
	var dfs func(string, string)
	dfs = func(nodeID, compID string) {
		visited[nodeID] = true
		components[compID] = append(components[compID], nodeID)
		
		// Visit neighbors
		for _, edge := range ig.edgesByFrom[nodeID] {
			if relevantNodes[edge.To] && !visited[edge.To] {
				dfs(edge.To, compID)
			}
		}
		
		// Also check incoming edges (treat as undirected for connectivity)
		for _, edge := range ig.graph.Edges {
			if edge.To == nodeID && relevantNodes[edge.From] && !visited[edge.From] {
				dfs(edge.From, compID)
			}
		}
	}
	
	// Start DFS from each unvisited relevant node
	for nodeID := range relevantNodes {
		if !visited[nodeID] {
			compID := fmt.Sprintf("component_%d", componentID)
			dfs(nodeID, compID)
			componentID++
		}
	}
	
	return components
}
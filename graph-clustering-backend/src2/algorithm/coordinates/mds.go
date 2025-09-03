package coordinates

import (
	"fmt"
	"sort"
	// "math"

	"gonum.org/v1/gonum/graph"
	// "gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/mds"
)

// MDSResult contains 2D coordinates from MDS
type MDSResult struct {
	Coordinates map[int64]Position // gonum nodeID -> 2D position
	MinX, MaxX  float64           // Coordinate bounds for normalization
	MinY, MaxY  float64
}

// Position represents a 2D coordinate
type Position struct {
	X, Y float64
}

// MDSCalculator computes 2D coordinates using Multidimensional Scaling
type MDSCalculator struct {
	maxDistance float64 // Maximum distance for unreachable nodes
}

// NewMDSCalculator creates a new MDS calculator
func NewMDSCalculator() *MDSCalculator {
	return &MDSCalculator{
		maxDistance: 10.0, // Default max distance for unreachable nodes
	}
}

// WithMaxDistance sets the maximum distance for unreachable nodes
func (mdsc *MDSCalculator) WithMaxDistance(maxDist float64) *MDSCalculator {
	mdsc.maxDistance = maxDist
	return mdsc
}

// Calculate computes 2D coordinates using classical MDS (Torgerson scaling)
func (mdsc *MDSCalculator) Calculate(g graph.Graph, nodeMapping map[int64]int) (*MDSResult, error) {
	if g.Nodes().Len() == 0 {
		return nil, fmt.Errorf("graph has no nodes")
	}

	// Build node list in DETERMINISTIC order
	nodeList := make([]int64, 0, g.Nodes().Len())
	nodes := g.Nodes()
	for nodes.Next() {
		nodeList = append(nodeList, nodes.Node().ID())
	}
	
	// SORT for deterministic ordering
	sort.Slice(nodeList, func(i, j int) bool {
		return nodeList[i] < nodeList[j]
	})

	if len(nodeList) == 1 {
		// Special case: single node at origin
		return &MDSResult{
			Coordinates: map[int64]Position{
				nodeList[0]: {X: 0.0, Y: 0.0},
			},
			MinX: 0.0, MaxX: 0.0,
			MinY: 0.0, MaxY: 0.0,
		}, nil
	}

	// Compute distance matrix
	distMatrix, err := mdsc.computeDistanceMatrix(g, nodeList)
	if err != nil {
		return nil, fmt.Errorf("failed to compute distance matrix: %w", err)
	}

	// Apply classical MDS (Torgerson scaling)
	coordinates, err := mdsc.applyTorgersonScaling(distMatrix)
	if err != nil {
		return nil, fmt.Errorf("MDS computation failed: %w", err)
	}

	// Convert to result format
	result := &MDSResult{
		Coordinates: make(map[int64]Position),
	}

	var minX, maxX, minY, maxY float64
	first := true

	for i, nodeID := range nodeList {
		x := coordinates.At(i, 0)
		y := coordinates.At(i, 1)
		
		result.Coordinates[nodeID] = Position{X: x, Y: y}

		if first {
			minX, maxX = x, x
			minY, maxY = y, y
			first = false
		} else {
			if x < minX { minX = x }
			if x > maxX { maxX = x }
			if y < minY { minY = y }
			if y > maxY { maxY = y }
		}
	}

	result.MinX, result.MaxX = minX, maxX
	result.MinY, result.MaxY = minY, maxY

	return result, nil
}

// computeDistanceMatrix computes shortest path distances between all node pairs
func (mdsc *MDSCalculator) computeDistanceMatrix(g graph.Graph, nodeList []int64) (*mat.SymDense, error) {
	n := len(nodeList)
	distMatrix := mat.NewSymDense(n, nil)

	// Compute distances using BFS from each node
	for i, sourceNode := range nodeList {
		distances := mdsc.bfsDistances(g, sourceNode, nodeList)
		
		for j, targetNode := range nodeList {
			dist := distances[targetNode]
			if dist < 0 {
				// Unreachable nodes get max distance
				dist = mdsc.maxDistance
			}
			distMatrix.SetSym(i, j, dist)
		}
	}

	return distMatrix, nil
}

// bfsDistances computes shortest path distances from source to all other nodes
func (mdsc *MDSCalculator) bfsDistances(g graph.Graph, sourceNode int64, nodeList []int64) map[int64]float64 {
	distances := make(map[int64]float64)
	visited := make(map[int64]bool)
	queue := []int64{sourceNode}
	
	distances[sourceNode] = 0
	visited[sourceNode] = true

	// BFS traversal
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		// Get neighbors
		neighbors := g.From(current)
		for neighbors.Next() {
			neighbor := neighbors.Node().ID()
			if !visited[neighbor] {
				visited[neighbor] = true
				distances[neighbor] = distances[current] + 1
				queue = append(queue, neighbor)
			}
		}
	}

	// Mark unreachable nodes
	for _, nodeID := range nodeList {
		if _, exists := distances[nodeID]; !exists {
			distances[nodeID] = -1 // Unreachable
		}
	}

	return distances
}

// applyTorgersonScaling applies classical MDS to distance matrix
func (mdsc *MDSCalculator) applyTorgersonScaling(distMatrix *mat.SymDense) (*mat.Dense, error) {
	var coordinates mat.Dense
	var eigenvals []float64

	// Apply Torgerson scaling to get 2D coordinates using gonum's mds package
	k, err := mds.TorgersonScaling(&coordinates, eigenvals, distMatrix)
	if err != nil {
		return nil, fmt.Errorf("Torgerson scaling failed: %w", err)
	}

	if k == 0 {
		return nil, fmt.Errorf("no positive eigenvalues found in MDS")
	}

	// Ensure we have at least 2 dimensions
	rows, cols := coordinates.Dims()
	if cols < 2 {
		// Pad with zeros if we don't have 2 dimensions
		paddedCoords := mat.NewDense(rows, 2, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols && j < 2; j++ {
				paddedCoords.Set(i, j, coordinates.At(i, j))
			}
			// Fill remaining columns with zeros
			for j := cols; j < 2; j++ {
				paddedCoords.Set(i, j, 0.0)
			}
		}
		return paddedCoords, nil
	}

	// Return only first 2 dimensions
	coords2D := mat.NewDense(rows, 2, nil)
	for i := 0; i < rows; i++ {
		coords2D.Set(i, 0, coordinates.At(i, 0)) // X coordinate
		coords2D.Set(i, 1, coordinates.At(i, 1)) // Y coordinate
	}

	return coords2D, nil
}

// GetNormalizedPosition returns coordinates normalized to [0,1] range
func (result *MDSResult) GetNormalizedPosition(nodeID int64) Position {
	pos, exists := result.Coordinates[nodeID]
	if !exists {
		return Position{X: 0.5, Y: 0.5} // Center if not found
	}

	// Normalize to [0,1] range
	var x, y float64
	
	if result.MaxX != result.MinX {
		x = (pos.X - result.MinX) / (result.MaxX - result.MinX)
	} else {
		x = 0.5
	}
	
	if result.MaxY != result.MinY {
		y = (pos.Y - result.MinY) / (result.MaxY - result.MinY)
	} else {
		y = 0.5
	}

	return Position{X: x, Y: y}
}

// GetScaledPosition returns coordinates scaled to specified range
func (result *MDSResult) GetScaledPosition(nodeID int64, minVal, maxVal float64) Position {
	normalized := result.GetNormalizedPosition(nodeID)
	scale := maxVal - minVal
	
	return Position{
		X: minVal + normalized.X*scale,
		Y: minVal + normalized.Y*scale,
	}
}
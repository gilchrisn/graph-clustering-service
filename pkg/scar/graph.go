package scar

import (
	"bufio"
	"os"
	"sort"
	"strconv"
	"strings"
)

// SymmetricVertex represents a vertex in an undirected graph
type SymmetricVertex struct {
	neighbors []int64
	degree    int64
}

func (v *SymmetricVertex) GetOutDegree() int64 {
	return v.degree
}

func (v *SymmetricVertex) GetNeighbors() []int64 {
	return v.neighbors
}

// GraphStructure represents the main graph data structure
type GraphStructure struct {
	V []SymmetricVertex
	n int64 // number of vertices
	m int64 // number of edges
}

func (g *GraphStructure) NumVertices() int64 {
	return g.n
}

func (g *GraphStructure) NumEdges() int64 {
	return g.m
}

// GraphReader handles reading graph data from files
type GraphReader struct{}

func NewGraphReader() *GraphReader {
	return &GraphReader{}
}

func (gr *GraphReader) ReadFromFile(filename string) (*GraphStructure, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	edges := make(map[int64][]int64)
	maxNode := int64(0)
	edgeCount := int64(0)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) >= 2 {
			src, err1 := strconv.ParseInt(parts[0], 10, 64)
			dst, err2 := strconv.ParseInt(parts[1], 10, 64)
			if err1 == nil && err2 == nil {
				edges[src] = append(edges[src], dst)
				edges[dst] = append(edges[dst], src)
				if src > maxNode {
					maxNode = src
				}
				if dst > maxNode {
					maxNode = dst
				}
				edgeCount++
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	vertices := gr.buildVertices(edges, maxNode)
	
	return &GraphStructure{
		V: vertices,
		n: maxNode + 1,
		m: edgeCount,
	}, nil
}

func (gr *GraphReader) buildVertices(edges map[int64][]int64, maxNode int64) []SymmetricVertex {
	vertices := make([]SymmetricVertex, maxNode+1)
	
	for i := int64(0); i <= maxNode; i++ {
		neighbors := edges[i]
		if len(neighbors) > 0 {
			neighbors = gr.removeDuplicatesAndSort(neighbors)
		}
		vertices[i] = SymmetricVertex{
			neighbors: neighbors,
			degree:    int64(len(neighbors)),
		}
	}
	
	return vertices
}

func (gr *GraphReader) removeDuplicatesAndSort(neighbors []int64) []int64 {
	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i] < neighbors[j]
	})
	
	if len(neighbors) == 0 {
		return neighbors
	}
	
	unique := neighbors[:1]
	for j := 1; j < len(neighbors); j++ {
		if neighbors[j] != neighbors[j-1] {
			unique = append(unique, neighbors[j])
		}
	}
	return unique
}

// Legacy function for backward compatibility
func ReadGraphFromFile(filename string) *GraphStructure {
	reader := NewGraphReader()
	graph, err := reader.ReadFromFile(filename)
	if err != nil {
		panic(err) // Maintain original behavior
	}
	return graph
}
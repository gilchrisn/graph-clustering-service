package scar

import (
	"bufio"
	"fmt"
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
	lineNum := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lineNum++
		
		if line == "" || strings.HasPrefix(line, "#") {
			fmt.Printf("Line %d: %s (skipped)\n", lineNum, line)
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
			} else {
				fmt.Printf("Line %d: %s (parse error: %v, %v)\n", lineNum, line, err1, err2)
			}
		} else {
			fmt.Printf("Line %d: %s (insufficient fields)\n", lineNum, line)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	// fmt.Printf("Graph summary: %d nodes (0-%d), %d edges\n", maxNode+1, maxNode, edgeCount)
	// fmt.Println("Adjacency list:")
	// for i := int64(0); i <= maxNode; i++ {
	// 	fmt.Printf("Node %d: neighbors %v\n", i, edges[i])
	// }

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
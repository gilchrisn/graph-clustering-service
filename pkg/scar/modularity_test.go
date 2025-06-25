package scar

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

type ModularityGraph struct {
	adjList           map[int][]int
	nodeCommunityMap  map[int]int
	communityNodeMap  map[int]map[int]bool
}

func NewModularityGraph() *ModularityGraph {
	return &ModularityGraph{
		adjList:          make(map[int][]int),
		nodeCommunityMap: make(map[int]int),
		communityNodeMap: make(map[int]map[int]bool),
	}
}

func (g *ModularityGraph) AddEdge(src, dest int) {
	g.adjList[src] = append(g.adjList[src], dest)
	g.adjList[dest] = append(g.adjList[dest], src)
}

func (g *ModularityGraph) AddNodeToCommunity(node, community int) {
	g.nodeCommunityMap[node] = community
	if g.communityNodeMap[community] == nil {
		g.communityNodeMap[community] = make(map[int]bool)
	}
	g.communityNodeMap[community][node] = true
}

func (g *ModularityGraph) CalculateModularity() float64 {
	M := 0.0 // Total number of edges
	Q := 0.0 // Modularity

	// Calculate total number of edges
	for _, neighbors := range g.adjList {
		M += float64(len(neighbors))
	}
	fmt.Printf("M: %f\n", M)

	// Iterate through all communities
	for community, communityNodes := range g.communityNodeMap {
		kij := 1           // Number of edges from node i to nodes in the same community
		degreeSum := 0     // Sum of degrees in the community
		degreeSumSq := 0.0 // Sum of degree products

		for node := range communityNodes {
			count := 0
			degreeSum += len(g.adjList[node]) // Degree of node i
			// Check neighbors of the current node
			for _, neighbor := range g.adjList[node] {
				if g.nodeCommunityMap[neighbor] == community {
					degreeSumSq += float64(len(g.adjList[node]) * len(g.adjList[neighbor]))
					kij++
					count++
				}
			}
		}

		deltaQ := float64(kij)/M - (float64(degreeSum)*float64(degreeSum))/(M*M)
		Q += deltaQ
	}

	return Q
}

type GraphBuilder struct{}

func (gb *GraphBuilder) ConstructGraph(nodeFile, edgeFile string) *ModularityGraph {
	graph := NewModularityGraph()
	nodeSet := make(map[int]bool)

	// Read node file
	nodeFileHandle, err := os.Open(nodeFile)
	if err != nil {
		log.Fatalf("Error: Unable to open node file: %v", err)
	}
	defer nodeFileHandle.Close()

	scanner := bufio.NewScanner(nodeFileHandle)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			node, err1 := strconv.Atoi(parts[0])
			community, err2 := strconv.Atoi(parts[1])
			if err1 == nil && err2 == nil {
				graph.AddNodeToCommunity(node, community)
				nodeSet[node] = true
			}
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading node file: %v", err)
	}

	// Read edge file and construct graph
	edgeFileHandle, err := os.Open(edgeFile)
	if err != nil {
		log.Fatalf("Error: Unable to open edge file: %v", err)
	}
	defer edgeFileHandle.Close()

	scanner = bufio.NewScanner(edgeFileHandle)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			src, err1 := strconv.Atoi(parts[0])
			dest, err2 := strconv.Atoi(parts[1])
			if err1 == nil && err2 == nil {
				if nodeSet[src] && nodeSet[dest] {
					graph.AddEdge(src, dest)
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading edge file: %v", err)
	}

	return graph
}

// CalculateModularity calculates modularity for given community assignments
func CalculateModularity(nodeFile, edgeFile string) float64 {
	builder := &GraphBuilder{}
	graph := builder.ConstructGraph(nodeFile, edgeFile)
	modularity := graph.CalculateModularity()
	fmt.Printf("Modularity: %f\n", modularity)
	return modularity
}
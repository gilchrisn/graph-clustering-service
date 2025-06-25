package scar

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// ModularityCalculator handles modularity calculations
type ModularityCalculator struct{}

func NewModularityCalculator() *ModularityCalculator {
	return &ModularityCalculator{}
}

func (mc *ModularityCalculator) Calculate(
	communitySketches map[int64][]uint32,
	nodesInCommunities map[int64][]int64,
	degreeSketches []uint32,
	community []int64,
	hashToNodeMap map[uint32]int64,
	sketches []uint32,
	k int64,
	wholeWeight float64,
) float64 {
	sumQ := 0.0
	
	for i := int64(0); i < int64(len(communitySketches)); i++ {
		degreeSum, kij := mc.calculateCommunityMetrics(
			i, nodesInCommunities, degreeSketches, community,
			hashToNodeMap, sketches, k,
		)
		
		deltaSumQ := kij - (degreeSum*degreeSum)/(2*wholeWeight)
		deltaSumQ /= (2 * wholeWeight)
		sumQ += deltaSumQ
	}
	
	return sumQ
}

func (mc *ModularityCalculator) calculateCommunityMetrics(
	currentCommunity int64,
	nodesInCommunities map[int64][]int64,
	degreeSketches []uint32,
	community []int64,
	hashToNodeMap map[uint32]int64,
	sketches []uint32,
	k int64,
) (float64, float64) {
	degreeSum := 0.0
	kij := 0.0
	
	if nodes, exists := nodesInCommunities[currentCommunity]; exists {
		for _, node := range nodes {
			degreeSum += float64(degreeSketches[node])
			
			for j := int64(0); j < k; j++ {
				sketchIdx := j + node*k
				if int(sketchIdx) < len(sketches) {
					if neighborNode, exists := hashToNodeMap[sketches[sketchIdx]]; exists {
						if community[neighborNode] == currentCommunity {
							kij += 1
						}
					}
				}
			}
			
			if degreeSketches[node] > uint32(k) {
				kij *= float64(degreeSketches[node]) / float64(k)
			}
		}
	}
	
	return degreeSum, kij
}

// ModularityGraph represents a graph for modularity testing
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
	M := g.calculateTotalEdges()
	Q := 0.0

	for community, communityNodes := range g.communityNodeMap {
		kij := 1
		degreeSum := 0
		
		for node := range communityNodes {
			degreeSum += len(g.adjList[node])
			for _, neighbor := range g.adjList[node] {
				if g.nodeCommunityMap[neighbor] == community {
					kij++
				}
			}
		}

		deltaQ := float64(kij)/M - (float64(degreeSum)*float64(degreeSum))/(M*M)
		Q += deltaQ
	}

	return Q
}

func (g *ModularityGraph) calculateTotalEdges() float64 {
	M := 0.0
	for _, neighbors := range g.adjList {
		M += float64(len(neighbors))
	}
	return M
}

// GraphBuilder constructs graphs from node and edge files
type GraphBuilder struct{}

func NewGraphBuilder() *GraphBuilder {
	return &GraphBuilder{}
}

func (gb *GraphBuilder) ConstructGraph(nodeFile, edgeFile string) (*ModularityGraph, error) {
	graph := NewModularityGraph()
	
	nodeSet, err := gb.readNodes(nodeFile, graph)
	if err != nil {
		return nil, err
	}
	
	err = gb.readEdges(edgeFile, graph, nodeSet)
	if err != nil {
		return nil, err
	}
	
	return graph, nil
}

func (gb *GraphBuilder) readNodes(nodeFile string, graph *ModularityGraph) (map[int]bool, error) {
	nodeSet := make(map[int]bool)
	
	file, err := os.Open(nodeFile)
	if err != nil {
		return nil, fmt.Errorf("unable to open node file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
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

	return nodeSet, scanner.Err()
}

func (gb *GraphBuilder) readEdges(edgeFile string, graph *ModularityGraph, nodeSet map[int]bool) error {
	file, err := os.Open(edgeFile)
	if err != nil {
		return fmt.Errorf("unable to open edge file: %v", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
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

	return scanner.Err()
}

// Legacy functions for backward compatibility
func calculateModularity(communitySketches map[int64][]uint32, nodesInCommunities map[int64][]int64, degreeSketches []uint32, community []int64, hashToNodeMap map[uint32]int64, sketches []uint32, k int64, wholeWeight float64) {
	calculator := NewModularityCalculator()
	modularity := calculator.Calculate(communitySketches, nodesInCommunities, degreeSketches, community, hashToNodeMap, sketches, k, wholeWeight)
	fmt.Printf("Modularity: %f\n", modularity)
}

func CalculateModularity(nodeFile, edgeFile string) float64 {
	builder := NewGraphBuilder()
	graph, err := builder.ConstructGraph(nodeFile, edgeFile)
	if err != nil {
		fmt.Printf("Error building graph: %v\n", err)
		return 0.0
	}
	modularity := graph.CalculateModularity()
	fmt.Printf("Modularity: %f\n", modularity)
	return modularity
}
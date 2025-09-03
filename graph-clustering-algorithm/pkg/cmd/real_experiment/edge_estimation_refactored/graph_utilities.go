package main

import "math/rand"

// ============================================================================
// GRAPH GENERATION UTILITIES
// ============================================================================

func generateErdosRenyi(numNodes int, edgeProb float64, rng *rand.Rand) *Graph {
    graph := &Graph{
        NumNodes: numNodes,
        Edges:    make(map[int]map[int]bool),
    }
    
    // Initialize adjacency
    for i := 0; i < numNodes; i++ {
        graph.Edges[i] = make(map[int]bool)
    }
    
    // Generate edges
    for i := 0; i < numNodes; i++ {
        for j := i + 1; j < numNodes; j++ {
            if rng.Float64() < edgeProb {
                graph.Edges[i][j] = true
                graph.Edges[j][i] = true
            }
        }
    }
    
    // Generate random communities
    numComms := max(1, numNodes/20)
    communities := make([][]int, numComms)
    for i := 0; i < numNodes; i++ {
        commId := rng.Intn(numComms)
        communities[commId] = append(communities[commId], i)
    }
    graph.Communities = communities
    
    return graph
}

func generatePlantedPartition(numNodes, numCommunities int, pIntra, pInter float64, rng *rand.Rand) *Graph {
    graph := &Graph{
        NumNodes: numNodes,
        Edges:    make(map[int]map[int]bool),
    }
    
    // Initialize adjacency
    for i := 0; i < numNodes; i++ {
        graph.Edges[i] = make(map[int]bool)
    }
    
    // Assign nodes to communities
    nodesPerComm := numNodes / numCommunities
    communities := make([][]int, numCommunities)
    nodeToComm := make([]int, numNodes)
    
    for i := 0; i < numNodes; i++ {
        commId := i / nodesPerComm
        if commId >= numCommunities {
            commId = numCommunities - 1
        }
        communities[commId] = append(communities[commId], i)
        nodeToComm[i] = commId
    }
    
    // Generate edges based on community membership
    for i := 0; i < numNodes; i++ {
        for j := i + 1; j < numNodes; j++ {
            prob := pInter // Default inter-community
            if nodeToComm[i] == nodeToComm[j] {
                prob = pIntra // Same community
            }
            
            if rng.Float64() < prob {
                graph.Edges[i][j] = true
                graph.Edges[j][i] = true
            }
        }
    }
    
    graph.Communities = communities
    return graph
}

// ============================================================================
// GROUND TRUTH CALCULATION
// ============================================================================

func calculateTrueWeight(graph *Graph, commA, commB []int) float64 {
    commBLookup := make(map[int]bool)
    for _, nodeB := range commB {
        commBLookup[nodeB] = true
    }
    
    weight := 0.0
    for _, nodeA := range commA {
        for neighbor := range graph.Edges[nodeA] {
            if commBLookup[neighbor] {
                weight += 1.0
            }
        }
    }
    
    return weight
}

func calculateEdgeMultiplicity(graph *Graph, commA, commB []int) float64 {
    if len(commA) == 0 || len(commB) == 0 {
        return 0.0
    }
    
    commBLookup := make(map[int]bool)
    for _, nodeB := range commB {
        commBLookup[nodeB] = true
    }
    
    uniqueConnections := make(map[int]bool)
    totalEdges := 0
    
    for _, nodeA := range commA {
        for neighbor := range graph.Edges[nodeA] {
            if commBLookup[neighbor] {
                uniqueConnections[neighbor] = true
                totalEdges++
            }
        }
    }
    
    if len(uniqueConnections) == 0 {
        return 0.0
    }
    
    return float64(totalEdges) / float64(len(uniqueConnections))
}

func calculateLocalDensity(graph *Graph, commA, commB []int) float64 {
    totalPossible := len(commA) * len(commB)
    if totalPossible == 0 {
        return 0.0
    }
    
    actualEdges := calculateTrueWeight(graph, commA, commB)
    return actualEdges / float64(totalPossible)
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
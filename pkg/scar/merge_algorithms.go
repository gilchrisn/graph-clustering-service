package scar

import (
	"math"
	"sort"
)

// MergeAlgorithm interface for different merging strategies
type MergeAlgorithm interface {
	FindMerges(
		communityEdgeTable map[int64]map[int64]bool,
		communitySketches map[int64][]uint32,
		community []int64,
		nodesInCommunities map[int64][]int64,
		communityDegreeSketches []uint32,
		wholeWeight float64,
	) [][]int64
}

// InitialMergeAlgorithm implements the initial best merge strategy
type InitialMergeAlgorithm struct {
	k  int64
	nk int64
}

func NewInitialMergeAlgorithm(k, nk int64) *InitialMergeAlgorithm {
	return &InitialMergeAlgorithm{k: k, nk: nk}
}

func (ima *InitialMergeAlgorithm) FindMerges(
	communityEdgeTable map[int64]map[int64]bool,
	communitySketches map[int64][]uint32,
	community []int64,
	nodesInCommunities map[int64][]int64,
	communityDegreeSketches []uint32,
	wholeWeight float64,
) [][]int64 {
	// Implementation of initial best merge using quality functions
	n2c := make(map[int64]int64)
	var nodeId []int64
	in := make([]int64, len(communityDegreeSketches))
	tot := make([]int64, len(communityDegreeSketches))
	nodeDegreeSketch := make(map[int64]int64)
	
	for i := range in {
		in[i] = 0
		tot[i] = 1
	}
	
	oldQuality := ima.qualityFunction(in, tot, wholeWeight)
	newQuality := oldQuality
	
	// Initialize node mappings
	for comm, nodes := range nodesInCommunities {
		for _, node := range nodes {
			n2c[node] = comm
			nodeId = append(nodeId, node)
			nodeDegreeSketch[node] = int64(communityDegreeSketches[comm])
		}
	}
	
	for i := 0; i < len(communityDegreeSketches); i++ {
		tot[i] = int64(communityDegreeSketches[i])
	}
	
	// Iterative optimization
	for iter := 0; iter < 20; iter++ {
		for i := 0; i < len(nodeId); i++ {
			node := nodeId[i]
			nodeCommunity := n2c[node]
			
			neighborCommunity := make(map[int64]bool)
			neighborCommunityWeight := make(map[int64]int64)
			neighborCommunityWeight[nodeCommunity] = 0
			
			ima.findNeighborCommunities(node, nodeCommunity, communityEdgeTable, nodesInCommunities, n2c, neighborCommunity, neighborCommunityWeight)
			
			ima.removeQuality(in, tot, n2c, nodeDegreeSketch[node], node, nodeCommunity, oldQuality)
			bestNeighborCommunity, bestWeight := ima.findBestMove(node, nodeCommunity, neighborCommunityWeight, in, tot, nodeDegreeSketch, wholeWeight)
			ima.insertQuality(in, tot, n2c, nodeDegreeSketch[node], node, bestNeighborCommunity, bestWeight)
		}
		
		newQuality = ima.qualityFunction(in, tot, wholeWeight)
		if !(newQuality > oldQuality && iter < 20) {
			break
		}
		oldQuality = newQuality
	}
	
	return ima.generateNewCommunities(n2c, community)
}

func (ima *InitialMergeAlgorithm) findNeighborCommunities(
	node, nodeCommunity int64,
	communityEdgeTable map[int64]map[int64]bool,
	nodesInCommunities map[int64][]int64,
	n2c map[int64]int64,
	neighborCommunity map[int64]bool,
	neighborCommunityWeight map[int64]int64,
) {
	if neighbors, exists := communityEdgeTable[nodeCommunity]; exists {
		for neighborCommunityId := range neighbors {
			if nodes, exists := nodesInCommunities[neighborCommunityId]; exists && len(nodes) > 0 {
				neighbor := nodes[0]
				if neighbor == node {
					continue
				}
				
				neighborComm := n2c[neighbor]
				neighborCommunity[neighborComm] = true
				neighborCommunityWeight[neighborComm]++
			}
		}
	}
}

func (ima *InitialMergeAlgorithm) findBestMove(
	node, nodeCommunity int64,
	neighborCommunityWeight map[int64]int64,
	in, tot []int64,
	nodeDegreeSketch map[int64]int64,
	wholeWeight float64,
) (int64, float64) {
	bestNeighborCommunity := nodeCommunity
	bestIncrease := 0.0
	bestWeight := 0.0
	
	for nei, weight := range neighborCommunityWeight {
		increase := ima.gainQuality(in, tot, node, nodeCommunity, nei, float64(weight), float64(nodeDegreeSketch[node]), wholeWeight)
		if increase > bestIncrease {
			bestIncrease = increase
			bestNeighborCommunity = nei
			bestWeight = float64(weight)
		}
	}
	
	return bestNeighborCommunity, bestWeight
}

func (ima *InitialMergeAlgorithm) qualityFunction(in []int64, tot []int64, wholeWeight float64) float64 {
	Q := 0.0
	for i := 0; i < len(in); i++ {
		deltaQ := float64(in[i]) - float64(tot[i])*float64(tot[i])/(2*wholeWeight)
		Q += deltaQ
	}
	Q /= (2 * wholeWeight)
	return Q
}

func (ima *InitialMergeAlgorithm) removeQuality(in []int64, tot []int64, n2c map[int64]int64, nodeDegree int64, node, nodeCommunity int64, value float64) {
	if nodeCommunity < 0 || nodeCommunity >= int64(len(in)) {
		return
	}
	in[nodeCommunity] -= int64(value * 2)
	tot[nodeCommunity] -= nodeDegree
	n2c[node] = -1
}

func (ima *InitialMergeAlgorithm) insertQuality(in []int64, tot []int64, n2c map[int64]int64, nodeDegree int64, node, nodeCommunity int64, value float64) {
	if nodeCommunity < 0 || nodeCommunity >= int64(len(in)) {
		return
	}
	in[nodeCommunity] += int64(value * 2)
	tot[nodeCommunity] += nodeDegree
	n2c[node] = nodeCommunity
}

func (ima *InitialMergeAlgorithm) gainQuality(in []int64, tot []int64, node, nodeCommunity, neighborCommunity int64, dnc, degc, wholeWeight float64) float64 {
	if neighborCommunity < 0 || neighborCommunity >= int64(len(in)) {
		return 0
	}
	totc := float64(tot[neighborCommunity])
	return dnc - totc*degc/wholeWeight/2
}

func (ima *InitialMergeAlgorithm) generateNewCommunities(n2c map[int64]int64, community []int64) [][]int64 {
	tempNewCommunities := make(map[int64][]int64)
	for node, communityId := range n2c {
		tempNewCommunities[communityId] = append(tempNewCommunities[communityId], community[node])
	}
	
	var newCommunities [][]int64
	for _, comms := range tempNewCommunities {
		newCommunities = append(newCommunities, comms)
	}
	return newCommunities
}

// QuickMergeAlgorithm implements the quick best merge strategy
type QuickMergeAlgorithm struct {
	k  int64
	nk int64
}

func NewQuickMergeAlgorithm(k, nk int64) *QuickMergeAlgorithm {
	return &QuickMergeAlgorithm{k: k, nk: nk}
}

func (qma *QuickMergeAlgorithm) FindMerges(
	communityEdgeTable map[int64]map[int64]bool,
	communitySketches map[int64][]uint32,
	community []int64,
	nodesInCommunities map[int64][]int64,
	communityDegreeSketches []uint32,
	wholeWeight float64,
) [][]int64 {
	communityToNewIndex := make(map[int64]int64)
	maxIndex := int64(0)
	var newCommunities [][]int64
	
	for currentCommunity, neighbors := range communityEdgeTable {
		bestNeighbor := int64(-1)
		bestDegreeSketch := math.MaxFloat64
		
		// Find the neighbor with the lowest degree sketch
		for neighborCommunity := range neighbors {
			if int(neighborCommunity) < len(communityDegreeSketches) {
				degreeSketch := float64(communityDegreeSketches[neighborCommunity])
				if degreeSketch < bestDegreeSketch {
					bestDegreeSketch = degreeSketch
					bestNeighbor = neighborCommunity
				}
			}
		}
		
		if bestNeighbor != -1 {
			qma.processMerge(currentCommunity, bestNeighbor, communityToNewIndex, &newCommunities, &maxIndex)
		}
	}
	
	return newCommunities
}

func (qma *QuickMergeAlgorithm) processMerge(
	currentCommunity, bestNeighbor int64,
	communityToNewIndex map[int64]int64,
	newCommunities *[][]int64,
	maxIndex *int64,
) {
	it1, exists1 := communityToNewIndex[currentCommunity]
	it2, exists2 := communityToNewIndex[bestNeighbor]
	
	if exists1 && exists2 {
		// Both communities already have an index, do nothing
	} else if exists1 {
		(*newCommunities)[it1] = append((*newCommunities)[it1], bestNeighbor)
		communityToNewIndex[bestNeighbor] = it1
	} else if exists2 {
		(*newCommunities)[it2] = append((*newCommunities)[it2], currentCommunity)
		communityToNewIndex[currentCommunity] = it2
	} else {
		*newCommunities = append(*newCommunities, []int64{currentCommunity, bestNeighbor})
		communityToNewIndex[currentCommunity] = *maxIndex
		communityToNewIndex[bestNeighbor] = *maxIndex
		(*maxIndex)++
	}
}

// AdvancedMergeAlgorithm implements the advanced merge strategy with threshold
type AdvancedMergeAlgorithm struct {
	k         int64
	nk        int64
	threshold float64
}

func NewAdvancedMergeAlgorithm(k, nk int64, threshold float64) *AdvancedMergeAlgorithm {
	return &AdvancedMergeAlgorithm{k: k, nk: nk, threshold: threshold}
}

func (ama *AdvancedMergeAlgorithm) FindMerges(
	communityEdgeTable map[int64]map[int64]bool,
	communitySketches map[int64][]uint32,
	community []int64,
	nodesInCommunities map[int64][]int64,
	communityDegreeSketches []uint32,
	wholeWeight float64,
) [][]int64 {
	communityToNewIndex := make(map[int64]int64)
	maxIndex := int64(0)
	var newCommunities [][]int64
	
	for currentCommunity, neighbors := range communityEdgeTable {
		bestNeighbor := int64(-1)
		bestEValue := ama.threshold
		
		c1Size := int64(len(nodesInCommunities[currentCommunity]))
		
		for neighborCommunity := range neighbors {
			eValue := ama.calculateEValue(currentCommunity, neighborCommunity, c1Size, communitySketches, nodesInCommunities, communityDegreeSketches, wholeWeight)
			
			if eValue > bestEValue {
				bestEValue = eValue
				bestNeighbor = neighborCommunity
			}
		}
		
		if bestNeighbor != -1 {
			ama.processMerge(currentCommunity, bestNeighbor, communityToNewIndex, &newCommunities, &maxIndex)
		}
	}
	
	return newCommunities
}

func (ama *AdvancedMergeAlgorithm) calculateEValue(
	currentCommunity, neighborCommunity, c1Size int64,
	communitySketches map[int64][]uint32,
	nodesInCommunities map[int64][]int64,
	communityDegreeSketches []uint32,
	wholeWeight float64,
) float64 {
	C2 := 1.0
	intersectK := 1.0
	
	neighborSketch := make([]uint32, len(communitySketches[neighborCommunity]))
	copy(neighborSketch, communitySketches[neighborCommunity])
	
	if int64(len(neighborSketch)) < ama.k*ama.nk {
		C2 = float64(len(neighborSketch)) / float64(ama.nk)
	} else {
		maxVbksC2 := neighborSketch[len(neighborSketch)-1]
		if maxVbksC2 > 0 {
			C2 = float64(ama.k-1) * float64(ama.nk) * float64(math.MaxUint32) / float64(maxVbksC2)
		}
	}
	
	// Add current community's hash values to neighbor sketch
	if currentSketches, exists := communitySketches[currentCommunity]; exists {
		neighborSketch = append(neighborSketch, currentSketches...)
	}
	
	sort.Slice(neighborSketch, func(i, j int) bool {
		return neighborSketch[i] < neighborSketch[j]
	})
	
	if int64(len(neighborSketch)) < ama.nk*ama.k {
		count := ama.countIntersections(currentCommunity, neighborSketch, communitySketches)
		intersectK = float64(c1Size) + C2 - float64(count)/float64(ama.nk)
	} else {
		if int64(len(neighborSketch)) > ama.k*ama.nk {
			neighborSketch = neighborSketch[:ama.k*ama.nk]
		}
		if len(neighborSketch) > 0 {
			intersectK = float64(ama.k-1) * float64(ama.nk) * float64(math.MaxUint32) / float64(neighborSketch[len(neighborSketch)-1])
		}
	}
	
	n1 := float64(communityDegreeSketches[currentCommunity])
	n2 := float64(communityDegreeSketches[neighborCommunity])
	
	return ama.calculateEFunction(c1Size, C2, intersectK, n1, n2, wholeWeight)
}

func (ama *AdvancedMergeAlgorithm) countIntersections(
	currentCommunity int64,
	neighborSketch []uint32,
	communitySketches map[int64][]uint32,
) int64 {
	count := int64(0)
	if currentSketches, exists := communitySketches[currentCommunity]; exists {
		for _, sketchValue := range currentSketches {
			// Binary search for sketchValue in neighborSketch
			idx := sort.Search(len(neighborSketch), func(i int) bool {
				return neighborSketch[i] >= sketchValue
			})
			for idx < len(neighborSketch) && neighborSketch[idx] == sketchValue {
				count++
				idx++
			}
		}
	}
	return count
}

func (ama *AdvancedMergeAlgorithm) calculateEFunction(
	c1Size int64,
	C2, intersectK float64,
	n1, n2, wholeWeight float64,
) float64 {
	return float64(c1Size) + C2 - intersectK - (n1*n2)/(2*wholeWeight)
}

func (ama *AdvancedMergeAlgorithm) processMerge(
	currentCommunity, bestNeighbor int64,
	communityToNewIndex map[int64]int64,
	newCommunities *[][]int64,
	maxIndex *int64,
) {
	it1, exists1 := communityToNewIndex[currentCommunity]
	it2, exists2 := communityToNewIndex[bestNeighbor]
	
	if exists1 && exists2 {
		// Both communities already have an index, do nothing
	} else if exists1 {
		(*newCommunities)[it1] = append((*newCommunities)[it1], bestNeighbor)
		communityToNewIndex[bestNeighbor] = it1
	} else if exists2 {
		(*newCommunities)[it2] = append((*newCommunities)[it2], currentCommunity)
		communityToNewIndex[currentCommunity] = it2
	} else {
		*newCommunities = append(*newCommunities, []int64{currentCommunity, bestNeighbor})
		communityToNewIndex[currentCommunity] = *maxIndex
		communityToNewIndex[bestNeighbor] = *maxIndex
		(*maxIndex)++
	}
}

// EdgeTableMerger handles merging of community edge tables
type EdgeTableMerger struct{}

func NewEdgeTableMerger() *EdgeTableMerger {
	return &EdgeTableMerger{}
}

func (etm *EdgeTableMerger) MergeEdgeTables(
	oldEdgeTable map[int64]map[int64]bool,
	newCommunities [][]int64,
) map[int64]map[int64]bool {
	newEdgeTable := make(map[int64]map[int64]bool)
	
	// Create a mapping from old community IDs to new community IDs
	oldToNewMap := make(map[int64]int64)
	for newCommId := 0; newCommId < len(newCommunities); newCommId++ {
		commsToMerge := newCommunities[newCommId]
		for _, oldComm := range commsToMerge {
			oldToNewMap[oldComm] = int64(newCommId)
		}
	}
	
	// Iterate through each old community in the old edge table
	for oldComm, neighbors := range oldEdgeTable {
		if newCommId, exists := oldToNewMap[oldComm]; exists {
			for neighborComm := range neighbors {
				if newNeighborComm, exists := oldToNewMap[neighborComm]; exists {
					if newNeighborComm != newCommId {
						if newEdgeTable[newCommId] == nil {
							newEdgeTable[newCommId] = make(map[int64]bool)
						}
						newEdgeTable[newCommId][newNeighborComm] = true
					}
				}
			}
		}
	}
	
	return newEdgeTable
}

// Legacy functions for backward compatibility
func initialBestMerge(
	k int64,
	communityEdgeTable map[int64]map[int64]bool,
	communitySketches map[int64][]uint32,
	community []int64,
	bestE []float64,
	nodesInCommunities map[int64][]int64,
	newCommunities *[][]int64,
	communityDegreeSketches []uint32,
	wholeWeight float64,
) {
	algorithm := NewInitialMergeAlgorithm(k, 4)
	*newCommunities = algorithm.FindMerges(communityEdgeTable, communitySketches, community, nodesInCommunities, communityDegreeSketches, wholeWeight)
}

func quickBestMerge(
	k int64,
	communityEdgeTable map[int64]map[int64]bool,
	communitySketches map[int64][]uint32,
	community []int64,
	bestE []float64,
	nodesInCommunities map[int64][]int64,
	newCommunities *[][]int64,
	communityDegreeSketches []uint32,
	wholeWeight float64,
) {
	algorithm := NewQuickMergeAlgorithm(k, 4)
	*newCommunities = algorithm.FindMerges(communityEdgeTable, communitySketches, community, nodesInCommunities, communityDegreeSketches, wholeWeight)
}

func calculateBestMerge(
	k, nk int64,
	iter int,
	threshold float64,
	communityEdgeTable map[int64]map[int64]bool,
	communitySketches map[int64][]uint32,
	community []int64,
	bestE []float64,
	nodesInCommunities map[int64][]int64,
	newCommunities *[][]int64,
	communityDegreeSketches []uint32,
	nodeHashValue []uint32,
	wholeWeight float64,
) {
	adjustedThreshold := threshold
	for i := 1; i < iter; i++ {
		adjustedThreshold *= threshold
	}
	
	algorithm := NewAdvancedMergeAlgorithm(k, nk, adjustedThreshold)
	*newCommunities = algorithm.FindMerges(communityEdgeTable, communitySketches, community, nodesInCommunities, communityDegreeSketches, wholeWeight)
}

func mergeEdgeTable(
	oldEdgeTable map[int64]map[int64]bool,
	newCommunities [][]int64,
	newEdgeTable map[int64]map[int64]bool,
) {
	merger := NewEdgeTableMerger()
	merged := merger.MergeEdgeTables(oldEdgeTable, newCommunities)
	
	// Copy results to the output map
	for k, v := range merged {
		newEdgeTable[k] = v
	}
}

func mergeCommunities(
	k, nk int64,
	community []int64,
	communitySketches map[int64][]uint32,
	mergedSketches map[int64][]uint32,
	newCommunities [][]int64,
	nodesInCommunities map[int64][]int64,
) {
	detector := NewCommunityDetector(k, nk, 0.5)
	data := &CommunityData{
		Community:          community,
		CommunitySketches:  communitySketches,
		NodesInCommunities: nodesInCommunities,
	}
	
	detector.MergeCommunities(data, newCommunities)
	
	// Copy results to mergedSketches
	for k, v := range data.CommunitySketches {
		mergedSketches[k] = v
	}
}
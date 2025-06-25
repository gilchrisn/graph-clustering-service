package scar

import (
	"math"
	"sort"
)

// CommunityDetector handles community detection algorithms
type CommunityDetector struct {
	k         int64
	nk        int64
	threshold float64
}

func NewCommunityDetector(k, nk int64, threshold float64) *CommunityDetector {
	return &CommunityDetector{
		k:         k,
		nk:        nk,
		threshold: threshold,
	}
}

// CommunityData holds all community-related data structures
type CommunityData struct {
	Community                []int64
	CommunitySketches        map[int64][]uint32
	CommunityDegreeSketches  []uint32
	NodesInCommunity         []int64
	NodesInCommunities       map[int64][]int64
	CommunityEdgeTable       map[int64]map[int64]bool
	HashToNodeMap            map[uint32]int64
	DegreeSketches           []uint32
}

func NewCommunityData(n int64) *CommunityData {
	return &CommunityData{
		Community:          make([]int64, n),
		CommunitySketches:  make(map[int64][]uint32),
		HashToNodeMap:      make(map[uint32]int64),
		DegreeSketches:     make([]uint32, n),
		NodesInCommunities: make(map[int64][]int64),
		CommunityEdgeTable: make(map[int64]map[int64]bool),
	}
}

func (cd *CommunityDetector) InitializeCommunities(
	n, k, nk int64,
	sketches []uint32,
	nodeHashValue []uint32,
	data *CommunityData,
) {
	for i := range data.Community {
		data.Community[i] = -1
	}

	cd.calculateSketchesAndMappings(n, k, nk, sketches, nodeHashValue, data)
	cd.assignInitialCommunities(data)
	cd.findCommunityEdges(data)
	cd.initializeNodesInCommunities(n, data)
}

func (cd *CommunityDetector) calculateSketchesAndMappings(
	n, k, nk int64,
	sketches []uint32,
	nodeHashValue []uint32,
	data *CommunityData,
) {
	for i := int64(0); i < n; i++ {
		if nodeHashValue[i*nk] != 0 && sketches[i*k+1] != 0 {
			currentNodeIndex := int64(len(data.NodesInCommunity))
			
			// Map hash values to nodes
			for j := int64(0); j < nk; j++ {
				data.HashToNodeMap[nodeHashValue[i*nk+j]] = i
			}
			
			data.NodesInCommunity = append(data.NodesInCommunity, i)
			data.CommunityDegreeSketches = append(data.CommunityDegreeSketches, 0)
			data.DegreeSketches[i] = 0
			
			cd.processNodeSketches(i, currentNodeIndex, k, nk, n, sketches, data)
		}
	}
}

func (cd *CommunityDetector) processNodeSketches(
	nodeId, nodeIndex, k, nk, n int64,
	sketches []uint32,
	data *CommunityData,
) {
	for j := int64(0); j < nk; j++ {
		flag := false
		var currentSketch uint32 = 0
		
		for ki := int64(0); ki < k; ki++ {
			sketchVal := sketches[j*n*k+nodeId*k+ki]
			if sketchVal != 0 {
				currentSketch = sketchVal
				data.CommunitySketches[nodeIndex] = append(data.CommunitySketches[nodeIndex], currentSketch)
			} else {
				flag = true
				data.DegreeSketches[nodeId] += uint32(ki - 1)
				break
			}
		}
		
		if !flag && currentSketch != 0 {
			degree := uint32(float64(math.MaxUint32)/float64(currentSketch) * float64(k-1))
			data.DegreeSketches[nodeId] += degree
		}
	}
	
	data.DegreeSketches[nodeId] /= uint32(nk)
	data.CommunityDegreeSketches[nodeIndex] = data.DegreeSketches[nodeId]
}

func (cd *CommunityDetector) assignInitialCommunities(data *CommunityData) {
	for communityId, nodeId := range data.NodesInCommunity {
		data.Community[nodeId] = int64(communityId)
	}
}

func (cd *CommunityDetector) findCommunityEdges(data *CommunityData) {
	for _, mappedNodeId := range data.HashToNodeMap {
		currentCommunity := data.Community[mappedNodeId]
		
		if sketches, exists := data.CommunitySketches[currentCommunity]; exists {
			for _, sketchValue := range sketches {
				if neighborNodeId, exists := data.HashToNodeMap[sketchValue]; exists {
					neighborCommunity := data.Community[neighborNodeId]
					if currentCommunity != neighborCommunity {
						if data.CommunityEdgeTable[currentCommunity] == nil {
							data.CommunityEdgeTable[currentCommunity] = make(map[int64]bool)
						}
						data.CommunityEdgeTable[currentCommunity][neighborCommunity] = true
					}
				}
			}
		}
	}
}

func (cd *CommunityDetector) initializeNodesInCommunities(n int64, data *CommunityData) {
	for i := int64(0); i < n; i++ {
		if data.Community[i] != -1 {
			data.NodesInCommunities[data.Community[i]] = append(data.NodesInCommunities[data.Community[i]], i)
		}
	}
}

// Community merging algorithms
func (cd *CommunityDetector) InitialBestMerge(
	data *CommunityData,
	wholeWeight float64,
) [][]int64 {
	// Implementation of initial best merge algorithm
	// This is a simplified version - the full implementation would be quite long
	var newCommunities [][]int64
	
	// Basic greedy merging based on degree sketches
	communityToNewIndex := make(map[int64]int64)
	maxIndex := int64(0)
	
	for currentCommunity, neighbors := range data.CommunityEdgeTable {
		bestNeighbor := int64(-1)
		bestDegreeSketch := math.MaxFloat64
		
		for neighborCommunity := range neighbors {
			if int(neighborCommunity) < len(data.CommunityDegreeSketches) {
				degreeSketch := float64(data.CommunityDegreeSketches[neighborCommunity])
				if degreeSketch < bestDegreeSketch {
					bestDegreeSketch = degreeSketch
					bestNeighbor = neighborCommunity
				}
			}
		}
		
		if bestNeighbor != -1 {
			cd.processCommunityMerge(currentCommunity, bestNeighbor, communityToNewIndex, &newCommunities, &maxIndex)
		}
	}
	
	return newCommunities
}

func (cd *CommunityDetector) processCommunityMerge(
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
		// Current community has an index, add bestNeighbor to the same index
		(*newCommunities)[it1] = append((*newCommunities)[it1], bestNeighbor)
		communityToNewIndex[bestNeighbor] = it1
	} else if exists2 {
		// Best neighbor has an index, add currentCommunity to the same index
		(*newCommunities)[it2] = append((*newCommunities)[it2], currentCommunity)
		communityToNewIndex[currentCommunity] = it2
	} else {
		// Neither has an index, create a new entry
		*newCommunities = append(*newCommunities, []int64{currentCommunity, bestNeighbor})
		communityToNewIndex[currentCommunity] = *maxIndex
		communityToNewIndex[bestNeighbor] = *maxIndex
		(*maxIndex)++
	}
}

func (cd *CommunityDetector) MergeCommunities(
	data *CommunityData,
	newCommunities [][]int64,
) {
	newCommunitySketches := make(map[int64][]uint32)
	newNodesInCommunity := make(map[int64][]int64)
	
	newCommId := int64(0)
	for ; newCommId < int64(len(newCommunities)); newCommId++ {
		commsToMerge := newCommunities[newCommId]
		if len(commsToMerge) == 0 {
			continue
		}
		
		for _, oldComm := range commsToMerge {
			for _, node := range data.NodesInCommunities[oldComm] {
				data.Community[node] = newCommId
				newNodesInCommunity[newCommId] = append(newNodesInCommunity[newCommId], node)
			}
			newCommunitySketches[newCommId] = append(newCommunitySketches[newCommId], data.CommunitySketches[oldComm]...)
			delete(data.NodesInCommunities, oldComm)
		}
	}
	
	// Process communities that were not merged
	for comm, nodes := range data.NodesInCommunities {
		newCommunitySketches[newCommId] = data.CommunitySketches[comm]
		for _, node := range nodes {
			data.Community[node] = newCommId
			newNodesInCommunity[newCommId] = append(newNodesInCommunity[newCommId], node)
		}
		newCommId++
	}
	
	// Move the new nodes into the original NodesInCommunities
	for k, v := range newNodesInCommunity {
		data.NodesInCommunities[k] = v
	}
	for k := range data.NodesInCommunities {
		if _, exists := newNodesInCommunity[k]; !exists {
			delete(data.NodesInCommunities, k)
		}
	}
	
	// Sort and resize sketches
	for comm, sketches := range newCommunitySketches {
		sort.Slice(sketches, func(i, j int) bool {
			return sketches[i] < sketches[j]
		})
		if int64(len(sketches)) > cd.k*cd.nk {
			sketches = sketches[:cd.k*cd.nk]
		}
		data.CommunitySketches[comm] = sketches
	}
}

// Legacy functions for backward compatibility
func calculateNewSketchesAndMappings(
	n, k, nk int64,
	sketches []uint32,
	nodeHashValue []uint32,
	hashToNodeMap map[uint32]int64,
	communitySketches map[int64][]uint32,
	degreeSketches []uint32,
	communityDegreeSketches *[]uint32,
	nodesInCommunity *[]int64,
) {
	detector := NewCommunityDetector(k, nk, 0.5)
	data := NewCommunityData(n)
	data.HashToNodeMap = hashToNodeMap
	data.CommunitySketches = communitySketches
	data.DegreeSketches = degreeSketches
	data.CommunityDegreeSketches = *communityDegreeSketches
	data.NodesInCommunity = *nodesInCommunity
	
	detector.calculateSketchesAndMappings(n, k, nk, sketches, nodeHashValue, data)
	
	*communityDegreeSketches = data.CommunityDegreeSketches
	*nodesInCommunity = data.NodesInCommunity
}

func assignInitialCommunities(nodeCommunityMap []int64, community []int64) {
	communityId := int64(0)
	for _, nodeId := range nodeCommunityMap {
		community[nodeId] = communityId
		communityId++
	}
}

func findCommunityEdges(
	hashToNodeMap map[uint32]int64,
	communitySketches map[int64][]uint32,
	community []int64,
	k int64,
	communityEdgeTable map[int64]map[int64]bool,
) {
	detector := NewCommunityDetector(k, 4, 0.5)
	data := &CommunityData{
		Community:          community,
		CommunitySketches:  communitySketches,
		CommunityEdgeTable: communityEdgeTable,
		HashToNodeMap:      hashToNodeMap,
	}
	detector.findCommunityEdges(data)
}
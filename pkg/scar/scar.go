package scar

import (
	// "bufio"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// SCARConfig holds configuration for SCAR algorithm
type SCARConfig struct {
	GraphFile    string
	PropertyFile string
	PathFile     string
	OutputFile   string
	EdgesFile    string
	K            int64
	NK           int64
	Threshold    float64
}

// RunSCAR executes the SCAR community detection algorithm
func RunSCAR(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: RunSCAR([]string{\"<inFile>\", \"[options]\"})") 
		return
	}
	
	// Parse command line arguments
	var (
		iFile         = args[0]
		outputFile    = "output.txt"
		edgelistFile  = ""
		propertyFile  = ""
		pathFile      = ""
		k        int64 = 10
		nk       int64 = 4
		threshold     = 0.5
	)
	
	// Parse additional arguments
	for i := 1; i < len(args); i++ {
		arg := args[i]
		if strings.HasPrefix(arg, "-o=") {
			outputFile = strings.TrimPrefix(arg, "-o=")
		} else if strings.HasPrefix(arg, "-edges=") {
			edgelistFile = strings.TrimPrefix(arg, "-edges=")
		} else if strings.HasPrefix(arg, "-pro=") {
			propertyFile = strings.TrimPrefix(arg, "-pro=")
		} else if strings.HasPrefix(arg, "-path=") {
			pathFile = strings.TrimPrefix(arg, "-path=")
		} else if strings.HasPrefix(arg, "-k=") {
			if val, err := strconv.ParseInt(strings.TrimPrefix(arg, "-k="), 10, 64); err == nil {
				k = val
			}
		} else if strings.HasPrefix(arg, "-nk=") {
			if val, err := strconv.ParseInt(strings.TrimPrefix(arg, "-nk="), 10, 64); err == nil {
				nk = val
			}
		} else if strings.HasPrefix(arg, "-th=") {
			if val, err := strconv.ParseFloat(strings.TrimPrefix(arg, "-th="), 64); err == nil {
				threshold = val
			}
		}
	}
	
	fmt.Println("Start reading graph")
	
	// Read graph using the new structure
	G := ReadGraphFromFile(iFile)
	n := G.n
	pathLength := int64(10)
	
	startTime := time.Now()
	
	// Initialize arrays
	oldSketches := make([]uint32, pathLength*n*k*nk)
	for i := range oldSketches {
		oldSketches[i] = math.MaxUint32
	}
	
	nodeHashValue := make([]uint32, n*nk)
	vertexProperties := ReadProperties(propertyFile, n)
	path, actualPathLength := ReadPath(pathFile)
	pathLength = actualPathLength
	
	fmt.Println("Start get graph parameters")
	
	// Compute sketch
	ComputeSketchForGraph(G, oldSketches, path, pathLength, vertexProperties, nodeHashValue, k, nk)
	
	sketches := oldSketches[(pathLength-1)*n*k*nk:]
	
	// Add 1 to all sketches
	for i := range sketches {
		if sketches[i] != math.MaxUint32 {
			sketches[i]++
		}
	}
	
	// Create new hash value table and sketch array for nodes with non-zero hash values
	hashToNodeMap := make(map[uint32]int64)
	communitySketches := make(map[int64][]uint32)
	degreeSketches := make([]uint32, n)
	var communityDegreeSketches []uint32
	var nodesInCommunity []int64
	
	calculateNewSketchesAndMappings(n, k, nk, sketches, nodeHashValue, hashToNodeMap, communitySketches, degreeSketches, &communityDegreeSketches, &nodesInCommunity)
	
	// Reconstruct edges from sketches if needed
	if edgelistFile != "" {
		reconstructEdges(edgelistFile, nodesInCommunity, communitySketches, hashToNodeMap)
	}
	
	fmt.Println("Finish calculate sketches")
	
	// Assign initial communities
	community := make([]int64, n)
	for i := range community {
		community[i] = -1
	}
	assignInitialCommunities(nodesInCommunity, community)
	
	// Calculate whole weight
	wholeWeight := 0.0
	for _, degree := range communityDegreeSketches {
		wholeWeight += float64(degree)
	}
	wholeWeight /= 2.0
	fmt.Printf("Whole weight: %f\n", wholeWeight)
	
	fmt.Println("Finish calculate sketches and mappings")
	
	// Create a table to store community edge connections
	communityEdgeTable := make(map[int64]map[int64]bool)
	findCommunityEdges(hashToNodeMap, communitySketches, community, k, communityEdgeTable)
	
	// Initialize nodesInCommunities
	nodesInCommunities := make(map[int64][]int64)
	for i := int64(0); i < n; i++ {
		if community[i] != -1 {
			nodesInCommunities[community[i]] = append(nodesInCommunities[community[i]], i)
		}
	}
	
	fmt.Println("Finish Initialization of the community")
	
	// Iterate for community detection
	for iter := 0; iter < 20; iter++ {
		start := time.Now()
		fmt.Printf("Number of communities: %d\n", len(communitySketches))
		
		bestE := make([]float64, n)
		for i := range bestE {
			bestE[i] = 0.00001 * wholeWeight
		}
		
		var newCommunities [][]int64
		
		if iter < 1 {
			initialBestMerge(k, communityEdgeTable, communitySketches, community, bestE, nodesInCommunities, &newCommunities, communityDegreeSketches, wholeWeight)
		} else if iter < 2 {
			quickBestMerge(k, communityEdgeTable, communitySketches, community, bestE, nodesInCommunities, &newCommunities, communityDegreeSketches, wholeWeight)
		} else {
			calculateBestMerge(k, nk, iter, threshold, communityEdgeTable, communitySketches, community, bestE, nodesInCommunities, &newCommunities, communityDegreeSketches, nodeHashValue, wholeWeight)
		}
		
		if len(newCommunities) == 0 {
			break
		}
		fmt.Printf("New communities: %d\n", len(newCommunities))
		
		// Create new community hash values for the next iteration
		mergedSketches := make(map[int64][]uint32)
		mergeCommunities(k, nk, community, communitySketches, mergedSketches, newCommunities, nodesInCommunities)
		
		// Recalculate the community edge table after merging communities
		newCommunityEdgeTable := make(map[int64]map[int64]bool)
		mergeEdgeTable(communityEdgeTable, newCommunities, newCommunityEdgeTable)
		
		// Update community sketches and edge table after merging
		communitySketches = mergedSketches
		communityEdgeTable = newCommunityEdgeTable
		
		// Recalculate community degree sketches
		communityDegreeSketches = communityDegreeSketches[:0]
		for i := int64(0); i < int64(len(communitySketches)); i++ {
			currentSketch := uint32(0)
			count := uint32(0)
			for _, sketch := range communitySketches[i] {
				count++
				if sketch > currentSketch {
					currentSketch = sketch
				}
			}
			if currentSketch != 0 {
				if count >= uint32(k-1) {
					degree := uint32(float64(math.MaxUint32)/float64(currentSketch) * float64(k-1))
					communityDegreeSketches = append(communityDegreeSketches, degree)
				} else {
					communityDegreeSketches = append(communityDegreeSketches, count)
				}
			} else {
				communityDegreeSketches = append(communityDegreeSketches, 1)
			}
		}
		
		fmt.Printf("Merge time: %v\n", time.Since(start))
	}
	
	fmt.Printf("Community detection time: %v\n", time.Since(startTime))
	
	// Write output
	writeOutput(outputFile, community, n)
	
	// Calculate and print modularity
	calculateModularity(communitySketches, nodesInCommunities, degreeSketches, community, hashToNodeMap, sketches, k, wholeWeight)
}

// RunSCARWithConfig runs SCAR with configuration struct
func RunSCARWithConfig(config SCARConfig) error {
	args := []string{config.GraphFile}
	
	if config.PropertyFile != "" {
		args = append(args, "-pro="+config.PropertyFile)
	}
	if config.PathFile != "" {
		args = append(args, "-path="+config.PathFile)
	}
	if config.OutputFile != "" {
		args = append(args, "-o="+config.OutputFile)
	}
	if config.EdgesFile != "" {
		args = append(args, "-edges="+config.EdgesFile)
	}
	if config.K > 0 {
		args = append(args, fmt.Sprintf("-k=%d", config.K))
	}
	if config.NK > 0 {
		args = append(args, fmt.Sprintf("-nk=%d", config.NK))
	}
	if config.Threshold > 0 {
		args = append(args, fmt.Sprintf("-th=%f", config.Threshold))
	}
	
	RunSCAR(args)
	return nil
}

// Helper functions - keep only one copy of each

func reconstructEdges(edgelistFile string, nodesInCommunity []int64, communitySketches map[int64][]uint32, hashToNodeMap map[uint32]int64) {
	file, err := os.Create(edgelistFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	
	uniqueEdges := make(map[string]bool)
	
	// For each node that has a sketch
	for nodeIndex, currentNode := range nodesInCommunity {
		// Look at all sketch values for this node
		if sketches, exists := communitySketches[int64(nodeIndex)]; exists {
			for _, sketchValue := range sketches {
				// Check if this sketch value corresponds to another node
				if neighborNode, exists := hashToNodeMap[sketchValue]; exists {
					// Avoid self-loops and duplicate edges
					if currentNode != neighborNode {
						// Store edge in canonical form (smaller node first)
						src := currentNode
						dst := neighborNode
						if src > dst {
							src, dst = dst, src
						}
						edgeKey := fmt.Sprintf("%d_%d", src, dst)
						if !uniqueEdges[edgeKey] {
							uniqueEdges[edgeKey] = true
							fmt.Fprintf(file, "%d %d\n", src, dst)
						}
					}
				}
			}
		}
	}
	
	fmt.Printf("Reconstructed %d edges\n", len(uniqueEdges))
}

func writeOutput(outputFile string, community []int64, n int64) {
	file, err := os.Create(outputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	
	for i := int64(0); i < n; i++ {
		if community[i] == -1 {
			continue
		}
		fmt.Fprintf(file, "%d %d\n", i, community[i])
	}
}

func calculateModularity(communitySketches map[int64][]uint32, nodesInCommunities map[int64][]int64, degreeSketches []uint32, community []int64, hashToNodeMap map[uint32]int64, sketches []uint32, k int64, wholeWeight float64) {
	sumQ := 0.0
	
	for i := int64(0); i < int64(len(communitySketches)); i++ {
		currentCommunity := i
		
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
		
		deltaSumQ := kij - (degreeSum*degreeSum)/(2*wholeWeight)
		deltaSumQ /= (2 * wholeWeight)
		sumQ += deltaSumQ
	}
	
	fmt.Printf("Modularity: %f\n", sumQ)
}

// All the other algorithm functions go here (calculateNewSketchesAndMappings, mergeEdgeTable, etc.)

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
	for i := int64(0); i < n; i++ {
		if nodeHashValue[i*nk] != 0 {
			if sketches[i*k+1] == 0 {
				continue
			}
			currentNodeIndex := int64(len(*nodesInCommunity))
			for j := int64(0); j < nk; j++ {
				hashToNodeMap[nodeHashValue[i*nk+j]] = i
			}
			*nodesInCommunity = append(*nodesInCommunity, i)
			*communityDegreeSketches = append(*communityDegreeSketches, 0)
			degreeSketches[i] = 0
			
			for j := int64(0); j < nk; j++ {
				flag := false
				var currentSketch uint32 = 0
				for ki := int64(0); ki < k; ki++ {
					if sketches[j*n*k+i*k+ki] != 0 {
						currentSketch = sketches[j*n*k+i*k+ki]
						communitySketches[currentNodeIndex] = append(communitySketches[currentNodeIndex], currentSketch)
					} else {
						flag = true
						degreeSketches[i] += uint32(ki - 1)
						break
					}
				}
				
				if (!flag) && (currentSketch != 0) {
					degreeSketches[i] += uint32(float64(math.MaxUint32)/float64(currentSketch) * float64(k-1))
				}
			}
			degreeSketches[i] /= uint32(nk)
			(*communityDegreeSketches)[currentNodeIndex] = degreeSketches[i]
		}
	}
}

func mergeEdgeTable(
	oldEdgeTable map[int64]map[int64]bool,
	newCommunities [][]int64,
	newEdgeTable map[int64]map[int64]bool,
) {
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
		// Check if the old community has a corresponding new community
		if newCommId, exists := oldToNewMap[oldComm]; exists {
			// Iterate through the neighbors of the old community
			for neighborComm := range neighbors {
				// Check if the neighbor community also has a mapping
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
}

func findCommunityEdges(
	hashToNodeMap map[uint32]int64,
	communitySketches map[int64][]uint32,
	community []int64,
	k int64,
	communityEdgeTable map[int64]map[int64]bool,
) {
	for _, mappedNodeId := range hashToNodeMap {
		originalNodeId := mappedNodeId
		currentCommunity := community[originalNodeId]
		
		for _, sketchValue := range communitySketches[currentCommunity] {
			if neighborNodeId, exists := hashToNodeMap[sketchValue]; exists {
				neighborCommunity := community[neighborNodeId]
				if currentCommunity != neighborCommunity {
					if communityEdgeTable[currentCommunity] == nil {
						communityEdgeTable[currentCommunity] = make(map[int64]bool)
					}
					communityEdgeTable[currentCommunity][neighborCommunity] = true
				}
			}
		}
	}
}

func calculateEFunction(
	c1Size int64,
	C2, IntersectK float64,
	k int64,
	n1, n2, wholeWeight float64,
) float64 {
	return float64(c1Size) + C2 - IntersectK - (n1*n2)/(2*wholeWeight)
}

func assignInitialCommunities(nodeCommunityMap []int64, community []int64) {
	communityId := int64(0)
	for _, nodeId := range nodeCommunityMap {
		community[nodeId] = communityId
		communityId++
	}
}

func qualityFunction(in []int64, tot []int64, wholeWeight float64) float64 {
	Q := 0.0
	for i := 0; i < len(in); i++ {
		deltaQ := float64(in[i]) - float64(tot[i])*float64(tot[i])/(2*wholeWeight)
		Q += deltaQ
		if deltaQ > 1e8 || deltaQ < -1e8 {
			fmt.Printf("Community: %d\n", i)
			fmt.Printf(" In: %d Tot: %d DeltaQ: %f\n", in[i], tot[i], deltaQ)
		}
	}
	Q /= (2 * wholeWeight)
	return Q
}

func removeQuality(
	in []int64,
	tot []int64,
	n2c map[int64]int64,
	nodeDegree int64,
	node, nodeCommunity int64,
	value float64,
) {
	if nodeCommunity < 0 || nodeCommunity >= int64(len(in)) {
		panic("invalid community index")
	}
	
	in[nodeCommunity] -= int64(value * 2)
	tot[nodeCommunity] -= nodeDegree
	n2c[node] = -1
}

func insertQuality(
	in []int64,
	tot []int64,
	n2c map[int64]int64,
	nodeDegree int64,
	node, nodeCommunity int64,
	value float64,
) {
	if nodeCommunity < 0 || nodeCommunity >= int64(len(in)) {
		panic("invalid community index")
	}
	
	in[nodeCommunity] += int64(value * 2)
	tot[nodeCommunity] += nodeDegree
	n2c[node] = nodeCommunity
}

func gainQuality(
	in []int64,
	tot []int64,
	n2c map[int64]int64,
	node, nodeCommunity, neighborCommunity int64,
	dnc, degc, wholeWeight float64,
) float64 {
	if neighborCommunity < 0 || neighborCommunity >= int64(len(in)) {
		panic("invalid neighbor community index")
	}
	
	totc := float64(tot[neighborCommunity])
	return dnc - totc*degc/wholeWeight/2
}

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
	// Initial community assignment
	n2c := make(map[int64]int64)
	var nodeId []int64
	in := make([]int64, len(communityDegreeSketches))
	tot := make([]int64, len(communityDegreeSketches))
	nodeDegreeSketch := make(map[int64]int64)
	
	for i := range in {
		in[i] = 0
		tot[i] = 1
	}
	
	oldQuality := qualityFunction(in, tot, wholeWeight)
	newQuality := oldQuality
	
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
	
	iter := 0
	for {
		for i := 0; i < len(nodeId); i++ {
			node := nodeId[i]
			nodeCommunity := n2c[node]
			
			neighborCommunity := make(map[int64]bool)
			neighborCommunityWeight := make(map[int64]int64)
			neighborCommunityWeight[nodeCommunity] = 0
			
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
			
			removeQuality(in, tot, n2c, nodeDegreeSketch[node], node, nodeCommunity, oldQuality)
			bestNeighborCommunity := nodeCommunity
			bestIncrease := 0.0
			bestWeight := 0.0
			
			for nei, weight := range neighborCommunityWeight {
				increase := gainQuality(in, tot, n2c, node, nodeCommunity, nei, float64(weight), float64(nodeDegreeSketch[node]), wholeWeight)
				if increase > bestIncrease {
					bestIncrease = increase
					bestNeighborCommunity = nei
					bestWeight = float64(weight)
				}
			}
			
			insertQuality(in, tot, n2c, nodeDegreeSketch[node], node, bestNeighborCommunity, bestWeight)
		}
		
		newQuality = qualityFunction(in, tot, wholeWeight)
		iter++
		
		if !(newQuality > oldQuality && iter < 20) {
			break
		}
		oldQuality = newQuality
	}
	
	// Generate newCommunities based on n2c
	tempNewCommunities := make(map[int64][]int64)
	for node, communityId := range n2c {
		tempNewCommunities[communityId] = append(tempNewCommunities[communityId], community[node])
	}
	
	// Convert tempNewCommunities to the format of newCommunities
	*newCommunities = (*newCommunities)[:0]
	for _, comms := range tempNewCommunities {
		*newCommunities = append(*newCommunities, comms)
	}
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
	communityToNewIndex := make(map[int64]int64)
	maxIndex := int64(0)
	
	for currentCommunity, neighbors := range communityEdgeTable {
		bestNeighbor := int64(-1)
		bestDegreeSketch := math.MaxFloat64
		
		// Find the neighbor with the lowest degree sketch
		for neighborCommunity := range neighbors {
			degreeSketch := float64(communityDegreeSketches[neighborCommunity])
			if degreeSketch < bestDegreeSketch {
				bestDegreeSketch = degreeSketch
				bestNeighbor = neighborCommunity
			}
		}
		
		if bestNeighbor != -1 {
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
				communityToNewIndex[currentCommunity] = maxIndex
				communityToNewIndex[bestNeighbor] = maxIndex
				maxIndex++
			}
		}
	}
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
	communityToNewIndex := make(map[int64]int64)
	maxIndex := int64(0)
	
	for currentCommunity, neighbors := range communityEdgeTable {
		bestNeighbor := int64(-1)
		bestEValue := 1.0
		
		for i := 1; i < iter; i++ {
			bestEValue *= threshold
		}
		
		c1Size := int64(len(nodesInCommunities[currentCommunity]))
		
		for neighborCommunity := range neighbors {
			C2 := 1.0
			intersectK := 1.0
			
			neighborSketch := make([]uint32, len(communitySketches[neighborCommunity]))
			copy(neighborSketch, communitySketches[neighborCommunity])
			
			if int64(len(neighborSketch)) < k*nk {
				C2 = float64(len(neighborSketch)) / float64(nk)
			} else {
				maxVbksC2 := neighborSketch[len(neighborSketch)-1]
				C2 = float64(k-1) * float64(nk) * float64(math.MaxUint32) / float64(maxVbksC2)
			}
			
			// Add current community's hash values
			for _, node := range nodesInCommunities[currentCommunity] {
				for j := int64(0); j < nk; j++ {
					neighborSketch = append(neighborSketch, nodeHashValue[node*nk+j])
				}
			}
			
			sort.Slice(neighborSketch, func(i, j int) bool {
				return neighborSketch[i] < neighborSketch[j]
			})
			
			if int64(len(neighborSketch)) < nk*k {
				count := int64(0)
				for _, node := range nodesInCommunities[currentCommunity] {
					for j := int64(0); j < nk; j++ {
						// Binary search for nodeHashValue[node*nk+j] in neighborSketch
						target := nodeHashValue[node*nk+j]
						idx := sort.Search(len(neighborSketch), func(i int) bool {
							return neighborSketch[i] >= target
						})
						for idx < len(neighborSketch) && neighborSketch[idx] == target {
							count++
							idx++
						}
					}
				}
				intersectK = float64(c1Size) + C2 - float64(count)/float64(nk)
			} else {
				if int64(len(neighborSketch)) > k*nk {
					neighborSketch = neighborSketch[:k*nk]
				}
				intersectK = float64(k-1) * float64(nk) * float64(math.MaxUint32) / float64(neighborSketch[len(neighborSketch)-1])
			}
			
			n1 := float64(communityDegreeSketches[currentCommunity])
			n2 := float64(communityDegreeSketches[neighborCommunity])
			eValue := calculateEFunction(c1Size, C2, intersectK, k, n1, n2, wholeWeight)
			
			if eValue > bestEValue {
				bestEValue = eValue
				bestNeighbor = neighborCommunity
			}
		}
		
		if bestNeighbor != -1 {
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
				communityToNewIndex[currentCommunity] = maxIndex
				communityToNewIndex[bestNeighbor] = maxIndex
				maxIndex++
			}
		}
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
	newCommunitySketches := make(map[int64][]uint32)
	newNodesInCommunity := make(map[int64][]int64)
	
	newCommId := int64(0)
	for ; newCommId < int64(len(newCommunities)); newCommId++ {
		commsToMerge := newCommunities[newCommId]
		if len(commsToMerge) == 0 {
			continue
		}
		
		for _, oldComm := range commsToMerge {
			for _, node := range nodesInCommunities[oldComm] {
				community[node] = newCommId
				newNodesInCommunity[newCommId] = append(newNodesInCommunity[newCommId], node)
			}
			newCommunitySketches[newCommId] = append(newCommunitySketches[newCommId], communitySketches[oldComm]...)
			delete(nodesInCommunities, oldComm)
		}
	}
	
	// Process communities that were not merged
	for comm, nodes := range nodesInCommunities {
		newCommunitySketches[newCommId] = communitySketches[comm]
		for _, node := range nodes {
			community[node] = newCommId
			newNodesInCommunity[newCommId] = append(newNodesInCommunity[newCommId], node)
		}
		newCommId++
	}
	
	// Move the new nodes into the original nodesInCommunities
	for k, v := range newNodesInCommunity {
		nodesInCommunities[k] = v
	}
	for k := range nodesInCommunities {
		if _, exists := newNodesInCommunity[k]; !exists {
			delete(nodesInCommunities, k)
		}
	}
	
	// Sort and resize sketches
	for comm, sketches := range newCommunitySketches {
		sort.Slice(sketches, func(i, j int) bool {
			return sketches[i] < sketches[j]
		})
		if int64(len(sketches)) > k*nk {
			sketches = sketches[:k*nk]
		}
		mergedSketches[comm] = sketches
	}
}
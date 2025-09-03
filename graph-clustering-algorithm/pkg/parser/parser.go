package parser

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Edge represents a weighted edge
type Edge struct {
	From   string
	To     string
	Weight float64
}

// SketchData represents sketch information for a node
type SketchData struct {
	NodeID     string
	Level      int
	SelfHashes []uint32
	Sketches   [][]uint32 // [nk][k] sketch layers
	K          int
	NK         int
	IsFull     bool
}

// SketchParams holds sketch parameters
type SketchParams struct {
	K         int
	NK        int
	Threshold float64
	ProFile   string
	PathFile  string
}

// func main() {
// 	if len(os.Args) < 2 {
// 		printUsage()
// 		os.Exit(1)
// 	}

// 	mode := strings.ToLower(os.Args[1])

// 	switch mode {
// 	case "traditional", "trad", "t":
// 		runTraditionalLouvain()
// 	case "sketch", "scar", "s":
// 		runSketchLouvain()
// 	case "help", "-h", "--help":
// 		printUsage()
// 	default:
// 		fmt.Printf("Unknown mode: %s\n\n", mode)
// 		printUsage()
// 		os.Exit(1)
// 	}
// }

func runTraditionalLouvain() {
	if len(os.Args) < 7 {
		fmt.Println("Traditional Louvain mode requires 5 arguments:")
		fmt.Println("Usage: program traditional <edgelist> <mapping> <hierarchy> <root> <output_prefix>")
		fmt.Println("")
		fmt.Println("Example:")
		fmt.Println("  program traditional graph.txt communities.mapping communities.hierarchy communities.root output")
		os.Exit(1)
	}

	edgelistFile := os.Args[2]
	mappingFile := os.Args[3]
	hierarchyFile := os.Args[4]
	rootFile := os.Args[5]
	outputPrefix := os.Args[6]

	fmt.Printf("=== TRADITIONAL LOUVAIN PARSER ===\n")
	fmt.Printf("Edgelist file:  %s\n", edgelistFile)
	fmt.Printf("Mapping file:   %s\n", mappingFile)
	fmt.Printf("Hierarchy file: %s\n", hierarchyFile)
	fmt.Printf("Root file:      %s\n", rootFile)
	fmt.Printf("Output prefix:  %s\n", outputPrefix)
	fmt.Printf("\n")

	// Check if files exist
	if err := checkFilesExist(edgelistFile, mappingFile, hierarchyFile, rootFile); err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	err := ParseLouvainHierarchy(edgelistFile, mappingFile, hierarchyFile, rootFile, outputPrefix)
	if err != nil {
		fmt.Printf("Error parsing traditional Louvain: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n=== TRADITIONAL LOUVAIN PARSING COMPLETED ===\n")
	printOutputFiles(outputPrefix)
}

func runSketchLouvain() {
	if len(os.Args) < 7 {
		fmt.Println("Sketch Louvain mode requires 5 arguments:")
		fmt.Println("Usage: program sketch <sketch_file> <mapping> <hierarchy> <root> <output_prefix>")
		fmt.Println("")
		fmt.Println("Example:")
		fmt.Println("  program sketch communities.sketch communities.mapping communities.hierarchy communities.root output")
		os.Exit(1)
	}

	sketchFile := os.Args[2]
	mappingFile := os.Args[3]
	hierarchyFile := os.Args[4]
	rootFile := os.Args[5]
	outputPrefix := os.Args[6]

	fmt.Printf("=== SKETCH LOUVAIN PARSER ===\n")
	fmt.Printf("Sketch file:    %s\n", sketchFile)
	fmt.Printf("Mapping file:   %s\n", mappingFile)
	fmt.Printf("Hierarchy file: %s\n", hierarchyFile)
	fmt.Printf("Root file:      %s\n", rootFile)
	fmt.Printf("Output prefix:  %s\n", outputPrefix)
	fmt.Printf("\n")

	// Check if files exist
	if err := checkFilesExist(sketchFile, mappingFile, hierarchyFile, rootFile); err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	err := ParseSketchLouvainHierarchy(sketchFile, mappingFile, hierarchyFile, rootFile, outputPrefix)
	if err != nil {
		fmt.Printf("Error parsing sketch Louvain: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n=== SKETCH LOUVAIN PARSING COMPLETED ===\n")
	printOutputFiles(outputPrefix)
}

func checkFilesExist(filenames ...string) error {
	for _, filename := range filenames {
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			return fmt.Errorf("file does not exist: %s", filename)
		}
	}
	return nil
}

func printOutputFiles(outputPrefix string) {
	fmt.Printf("Generated files:\n")
	
	// Find all output files
	pattern := outputPrefix + "_level_*.txt"
	matches, err := filepath.Glob(pattern)
	if err != nil {
		fmt.Printf("Error finding output files: %v\n", err)
		return
	}

	if len(matches) == 0 {
		fmt.Printf("No output files found with pattern: %s\n", pattern)
		return
	}

	for _, match := range matches {
		if stat, err := os.Stat(match); err == nil {
			fmt.Printf("  - %s (%d bytes)\n", match, stat.Size())
		} else {
			fmt.Printf("  - %s (error: %v)\n", match, err)
		}
	}
}

func printUsage() {
	fmt.Printf("Louvain Hierarchy Parser\n")
	fmt.Printf("========================\n\n")
	
	fmt.Printf("Usage: %s <mode> [arguments...]\n\n", os.Args[0])
	
	fmt.Printf("Modes:\n")
	fmt.Printf("  traditional (t, trad)  - Parse traditional Louvain output\n")
	fmt.Printf("  sketch (s, scar)       - Parse sketch-based Louvain output\n")
	fmt.Printf("  help (-h, --help)      - Show this help message\n\n")
	
	fmt.Printf("Traditional Louvain:\n")
	fmt.Printf("  %s traditional <edgelist> <mapping> <hierarchy> <root> <output_prefix>\n\n", os.Args[0])
	fmt.Printf("  Arguments:\n")
	fmt.Printf("    edgelist      - Original graph edges file\n")
	fmt.Printf("    mapping       - Community mapping file\n")
	fmt.Printf("    hierarchy     - Community hierarchy file\n")
	fmt.Printf("    root          - Root community file\n")
	fmt.Printf("    output_prefix - Prefix for output files\n\n")
	
	fmt.Printf("Sketch Louvain:\n")
	fmt.Printf("  %s sketch <sketch_file> <mapping> <hierarchy> <root> <output_prefix>\n\n", os.Args[0])
	fmt.Printf("  Arguments:\n")
	fmt.Printf("    sketch_file   - Sketch data file (.sketch)\n")
	fmt.Printf("    mapping       - Community mapping file\n")
	fmt.Printf("    hierarchy     - Community hierarchy file\n")
	fmt.Printf("    root          - Root community file\n")
	fmt.Printf("    output_prefix - Prefix for output files\n\n")
	
	fmt.Printf("Examples:\n")
	fmt.Printf("  # Traditional Louvain\n")
	fmt.Printf("  %s traditional graph.txt communities.mapping communities.hierarchy communities.root network\n\n", os.Args[0])
	fmt.Printf("  # Sketch Louvain\n")
	fmt.Printf("  %s sketch communities.sketch communities.mapping communities.hierarchy communities.root network\n\n", os.Args[0])
	
	fmt.Printf("Output:\n")
	fmt.Printf("  Both modes generate hierarchical level files:\n")
	fmt.Printf("    <output_prefix>_level_0.txt  - Level 0 (leaf/original graph)\n")
	fmt.Printf("    <output_prefix>_level_1.txt  - Level 1 (first aggregation)\n")
	fmt.Printf("    <output_prefix>_level_2.txt  - Level 2 (second aggregation)\n")
	fmt.Printf("    ...\n")
}

// ======= SKETCH LOUVAIN FUNCTIONS =======

// ParseSketchLouvainHierarchy parses sketch-based Louvain output and creates hierarchical level graphs
func ParseSketchLouvainHierarchy(sketchFile, mappingFile, hierarchyFile, rootFile, outputPrefix string) error {
	// 1. Parse input files
	sketchParams, nodeToSketch, hashToNode, err := parseSketchFile(sketchFile)
	if err != nil {
		return fmt.Errorf("failed to parse sketch file: %w", err)
	}
	
	mapping, err := parseMappingFile(mappingFile)
	if err != nil {
		return fmt.Errorf("failed to parse mapping: %w", err)
	}
	
	hierarchy, err := parseHierarchyFile(hierarchyFile)
	if err != nil {
		return fmt.Errorf("failed to parse hierarchy: %w", err)
	}
	
	rootID, err := parseRootFile(rootFile)
	if err != nil {
		return fmt.Errorf("failed to parse root: %w", err)
	}
	
	fmt.Printf("Parsed files successfully:\n")
	fmt.Printf("  - Sketch nodes: %d levels\n", len(nodeToSketch))
	fmt.Printf("  - Hash mappings: %d\n", len(hashToNode))
	fmt.Printf("  - Communities: %d\n", len(mapping))
	fmt.Printf("  - Hierarchy entries: %d\n", len(hierarchy))
	fmt.Printf("  - Root: %s\n", rootID)
	fmt.Printf("  - Sketch params: k=%d, nk=%d\n", sketchParams.K, sketchParams.NK)
	
	// 2. Build level mappings
	levels, maxLevel := buildLevelMappings(mapping, hierarchy, rootID)
	
	fmt.Printf("Built %d levels\n", maxLevel+1)
	
	// 3. Build adjacency list for level 0 (leaf nodes)
	adjacencyList := buildLeafAdjacencyList(nodeToSketch[0], hashToNode)
	
	// 4. Generate leaf graph (level 0)
	leafEdges := buildLeafGraph(nodeToSketch[0], hashToNode)
	err = saveEdges(leafEdges, outputPrefix+"_level_0.txt")
	if err != nil {
		return fmt.Errorf("failed to save level 0: %w", err)
	}
	fmt.Printf("Saved level 0: %d edges\n", len(leafEdges))
	
	// 5. Generate higher level graphs
	for level := 1; level <= maxLevel; level++ {
		levelEdges := buildLevelGraph(level, nodeToSketch, levels, adjacencyList, sketchParams)
		filename := fmt.Sprintf("%s_level_%d.txt", outputPrefix, level)
		err = saveEdges(levelEdges, filename)
		if err != nil {
			return fmt.Errorf("failed to save level %d: %w", level, err)
		}
		fmt.Printf("Saved level %d: %d edges\n", level, len(levelEdges))
	}
	
	return nil
}

// parseSketchFile parses the sketch file and extracts sketch data
func parseSketchFile(filename string) (*SketchParams, map[int]map[string]*SketchData, map[uint32]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, nil, err
	}
	defer file.Close()
	
	scanner := bufio.NewScanner(file)
	nodeToSketch := make(map[int]map[string]*SketchData)
	hashToNode := make(map[uint32]string)
	var params *SketchParams
	
	// Parse parameters from first line
	if scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		params, err = parseSketchParams(line)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to parse sketch parameters: %w", err)
		}
	}
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		
		// Parse node header: just the nodeID (no level= part anymore)
		nodeID := line
		
		// Determine level from nodeID format
		var level int
		if _, err := strconv.Atoi(nodeID); err == nil {
			// Pure number = level 0 (leaf node)
			level = 0
		} else if strings.HasPrefix(nodeID, "c0_l") {
			// Extract level from community ID (e.g., "c0_l1_2" -> 1)
			parts := strings.Split(nodeID, "_")
			if len(parts) >= 2 && strings.HasPrefix(parts[1], "l") {
				if l, err := strconv.Atoi(parts[1][1:]); err == nil {
					level = l
				}
			}
		} else {
			continue // Skip unknown format
		}
		
		// Read self-hashes line (first line after node header)
		if !scanner.Scan() {
			break
		}
		selfHashLine := strings.TrimSpace(scanner.Text())
		selfHashes, err := parseHashLine(selfHashLine)
		if err != nil {
			continue
		}
		
		// Build hash to node mapping from self-hashes (only for level 0)
		if level == 0 {
			for _, hash := range selfHashes {
				hashToNode[hash] = nodeID
			}
		}
		
		// Read sketch layers (nk layers)
		sketches := make([][]uint32, params.NK)
		for i := 0; i < params.NK; i++ {
			if !scanner.Scan() {
				break
			}
			sketchLine := strings.TrimSpace(scanner.Text())
			sketch, err := parseHashLine(sketchLine)
			if err != nil {
				sketch = []uint32{}
			}
			sketches[i] = sketch
		}
		
		// Create sketch data
		sketchData := &SketchData{
			NodeID:     nodeID,
			Level:      level,
			SelfHashes: selfHashes,
			Sketches:   sketches,
			K:          params.K,
			NK:         params.NK,
			IsFull:     isSketchFull(sketches[0], params.K),
		}
		
		// Store in nodeToSketch map
		if nodeToSketch[level] == nil {
			nodeToSketch[level] = make(map[string]*SketchData)
		}
		nodeToSketch[level][nodeID] = sketchData
	}
	
	fmt.Printf("Parsed sketch parameters: %+v\n", params)
	fmt.Printf("Parsed node to sketch mapping: %v\n", func() map[int]map[string]string {
		result := make(map[int]map[string]string)
		for level, nodes := range nodeToSketch {
			result[level] = make(map[string]string)
			for nodeID := range nodes {
				result[level][nodeID] = fmt.Sprintf("%p", nodes[nodeID])
			}
		}
		return result
	}())
	fmt.Printf("Parsed hash to node mapping: %v\n", hashToNode)
	
	return params, nodeToSketch, hashToNode, scanner.Err()
}

// parseSketchParams parses the parameter line
func parseSketchParams(line string) (*SketchParams, error) {
	params := &SketchParams{}
	
	parts := strings.Fields(line)
	for _, part := range parts {
		if strings.HasPrefix(part, "k=") {
			val, err := strconv.Atoi(strings.TrimPrefix(part, "k="))
			if err != nil {
				return nil, err
			}
			params.K = val
		} else if strings.HasPrefix(part, "nk=") {
			val, err := strconv.Atoi(strings.TrimPrefix(part, "nk="))
			if err != nil {
				return nil, err
			}
			params.NK = val
		} else if strings.HasPrefix(part, "th=") {
			val, err := strconv.ParseFloat(strings.TrimPrefix(part, "th="), 64)
			if err != nil {
				return nil, err
			}
			params.Threshold = val
		} else if strings.HasPrefix(part, "pro=") {
			params.ProFile = strings.TrimPrefix(part, "pro=")
		} else if strings.HasPrefix(part, "path=") {
			params.PathFile = strings.TrimPrefix(part, "path=")
		}
	}
	
	return params, nil
}

// parseHashLine parses a comma-separated line of hash values
func parseHashLine(line string) ([]uint32, error) {
	if line == "" {
		return []uint32{}, nil
	}
	
	parts := strings.Split(line, ",")
	hashes := make([]uint32, 0, len(parts))
	
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		
		val, err := strconv.ParseUint(part, 10, 32)
		if err != nil {
			continue
		}
		hashes = append(hashes, uint32(val))
	}
	
	return hashes, nil
}

// isSketchFull checks if a sketch is full
func isSketchFull(sketch []uint32, k int) bool {
	return len(sketch) >= k
}

// buildLeafAdjacencyList builds adjacency list for leaf level nodes
func buildLeafAdjacencyList(leafSketches map[string]*SketchData, hashToNode map[uint32]string) map[string][]string {
	adjacencyList := make(map[string][]string)
	
	for nodeID, sketchData := range leafSketches {
		neighbors := make(map[string]bool)
		
		// For non-full sketches, use direct hash-to-node mapping
		if !sketchData.IsFull {
			for _, layer := range sketchData.Sketches {
				for _, hash := range layer {
					if neighborID, exists := hashToNode[hash]; exists && neighborID != nodeID {
						neighbors[neighborID] = true
					}
				}
			}
		} else {
			// For full sketches, find all nodes with sketch intersections
			for otherNodeID, otherSketch := range leafSketches {
				if otherNodeID == nodeID {
					continue
				}
				
				if hasSketchIntersection(sketchData, otherSketch) {
					neighbors[otherNodeID] = true
				}
			}
		}
		
		// Convert to slice
		neighborList := make([]string, 0, len(neighbors))
		for neighbor := range neighbors {
			neighborList = append(neighborList, neighbor)
		}
		adjacencyList[nodeID] = neighborList
	}
	
	return adjacencyList
}

// buildLeafGraph creates edges for the leaf level
func buildLeafGraph(leafSketches map[string]*SketchData, hashToNode map[uint32]string) []Edge {
	var edges []Edge
	processed := make(map[string]bool)
	
	// fmt.Printf("DEBUG: Building leaf graph with %d nodes\n", len(leafSketches))
	
	for nodeID, sketchData := range leafSketches {
		// fmt.Printf("DEBUG: Node %s has %d sketch layers\n", nodeID, len(sketchData.Sketches))
		nodeEdges := 0
		
		for layerIdx, layer := range sketchData.Sketches {
			fmt.Printf("DEBUG: Node %s layer %d has %d hashes: %v\n", nodeID, layerIdx, len(layer), layer)
			
			for _, hash := range layer {
				if neighborID, exists := hashToNode[hash]; exists && neighborID != nodeID {
					// Create edge key to avoid duplicates
					var edgeKey string
					if nodeID <= neighborID {
						edgeKey = nodeID + "|" + neighborID
					} else {
						edgeKey = neighborID + "|" + nodeID
					}
					
					if !processed[edgeKey] {
						// fmt.Printf("DEBUG: Creating edge %s <-> %s (hash %d)\n", nodeID, neighborID, hash)
						edges = append(edges, Edge{From: nodeID, To: neighborID, Weight: 1.0})
						edges = append(edges, Edge{From: neighborID, To: nodeID, Weight: 1.0})
						processed[edgeKey] = true
						nodeEdges++
					}
				} else if exists {
					// fmt.Printf("DEBUG: Skipping self-loop: node %s hash %d maps to same node\n", nodeID, hash)
				} else {
					// fmt.Printf("DEBUG: Hash %d not found in hashToNode mapping\n", hash)
				}
			}
		}
		// fmt.Printf("DEBUG: Node %s generated %d unique edges\n", nodeID, nodeEdges)
	}
	
	// fmt.Printf("DEBUG: Total edges generated: %d\n", len(edges))
	return edges
}

// buildLevelGraph creates edges for a specific level
func buildLevelGraph(level int, nodeToSketch map[int]map[string]*SketchData, 
	levels map[int]map[string]string, adjacencyList map[string][]string, 
	params *SketchParams) []Edge {
	
	levelSketches := nodeToSketch[level]
	if levelSketches == nil {
		return []Edge{}
	}
	
	// Build reverse mapping: leaf node -> level N community
	leafToLevel := make(map[string]string)
	for leafNode, communityID := range levels[level] {
		leafToLevel[leafNode] = communityID
	}
	
	// Build community adjacency list
	commAdjacency := make(map[string]map[string]float64)
	
	// Process non-full sketch communities
	for nodeID, sketchData := range levelSketches {
		if !sketchData.IsFull {
			// Get leaf nodes in this community from mapping[node]
			communityNodes := []string{}
			for leafNode, communityID := range levels[level] {
				if communityID == nodeID {
					communityNodes = append(communityNodes, leafNode)
				}
			}
			
			// For each leaf in mapping[node]
			for _, leafNode := range communityNodes {
				// For each neighbor in adjacencyList[leaf]
				if neighbors, exists := adjacencyList[leafNode]; exists {
					for _, neighbor := range neighbors {
						// Find which community this neighbor belongs to
						if targetCommunity, exists := leafToLevel[neighbor]; exists {
							// Add to community adjacency list
							if commAdjacency[nodeID] == nil {
								commAdjacency[nodeID] = make(map[string]float64)
							}
							commAdjacency[nodeID][targetCommunity] += 1.0
						}
					}
				}
			}
		} else {
			// Full sketch: use estimation method
			if commAdjacency[nodeID] == nil {
				commAdjacency[nodeID] = make(map[string]float64)
			}
			
			for otherNodeID, otherSketch := range levelSketches {
				var weight float64
				if nodeID == otherNodeID {
					// Self-loop: use estimated degree
					weight = estimateCardinality(sketchData, params.K)
				} else {
					// Edge to other node: use estimated edge weight
					weight = estimateEdgeWeight(sketchData, otherSketch, params.K)
				}
				
				if weight > 0 {
					commAdjacency[nodeID][otherNodeID] = weight
				}
			}
		}
	}
	
	// Convert community adjacency list to edge list
	var edges []Edge
	processed := make(map[string]bool)
	
	for commU, neighbors := range commAdjacency {
		for commV, weight := range neighbors {
			// Create edge key to avoid duplicates
			var edgeKey string
			if commU <= commV {
				edgeKey = commU + "|" + commV
			} else {
				edgeKey = commV + "|" + commU
			}
			
			if processed[edgeKey] {
				continue
			}
			
			if commU == commV {
				// Self-loop: divide by 2 (double counted) and add single edge
				finalWeight := weight / 2.0
				edges = append(edges, Edge{From: commU, To: commV, Weight: finalWeight})
			} else {
				// Make edges symmetrical - use same weight in both directions
				// For non-full to full case, get max weight from both directions
				reverseWeight := 0.0
				if commAdjacency[commV] != nil {
					reverseWeight = commAdjacency[commV][commU]
				}
				
				finalWeight := weight
				if reverseWeight > finalWeight {
					finalWeight = reverseWeight
				}
				
				// Add bidirectional edges with same weight
				edges = append(edges, Edge{From: commU, To: commV, Weight: finalWeight})
				edges = append(edges, Edge{From: commV, To: commU, Weight: finalWeight})
			}
			
			processed[edgeKey] = true
		}
	}
	
	return edges
}

// buildNonFullSketchEdges handles non-full sketch nodes
func buildNonFullSketchEdges(nodeID string, sketchData *SketchData, 
	levels map[int]map[string]string, adjacencyList map[string][]string,
	leafToLevel map[string]string, level int, nodeToSketch map[int]map[string]*SketchData) []Edge {
	
	var edges []Edge
	edgeWeights := make(map[string]float64) // Track edge weights between communities
	
	// Get leaf nodes in this community from mapping[node]
	communityNodes := []string{}
	for leafNode, communityID := range levels[level] {
		if communityID == nodeID {
			communityNodes = append(communityNodes, leafNode)
		}
	}
	
	// For each leaf in mapping[node]
	for _, leafNode := range communityNodes {
		// For each neighbor in adjacencyList[leaf]
		if neighbors, exists := adjacencyList[leafNode]; exists {
			for _, neighbor := range neighbors {
				// Find which community this neighbor belongs to
				if targetCommunity, exists := leafToLevel[neighbor]; exists {
					// Create edge key
					var edgeKey string
					if nodeID <= targetCommunity {
						edgeKey = nodeID + "|" + targetCommunity
					} else {
						edgeKey = targetCommunity + "|" + nodeID
					}
					
					// Count the edge (increment weight)
					edgeWeights[edgeKey] += 1.0
				}
			}
		}
	}
	
	// Convert edge weights to actual edges
	for edgeKey, weight := range edgeWeights {
		parts := strings.Split(edgeKey, "|")
		if len(parts) != 2 {
			continue
		}
		
		from := parts[0]
		to := parts[1]
		
		// Check if target has full sketch for double-counting logic
		targetSketch := nodeToSketch[level][to]
		if from != nodeID {
			targetSketch = nodeToSketch[level][from]
		}
		
		if targetSketch != nil && targetSketch.IsFull {
			// Target has full sketch, add bidirectional edge
			edges = append(edges, Edge{From: from, To: to, Weight: weight})
			edges = append(edges, Edge{From: to, To: from, Weight: weight})
		} else {
			// Target has non-full sketch, add only one direction
			edges = append(edges, Edge{From: nodeID, To: to, Weight: weight})
		}
	}
	
	return edges
}

// buildFullSketchEdges handles full sketch nodes using estimation
func buildFullSketchEdges(nodeID string, sketchData *SketchData, 
	levelSketches map[string]*SketchData, processed map[string]bool, 
	params *SketchParams) []Edge {
	
	var edges []Edge
	
	for otherNodeID, otherSketch := range levelSketches {
		// Create edge key
		var edgeKey string
		if nodeID <= otherNodeID {
			edgeKey = nodeID + "|" + otherNodeID
		} else {
			edgeKey = otherNodeID + "|" + nodeID
		}
		
		if processed[edgeKey] {
			continue
		}
		
		var weight float64
		if nodeID == otherNodeID {
			// Self-loop: use estimated degree
			weight = estimateCardinality(sketchData, params.K)
		} else {
			// Edge to other node: use estimated edge weight
			weight = estimateEdgeWeight(sketchData, otherSketch, params.K)
		}
		
		if weight > 0 {
			edges = append(edges, Edge{From: nodeID, To: otherNodeID, Weight: weight})
			if nodeID != otherNodeID {
				edges = append(edges, Edge{From: otherNodeID, To: nodeID, Weight: weight})
			}
		}
		
		processed[edgeKey] = true
	}
	
	return edges
}

// estimateCardinality estimates cardinality using bottom-k sketch
func estimateCardinality(sketchData *SketchData, k int) float64 {
	if !sketchData.IsFull || len(sketchData.Sketches) == 0 {
		// Not full, use exact count
		count := 0
		for _, layer := range sketchData.Sketches {
			count += len(layer)
		}
		return float64(count)
	}
	
	// Full sketch, use estimation formula
	firstLayer := sketchData.Sketches[0]
	if len(firstLayer) >= k {
		rK := float64(firstLayer[k-1])
		if rK > 0 {
			estimate := float64(k-1) * float64(math.MaxUint32) / rK
			return estimate
		}
	}
	
	return 0.0
}

// estimateEdgeWeight estimates edge weight between two sketches using inclusion-exclusion
func estimateEdgeWeight(sketch1, sketch2 *SketchData, k int) float64 {
	if !sketch1.IsFull || !sketch2.IsFull {
		// Use intersection counting for non-full sketches
		return float64(countSketchIntersection(sketch1, sketch2))
	}
	
	// Use inclusion-exclusion principle for full sketches
	card1 := estimateCardinality(sketch1, k)
	card2 := estimateCardinality(sketch2, k)
	
	// Calculate union cardinality
	unionSketch := unionSketches(sketch1, sketch2, k)
	unionCard := estimateCardinality(unionSketch, k)
	
	// Inclusion-exclusion: |A ∩ B| = |A| + |B| - |A ∪ B|
	intersection := card1 + card2 - unionCard
	
	if intersection < 0 {
		intersection = 0
	}
	
	return intersection
}

// hasSketchIntersection checks if two sketches have any common elements
func hasSketchIntersection(sketch1, sketch2 *SketchData) bool {
	return countSketchIntersection(sketch1, sketch2) > 0
}

// countSketchIntersection counts common elements between two sketches
func countSketchIntersection(sketch1, sketch2 *SketchData) int {
	count := 0
	
	for i := 0; i < len(sketch1.Sketches) && i < len(sketch2.Sketches); i++ {
		layer1 := sketch1.Sketches[i]
		layer2 := sketch2.Sketches[i]
		
		// Count intersection in this layer
		hashSet := make(map[uint32]bool)
		for _, hash := range layer1 {
			hashSet[hash] = true
		}
		
		for _, hash := range layer2 {
			if hashSet[hash] {
				count++
			}
		}
	}
	
	return count
}

// unionSketches creates a union of two sketches using bottom-k union
func unionSketches(sketch1, sketch2 *SketchData, k int) *SketchData {
	result := &SketchData{
		NodeID:   sketch1.NodeID + "_union_" + sketch2.NodeID,
		Level:    sketch1.Level,
		Sketches: make([][]uint32, len(sketch1.Sketches)),
		K:        k,
		NK:       sketch1.NK,
	}
	
	for i := 0; i < len(sketch1.Sketches) && i < len(sketch2.Sketches); i++ {
		result.Sketches[i] = bottomKUnion(sketch1.Sketches[i], sketch2.Sketches[i], k)
	}
	
	result.IsFull = isSketchFull(result.Sketches[0], k)
	
	return result
}

// bottomKUnion performs bottom-k union of two sketch layers
func bottomKUnion(sketch1, sketch2 []uint32, k int) []uint32 {
	// Merge and sort all values
	allValues := make([]uint32, 0, len(sketch1)+len(sketch2))
	allValues = append(allValues, sketch1...)
	allValues = append(allValues, sketch2...)
	
	// Remove duplicates and sort
	valueSet := make(map[uint32]bool)
	for _, val := range allValues {
		valueSet[val] = true
	}
	
	uniqueValues := make([]uint32, 0, len(valueSet))
	for val := range valueSet {
		uniqueValues = append(uniqueValues, val)
	}
	
	// Sort values
	for i := 0; i < len(uniqueValues); i++ {
		for j := i + 1; j < len(uniqueValues); j++ {
			if uniqueValues[i] > uniqueValues[j] {
				uniqueValues[i], uniqueValues[j] = uniqueValues[j], uniqueValues[i]
			}
		}
	}
	
	// Take bottom k
	result := make([]uint32, 0, k)
	for i := 0; i < len(uniqueValues) && i < k; i++ {
		result = append(result, uniqueValues[i])
	}
	
	return result
}

// ======= TRADITIONAL LOUVAIN FUNCTIONS =======

// ParseLouvainHierarchy parses Louvain output and creates hierarchical level graphs
func ParseLouvainHierarchy(edgelistFile, mappingFile, hierarchyFile, rootFile, outputPrefix string) error {
	// 1. Parse input files
	leafEdges, err := parseEdgelistFile(edgelistFile)
	if err != nil {
		return fmt.Errorf("failed to parse edgelist: %w", err)
	}
	
	mapping, err := parseMappingFile(mappingFile)
	if err != nil {
		return fmt.Errorf("failed to parse mapping: %w", err)
	}
	
	hierarchy, err := parseHierarchyFile(hierarchyFile)
	if err != nil {
		return fmt.Errorf("failed to parse hierarchy: %w", err)
	}
	
	rootID, err := parseRootFile(rootFile)
	if err != nil {
		return fmt.Errorf("failed to parse root: %w", err)
	}
	
	fmt.Printf("Parsed files successfully:\n")
	fmt.Printf("  - Leaf edges: %d\n", len(leafEdges))
	fmt.Printf("  - Communities: %d\n", len(mapping))
	fmt.Printf("  - Hierarchy entries: %d\n", len(hierarchy))
	fmt.Printf("  - Root: %s\n", rootID)
	
	// 2. Build level mappings
	levels, maxLevel := buildLevelMappings(mapping, hierarchy, rootID)
	
	fmt.Printf("Built %d levels\n", maxLevel+1)
	
	// 3. Save level 0 (original graph)
	err = saveEdges(leafEdges, outputPrefix+"_level_0.txt")
	if err != nil {
		return fmt.Errorf("failed to save level 0: %w", err)
	}
	fmt.Printf("Saved level 0: %d edges\n", len(leafEdges))
	
	// 4. Create and save aggregated levels
	for level := 1; level <= maxLevel; level++ {
		levelEdges := aggregateEdges(leafEdges, levels[level])
		filename := fmt.Sprintf("%s_level_%d.txt", outputPrefix, level)
		err = saveEdges(levelEdges, filename)
		if err != nil {
			return fmt.Errorf("failed to save level %d: %w", level, err)
		}
		fmt.Printf("Saved level %d: %d edges\n", level, len(levelEdges))
	}
	
	return nil
}

// parseEdgelistFile parses the materialization output edgelist
func parseEdgelistFile(filename string) ([]Edge, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	var edges []Edge
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		
		from := parts[0]
		to := parts[1]
		weight := 1.0
		
		if len(parts) >= 3 {
			if w, err := strconv.ParseFloat(parts[2], 64); err == nil {
				weight = w
			}
		}
		
		edges = append(edges, Edge{From: from, To: to, Weight: weight})
	}
	
	return edges, scanner.Err()
}

// parseMappingFile parses the Louvain mapping file
// Format: community_id\nnode_count\nnode1\nnode2\n...
func parseMappingFile(filename string) (map[string][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	mapping := make(map[string][]string)
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		// Read community ID
		communityID := strings.TrimSpace(scanner.Text())
		if communityID == "" {
			continue
		}
		
		// Read node count
		if !scanner.Scan() {
			break
		}
		countStr := strings.TrimSpace(scanner.Text())
		count, err := strconv.Atoi(countStr)
		if err != nil {
			return nil, fmt.Errorf("invalid node count for community %s: %s", communityID, countStr)
		}
		
		// Read nodes
		nodes := make([]string, 0, count)
		for i := 0; i < count && scanner.Scan(); i++ {
			node := strings.TrimSpace(scanner.Text())
			nodes = append(nodes, node)
		}
		
		if len(nodes) != count {
			return nil, fmt.Errorf("expected %d nodes for community %s, got %d", count, communityID, len(nodes))
		}
		
		mapping[communityID] = nodes
	}
	
	return mapping, scanner.Err()
}

// parseHierarchyFile parses the Louvain hierarchy file
// Format: parent_id\nchild_count\nchild1\nchild2\n...
func parseHierarchyFile(filename string) (map[string][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	hierarchy := make(map[string][]string)
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		// Read parent ID
		parentID := strings.TrimSpace(scanner.Text())
		if parentID == "" {
			continue
		}
		
		// Read child count
		if !scanner.Scan() {
			break
		}
		countStr := strings.TrimSpace(scanner.Text())
		count, err := strconv.Atoi(countStr)
		if err != nil {
			return nil, fmt.Errorf("invalid child count for parent %s: %s", parentID, countStr)
		}
		
		// Read children
		children := make([]string, 0, count)
		for i := 0; i < count && scanner.Scan(); i++ {
			child := strings.TrimSpace(scanner.Text())
			children = append(children, child)
		}
		
		if len(children) != count {
			return nil, fmt.Errorf("expected %d children for parent %s, got %d", count, parentID, len(children))
		}
		
		hierarchy[parentID] = children
	}
	
	return hierarchy, scanner.Err()
}

// parseRootFile parses the Louvain root file
// Format: root_community_id\n
func parseRootFile(filename string) (string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return "", err
	}
	defer file.Close()
	
	scanner := bufio.NewScanner(file)
	if scanner.Scan() {
		return strings.TrimSpace(scanner.Text()), nil
	}
	
	if err := scanner.Err(); err != nil {
		return "", err
	}
	
	return "", fmt.Errorf("empty root file")
}

// buildLevelMappings creates node->community mappings for each level
func buildLevelMappings(mapping map[string][]string, hierarchy map[string][]string, rootID string) (map[int]map[string]string, int) {
	levels := make(map[int]map[string]string)
	
	// Extract level from community ID (e.g., "c0_l1_0" -> level 1)
	extractLevel := func(communityID string) int {
		parts := strings.Split(communityID, "_")
		if len(parts) >= 2 && strings.HasPrefix(parts[1], "l") {
			if level, err := strconv.Atoi(parts[1][1:]); err == nil {
				return level
			}
		}
		return 0
	}
	
	maxLevel := 0
	
	// Build mappings for each level from the mapping file
	for communityID, nodes := range mapping {
		level := extractLevel(communityID)
		if level > maxLevel {
			maxLevel = level
		}
		
		if levels[level] == nil {
			levels[level] = make(map[string]string)
		}
		
		// Map each original node to this community at this level
		for _, node := range nodes {
			levels[level][node] = communityID
		}
	}
	
	fmt.Printf("Level mappings built:\n")
	for level := 1; level <= maxLevel; level++ {
		fmt.Printf("  Level %d: %d nodes mapped\n", level, len(levels[level]))
	}
	
	return levels, maxLevel
}

// aggregateEdges creates aggregated edges for a specific level
func aggregateEdges(leafEdges []Edge, nodeToComm map[string]string) []Edge {
	edgeWeights := make(map[string]float64)
	
	// Aggregate edge weights between communities
	for _, edge := range leafEdges {
		fromComm, fromExists := nodeToComm[edge.From]
		toComm, toExists := nodeToComm[edge.To]
		
		if !fromExists || !toExists {
			continue // Skip edges involving nodes not in this level
		}
		// fmt.Printf("DEBUG: Processing edge %s -> %s (weight %.6f) in communities %s -> %s\n", edge.From, edge.To, edge.Weight, fromComm, toComm)
		
		// Create edge key (use consistent ordering for undirected edges)
		var edgeKey string
		if fromComm <= toComm {
			edgeKey = fromComm + "|" + toComm
		} else {
			edgeKey = toComm + "|" + fromComm
		}
		// fmt.Printf("DEBUG: Edge key: %s\n", edgeKey)
		edgeWeights[edgeKey] += edge.Weight
	}
	
	// Convert to edge list and divide by 2 for undirected graphs
	var result []Edge
	processed := make(map[string]bool)
	
	for edgeKey, totalWeight := range edgeWeights {
		if processed[edgeKey] {
			continue
		}
		
		parts := strings.Split(edgeKey, "|")
		if len(parts) != 2 {
			continue
		}
		
		from := parts[0]
		to := parts[1]
		weight := totalWeight / 2.0 // Divide by 2 for undirected graphs
		
		// Add both directions for undirected graph
		result = append(result, Edge{From: from, To: to, Weight: weight})
		if from != to {
			result = append(result, Edge{From: to, To: from, Weight: weight})
		}
		
		processed[edgeKey] = true
	}
	
	return result
}

// saveEdges saves edges to a file in "from to weight" format
func saveEdges(edges []Edge, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	for _, edge := range edges {
		_, err := fmt.Fprintf(file, "%s %s %.6f\n", edge.From, edge.To, edge.Weight)
		if err != nil {
			return err
		}
	}
	
	return nil
}
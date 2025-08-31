
package scar

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	// "time"
	
	"github.com/rs/zerolog"
)

type NodeMapping struct {
    OriginalToCompressed map[int]int  `json:"original_to_compressed"`
    CompressedToOriginal []int        `json:"compressed_to_original"` 
    NumTargetNodes       int          `json:"num_target_nodes"`
}

// BuildSketchGraph performs sketch preprocessing and returns a SketchGraph
func BuildSketchGraph(graphFile, propertiesFile, pathFile string, config *Config, logger zerolog.Logger) (*SketchGraph, *NodeMapping, error) {
	logger.Info().Msg("Starting sketch preprocessing")
	
	// Step 1: Read raw graph
	rawGraph, err := readRawGraph(graphFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read graph: %w", err)
	}
	
	// Step 2: Read properties and path
	properties, err := readProperties(propertiesFile, int64(rawGraph.NumNodes))
	if err != nil {
		logger.Warn().Err(err).Msg("Failed to read properties, using defaults")
		properties = make([]uint32, rawGraph.NumNodes) // All zeros
	}
	
	path, pathLength, err := readPath(pathFile)
	if err != nil {
		logger.Warn().Err(err).Msg("Failed to read path, using default")
		path = []uint32{0}
		pathLength = 1
	}

	// Step 3: Compute sketches
	sketches, nodeHashValues, err := computeSketches(rawGraph, properties, path, pathLength, config)
	if err != nil {
		return nil, nil, fmt.Errorf("sketch computation failed: %w", err)
	}
	
	// Step 4: Build SketchGraph
	sketchGraph, nodeMapping, err := buildSketchGraphFromSketches(rawGraph, sketches, nodeHashValues, config, path[0], properties)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to build sketch graph: %w", err)
	}
	
	logger.Info().
		Int("nodes", sketchGraph.NumNodes).
		Float64("total_weight", sketchGraph.TotalWeight).
		Msg("Sketch preprocessing completed")

	return sketchGraph, nodeMapping, nil
}

// RawGraph represents the input graph structure
type RawGraph struct {
	NumNodes  int
	Adjacency [][]int64
}

// readRawGraph reads the graph from edge list format
func readRawGraph(filename string) (*RawGraph, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	edges := make(map[int64][]int64)
	maxNode := int64(0)

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
			}
		}
	}

	// Convert to adjacency list format
	numNodes := int(maxNode + 1)
	adjacency := make([][]int64, numNodes)
	for i := int64(0); i < int64(numNodes); i++ {
		adjacency[i] = edges[i]
	}

	return &RawGraph{
		NumNodes:  numNodes,
		Adjacency: adjacency,
	}, scanner.Err()
}

// readProperties reads node properties from file
func readProperties(filename string, n int64) ([]uint32, error) {
	properties := make([]uint32, n)
	
	if filename == "" {
		return properties, nil
	}
	
	file, err := os.Open(filename)
	if err != nil {
		return properties, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			nodeId, err1 := strconv.ParseInt(parts[0], 10, 64)
			property, err2 := strconv.ParseInt(parts[1], 10, 32)
			if err1 == nil && err2 == nil && nodeId < n {
				properties[nodeId] = uint32(property)
			}
		}
	}
	
	return properties, scanner.Err()
}

// readPath reads the path specification from file
func readPath(filename string) ([]uint32, int64, error) {
	path := []uint32{0}
	pathLength := int64(1)
	
	if filename == "" {
		return path, pathLength, nil
	}
	
	file, err := os.Open(filename)
	if err != nil {
		return path, pathLength, err
	}
	defer file.Close()

	path = make([]uint32, 0, 20)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		
		if label, err := strconv.ParseInt(line, 10, 32); err == nil {
			path = append(path, uint32(label))
		}
	}
	
	if len(path) == 0 {
		path = []uint32{0}
		pathLength = 1
	} else {
		pathLength = int64(len(path))
	}
	
	return path, pathLength, scanner.Err()
}

// computeSketches performs the sketch propagation algorithm
func computeSketches(rawGraph *RawGraph, properties []uint32, path []uint32, pathLength int64, config *Config) ([]uint32, []uint32, error) {
	n := int64(rawGraph.NumNodes)
	propK := config.K() + 1 // +1 to accommodate for self hash
	nk := config.NK()
	
	// Initialize sketch storage
	sketches := make([]uint32, pathLength*n*nk*propK)
	for i := range sketches {
		sketches[i] = math.MaxUint32
	}
	
	nodeHashValues := make([]uint32, n*nk)
	
	// GLOBAL hash deduplication set
	usedHashes := make(map[uint32]struct{})
	// Reserve MaxUint32 as sentinel value
	usedHashes[math.MaxUint32] = struct{}{}
	
	// Initialize frontier with nodes having the first label
	// rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	rng := rand.New(rand.NewSource(config.RandomSeed() + 1))
	firstLabel := path[0]
	
	// Helper function to generate unique hash
	generateUniqueHash := func() uint32 {
		maxAttempts := 1000 // Prevent infinite loops
		for attempts := 0; attempts < maxAttempts; attempts++ {
			candidate := rng.Uint32()
			
			// Check if hash is already used
			if _, exists := usedHashes[candidate]; !exists {
				usedHashes[candidate] = struct{}{}
				return candidate
			}
		}
		
		// Fallback: linear search for unused value (should be extremely rare)
		for candidate := uint32(0); candidate < math.MaxUint32; candidate++ {
			if _, exists := usedHashes[candidate]; !exists {
				usedHashes[candidate] = struct{}{}
				return candidate
			}
		}
		
		// Should never reach here with reasonable graph sizes
		panic("unable to generate unique hash - hash space exhausted")
	}
	
	for i := int64(0); i < n; i++ {
		if properties[i] == firstLabel {
			for j := int64(0); j < nk; j++ {
				uniqueHash := generateUniqueHash()
				sketches[j*n*propK+i*propK] = uniqueHash
				nodeHashValues[i*nk+j] = uniqueHash + 1
			}
		}
	}
	
	// Propagate sketches along the path (unchanged - union handles duplicates correctly)
	for iter := int64(1); iter < pathLength; iter++ {
		currentLabel := path[iter]
		
		// For each node with current label, collect sketches from neighbors
		for node := int64(0); node < n; node++ {
			if properties[node] != currentLabel {
				continue
			}
		
			// Union sketches from all neighbors
			for _, neighbor := range rawGraph.Adjacency[node] {
				for layer := int64(0); layer < nk; layer++ {
					srcIdx := (iter-1)*n*nk*propK + layer*n*propK + neighbor*propK 
					dstIdx := iter*n*nk*propK + layer*n*propK + node*propK

					// Extract source and destination sketches
					srcSketch := make([]uint32, propK)
					dstSketch := make([]uint32, propK)

					for ki := int64(0); ki < propK; ki++ {
						if srcIdx+ki < int64(len(sketches)) {
							srcSketch[ki] = sketches[srcIdx+ki]
						} else {
							srcSketch[ki] = math.MaxUint32
						}
						if dstIdx+ki < int64(len(sketches)) {
							dstSketch[ki] = sketches[dstIdx+ki]
						} else {
							dstSketch[ki] = math.MaxUint32
						}
					}
					
					// Proper bottom-k union (merge k smallest values)
					result := make([]uint32, propK)
					for i := range result {
						result[i] = math.MaxUint32
					}
					
					i, j, t := 0, 0, 0
					for t < int(propK) {
						var val1, val2 uint32 = math.MaxUint32, math.MaxUint32
						
						if i < int(propK) && srcSketch[i] != math.MaxUint32 {
							val1 = srcSketch[i]
						}
						if j < int(propK) && dstSketch[j] != math.MaxUint32 {
							val2 = dstSketch[j]
						}
						
						if val1 == val2 && val1 != math.MaxUint32 {
							result[t] = val1
							t++; i++; j++
						} else if val1 < val2 {
							result[t] = val1
							t++; i++
						} else if val2 < math.MaxUint32 {
							result[t] = val2
							t++; j++
						} else {
							break
						}
					}
					
					// Write back the corrected union
					for ki := int64(0); ki < propK; ki++ {
						if dstIdx+ki < int64(len(sketches)) {
							sketches[dstIdx+ki] = result[ki]
						}
					}
				}
			}
		}
	}
	
	// Extract final sketches (last iteration)
	finalSketches := sketches[(pathLength-1)*n*propK*nk:]

	// Add 1 to all sketches (original SCAR logic)
	for i := range finalSketches {
		if finalSketches[i] != math.MaxUint32 {
			finalSketches[i]++
		}
	}
	
	return finalSketches, nodeHashValues, nil
}

// buildSketchGraphFromSketches creates a SketchGraph from computed sketches
func buildSketchGraphFromSketches(rawGraph *RawGraph, sketches []uint32, nodeHashValues []uint32, config *Config, targetType uint32, properties []uint32) (*SketchGraph, *NodeMapping, error) {
	// Build node mapping for target type
	nodeMapping := buildNodeMapping(properties, targetType, sketches, nodeHashValues, config)

	// Create sketch graph with the number of target nodes (compressed)
	sketchGraph := NewSketchGraph(nodeMapping.NumTargetNodes)
	sketchGraph.sketchManager = NewSketchManager(config.K(), config.NK())
	
	n := rawGraph.NumNodes
	finalK := config.K()
	propK := finalK + 1 // +1 to accommodate for self hash
	nk := config.NK()

	// Pre-build per-node self-hash set (for all original nodes - needed for sketch filtering)
	selfHashes := make([]map[uint32]struct{}, n)
	for i := int64(0); i < int64(n); i++ {
		m := make(map[uint32]struct{}, nk)
		for j := int64(0); j < nk; j++ {
			hv := nodeHashValues[i*nk+j]
			if hv != 0 {
				m[hv] = struct{}{}
			}
		}
		selfHashes[i] = m
	}
	
	// Convert sketches to VertexBottomKSketch objects - ONLY for target nodes
	for originalId, compressedId := range nodeMapping.OriginalToCompressed {
		if nodeHashValues[int64(originalId)*nk] != 0 { // Node has sketches
			sketch := NewVertexBottomKSketch(int64(compressedId), finalK, nk) 
			
			// Fill sketch layers: convert from propK to finalK, dropping self-hashes
			for layer := int64(0); layer < nk; layer++ {
				// Collect candidates from propK-sized sketch, dropping self-hashes
				out := make([]uint32, 0, finalK)
				
				for ki := int64(0); ki < propK; ki++ {
					// Use originalId to access sketch data (sketches computed for all nodes)
					idx := layer*int64(n)*propK + int64(originalId)*propK + ki
					if int(idx) >= len(sketches) {
						break
					}
					
					v := sketches[idx]
					if v == math.MaxUint32 {
						break // remaining are empty
					}
					
					// Drop self-hash
					if _, isSelf := selfHashes[originalId][v]; isSelf {
						continue
					}
					
					// Optional dedup within layer
					if len(out) > 0 && v == out[len(out)-1] {
						continue
					}
					
					out = append(out, v)
					if len(out) == int(finalK) { 
						break
					}
				}
				
				// Pad if under-filled to finalK size
				layerSketch := make([]uint32, finalK) 
				for idx := range layerSketch {
					if idx < len(out) {
						layerSketch[idx] = out[idx]
					} else {
						layerSketch[idx] = math.MaxUint32
					}
				}
				
				sketch.sketches[layer] = layerSketch
			}
			
			sketch.UpdateFilledCount()
			
			// Only add if it has non-self content
			if sketch.filledCount > 0 {
				// Store using compressedId in the new graph
				sketchGraph.sketchManager.vertexSketches[int64(compressedId)] = sketch
			}
			
			// Create identifying hash sketch for this node  
			identifyingHashSketch := NewVertexBottomKSketch(int64(compressedId), finalK, nk)

			// Fill with this node's identifying hashes (one per layer)
			for layer := int64(0); layer < nk; layer++ {
				hashValue := nodeHashValues[int64(originalId)*nk+layer]
				if hashValue != 0 {
					identifyingHashSketch.sketches[layer][0] = hashValue  // One hash per layer
					// Rest of layer is empty (MaxUint32)
					for i := int64(1); i < finalK; i++ {
						identifyingHashSketch.sketches[layer][i] = math.MaxUint32
					}
					
					// Build hashToNodeMap (all nk hashes point to same node)
					sketchGraph.sketchManager.hashToNodeMap[hashValue] = int64(compressedId)
				}
			}

		 	sketchGraph.sketchManager.nodeToHashMap[int64(compressedId)] = identifyingHashSketch
		}
	}
	
	// Build adjacency list for non-full sketches (using compressed graph)
	sketchGraph.buildAdjacencyList(rawGraph, nodeMapping)
	
	return sketchGraph, nodeMapping, nil
}


// buildNodeMapping creates a mapping from original node IDs to compressed indices
func buildNodeMapping(properties []uint32, targetType uint32, sketches []uint32, nodeHashValues []uint32, config *Config) *NodeMapping {
    mapping := &NodeMapping{
        OriginalToCompressed: make(map[int]int),
        CompressedToOriginal: make([]int, 0),
    }
    
    n := len(properties)
    finalK := config.K()
    propK := finalK + 1
    nk := config.NK()
    
    // Pre-build self-hash sets for filtering
    selfHashes := make([]map[uint32]struct{}, n)
    for i := 0; i < n; i++ {
        m := make(map[uint32]struct{}, nk)
        for j := int64(0); j < nk; j++ {
            hv := nodeHashValues[int64(i)*nk+j]
            if hv != 0 {
                m[hv] = struct{}{}
            }
        }
        selfHashes[i] = m
    }
    
    compressedId := 0
    for originalId := 0; originalId < n; originalId++ {
        // Check 1: Must be target type
        if properties[originalId] != targetType {
            continue
        }
        
        // Check 2: Must have non-empty sketch
        if hasNonEmptySketch(originalId, sketches, selfHashes[originalId], int64(n), finalK, propK, nk) {
            mapping.OriginalToCompressed[originalId] = compressedId
            mapping.CompressedToOriginal = append(mapping.CompressedToOriginal, originalId)
            compressedId++
        }
    }
    
    mapping.NumTargetNodes = compressedId
    return mapping
}

// Check if a node has a non-empty sketch (after removing self-hashes)
func hasNonEmptySketch(originalId int, sketches []uint32, selfHashSet map[uint32]struct{}, n, finalK, propK, nk int64) bool {
    // Must have hash values first
    if len(selfHashSet) == 0 {
        return false
    }
    
    // Check if any layer has non-self content
    for layer := int64(0); layer < nk; layer++ {
        for ki := int64(0); ki < propK; ki++ {
			idx := layer*int64(n)*propK + int64(originalId)*propK + ki
            if int(idx) >= len(sketches) {
                break
            }
            v := sketches[idx]
            if v == math.MaxUint32 {
                break // Remaining are empty
            }
            
            // If we find a non-self hash, sketch is non-empty
            if _, isSelf := selfHashSet[v]; !isSelf {
                return true
            }
        }
    }
    
    return false // All hashes were self-hashes or empty
}
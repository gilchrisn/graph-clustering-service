package scar

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"testing"
	"time"
)

// Test 1: Test estimation with exact matching and file logging
func TestEstimation(t *testing.T) {
	k := int64(10)
	nk := int64(1)
	n := 1000 // number of hashes to generate
	numTestCases := 1000 // Run 1000 cases

	config := NewConfig()
	config.Set("scar.k", k)
	config.Set("scar.nk", nk)

	for testCase := 0; testCase < numTestCases; testCase++ {
		t.Run(fmt.Sprintf("TestCase_%d", testCase), func(t *testing.T) {
			// Generate n unique random hashes
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(testCase)))
			hashSet := make(map[uint32]bool)
			hashes := make([]uint32, 0, n)
			
			for len(hashes) < n {
				hash := rng.Uint32()
				if hash == math.MaxUint32 {
					hash-- // Avoid MaxUint32
				}
				
				// Only add if not already seen
				if !hashSet[hash] {
					hashSet[hash] = true
					hashes = append(hashes, hash)
				}
			}

			// Sort hashes for baseline
			sort.Slice(hashes, func(i, j int) bool {
				return hashes[i] < hashes[j]
			})

			// Ensure we have enough unique hashes for bottom-k
			if len(hashes) < int(k) {
				t.Fatalf("Not enough unique hashes generated: got %d, need %d", len(hashes), k)
			}

			// Ensure we have enough unique hashes for bottom-k
			if len(hashes) < int(k) {
				t.Fatalf("Not enough unique hashes generated: got %d, need %d", len(hashes), k)
			}

			// Baseline: pick bottom k and estimate degree
			bottomK := hashes[k-1] // k-th smallest (0-indexed)
			baselineEstimate := float64(k-1) * float64(math.MaxUint32) / float64(bottomK)

			// Create temporary files for SCAR preprocessing
			graphFile, err := createTestGraphFile(int(k), testCase)
			if err != nil {
				t.Fatalf("Failed to create graph file: %v", err)
			}
			defer os.Remove(graphFile)

			propertiesFile, err := createTestPropertiesFile(2*int(k), int(k), testCase)
			if err != nil {
				t.Fatalf("Failed to create properties file: %v", err)
			}
			defer os.Remove(propertiesFile)

			pathFile, err := createTestPathFile(testCase)
			if err != nil {
				t.Fatalf("Failed to create path file: %v", err)
			}
			defer os.Remove(pathFile)

			// Mock the sketch computation to use our sorted hashes
			originalSeed := config.RandomSeed()
			mockSeed := int64(testCase) + 12345 // Deterministic seed
			config.Set("algorithm.random_seed", mockSeed)

			// Build sketch graph using actual SCAR pipeline
			sketchGraph, nodeMapping, err := BuildSketchGraph(graphFile, propertiesFile, pathFile, config, config.CreateLogger())
			if err != nil {
				t.Fatalf("BuildSketchGraph failed: %v", err)
			}

			// Restore original seed
			config.Set("algorithm.random_seed", originalSeed)

			// Find compressed ID for original node 0
			compressedNode0, exists := nodeMapping.OriginalToCompressed[0]
			if !exists {
				t.Fatalf("Node 0 not found in mapping")
			}

			// CAPTURE ORIGINAL STATE FOR DEBUGGING
			originalSketch := sketchGraph.sketchManager.GetVertexSketch(int64(compressedNode0))
			var originalSketchData [][]uint32
			if originalSketch != nil {
				originalSketchData = make([][]uint32, nk)
				for layer := int64(0); layer < nk; layer++ {
					originalSketchData[layer] = make([]uint32, k)
					copy(originalSketchData[layer], originalSketch.GetSketch(layer))
				}
			}

			// Override node 0's sketch with our controlled hashes
			node0Sketch := NewVertexBottomKSketch(int64(compressedNode0), k, nk)
			for layer := int64(0); layer < nk; layer++ {
				for i := int64(0); i < k; i++ {
					node0Sketch.sketches[layer][i] = hashes[i]
				}
			}
			node0Sketch.UpdateFilledCount()
			sketchGraph.sketchManager.vertexSketches[int64(compressedNode0)] = node0Sketch

			// CAPTURE HASH MAPPINGS FOR DEBUGGING
			originalHashMappings := make(map[uint32]int64)
			for hash, nodeId := range sketchGraph.sketchManager.hashToNodeMap {
				originalHashMappings[hash] = nodeId
			}

			// Override hash-to-node mapping with our controlled assignments
			for i := int64(0); i < k; i++ {
				if compressedId, exists := nodeMapping.OriginalToCompressed[int(i)]; exists {
					sketchGraph.sketchManager.hashToNodeMap[hashes[i]] = int64(compressedId)
				}
			}

			// Test: Estimate degree of node 0 using sketch
			estimatedDegree := sketchGraph.GetDegree(compressedNode0)

			// EXACT MATCHING - No tolerance
			exactTolerance := 0.001 // Very small tolerance for floating point precision
			diff := math.Abs(estimatedDegree - baselineEstimate)

			if diff > exactTolerance {
				// WRITE FAILURE TO FILE
				filename := fmt.Sprintf("test_failure_estimation_case_%d.log", testCase)
				logFile, err := os.Create(filename)
				if err != nil {
					t.Fatalf("Could not create log file: %v", err)
				}
				defer logFile.Close()

				fmt.Fprintf(logFile, "=== ESTIMATION TEST FAILURE - CASE %d ===\n", testCase)
				fmt.Fprintf(logFile, "Timestamp: %s\n", time.Now().Format("2006-01-02 15:04:05"))
				fmt.Fprintf(logFile, "Test Seed: %d\n", mockSeed)
				fmt.Fprintf(logFile, "Parameters: k=%d, nk=%d, n=%d\n", k, nk, n)
				fmt.Fprintf(logFile, "Generated unique hashes: %d (requested %d)\n", len(hashes), n)
				fmt.Fprintf(logFile, "Expected: %.10f\n", baselineEstimate)
				fmt.Fprintf(logFile, "Got: %.10f\n", estimatedDegree)
				fmt.Fprintf(logFile, "Difference: %.10f\n", diff)
				fmt.Fprintf(logFile, "Tolerance: %.10f\n", exactTolerance)
				fmt.Fprintf(logFile, "\n")

				// Log the generated hash sequence
				fmt.Fprintf(logFile, "Generated hashes (first 20):\n")
				for i := 0; i < 20 && i < len(hashes); i++ {
					fmt.Fprintf(logFile, "  [%d]: %d\n", i, hashes[i])
				}
				fmt.Fprintf(logFile, "\n")

				fmt.Fprintf(logFile, "Bottom-k hashes:\n")
				for i := int64(0); i < k; i++ {
					fmt.Fprintf(logFile, "  [%d]: %d\n", i, hashes[i])
				}
				fmt.Fprintf(logFile, "Critical bottom-k hash (position %d): %d\n", k-1, bottomK)
				fmt.Fprintf(logFile, "\n")

				// Log original vs modified sketch
				if originalSketch != nil {
					fmt.Fprintf(logFile, "Original sketch state:\n")
					fmt.Fprintf(logFile, "  Filled count: %d, Is full: %v\n", originalSketch.GetFilledCount(), originalSketch.IsSketchFull())
					for layer := int64(0); layer < nk; layer++ {
						fmt.Fprintf(logFile, "  Layer %d: ", layer)
						for i := int64(0); i < 10 && i < k; i++ { // First 10 values
							fmt.Fprintf(logFile, "%d ", originalSketchData[layer][i])
						}
						fmt.Fprintf(logFile, "\n")
					}
					fmt.Fprintf(logFile, "\n")
				}

				fmt.Fprintf(logFile, "Modified sketch state:\n")
				fmt.Fprintf(logFile, "  Filled count: %d, Is full: %v\n", node0Sketch.GetFilledCount(), node0Sketch.IsSketchFull())
				for layer := int64(0); layer < nk; layer++ {
					layerData := node0Sketch.GetSketch(layer)
					fmt.Fprintf(logFile, "  Layer %d: ", layer)
					for i := int64(0); i < 10 && i < k; i++ { // First 10 values
						fmt.Fprintf(logFile, "%d ", layerData[i])
					}
					fmt.Fprintf(logFile, "\n")
				}
				fmt.Fprintf(logFile, "\n")

				// Log estimation formula components
				if node0Sketch.IsSketchFull() {
					sum := uint32(0)
					fmt.Fprintf(logFile, "Estimation formula breakdown:\n")
					for layer := int64(0); layer < nk; layer++ {
						kthValue := node0Sketch.sketches[layer][k-1]
						sum += kthValue
						fmt.Fprintf(logFile, "  Layer %d k-th value: %d\n", layer, kthValue)
					}
					avgKthValue := float64(sum) / float64(nk)
					manualEstimate := float64(k-1) * float64(math.MaxUint32) / avgKthValue
					fmt.Fprintf(logFile, "  Sum of k-th values: %d\n", sum)
					fmt.Fprintf(logFile, "  Average k-th value: %.6f\n", avgKthValue)
					fmt.Fprintf(logFile, "  Manual calculation: %.10f\n", manualEstimate)
					fmt.Fprintf(logFile, "  MaxUint32: %d\n", math.MaxUint32)
					fmt.Fprintf(logFile, "  Formula: (%d-1) * %d / %.6f = %.10f\n", k, math.MaxUint32, avgKthValue, manualEstimate)
				}
				fmt.Fprintf(logFile, "\n")

				// Log hash mapping info
				fmt.Fprintf(logFile, "Hash mapping changes:\n")
				changedMappings := 0
				for hash, newNodeId := range sketchGraph.sketchManager.hashToNodeMap {
					if originalNodeId, existed := originalHashMappings[hash]; !existed || originalNodeId != newNodeId {
						if changedMappings < 20 { // Limit output
							fmt.Fprintf(logFile, "  Hash %d: %d -> %d\n", hash, originalNodeId, newNodeId)
						}
						changedMappings++
					}
				}
				fmt.Fprintf(logFile, "  Total mapping changes: %d\n", changedMappings)
				fmt.Fprintf(logFile, "\n")

				// Log graph structure info
				fmt.Fprintf(logFile, "Graph structure:\n")
				fmt.Fprintf(logFile, "  Total nodes: %d\n", sketchGraph.NumNodes)
				fmt.Fprintf(logFile, "  Total weight: %.6f\n", sketchGraph.TotalWeight)
				fmt.Fprintf(logFile, "  Node %d adjacency count: %d\n", compressedNode0, len(sketchGraph.adjacencyList[compressedNode0]))
				fmt.Fprintf(logFile, "  Original node 0 -> compressed node %d\n", compressedNode0)
				fmt.Fprintf(logFile, "  Node mapping: %d target nodes\n", nodeMapping.NumTargetNodes)

				fmt.Fprintf(logFile, "\n=== END FAILURE LOG ===\n")

				// Fail the test with minimal CLI output
				t.Errorf("Estimation test failed (case %d). Details logged to: %s", testCase, filename)
				return // Stop after first failure
			}

			// Only log progress every 100 cases to keep CLI clean
			if testCase%100 == 0 {
				t.Logf("Estimation test: %d/%d cases passed", testCase+1, numTestCases)
			}
		})
	}
}

// Test 2: Test union with exact matching and file logging
func TestUnion(t *testing.T) {
	k := int64(512)
	nk := int64(1)
	n := 10000 // number of hashes to generate
	numTestCases := 10000 // Run 1000 cases

	for testCase := 0; testCase < numTestCases; testCase++ {
		t.Run(fmt.Sprintf("TestCase_%d", testCase), func(t *testing.T) {
			// Generate n unique random hashes
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(testCase)))
			hashSet := make(map[uint32]bool)
			hashes := make([]uint32, 0, n)
			
			for len(hashes) < n {
				hash := rng.Uint32()
				if hash == math.MaxUint32 {
					hash-- // Avoid MaxUint32
				}
				
				// Only add if not already seen
				if !hashSet[hash] {
					hashSet[hash] = true
					hashes = append(hashes, hash)
				}
			}

			// Sort hashes for baseline
			sort.Slice(hashes, func(i, j int) bool {
				return hashes[i] < hashes[j]
			})

			// Baseline: pick bottom k and estimate degree
			bottomK := hashes[k-1] // k-th smallest (0-indexed)
			baselineEstimate := float64(k-1) * float64(math.MaxUint32) / float64(bottomK)

			// Create sketch graph with k nodes (type 0 nodes only)
			numNodes := int(k)
			sketchGraph := NewSketchGraph(numNodes)
			sketchGraph.sketchManager = NewSketchManager(k, nk)

			// CAPTURE SETUP DATA FOR DEBUGGING
			nodeSketchSetup := make(map[int64][][]uint32)

			// Set hash assignments: node 0 gets lowest hash, node 1 gets second lowest, etc.
			for nodeId := int64(0); nodeId < k; nodeId++ {
				nodeSketch := NewVertexBottomKSketch(nodeId, k, nk)
				nodeSketchData := make([][]uint32, nk)

				// Each node gets assigned its corresponding hash from the sorted list
				for layer := int64(0); layer < nk; layer++ {
					nodeSketch.sketches[layer][0] = hashes[nodeId] // Put hash in first position
					for i := int64(1); i < k; i++ {
						nodeSketch.sketches[layer][i] = math.MaxUint32 // Rest are empty
					}

					// Capture for debugging
					nodeSketchData[layer] = make([]uint32, k)
					copy(nodeSketchData[layer], nodeSketch.sketches[layer])
				}
				nodeSketch.UpdateFilledCount()
				nodeSketchSetup[nodeId] = nodeSketchData

				sketchGraph.sketchManager.vertexSketches[nodeId] = nodeSketch
				sketchGraph.sketchManager.hashToNodeMap[hashes[nodeId]] = nodeId
			}

			// Initialize each node in their own community
			comm := NewCommunity(sketchGraph)

			// Move all nodes into node 0's community
			for nodeId := 1; nodeId < int(k); nodeId++ {
				oldComm := comm.NodeToCommunity[nodeId]
				newComm := 0

				// Remove from old community
				oldNodes := comm.CommunityNodes[oldComm]
				for i, n := range oldNodes {
					if n == nodeId {
						comm.CommunityNodes[oldComm] = append(oldNodes[:i], oldNodes[i+1:]...)
						break
					}
				}

				// Add to new community
				comm.CommunityNodes[newComm] = append(comm.CommunityNodes[newComm], nodeId)
				comm.NodeToCommunity[nodeId] = newComm
			}

			// Update community sketch for community 0 (union of all member sketches)
			sketchGraph.UpdateCommunitySketch(0, comm.CommunityNodes[0], comm)

			// Test: Estimate community degree
			estimatedCommunityDegree := sketchGraph.EstimateCommunityCardinality(0, comm)

			// EXACT MATCHING - No tolerance
			exactTolerance := 0.001 // Very small tolerance for floating point precision
			diff := math.Abs(estimatedCommunityDegree - baselineEstimate)

			if diff > exactTolerance {
				// WRITE FAILURE TO FILE
				filename := fmt.Sprintf("test_failure_union_case_%d.log", testCase)
				logFile, err := os.Create(filename)
				if err != nil {
					t.Fatalf("Could not create log file: %v", err)
				}
				defer logFile.Close()

				fmt.Fprintf(logFile, "=== UNION TEST FAILURE - CASE %d ===\n", testCase)
				fmt.Fprintf(logFile, "Timestamp: %s\n", time.Now().Format("2006-01-02 15:04:05"))
				fmt.Fprintf(logFile, "Parameters: k=%d, nk=%d, n=%d\n", k, nk, n)
				fmt.Fprintf(logFile, "Generated unique hashes: %d (requested %d)\n", len(hashes), n)
				fmt.Fprintf(logFile, "Expected: %.10f\n", baselineEstimate)
				fmt.Fprintf(logFile, "Got: %.10f\n", estimatedCommunityDegree)
				fmt.Fprintf(logFile, "Difference: %.10f\n", diff)
				fmt.Fprintf(logFile, "Tolerance: %.10f\n", exactTolerance)
				fmt.Fprintf(logFile, "\n")

				// Log the hash assignments
				fmt.Fprintf(logFile, "Hash assignments:\n")
				for i := int64(0); i < k; i++ {
					fmt.Fprintf(logFile, "  Node %d gets hash: %d (position %d in sorted list)\n", i, hashes[i], i)
				}
				fmt.Fprintf(logFile, "Critical bottom-k hash (position %d): %d\n", k-1, bottomK)
				fmt.Fprintf(logFile, "\n")

				// Log individual node sketches before union
				fmt.Fprintf(logFile, "Individual node sketches:\n")
				for nodeId := int64(0); nodeId < k; nodeId++ {
					nodeData := nodeSketchSetup[nodeId]
					fmt.Fprintf(logFile, "  Node %d:\n", nodeId)
					for layer := int64(0); layer < nk; layer++ {
						fmt.Fprintf(logFile, "    Layer %d: ", layer)
						for i := int64(0); i < 5 && i < k; i++ { // First 5 values
							fmt.Fprintf(logFile, "%d ", nodeData[layer][i])
						}
						fmt.Fprintf(logFile, "\n")
					}
				}
				fmt.Fprintf(logFile, "\n")

				// Verify community sketch was created
				communitySketch := comm.GetCommunitySketch(0)
				if communitySketch == nil {
					fmt.Fprintf(logFile, "ERROR: Community sketch was not created!\n")
				} else {
					fmt.Fprintf(logFile, "Community sketch after union:\n")
					fmt.Fprintf(logFile, "  Filled count: %d, Is full: %v\n", communitySketch.GetFilledCount(), communitySketch.IsSketchFull())
					for layer := int64(0); layer < nk; layer++ {
						layerData := communitySketch.GetSketch(layer)
						fmt.Fprintf(logFile, "  Layer %d: ", layer)
						for i := int64(0); i < 10 && i < k; i++ { // First 10 values
							fmt.Fprintf(logFile, "%d ", layerData[i])
						}
						fmt.Fprintf(logFile, "\n")
					}

					// Log union estimation details
					if communitySketch.IsSketchFull() {
						sum := uint32(0)
						fmt.Fprintf(logFile, "\nUnion estimation formula breakdown:\n")
						for layer := int64(0); layer < nk; layer++ {
							kthValue := communitySketch.sketches[layer][k-1]
							sum += kthValue
							fmt.Fprintf(logFile, "  Layer %d k-th value: %d\n", layer, kthValue)
						}
						avgKthValue := float64(sum) / float64(nk)
						manualEstimate := float64(k-1) * float64(math.MaxUint32) / avgKthValue
						fmt.Fprintf(logFile, "  Sum of k-th values: %d\n", sum)
						fmt.Fprintf(logFile, "  Average k-th value: %.6f\n", avgKthValue)
						fmt.Fprintf(logFile, "  Manual calculation: %.10f\n", manualEstimate)
					}
				}

				fmt.Fprintf(logFile, "\nCommunity nodes: %v\n", comm.CommunityNodes[0])
				fmt.Fprintf(logFile, "\n=== END FAILURE LOG ===\n")

				// Fail the test with minimal CLI output
				t.Errorf("Union test failed (case %d). Details logged to: %s", testCase, filename)
				return // Stop after first failure
			}

			// Only log progress every 100 cases to keep CLI clean
			if testCase%100 == 0 {
				t.Logf("Union test: %d/%d cases passed", testCase+1, numTestCases)
			}
		})
	}
}

// Helper functions remain the same
func createTestGraphFile(k, testCase int) (string, error) {
	filename := fmt.Sprintf("test_graph_%d_%d.txt", k, testCase)
	file, err := os.Create(filename)
	if err != nil {
		return "", err
	}
	defer file.Close()

	for i := 1; i < k; i++ {
		type1Node := k - 1 + i
		fmt.Fprintf(file, "%d %d\n", i, type1Node)
		fmt.Fprintf(file, "%d %d\n", type1Node, 0)
	}

	return filename, nil
}

func createTestPropertiesFile(totalNodes, k, testCase int) (string, error) {
	filename := fmt.Sprintf("test_properties_%d_%d.txt", k, testCase)
	file, err := os.Create(filename)
	if err != nil {
		return "", err
	}
	defer file.Close()

	for i := 0; i < totalNodes; i++ {
		if i < k {
			fmt.Fprintf(file, "%d 0\n", i) // type 0
		} else {
			fmt.Fprintf(file, "%d 1\n", i) // type 1
		}
	}

	return filename, nil
}

func createTestPathFile(testCase int) (string, error) {
	filename := fmt.Sprintf("test_path_%d.txt", testCase)
	file, err := os.Create(filename)
	if err != nil {
		return "", err
	}
	defer file.Close()

	fmt.Fprintf(file, "0\n")
	fmt.Fprintf(file, "1\n")
	fmt.Fprintf(file, "0\n")

	return filename, nil
}

func TestSCARBasicFunctionalities(t *testing.T) {
	t.Run("Estimation", TestEstimation)
	t.Run("Union", TestUnion)
}
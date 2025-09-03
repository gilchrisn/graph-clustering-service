// utils/nmi_calculator.go - Normalized Mutual Information implementation
package utils

import (
	"fmt"
	"log"
	"math"
	"sort"
)

// ClusterStats represents basic statistics for a clustering
type ClusterStats struct {
	Mean float64 `json:"mean"`
	Max  int     `json:"max"`
	Min  int     `json:"min"`
	Std  float64 `json:"std"`
}

// ComparisonMetrics represents the result of comparing two clusterings
type ComparisonMetrics struct {
	NMI           float64                    `json:"nmi"`
	ClusterCounts map[string]int             `json:"clusterCounts"`
	ClusterSizeStats map[string]ClusterStats `json:"clusterSizeStats"`
	Similarity    string                     `json:"similarity"`
	Details       map[string]interface{}     `json:"details"`
	Error         string                     `json:"error,omitempty"`
}

// NormalizedMutualInfo calculates Normalized Mutual Information (NMI) between two clusterings
// Returns NMI score between 0 and 1
func NormalizedMutualInfo(clustering1, clustering2 []int) (float64, error) {
	if len(clustering1) != len(clustering2) {
		return 0, fmt.Errorf("clusterings must have the same length")
	}
	
	n := len(clustering1)
	if n == 0 {
		return 0, nil
	}
	
	// Build contingency table
	contingencyTable := buildContingencyTable(clustering1, clustering2)
	
	// Calculate mutual information
	mi := calculateMutualInformation(contingencyTable, n)
	
	// Calculate entropies for normalization
	h1 := calculateEntropy(clustering1)
	h2 := calculateEntropy(clustering2)
	
	// Normalize MI by average entropy
	avgEntropy := (h1 + h2) / 2
	
	// Handle edge case where both clusterings have only one cluster
	if avgEntropy == 0 {
		return 1.0, nil
	}
	
	return mi / avgEntropy, nil
}

// buildContingencyTable builds contingency table showing overlap between clusterings
func buildContingencyTable(clustering1, clustering2 []int) map[string]int {
	table := make(map[string]int)
	
	for i := 0; i < len(clustering1); i++ {
		c1 := clustering1[i]
		c2 := clustering2[i]
		key := fmt.Sprintf("%d_%d", c1, c2)
		
		table[key]++
	}
	
	return table
}

// calculateMutualInformation calculates mutual information from contingency table
func calculateMutualInformation(contingencyTable map[string]int, n int) float64 {
	// Get cluster counts for each clustering
	counts1 := make(map[int]int)
	counts2 := make(map[int]int)
	
	for key, count := range contingencyTable {
		var c1, c2 int
		fmt.Sscanf(key, "%d_%d", &c1, &c2)
		counts1[c1] += count
		counts2[c2] += count
	}
	
	mi := 0.0
	
	for key, nij := range contingencyTable {
		var c1, c2 int
		fmt.Sscanf(key, "%d_%d", &c1, &c2)
		ni := counts1[c1]
		nj := counts2[c2]
		
		if nij > 0 {
			mi += float64(nij)/float64(n) * math.Log2(float64(nij*n)/float64(ni*nj))
		}
	}
	
	return mi
}

// calculateEntropy calculates entropy of a clustering
func calculateEntropy(clustering []int) float64 {
	counts := make(map[int]int)
	
	// Count occurrences of each cluster
	for _, cluster := range clustering {
		counts[cluster]++
	}
	
	n := len(clustering)
	entropy := 0.0
	
	for _, count := range counts {
		p := float64(count) / float64(n)
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	
	return entropy
}

// ExtractClusteringFromHierarchy extracts clustering assignments from hierarchy data
// Convert hierarchy structure to flat cluster assignments for NMI calculation
func ExtractClusteringFromHierarchy(hierarchyData, mappingData interface{}) ([]int, error) {
	// clusterAssignments := []int{}
	nodeToCluster := make(map[int]int)
	
	// Get all leaf nodes from mapping data
	allLeafNodes := make(map[int]bool)
	
	// Convert mappingData to map[string]interface{}
	mapping, ok := mappingData.(map[string]interface{})
	if !ok {
		return []int{}, fmt.Errorf("mapping data is not in expected format")
	}
	
	for _, leavesData := range mapping {
		if leavesMap, ok := leavesData.(map[string]interface{}); ok {
			if leaves, ok := leavesMap["leaves"]; ok {
				if leavesSlice, ok := leaves.([]interface{}); ok {
					for _, leafData := range leavesSlice {
						var leafID int
						switch v := leafData.(type) {
						case int:
							leafID = v
						case float64:
							leafID = int(v)
						default:
							continue
						}
						allLeafNodes[leafID] = true
					}
				} else if leavesIntSlice, ok := leaves.([]int); ok {
					for _, leafID := range leavesIntSlice {
						allLeafNodes[leafID] = true
					}
				}
			}
		}
	}
	
	// Assign cluster IDs to leaf nodes
	clusterID := 0
	for supernodeID, leavesData := range mapping {
		_ = supernodeID // Use supernodeID if needed for debugging
		
		var leaves []int
		if leavesMap, ok := leavesData.(map[string]interface{}); ok {
			if leavesInterface, ok := leavesMap["leaves"]; ok {
				if leavesSlice, ok := leavesInterface.([]interface{}); ok {
					for _, leafData := range leavesSlice {
						var leafID int
						switch v := leafData.(type) {
						case int:
							leafID = v
						case float64:
							leafID = int(v)
						default:
							continue
						}
						leaves = append(leaves, leafID)
					}
				} else if leavesIntSlice, ok := leavesInterface.([]int); ok {
					leaves = leavesIntSlice
				}
			}
		}
		
		for _, leafID := range leaves {
			nodeToCluster[leafID] = clusterID
		}
		clusterID++
	}
	
	// Create ordered array of cluster assignments
	var sortedLeafNodes []int
	for leafID := range allLeafNodes {
		sortedLeafNodes = append(sortedLeafNodes, leafID)
	}
	sort.Ints(sortedLeafNodes)
	
	assignments := make([]int, len(sortedLeafNodes))
	for i, nodeID := range sortedLeafNodes {
		if clusterID, exists := nodeToCluster[nodeID]; exists {
			assignments[i] = clusterID
		} else {
			assignments[i] = 0
		}
	}
	
	log.Printf("Extracted clustering: %d nodes, %d clusters", len(assignments), clusterID)
	return assignments, nil
}

// CompareAlgorithmResults compares two algorithm results and calculates comprehensive metrics
func CompareAlgorithmResults(heteroHierarchy, scarHierarchy, heteroMapping, scarMapping interface{}) (*ComparisonMetrics, error) {
	log.Println("🔄 Calculating comparison metrics...")
	
	// Extract flat clusterings for NMI calculation
	heteroClustering, err1 := ExtractClusteringFromHierarchy(heteroHierarchy, heteroMapping)
	scarClustering, err2 := ExtractClusteringFromHierarchy(scarHierarchy, scarMapping)
	
	if err1 != nil {
		log.Printf("Error extracting heterogeneous clustering: %v", err1)
	}
	if err2 != nil {
		log.Printf("Error extracting SCAR clustering: %v", err2)
	}
	
	// Calculate NMI if we have valid clusterings
	nmi := 0.0
	if len(heteroClustering) > 0 && len(scarClustering) > 0 {
		// Ensure both clusterings have the same length
		minLength := len(heteroClustering)
		if len(scarClustering) < minLength {
			minLength = len(scarClustering)
		}
		
		heteroTrimmed := heteroClustering[:minLength]
		scarTrimmed := scarClustering[:minLength]
		
		if minLength > 0 {
			nmiVal, err := NormalizedMutualInfo(heteroTrimmed, scarTrimmed)
			if err == nil {
				nmi = nmiVal
				log.Printf("🔎 NMI calculated: %.4f", nmi)
			} else {
				log.Printf("Error calculating NMI: %v", err)
			}
		}
	}
	
	// Calculate basic metrics
	heteroClusterCount := 0
	scarClusterCount := 0
	
	if heteroHierarchy != nil {
		if hierarchy, ok := heteroHierarchy.(map[string]interface{}); ok {
			heteroClusterCount = len(hierarchy)
		}
	}
	
	if scarHierarchy != nil {
		if hierarchy, ok := scarHierarchy.(map[string]interface{}); ok {
			scarClusterCount = len(hierarchy)
		}
	}
	
	// Calculate cluster size distributions
	heteroSizes := extractClusterSizes(heteroMapping)
	scarSizes := extractClusterSizes(scarMapping)
	
	// Determine similarity level
	similarity := "Low"
	if nmi > 0.7 {
		similarity = "High"
	} else if nmi > 0.4 {
		similarity = "Moderate"
	}
	
	metrics := &ComparisonMetrics{
		NMI: nmi,
		ClusterCounts: map[string]int{
			"heterogeneous": heteroClusterCount,
			"scar":          scarClusterCount,
		},
		ClusterSizeStats: map[string]ClusterStats{
			"heterogeneous": calculateStats(heteroSizes),
			"scar":          calculateStats(scarSizes),
		},
		Similarity: similarity,
		Details: map[string]interface{}{
			"heteroNodes":           len(heteroClustering),
			"scarNodes":             len(scarClustering),
			"heteroUniqueClusters":  countUniqueClusters(heteroClustering),
			"scarUniqueClusters":    countUniqueClusters(scarClustering),
		},
	}
	
	log.Printf("✅ Comparison metrics completed: %+v", metrics)
	return metrics, nil
}

// extractClusterSizes extracts cluster sizes from mapping data
func extractClusterSizes(mappingData interface{}) []int {
	var sizes []int
	
	if mapping, ok := mappingData.(map[string]interface{}); ok {
		for _, leavesData := range mapping {
			if leavesMap, ok := leavesData.(map[string]interface{}); ok {
				if leaves, ok := leavesMap["leaves"]; ok {
					if leavesSlice, ok := leaves.([]interface{}); ok {
						if len(leavesSlice) > 0 {
							sizes = append(sizes, len(leavesSlice))
						}
					} else if leavesIntSlice, ok := leaves.([]int); ok {
						if len(leavesIntSlice) > 0 {
							sizes = append(sizes, len(leavesIntSlice))
						}
					}
				}
			}
		}
	}
	
	return sizes
}

// countUniqueClusters counts unique clusters in a clustering
func countUniqueClusters(clustering []int) int {
	unique := make(map[int]bool)
	for _, cluster := range clustering {
		unique[cluster] = true
	}
	return len(unique)
}

// calculateStats calculates basic statistics for an array of numbers
func calculateStats(values []int) ClusterStats {
	if len(values) == 0 {
		return ClusterStats{Mean: 0, Max: 0, Min: 0, Std: 0}
	}
	
	// Calculate mean
	sum := 0
	for _, v := range values {
		sum += v
	}
	mean := float64(sum) / float64(len(values))
	
	// Find max and min
	max := values[0]
	min := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
		if v < min {
			min = v
		}
	}
	
	// Calculate standard deviation
	variance := 0.0
	for _, v := range values {
		diff := float64(v) - mean
		variance += diff * diff
	}
	variance /= float64(len(values))
	std := math.Sqrt(variance)
	
	return ClusterStats{
		Mean: math.Round(mean*100) / 100, // Round to 2 decimal places
		Max:  max,
		Min:  min,
		Std:  math.Round(std*100) / 100, // Round to 2 decimal places
	}
}
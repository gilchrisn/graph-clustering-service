package louvain

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// GraphParser handles parsing and normalizing graph files
type GraphParser struct {
	// Mapping from original node ID to normalized index
	OriginalToNormalized map[string]int
	// Mapping from normalized index to original node ID
	NormalizedToOriginal map[int]string
	// Total number of nodes
	NumNodes int
}

// ParseResult contains the parsed graph and mappings
type ParseResult struct {
	Graph   *NormalizedGraph
	Parser  *GraphParser
}

// NewGraphParser creates a new graph parser
func NewGraphParser() *GraphParser {
	return &GraphParser{
		OriginalToNormalized: make(map[string]int),
		NormalizedToOriginal: make(map[int]string),
		NumNodes:             0,
	}
}

// ParseEdgeList parses an edge list file and returns a normalized graph
// Expected format: "from to weight" or "from to" (weight defaults to 1.0)
func (p *GraphParser) ParseEdgeList(filename string) (*ParseResult, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// First pass: collect all unique node IDs
	nodeSet := make(map[string]bool)
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
		
		// Skip self-loops
		if from == to {
			continue
		}
		
		nodeSet[from] = true
		nodeSet[to] = true
	}
	
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	// Create normalized mapping
	if err := p.createNormalizedMapping(nodeSet); err != nil {
		return nil, err
	}

	// Second pass: create the graph
	file.Seek(0, 0) // Reset file pointer
	scanner = bufio.NewScanner(file)
	
	graph := NewNormalizedGraph(p.NumNodes)
	
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
		
		// Skip self-loops
		if from == to {
			continue
		}
		
		weight := 1.0
		if len(parts) >= 3 {
			if w, err := strconv.ParseFloat(parts[2], 64); err == nil {
				weight = w
			}
		}
		
		fromIdx := p.OriginalToNormalized[from]
		toIdx := p.OriginalToNormalized[to]
		
		graph.AddEdge(fromIdx, toIdx, weight)
	}
	
	return &ParseResult{
		Graph:  graph,
		Parser: p,
	}, nil
}

// createNormalizedMapping creates the bidirectional mapping between original and normalized IDs
func (p *GraphParser) createNormalizedMapping(nodeSet map[string]bool) error {
	// Convert to slice for sorting
	nodes := make([]string, 0, len(nodeSet))
	for node := range nodeSet {
		nodes = append(nodes, node)
	}
	
	// Sort nodes to ensure consistent ordering
	// Sort nodes to ensure consistent ordering
	// Try to sort numerically if all nodes are integers
	allIntegers := p.allNodesAreIntegers(nodes)
	if allIntegers {
		sort.Slice(nodes, func(i, j int) bool {
			a, _ := strconv.Atoi(nodes[i])
			b, _ := strconv.Atoi(nodes[j])
			return a < b
		})
	} else {
		sort.Strings(nodes)
	}
	
	// Create mappings
	p.NumNodes = len(nodes)
	for i, node := range nodes {
		p.OriginalToNormalized[node] = i
		p.NormalizedToOriginal[i] = node
	}
	
	// ============= ADD DEBUG PRINTING HERE =============
	fmt.Printf("\n=== NODE NORMALIZATION MAPPING ===\n")
	fmt.Printf("All nodes are integers: %t\n", allIntegers)
	fmt.Printf("Total nodes: %d\n", p.NumNodes)
	fmt.Printf("Sorted order: %v\n", nodes)
	fmt.Printf("\nOriginal -> Normalized mapping:\n")
	for i, originalNode := range nodes {
		fmt.Printf("  %s -> %d\n", originalNode, i)
	}
	fmt.Printf("===================================\n\n")
	// ============= END DEBUG PRINTING =============
	
	return nil
}

// allNodesAreIntegers checks if all node IDs can be parsed as integers
func (p *GraphParser) allNodesAreIntegers(nodes []string) bool {
	for _, node := range nodes {
		if _, err := strconv.Atoi(node); err != nil {
			return false
		}
	}
	return true
}

// GetOriginalID returns the original node ID for a normalized index
func (p *GraphParser) GetOriginalID(normalizedID int) (string, bool) {
	originalID, exists := p.NormalizedToOriginal[normalizedID]
	return originalID, exists
}

// GetNormalizedID returns the normalized index for an original node ID
func (p *GraphParser) GetNormalizedID(originalID string) (int, bool) {
	normalizedID, exists := p.OriginalToNormalized[originalID]
	return normalizedID, exists
}

// ConvertCommunityMapping converts a community mapping from normalized indices back to original IDs
func (p *GraphParser) ConvertCommunityMapping(normalizedMapping map[int]int) map[string]int {
	originalMapping := make(map[string]int)
	
	for normalizedID, communityID := range normalizedMapping {
		if originalID, exists := p.NormalizedToOriginal[normalizedID]; exists {
			originalMapping[originalID] = communityID
		}
	}
	
	return originalMapping
}
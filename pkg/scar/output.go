package scar

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// FileWriter handles writing SCAR results to files in Louvain-compatible format
type FileWriter struct {
	OutputDir string
	Prefix    string
	Graph     *HeterogeneousGraph
	Result    *ScarResult
}

// NewFileWriter creates a new file writer
func NewFileWriter(outputDir, prefix string, graph *HeterogeneousGraph, result *ScarResult) *FileWriter {
	return &FileWriter{
		OutputDir: outputDir,
		Prefix:    prefix,
		Graph:     graph,
		Result:    result,
	}
}

// WriteAll writes all output files in Louvain-compatible format
func WriteAll(result *ScarResult, graph *HeterogeneousGraph, outputDir, prefix string) error {
	writer := NewFileWriter(outputDir, prefix, graph, result)
	
	// Ensure output directory exists
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %v", err)
	}

	// Write all required files
	if err := writer.WriteHierarchyFiles(); err != nil {
		return fmt.Errorf("failed to write hierarchy files: %v", err)
	}

	if err := writer.WriteMappingFiles(); err != nil {
		return fmt.Errorf("failed to write mapping files: %v", err)
	}

	if err := writer.WriteRootFile(); err != nil {
		return fmt.Errorf("failed to write root file: %v", err)
	}

	if err := writer.WriteEdgesFiles(); err != nil {
		return fmt.Errorf("failed to write edges files: %v", err)
	}

	return nil
}

// WriteHierarchyFiles writes hierarchy files for each level
// Format: supernode_name \n num_children \n child1 \n child2 \n ...
func (w *FileWriter) WriteHierarchyFiles() error {
	for level, hierarchy := range w.Result.HierarchyLevels {
		filename := filepath.Join(w.OutputDir, "hierarchy-output", fmt.Sprintf("%s_%d.dat", w.Prefix, level))
		
		// Ensure directory exists
		if err := os.MkdirAll(filepath.Dir(filename), 0755); err != nil {
			return err
		}

		file, err := os.Create(filename)
		if err != nil {
			return err
		}
		defer file.Close()

		writer := bufio.NewWriter(file)

		// Sort supernodes for consistent output
		supernodes := make([]string, 0, len(hierarchy))
		for supernode := range hierarchy {
			supernodes = append(supernodes, supernode)
		}
		sort.Strings(supernodes)

		for _, supernode := range supernodes {
			children := hierarchy[supernode]
			
			// Write supernode name
			fmt.Fprintf(writer, "%s\n", supernode)
			
			// Write number of children
			fmt.Fprintf(writer, "%d\n", len(children))
			
			// Write children (convert to numeric IDs if needed)
			sortedChildren := make([]string, len(children))
			copy(sortedChildren, children)
			sort.Strings(sortedChildren)
			
			for _, child := range sortedChildren {
				// Convert node ID to numeric ID for output
				numericID := w.nodeToNumericID(child)
				fmt.Fprintf(writer, "%d\n", numericID)
			}
		}

		writer.Flush()
	}

	return nil
}

// WriteMappingFiles writes mapping files for each level
// Format: supernode_name \n num_leafs \n leaf1 \n leaf2 \n ...
func (w *FileWriter) WriteMappingFiles() error {
	for level, mapping := range w.Result.MappingLevels {
		filename := filepath.Join(w.OutputDir, "mapping-output", fmt.Sprintf("%s_%d.dat", w.Prefix, level))
		
		// Ensure directory exists
		if err := os.MkdirAll(filepath.Dir(filename), 0755); err != nil {
			return err
		}

		file, err := os.Create(filename)
		if err != nil {
			return err
		}
		defer file.Close()

		writer := bufio.NewWriter(file)

		// Sort supernodes for consistent output
		supernodes := make([]string, 0, len(mapping))
		for supernode := range mapping {
			supernodes = append(supernodes, supernode)
		}
		sort.Strings(supernodes)

		for _, supernode := range supernodes {
			leafs := mapping[supernode]
			
			// Write supernode name
			fmt.Fprintf(writer, "%s\n", supernode)
			
			// Write number of leafs
			fmt.Fprintf(writer, "%d\n", len(leafs))
			
			// Write leaf nodes (original nodes as numeric IDs)
			sortedLeafs := make([]string, len(leafs))
			copy(sortedLeafs, leafs)
			sort.Strings(sortedLeafs)
			
			for _, leaf := range sortedLeafs {
				numericID := w.nodeToNumericID(leaf)
				fmt.Fprintf(writer, "%d\n", numericID)
			}
		}

		writer.Flush()
	}

	return nil
}

// WriteRootFile writes the root file containing the top-level community
func (w *FileWriter) WriteRootFile() error {
	filename := filepath.Join(w.OutputDir, fmt.Sprintf("%s.root", w.Prefix))
	
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Find the top-level supernode (from the highest level)
	if len(w.Result.HierarchyLevels) == 0 {
		return fmt.Errorf("no hierarchy levels found")
	}

	topLevel := len(w.Result.HierarchyLevels) - 1
	topLevelHierarchy := w.Result.HierarchyLevels[topLevel]
	
	// Get the first (and usually only) supernode at the top level
	var rootSupernode string
	for supernode := range topLevelHierarchy {
		rootSupernode = supernode
		break
	}

	if rootSupernode == "" {
		// If no clear root, create one based on the final communities
		rootSupernode = fmt.Sprintf("c0_l%d_0", topLevel)
	}

	fmt.Fprintf(file, "%s\n", rootSupernode)
	return nil
}

// WriteEdgesFiles writes edge files for each level containing intra-community edges
func (w *FileWriter) WriteEdgesFiles() error {
	for level, levelInfo := range w.Result.Levels {
		filename := filepath.Join(w.OutputDir, "edges-output", fmt.Sprintf("%s_%d.dat", w.Prefix, level))
		
		// Ensure directory exists
		if err := os.MkdirAll(filepath.Dir(filename), 0755); err != nil {
			return err
		}

		file, err := os.Create(filename)
		if err != nil {
			return err
		}
		defer file.Close()

		writer := bufio.NewWriter(file)

		// Write intra-community edges
		for _, nodes := range levelInfo.C2N {
			if len(nodes) < 2 {
				continue // No internal edges possible
			}

			// Find all edges within this community
			nodeSet := make(map[string]bool)
			for _, node := range nodes {
				nodeSet[node] = true
			}

			// Check all edges in the original graph
			for _, edge := range w.Graph.Edges {
				// Check if both endpoints are in this community
				if nodeSet[edge.From] && nodeSet[edge.To] {
					fromID := w.nodeToNumericID(edge.From)
					toID := w.nodeToNumericID(edge.To)
					
					// Write edge (format: from_id to_id weight)
					fmt.Fprintf(writer, "%d %d %.6f\n", fromID, toID, edge.Weight)
				}
			}
		}

		writer.Flush()
	}

	return nil
}

// WriteStatistics writes algorithm statistics to a file
func (w *FileWriter) WriteStatistics() error {
	filename := filepath.Join(w.OutputDir, fmt.Sprintf("%s.stats", w.Prefix))
	
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	// Write statistics
	fmt.Fprintf(writer, "SCAR Algorithm Statistics\n")
	fmt.Fprintf(writer, "========================\n")
	fmt.Fprintf(writer, "Total Levels: %d\n", w.Result.Statistics.TotalLevels)
	fmt.Fprintf(writer, "Total Iterations: %d\n", w.Result.Statistics.TotalIterations)
	fmt.Fprintf(writer, "Total Duration: %v\n", w.Result.Statistics.TotalDuration)
	fmt.Fprintf(writer, "Final Modularity: %.6f\n", w.Result.Statistics.FinalModularity)
	fmt.Fprintf(writer, "Initial Nodes: %d\n", w.Result.Statistics.InitialNodes)
	fmt.Fprintf(writer, "Initial Edges: %d\n", w.Result.Statistics.InitialEdges)
	fmt.Fprintf(writer, "Final Nodes: %d\n", w.Result.Statistics.FinalNodes)
	fmt.Fprintf(writer, "\nLevel Details:\n")
	
	for _, level := range w.Result.Levels {
		fmt.Fprintf(writer, "Level %d: %d nodes -> %d communities, modularity=%.6f, iterations=%d, duration=%v\n",
			level.Level, level.Nodes, level.Communities, level.Modularity, level.Iterations, level.Duration)
	}

	writer.Flush()
	return nil
}

// WriteCommunities writes final community assignments
func (w *FileWriter) WriteCommunities() error {
	filename := filepath.Join(w.OutputDir, fmt.Sprintf("%s.communities", w.Prefix))
	
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)

	// Sort nodes for consistent output
	nodes := make([]string, 0, len(w.Result.FinalCommunities))
	for node := range w.Result.FinalCommunities {
		nodes = append(nodes, node)
	}
	sort.Strings(nodes)

	// Write node -> community assignments
	for _, node := range nodes {
		community := w.Result.FinalCommunities[node]
		numericID := w.nodeToNumericID(node)
		fmt.Fprintf(writer, "%d %d\n", numericID, community)
	}

	writer.Flush()
	return nil
}

// WriteAllFiles writes all output files including optional statistics and communities
func (w *FileWriter) WriteAllFiles() error {
	// Write required files
	if err := w.WriteHierarchyFiles(); err != nil {
		return err
	}
	if err := w.WriteMappingFiles(); err != nil {
		return err
	}
	if err := w.WriteRootFile(); err != nil {
		return err
	}
	if err := w.WriteEdgesFiles(); err != nil {
		return err
	}

	// Write optional files
	if err := w.WriteStatistics(); err != nil {
		return err
	}
	if err := w.WriteCommunities(); err != nil {
		return err
	}

	return nil
}

// nodeToNumericID converts node ID to numeric ID for output compatibility
// This creates a mapping from string IDs to numeric IDs
func (w *FileWriter) nodeToNumericID(nodeID string) int {
	// Create a consistent mapping from string IDs to numeric IDs
	// This is a simple approach - in practice, you might want to store this mapping
	
	// Find the node in the original graph's node list
	for i, node := range w.Graph.NodeList {
		if node == nodeID {
			return i
		}
	}
	
	// If not found, hash the string to create a consistent numeric ID
	hash := 0
	for _, char := range nodeID {
		hash = hash*31 + int(char)
	}
	
	// Ensure positive
	if hash < 0 {
		hash = -hash
	}
	
	return hash
}

// numericIDToNode converts numeric ID back to node ID
func (w *FileWriter) numericIDToNode(numericID int) string {
	if numericID >= 0 && numericID < len(w.Graph.NodeList) {
		return w.Graph.NodeList[numericID]
	}
	
	// Return string representation if not found in list
	return strconv.Itoa(numericID)
}

// ValidateOutputFiles validates that output files are correctly formatted
func ValidateOutputFiles(outputDir, prefix string) error {
	// Check that all required files exist
	requiredDirs := []string{"hierarchy-output", "mapping-output", "edges-output"}
	
	for _, dir := range requiredDirs {
		dirPath := filepath.Join(outputDir, dir)
		if _, err := os.Stat(dirPath); os.IsNotExist(err) {
			return fmt.Errorf("required directory not found: %s", dirPath)
		}
	}
	
	// Check root file
	rootFile := filepath.Join(outputDir, fmt.Sprintf("%s.root", prefix))
	if _, err := os.Stat(rootFile); os.IsNotExist(err) {
		return fmt.Errorf("root file not found: %s", rootFile)
	}
	
	return nil
}

// LoadScarResult loads SCAR results from output files (for testing/validation)
func LoadScarResult(outputDir, prefix string) (*ScarResult, error) {
	result := &ScarResult{
		Levels:           make([]LevelInfo, 0),
		FinalCommunities: make(map[string]int),
		HierarchyLevels:  make([]map[string][]string, 0),
		MappingLevels:    make([]map[string][]string, 0),
	}
	
	// Load hierarchy files
	hierarchyDir := filepath.Join(outputDir, "hierarchy-output")
	if files, err := filepath.Glob(filepath.Join(hierarchyDir, fmt.Sprintf("%s_*.dat", prefix))); err == nil {
		for _, file := range files {
			level, hierarchy, err := loadHierarchyFile(file)
			if err != nil {
				return nil, err
			}
			
			// Ensure we have enough levels
			for len(result.HierarchyLevels) <= level {
				result.HierarchyLevels = append(result.HierarchyLevels, make(map[string][]string))
			}
			
			result.HierarchyLevels[level] = hierarchy
		}
	}
	
	// Load mapping files
	mappingDir := filepath.Join(outputDir, "mapping-output")
	if files, err := filepath.Glob(filepath.Join(mappingDir, fmt.Sprintf("%s_*.dat", prefix))); err == nil {
		for _, file := range files {
			level, mapping, err := loadMappingFile(file)
			if err != nil {
				return nil, err
			}
			
			// Ensure we have enough levels
			for len(result.MappingLevels) <= level {
				result.MappingLevels = append(result.MappingLevels, make(map[string][]string))
			}
			
			result.MappingLevels[level] = mapping
		}
	}
	
	result.NumLevels = len(result.HierarchyLevels)
	
	return result, nil
}

// Helper function to load hierarchy file
func loadHierarchyFile(filename string) (int, map[string][]string, error) {
	// Extract level from filename
	base := filepath.Base(filename)
	parts := strings.Split(base, "_")
	if len(parts) < 2 {
		return 0, nil, fmt.Errorf("invalid hierarchy filename format: %s", filename)
	}
	
	levelStr := strings.TrimSuffix(parts[len(parts)-1], ".dat")
	level, err := strconv.Atoi(levelStr)
	if err != nil {
		return 0, nil, fmt.Errorf("invalid level in filename: %s", filename)
	}
	
	file, err := os.Open(filename)
	if err != nil {
		return 0, nil, err
	}
	defer file.Close()
	
	hierarchy := make(map[string][]string)
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		supernode := strings.TrimSpace(scanner.Text())
		if supernode == "" {
			continue
		}
		
		// Read number of children
		if !scanner.Scan() {
			break
		}
		numChildren, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
		if err != nil {
			return 0, nil, fmt.Errorf("invalid number of children: %v", err)
		}
		
		// Read children
		children := make([]string, 0, numChildren)
		for i := 0; i < numChildren && scanner.Scan(); i++ {
			child := strings.TrimSpace(scanner.Text())
			children = append(children, child)
		}
		
		hierarchy[supernode] = children
	}
	
	return level, hierarchy, scanner.Err()
}

// Helper function to load mapping file
func loadMappingFile(filename string) (int, map[string][]string, error) {
	// Extract level from filename (same logic as hierarchy)
	base := filepath.Base(filename)
	parts := strings.Split(base, "_")
	if len(parts) < 2 {
		return 0, nil, fmt.Errorf("invalid mapping filename format: %s", filename)
	}
	
	levelStr := strings.TrimSuffix(parts[len(parts)-1], ".dat")
	level, err := strconv.Atoi(levelStr)
	if err != nil {
		return 0, nil, fmt.Errorf("invalid level in filename: %s", filename)
	}
	
	file, err := os.Open(filename)
	if err != nil {
		return 0, nil, err
	}
	defer file.Close()
	
	mapping := make(map[string][]string)
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		supernode := strings.TrimSpace(scanner.Text())
		if supernode == "" {
			continue
		}
		
		// Read number of leafs
		if !scanner.Scan() {
			break
		}
		numLeafs, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
		if err != nil {
			return 0, nil, fmt.Errorf("invalid number of leafs: %v", err)
		}
		
		// Read leaf nodes
		leafs := make([]string, 0, numLeafs)
		for i := 0; i < numLeafs && scanner.Scan(); i++ {
			leaf := strings.TrimSpace(scanner.Text())
			leafs = append(leafs, leaf)
		}
		
		mapping[supernode] = leafs
	}
	
	return level, mapping, scanner.Err()
}
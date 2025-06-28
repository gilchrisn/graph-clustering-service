package materialization
import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"github.com/gilchrisn/graph-clustering-service/pkg/models"
)

// ParseSCARInput parses SCAR-format input files and creates a heterogeneous graph
func ParseSCARInput(graphFile, propertiesFile, pathFile string) (*models.HeterogeneousGraph, *models.MetaPath, error) {
	// Parse properties first to understand node types
	nodeTypes, err := parsePropertiesFile(propertiesFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse properties: %w", err)
	}
	
	// Parse graph edges
	graph, err := parseGraphFile(graphFile, nodeTypes)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse graph: %w", err)
	}
	
	// Parse meta path
	metaPath, err := parsePathFile(pathFile, nodeTypes)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse path: %w", err)
	}
	
	return graph, metaPath, nil
}

// parsePropertiesFile reads the properties file and returns node ID -> type mapping
func parsePropertiesFile(filename string) (map[string]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	nodeTypes := make(map[string]string)
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid properties line: %s", line)
		}
		
		nodeID := parts[0]
		typeID := parts[1]
		
		// Convert type ID to type name
		typeName := fmt.Sprintf("Type_%s", typeID)
		nodeTypes[nodeID] = typeName
	}
	
	return nodeTypes, scanner.Err()
}

// parseGraphFile reads the graph file and creates edges
func parseGraphFile(filename string, nodeTypes map[string]string) (*models.HeterogeneousGraph, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	graph := &models.HeterogeneousGraph{
		Nodes: make(map[string]models.Node),
		Edges: make([]models.Edge, 0),
	}
	
	// Create nodes from nodeTypes
	for nodeID, nodeType := range nodeTypes {
		graph.Nodes[nodeID] = models.Node{
			ID:         nodeID,
			Type:       nodeType,
			Properties: make(map[string]interface{}),
		}
	}
	
	// Parse edges
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) < 2 {
			return nil, fmt.Errorf("invalid edge line: %s", line)
		}
		
		fromNode := parts[0]
		toNode := parts[1]
		weight := 1.0
		
		// Parse weight if provided
		if len(parts) >= 3 {
			if w, err := strconv.ParseFloat(parts[2], 64); err == nil {
				weight = w
			}
		}
		
		// Determine edge type based on node types
		fromType := nodeTypes[fromNode]
		toType := nodeTypes[toNode]
		edgeType := fmt.Sprintf("%s_to_%s", fromType, toType)
		
		graph.Edges = append(graph.Edges, models.Edge{
			From:   fromNode,
			To:     toNode,
			Type:   edgeType,
			Weight: weight,
		})
	}
	
	return graph, scanner.Err()
}

// parsePathFile reads the path file and creates a meta path
func parsePathFile(filename string, nodeTypes map[string]string) (*models.MetaPath, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	
	var pathTypes []string
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		
		typeID := line
		typeName := fmt.Sprintf("Type_%s", typeID)
		pathTypes = append(pathTypes, typeName)
	}
	
	if len(pathTypes) < 2 {
		return nil, fmt.Errorf("path must have at least 2 node types")
	}
	
	// Create edge sequence from node sequence
	edgeSequence := make([]string, len(pathTypes)-1)
	for i := 0; i < len(pathTypes)-1; i++ {
		edgeSequence[i] = fmt.Sprintf("%s_to_%s", pathTypes[i], pathTypes[i+1])
	}
	
	metaPath := &models.MetaPath{
		ID:           "scar_path",
		NodeSequence: pathTypes,
		EdgeSequence: edgeSequence,
	}
	
	return metaPath, nil
}



// SCARToMaterialization runs the complete conversion from SCAR input to edge list output
func SCARToMaterialization(graphFile, propertiesFile, pathFile, outputFile string) error {
	// Parse SCAR input
	graph, metaPath, err := ParseSCARInput(graphFile, propertiesFile, pathFile)
	if err != nil {
		return fmt.Errorf("failed to parse SCAR input: %w", err)
	}
	
	// Run materialization
	config := DefaultMaterializationConfig()
	config.Aggregation.Strategy = Count
	config.Aggregation.Symmetric = true
	
	engine := NewMaterializationEngine(graph, metaPath, config, nil)
	result, err := engine.Materialize()
	if err != nil {
		return fmt.Errorf("materialization failed: %w", err)
	}
	
	// Save as simple edge list
	err = SaveAsSimpleEdgeList(result.HomogeneousGraph, outputFile)
	if err != nil {
		return fmt.Errorf("failed to save edge list: %w", err)
	}
	
	fmt.Printf("Converted SCAR input to edge list: %s\n", outputFile)
	fmt.Printf("Nodes: %d, Edges: %d\n", 
		len(result.HomogeneousGraph.Nodes), 
		len(result.HomogeneousGraph.Edges))
	
	return nil
}
# Validation Module

The validation module provides comprehensive validation for heterogeneous graphs and meta paths, ensuring data integrity before processing.

## Features

- Load and validate heterogeneous graphs from JSON files
- Load and validate meta paths from JSON files  
- Check compatibility between graphs and meta paths
- Comprehensive error reporting with specific field-level validation

## Usage

### Basic Validation

```go
package main

import (
    "fmt"
    "log"
    "github.com/gilchrisn/graph-clustering-service/pkg/validation"
)

func main() {
    // Load and validate graph
    graph, err := validation.LoadAndValidateGraph("data/graph_input.json")
    if err != nil {
        log.Fatalf("Graph validation failed: %v", err)
    }
    
    // Load and validate meta path
    metaPath, err := validation.LoadAndValidateMetaPath("data/meta_path.json")
    if err != nil {
        log.Fatalf("Meta path validation failed: %v", err)
    }
    
    // Check compatibility
    err = validation.ValidateMetaPathAgainstGraph(metaPath, graph)
    if err != nil {
        log.Fatalf("Incompatible: %v", err)
    }
    
    fmt.Println("✓ All validations passed!")
}
```

### Validation Functions

#### `LoadAndValidateGraph(filePath string) (*models.HeterogeneousGraph, error)`
- Loads graph from JSON file
- Validates node and edge structure
- Checks for isolated nodes and invalid references
- Returns populated graph with type maps

#### `LoadAndValidateMetaPath(filePath string) (*models.MetaPath, error)`
- Loads meta path from JSON file
- Validates sequence lengths match
- Ensures non-empty IDs and descriptions

#### `ValidateMetaPathAgainstGraph(metaPath, graph) error`
- Checks all node types in meta path exist in graph
- Checks all edge types in meta path exist in graph
- Validates each transition in meta path is possible in graph

#### `ValidateGraphStructure(graph *models.HeterogeneousGraph) error`
- Comprehensive structural validation
- Node and edge consistency checks
- Connectivity analysis

## Input Formats

### Graph JSON Structure
```json
{
  "nodes": {
    "node_id": {
      "type": "NodeType",
      "property1": "value1",
      "property2": "value2"
    }
  },
  "edges": [
    {
      "from": "source_node_id",
      "to": "target_node_id", 
      "type": "EdgeType",
      "weight": 1.0
    }
  ]
}
```

### Meta Path JSON Structure
```json
{
  "id": "unique_identifier",
  "node_sequence": ["NodeType1", "NodeType2", "NodeType3"],
  "edge_sequence": ["EdgeType1", "EdgeType2"],
  "description": "Human readable description"
}
```

## Validation Rules

### Graph Validation
- ✅ At least one node and one edge
- ✅ All edge references point to existing nodes
- ✅ No negative edge weights
- ✅ No self-loops
- ✅ No duplicate edges
- ✅ At least 2 different node types (heterogeneous requirement)
- ✅ No isolated nodes

### Meta Path Validation
- ✅ Non-empty ID
- ✅ At least 2 nodes in sequence
- ✅ Edge sequence length = node sequence length - 1
- ✅ All node/edge types exist in target graph
- ✅ Each transition is possible in graph

## Error Handling

The module uses structured error types:

```go
type ValidationError struct {
    Field   string // Which field failed validation
    Message string // What went wrong
    Value   string // The problematic value (optional)
}

type ValidationErrors []ValidationError // Multiple errors
```

Example error output:
```
validation error in field 'edge[2].from': from node does not exist in graph (value: nonexistent_node)
```

## Testing

```bash
# Run validation tests
go test ./pkg/validation/

# Run with verbose output
go test -v ./pkg/validation/

# Run benchmarks
go test -bench=. ./pkg/validation/
```

## Best Practices

1. **Always validate before processing**: Run validation before materialization or SCAR
2. **Handle validation errors gracefully**: Check specific error fields for targeted fixes
3. **Use appropriate file formats**: Ensure JSON files are properly formatted
4. **Verify meta path compatibility**: Always check meta path works with your specific graph
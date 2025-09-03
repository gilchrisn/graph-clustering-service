# Input/Output API Documentation

## Input Processing

### Edge List Format

The parser accepts simple edge list files with the following format:

```
# Comments start with # and are ignored
from_node to_node [weight]
```

**Format Specifications:**
- **Node IDs**: Can be strings or integers (e.g., "1", "node_a", "user123")
- **Weight**: Optional floating point number (defaults to 1.0 if omitted)
- **Self-loops**: Not supported - edges where `from_node == to_node` are automatically skipped
- **Comments**: Lines starting with `#` are ignored
- **Empty lines**: Ignored

**Example Input File:**
```
# Social network edges
1 2 1.5
2 3 2.0
1 3
4 5 0.8
# This is a comment
3 4 1.2
```

### Node Normalization

The parser automatically converts original node IDs to normalized integer indices (0, 1, 2, ...) for internal processing:

1. **Collection Phase**: All unique node IDs are collected from the input
2. **Sorting Phase**: Node IDs are sorted for consistent ordering:
   - If all nodes are integers: sorted numerically (1, 2, 10, 20)
   - Otherwise: sorted lexicographically ("a", "b", "node1", "node2")
3. **Mapping Phase**: Creates bidirectional mappings between original IDs and indices

### Parsing Functions

```go
// Create parser
parser := NewGraphParser()

// Parse edge list file
result, err := parser.ParseEdgeList("input.txt")
if err != nil {
    return fmt.Errorf("parsing failed: %w", err)
}

graph := result.Graph   // NormalizedGraph with integer indices
parser := result.Parser // Contains ID mappings for output conversion
```

### Manual Graph Construction

```go
// Create empty graph with 5 nodes (indices 0-4)
graph := NewNormalizedGraph(5)

// Add edges using normalized indices
graph.AddEdge(0, 1, 2.5)  // Node 0 to Node 1, weight 2.5
graph.AddEdge(1, 2, 1.0)  // Node 1 to Node 2, weight 1.0

// Validate graph structure
if err := graph.Validate(); err != nil {
    return fmt.Errorf("invalid graph: %w", err)
}
```

## Output Generation

### Generated Files

The output writer creates a hierarchical community structure across multiple files:

```go
writer := NewFileWriter()
err := writer.WriteAll(result, parser, "output_dir", "prefix")
```

This generates:
- `prefix.mapping` - Community membership mappings
- `prefix.hierarchy` - Hierarchical structure relationships  
- `prefix.root` - Root community identifier
- `prefix.edges` - Inter-community connections

### Community ID Format

Communities are identified using the pattern: `c0_l{level}_{index}`

- `c0` - Fixed prefix
- `l{level}` - Level in hierarchy (l1, l2, l3, ...)
- `{index}` - Community index at that level (0, 1, 2, ...)

**Examples:**
- `c0_l1_0` - First community at level 1
- `c0_l2_3` - Fourth community at level 2

### File Format Specifications

#### 1. Mapping File (`prefix.mapping`)

Maps each community to its member nodes using original node IDs:

```
community_id
node_count
original_node_id_1
original_node_id_2
...
community_id
node_count
original_node_id_1
...
```

**Example:**
```
c0_l1_0
3
user_1
user_2
user_5
c0_l1_1
2
user_3
user_4
```

#### 2. Hierarchy File (`prefix.hierarchy`)

Shows parent-child relationships between communities across levels:

```
parent_community_id
child_count
child_community_index_1
child_community_index_2
...
```

**Example:**
```
c0_l2_0
2
0
1
c0_l2_1
1
2
```

This means:
- Community `c0_l2_0` contains child communities 0 and 1 from level 1
- Community `c0_l2_1` contains child community 2 from level 1

#### 3. Root File (`prefix.root`)

Contains the single root community ID that encompasses the entire graph:

```
root_community_id
```

**Example:**
```
c0_l3_0
```

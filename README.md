# SCAR-based Heterogeneous Graph Clustering Service

This project implements two approaches for community detection in heterogeneous graphs:

1. **Materialization + Louvain**: Traditional approach that materializes the heterogeneous graph into a homogeneous graph, then applies Louvain algorithm
2. **SCAR**: Sketch-based approach that works directly on heterogeneous graphs without full materialization

## Architecture Overview

```
Heterogeneous Graph Input
         │
         ▼
    [Validation]
         │
    ┌────┴────┐
    ▼         ▼
[Materialization] [SCAR]
    │         │
    ▼         │
[Louvain]     │
    │         │
    ▼         ▼
Communities Communities
```

## Quick Start

```bash
# Setup and run comparison
make quickstart
make compare

# Run individual pipelines
make validate      # Just validation
make materialize   # Validation + materialization + Louvain  
make scar          # Validation + SCAR

# Development
make test          # Run all tests
make benchmark     # Performance benchmarks
```

## Example Usage

```bash
# Compare both approaches on your data
go run comparison_main.go data/graph_input.json data/meta_path.json

# Run individual approaches
go run main.go materialize data/graph_input.json data/meta_path.json
go run main.go validate data/graph_input.json data/meta_path.json
```

## Modules

### 1. [Validation Module](pkg/validation/README.md)
- Loads and validates heterogeneous graphs from JSON
- Loads and validates meta paths
- Ensures compatibility between graphs and meta paths

### 2. [Materialization Module](pkg/materialization/README.md) 
- Converts heterogeneous graphs to homogeneous graphs using meta paths
- Supports multiple aggregation strategies (Count, Sum, Average, etc.)
- Two interpretation modes: DirectTraversal and MeetingBased

### 3. [Louvain Module](pkg/louvain/README.md)
- Traditional hierarchical community detection
- Works on homogeneous graphs (output of materialization)
- Multi-level optimization with modularity maximization

### 4. [SCAR Module](pkg/scar/README.md)
- Sketch-based community detection for heterogeneous graphs
- Works directly on heterogeneous graphs without full materialization
- Uses Bottom-K sketches for efficient similarity estimation

## Example Data Format

### Graph Input (`data/graph_input.json`)
```json
{
  "nodes": {
    "a1": {"type": "Author", "name": "Alice"},
    "a2": {"type": "Author", "name": "Bob"},
    "p1": {"type": "Paper", "title": "ML Paper"},
    "v1": {"type": "Venue", "name": "ICML"}
  },
  "edges": [
    {"from": "a1", "to": "p1", "type": "writes", "weight": 1.0},
    {"from": "a2", "to": "p1", "type": "writes", "weight": 1.0},
    {"from": "p1", "to": "v1", "type": "published_in", "weight": 1.0}
  ]
}
```

### Meta Path (`data/meta_path.json`)
```json
{
  "id": "author_coauthorship",
  "node_sequence": ["Author", "Paper", "Author"],
  "edge_sequence": ["writes", "writes"],
  "description": "Authors connected through co-authored papers"
}
```

## Performance Comparison

| Aspect | Materialization + Louvain | SCAR |
|--------|---------------------------|------|
| **Memory Usage** | High (full edge materialization) | Low (sketch-based) |
| **Runtime** | O(m²) for dense meta paths | O(k×n) where k << m |
| **Accuracy** | Exact | Approximation with high accuracy |
| **Scalability** | Limited by memory | Scales to large graphs |
| **Graph Support** | Any via materialization | Heterogeneous native |

## Build and Test

```bash
# Build all modules
make build

# Run tests
make test

# Run benchmarks
make benchmark

# Create example data
make example

# Full development workflow
make dev
```

## Dependencies

- Go 1.19+
- No external dependencies (pure Go implementation)

## Contributing

1. Each module has its own tests and benchmarks
2. Follow the established patterns for error handling
3. Update documentation when adding features
4. Run `make check` before submitting PRs

## License

MIT License - see LICENSE file for details.
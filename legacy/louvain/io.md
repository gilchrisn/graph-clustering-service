# Louvain Algorithm - Input/Output Format Specification

This document defines the standard input and output formats for the Louvain community detection algorithm implementation. Use this as a reference to ensure compatibility when comparing with other community detection methods.

## Input Format

### Edge List File Format

The algorithm accepts **edge list files** in plain text format.

#### Basic Format
```
from_node to_node [weight]
```

- **`from_node`**: Source node identifier
- **`to_node`**: Target node identifier  
- **`weight`**: Optional edge weight (defaults to 1.0 if omitted)
- **Separator**: Whitespace (spaces or tabs)

#### Node Identifiers
Node IDs can be:
- **Strings**: `alice`, `bob`, `user_123`, `group_a`
- **Numbers**: `1`, `2`, `42`, `1001`
- **Mixed**: `user1`, `node_42`, `cluster_a`

#### Edge Weight Rules
- **Optional**: If weight is omitted, defaults to `1.0`
- **Format**: Floating point numbers (`1.0`, `2.5`, `0.8`)
- **Self-loops**: Allowed (e.g., `alice alice 1.5`)
- **Undirected**: The algorithm treats all edges as undirected

#### Comments and Empty Lines
- **Comments**: Lines starting with `#` are ignored
- **Empty lines**: Ignored
- **Whitespace**: Leading/trailing whitespace is trimmed

### Example Input Files

#### Example 1: Social Network (String IDs with Weights)
```
# Social network connections
alice bob 2.5
bob charlie 1.0
alice charlie 1.8
david eve 3.2
eve alice 0.8
charlie david 1.2
```

#### Example 2: Simple Graph (Numeric IDs, No Weights)
```
# Simple graph - all edges have weight 1.0
1 2
2 3
1 3
4 5
5 1
3 4
```

#### Example 3: Mixed Format with Self-loops
```
# Mixed node types with self-loops
user1 user2 0.8
user2 group_a 1.5
group_a user1 2.0
user1 user1 1.0
brand_x user2 0.7
```

#### Example 4: Protein Interaction Network
```
# Protein interaction weights
PROT_001 PROT_002 0.85
PROT_002 PROT_003 0.92
PROT_001 PROT_003 0.76
PROT_004 PROT_005 0.88
PROT_003 PROT_004 0.71
```

### Input File Requirements

- **Encoding**: UTF-8
- **Line endings**: Unix (`\n`) or Windows (`\r\n`)
- **File size**: No explicit limits (memory dependent)
- **Node count**: No explicit limits (memory dependent)
- **Edge count**: No explicit limits (memory dependent)

---

## Output Format

The algorithm generates multiple output files with different levels of detail.

### Output Files Overview

Given prefix `communities`, the following files are created:

1. **`communities.mapping`** - Community membership
2. **`communities.hierarchy`** - Hierarchical structure  
3. **`communities.root`** - Top-level communities
4. **`communities.edges`** - Inter-community connections
5. **`simple_mapping.txt`** - Simple node→community mapping

### 1. Community Mapping File (`prefix.mapping`)

Contains the final community assignments with original node IDs.

#### Format
```
c0_l{level}_{community_id}
{node_count}
{node_id_1}
{node_id_2}
...
{node_id_n}
```

#### Example
```
c0_l2_0
3
alice
bob
charlie
c0_l2_1
2
david
eve
```

**Explanation:**
- `c0_l2_0`: Community identifier (level 2, community 0)
- `3`: Number of nodes in this community
- `alice`, `bob`, `charlie`: Original node IDs in this community

### 2. Hierarchy File (`prefix.hierarchy`)

Shows how communities are nested within higher-level communities.

#### Format
```
c0_l{level}_{community_id}
{sub_community_count}
{sub_community_id_1}
{sub_community_id_2}
...
{sub_community_id_n}
```

#### Example
```
c0_l2_0
2
0
1
c0_l2_1
1
2
```

**Explanation:**
- `c0_l2_0`: Higher-level community
- `2`: Contains 2 sub-communities from previous level
- `0`, `1`: Sub-community IDs from level 1

### 3. Root Communities File (`prefix.root`)

Lists all top-level communities from the final level.

#### Format
```
c0_l{final_level}_{community_id_1}
c0_l{final_level}_{community_id_2}
...
```

#### Example
```
c0_l2_0
c0_l2_1
```

### 4. Community Edges File (`prefix.edges`)

Shows edges between communities at each level.

#### Format
```
c0_l{level}_{comm1} c0_l{level}_{comm2}
...
```

#### Example
```
c0_l1_0 c0_l1_1
c0_l1_1 c0_l1_2
c0_l2_0 c0_l2_1
```

### 5. Simple Mapping File (`simple_mapping.txt`)

A simplified node→community mapping for easy processing.

#### Format
```
{original_node_id} {community_id}
```

#### Example
```
alice 0
bob 0
charlie 0
david 1
eve 1
```

---

## Complete Example

### Input File (`social_network.txt`)
```
# Social media connections
alice bob 2.5
bob charlie 1.0
alice charlie 1.8
david eve 3.2
eve frank 2.1
alice david 0.5
```

### Algorithm Execution
```bash
./louvain -input social_network.txt -output results -prefix social
```

### Output Files

#### `social.mapping`
```
c0_l2_0
3
alice
bob
charlie
c0_l2_1
3
david
eve
frank
```

#### `social.hierarchy`
```
c0_l2_0
2
0
1
c0_l2_1
1
2
```

#### `social.root`
```
c0_l2_0
c0_l2_1
```

#### `social.edges`
```
c0_l1_0 c0_l1_1
c0_l1_1 c0_l1_2
c0_l2_0 c0_l2_1
```

#### `simple_mapping.txt`
```
alice 0
bob 0
charlie 0
david 1
eve 1
frank 1
```

---

## Implementation Notes

### Node ID Handling
- **Normalization**: Original node IDs are mapped to integer indices (0, 1, 2, ...)
- **Sorting**: Node IDs are sorted (numerically if all numeric, alphabetically otherwise)
- **Mapping**: Bidirectional mappings maintain original→normalized and normalized→original
- **Output**: All output files use original node IDs

### Community ID Assignment
- **Level numbering**: Starts from 1 (level 1, level 2, etc.)
- **Community numbering**: Starts from 0 within each level
- **Hierarchical tracking**: Community IDs are consistent across hierarchy files

### Edge Weight Handling
- **Self-loops**: Counted twice in degree calculation (standard graph theory)
- **Undirected**: All edges treated as undirected regardless of input order
- **Aggregation**: Edge weights are summed when creating super-graphs

### Algorithm Parameters
- **Modularity threshold**: Minimum improvement threshold (default: 0.001)
- **Maximum iterations**: Per-level iteration limit (default: 1)
- **Random seed**: For reproducible results (default: 42)

---

## Validation and Testing

### Input Validation
To validate your input file:
1. Check edge list format compliance
2. Verify node ID consistency  
3. Ensure weight values are valid numbers
4. Test with small sample files

### Output Validation
To validate algorithm output:
1. Verify all original node IDs appear in mapping files
2. Check hierarchy consistency between levels
3. Ensure community assignments are complete
4. Validate simple mapping totals

### Baseline Comparison
When comparing with other community detection algorithms:
1. Use identical input files
2. Convert other outputs to match this format
3. Compare simple mapping files for equivalence
4. Validate modularity scores
5. Check hierarchical structure if supported

---

## File Size Estimates

| Graph Size | Nodes | Edges | Input Size | Output Size |
|------------|-------|-------|------------|-------------|
| Small      | 100   | 500   | ~15 KB     | ~5 KB       |
| Medium     | 1K    | 10K   | ~200 KB    | ~50 KB      |
| Large      | 10K   | 100K  | ~2 MB      | ~500 KB     |
| Very Large | 100K  | 1M    | ~20 MB     | ~5 MB       |

*Estimates assume average node ID length of 8 characters and default community structure.*
# Graph Visualization Backend API

> **Complete graph clustering with coordinate generation, edge extraction, and multi-experiment comparison**

## 🚀 Quick Start

**Base URL:** `http://localhost:8080/api/v1`

### 1. Upload Dataset
```javascript
const formData = new FormData();
formData.append('name', 'My Graph Dataset');
formData.append('graphFile', graphFile);        // Edge list file
formData.append('propertiesFile', propertyFile); // Node types file  
formData.append('pathFile', pathFile);           // Meta-path file

const response = await fetch('/api/v1/datasets', {
  method: 'POST',
  body: formData
});

const result = await response.json();
const datasetId = result.data.datasetId;
```

### 2. Start Clustering with Coordinate Generation
```javascript
const response = await fetch(`/api/v1/datasets/${datasetId}/clustering`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    algorithm: 'scar', // or 'louvain'
    parameters: {
      // SCAR parameters
      k: 64,
      nk: 4,
      threshold: 0.5,
      // 🆕 NEW: SCAR reconstruction parameters
      reconstructionThreshold: 0.1,
      reconstructionMode: "inclusion_exclusion", // or "full"
      edgeWeightNormalization: true,
      
      // Louvain parameters (if using louvain)
      maxLevels: 5,
      maxIterations: 100,
      minModularityGain: 0.000001
    }
  })
});

const result = await response.json();
const jobId = result.data.jobId;
```

### 3. Monitor Progress
```javascript
const pollJobStatus = async () => {
  const response = await fetch(`/api/v1/datasets/${datasetId}/clustering/${jobId}`);
  const result = await response.json();
  
  if (result.data.status === 'completed') {
    console.log('Modularity:', result.data.result.modularity);
    return result.data;
  } else if (result.data.status === 'failed') {
    throw new Error(result.data.error);
  } else {
    // Still running, check again in 2 seconds
    setTimeout(pollJobStatus, 2000);
  }
};

await pollJobStatus();
```

### 4. Get Complete Visualization Data
```javascript
// Get hierarchy with coordinates
const hierarchyResponse = await fetch(`/api/v1/datasets/${datasetId}/hierarchy?jobId=${jobId}`);
const hierarchyResult = await hierarchyResponse.json();
const hierarchy = hierarchyResult.data.hierarchy;

// Get cluster details with edges
const clusterResponse = await fetch(`/api/v1/datasets/${datasetId}/clusters/c0_l0_1/nodes?jobId=${jobId}`);
const clusterResult = await clusterResponse.json();
const cluster = clusterResult.data;

console.log(`Cluster ${cluster.clusterId}:`);
console.log(`- ${cluster.nodes.length} nodes`);
console.log(`- ${cluster.edges.length} edges`);
console.log(`- Coordinates available: ${Object.keys(hierarchy.coordinates).length} nodes`);
```

### 5. 🆕 Compare Multiple Experiments
```javascript
// Compare multiple SCAR configurations against Louvain baseline
const multiComparisonResponse = await fetch('/api/v1/comparisons/multi', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: "SCAR Parameter Study vs Louvain",
    selectedExperiments: [
      { datasetId: datasetId, jobId: scarK32JobId },
      { datasetId: datasetId, jobId: scarK64JobId },
      { datasetId: datasetId, jobId: scarK128JobId }
    ],
    metrics: ["hmi", "custom_leaf_metric", "custom_displayed_metric"],
    ensureLouvainBaseline: true // Automatically creates standard Louvain baseline
  })
});

const multiResult = await multiComparisonResponse.json();
const comparisonId = multiResult.data.comparisonId;

// Get multi-comparison results
const resultResponse = await fetch(`/api/v1/comparisons/${comparisonId}`);
const result = await resultResponse.json();

console.log('Multi-Comparison Results:');
console.log(`- Baseline: ${result.data.result.baselineConfig.description}`);
result.data.result.experiments.forEach(exp => {
  console.log(`- ${exp.label}: HMI=${exp.metrics.hmi.toFixed(3)}, Custom=${exp.metrics.custom_leaf_metric.toFixed(3)}`);
});
```

### 6. Legacy Two-Experiment Comparison (Still Available)
```javascript
// Compare two specific experiments
const comparisonResponse = await fetch('/api/v1/comparisons', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: "Louvain vs SCAR Direct Comparison",
    experimentA: { datasetId: datasetId, jobId: louvainJobId },
    experimentB: { datasetId: datasetId, jobId: scarJobId },
    metrics: ["agds", "jaccard", "hmi", "ari"],
    options: { levelWise: true, includeVisualization: true }
  })
});
```

## 📡 API Reference

### Datasets

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/datasets` | POST | Upload new dataset |
| `/datasets` | GET | List all datasets |
| `/datasets/{id}` | GET | Get dataset details |
| `/datasets/{id}` | DELETE | Delete dataset |

### Clustering Jobs

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/datasets/{id}/clustering` | POST | Start clustering job with coordinate generation |
| `/datasets/{id}/clustering/{jobId}` | GET | Get job status |
| `/datasets/{id}/clustering/{jobId}` | DELETE | Cancel job |

### Results

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/datasets/{id}/hierarchy?jobId={jobId}` | GET | Get full hierarchy with coordinates |
| `/datasets/{id}/hierarchy/levels/{level}?jobId={jobId}` | GET | Get specific level |
| `/datasets/{id}/clusters/{clusterId}/nodes?jobId={jobId}` | GET | **Get cluster nodes and edges** |

### 🆕 Multi-Experiment Comparisons

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/comparisons/multi` | POST | **🆕 Compare multiple experiments vs baseline** |
| `/comparisons/{comparisonId}` | GET | **Get comparison results** |
| `/comparisons/{comparisonId}` | DELETE | **Delete comparison** |
| `/comparisons` | GET | **List all comparisons** |

### Legacy Two-Experiment Comparisons

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/comparisons` | POST | **Compare two specific experiments (legacy)** |

### System

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/health` | GET | Service health check |
| `/algorithms` | GET | Available algorithms |

## 📋 Data Formats

### Upload Request (multipart/form-data)
```
name: string           // Dataset display name
graphFile: file        // Edge list: "source target [weight]"
propertiesFile: file   // Node types: "nodeId typeId"  
pathFile: file         // Meta-path: "typeId" (one per line)
```

### Clustering Request
```json
{
  "algorithm": "louvain|scar",
  "parameters": {
    // Louvain parameters
    "maxLevels": 10,
    "maxIterations": 100, 
    "minModularityGain": 0.000001,
    
    // SCAR parameters  
    "k": 64,
    "nk": 4,
    "threshold": 0.5,
    
    // 🆕 NEW: SCAR reconstruction parameters
    "reconstructionThreshold": 0.1,     // 0.0-1.0, edge inclusion threshold
    "reconstructionMode": "inclusion_exclusion", // "inclusion_exclusion" | "full"
    "edgeWeightNormalization": true     // normalize weights during reconstruction
  }
}
```

### 🆕 Multi-Experiment Comparison Request
```json
{
  "name": "SCAR Parameter Study",
  "selectedExperiments": [
    { "datasetId": "uuid-dataset-1", "jobId": "uuid-scar-k32" },
    { "datasetId": "uuid-dataset-1", "jobId": "uuid-scar-k64" },
    { "datasetId": "uuid-dataset-1", "jobId": "uuid-scar-k128" }
  ],
  "metrics": ["hmi", "custom_leaf_metric", "custom_displayed_metric"],
  "ensureLouvainBaseline": true // Creates standard Louvain baseline if missing
}
```

### Legacy Two-Experiment Comparison Request
```json
{
  "name": "Direct Algorithm Comparison",
  "experimentA": {
    "datasetId": "uuid-dataset-1", 
    "jobId": "uuid-job-a"
  },
  "experimentB": {
    "datasetId": "uuid-dataset-1",
    "jobId": "uuid-job-b"
  },
  "metrics": ["agds", "jaccard", "hmi", "ari"],
  "options": {
    "levelWise": true,
    "includeVisualization": true
  }
}
```

### 🆕 Multi-Comparison Response
```json
{
  "success": true,
  "message": "Multi-comparison started - will use standard Louvain baseline (maxLevels:10, maxIterations:100, minModularityGain:1e-6)",
  "data": {
    "comparisonId": "comparison-uuid",
    "comparison": {
      "id": "comparison-uuid",
      "name": "SCAR Parameter Study",
      "status": "completed",
      "result": {
        "baselineJobId": "louvain-baseline-uuid",
        "baselineConfig": {
          "algorithm": "louvain",
          "parameters": {
            "maxLevels": 10,
            "maxIterations": 100,
            "minModularityGain": 1e-6
          },
          "description": "Standard Louvain Baseline (industry default parameters)",
          "isStandard": true
        },
        "experiments": [
          {
            "jobId": "uuid-scar-k32",
            "label": "SCAR k=32",
            "metrics": {
              "hmi": 0.756,
              "custom_leaf_metric": 0.823,
              "custom_displayed_metric": 0.734
            }
          },
          {
            "jobId": "uuid-scar-k64",
            "label": "SCAR k=64",
            "metrics": {
              "hmi": 0.791,
              "custom_leaf_metric": 0.845,
              "custom_displayed_metric": 0.782
            }
          }
        ]
      }
    }
  }
}
```

### Hierarchy Response with Coordinates
```json
{
  "success": true,
  "data": {
    "hierarchy": {
      "datasetId": "uuid",
      "jobId": "uuid", 
      "algorithm": "louvain",
      "levels": [
        {
          "level": 0,
          "communities": {
            "c0_l0_0": ["1", "5", "12"],  // Leaf nodes
            "c0_l0_1": ["2", "8", "15"]
          },
          "parentMap": {
            "c0_l0_0": "c0_l1_0"  // Parent community
          }
        }
      ],
      "rootNode": "c0_l3_0",
      "coordinates": {
        "1": {
          "x": -45.2,
          "y": 23.1,
          "radius": 8.5
        },
        "5": {
          "x": 67.8,
          "y": -12.4,
          "radius": 6.2
        },
        "c0_l1_0": {
          "x": 23.5,
          "y": 89.1,
          "radius": 15.7
        }
      }
    }
  }
}
```

### Legacy Comparison Response
```json
{
  "success": true,
  "data": {
    "id": "comparison-uuid",
    "name": "Louvain vs SCAR Test",
    "status": "completed",
    "result": {
      "agds": 0.756,
      "jaccard": 0.823,
      "hmi": 0.791,
      "ari": 0.645,
      "levelWise": [
        {
          "level": 0,
          "jaccard": 0.823,
          "ari": 0.645
        }
      ],
      "summary": {
        "overallSimilarity": "High",
        "keyDifferences": [],
        "recommendation": "Experiments show very similar clustering patterns. Consider using either approach."
      }
    }
  }
}
```

### Cluster Response with Nodes and Edges
```json
{
  "success": true,
  "data": {
    "clusterId": "c0_l0_1",
    "level": 0,
    "nodes": [
      {
        "id": "1",
        "label": "1",
        "position": {
          "x": -45.2,
          "y": 23.1,
          "radius": 8.5
        },
        "type": "leaf",
        "metadata": {
          "level": 0
        }
      }
    ],
    "edges": [
      {
        "source": "1",
        "target": "2",
        "weight": 1.0
      }
    ]
  }
}
```

### Job Status Response
```json
{
  "success": true,
  "data": {
    "id": "job-uuid",
    "status": "queued|running|completed|failed|cancelled",
    "progress": {
      "percentage": 75,
      "message": "Building hierarchy with coordinates..."
    },
    "result": {  // Only when status === "completed"
      "modularity": 0.845,
      "numLevels": 4,
      "numCommunities": 28,
      "processingTimeMS": 15430
    },
    "error": "Error message"  // Only when status === "failed"
  }
}
```

## 🎯 Metrics Explained

### 🆕 New Multi-Experiment Metrics

| Metric | Description | Range | Purpose |
|--------|-------------|-------|---------|
| `hmi` | Hierarchical Mutual Information | 0.0-1.0 | Hierarchy-wide similarity |
| `custom_leaf_metric` | Jaccard × AM/QM displacement × AM/QM radius | 0.0-1.0 | Entire leaf level visualization similarity |
| `custom_displayed_metric` | Same formula, largest community only | 0.0-1.0 | Focus area visualization similarity |

**AM/QM Formula:** Arithmetic Mean ÷ Quadratic Mean of Euclidean distances between matching nodes  
**Edge Case:** Returns 1.0 when AM = QM to avoid division by zero

### Legacy Two-Experiment Metrics

| Metric | Description | Range | Purpose |
|--------|-------------|-------|---------|
| `agds` | Average Geometric Distance Similarity | 0.0-1.0 | Position similarity between node coordinates |
| `jaccard` | Jaccard Index | 0.0-1.0 | Community overlap similarity |
| `hmi` | Hierarchical Mutual Information | 0.0-1.0 | Hierarchy structure similarity |
| `ari` | Adjusted Rand Index | 0.0-1.0 | Clustering agreement similarity |

## 🔧 Error Handling

All responses follow consistent format:
```json
{
  "success": false,
  "message": "Human-readable error",
  "error": "Technical error details"
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `400` - Bad request (validation errors)
- `500` - Internal server error

**🆕 Multi-Comparison Error Cases:**
- `400` - Less than 2 experiments selected
- `400` - More than 10 experiments selected
- `400` - Experiments from different datasets
- `400` - Invalid metric names
- `500` - Failed to create Louvain baseline

## 🎯 Frontend Integration

### 🆕 Multi-Comparison Hook
```javascript
import { useState } from 'react';

export const useMultiComparison = () => {
  const [loading, setLoading] = useState(false);
  
  const startMultiComparison = async (comparisonData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/comparisons/multi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(comparisonData)
      });
      
      const result = await response.json();
      if (!result.success) throw new Error(result.message);
      
      return result.data;
    } finally {
      setLoading(false);
    }
  };
  
  const getComparison = async (comparisonId) => {
    const response = await fetch(`/api/v1/comparisons/${comparisonId}`);
    const result = await response.json();
    if (!result.success) throw new Error(result.message);
    return result.data;
  };
  
  return { startMultiComparison, getComparison, loading };
};
```

### 🆕 Multi-Comparison Component
```javascript
const MultiComparisonResults = ({ comparisonId }) => {
  const [comparison, setComparison] = useState(null);
  
  useEffect(() => {
    const fetchComparison = async () => {
      const response = await fetch(`/api/v1/comparisons/${comparisonId}`);
      const result = await response.json();
      setComparison(result.data);
    };
    
    if (comparisonId) fetchComparison();
  }, [comparisonId]);

  if (!comparison?.result) return <div>Loading comparison...</div>;

  return (
    <div className="multi-comparison-results">
      <h2>{comparison.name}</h2>
      
      <div className="baseline-info">
        <h3>Baseline Configuration</h3>
        <p><strong>{comparison.result.baselineConfig.description}</strong></p>
        <p>Parameters: maxLevels={comparison.result.baselineConfig.parameters.maxLevels}, 
           maxIterations={comparison.result.baselineConfig.parameters.maxIterations}</p>
      </div>

      <div className="experiments-table">
        <h3>Experiment Results</h3>
        <table>
          <thead>
            <tr>
              <th>Configuration</th>
              <th>HMI</th>
              <th>Leaf Metric</th>
              <th>Displayed Metric</th>
            </tr>
          </thead>
          <tbody>
            {comparison.result.experiments.map(exp => (
              <tr key={exp.jobId}>
                <td>{exp.label}</td>
                <td>{(exp.metrics.hmi * 100).toFixed(1)}%</td>
                <td>{(exp.metrics.custom_leaf_metric * 100).toFixed(1)}%</td>
                <td>{(exp.metrics.custom_displayed_metric * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Add chart visualization here using your preferred chart library */}
    </div>
  );
};
```

### Updated SCAR Parameter Form
```javascript
const SCARParameterForm = ({ onSubmit }) => {
  const [params, setParams] = useState({
    k: 64,
    nk: 4,
    threshold: 0.5,
    // 🆕 NEW: Reconstruction parameters
    reconstructionThreshold: 0.1,
    reconstructionMode: 'inclusion_exclusion',
    edgeWeightNormalization: true
  });

  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      onSubmit({ algorithm: 'scar', parameters: params });
    }}>
      <div>
        <label>K (sketch size): {params.k}</label>
        <input type="range" min="16" max="256" step="16" 
               value={params.k} 
               onChange={(e) => setParams({...params, k: parseInt(e.target.value)})} />
      </div>
      
      <div>
        <label>NK (layers): {params.nk}</label>
        <input type="range" min="2" max="8" 
               value={params.nk}
               onChange={(e) => setParams({...params, nk: parseInt(e.target.value)})} />
      </div>
      
      <div>
        <label>Threshold: {params.threshold}</label>
        <input type="range" min="0" max="1" step="0.1" 
               value={params.threshold}
               onChange={(e) => setParams({...params, threshold: parseFloat(e.target.value)})} />
      </div>

      {/* 🆕 NEW: Reconstruction parameters */}
      <div>
        <label>Reconstruction Threshold: {params.reconstructionThreshold}</label>
        <input type="range" min="0" max="1" step="0.1" 
               value={params.reconstructionThreshold}
               onChange={(e) => setParams({...params, reconstructionThreshold: parseFloat(e.target.value)})} />
        <small>Higher values = more selective edge inclusion</small>
      </div>
      
      <div>
        <label>Reconstruction Mode:</label>
        <select value={params.reconstructionMode} 
                onChange={(e) => setParams({...params, reconstructionMode: e.target.value})}>
          <option value="inclusion_exclusion">Inclusion-Exclusion (Standard)</option>
          <option value="full">Full Reconstruction</option>
        </select>
      </div>
      
      <div>
        <label>
          <input type="checkbox" 
                 checked={params.edgeWeightNormalization}
                 onChange={(e) => setParams({...params, edgeWeightNormalization: e.target.checked})} />
          Normalize Edge Weights
        </label>
      </div>

      <button type="submit">Start SCAR Clustering</button>
    </form>
  );
};
```

### Legacy Two-Experiment Comparison Hook (Still Available)
```javascript
export const useComparison = () => {
  const [loading, setLoading] = useState(false);
  
  const startComparison = async (comparisonData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/comparisons', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(comparisonData)
      });
      
      const result = await response.json();
      if (!result.success) throw new Error(result.message);
      
      return result.data;
    } finally {
      setLoading(false);
    }
  };
  
  return { startComparison, loading };
};
```

## ✨ Features

### 🆕 Multi-Experiment Comparison
- **Batch Comparison**: Compare 2-10 experiments against standard Louvain baseline
- **Automatic Baseline**: Creates standard Louvain baseline if missing (maxLevels:10, maxIterations:100, minModularityGain:1e-6)
- **Custom Visualization Metrics**: 
  - `custom_leaf_metric`: Jaccard × AM/QM displacement × AM/QM radius (entire leaf level)
  - `custom_displayed_metric`: Same formula for largest community comparison
- **Transparency**: Full baseline configuration shown in results
- **Industry Standard**: Follows ML benchmarking best practices

### Enhanced SCAR Algorithm
- **🆕 Reconstruction Parameters**: Fine-tune graph reconstruction process
  - `reconstructionThreshold`: Edge inclusion selectivity (0.0-1.0)
  - `reconstructionMode`: Standard vs full reconstruction
  - `edgeWeightNormalization`: Weight normalization toggle

### Coordinate Generation
- **PageRank-Based Node Sizing**: Larger nodes = more important/central (radius 3.0-20.0)
- **MDS-Based Layout**: Optimal 2D positioning using shortest paths (-100 to +100 range)
- **Multi-Level Support**: Coordinates for both leaf nodes and supernodes

### Edge Extraction
- **Intra-Cluster Edges**: Edges between nodes within the same cluster
- **Algorithm-Agnostic**: Works with both Louvain and SCAR algorithms
- **Multi-Level**: Extract edges at any hierarchy level

### Legacy Two-Experiment Comparison (Still Available)
- **Direct Comparison**: Compare any two specific experiments
- **Multi-Metric Analysis**: AGDS, Jaccard, HMI, ARI similarity metrics
- **Level-Wise Analysis**: Compare hierarchies at each level
- **Coordinate Similarity**: AGDS measures position similarity

### Algorithm Support
- **Louvain**: Classic modularity optimization with graph materialization
- **SCAR**: Sketch-based clustering with enhanced reconstruction parameters
- **Unified Interface**: Same API for both algorithms

### Performance & Reliability
- **Background Processing**: Non-blocking job execution with progress tracking
- **Resource Management**: Worker pools and memory limits
- **Graceful Degradation**: Fallback handling for failed operations
- **Structured Logging**: Comprehensive monitoring and debugging

## 🔄 API Migration Guide

### 🆕 New Endpoints

| Endpoint | Method | Description | Status |
|----------|---------|-------------|---------|
| `/comparisons/multi` | POST | **🆕 Multi-experiment comparison** | New |

### Updated Endpoints

| Endpoint | Changes | Status |
|----------|---------|---------|
| `/datasets/{id}/clustering` | **🆕 Added SCAR reconstruction parameters** | Enhanced |

### Unchanged Endpoints

| Endpoint | Method | Description | Status |
|----------|---------|-------------|---------|
| `/datasets` | POST | Upload dataset | ✅ Unchanged |
| `/datasets/{id}/clustering` | POST | Start clustering | ✅ Enhanced |
| `/datasets/{id}/hierarchy?jobId={jobId}` | GET | Get hierarchy with coordinates | ✅ Unchanged |
| `/datasets/{id}/clusters/{clusterId}/nodes?jobId={jobId}` | GET | Get cluster nodes and edges | ✅ Unchanged |
| `/comparisons` | POST | Legacy two-experiment comparison | ✅ Unchanged |

## 📞 Support

- **Health Check:** `GET /api/v1/health`
- **Server Logs:** Check console for structured logs
- **Algorithm Info:** `GET /api/v1/algorithms`

---

**Complete graph visualization backend with multi-experiment comparison, enhanced SCAR parameters, and industry-standard baselines.**
// services/clusteringApi.ts 
import {
  UploadResponse,
  ClusteringResponse,
  JobStatusResponse,
  HierarchyResponse,
  ClusteringParameters,
  ClusterDetailsResponse,
} from '../types/api';

import { 
  ComparisonRequest, 
  ComparisonStartResponse, 
  ComparisonStatusResponse 
} from '../types/comparison';

// Multi-comparison request interface
interface MultiComparisonRequest {
  name: string;
  selectedExperiments: Array<{ datasetId: string; jobId: string }>;
  baselineExperiment: { datasetId: string; jobId: string };
  metrics: string[];
}

export class ClusteringApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8080/api/v1') {
    this.baseUrl = baseUrl;
  }

  async uploadDataset(
    name: string, 
    files: { 
      graphFile: File; 
      propertiesFile: File; 
      pathFile: File; 
    }
  ): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('name', name);
    formData.append('graphFile', files.graphFile);
    formData.append('propertiesFile', files.propertiesFile);
    formData.append('pathFile', files.pathFile);

    const response = await fetch(`${this.baseUrl}/datasets`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Upload failed: ${response.status} ${errorText}`);
    }

    return response.json();
  }

  async startClustering(
    datasetId: string, 
    algorithm: string, 
    parameters: ClusteringParameters
  ): Promise<ClusteringResponse> {
    const response = await fetch(`${this.baseUrl}/datasets/${datasetId}/clustering`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ algorithm, parameters })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Clustering failed: ${response.status} ${errorText}`);
    }

    return response.json();
  }

  async getClusterDetails(
    datasetId: string, 
    clusterId: string, 
    jobId: string
  ): Promise<ClusterDetailsResponse> {
    const response = await fetch(
      `${this.baseUrl}/datasets/${datasetId}/clusters/${clusterId}/nodes?jobId=${jobId}`
    );
    
    if (!response.ok) {
      throw new Error(`Failed to get cluster details: ${response.statusText}`);
    }
    
    return response.json();
  }

  async getJobStatus(datasetId: string, jobId: string): Promise<JobStatusResponse> {
    const response = await fetch(`${this.baseUrl}/datasets/${datasetId}/clustering/${jobId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get job status: ${response.statusText}`);
    }
    
    return response.json();
  }

  async getHierarchy(datasetId: string, jobId: string): Promise<HierarchyResponse> {
    const response = await fetch(`${this.baseUrl}/datasets/${datasetId}/hierarchy?jobId=${jobId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get hierarchy: ${response.statusText}`);
    }
    
    return response.json();
  }

  async cancelJob(datasetId: string, jobId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/datasets/${datasetId}/clustering/${jobId}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      throw new Error(`Failed to cancel job: ${response.statusText}`);
    }
  }

  async getDatasets(): Promise<{ success: boolean; data: any[] }> {
    const response = await fetch(`${this.baseUrl}/datasets`);
    
    if (!response.ok) {
      throw new Error(`Failed to get datasets: ${response.statusText}`);
    }
    
    return response.json();
  }

  async deleteDataset(datasetId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/datasets/${datasetId}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      throw new Error(`Failed to delete dataset: ${response.statusText}`);
    }
  }

  async startComparison(request: ComparisonRequest): Promise<ComparisonStartResponse> {
    const response = await fetch(`${this.baseUrl}/comparisons`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Comparison failed: ${response.status} ${errorText}`);
    }

    return response.json();
  }

  async getComparisonStatus(comparisonId: string): Promise<ComparisonStatusResponse> {
    const response = await fetch(`${this.baseUrl}/comparisons/${comparisonId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get comparison status: ${response.statusText}`);
    }
    
    return response.json();
  }

  async cancelComparison(comparisonId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/comparisons/${comparisonId}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      throw new Error(`Failed to cancel comparison: ${response.statusText}`);
    }
  }
  
  // Updated: Send baselineExperiment instead of ensureLouvainBaseline
  async startMultiComparison(request: MultiComparisonRequest): Promise<{ success: boolean; data: { comparisonId: string } }> {
    const response = await fetch(`${this.baseUrl}/comparisons/multi`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Multi-comparison failed: ${response.status} ${errorText}`);
    }

    return response.json();
  }

  async getMultiComparison(comparisonId: string): Promise<{ success: boolean; data: any }> {
    const response = await fetch(`${this.baseUrl}/comparisons/${comparisonId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get multi-comparison: ${response.statusText}`);
    }
    
    return response.json();
  }

  async healthCheck(): Promise<{ status: string }> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }
}

// Singleton instance
export const apiClient = new ClusteringApiClient();
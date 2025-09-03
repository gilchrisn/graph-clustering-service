// types/api.ts
export interface NodePosition {
  x: number;
  y: number;
  radius: number;
}

export interface HierarchyLevel {
  level: number;
  communities: Record<string, string[]>;
  parentMap?: Record<string, string>;
}

export interface ClusteringResult {
  datasetId: string;
  jobId: string;
  algorithm: string;
  levels: HierarchyLevel[];
  coordinates: Record<string, NodePosition>;
  modularity: number;
  numLevels: number;
  numCommunities: number;
  processingTimeMS: number;
}

export interface Dataset {
  id: string;
  name: string;
  uploadedAt: string;
  nodeCount?: number;
  edgeCount?: number;
}

export interface JobProgress {
  percentage: number;
  message: string;
}

export interface JobStatus {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress?: JobProgress;
  result?: {
    modularity: number;
    numLevels: number;
    numCommunities: number;
    processingTimeMS: number;
  };
  error?: string;
}

export interface UploadResponse {
  success: boolean;
  data: {
    datasetId: string;
  };
}

export interface ClusteringResponse {
  success: boolean;
  data: {
    jobId: string;
  };
}

export interface ClusterEdge {
  source: string;
  target: string;
  weight: number;
}

export interface ClusterNode {
  id: string;
  label: string;
  position: NodePosition;
  type: 'leaf' | 'community';
  metadata: {
    level: number;
  };
}

export interface ClusterDetails {
  clusterId: string;
  level: number;
  nodes: ClusterNode[];
  edges: ClusterEdge[];
}

export interface ClusterDetailsResponse {
  success: boolean;
  data: ClusterDetails;
}

export interface JobStatusResponse {
  success: boolean;
  data: JobStatus;
}

export interface HierarchyResponse {
  success: boolean;
  data: {
    hierarchy: ClusteringResult;
  };
}

export interface ClusteringParameters {
  maxLevels: number;
  maxIterations: number;
  minModularityGain: number;
  k?: number;
  nk?: number;
  threshold?: number;
}


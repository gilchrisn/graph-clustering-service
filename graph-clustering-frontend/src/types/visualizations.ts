// types/visualizations.ts
import { ClusteringResult, JobProgress } from './api';

export interface Experiment {
  id: string;
  datasetId: string;
  algorithm: 'louvain' | 'scar';
  parameters: ClusteringParameters;
  jobId?: string;
  status: 'configuring' | 'running' | 'completed' | 'failed';
  result?: ClusteringResult;
  error?: string;
  progress?: JobProgress;
  createdAt: string;
}

export interface ClusteringParameters {
  maxLevels: number;
  maxIterations: number;
  minModularityGain: number;
  // SCAR specific parameters
  k?: number;
  nk?: number;
  threshold?: number;

  reconstructionThreshold?: number;
  reconstructionMode?: 'inclusion_exclusion' | 'full';
  edgeWeightNormalization?: boolean;
}

export interface DrillDownStrategy {
  id: string;
  name: string;
  description: string;
  selectBest: (communities: [string, string[]][], result: ClusteringResult) => string;
}

export interface VisualizationConfig {
  type: 'timeline-hierarchy' | 'animated-graph' | 'network-dashboard';
  title: string;
  description: string;
}

export interface HierarchyPath {
  experimentId: string;
  communities: string[];
  strategy: string;
}


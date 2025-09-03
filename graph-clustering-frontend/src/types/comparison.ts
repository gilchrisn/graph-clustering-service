// types/comparison.ts
export interface ComparisonRequest {
  name: string;
  experimentA: {
    datasetId: string;
    jobId: string;
  };
  experimentB: {
    datasetId: string;
    jobId: string;
  };
  metrics: string[];
  options: {
    levelWise: boolean;
    includeVisualization: boolean;
  };
}

export interface ComparisonJob {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress?: {
    percentage: number;
    message: string;
  };
  experimentA: {
    datasetId: string;
    jobId: string;
  };
  experimentB: {
    datasetId: string;
    jobId: string;
  };
  metrics: string[];
  result?: ComparisonResult;
  error?: string;
  createdAt: string;
  completedAt?: string;
  processingTimeMs?: number;
}

export interface ComparisonResult {
  agds?: number;
  hmi?: number;
  jaccard?: number;
  ari?: number;
  levelMetrics?: LevelComparisonMetrics[];
  significantDifferences?: string[];
  recommendations?: string[];
  summary: ComparisonSummary;
}

export interface LevelComparisonMetrics {
  level: number;
  agds: number;
  hmi: number;
  jaccard: number;
  communityOverlap: number;
}

export interface ComparisonSummary {
  overallSimilarity: string;
  keyDifferences: string[];
  recommendation: string;
}

// API Response types
export interface ComparisonStartResponse {
  success: boolean;
  message: string;
  data: {
    comparisonId: string;
  };
}

export interface ComparisonStatusResponse {
  success: boolean;
  message?: string;
  data: ComparisonJob;
}
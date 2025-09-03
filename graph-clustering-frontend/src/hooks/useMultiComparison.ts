// hooks/useMultiComparison.ts
import { useState, useCallback } from 'react';
import { apiClient } from '../services/clusteringApi';
import { Experiment } from '../types/visualizations';

interface MultiComparisonResult {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  result?: {
    baselineJobId: string;
    baselineConfig: {
      algorithm: string;
      parameters: Record<string, any>;
      description: string;
      isStandard: boolean;
    };
    experiments: Array<{
      jobId: string;
      label: string;
      metrics: {
        hmi: number;
        custom_leaf_metric: number;
        custom_displayed_metric: number;
      };
    }>;
  };
  error?: string;
}

export const useMultiComparison = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startMultiComparison = useCallback(async (
    experiments: Experiment[],
    baselineExperiment: Experiment // Added baseline parameter
  ): Promise<{ comparisonId: string }> => {
    setLoading(true);
    setError(null);

    try {
      if (!baselineExperiment.jobId) {
        throw new Error('Baseline experiment has no job ID');
      }

      // Validate that baseline is in the experiment list
      if (!experiments.find(exp => exp.id === baselineExperiment.id)) {
        throw new Error('Baseline experiment must be in the selected experiments list');
      }

      const request = {
        name: `Multi-Experiment Comparison (${experiments.length} experiments)`,
        selectedExperiments: experiments.map(exp => ({
          datasetId: exp.datasetId,
          jobId: exp.jobId!
        })),
        baselineExperiment: { // Send chosen baseline instead of ensureLouvainBaseline
          datasetId: baselineExperiment.datasetId,
          jobId: baselineExperiment.jobId
        },
        metrics: ["hmi", "custom_leaf_metric", "custom_displayed_metric"]
      };

      const response = await apiClient.startMultiComparison(request);
      return { comparisonId: response.data.comparisonId };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Multi-comparison failed';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const getMultiComparison = useCallback(async (
    comparisonId: string
  ): Promise<MultiComparisonResult> => {
    try {
      const response = await apiClient.getMultiComparison(comparisonId);
      console.log('Backend Response:', JSON.stringify(response.data, null, 2));
      console.log('Status:', response.data.status);
      console.log('Has result:', !!response.data.result);
      console.log('Has baselineConfig:', !!response.data.result?.baselineConfig);
      return response.data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get comparison';
      setError(errorMessage);
      throw err;
    }
  }, []);

  return {
    startMultiComparison,
    getMultiComparison,
    loading,
    error
  };
};
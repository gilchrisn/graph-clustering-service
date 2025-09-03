// hooks/useComparison.ts
import { useState, useCallback, useRef } from 'react';
import { apiClient } from '../services/clusteringApi';
import { useVisualizationStore } from '../store/visualizationStore';
import { ComparisonJob, ComparisonRequest } from '../types/comparison';



export const useComparison = () => {
  const [comparison, setComparison] = useState<ComparisonJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { experiments } = useVisualizationStore();
  
  // Track current request to handle cleanup 
  const currentRequestRef = useRef<AbortController | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const clearPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  }, []);

  const startComparison = useCallback(async (
    experimentAId: string,
    experimentBId: string,
    metrics: string[]
  ) => {
    setLoading(true);
    setError(null);
    setComparison(null);

    // Cancel previous request if still pending 
    if (currentRequestRef.current) {
      currentRequestRef.current.abort();
    }

    // Create new abort controller
    const abortController = new AbortController();
    currentRequestRef.current = abortController;
    try {
      const experimentA = experiments.find(exp => exp.id === experimentAId);
      const experimentB = experiments.find(exp => exp.id === experimentBId);

      if (!experimentA) {
        throw new Error(`Experiment A not found: ${experimentAId}`);
      }
      if (!experimentB) {
        throw new Error(`Experiment B not found: ${experimentBId}`);
      }
      if (!experimentA.jobId) {
        throw new Error(`Experiment A has no job ID: ${experimentAId}`);
      }
      if (!experimentB.jobId) {
        throw new Error(`Experiment B has no job ID: ${experimentBId}`);
      }
      if (experimentA.status !== 'completed') {
        throw new Error(`Experiment A is not completed: ${experimentA.status}`);
      }
      if (experimentB.status !== 'completed') {
        throw new Error(`Experiment B is not completed: ${experimentB.status}`);
      }

      const request: ComparisonRequest = {
        name: `Experiment ${experimentAId.slice(-4)} vs ${experimentBId.slice(-4)}`,
        experimentA: {
          datasetId: experimentA.datasetId, 
          jobId: experimentA.jobId         
        },
        experimentB: {
          datasetId: experimentB.datasetId, 
          jobId: experimentB.jobId         
        },
        metrics,
        options: {
          levelWise: true,
          includeVisualization: true
        }
      };

      const response = await apiClient.startComparison(request);
      
      // Check if request was aborted
      if (abortController.signal.aborted) {
        return;
      }

      const comparisonId = response.data.comparisonId;
      
      // Start polling for status
      startPolling(comparisonId);
      
    } catch (err) {
      // Don't set error if request was aborted
      if (!abortController.signal.aborted) {
        const errorMessage = err instanceof Error ? err.message : 'Comparison failed';
        setError(errorMessage);
        setLoading(false);
      }
    } finally {
      // Clear the ref if this was the current request
      if (currentRequestRef.current === abortController) {
        currentRequestRef.current = null;
      }
    }
  }, [experiments]);

  const startPolling = useCallback((comparisonId: string) => {
    const pollComparison = async (): Promise<boolean> => {
      try {
        const response = await apiClient.getComparisonStatus(comparisonId);
        const comparisonData = response.data;

        setComparison(comparisonData);

        if (comparisonData.status === 'completed') {
          setLoading(false);
          clearPolling();
          return false; // Stop polling
        } else if (comparisonData.status === 'failed') {
          setError(comparisonData.error || 'Comparison failed');
          setLoading(false);
          clearPolling();
          return false; // Stop polling
        } else if (comparisonData.status === 'cancelled') {
          setError('Comparison was cancelled');
          setLoading(false);
          clearPolling();
          return false; // Stop polling
        }

        return comparisonData.status === 'running' || comparisonData.status === 'queued';
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to get comparison status');
        setLoading(false);
        clearPolling();
        return false; // Stop polling
      }
    };

    // Initial poll
    pollComparison().then((shouldContinue) => {
      if (shouldContinue) {
        // Set up interval polling (same pattern as your experiment polling)
        pollingIntervalRef.current = setInterval(async () => {
          const shouldContinue = await pollComparison();
          if (!shouldContinue) {
            clearPolling();
          }
        }, 2000); // Poll every 2 seconds
      }
    });
  }, [clearPolling]);

  const cancelComparison = useCallback(async (comparisonId: string) => {
    try {
      await apiClient.cancelComparison(comparisonId);
      clearPolling();
      setLoading(false);
      setError('Comparison cancelled');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel comparison');
    }
  }, [clearPolling]);

  // Cleanup on unmount (same pattern as useClusterEdges)
  const cleanup = useCallback(() => {
    if (currentRequestRef.current) {
      currentRequestRef.current.abort();
      currentRequestRef.current = null;
    }
    clearPolling();
  }, [clearPolling]);

  return {
    comparison,
    loading,
    error,
    startComparison,
    cancelComparison,
    cleanup
  };
};
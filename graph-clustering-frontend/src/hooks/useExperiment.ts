// hooks/useExperiment.ts
import { useEffect, useCallback } from 'react';
import { useVisualizationStore } from '../store/visualizationStore';
import { apiClient } from '../services/clusteringApi';
import { Experiment } from '../types/visualizations';

export const usePolling = (
  callback: () => Promise<boolean>,
  interval: number,
  enabled: boolean
) => {
  useEffect(() => {
    if (!enabled) return;

    const poll = async () => {
      try {
        const shouldContinue = await callback();
        if (shouldContinue) {
          setTimeout(poll, interval);
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    };

    poll();
  }, [callback, interval, enabled]);
};

export const useExperimentPolling = (experiment: Experiment) => {
  const { updateExperiment } = useVisualizationStore();

  const pollJobStatus = useCallback(async (): Promise<boolean> => {
    if (!experiment.jobId || experiment.status !== 'running') return false;

    try {
      const response = await apiClient.getJobStatus(experiment.datasetId, experiment.jobId);
      const jobData = response.data;

      updateExperiment(experiment.id, {
        status: jobData.status as any,
        progress: jobData.progress,
        error: jobData.error
      });

      if (jobData.status === 'completed') {
        // Fetch hierarchy data
        const hierarchyResponse = await apiClient.getHierarchy(experiment.datasetId, experiment.jobId);
        const hierarchy = hierarchyResponse.data.hierarchy;
        
        if (jobData.result === undefined) {
          throw new Error('Job result is undefined');
        }
        
        // Merge job result stats with hierarchy data
        const result = {
          ...hierarchy,                                    // Hierarchy structure
          modularity: jobData.result.modularity,           // From job status
          numLevels: jobData.result.numLevels,             // From job status  
          numCommunities: jobData.result.numCommunities,   // From job status
          processingTimeMS: jobData.result.processingTimeMS // From job status
        };
        

        updateExperiment(experiment.id, { result });
        return false; // Stop polling
      }

      return jobData.status === 'running';
    } catch (error) {
      updateExperiment(experiment.id, {
        status: 'failed',
        error: error instanceof Error ? error.message : 'Job polling failed'
      });
      return false;
    }
  }, [experiment.id, experiment.jobId, experiment.datasetId, experiment.status, updateExperiment]);

  usePolling(pollJobStatus, 2000, experiment.status === 'running');
};

export const useExperimentActions = () => {
  const { addExperiment, currentDataset } = useVisualizationStore();

  const startExperiment = useCallback(async (
    algorithm: 'louvain' | 'scar',
    parameters: any
  ) => {
    if (!currentDataset) {
      throw new Error('No dataset selected');
    }

    const experiment: Experiment = {
      id: `exp_${Date.now()}`,
      datasetId: currentDataset.id,
      algorithm,
      parameters,
      status: 'configuring',
      createdAt: new Date().toISOString()
    };

    addExperiment(experiment);

    try {
      const response = await apiClient.startClustering(
        currentDataset.id,
        algorithm,
        parameters
      );

      useVisualizationStore.getState().updateExperiment(experiment.id, {
        jobId: response.data.jobId,
        status: 'running'
      });

      return experiment.id;
    } catch (error) {
      useVisualizationStore.getState().updateExperiment(experiment.id, {
        status: 'failed',
        error: error instanceof Error ? error.message : 'Failed to start clustering'
      });
      throw error;
    }
  }, [currentDataset, addExperiment]);

  return {
    startExperiment
  };
};

export const useCompletedExperiments = () => {
  return useVisualizationStore(state => 
    state.experiments.filter(exp => exp.status === 'completed')
  );
};

export const useRunningExperiments = () => {
  return useVisualizationStore(state => 
    state.experiments.filter(exp => exp.status === 'running')
  );
};

export const useExperimentById = (id: string) => {
  return useVisualizationStore(state => 
    state.experiments.find(exp => exp.id === id)
  );
};
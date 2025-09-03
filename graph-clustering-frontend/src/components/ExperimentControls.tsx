// components/ExperimentControls.tsx
import React, { useState } from 'react';
import { useVisualizationStore } from '../store/visualizationStore';
import { useExperimentActions } from '../hooks/useExperiment';
import { ClusteringParameters } from '../types/visualizations';
import { Button } from './ui/Button';
import { Card } from './ui/Card';

export const ExperimentControls: React.FC = () => {
  const { currentDataset } = useVisualizationStore();
  const { startExperiment } = useExperimentActions();
  
  const [algorithm, setAlgorithm] = useState<'louvain' | 'scar'>('louvain');
  const [parameters, setParameters] = useState<ClusteringParameters>({
    maxLevels: 5,
    maxIterations: 100,
    minModularityGain: -100.0,
    reconstructionThreshold: 0.1,
    reconstructionMode: 'inclusion_exclusion',
    edgeWeightNormalization: true
  });
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleParameterChange = <K extends keyof ClusteringParameters>(
    key: K,
    value: ClusteringParameters[K]
  ) => {
    setParameters(prev => ({ ...prev, [key]: value }));
    setError(null);
  };

  const handleStartExperiment = async () => {
    if (!currentDataset) {
      setError('No dataset selected');
      return;
    }

    setIsStarting(true);
    setError(null);

    try {
      await startExperiment(algorithm, parameters);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start experiment');
    } finally {
      setIsStarting(false);
    }
  };

  const resetToDefaults = () => {
    if (algorithm === 'louvain') {
      setParameters({
        maxLevels: 5,
        maxIterations: 100,
        minModularityGain: 0.000001
      });
    } else {
      setParameters({
        maxLevels: 5,
        maxIterations: 100,
        minModularityGain: 0.000001,
        k: 64,
        nk: 1,
        threshold: 0.0
      });
    }
  };

  const handleAlgorithmChange = (newAlgorithm: 'louvain' | 'scar') => {
    setAlgorithm(newAlgorithm);
    
    // Set algorithm-specific defaults
    if (newAlgorithm === 'scar') {
      setParameters(prev => ({
        ...prev,
        k: 64,
        nk: 1,
        threshold: 0.0
      }));
    } else {
      // Remove SCAR-specific parameters for Louvain
      const { k, nk, threshold, ...louvainParams } = parameters;
      setParameters(louvainParams);
    }
    
    setError(null);
  };

  if (!currentDataset) {
    return (
      <Card className="border-l-4 border-gray-400">
        <div className="text-center py-8">
          <p className="text-gray-500">Please upload a dataset to create experiments</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="border-l-4 border-blue-500">
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">New Experiment</h3>
          <p className="text-sm text-gray-600">
            Configure clustering parameters for {currentDataset.name}
          </p>
        </div>
        
        <div className="space-y-4">
          {/* Algorithm Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Algorithm
            </label>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="louvain"
                  checked={algorithm === 'louvain'}
                  onChange={(e) => handleAlgorithmChange(e.target.value as 'louvain')}
                  className="mr-2"
                />
                <span className="text-sm text-gray-500">Louvain</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="scar"
                  checked={algorithm === 'scar'}
                  onChange={(e) => handleAlgorithmChange(e.target.value as 'scar')}
                  className="mr-2"
                />
                <span className="text-sm text-gray-500">SCAR</span>
              </label>
            </div>
          </div>

          {/* Common Parameters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max Levels
              </label>
              <input
                type="number"
                value={parameters.maxLevels}
                onChange={(e) => handleParameterChange('maxLevels', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs text-gray-500"
                min="1"
                max="20"
              />
              <p className="text-xs text-gray-500 mt-1">Hierarchy depth (1-20)</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max Iterations
              </label>
              <input
                type="number"
                value={parameters.maxIterations}
                onChange={(e) => handleParameterChange('maxIterations', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs text-gray-500"
                min="10"
                max="1000"
              />
              <p className="text-xs text-gray-500 mt-1">Optimization rounds (10-1000)</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Min Modularity Gain
              </label>
              <input
                type="number"
                value={parameters.minModularityGain}
                onChange={(e) => handleParameterChange('minModularityGain', parseFloat(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs text-gray-500"
                step="0.000001"
                min="-100.0"
                max="1"
              />
              <p className="text-xs text-gray-500 mt-1">Convergence threshold</p>
            </div>
          </div>

          {/* SCAR Specific Parameters */}
          {algorithm === 'scar' && (
            <div className="border-t pt-4">
              <h4 className="text-sm font-medium text-gray-700 mb-3">SCAR Parameters</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    K (Sketch Size)
                  </label>
                  <input
                    type="number"
                    value={parameters.k || 64}
                    onChange={(e) => handleParameterChange('k', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs text-gray-500"
                    min="16"
                    max="512"
                  />
                  <p className="text-xs text-gray-500 mt-1">Sketch Size</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    NK (Sketch per Node)
                  </label>
                  <input
                    type="number"
                    value={parameters.nk || 1}
                    onChange={(e) => handleParameterChange('nk', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs text-gray-500"
                    min="2"
                    max="20"
                  />
                  <p className="text-xs text-gray-500 mt-1">Sketches per Node</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Threshold
                  </label>
                  <input
                    type="number"
                    value={parameters.threshold || 0.0}
                    onChange={(e) => handleParameterChange('threshold', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs text-gray-500"
                    step="0.1"
                    min="0"
                    max="1"
                  />
                  <p className="text-xs text-gray-500 mt-1">Threshold</p>
                </div>

                {/* New Reconstruction Parameters Section */}
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <h5 className="text-sm font-medium text-gray-700 mb-3">🆕 Graph Reconstruction</h5>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Reconstruction Threshold: {parameters.reconstructionThreshold}
                      </label>
                      <input
                        type="number"
                        value={parameters.reconstructionThreshold}
                        onChange={(e) => handleParameterChange('reconstructionThreshold', parseFloat(e.target.value))}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs text-gray-500"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Edge inclusion selectivity (any real number)
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Reconstruction Mode
                      </label>
                      <select
                        value={parameters.reconstructionMode}
                        onChange={(e) => handleParameterChange('reconstructionMode', e.target.value as 'inclusion_exclusion' | 'full')}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-xs text-gray-500"
                      >
                        <option value="inclusion_exclusion">Inclusion-Exclusion (Standard)</option>
                        <option value="full">Full Reconstruction</option>
                      </select>
                      <p className="text-xs text-gray-500 mt-1">Graph reconstruction method</p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Edge Weight Options
                      </label>
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          checked={parameters.edgeWeightNormalization}
                          onChange={(e) => handleParameterChange('edgeWeightNormalization', e.target.checked)}
                          className="mr-2 rounded border-gray-300"
                        />
                        <span className="text-xs text-gray-500">Normalize Edge Weights</span>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">Apply weight normalization during reconstruction</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded-md">
            <div className="text-sm">{error}</div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-between items-center">
          <Button 
            variant="secondary" 
            size="sm"
            onClick={resetToDefaults}
          >
            Reset to Defaults
          </Button>
          
          <div className="flex space-x-3">
            <Button
              variant="success"
              onClick={handleStartExperiment}
              disabled={isStarting || !currentDataset}
            >
              {isStarting ? 'Starting...' : 'Start Experiment'}
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};
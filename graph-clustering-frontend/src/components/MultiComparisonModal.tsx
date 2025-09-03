// components/MultiComparisonModal.tsx 
import React, { useState, useEffect } from 'react';
import { FullscreenModal } from './ui/FullscreenModal';
import { Card } from './ui/Card';
import { Button } from './ui/Button';
import { Loading } from './ui/Loading';
import { MultiComparisonChart } from './MultiComparisonChart';
import { useMultiComparison } from '../hooks/useMultiComparison';
import { Experiment } from '../types/visualizations';

interface MultiComparisonModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedExperiments: Experiment[];
}

export const MultiComparisonModal: React.FC<MultiComparisonModalProps> = ({
  isOpen,
  onClose,
  selectedExperiments
}) => {
  const [step, setStep] = useState<'baseline-selection' | 'comparison'>('baseline-selection');
  const [selectedBaseline, setSelectedBaseline] = useState<string | null>(null);
  const [comparisonId, setComparisonId] = useState<string | null>(null);
  const [comparison, setComparison] = useState<any>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  
  const { 
    startMultiComparison, 
    getMultiComparison, 
    loading: apiLoading, 
    error: apiError 
  } = useMultiComparison();

  // Reset state when modal opens/closes
  useEffect(() => {
    if (isOpen && selectedExperiments.length >= 2) {
      setStep('baseline-selection');
      setSelectedBaseline(null);
      setComparisonId(null);
      setComparison(null);
      
      // Auto-select first Louvain experiment as default baseline
      const louvainExperiment = selectedExperiments.find(exp => exp.algorithm === 'louvain');
      if (louvainExperiment) {
        setSelectedBaseline(louvainExperiment.id);
      } else {
        // If no Louvain, select first experiment
        setSelectedBaseline(selectedExperiments[0].id);
      }
    }
  }, [isOpen, selectedExperiments]);

  // Poll for results
  useEffect(() => {
    if (comparisonId && !comparison?.result) {
      const pollComparison = async () => {
        try {
          const result = await getMultiComparison(comparisonId);
          setComparison(result);
          
          if (result.status === 'completed' || result.status === 'failed') {
            if (pollingInterval) {
              clearInterval(pollingInterval);
              setPollingInterval(null);
            }
          }
        } catch (error) {
          console.error('Polling error:', error);
        }
      };

      // Initial poll
      pollComparison();
      
      // Set up interval
      const interval = setInterval(pollComparison, 2000);
      setPollingInterval(interval);
      
      return () => {
        if (interval) clearInterval(interval);
      };
    }
  }, [comparisonId, comparison?.result]);

  const handleStartComparison = async () => {
    if (!selectedBaseline) return;
    
    try {
      const baselineExperiment = selectedExperiments.find(exp => exp.id === selectedBaseline);
      if (!baselineExperiment) return;

      const result = await startMultiComparison(selectedExperiments, baselineExperiment);
      setComparisonId(result.comparisonId);
      setStep('comparison');
    } catch (error) {
      console.error('Failed to start comparison:', error);
    }
  };

  const handleClose = () => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
    setComparisonId(null);
    setComparison(null);
    setStep('baseline-selection');
    setSelectedBaseline(null);
    onClose();
  };

  const getAlgorithmColor = (algorithm: string) => {
    return algorithm === 'louvain' 
      ? 'bg-purple-100 text-purple-700' 
      : 'bg-orange-100 text-orange-700';
  };

  const formatParameterString = (params: any) => {
    return Object.entries(params)
      .map(([key, value]) => `${key}=${value}`)
      .join(', ');
  };

  const baselineExperiment = selectedExperiments.find(exp => exp.id === selectedBaseline);

  return (
    <FullscreenModal
      isOpen={isOpen}
      onClose={handleClose}
      title="Multi-Experiment Comparison"
    >
      <div className="h-full p-6">
        <div className="max-w-6xl mx-auto space-y-6">
          {step === 'baseline-selection' && (
            <>
              {/* Baseline Selection Step */}
              <Card>
                <h3 className="text-lg font-semibold mb-4">
                  Step 1: Choose Baseline Experiment
                </h3>
                <p className="text-gray-600 mb-6">
                  Select which experiment to use as the baseline for comparison. 
                  Other experiments will be compared against this baseline.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  {selectedExperiments.map(experiment => (
                    <Card
                      key={experiment.id}
                      className={`cursor-pointer transition-all ${
                        selectedBaseline === experiment.id
                          ? 'border-blue-500 bg-blue-50'
                          : 'hover:border-gray-300 hover:shadow-md'
                      }`}
                      onClick={() => setSelectedBaseline(experiment.id)}
                    >
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <input
                              type="radio"
                              checked={selectedBaseline === experiment.id}
                              onChange={() => setSelectedBaseline(experiment.id)}
                              className="text-blue-600 focus:ring-blue-500"
                            />
                            <h4 className="font-medium text-gray-900">
                              Experiment {experiment.id.slice(-4)}
                            </h4>
                            <span className={`px-2 py-1 text-xs rounded font-medium ${getAlgorithmColor(experiment.algorithm)}`}>
                              {experiment.algorithm.toUpperCase()}
                            </span>
                            {selectedBaseline === experiment.id && (
                              <span className="px-2 py-1 text-xs rounded font-medium bg-blue-100 text-blue-700">
                                Baseline
                              </span>
                            )}
                          </div>
                          <div className="text-sm text-gray-600">
                            Modularity: {experiment.result?.modularity.toFixed(3)}
                          </div>
                        </div>
                        
                        <p className="text-xs text-gray-600 ml-6">
                          {formatParameterString(experiment.parameters)}
                        </p>

                        {experiment.result && (
                          <div className="text-xs text-gray-500 grid grid-cols-3 gap-2 ml-6">
                            <span>Levels: {experiment.result.levels.length}</span>
                            <span>Communities: {experiment.result.numCommunities}</span>
                            <span>
                              Time: {experiment.result.processingTimeMS 
                                ? `${(experiment.result.processingTimeMS / 1000).toFixed(1)}s`
                                : 'N/A'
                              }
                            </span>
                          </div>
                        )}
                      </div>
                    </Card>
                  ))}
                </div>

                {/* Baseline Preview */}
                {baselineExperiment && (
                  <Card className="border-l-4 border-blue-500 bg-blue-50">
                    <h4 className="font-semibold text-blue-900 mb-2">Selected Baseline</h4>
                    <div className="text-blue-800">
                      <p><strong>Algorithm:</strong> {baselineExperiment.algorithm.toUpperCase()}</p>
                      <p><strong>Parameters:</strong> {formatParameterString(baselineExperiment.parameters)}</p>
                      <p><strong>Modularity:</strong> {baselineExperiment.result?.modularity.toFixed(3)}</p>
                    </div>
                  </Card>
                )}

                {/* Action Buttons */}
                <div className="flex justify-end gap-3 pt-6 border-t border-gray-200">
                  <Button variant="secondary" onClick={handleClose}>
                    Cancel
                  </Button>
                  <Button 
                    onClick={handleStartComparison}
                    disabled={!selectedBaseline || apiLoading}
                  >
                    {apiLoading ? 'Starting...' : 'Start Comparison'}
                  </Button>
                </div>
              </Card>
            </>
          )}

          {step === 'comparison' && (
            <>
              {/* Comparison Info */}
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">
                    Comparing {selectedExperiments.length} Experiments
                  </h3>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setStep('baseline-selection')}
                  >
                    Change Baseline
                  </Button>
                </div>
                
                {baselineExperiment && (
                  <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="text-sm">
                      <strong className="text-blue-900">Baseline:</strong>
                      <span className="ml-2 text-blue-800">
                        {baselineExperiment.algorithm.toUpperCase()} 
                        (Exp {baselineExperiment.id.slice(-4)})
                      </span>
                    </div>
                  </div>
                )}
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {selectedExperiments.map(exp => (
                    <div key={exp.id} className="text-center">
                      <div className={`px-3 py-1 rounded text-sm font-medium ${
                        exp.algorithm === 'louvain' 
                          ? 'bg-purple-100 text-purple-700'
                          : 'bg-orange-100 text-orange-700'
                      } ${exp.id === selectedBaseline ? 'ring-2 ring-blue-500' : ''}`}>
                        {exp.algorithm.toUpperCase()}
                        {exp.id === selectedBaseline && (
                          <span className="ml-1 text-xs">(Baseline)</span>
                        )}
                      </div>
                      <div className="text-xs text-gray-600 mt-1">
                        {exp.algorithm === 'scar' && exp.parameters.k && `k=${exp.parameters.k}`}
                        {exp.algorithm === 'louvain' && `levels=${exp.parameters.maxLevels}`}
                      </div>
                    </div>
                  ))}
                </div>
              </Card>

              {/* Loading State */}
              {(apiLoading || !comparison || comparison.status === 'running') && (
                <Card>
                  <Loading
                    message={
                      apiLoading ? "Starting multi-comparison..." :
                      !comparison ? "Initializing..." :
                      "Computing similarity metrics..."
                    }
                    variant="spinner"
                    size="lg"
                  />
                </Card>
              )}

              {/* Error State */}
              {(apiError || comparison?.status === 'failed') && (
                <Card className="border-red-200 bg-red-50">
                  <div className="text-red-800">
                    <h4 className="font-semibold mb-2">Comparison Failed</h4>
                    <p>{apiError || comparison?.error}</p>
                  </div>
                </Card>
              )}

              {/* Results */}
              {comparison?.result && (
                <>
                  {/* Chart */}
                  <MultiComparisonChart data={comparison.result.experiments} />

                  {/* Results Table */}
                  <Card>
                    <h4 className="text-lg font-semibold mb-4">Detailed Results</h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-gray-200">
                            <th className="text-left py-2 px-3">Configuration</th>
                            <th className="text-center py-2 px-3">HMI</th>
                            <th className="text-center py-2 px-3">Leaf Level Metric</th>
                            <th className="text-center py-2 px-3">Displayed Metric</th>
                          </tr>
                        </thead>
                        <tbody>
                          {comparison.result.experiments.map((exp: any) => (
                            <tr key={exp.jobId} className="border-b border-gray-100">
                              <td className="py-2 px-3 font-medium">
                                {exp.label}
                                {exp.jobId === baselineExperiment?.jobId && (
                                  <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                                    Baseline
                                  </span>
                                )}
                              </td>
                              <td className="py-2 px-3 text-center">
                                {(exp.metrics.hmi * 100).toFixed(1)}%
                              </td>
                              <td className="py-2 px-3 text-center">
                                {(exp.metrics.custom_leaf_metric * 100).toFixed(1)}%
                              </td>
                              <td className="py-2 px-3 text-center">
                                {(exp.metrics.custom_displayed_metric * 100).toFixed(1)}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </FullscreenModal>
  );
};
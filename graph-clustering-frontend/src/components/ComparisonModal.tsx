// components/ComparisonModal.tsx
import React, { useState } from 'react';
import { Experiment } from '../types/visualizations';
import { useComparison } from '../hooks/useComparison';
import { FullscreenModal } from './ui/FullscreenModal';
import { Card } from './ui/Card';
import { Button } from './ui/Button';
import { Loading } from './ui/Loading';

interface ComparisonModalProps {
  isOpen: boolean;
  onClose: () => void;
  sourceExperiment: Experiment;
  experiments: Experiment[]; // Available experiments for comparison
}

export const ComparisonModal: React.FC<ComparisonModalProps> = ({
  isOpen,
  onClose,
  sourceExperiment,
  experiments
}) => {
  const [targetExperiment, setTargetExperiment] = useState<Experiment | null>(null);
  const [comparisonStarted, setComparisonStarted] = useState(false);
  
  const { comparison, loading, error, startComparison } = useComparison();

  const handleStartComparison = async () => {
    if (!targetExperiment) return;
    
    setComparisonStarted(true);
    
    try {
      await startComparison(
        sourceExperiment.id,
        targetExperiment.id,
        ['agds', 'hmi', 'jaccard', 'ari']
      );
    } catch (err) {
      console.error('Failed to start comparison:', err);
      setComparisonStarted(false);
    }
  };

  const handleBack = () => {
    setComparisonStarted(false);
    setTargetExperiment(null);
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

  return (
    <FullscreenModal 
      isOpen={isOpen} 
      onClose={onClose} 
      title="Compare Experiments"
    >
      <div className="h-full flex flex-col">
        {!comparisonStarted ? (
          // Step 1: Select target experiment
          <div className="flex-1 p-6">
            <div className="max-w-6xl mx-auto space-y-6">
              {/* Source Experiment Display */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Compare From:
                </h3>
                <Card className="border-l-4 border-blue-500">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <h4 className="font-medium text-gray-900">
                        Experiment {sourceExperiment.id.slice(-4)}
                      </h4>
                      <span className={`px-2 py-1 text-xs rounded font-medium ${getAlgorithmColor(sourceExperiment.algorithm)}`}>
                        {sourceExperiment.algorithm.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      Modularity: {sourceExperiment.result?.modularity.toFixed(3)}
                    </div>
                  </div>
                  <p className="text-xs text-gray-600 mt-2">
                    {formatParameterString(sourceExperiment.parameters)}
                  </p>
                </Card>
              </div>

              {/* Target Experiment Selection */}
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Select Target Experiment:
                </h3>
                
                {experiments.length === 0 ? (
                  <Card className="text-center py-12">
                    <div className="text-gray-500">
                      <p className="text-lg font-medium mb-2">No Completed Experiments Available</p>
                      <p className="text-sm">
                        You need at least one other completed experiment to run a comparison.
                      </p>
                    </div>
                  </Card>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {experiments.map(experiment => (
                      <Card
                        key={experiment.id}
                        className={`cursor-pointer transition-all ${
                          targetExperiment?.id === experiment.id
                            ? 'border-green-500 bg-green-50'
                            : 'hover:border-gray-300 hover:shadow-md'
                        }`}
                        onClick={() => setTargetExperiment(experiment)}
                      >
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <h4 className="font-medium text-gray-900">
                                Experiment {experiment.id.slice(-4)}
                              </h4>
                              <span className={`px-2 py-1 text-xs rounded font-medium ${getAlgorithmColor(experiment.algorithm)}`}>
                                {experiment.algorithm.toUpperCase()}
                              </span>
                              {targetExperiment?.id === experiment.id && (
                                <span className="px-2 py-1 text-xs rounded font-medium bg-green-100 text-green-700">
                                  Selected
                                </span>
                              )}
                            </div>
                            <div className="text-sm text-gray-600">
                              Modularity: {experiment.result?.modularity.toFixed(3)}
                            </div>
                          </div>
                          
                          <p className="text-xs text-gray-600">
                            {formatParameterString(experiment.parameters)}
                          </p>

                          {/* Mini preview */}
                          {experiment.result && (
                            <div className="text-xs text-gray-500 grid grid-cols-3 gap-2">
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
                )}
              </div>

              {/* Comparison Options */}
              {targetExperiment && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Comparison Preview:
                  </h3>
                  <Card className="border-l-4 border-green-500">
                    <div className="grid grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-medium text-gray-700 mb-2">Source</h4>
                        <div className="text-sm space-y-1">
                          <div className="flex justify-between">
                            <span>Algorithm:</span>
                            <span className="font-medium">{sourceExperiment.algorithm.toUpperCase()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Modularity:</span>
                            <span className="font-medium">{sourceExperiment.result?.modularity.toFixed(3)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Levels:</span>
                            <span className="font-medium">{sourceExperiment.result?.levels.length}</span>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-medium text-gray-700 mb-2">Target</h4>
                        <div className="text-sm space-y-1">
                          <div className="flex justify-between">
                            <span>Algorithm:</span>
                            <span className="font-medium">{targetExperiment.algorithm.toUpperCase()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Modularity:</span>
                            <span className="font-medium">{targetExperiment.result?.modularity.toFixed(3)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Levels:</span>
                            <span className="font-medium">{targetExperiment.result?.levels.length}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <div className="text-sm text-gray-600">
                        <strong>Metrics to compute:</strong> AGDS (position similarity), HMI (hierarchy similarity), 
                        Jaccard (community overlap), ARI (clustering agreement)
                      </div>
                    </div>
                  </Card>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex justify-end gap-3 pt-6 border-t border-gray-200">
                <Button variant="secondary" onClick={onClose}>
                  Cancel
                </Button>
                <Button 
                  onClick={handleStartComparison}
                  disabled={!targetExperiment}
                >
                  Start Comparison
                </Button>
              </div>
            </div>
          </div>
        ) : (
          // Step 2: Show comparison progress/results
          <div className="flex-1 p-6">
            <div className="max-w-6xl mx-auto">
              {/* Header with back button */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleBack}
                    className="p-2 text-gray-400 hover:text-gray-600 rounded-lg transition-colors"
                  >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M19 12H5m7-7l-7 7 7 7"/>
                    </svg>
                  </button>
                  <h3 className="text-lg font-semibold text-gray-900">
                    Comparison: {sourceExperiment.id.slice(-4)} vs {targetExperiment?.id.slice(-4)}
                  </h3>
                </div>
              </div>

              {/* Comparison Results */}
              <ComparisonResults 
                comparison={comparison}
                loading={loading}
                error={error}
                sourceExperiment={sourceExperiment}
                targetExperiment={targetExperiment!}
              />
            </div>
          </div>
        )}
      </div>
    </FullscreenModal>
  );
};

// Comparison Results Component
const ComparisonResults: React.FC<{
  comparison: any;
  loading: boolean;
  error: string | null;
  sourceExperiment: Experiment;
  targetExperiment: Experiment;
}> = ({ comparison, loading, error, sourceExperiment, targetExperiment }) => {
  
  if (loading) {
    return (
      <Card className="text-center py-12">
        <Loading 
          message={comparison?.progress?.message || "Computing similarity metrics..."}
          progress={comparison?.progress?.percentage}
          variant="progress"
          size="lg"
        />
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border-l-4 border-red-500 p-6">
        <div className="text-red-700">
          <h4 className="font-medium mb-2">Comparison Failed</h4>
          <p className="text-sm">{error}</p>
        </div>
      </Card>
    );
  }

  if (!comparison?.result) {
    return (
      <Card className="text-center py-12">
        <div className="text-gray-500">
          <p>No comparison results available yet.</p>
        </div>
      </Card>
    );
  }

  const result = comparison.result;

  return (
    <div className="space-y-6">
      {/* Overall Similarity Metrics */}
      <Card>
        <h4 className="text-lg font-semibold text-gray-900 mb-4">Overall Similarity</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {result.agds ? (result.agds * 100).toFixed(1) : '0.0'}%
              </div>
              <div className="text-sm text-gray-600">AGDS</div>
              <div className="text-xs text-gray-500">Position Similarity</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {result.hmi ? (result.hmi * 100).toFixed(1) : '0.0'}%
              </div>
              <div className="text-sm text-gray-600">HMI</div>
              <div className="text-xs text-gray-500">Hierarchy Similarity</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {result.jaccard ? (result.jaccard * 100).toFixed(1) : '0.0'}%
              </div>
              <div className="text-sm text-gray-600">Jaccard</div>
              <div className="text-xs text-gray-500">Community Overlap</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {result.ari ? (result.ari * 100).toFixed(1) : '0.0'}%
              </div>
              <div className="text-sm text-gray-600">ARI</div>
              <div className="text-xs text-gray-500">Clustering Agreement</div>
            </div>
          </div>
      </Card>

      {/* Insights and Recommendations */}
      {(result.significantDifferences?.length > 0 || result.recommendations?.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Significant Differences */}
          {result.significantDifferences?.length > 0 && (
            <Card className="border-l-4 border-yellow-500">
              <h4 className="text-lg font-semibold text-gray-900 mb-3">Key Differences</h4>
              <ul className="space-y-2">
                {result.significantDifferences.map((diff: string, index: number) => (
                  <li key={index} className="flex items-start gap-2 text-sm">
                    <span className="text-yellow-500 mt-0.5">⚠</span>
                    <span className="text-gray-700">{diff}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* Recommendations */}
          {result.recommendations?.length > 0 && (
            <Card className="border-l-4 border-green-500">
              <h4 className="text-lg font-semibold text-gray-900 mb-3">Recommendations</h4>
              <ul className="space-y-2">
                {result.recommendations.map((rec: string, index: number) => (
                  <li key={index} className="flex items-start gap-2 text-sm">
                    <span className="text-green-500 mt-0.5">💡</span>
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}

      {/* Level-wise Comparison */}
      {result.levelMetrics?.length > 0 && (
        <Card>
          <h4 className="text-lg font-semibold text-gray-900 mb-4">Level-wise Analysis</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3">Level</th>
                  <th className="text-center py-2 px-3">AGDS</th>
                  <th className="text-center py-2 px-3">HMI</th>
                  <th className="text-center py-2 px-3">Jaccard</th>
                  <th className="text-center py-2 px-3">Community Overlap</th>
                </tr>
              </thead>
              <tbody>
                {result.levelMetrics.map((level: any, index: number) => (
                  <tr key={index} className="border-b border-gray-100">
                    <td className="py-2 px-3 font-medium">Level {level.level}</td>
                    <td className="py-2 px-3 text-center">{(level.agds * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-center">{(level.hmi * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-center">{(level.jaccard * 100).toFixed(1)}%</td>
                    <td className="py-2 px-3 text-center">{(level.communityOverlap * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
};
// components/FullscreenTimelineHierarchy.tsx - Enhanced with edge support
import React from 'react';
import { useVisualizationStore } from '../store/visualizationStore';
import { TimelineColumn } from './TimelineColumn';
import { GraphStatsPanel } from './GraphStatsPanel';
import { Card } from './ui/Card';
import { useEdgeCache } from '../hooks/useClusterEdges';

export interface FullscreenTimelineHierarchyProps {
  selectedExperimentId: string | null;
  onExperimentSelect: (id: string | null) => void;
  enableEdges?: boolean;
  onEdgeToggle?: () => void;
}

export const FullscreenTimelineHierarchy: React.FC<FullscreenTimelineHierarchyProps> = ({
  selectedExperimentId,
  onExperimentSelect,
  enableEdges = true,
  onEdgeToggle
}) => {
  const { experiments, currentDataset } = useVisualizationStore();
  const { clearCache, getCacheSize } = useEdgeCache();

  if (!currentDataset) return null;

  // Filter and sort experiments
  const datasetExperiments = experiments.filter(exp => exp.datasetId === currentDataset.id);
  const sortedExperiments = [...datasetExperiments].sort((a, b) => 
    new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );

  
  const selectedExperiment = sortedExperiments.find(exp => exp.id === selectedExperimentId);

  const getExperimentStats = () => {
    const total = datasetExperiments.length;
    const running = datasetExperiments.filter(exp => exp.status === 'running').length;
    const completed = datasetExperiments.filter(exp => exp.status === 'completed').length;
    const failed = datasetExperiments.filter(exp => exp.status === 'failed').length;
    
    return { total, running, completed, failed };
  };

  const stats = getExperimentStats();

  const handleClearCache = () => {
    clearCache();
  };

  return (
    <div className="flex flex-col h-full bg-gray-100">
      {/* Enhanced Header for Fullscreen */}
      <div className="flex-shrink-0 border-b border-gray-200 p-6 bg-white">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-2xl font-bold text-gray-900">Graph Evolution Analysis</h3>
            <p className="text-gray-600 mt-1">
              Compare clustering algorithms across experiments • Dataset: {currentDataset.name}
            </p>
          </div>
          
          {/* Edge Controls in Fullscreen */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-2 text-sm text-gray-600">
                <input
                  type="checkbox"
                  checked={enableEdges}
                  onChange={onEdgeToggle}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span>Show Edges</span>
              </label>
              
              {enableEdges && (
                <button
                  onClick={handleClearCache}
                  className="px-2 py-1 text-xs text-gray-500 hover:text-gray-700 border border-gray-300 hover:border-gray-400 rounded transition-colors"
                  title={`Clear edge cache (${getCacheSize()} entries)`}
                >
                  Clear Cache ({getCacheSize()})
                </button>
              )}
            </div>

            {/* Research Tip */}
            <div className="text-right">
              <div className="text-sm font-medium text-blue-900">Research Hypothesis</div>
              <div className="text-xs text-blue-700 max-w-md">
                As SCAR k increases → Graph structure approaches Louvain behavior
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Stats for Comparison */}
        {stats.total > 0 && (
          <div className="flex items-center gap-8 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-gray-600">Total:</span>
              <span className="font-semibold text-lg">{stats.total}</span>
            </div>
            
            {stats.running > 0 && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                <span className="text-blue-600 font-medium">{stats.running} running</span>
              </div>
            )}
            
            {stats.completed > 0 && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full" />
                <span className="text-green-600 font-medium">{stats.completed} completed</span>
              </div>
            )}

            {/* Algorithm Breakdown */}
            <div className="flex items-center gap-4 ml-auto">
              <div className="text-xs text-gray-500">
                Louvain: {datasetExperiments.filter(e => e.algorithm === 'louvain').length}
              </div>
              <div className="text-xs text-gray-500">
                SCAR: {datasetExperiments.filter(e => e.algorithm === 'scar').length}
              </div>
              {/* Edge Status */}
              {enableEdges && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  <span className="text-green-600 text-xs">Edges enabled</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Fullscreen Timeline Container */}
      <div className="flex-1 overflow-hidden">
        {sortedExperiments.length > 0 ? (
          <div className="flex gap-8 p-6 overflow-x-auto h-full">

            {/* Selected experiment detailed stats panel */}
            {selectedExperiment?.result && (
              <div className="min-w-[25rem] max-w-[25rem] flex-shrink-0">
                <Card className="h-full border-l-4 border-green-500">
                  <h3 className="text-lg font-semibold mb-4">
                    Detailed Analysis: Exp {selectedExperiment.id.slice(-4)}
                  </h3>
                  <div className="h-[calc(100%-2rem)] overflow-y-auto">
                    <GraphStatsPanel 
                      result={selectedExperiment.result} 
                      variant="detailed" 
                    />
                  </div>
                </Card>
              </div>
            )}
            
            {sortedExperiments.map(experiment => (
              <div key={experiment.id} className="min-w-[30rem] max-w-[30rem]">
                <TimelineColumn
                  experiment={experiment}
                  isSelected={selectedExperimentId === experiment.id}
                  onSelect={() => onExperimentSelect(
                    selectedExperimentId === experiment.id ? null : experiment.id
                  )}
                  enableEdges={enableEdges}
                />
              </div>
            ))}
          </div>
        ) : (
          <Card className="m-6 h-[calc(100%-3rem)] flex items-center justify-center">
            <div className="text-center py-12">
              <div className="text-gray-400 mb-4">
                <svg className="mx-auto h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} 
                    d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Experiments Yet</h3>
              <p className="text-gray-600 mb-6 max-w-lg">
                Create experiments with different SCAR k values and Louvain parameters to compare their graph evolution patterns.
              </p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};
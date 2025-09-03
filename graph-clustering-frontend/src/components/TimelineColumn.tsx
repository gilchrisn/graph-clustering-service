// components/TimelineColumn.tsx 
import React, { useState, useEffect } from 'react';
import { Experiment } from '../types/visualizations';
import { useExperimentPolling } from '../hooks/useExperiment';
import { useVisualizationStore } from '../store/visualizationStore'; 
import { generateDrillDownPath } from '../utils/drillDownEngine';
import { Card } from './ui/Card';
import { Loading } from './ui/Loading';
import { ExperimentSummary } from './ExperimentSummary';
import { HierarchyLevelCard } from './HierarchyLevelCard';
import { GraphStatsPanel } from './GraphStatsPanel';
import { ConfirmationModal } from './ui/ConfirmationModal';

import { useEdgeCache } from '../hooks/useClusterEdges';


export interface TimelineColumnProps {
  experiment: Experiment;
  isSelected?: boolean;
  onSelect?: () => void;
  enableEdges?: boolean;
  onCompare?: (experimentId: string) => void; 
  selectionMode?: boolean;
  isInSelection?: boolean;
  onToggleSelection?: () => void;
}

export const TimelineColumn: React.FC<TimelineColumnProps> = ({
  experiment,
  isSelected = false,
  onSelect,
  enableEdges = true,
  onCompare, 
  selectionMode = false,
  isInSelection = false,
  onToggleSelection
}) => {
  const { removeExperiment } = useVisualizationStore();
  const { getCacheSize } = useEdgeCache();
  const [hierarchyPath, setHierarchyPath] = useState<string[]>([]);
  const [selectedCommunities, setSelectedCommunities] = useState<Record<number, string>>({});
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Use polling hook for running experiments
  useExperimentPolling(experiment);

  // Generate hierarchy path when result is available
  useEffect(() => {
    if (experiment.result) {
      const path = generateDrillDownPath(experiment.result);
      setHierarchyPath(path);
      
      // Initialize selected communities from path
      const selections: Record<number, string> = {};
      path.forEach((communityId, index) => {
        const levelIndex = experiment.result!.levels.length - 1 - index;
        selections[levelIndex] = communityId;
      });
      setSelectedCommunities(selections);
    }
  }, [experiment.result]);

  const getParameterString = () => {
    return Object.entries(experiment.parameters)
      .map(([key, value]) => `${key}=${value}`)
      .join(', ');
  };

  const getStatusDisplay = () => {
    switch (experiment.status) {
      case 'configuring':
        return { color: 'gray', text: 'Configuring...' };
      case 'running':
        return { color: 'blue', text: 'Running' };
      case 'completed':
        return { color: 'green', text: 'Completed' };
      case 'failed':
        return { color: 'red', text: 'Failed' };
      default:
        return { color: 'gray', text: 'Unknown' };
    }
  };

  const statusDisplay = getStatusDisplay();

  const handleCommunitySelect = (levelIndex: number, communityId: string) => {
    setSelectedCommunities(prev => ({
      ...prev,
      [levelIndex]: communityId
    }));
  };

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      removeExperiment(experiment.id);
      setShowDeleteConfirm(false);
    } catch (error) {
      console.error('Failed to delete experiment:', error);
    } finally {
      setIsDeleting(false);
    }
  };

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  return (
    <>
      <div 
        className={`min-w-[32rem] max-w-[32rem] transition-all duration-200 ${
          isSelected ? 'transform scale-[1.02]' : ''
        }`}
        onClick={onSelect}
      >
        {/* Selection checkbox overlay */}
        {selectionMode && (
          <div className="absolute top-3 left-3 z-20">
            <input
              type="checkbox"
              checked={isInSelection}
              onChange={onToggleSelection}
              onClick={(e) => e.stopPropagation()}
              className="w-5 h-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
          </div>
        )}

        <Card 
          className={`h-full ${
            selectionMode && isInSelection
              ? 'border-blue-500 bg-blue-50'
              : isSelected 
                ? 'border-blue-500 shadow-lg bg-blue-50' 
                : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
          }`}
          hoverable={!selectionMode}
        >
          <div className="space-y-4">
            {/* Column Header */}
            <div className={`border-b pb-3 ${
              isSelected ? 'border-blue-200' : 'border-gray-200'
            }`}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold text-gray-900">
                    Experiment {experiment.id.slice(-4)}
                  </h3>
                  <span className={`px-2 py-1 text-xs rounded font-medium ${
                    experiment.algorithm === 'louvain' 
                      ? 'bg-purple-100 text-purple-700' 
                      : 'bg-orange-100 text-orange-700'
                  }`}>
                    {experiment.algorithm.toUpperCase()}
                  </span>
                  {/* Edge indicator */}
                  {enableEdges && experiment.result && (
                    <span className="px-2 py-1 text-xs rounded font-medium bg-green-100 text-green-700">
                      Edges
                    </span>
                  )}
                </div>
                
                {/* Status and Action Buttons */}
                <div className="flex items-center gap-2">
                  {/* Status Badge */}
                  <div className={`flex items-center gap-1 text-xs px-2 py-1 rounded ${
                    statusDisplay.color === 'green' ? 'bg-green-100 text-green-700' :
                    statusDisplay.color === 'blue' ? 'bg-blue-100 text-blue-700' :
                    statusDisplay.color === 'red' ? 'bg-red-100 text-red-700' :
                    'bg-gray-100 text-gray-700'
                  }`}>
                    {statusDisplay.color === 'running' && (
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse mr-1" />
                    )}
                    {statusDisplay.text}
                  </div>

                  {/* Compare Button - Only show for completed experiments */}
                  {experiment.status === 'completed' && onCompare && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent column selection
                        onCompare(experiment.id);
                      }}
                      className="px-2 py-1 text-xs bg-blue-50 text-blue-700 hover:bg-blue-100 rounded transition-colors border border-blue-200"
                      title="Compare with another experiment"
                    >
                      Compare
                    </button>
                  )}

                  {/* Delete Button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation(); // Prevent column selection
                      setShowDeleteConfirm(true);
                    }}
                    className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors"
                    title="Delete experiment"
                  >
                    {/* Trash Icon */}
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="3,6 5,6 21,6"></polyline>
                      <path d="m19,6v14a2,2 0 0,1 -2,2H7a2,2 0 0,1 -2,-2V6m3,0V4a2,2 0 0,1 2,-2h4a2,2 0 0,1 2,2v2"></path>
                      <line x1="10" y1="11" x2="10" y2="17"></line>
                      <line x1="14" y1="11" x2="14" y2="17"></line>
                    </svg>
                  </button>
                </div>
              </div>

              <p className="text-xs text-gray-600 mb-2 break-all">
                {getParameterString()}
              </p>

              {experiment.result && (
                <div className="space-y-3">
                  {/* Compact view by default */}
                  <GraphStatsPanel result={experiment.result} variant="compact" />
                  
                  {/* Detailed view when selected */}
                  {isSelected && (
                    <div className="pt-3 border-t border-gray-200">
                      <GraphStatsPanel result={experiment.result} variant="detailed" />
                    </div>
                  )}
                </div>
              )}

              {/* Edge cache info for debugging */}
              {enableEdges && process.env.NODE_ENV === 'development' && (
                <div className="text-xs text-gray-400 mt-1">
                  Cache: {getCacheSize()} entries
                </div>
              )}
            </div>

            {/* Status Content */}
            {experiment.status === 'running' && (
              <div className="space-y-3">
                <Loading
                  message={experiment.progress?.message || 'Processing...'}
                  progress={experiment.progress?.percentage}
                  variant="progress"
                  size="sm"
                />
              </div>
            )}

            {experiment.status === 'failed' && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-md">
                <div className="text-red-700 text-sm">
                  <div className="font-medium">Error occurred:</div>
                  <div className="mt-1">{experiment.error}</div>
                </div>
              </div>
            )}

            {/* Hierarchy Path with Edge Support */}
            {experiment.result && hierarchyPath.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-medium text-gray-700">Hierarchy Path</h4>
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span>{hierarchyPath.length} levels</span>
                    {enableEdges && (
                      <span className="px-1.5 py-0.5 bg-green-100 text-green-700 rounded">
                        Edge support
                      </span>
                    )}
                  </div>
                </div>
                
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {hierarchyPath.map((communityId, pathIndex) => {
                    const levelIndex = experiment.result!.levels.length - 1 - pathIndex;
                    const level = experiment.result!.levels[levelIndex];
                    
                    if (!level) return null;
                    
                    return (
                      <HierarchyLevelCard
                        key={`${levelIndex}-${communityId}`}
                        level={level}
                        coordinates={experiment.result!.coordinates}
                        selectedCommunity={selectedCommunities[levelIndex] || communityId}
                        onCommunitySelect={(selectedId) => 
                          handleCommunitySelect(levelIndex, selectedId)
                        }
                        datasetId={experiment.datasetId}
                        jobId={experiment.jobId}
                        enableEdges={enableEdges}
                      />
                    );
                  })}
                </div>
              </div>
            )}

            {/* Empty State */}
            {experiment.status === 'completed' && (!experiment.result || hierarchyPath.length === 0) && (
              <div className="text-center py-8 text-gray-500">
                <div className="text-sm">No hierarchy data available</div>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Delete Confirmation Modal */}
      <ConfirmationModal
        isOpen={showDeleteConfirm}
        onClose={() => setShowDeleteConfirm(false)}
        onConfirm={handleDelete}
        title="Delete Experiment"
        message={`Are you sure you want to delete experiment ${experiment.id.slice(-4)}? This action cannot be undone.`}
        confirmText="Delete"
        confirmVariant="danger"
        isLoading={isDeleting}
      />
    </>
  );
};
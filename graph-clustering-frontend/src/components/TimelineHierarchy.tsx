// components/TimelineHierarchy.tsx - Updated section
import React, { useState } from 'react';
import { useVisualizationStore } from '../store/visualizationStore';
import { TimelineColumn } from './TimelineColumn';
import { FullscreenModal } from './ui/FullscreenModal';
import { FullscreenTimelineHierarchy } from './FullscreenTimelineHierarchy';
import { ComparisonModal } from './ComparisonModal'; 
import { MultiComparisonModal } from './MultiComparisonModal';
import { Card } from './ui/Card';
import { Button } from './ui/Button';
import { useEdgeCache } from '../hooks/useClusterEdges';
import { useExperimentSelection } from '@/hooks/useExperimentSelection';
import { Experiment } from '@/types/visualizations';

export const TimelineHierarchy: React.FC = () => {
  const { experiments, currentDataset } = useVisualizationStore();
  const { clearCache, getCacheSize } = useEdgeCache();
  const [selectedExperimentId, setSelectedExperimentId] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [enableEdges, setEnableEdges] = useState(true);
  
  const {
    selectionMode,
    selectedExperiments,
    toggleSelection,
    clearSelection,
    enableSelectionMode
  } = useExperimentSelection();

  const [showMultiComparison, setShowMultiComparison] = useState(false);

  const selectedExperimentObjects = Array.from(selectedExperiments)
    .map(id => experiments.find(exp => exp.id === id))
    .filter(Boolean) as Experiment[];
    
  // Comparison modal state
  const [comparisonModal, setComparisonModal] = useState<{
    isOpen: boolean;
    sourceExperimentId?: string;
  }>({ isOpen: false });

  if (!currentDataset) {
    return (
      <Card className="h-full flex items-center justify-center">
        <div className="text-center py-12">
          <div className="text-gray-400 mb-4">
            <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} 
                d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2h2a2 2 0 002-2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Dataset Selected</h3>
          <p className="text-gray-500 mb-4">
            Upload a dataset to start creating clustering experiments
          </p>
        </div>
      </Card>
    );
  }

  // Filter experiments for current dataset
  const datasetExperiments = experiments.filter(exp => exp.datasetId === currentDataset.id);
  const completedExperiments = datasetExperiments.filter(exp => exp.status === 'completed');

  // Sort experiments by creation time (newest first)
  const sortedExperiments = [...datasetExperiments].sort((a, b) => 
    new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );

  const getExperimentStats = () => {
    const total = datasetExperiments.length;
    const running = datasetExperiments.filter(exp => exp.status === 'running').length;
    const completed = datasetExperiments.filter(exp => exp.status === 'completed').length;
    const failed = datasetExperiments.filter(exp => exp.status === 'failed').length;
    
    return { total, running, completed, failed };
  };

  const stats = getExperimentStats();

  const handleEdgeToggle = () => {
    setEnableEdges(!enableEdges);
    if (!enableEdges) {
      clearCache();
    }
  };

  const handleClearCache = () => {
    clearCache();
  };

  // Handle compare button click
  const handleCompareExperiment = (experimentId: string) => {
    setComparisonModal({
      isOpen: true,
      sourceExperimentId: experimentId
    });
  };

  // Handle comparison modal close
  const handleComparisonModalClose = () => {
    setComparisonModal({ isOpen: false });
  };

  // QoL Enhancement: Auto-select all completed experiments
  const handleEnableSelectionMode = () => {
    const completedIds = completedExperiments.map(exp => exp.id);
    enableSelectionMode(completedIds);
  };

  return (
    <>
      <div className="flex flex-col h-full">
        {/* Header with Fullscreen Button and Edge Controls */}
        <div className="flex-shrink-0 border-b border-gray-200 p-4 bg-white">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xl font-bold text-gray-900">Timeline Hierarchy</h2>
            
            <div className="flex items-center gap-3">
              {/* Selection Mode Toggle */}
              {!selectionMode ? (
                <Button onClick={handleEnableSelectionMode} variant="secondary">
                  Select Multiple
                </Button>
              ) : (
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-600">
                    {selectedExperiments.size} selected
                  </span>
                  <Button 
                    onClick={() => setShowMultiComparison(true)}
                    disabled={selectedExperiments.size < 2}
                  >
                    Compare ({selectedExperiments.size})
                  </Button>
                  <Button variant="secondary" onClick={clearSelection}>
                    Cancel
                  </Button>
                </div>
              )}

              {/* Edge Controls */}
              <div className="flex items-center gap-2">
                <label className="flex items-center gap-2 text-sm text-gray-600">
                  <input
                    type="checkbox"
                    checked={enableEdges}
                    onChange={handleEdgeToggle}
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

              {/* Dataset Info */}
              <div className="text-sm text-gray-600">
                <span className="font-medium">{currentDataset.name}</span>
                <span className="text-gray-400 ml-2">
                  • {new Date(currentDataset.uploadedAt).toLocaleDateString()}
                </span>
              </div>

              {/* FULLSCREEN BUTTON */}
              <button
                onClick={() => setIsFullscreen(true)}
                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-50 text-blue-700 hover:bg-blue-100 rounded-lg transition-colors border border-blue-200"
                title="Open in fullscreen for detailed analysis"
              >
                {/* Fullscreen icon */}
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 7V5a2 2 0 0 1 2-2h2m0 0h8m-8 0v2m8-2v2m0-2h2a2 2 0 0 1 2 2v2m0 0v8m0-8h-2m2 8v2a2 2 0 0 1-2 2h-2m0 0H7m8 0v-2M7 21v-2m0 2H5a2 2 0 0 1-2-2v-2m0 0V7"></path>
                </svg>
                <span className="hidden sm:inline">Fullscreen Analysis</span>
                <span className="sm:hidden">Expand</span>
              </button>
            </div>
          </div>

          {/* Stats */}
          {stats.total > 0 && (
            <div className="flex items-center gap-6 text-sm">
              <div className="flex items-center gap-1">
                <span className="text-gray-600">Total:</span>
                <span className="font-medium">{stats.total}</span>
              </div>
              
              {stats.running > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                  <span className="text-blue-600">{stats.running} running</span>
                </div>
              )}
              
              {stats.completed > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  <span className="text-green-600">{stats.completed} completed</span>
                </div>
              )}
              
              {stats.failed > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-red-500 rounded-full" />
                  <span className="text-red-600">{stats.failed} failed</span>
                </div>
              )}

              {/* Edge Status */}
              {enableEdges && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  <span className="text-green-600">Edges enabled</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Timeline Container - Rest remains the same */}
        <div className="flex-1 overflow-hidden min-h-[700px]">
          {sortedExperiments.length > 0 ? (
            <div className="flex gap-6 p-6 overflow-x-auto h-full">
              {sortedExperiments.map(experiment => (
                <TimelineColumn
                  key={experiment.id}
                  experiment={experiment}
                  isSelected={selectedExperimentId === experiment.id}
                  onSelect={() => setSelectedExperimentId(
                    selectedExperimentId === experiment.id ? null : experiment.id
                  )}
                  enableEdges={enableEdges}
                  onCompare={handleCompareExperiment}
                  selectionMode={selectionMode}
                  isInSelection={selectedExperiments.has(experiment.id)}
                  onToggleSelection={() => toggleSelection(experiment.id)}
                />
              ))}
            </div>
          ) : (
            <Card className="m-6 h-[calc(100%-2rem)] flex items-center justify-center">
              <div className="text-center py-12">
                <div className="text-gray-400 mb-4">
                  <svg className="mx-auto h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} 
                      d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Experiments Yet</h3>
                <p className="text-gray-600 mb-6 max-w-md">
                  Create your first clustering experiment using the controls above. 
                  Each experiment will appear as a column showing the hierarchy drill-down path.
                </p>
                
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-md mx-auto">
                  <h4 className="text-sm font-medium text-blue-900 mb-2">Research Tip:</h4>
                  <ul className="text-sm text-blue-800 space-y-1 text-left">
                    <li>• Create multiple SCAR experiments with different k values</li>
                    <li>• Compare with Louvain experiments</li>
                    <li>• Use fullscreen mode for detailed analysis</li>
                    <li>• Enable edges to see graph structure</li>
                    <li>• Observe graph evolution patterns</li>
                  </ul>
                </div>
              </div>
            </Card>
          )}
        </div>

        {/* Help Text with Edge Info */}
        {sortedExperiments.length > 0 && (
          <div className="flex-shrink-0 border-t border-gray-200 p-3 bg-gray-50">
            <div className="text-xs text-gray-600 text-center">
              <span className="font-medium">Tips:</span>
              <span className="ml-2">Click columns to select • Use fullscreen for detailed comparison • Toggle edges for complete graph visualization • Each column shows hierarchy drill-down from root to leaves</span>
            </div>
          </div>
        )}
      </div>

      {/*  Multi-Comparison Modal */}
      {showMultiComparison && (
        <MultiComparisonModal
          isOpen={showMultiComparison}
          onClose={() => setShowMultiComparison(false)}
          selectedExperiments={selectedExperimentObjects}
        />
      )}

      {/* FULLSCREEN MODAL */}
      <FullscreenModal
        isOpen={isFullscreen}
        onClose={() => setIsFullscreen(false)}
        title="Graph Evolution Analysis"
      >
        <FullscreenTimelineHierarchy
          selectedExperimentId={selectedExperimentId}
          onExperimentSelect={setSelectedExperimentId}
          enableEdges={enableEdges}
          onEdgeToggle={handleEdgeToggle}
        />
      </FullscreenModal>

      {/* COMPARISON MODAL */}
      {comparisonModal.sourceExperimentId && (
        <ComparisonModal
          isOpen={comparisonModal.isOpen}
          onClose={handleComparisonModalClose}
          sourceExperiment={sortedExperiments.find(exp => exp.id === comparisonModal.sourceExperimentId)!}
          experiments={sortedExperiments.filter(exp => 
            exp.status === 'completed' && exp.id !== comparisonModal.sourceExperimentId
          )}
        />
      )}
    </>
  );
};
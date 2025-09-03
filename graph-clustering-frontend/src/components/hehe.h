// Integration Guide: How to add comprehensive graph statistics to your existing components

// 1. Update TimelineColumn.tsx to include comprehensive stats
import { GraphStatsPanel } from './GraphStatsPanel';

// In your TimelineColumn component, replace the basic stats section:

// OLD CODE (around line 140):
// {experiment.result && (
//   <div className="grid grid-cols-3 gap-2 text-xs">
//     <div>
//       <span className="text-gray-500">Modularity:</span>
//       <div className="font-medium text-gray-900">{experiment.result.modularity.toFixed(3)}</div>
//     </div>
//     <div>
//       <span className="text-gray-500">Levels:</span>
//       <div className="font-medium text-gray-900">{experiment.result.levels.length}</div>
//     </div>
//     <div>
//       <span className="text-gray-500">Time:</span>
//       <div className="font-medium text-gray-900">
//         {experiment.result.processingTimeMS 
//           ? formatDuration(experiment.result.processingTimeMS)
//           : 'N/A'
//         }
//       </div>
//     </div>
//   </div>
// )}

// NEW CODE - Replace with:


// 2. Update HierarchyLevelCard.tsx for community statistics
import { CommunityGraphStats } from './GraphStatsPanel';

// Replace the CommunityItem component's content:

const CommunityItem: React.FC<{
  communityId: string;
  nodes: string[];
  coordinates: Record<string, NodePosition>;
  isSelected: boolean;
  onSelect?: () => void;
  datasetId?: string;
  jobId?: string;
  enableEdges?: boolean;
}> = ({ 
  communityId, 
  nodes, 
  coordinates, 
  isSelected, 
  onSelect,
  datasetId,
  jobId,
  enableEdges = true
}) => {
  // Your existing edge fetching logic...
  const { edges, loading: edgesLoading, error: edgesError } = useClusterEdges(
    datasetId || '',
    communityId,
    jobId || '',
    { enabled: enableEdges && !!datasetId && !!jobId }
  );

  return (
    <div className={`p-3 rounded-lg border transition-all w-full ${
      isSelected 
        ? 'border-blue-300 bg-blue-50' 
        : 'border-gray-200 hover:border-gray-300'
    }`}>
      
      {/* NEW: Comprehensive community statistics */}
      <CommunityGraphStats
        communityId={communityId}
        nodes={nodes}
        coordinates={coordinates}
        edges={edges}
        className="mb-3"
      />

      {/* Your existing MiniGraph component */}
      <CommunityPreview
        communityId={communityId}
        nodes={nodes}
        coordinates={coordinates}
        edges={edges}
        showEdgeLoading={edgesLoading}
        onClick={onSelect}
      />

      {/* Selection button */}
      {onSelect && (
        <button
          onClick={onSelect}
          className="w-full mt-2 px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded transition-colors"
        >
          {isSelected ? 'Selected' : 'Select'}
        </button>
      )}

      {/* Error handling */}
      {edgesError && (
        <div className="mt-2 text-xs text-red-600 bg-red-50 p-2 rounded">
          Edge loading failed: {edgesError}
        </div>
      )}
    </div>
  );
};

// 3. Optional: Add to FullscreenTimelineHierarchy for better overview
import { GraphStatsPanel } from './GraphStatsPanel';

// In FullscreenTimelineHierarchy.tsx, you can add experiment comparison:
export const FullscreenTimelineHierarchy: React.FC<FullscreenTimelineHierarchyProps> = ({
  selectedExperimentId,
  onExperimentSelect,
  enableEdges = true,
  onEdgeToggle
}) => {
  // ... existing code ...

  const selectedExperiment = sortedExperiments.find(exp => exp.id === selectedExperimentId);

  return (
    <div className="flex flex-col h-full bg-gray-100">
      {/* Existing header... */}

      <div className="flex-1 overflow-hidden">
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

          {/* Existing timeline columns */}
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
      </div>
    </div>
  );
};

// 4. Performance optimization: Add these imports to your main files
import React from 'react'; // Make sure React is imported for useMemo

// In your existing components, you can now access rich statistics like:

const SomeComponent = () => {
  const { experiments } = useVisualizationStore();
  
  return (
    <div>
      {experiments.map(experiment => {
        if (!experiment.result) return null;
        
        const analysis = GraphAnalyzer.analyzeExperiment(experiment.result);
        
        return (
          <div key={experiment.id}>
            <h3>Experiment {experiment.id.slice(-4)}</h3>
            <p>Branching Factor: {analysis.hierarchyMetrics.branchingFactor.toFixed(2)}</p>
            <p>Avg Degree: {analysis.levelAnalysis[0]?.stats.averageDegree.toFixed(2)}</p>
            <p>Graph Density: {(analysis.levelAnalysis[0]?.stats.density * 100).toFixed(1)}%</p>
          </div>
        );
      })}
    </div>
  );
};

// 5. File structure - add these new files to your project:
// utils/graphStats.ts (the comprehensive analyzer)
// components/GraphStatsPanel.tsx (the UI components)
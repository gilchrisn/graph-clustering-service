// components/HierarchyLevelCard.tsx 
import React from 'react';
import { HierarchyLevel, NodePosition } from '../types/api';
import { Card } from './ui/Card';
import { CommunityPreview } from './MiniGraph';
import { useClusterEdges } from '../hooks/useClusterEdges';
import { CommunityGraphStats } from './GraphStatsPanel';

export interface HierarchyLevelCardProps {
  level: HierarchyLevel;
  coordinates: Record<string, NodePosition>;
  selectedCommunity?: string;
  onCommunitySelect?: (communityId: string) => void;
  showAllCommunities?: boolean;
  // New props for edge support
  datasetId?: string;
  jobId?: string;
  enableEdges?: boolean;
}

// Individual community component with edge fetching
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
  // Fetch edges for this community
  const { 
    edges, 
    loading: edgesLoading, 
    error: edgesError 
  } = useClusterEdges(
    datasetId || '',
    communityId,
    jobId || '',
    { enabled: enableEdges && !!datasetId && !!jobId }
  );

  const getNodeTypeStats = (nodeList: string[]) => {
    const communities = nodeList.filter(id => id.startsWith('c0_')).length;
    const leafNodes = nodeList.length - communities;
    return { communities, leafNodes };
  };

  const stats = getNodeTypeStats(nodes);

  
  return (
    <div 
      className={`p-3 rounded-lg border transition-all w-full ${
        isSelected 
          ? 'border-blue-300 bg-blue-50' 
          : 'border-gray-200 hover:border-gray-300'
      }`}
    >
      <div className="space-y-2">
        {/* Community Header */}
        <div className="flex justify-between items-start">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-900 truncate">
                {communityId}
              </span>
              {isSelected && (
                <span className="px-1.5 py-0.5 bg-blue-200 text-blue-800 text-xs rounded">
                  Selected
                </span>
              )}
              {edgesError && (
                <span 
                  className="px-1.5 py-0.5 bg-yellow-100 text-yellow-800 text-xs rounded"
                  title={`Edge loading failed: ${edgesError}`}
                >
                  !
                </span>
              )}
            </div>
            
            <div className="flex gap-4 text-xs text-gray-600 mt-1">
              <span>{nodes.length} total nodes</span>
              {stats.communities > 0 && (
                <span>{stats.communities} communities</span>
              )}
              {stats.leafNodes > 0 && (
                <span>{stats.leafNodes} leaf nodes</span>
              )}
              {edges.length > 0 && (
                <span>• {edges.length} edges</span>
              )}
            </div>
          </div>
          
          {onSelect && (
            <button
              onClick={onSelect}
              className="ml-2 px-2 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded transition-colors"
            >
              {isSelected ? 'Selected' : 'Select'}
            </button>
          )}
        </div>

        {/* Comprehensive community statistics */}
        <CommunityGraphStats
          communityId={communityId}
          nodes={nodes}
          coordinates={coordinates}
          edges={edges}
          className="mb-3"
        />
        
        {/* Community Visualization with Edges */}
        <CommunityPreview
          communityId={communityId}
          nodes={nodes}
          coordinates={coordinates}
          edges={edges}
          showEdgeLoading={edgesLoading}
          onClick={onSelect}
        />

        {/* Edge Loading Indicator */}
        {edgesLoading && (
          <div className="text-xs text-blue-600 flex items-center gap-1">
            <div className="w-3 h-3 border border-blue-600 border-t-transparent rounded-full animate-spin" />
            <span>Loading edges...</span>
          </div>
        )}

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
    </div>
  );
};

export const HierarchyLevelCard: React.FC<HierarchyLevelCardProps> = ({
  level,
  coordinates,
  selectedCommunity,
  onCommunitySelect,
  showAllCommunities = false,
  datasetId,
  jobId,
  enableEdges = true
}) => {
  const communities = Object.entries(level.communities);
  const displayCommunities = showAllCommunities 
    ? communities 
    : selectedCommunity 
      ? communities.filter(([id]) => id === selectedCommunity)
      : communities.slice(0, 1);

  const getLevelTypeDisplay = (): { color: string; label: string } => {
    if (level.level === 0) {
      return { color: 'green', label: 'Leaf Level' };
    } else {
      return { color: 'blue', label: `Level ${level.level}` };
    }
  };

  const { color, label } = getLevelTypeDisplay();

  return (
    <Card 
      className={`border-l-4 border-${color}-500 w-full`}
      padding="md"
    >
      <div className="space-y-3">
        {/* Level Header */}
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <h4 className={`font-semibold text-${color}-900`}>{label}</h4>
            <span className={`px-2 py-1 bg-${color}-100 text-${color}-700 text-xs rounded-full`}>
              {communities.length} communities
            </span>
            {enableEdges && datasetId && jobId && (
              <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                Edges enabled
              </span>
            )}
          </div>
          
          {level.parentMap && (
            <span className="text-xs text-gray-500">
              ↗ Parent links
            </span>
          )}
        </div>

        {/* Communities Display */}
        <div className="space-y-3">
          {displayCommunities.map(([communityId, nodes]) => {
            const isSelected = selectedCommunity === communityId;
            
            return (
              <CommunityItem
                key={communityId}
                communityId={communityId}
                nodes={nodes}
                coordinates={coordinates}
                isSelected={isSelected}
                onSelect={() => onCommunitySelect?.(communityId)}
                datasetId={datasetId}
                jobId={jobId}
                enableEdges={enableEdges}
              />
            );
          })}
        </div>

        {/* Show All Toggle */}
        {!showAllCommunities && communities.length > 1 && !selectedCommunity && (
          <button
            onClick={() => onCommunitySelect?.('show-all')}
            className="w-full py-2 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 hover:border-gray-400 rounded transition-colors"
          >
            Show all {communities.length} communities
          </button>
        )}

        {/* Level Statistics */}
        <div className="pt-2 border-t border-gray-200">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-lg font-semibold text-gray-900">
                {communities.length}
              </div>
              <div className="text-xs text-gray-500">Communities</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-gray-900">
                {communities.reduce((sum, [, nodes]) => sum + nodes.length, 0)}
              </div>
              <div className="text-xs text-gray-500">Total Nodes</div>
            </div>
            <div>
              <div className="text-lg font-semibold text-gray-900">
                {level.level}
              </div>
              <div className="text-xs text-gray-500">Level</div>
            </div>
          </div>
        </div>

        {/* Edge Support Info */}
        {!enableEdges && (
          <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
            Edge rendering disabled. Enable for complete graph visualization.
          </div>
        )}
      </div>
    </Card>
  );
};
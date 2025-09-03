// components/GraphStatsPanel.tsx
import React from 'react';
import { ClusteringResult, ClusterEdge } from '../types/api';
import { GraphAnalyzer, useExperimentAnalysis, useCommunityAnalysis } from '../utils/graphStats';
import { Card } from './ui/Card';
import { NodePosition } from '../types/api';

interface GraphStatsPanelProps {
  result: ClusteringResult;
  className?: string;
  variant?: 'compact' | 'detailed';
}

export const GraphStatsPanel: React.FC<GraphStatsPanelProps> = ({ 
  result, 
  className = '',
  variant = 'detailed'
}) => {
  const analysis = useExperimentAnalysis(result);
  
  if (!analysis) return null;

  if (variant === 'compact') {
    return (
      <div className={`flex items-center gap-4 text-xs text-gray-600 ${className}`}>
        <span><strong>{analysis.overview.totalLevels}</strong> levels</span>
        <span><strong>{analysis.overview.totalCommunities}</strong> communities</span>
        <span><strong>{analysis.overview.totalNodes}</strong> nodes</span>
        <span><strong>{analysis.hierarchyMetrics.branchingFactor.toFixed(2)}</strong> branching</span>
        <span><strong>{(analysis.overview.processingTimeMs / 1000).toFixed(1)}s</strong></span>
      </div>
    );
  }

  return (
    <Card className={`border-l-4 border-blue-500 ${className}`}>
      <div className="space-y-6">
        {/* Overview Section */}
        <div>
          <h4 className="font-semibold text-blue-900 mb-3">Graph Analysis Overview</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {analysis.overview.totalLevels}
              </div>
              <div className="text-sm text-gray-600">Hierarchy Levels</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {analysis.overview.totalNodes.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Total Nodes</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {analysis.overview.totalCommunities.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Communities</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {analysis.overview.overallModularity.toFixed(3)}
              </div>
              <div className="text-sm text-gray-600">Modularity</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {analysis.hierarchyMetrics.branchingFactor.toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Branching Factor</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-600">
                {(analysis.overview.processingTimeMs / 1000).toFixed(1)}s
              </div>
              <div className="text-sm text-gray-600">Processing Time</div>
            </div>
          </div>
        </div>

        {/* Hierarchy Metrics */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h5 className="font-medium text-gray-800 mb-2">Hierarchy Structure</h5>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
            <div>
              <span className="text-gray-600">Avg Communities/Level:</span>
              <span className="ml-2 font-medium">{analysis.hierarchyMetrics.avgCommunitiesPerLevel.toFixed(1)}</span>
            </div>
            <div>
              <span className="text-gray-600">Avg Nodes/Level:</span>
              <span className="ml-2 font-medium">{analysis.hierarchyMetrics.avgNodesPerLevel.toFixed(1)}</span>
            </div>
            <div>
              <span className="text-gray-600">Hierarchy Depth:</span>
              <span className="ml-2 font-medium">{analysis.hierarchyMetrics.hierarchyDepth}</span>
            </div>
          </div>
        </div>

        {/* Level-by-Level Analysis */}
        <div>
          <h5 className="font-medium text-gray-800 mb-3">Level Analysis</h5>
          <div className="space-y-3">
            {analysis.levelAnalysis.map((levelAnalysis) => (
              <LevelStatsCard key={levelAnalysis.level} levelAnalysis={levelAnalysis} />
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
};

// Individual level statistics component
interface LevelStatsCardProps {
  levelAnalysis: any; // LevelAnalysis type
}

const LevelStatsCard: React.FC<LevelStatsCardProps> = ({ levelAnalysis }) => {
  const formattedStats = GraphAnalyzer.formatStats(levelAnalysis.stats);
  
  return (
    <div className="border border-gray-200 rounded-lg p-3 bg-white">
      <div className="flex items-center justify-between mb-2">
        <h6 className="font-medium text-gray-800">
          Level {levelAnalysis.level}
          {levelAnalysis.level === 0 && <span className="text-green-600 ml-1">(Leaf)</span>}
        </h6>
        <div className="text-xs text-gray-500">
          {levelAnalysis.stats.communityCount} communities
        </div>
      </div>
      
      <div className="grid grid-cols-3 md:grid-cols-5 gap-3 text-xs">
        {Object.entries(formattedStats).slice(0, 5).map(([key, value]) => (
          <div key={key} className="text-center">
            <div className="font-semibold text-gray-900">{value}</div>
            <div className="text-gray-500">{key}</div>
          </div>
        ))}
      </div>

      {/* Top communities */}
      {levelAnalysis.topCommunities.length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-100">
          <div className="text-xs text-gray-600">
            <span className="font-medium">Top communities:</span>
            <span className="ml-1">
              {levelAnalysis.topCommunities.slice(0, 3).map((comm: any, index: number) => 
                `${comm.id.slice(-6)}(${comm.size})`
              ).join(', ')}
              {levelAnalysis.topCommunities.length > 3 && '...'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

// Enhanced community preview with comprehensive graph stats
interface CommunityGraphStatsProps {
  communityId: string;
  nodes: string[];
  coordinates: Record<string, NodePosition>;
  edges?: ClusterEdge[];
  className?: string;
}

export const CommunityGraphStats: React.FC<CommunityGraphStatsProps> = ({
  communityId,
  nodes,
  coordinates,
  edges = [],
  className = ''
}) => {
  const stats = useCommunityAnalysis(nodes, coordinates, edges);
  const formattedStats = GraphAnalyzer.formatStats(stats);

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Header with community info */}
      <div className="flex justify-between items-center">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-blue-600 truncate">
              {communityId}
            </span>
            {stats.density > 0 && (
              <span className="px-1.5 py-0.5 bg-purple-100 text-purple-700 text-xs rounded">
                {(stats.density * 100).toFixed(1)}% dense
              </span>
            )}
            {stats.clusteringCoefficient > 0.5 && (
              <span className="px-1.5 py-0.5 bg-green-100 text-green-700 text-xs rounded">
                High clustering
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Core Statistics */}
      <div className="grid grid-cols-4 gap-2 text-xs text-center">
        <div>
          <div className="font-semibold text-gray-900">{stats.nodeCount}</div>
          <div className="text-gray-500">nodes</div>
        </div>
        <div>
          <div className="font-semibold text-gray-900">{stats.edgeCount}</div>
          <div className="text-gray-500">edges</div>
        </div>
        <div>
          <div className="font-semibold text-gray-900">{stats.averageDegree.toFixed(1)}</div>
          <div className="text-gray-500">avg degree</div>
        </div>
        <div>
          <div className="font-semibold text-gray-900">{(stats.density * 100).toFixed(1)}%</div>
          <div className="text-gray-500">density</div>
        </div>
      </div>

      {/* Advanced metrics (expandable) */}
      <details className="text-xs">
        <summary className="cursor-pointer text-gray-600 hover:text-gray-800 flex items-center gap-1">
          <span>Advanced graph metrics</span>
          <svg className="w-3 h-3 transform group-open:rotate-90 transition-transform" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
          </svg>
        </summary>
        
        <div className="mt-2 grid grid-cols-2 gap-2 bg-gray-50 p-2 rounded">
          <div className="flex justify-between">
            <span className="text-gray-600">Max Degree:</span>
            <span className="font-medium">{stats.maxDegree}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Min Degree:</span>
            <span className="font-medium">{stats.minDegree}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Degree Variance:</span>
            <span className="font-medium">{stats.degreeVariance.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Clustering Coeff:</span>
            <span className="font-medium">{stats.clusteringCoefficient.toFixed(3)}</span>
          </div>
          {stats.leafNodeCount !== undefined && (
            <>
              <div className="flex justify-between">
                <span className="text-gray-600">Leaf Nodes:</span>
                <span className="font-medium">{stats.leafNodeCount}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Internal Nodes:</span>
                <span className="font-medium">{stats.internalNodeCount}</span>
              </div>
            </>
          )}
        </div>
      </details>
    </div>
  );
};

// components/ExperimentSummary.tsx 
import React from 'react';
import { ClusteringResult } from '../types/api';
import { getBasicExperimentStats } from '../utils/basicStats';

interface ExperimentSummaryProps {
  result: ClusteringResult;
  variant?: 'compact' | 'detailed';
}

export const ExperimentSummary: React.FC<ExperimentSummaryProps> = ({ 
  result, 
  variant = 'compact' 
}) => {
  const stats = getBasicExperimentStats(result);

  if (variant === 'compact') {
    return (
      <div className="flex items-center gap-4 text-xs text-gray-600">
        <span><strong>{stats.totalLevels}</strong> levels</span>
        <span><strong>{stats.totalCommunities}</strong> communities</span>
        <span><strong>{stats.totalNodes}</strong> nodes</span>
        {stats.processingTime && (
          <span><strong>{(stats.processingTime / 1000).toFixed(1)}s</strong></span>
        )}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
      <div>
        <div className="text-lg font-bold text-gray-900">{stats.totalLevels}</div>
        <div className="text-xs text-gray-600">Levels</div>
      </div>
      <div>
        <div className="text-lg font-bold text-gray-900">{stats.totalCommunities}</div>
        <div className="text-xs text-gray-600">Communities</div>
      </div>
      <div>
        <div className="text-lg font-bold text-gray-900">{stats.totalNodes}</div>
        <div className="text-xs text-gray-600">Nodes</div>
      </div>
      <div>
        <div className="text-lg font-bold text-gray-900">{result.modularity.toFixed(3)}</div>
        <div className="text-xs text-gray-600">Modularity</div>
      </div>
    </div>
  );
};

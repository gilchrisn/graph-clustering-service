
// components/CommunityStats.tsx
import React from 'react';
import { getCommunityStats } from '../utils/basicStats';

interface CommunityStatsProps {
  nodes: string[];
  edges?: { length: number };
  className?: string;
}

export const CommunityStats: React.FC<CommunityStatsProps> = ({ 
  nodes, 
  edges, 
  className = '' 
}) => {
  const stats = getCommunityStats(nodes);

  return (
    <div className={`grid grid-cols-3 gap-2 text-xs text-center ${className}`}>
      <div>
        <div className="font-semibold text-gray-900">{stats.totalNodes}</div>
        <div className="text-gray-500">nodes</div>
      </div>
      <div>
        <div className="font-semibold text-gray-900">{stats.leafNodes}</div>
        <div className="text-gray-500">leaf</div>
      </div>
      <div>
        <div className="font-semibold text-gray-900">
          {edges?.length ?? '—'}
        </div>
        <div className="text-gray-500">edges</div>
      </div>
    </div>
  );
};

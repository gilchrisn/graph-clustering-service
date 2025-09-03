// utils/basicStats.ts 
import { ClusteringResult, HierarchyLevel } from '../types/api';

/**
 * Simple statistics utilities following KISS principle
 * Each function does one thing well
 */

export const getBasicExperimentStats = (result: ClusteringResult) => ({
  totalLevels: result.levels.length,
  totalNodes: Object.keys(result.coordinates).length,
  totalCommunities: result.levels.reduce((sum, level) => sum + Object.keys(level.communities).length, 0),
  processingTime: result.processingTimeMS
});

export const getLevelStats = (level: HierarchyLevel) => {
  const communities = Object.values(level.communities);
  const allNodes = communities.flat();
  const leafNodes = allNodes.filter(id => !id.startsWith('c0_')).length;
  
  return {
    numCommunities: communities.length,
    totalNodes: allNodes.length,
    leafNodes,
    avgCommunitySize: Math.round(allNodes.length / communities.length * 10) / 10
  };
};

export const getCommunityStats = (nodes: string[]) => {
  const leafNodes = nodes.filter(id => !id.startsWith('c0_')).length;
  return {
    totalNodes: nodes.length,
    leafNodes,
    communityNodes: nodes.length - leafNodes
  };
};
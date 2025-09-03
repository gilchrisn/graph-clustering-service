// utils/drillDownEngine.ts
import { ClusteringResult } from '../types/api';
import { DrillDownStrategy } from '../types/visualizations';

export const drillDownStrategies: Record<string, DrillDownStrategy> = {
  'most-leaf-nodes': {
    id: 'most-leaf-nodes',
    name: 'Most Leaf Nodes',
    description: 'Select community with the most leaf nodes',
    selectBest: (communities) => {
      return communities.reduce((best, current) => 
        current[1].length > best[1].length ? current : best
      )[0];
    }
  },

  'largest-cluster': {
    id: 'largest-cluster',
    name: 'Largest Cluster',
    description: 'Select the largest community by node count',
    selectBest: (communities) => {
      return communities.reduce((best, current) => 
        current[1].length > best[1].length ? current : best
      )[0];
    }
  },

  'random': {
    id: 'random',
    name: 'Random',
    description: 'Select a random community',
    selectBest: (communities) => {
      const randomIndex = Math.floor(Math.random() * communities.length);
      return communities[randomIndex][0];
    }
  },

  'highest-modularity': {
    id: 'highest-modularity',
    name: 'Highest Modularity',
    description: 'Select community that contributes most to modularity',
    selectBest: (communities) => {
      // For now, fallback to largest - can be enhanced with modularity calculation
      return communities.reduce((best, current) => 
        current[1].length > best[1].length ? current : best
      )[0];
    }
  }
};

export class DrillDownEngine {
  private strategy: DrillDownStrategy;

  constructor(strategyId: string = 'most-leaf-nodes') {
    this.strategy = drillDownStrategies[strategyId] || drillDownStrategies['most-leaf-nodes'];
  }

  generateHierarchyPath(result: ClusteringResult): string[] {
    if (!result.levels.length) return [];
    
    const path: string[] = [];
    
    // Start from the highest level (root) and drill down
    for (let i = result.levels.length - 1; i >= 0; i--) {
      const level = result.levels[i];
      const communities = Object.entries(level.communities);
      
      if (communities.length === 0) continue;
      
      // Use strategy to select best community at this level
      const selectedCommunity = this.strategy.selectBest(communities, result);
      path.push(selectedCommunity);
    }
    
    return path;
  }

  generateAlternativePaths(result: ClusteringResult, maxPaths: number = 3): string[][] {
    const paths: string[][] = [];
    
    // Generate multiple paths using different strategies
    const strategies = Object.keys(drillDownStrategies).slice(0, maxPaths);
    
    strategies.forEach(strategyId => {
      const engine = new DrillDownEngine(strategyId);
      const path = engine.generateHierarchyPath(result);
      if (path.length > 0) {
        paths.push(path);
      }
    });
    
    return paths;
  }

  getPathDescription(): string {
    return this.strategy.description;
  }

  getStrategyName(): string {
    return this.strategy.name;
  }
}

export const generateDrillDownPath = (
  result: ClusteringResult, 
  strategyId: string = 'most-leaf-nodes'
): string[] => {
  const engine = new DrillDownEngine(strategyId);
  return engine.generateHierarchyPath(result);
};

export const getPathMetrics = (path: string[], result: ClusteringResult) => {
  const metrics = {
    totalLevels: path.length,
    totalNodes: 0,
    levelSizes: [] as number[]
  };

  path.forEach((communityId, index) => {
    const levelIndex = result.levels.length - 1 - index;
    const level = result.levels[levelIndex];
    const community = level.communities[communityId];
    
    if (community) {
      const nodeCount = community.length;
      metrics.totalNodes += nodeCount;
      metrics.levelSizes.push(nodeCount);
    }
  });

  return metrics;
};
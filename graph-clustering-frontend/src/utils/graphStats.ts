// utils/graphStats.ts 
import React from 'react';
import { ClusteringResult, HierarchyLevel, NodePosition, ClusterEdge } from '../types/api';


export interface GraphStats {
  // Basic counts
  nodeCount: number;
  edgeCount: number;
  communityCount: number;
  
  // Degree statistics
  averageDegree: number;
  maxDegree: number;
  minDegree: number;
  degreeVariance: number;
  
  // Graph structure
  density: number;
  clusteringCoefficient: number;
  
  // Community specific
  averageCommunitySize: number;
  maxCommunitySize: number;
  minCommunitySize: number;
  communityModularity?: number;
  
  // Hierarchy specific (if applicable)
  hierarchyLevel?: number;
  leafNodeCount?: number;
  internalNodeCount?: number;
}

export interface LevelAnalysis {
  level: number;
  stats: GraphStats;
  topCommunities: Array<{
    id: string;
    size: number;
    localModularity?: number;
  }>;
}

export interface ExperimentAnalysis {
  overview: {
    totalLevels: number;
    totalNodes: number;
    totalCommunities: number;
    overallModularity: number;
    processingTimeMs: number;
  };
  levelAnalysis: LevelAnalysis[];
  hierarchyMetrics: {
    avgCommunitiesPerLevel: number;
    avgNodesPerLevel: number;
    hierarchyDepth: number;
    branchingFactor: number;
  };
}

export class GraphAnalyzer {
  
  /**
   * Analyze entire experiment with comprehensive statistics
   */
  static analyzeExperiment(result: ClusteringResult): ExperimentAnalysis {
    const levelAnalysis = result.levels.map((level, index) => 
      this.analyzeLevel(level, result.coordinates, index)
    );

    const totalCommunities = result.levels.reduce(
      (sum, level) => sum + Object.keys(level.communities).length, 0
    );

    return {
      overview: {
        totalLevels: result.levels.length,
        totalNodes: Object.keys(result.coordinates).length,
        totalCommunities,
        overallModularity: result.modularity,
        processingTimeMs: result.processingTimeMS || 0
      },
      levelAnalysis,
      hierarchyMetrics: {
        avgCommunitiesPerLevel: totalCommunities / result.levels.length,
        avgNodesPerLevel: Object.keys(result.coordinates).length / result.levels.length,
        hierarchyDepth: result.levels.length - 1,
        branchingFactor: this.calculateBranchingFactor(result.levels)
      }
    };
  }

  /**
   * Analyze a specific hierarchy level
   */
  static analyzeLevel(
    level: HierarchyLevel, 
    coordinates: Record<string, NodePosition>,
    levelIndex: number
  ): LevelAnalysis {
    const communities = Object.entries(level.communities);
    const allNodes = communities.flatMap(([_, nodes]) => nodes);
    const validNodes = allNodes.filter(nodeId => coordinates[nodeId]);

    // Community size analysis
    const communitySizes = communities.map(([_, nodes]) => nodes.length);
    const leafNodes = allNodes.filter(nodeId => !nodeId.startsWith('c0_'));
    const internalNodes = allNodes.filter(nodeId => nodeId.startsWith('c0_'));

    // Top communities by size
    const topCommunities = communities
      .map(([id, nodes]) => ({ id, size: nodes.length }))
      .sort((a, b) => b.size - a.size)
      .slice(0, 5);

    const stats: GraphStats = {
      nodeCount: validNodes.length,
      edgeCount: 0, // Will be computed when edges are available
      communityCount: communities.length,
      
      // Degree stats (placeholder - need edges to compute properly)
      averageDegree: 0,
      maxDegree: 0,
      minDegree: 0,
      degreeVariance: 0,
      
      // Graph structure
      density: 0, // edges / (nodes * (nodes-1) / 2)
      clusteringCoefficient: 0,
      
      // Community stats
      averageCommunitySize: communitySizes.length > 0 
        ? communitySizes.reduce((a, b) => a + b, 0) / communitySizes.length 
        : 0,
      maxCommunitySize: Math.max(...communitySizes, 0),
      minCommunitySize: Math.min(...communitySizes, 0),
      
      // Hierarchy specific
      hierarchyLevel: level.level,
      leafNodeCount: leafNodes.length,
      internalNodeCount: internalNodes.length
    };

    return {
      level: level.level,
      stats,
      topCommunities
    };
  }

  /**
   * Analyze community with edge information
   */
  static analyzeCommunityWithEdges(
    nodes: string[],
    coordinates: Record<string, NodePosition>,
    edges: ClusterEdge[] = []
  ): GraphStats {
    const validNodes = nodes.filter(nodeId => coordinates[nodeId]);
    const nodeSet = new Set(validNodes);
    
    // Filter edges to only include edges within this community
    const communityEdges = edges.filter(edge => 
      nodeSet.has(edge.source) && nodeSet.has(edge.target)
    );

    // Calculate degree statistics
    const degrees = this.calculateDegrees(validNodes, communityEdges);
    const degreeValues = Object.values(degrees);
    
    const avgDegree = degreeValues.length > 0 
      ? degreeValues.reduce((a, b) => a + b, 0) / degreeValues.length 
      : 0;
    
    const maxDegree = Math.max(...degreeValues, 0);
    const minDegree = Math.min(...degreeValues, 0);
    
    // Degree variance
    const degreeVariance = degreeValues.length > 0
      ? degreeValues.reduce((sum, d) => sum + Math.pow(d - avgDegree, 2), 0) / degreeValues.length
      : 0;

    // Graph density
    const maxPossibleEdges = validNodes.length * (validNodes.length - 1) / 2;
    const density = maxPossibleEdges > 0 ? communityEdges.length / maxPossibleEdges : 0;

    // Clustering coefficient (simplified)
    const clusteringCoefficient = this.calculateClusteringCoefficient(validNodes, communityEdges);

    const leafNodes = validNodes.filter(nodeId => !nodeId.startsWith('c0_'));
    const internalNodes = validNodes.filter(nodeId => nodeId.startsWith('c0_'));

    return {
      nodeCount: validNodes.length,
      edgeCount: communityEdges.length,
      communityCount: 1, // This is a single community
      
      averageDegree: avgDegree,
      maxDegree,
      minDegree,
      degreeVariance,
      
      density,
      clusteringCoefficient,
      
      averageCommunitySize: validNodes.length,
      maxCommunitySize: validNodes.length,
      minCommunitySize: validNodes.length,
      
      leafNodeCount: leafNodes.length,
      internalNodeCount: internalNodes.length
    };
  }

  /**
   * Calculate node degrees from edge list
   */
  private static calculateDegrees(nodes: string[], edges: ClusterEdge[]): Record<string, number> {
    const degrees: Record<string, number> = {};
    
    // Initialize all nodes with degree 0
    nodes.forEach(node => degrees[node] = 0);
    
    // Count edges for each node
    edges.forEach(edge => {
      if (degrees[edge.source] !== undefined) degrees[edge.source]++;
      if (degrees[edge.target] !== undefined) degrees[edge.target]++;
    });
    
    return degrees;
  }

  /**
   * Calculate clustering coefficient (simplified local version)
   */
  private static calculateClusteringCoefficient(nodes: string[], edges: ClusterEdge[]): number {
    if (nodes.length < 3) return 0;

    const nodeSet = new Set(nodes);
    const edgeSet = new Set(edges.map(e => `${e.source}-${e.target}`));
    
    let totalTriangles = 0;
    let totalPossibleTriangles = 0;

    // For each node, count triangles
    nodes.forEach(node => {
      const neighbors = edges
        .filter(edge => edge.source === node || edge.target === node)
        .map(edge => edge.source === node ? edge.target : edge.source)
        .filter(neighbor => nodeSet.has(neighbor));

      if (neighbors.length < 2) return;

      // Count triangles involving this node
      for (let i = 0; i < neighbors.length; i++) {
        for (let j = i + 1; j < neighbors.length; j++) {
          const neighbor1 = neighbors[i];
          const neighbor2 = neighbors[j];
          
          totalPossibleTriangles++;
          
          // Check if neighbors are connected
          if (edgeSet.has(`${neighbor1}-${neighbor2}`) || edgeSet.has(`${neighbor2}-${neighbor1}`)) {
            totalTriangles++;
          }
        }
      }
    });

    return totalPossibleTriangles > 0 ? totalTriangles / totalPossibleTriangles : 0;
  }

  /**
   * Calculate hierarchy branching factor
   */
  private static calculateBranchingFactor(levels: HierarchyLevel[]): number {
    if (levels.length <= 1) return 0;

    let totalBranching = 0;
    let validLevels = 0;

    for (let i = 0; i < levels.length - 1; i++) {
      const currentLevel = levels[i];
      const nextLevel = levels[i + 1];
      
      const currentCommunities = Object.keys(currentLevel.communities).length;
      const nextCommunities = Object.keys(nextLevel.communities).length;
      
      if (nextCommunities > 0) {
        totalBranching += currentCommunities / nextCommunities;
        validLevels++;
      }
    }

    return validLevels > 0 ? totalBranching / validLevels : 0;
  }

  /**
   * Format statistics for display
   */
  static formatStats(stats: GraphStats): Record<string, string> {
    return {
      'Nodes': stats.nodeCount.toLocaleString(),
      'Edges': stats.edgeCount.toLocaleString(),
      'Avg Degree': stats.averageDegree.toFixed(2),
      'Max Degree': stats.maxDegree.toString(),
      'Density': `${(stats.density * 100).toFixed(1)}%`,
      'Clustering': stats.clusteringCoefficient.toFixed(3),
      'Communities': stats.communityCount.toLocaleString(),
      'Avg Community Size': stats.averageCommunitySize.toFixed(1),
      ...(stats.leafNodeCount !== undefined && {
        'Leaf Nodes': stats.leafNodeCount.toLocaleString()
      })
    };
  }
}

// Performance-optimized hooks
export const useExperimentAnalysis = (result?: ClusteringResult) => {
  return React.useMemo(() => {
    if (!result) return null;
    return GraphAnalyzer.analyzeExperiment(result);
  }, [result]);
};

export const useCommunityAnalysis = (
  nodes: string[],
  coordinates: Record<string, NodePosition>,
  edges: ClusterEdge[] = []
) => {
  return React.useMemo(() => {
    return GraphAnalyzer.analyzeCommunityWithEdges(nodes, coordinates, edges);
  }, [nodes, coordinates, edges]);
};
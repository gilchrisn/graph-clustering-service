// components/MiniGraph.tsx 
import React from 'react';
import { NodePosition, ClusterEdge } from '../types/api';
import { CommunityStats } from './CommunityStat';

export interface MiniGraphProps {
  nodes: string[];
  coordinates: Record<string, NodePosition>;
  edges?: ClusterEdge[];
  onNodeClick?: (nodeId: string) => void;
  width?: number;
  height?: number;
  showLabels?: boolean;
  showEdgeLoading?: boolean;
  className?: string;
  edgeConfig?: {
    opacity?: number;
    maxWidth?: number;
    color?: string;
    animate?: boolean;
  };
}

export const MiniGraph: React.FC<MiniGraphProps> = ({
  nodes,
  coordinates,
  edges = [],
  onNodeClick,
  width = 200,
  height = 140,
  showLabels = false,
  showEdgeLoading = false,
  className = '',
  edgeConfig = {}
}) => {
  const {
    opacity = 0.6,
    maxWidth = 3,
    color = '#888',
    animate = true
  } = edgeConfig;

  const scale = 0.4;
  const centerX = 200;  
  const centerY = 100; 

  // Filter nodes that have coordinates
  const validNodes = nodes.filter(nodeId => coordinates[nodeId]);

  if (validNodes.length === 0) {
    return (
      <div 
        className={`flex items-center justify-center bg-gray-50 border rounded text-xs text-gray-500 ${className}`}  
      >
        No coordinates
      </div>
    );
  }

  // Calculate bounds for better scaling
  const positions = validNodes.map(nodeId => coordinates[nodeId]);
  const bounds = positions.reduce(
    (acc, pos) => ({
      minX: Math.min(acc.minX, pos.x),
      maxX: Math.max(acc.maxX, pos.x),
      minY: Math.min(acc.minY, pos.y),
      maxY: Math.max(acc.maxY, pos.y)
    }),
    { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity }
  );

  // Calculate scale to fit all nodes
  const rangeX = bounds.maxX - bounds.minX;
  const rangeY = bounds.maxY - bounds.minY;
  const maxRange = Math.max(rangeX, rangeY);
  const padding = 10;
  const availableSpace = Math.min(width, height) - 2 * padding;
  const dynamicScale = maxRange > 0 ? availableSpace / maxRange : scale;

  // Transform coordinate helper
  const transformCoordinate = (coord: NodePosition) => ({
    x: centerX + (coord.x - (bounds.minX + bounds.maxX) / 2) * dynamicScale,
    y: centerY + (coord.y - (bounds.minY + bounds.maxY) / 2) * dynamicScale
  });

  // Render edges
  const renderEdges = () => {
    if (!edges.length) {
      // Show edge loading indicator if loading
      if (showEdgeLoading) {
        return (
          <g>
            <text
              x={centerX}
              y={centerY + 60}
              textAnchor="middle"
              className="text-xs fill-gray-400 animate-pulse"
            >
              Loading edges...
            </text>
          </g>
        );
      }
      return null;
    }

    return (
      <g className={animate ? 'transition-opacity duration-300' : ''}>
        {edges.map((edge, index) => {
          const sourceCoord = coordinates[edge.source];
          const targetCoord = coordinates[edge.target];
          
          // Skip edge if either node is not visible
          if (!sourceCoord || !targetCoord || !validNodes.includes(edge.source) || !validNodes.includes(edge.target)) {
            return null;
          }

          const sourcePos = transformCoordinate(sourceCoord);
          const targetPos = transformCoordinate(targetCoord);
          
          // Calculate edge width based on weight
          const edgeWidth = Math.max(0.5, Math.min(maxWidth, edge.weight * 2));
          
          return (
            <line
              key={`edge-${index}-${edge.source}-${edge.target}`}
              x1={sourcePos.x}
              y1={sourcePos.y}
              x2={targetPos.x}
              y2={targetPos.y}
              stroke={color}
              strokeWidth={edgeWidth}
              opacity={opacity}
              className={animate ? 'transition-all duration-200' : ''}
            >
              <title>{`${edge.source} ↔ ${edge.target} (${edge.weight.toFixed(2)})`}</title>
            </line>
          );
        })}
      </g>
    );
  };

  // Render nodes
  const renderNodes = () => {
    return (
      <g>
        {validNodes.map(nodeId => {
          const coord = coordinates[nodeId];
          const pos = transformCoordinate(coord);
          const radius = Math.max(1.5, Math.min(8, coord.radius * 0.4));
          
          // Determine node color based on type
          const isCommunity = nodeId.startsWith('c0_');
          const isLeafNode = !isCommunity;
          
          let fillColor = '#06b6d4'; // Default cyan
          if (isCommunity) {
            fillColor = '#ef4444'; // Red for communities
          } else if (isLeafNode) {
            fillColor = '#10b981'; // Green for leaf nodes
          }

          return (
            <g key={nodeId}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={radius}
                fill={fillColor}
                stroke="#fff"
                strokeWidth="1"
                className={`transition-all duration-200 ${
                  onNodeClick ? 'cursor-pointer hover:stroke-blue-500 hover:stroke-2' : ''
                }`}
                onClick={() => onNodeClick?.(nodeId)}
                filter="url(#shadow)"
                style={{ zIndex: 10 }} // Ensure nodes are above edges
              >
                <title>{nodeId}</title>
              </circle>
              
              {/* Optional labels for larger nodes */}
              {showLabels && radius > 4 && (
                <text
                  x={pos.x}
                  y={pos.y + 1}
                  textAnchor="middle"
                  className="text-xs fill-white font-medium pointer-events-none"
                  style={{ fontSize: Math.max(6, radius * 0.8) }}
                >
                  {nodeId.length > 8 ? `${nodeId.substring(0, 6)}...` : nodeId}
                </text>
              )}
            </g>
          );
        })}
      </g>
    );
  };

  return (
    <svg 
      className={`w-full h-48 border rounded bg-gray-50 ${className}`}
      viewBox="0 0 400 200"
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="1" dy="1" stdDeviation="1" floodColor="rgba(0,0,0,0.2)" />
        </filter>
        
        {/* Grid pattern for background */}
        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#f3f4f6" strokeWidth="0.5"/>
        </pattern>
      </defs>
      
      {/* Background grid */}
      <rect width="100%" height="100%" fill="url(#grid)" />
      
      {/* Render edges first (so they appear behind nodes) */}
      {renderEdges()}
      
      {/* Render nodes on top */}
      {renderNodes()}
      
      {/* Stats overlay */}
      <g className="pointer-events-none">
        {/* Node count */}
        <text
          x={width - 5}
          y={height - 5}
          textAnchor="end"
          className="text-xs fill-gray-600 font-medium"
        >
          {validNodes.length} nodes
        </text>
        
        {/* Edge count */}
        {edges.length > 0 && (
          <text
            x={5}
            y={height - 5}
            textAnchor="start"
            className="text-xs fill-gray-600 font-medium"
          >
            {edges.length} edges
          </text>
        )}
      </g>
    </svg>
  );
};

// Specialized version for community preview with edge support
export const CommunityPreview: React.FC<{
  communityId: string;
  nodes: string[];
  coordinates: Record<string, NodePosition>;
  edges?: ClusterEdge[];
  showEdgeLoading?: boolean;
  onClick?: () => void;
}> = ({ communityId, nodes, coordinates, edges, showEdgeLoading, onClick }) => {
  return (
    <div className="space-y-2 w-full">
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-blue-600 truncate">
          {communityId}
        </span>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span>{nodes.length} nodes</span>
          {edges && edges.length > 0 && (
            <span>• {edges.length} edges</span>
          )}
        </div>
      </div>
      
      <div className="cursor-pointer w-full" onClick={onClick}>
        <CommunityStats 
          nodes={nodes} 
          edges={edges} 
          className="mb-2" 
        />

        <MiniGraph
          nodes={nodes}
          coordinates={coordinates}
          edges={edges}
          showEdgeLoading={showEdgeLoading}
          width={200}
          height={200}
          showLabels={false}
          className="hover:border-blue-300 transition-colors"
          edgeConfig={{
            opacity: 0.7,
            maxWidth: 2.5,
            color: '#6b7280',
            animate: true
          }}
        />
      </div>
    </div>
  );
};
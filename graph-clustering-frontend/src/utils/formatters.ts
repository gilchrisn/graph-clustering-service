// utils/formatters.ts

/**
 * Format duration from milliseconds to human readable string
 */
export const formatDuration = (ms: number): string => {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`;
  return `${(ms / 3600000).toFixed(1)}h`;
};

/**
 * Format modularity score with appropriate precision
 */
export const formatModularity = (modularity: number): string => {
  return modularity.toFixed(3);
};

/**
 * Format clustering parameters as readable string
 */
export const formatParameterSet = (parameters: Record<string, any>): string => {
  return Object.entries(parameters)
    .map(([key, value]) => {
      // Format specific parameter types
      if (key === 'minModularityGain') {
        return `${key}=${(value as number).toExponential(1)}`;
      }
      if (typeof value === 'number' && value < 1) {
        return `${key}=${value.toFixed(3)}`;
      }
      return `${key}=${value}`;
    })
    .join(', ');
};

/**
 * Format file size in bytes to human readable format
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Format large numbers with appropriate units
 */
export const formatNumber = (num: number): string => {
  if (num < 1000) return num.toString();
  if (num < 1000000) return `${(num / 1000).toFixed(1)}K`;
  if (num < 1000000000) return `${(num / 1000000).toFixed(1)}M`;
  return `${(num / 1000000000).toFixed(1)}B`;
};

/**
 * Format date to readable string
 */
export const formatDate = (date: string | Date): string => {
  const d = new Date(date);
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

/**
 * Format relative time (e.g., "2 minutes ago")
 */
export const formatRelativeTime = (date: string | Date): string => {
  const now = new Date();
  const then = new Date(date);
  const diffMs = now.getTime() - then.getTime();
  
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);
  
  if (diffSeconds < 60) return 'just now';
  if (diffMinutes < 60) return `${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
  if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
  
  return formatDate(date);
};

/**
 * Format percentage with appropriate precision
 */
export const formatPercentage = (value: number, total: number): string => {
  if (total === 0) return '0%';
  const percentage = (value / total) * 100;
  if (percentage < 0.1) return '<0.1%';
  if (percentage < 1) return `${percentage.toFixed(1)}%`;
  return `${Math.round(percentage)}%`;
};

/**
 * Truncate text with ellipsis
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return `${text.substring(0, maxLength - 3)}...`;
};

/**
 * Format experiment status with proper capitalization
 */
export const formatStatus = (status: string): string => {
  return status.charAt(0).toUpperCase() + status.slice(1).toLowerCase();
};

/**
 * Format algorithm name for display
 */
export const formatAlgorithmName = (algorithm: string): string => {
  switch (algorithm.toLowerCase()) {
    case 'louvain':
      return 'Louvain';
    case 'scar':
      return 'SCAR';
    default:
      return algorithm.toUpperCase();
  }
};

/**
 * Color utilities for consistent theming
 */
export const getStatusColor = (status: string): { bg: string; text: string; border: string } => {
  switch (status.toLowerCase()) {
    case 'running':
      return { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-200' };
    case 'completed':
      return { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-200' };
    case 'failed':
      return { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-200' };
    case 'configuring':
      return { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200' };
    default:
      return { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200' };
  }
};

/**
 * Get algorithm color for consistent theming
 */
export const getAlgorithmColor = (algorithm: string): { bg: string; text: string } => {
  switch (algorithm.toLowerCase()) {
    case 'louvain':
      return { bg: 'bg-purple-100', text: 'text-purple-700' };
    case 'scar':
      return { bg: 'bg-orange-100', text: 'text-orange-700' };
    default:
      return { bg: 'bg-gray-100', text: 'text-gray-700' };
  }
};

/**
 * Generate a readable ID from timestamp and random string
 */
export const generateReadableId = (prefix: string = 'item'): string => {
  const timestamp = Date.now().toString(36);
  const randomStr = Math.random().toString(36).substring(2, 8);
  return `${prefix}_${timestamp}_${randomStr}`;
};

/**
 * Validate and format coordinates for display
 */
export const formatCoordinate = (coord: number): string => {
  if (isNaN(coord)) return '0.0';
  return coord.toFixed(1);
};

/**
 * Format node radius for consistent display
 */
export const formatRadius = (radius: number): string => {
  if (isNaN(radius) || radius < 0) return '1.0';
  return Math.max(1.0, radius).toFixed(1);
};
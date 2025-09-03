// hooks/useClusterEdges.ts
import { useState, useEffect, useRef, useCallback } from 'react';
import { apiClient } from '../services/clusteringApi';
import { ClusterEdge, ClusterDetails } from '../types/api';

interface CacheEntry {
  data: ClusterDetails;
  timestamp: number;
}

interface UseClusterEdgesOptions {
  enabled?: boolean;
  cacheTTL?: number; // milliseconds, default 5 minutes
}

interface UseClusterEdgesReturn {
  edges: ClusterEdge[];
  loading: boolean;
  error: string | null;
  refetch: () => void;
  clusterDetails: ClusterDetails | null;
}

// Simple in-memory cache with TTL
class EdgeCache {
  private cache = new Map<string, CacheEntry>();
  private maxEntries = 100;
  private defaultTTL = 5 * 60 * 1000; // 5 minutes

  private isExpired(entry: CacheEntry, ttl: number): boolean {
    return Date.now() - entry.timestamp > ttl;
  }

  private evictExpired(ttl: number): void {
    for (const [key, entry] of this.cache.entries()) {
      if (this.isExpired(entry, ttl)) {
        this.cache.delete(key);
      }
    }
  }

  private evictOldest(): void {
    if (this.cache.size >= this.maxEntries) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }
  }

  get(key: string, ttl: number = this.defaultTTL): ClusterDetails | null {
    this.evictExpired(ttl);
    
    const entry = this.cache.get(key);
    if (!entry || this.isExpired(entry, ttl)) {
      return null;
    }
    
    return entry.data;
  }

  set(key: string, data: ClusterDetails): void {
    this.evictOldest();
    
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }
}

// Singleton cache instance
const edgeCache = new EdgeCache();

export const useClusterEdges = (
  datasetId: string,
  clusterId: string,
  jobId: string,
  options: UseClusterEdgesOptions = {}
): UseClusterEdgesReturn => {
  const { enabled = true, cacheTTL = 5 * 60 * 1000 } = options;
  
  const [clusterDetails, setClusterDetails] = useState<ClusterDetails | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Track current request to handle cleanup
  const currentRequestRef = useRef<AbortController | null>(null);
  
  // Generate cache key
  const cacheKey = `${datasetId}:${clusterId}:${jobId}`;

  const fetchClusterEdges = useCallback(async (forceRefresh = false) => {
    if (!enabled || !datasetId || !clusterId || !jobId) {
      return;
    }

    // Check cache first (unless force refresh)
    if (!forceRefresh) {
      const cachedData = edgeCache.get(cacheKey, cacheTTL);
      if (cachedData) {
        setClusterDetails(cachedData);
        setError(null);
        return;
      }
    }

    // Cancel previous request if still pending
    if (currentRequestRef.current) {
      currentRequestRef.current.abort();
    }

    // Create new abort controller for this request
    const abortController = new AbortController();
    currentRequestRef.current = abortController;

    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.getClusterDetails(datasetId, clusterId, jobId);
      
      // Check if request was aborted
      if (abortController.signal.aborted) {
        return;
      }

      const details = response.data;
      
      // Update state
      setClusterDetails(details);
      
      // Cache the result
      edgeCache.set(cacheKey, details);
      
    } catch (err) {
      // Don't set error if request was aborted
      if (!abortController.signal.aborted) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch cluster edges';
        setError(errorMessage);
        setClusterDetails(null);
      }
    } finally {
      // Only clear loading if this is still the current request
      if (!abortController.signal.aborted) {
        setLoading(false);
      }
      
      // Clear the ref if this was the current request
      if (currentRequestRef.current === abortController) {
        currentRequestRef.current = null;
      }
    }
  }, [datasetId, clusterId, jobId, enabled, cacheKey, cacheTTL]);

  const refetch = useCallback(() => {
    fetchClusterEdges(true);
  }, [fetchClusterEdges]);

  // Effect to clear state when hook is disabled
  useEffect(() => {
    if (!enabled) {
      // Clear state immediately when disabled
      setClusterDetails(null);
      setError(null);
      setLoading(false);
      
      // Cancel any pending requests
      if (currentRequestRef.current) {
        currentRequestRef.current.abort();
        currentRequestRef.current = null;
      }
    }
  }, [enabled]);

  // Effect to fetch data when dependencies change
  useEffect(() => {
    if (enabled) {
      fetchClusterEdges();
    }
    
    // Cleanup function to abort request if component unmounts
    return () => {
      if (currentRequestRef.current) {
        currentRequestRef.current.abort();
        currentRequestRef.current = null;
      }
    };
  }, [fetchClusterEdges, enabled]);

  // Extract edges from cluster details
  const edges = clusterDetails?.edges || [];

  return {
    edges,
    loading,
    error,
    refetch,
    clusterDetails
  };
};

// Utility hook for cache management
export const useEdgeCache = () => {
  const clearCache = useCallback(() => {
    edgeCache.clear();
  }, []);

  const getCacheSize = useCallback(() => {
    return edgeCache.size();
  }, []);

  return {
    clearCache,
    getCacheSize
  };
};
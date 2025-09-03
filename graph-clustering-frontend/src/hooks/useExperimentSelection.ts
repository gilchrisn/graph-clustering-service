// hooks/useExperimentSelection.ts 
import { useState, useCallback } from 'react';

export const useExperimentSelection = () => {
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedExperiments, setSelectedExperiments] = useState<Set<string>>(new Set());
  
  const toggleSelection = useCallback((experimentId: string) => {
    setSelectedExperiments(prev => {
      const newSelected = new Set(prev);
      if (newSelected.has(experimentId)) {
        newSelected.delete(experimentId);
      } else {
        newSelected.add(experimentId);
      }
      return newSelected;
    });
  }, []);
  
  const clearSelection = useCallback(() => {
    setSelectedExperiments(new Set());
    setSelectionMode(false);
  }, []);
  
  const enableSelectionMode = useCallback((completedExperimentIds: string[] = []) => {
    setSelectionMode(true);
    setSelectedExperiments(new Set(completedExperimentIds));
  }, []);
  
  return {
    selectionMode,
    selectedExperiments,
    toggleSelection,
    clearSelection,
    enableSelectionMode
  };
};
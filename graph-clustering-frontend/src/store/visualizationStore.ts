// store/visualizationStore.ts
import { create } from 'zustand';
import { Experiment, HierarchyPath } from '../types/visualizations';
import { Dataset } from '../types/api';

interface VisualizationStore {
  // State
  datasets: Dataset[];
  currentDataset: Dataset | null;
  experiments: Experiment[];
  currentViz: string;
  hierarchyPaths: Record<string, HierarchyPath>;
  
  // Dataset actions
  addDataset: (dataset: Dataset) => void;
  setCurrentDataset: (dataset: Dataset | null) => void;
  removeDataset: (datasetId: string) => void;
  
  // Experiment actions
  addExperiment: (experiment: Experiment) => void;
  updateExperiment: (id: string, updates: Partial<Experiment>) => void;
  removeExperiment: (id: string) => void;
  getExperimentsByDataset: (datasetId: string) => Experiment[];
  
  // Visualization actions
  setCurrentViz: (viz: string) => void;
  setHierarchyPath: (experimentId: string, path: HierarchyPath) => void;
  
  // Utility actions
  reset: () => void;
}

export const useVisualizationStore = create<VisualizationStore>((set, get) => ({
  // Initial state
  datasets: [],
  currentDataset: null,
  experiments: [],
  currentViz: 'timeline-hierarchy',
  hierarchyPaths: {},
  
  // Dataset actions
  addDataset: (dataset) => set((state) => ({ 
    datasets: [...state.datasets, dataset],
    currentDataset: dataset
  })),
  
  setCurrentDataset: (dataset) => set({ currentDataset: dataset }),
  
  removeDataset: (datasetId) => set((state) => ({
    datasets: state.datasets.filter(d => d.id !== datasetId),
    experiments: state.experiments.filter(e => e.datasetId !== datasetId),
    currentDataset: state.currentDataset?.id === datasetId ? null : state.currentDataset
  })),
  
  // Experiment actions
  addExperiment: (experiment) => set((state) => ({
    experiments: [...state.experiments, experiment]
  })),
  
  updateExperiment: (id, updates) => set((state) => ({
    experiments: state.experiments.map(exp => 
      exp.id === id ? { ...exp, ...updates } : exp
    )
  })),
  
  removeExperiment: (id) => set((state) => ({
    experiments: state.experiments.filter(exp => exp.id !== id)
  })),
  
  getExperimentsByDataset: (datasetId) => {
    const state = get();
    return state.experiments.filter(exp => exp.datasetId === datasetId);
  },
  
  // Visualization actions
  setCurrentViz: (viz) => set({ currentViz: viz }),
  
  setHierarchyPath: (experimentId, path) => set((state) => ({
    hierarchyPaths: {
      ...state.hierarchyPaths,
      [experimentId]: path
    }
  })),
  
  // Utility actions
  reset: () => set({
    datasets: [],
    currentDataset: null,
    experiments: [],
    currentViz: 'timeline-hierarchy',
    hierarchyPaths: {}
  })
}));
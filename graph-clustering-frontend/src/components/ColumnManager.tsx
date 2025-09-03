
import React, { useState } from 'react';
import { Experiment } from '../types/visualizations';

interface ColumnManagerProps {
  experiments: Experiment[];
  hiddenExperiments: string[];
  columnOrder: string[];
  onToggleExperiment: (experimentId: string) => void;
  onReorderColumns: (newOrder: string[]) => void;
}

export const ColumnManager: React.FC<ColumnManagerProps> = ({
  experiments,
  hiddenExperiments,
  onToggleExperiment
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const visibleCount = experiments.length - hiddenExperiments.length;

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 text-sm border border-gray-300 rounded-md hover:bg-gray-50"
      >
        <span>📋</span>
        <span>Columns ({visibleCount}/{experiments.length})</span>
      </button>

      {isOpen && (
        <div className="absolute top-full right-0 mt-2 w-80 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
          <div className="p-3 border-b border-gray-200">
            <h4 className="font-medium text-gray-900">Manage Columns</h4>
            <p className="text-sm text-gray-600">Show/hide experiment columns</p>
          </div>
          
          <div className="max-h-60 overflow-y-auto">
            {experiments.map((experiment) => {
              const isHidden = hiddenExperiments.includes(experiment.id);
              
              return (
                <div key={experiment.id} className="flex items-center gap-3 p-3 hover:bg-gray-50">
                  <input
                    type="checkbox"
                    checked={!isHidden}
                    onChange={() => onToggleExperiment(experiment.id)}
                    className="rounded border-gray-300"
                  />
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 text-xs rounded font-medium ${
                        experiment.algorithm === 'louvain' 
                          ? 'bg-purple-100 text-purple-700' 
                          : 'bg-orange-100 text-orange-700'
                      }`}>
                        {experiment.algorithm.toUpperCase()}
                      </span>
                      <span className="text-sm font-medium text-gray-900 truncate">
                        Exp {experiment.id.slice(-4)}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {new Date(experiment.createdAt).toLocaleString()}
                    </div>
                  </div>
                  
                  <div className={`w-2 h-2 rounded-full ${
                    experiment.status === 'completed' ? 'bg-green-500' :
                    experiment.status === 'running' ? 'bg-blue-500 animate-pulse' :
                    experiment.status === 'failed' ? 'bg-red-500' :
                    'bg-gray-400'
                  }`} />
                </div>
              );
            })}
          </div>
          
          <div className="p-3 border-t border-gray-200 bg-gray-50">
            <div className="flex gap-2">
              <button
                onClick={() => experiments.forEach(exp => 
                  onToggleExperiment(exp.id)
                )}
                className="text-sm text-blue-600 hover:text-blue-700"
              >
                Show All
              </button>
              <button
                onClick={() => experiments.forEach(exp => 
                  !hiddenExperiments.includes(exp.id) && onToggleExperiment(exp.id)
                )}
                className="text-sm text-gray-600 hover:text-gray-700"
              >
                Hide All
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
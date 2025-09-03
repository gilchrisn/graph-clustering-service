// App.tsx - Cleaned up version with proper imports
import React, { useState } from 'react';
import { useVisualizationStore } from './store/visualizationStore';
import { DatasetUpload } from './components/DatasetUpload';
import { ExperimentControls } from './components/ExperimentControls';
import { TimelineHierarchy } from './components/TimelineHierarchy';
import { Button } from './components/ui/Button';

const App: React.FC = () => {
  const { currentDataset } = useVisualizationStore();
  const [showUpload, setShowUpload] = useState(false);

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div>
              <h1 className="text-xl font-bold text-gray-900">Graph Clustering Visualization</h1>
              {currentDataset && (
                <p className="text-sm text-gray-600">Current: {currentDataset.name}</p>
              )}
            </div>
            
            <div className="flex gap-2">
              <Button onClick={() => setShowUpload(true)}>
                Upload Dataset
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {showUpload && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold">Upload New Dataset</h2>
                  <button
                    onClick={() => setShowUpload(false)}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    ✕
                  </button>
                </div>
                <DatasetUpload onComplete={() => setShowUpload(false)} />
              </div>
            </div>
          </div>
        )}

        {currentDataset && (
          <div className="space-y-6">
            <ExperimentControls />
            <div className="bg-white rounded-lg shadow h-[600px]">
              <TimelineHierarchy />
            </div>
          </div>
        )}

        {!currentDataset && (
          <div className="text-center py-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Welcome to Graph Clustering Visualization
            </h2>
            <p className="text-gray-600 mb-8">
              Upload your graph dataset to get started with clustering analysis
            </p>
            <Button onClick={() => setShowUpload(true)} size="lg">
              Upload Your First Dataset
            </Button>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
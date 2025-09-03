// components/DatasetUpload.tsx
import React, { useState } from 'react';
import { useVisualizationStore } from '../store/visualizationStore';
import { apiClient } from '../services/clusteringApi';
import { Dataset } from '../types/api';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { Loading } from './ui/Loading';

export interface DatasetUploadProps {
  onComplete: () => void;
  onCancel?: () => void;
}

interface UploadFiles {
  graphFile?: File;
  propertiesFile?: File;
  pathFile?: File;
}

export const DatasetUpload: React.FC<DatasetUploadProps> = ({ onComplete, onCancel }) => {
  const [name, setName] = useState('');
  const [files, setFiles] = useState<UploadFiles>({});
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { addDataset } = useVisualizationStore();

  const handleFileChange = (type: keyof UploadFiles) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setFiles(prev => ({ ...prev, [type]: file }));
      setError(null); // Clear error when file is selected
    }
  };

  const validateForm = (): boolean => {
    if (!name.trim()) {
      setError('Please provide a dataset name');
      return false;
    }
    
    if (!files.graphFile || !files.propertiesFile || !files.pathFile) {
      setError('Please provide all required files');
      return false;
    }
    
    return true;
  };

  const handleSubmit = async () => {
    if (!validateForm()) return;

    setUploading(true);
    setError(null);

    try {
      const response = await apiClient.uploadDataset(name.trim(), {
        graphFile: files.graphFile!,
        propertiesFile: files.propertiesFile!,
        pathFile: files.pathFile!
      });

      const dataset: Dataset = {
        id: response.data.datasetId,
        name: name.trim(),
        uploadedAt: new Date().toISOString()
      };

      addDataset(dataset);
      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const getFileDisplayName = (file: File | undefined): string => {
    if (!file) return 'No file selected';
    return file.name.length > 30 ? `${file.name.substring(0, 27)}...` : file.name;
  };

  const isFileSelected = (file: File | undefined): boolean => !!file;

  if (uploading) {
    return (
      <Card className="max-w-md mx-auto">
        <Loading 
          message="Uploading dataset..." 
          variant="spinner" 
          size="lg" 
        />
      </Card>
    );
  }

  return (
    <Card className="max-w-2xl mx-auto">
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900">Upload Dataset</h2>
          <p className="text-gray-600 mt-2">
            Upload your graph files to start clustering analysis
          </p>
        </div>
        
        <div className="space-y-4">
          {/* Dataset Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Dataset Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-md text-gray-900"
              placeholder="My Graph Dataset"
              maxLength={100}
            />
          </div>

          {/* File Upload Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Graph File */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Graph File *
              </label>
              <div className="relative">
                <input
                  type="file"
                  onChange={handleFileChange('graphFile')}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  accept=".txt,.csv"
                />
                <div className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
                  isFileSelected(files.graphFile) 
                    ? 'border-green-400 bg-green-50' 
                    : 'border-gray-300 hover:border-gray-400'
                }`}>
                  <div className="text-sm">
                    {isFileSelected(files.graphFile) ? (
                      <div>
                        <div className="text-green-600 font-medium">✓ Selected</div>
                        <div className="text-gray-600 mt-1 break-all">
                          {getFileDisplayName(files.graphFile)}
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="text-gray-600">Click to select</div>
                        <div className="text-xs text-gray-500 mt-1">Edge list file</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">Format: source target [weight]</p>
            </div>

            {/* Properties File */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Properties File *
              </label>
              <div className="relative">
                <input
                  type="file"
                  onChange={handleFileChange('propertiesFile')}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  accept=".txt,.csv"
                />
                <div className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
                  isFileSelected(files.propertiesFile) 
                    ? 'border-green-400 bg-green-50' 
                    : 'border-gray-300 hover:border-gray-400'
                }`}>
                  <div className="text-sm">
                    {isFileSelected(files.propertiesFile) ? (
                      <div>
                        <div className="text-green-600 font-medium">✓ Selected</div>
                        <div className="text-gray-600 mt-1 break-all">
                          {getFileDisplayName(files.propertiesFile)}
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="text-gray-600">Click to select</div>
                        <div className="text-xs text-gray-500 mt-1">Node types file</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">Format: nodeId typeId</p>
            </div>

            {/* Path File */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Path File *
              </label>
              <div className="relative">
                <input
                  type="file"
                  onChange={handleFileChange('pathFile')}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  accept=".txt,.csv"
                />
                <div className={`border-2 border-dashed rounded-lg p-4 text-center transition-colors ${
                  isFileSelected(files.pathFile) 
                    ? 'border-green-400 bg-green-50' 
                    : 'border-gray-300 hover:border-gray-400'
                }`}>
                  <div className="text-sm">
                    {isFileSelected(files.pathFile) ? (
                      <div>
                        <div className="text-green-600 font-medium">✓ Selected</div>
                        <div className="text-gray-600 mt-1 break-all">
                          {getFileDisplayName(files.pathFile)}
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="text-gray-600">Click to select</div>
                        <div className="text-xs text-gray-500 mt-1">Meta-path file</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-1">Format: typeId (one per line)</p>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded-md">
            <div className="text-sm">{error}</div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3">
          {onCancel && (
            <Button variant="secondary" onClick={onCancel}>
              Cancel
            </Button>
          )}
          <Button 
            onClick={handleSubmit}
            disabled={!name.trim() || !files.graphFile || !files.propertiesFile || !files.pathFile}
          >
            Upload Dataset
          </Button>
        </div>
      </div>
    </Card>
  );
};
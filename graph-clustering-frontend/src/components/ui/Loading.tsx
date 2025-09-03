// components/ui/Loading.tsx
import React from 'react';

export interface LoadingProps {
  message?: string;
  progress?: number;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'spinner' | 'pulse' | 'progress';
}

const Spinner: React.FC<{ size: string }> = ({ size }) => (
  <div className={`animate-spin rounded-full border-2 border-gray-300 border-t-blue-600 ${size}`} />
);

const Pulse: React.FC = () => (
  <div className="flex space-x-1">
    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" />
    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.1s' }} />
    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
  </div>
);

const sizeClasses = {
  sm: 'w-4 h-4',
  md: 'w-6 h-6',
  lg: 'w-8 h-8'
};

export const Loading: React.FC<LoadingProps> = ({ 
  message = 'Loading...', 
  progress,
  size = 'md',
  variant = 'spinner'
}) => {
  return (
    <div className="flex flex-col items-center justify-center space-y-3">
      {variant === 'spinner' && <Spinner size={sizeClasses[size]} />}
      {variant === 'pulse' && <Pulse />}
      
      {message && (
        <p className="text-sm text-gray-600 text-center">{message}</p>
      )}
      
      {progress !== undefined && (
        <div className="w-full max-w-xs">
          <div className="bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 text-center mt-1">
            {Math.round(progress || 0)}%
          </div>
        </div>
      )}
      
      {variant === 'progress' && progress === undefined && (
        <div className="w-full max-w-xs">
          <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
            <div className="bg-blue-600 h-2 rounded-full animate-pulse" />
          </div>
        </div>
      )}
    </div>
  );
};
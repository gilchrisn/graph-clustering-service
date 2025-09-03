
// components/ui/FullscreenModal.tsx
import React, { useEffect } from 'react';

export interface FullscreenModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
}

export const FullscreenModal: React.FC<FullscreenModalProps> = ({
  isOpen,
  onClose,
  children,
  title
}) => {
  // Escape key handling
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-white">
      {/* Header Bar */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-3">
          {title && (
            <h2 className="text-xl font-bold text-gray-900">{title}</h2>
          )}
          <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full font-medium">
            Fullscreen Mode
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Escape hint */}
          <span className="text-xs text-gray-500 hidden sm:block">
            Press ESC to exit
          </span>
          
          {/* Close button */}
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
            title="Exit fullscreen (ESC)"
          >
            {/* Exit fullscreen icon */}
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path>
            </svg>
          </button>
        </div>
      </div>

      {/* Content Area */}
      <div className="h-[calc(100vh-73px)] overflow-hidden">
        {children}
      </div>
    </div>
  );
};
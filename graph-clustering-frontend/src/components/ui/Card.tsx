// components/ui/Card.tsx
import React from 'react';

export interface CardProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  selected?: boolean;
  hoverable?: boolean;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

const paddingClasses = {
  none: '',
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6'
};

export const Card: React.FC<CardProps> = ({ 
  children, 
  className = '', 
  onClick, 
  selected = false,
  hoverable = true,
  padding = 'md'
}) => {
  const baseClasses = 'bg-white rounded-lg shadow border';
  const interactiveClasses = onClick ? 'cursor-pointer' : '';
  const selectedClasses = selected ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-200';
  const hoverClasses = hoverable && onClick ? 'hover:shadow-md hover:border-gray-300 transition-all duration-200' : '';
  
  const classes = [
    baseClasses,
    paddingClasses[padding],
    interactiveClasses,
    selectedClasses,
    hoverClasses,
    className
  ].filter(Boolean).join(' ');

  return (
    <div 
      className={classes}
      onClick={onClick}
    >
      {children}
    </div>
  );
};
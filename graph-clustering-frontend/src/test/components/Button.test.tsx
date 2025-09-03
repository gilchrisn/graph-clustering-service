// import { render, screen, fireEvent } from '@testing-library/react';
// import { describe, it, expect, vi } from 'vitest';
// import { Button } from '../../components/ui/Button';

// describe('Button Component', () => {
//   it('renders with correct text', () => {
//     render(<Button>Click me</Button>);
//     expect(screen.getByRole('button', { name: /click me/i })).toBeInTheDocument();
//   });

//   it('calls onClick handler when clicked', () => {
//     const handleClick = vi.fn();
//     render(<Button onClick={handleClick}>Click me</Button>);
    
//     fireEvent.click(screen.getByRole('button'));
//     expect(handleClick).toHaveBeenCalledTimes(1);
//   });

//   it('is disabled when disabled prop is true', () => {
//     render(<Button disabled>Disabled button</Button>);
//     expect(screen.getByRole('button')).toBeDisabled();
//   });

//   it('applies correct variant classes', () => {
//     const { rerender } = render(<Button variant="primary">Primary</Button>);
//     expect(screen.getByRole('button')).toHaveClass('bg-blue-600');

//     rerender(<Button variant="secondary">Secondary</Button>);
//     expect(screen.getByRole('button')).toHaveClass('bg-gray-200');

//     rerender(<Button variant="danger">Danger</Button>);
//     expect(screen.getByRole('button')).toHaveClass('bg-red-600');
//   });

//   it('applies correct size classes', () => {
//     const { rerender } = render(<Button size="sm">Small</Button>);
//     expect(screen.getByRole('button')).toHaveClass('px-3', 'py-1.5', 'text-sm');

//     rerender(<Button size="lg">Large</Button>);
//     expect(screen.getByRole('button')).toHaveClass('px-6', 'py-3', 'text-lg');
//   });

//   it('applies custom className', () => {
//     render(<Button className="custom-class">Custom</Button>);
//     expect(screen.getByRole('button')).toHaveClass('custom-class');
//   });

//   it('does not call onClick when disabled', () => {
//     const handleClick = vi.fn();
//     render(<Button onClick={handleClick} disabled>Disabled</Button>);
    
//     fireEvent.click(screen.getByRole('button'));
//     expect(handleClick).not.toHaveBeenCalled();
//   });
// });
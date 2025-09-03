/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Custom color palette for graph visualization
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        // Node type colors
        node: {
          community: '#ef4444',
          leaf: '#10b981',
          supernode: '#f59e0b',
        },
        // Status colors
        status: {
          running: '#3b82f6',
          completed: '#10b981',
          failed: '#ef4444',
          configuring: '#6b7280',
        }
      },
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"Segoe UI"',
          'Roboto',
          '"Oxygen"',
          'Ubuntu',
          'Cantarell',
          '"Fira Sans"',
          '"Droid Sans"',
          '"Helvetica Neue"',
          'sans-serif'
        ],
        mono: [
          '"SF Mono"',
          'Monaco',
          'Inconsolata',
          '"Roboto Mono"',
          '"Source Code Pro"',
          'monospace'
        ]
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      minWidth: {
        '80': '20rem',
        '96': '24rem',
      },
      maxWidth: {
        '8xl': '88rem',
        '9xl': '96rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-in': 'slideIn 0.3s ease-out',
        'bounce-gentle': 'bounceGentle 1s ease-in-out infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideIn: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(0)' },
        },
        bounceGentle: {
          '0%, 100%': { 
            transform: 'translateY(-5%)',
            animationTimingFunction: 'cubic-bezier(0.8,0,1,1)'
          },
          '50%': { 
            transform: 'none',
            animationTimingFunction: 'cubic-bezier(0,0,0.2,1)'
          },
        },
      },
      boxShadow: {
        'inner-lg': 'inset 0 10px 15px -3px rgba(0, 0, 0, 0.1)',
        'colored': '0 10px 15px -3px rgba(59, 130, 246, 0.1)',
      },
      backdropBlur: {
        xs: '2px',
      },
      gridTemplateColumns: {
        'auto-fit': 'repeat(auto-fit, minmax(280px, 1fr))',
        'auto-fill': 'repeat(auto-fill, minmax(280px, 1fr))',
      },
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      }
    },
  },
  plugins: [
    // Add forms plugin for better form styling
    require('@tailwindcss/forms')({
      strategy: 'class',
    }),
    
    // Custom plugin for graph visualization utilities
    function({ addUtilities, theme }) {
      const newUtilities = {
        // Timeline specific utilities
        '.timeline-scroll': {
          scrollbarWidth: 'thin',
          scrollbarColor: `${theme('colors.gray.300')} ${theme('colors.gray.100')}`,
          '&::-webkit-scrollbar': {
            width: '8px',
            height: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: theme('colors.gray.100'),
          },
          '&::-webkit-scrollbar-thumb': {
            background: theme('colors.gray.300'),
            borderRadius: '4px',
          },
          '&::-webkit-scrollbar-thumb:hover': {
            background: theme('colors.gray.400'),
          },
        },
        
        // Node visualization utilities
        '.node-community': {
          fill: theme('colors.node.community'),
          stroke: theme('colors.white'),
          strokeWidth: '1px',
        },
        '.node-leaf': {
          fill: theme('colors.node.leaf'),
          stroke: theme('colors.white'),
          strokeWidth: '1px',
        },
        '.node-supernode': {
          fill: theme('colors.node.supernode'),
          stroke: theme('colors.white'),
          strokeWidth: '1px',
        },
        
        // Status indicators
        '.status-indicator': {
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.25rem',
          padding: '0.25rem 0.5rem',
          borderRadius: '0.375rem',
          fontSize: '0.75rem',
          fontWeight: '500',
        },
        
        // Card hover effects
        '.card-hover-lift': {
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: theme('boxShadow.lg'),
          },
        },
        
        // Loading states
        '.loading-skeleton': {
          backgroundColor: theme('colors.gray.200'),
          backgroundImage: `linear-gradient(
            90deg,
            ${theme('colors.gray.200')} 0%,
            ${theme('colors.gray.300')} 50%,
            ${theme('colors.gray.200')} 100%
          )`,
          backgroundSize: '200px 100%',
          backgroundRepeat: 'no-repeat',
          animation: 'loading 1.2s ease-in-out infinite',
        },
      };
      
      addUtilities(newUtilities);
    },
  ],
  
  // Safelist important classes that might be generated dynamically
  safelist: [
    'bg-blue-100',
    'bg-green-100', 
    'bg-red-100',
    'bg-gray-100',
    'bg-purple-100',
    'bg-orange-100',
    'text-blue-700',
    'text-green-700',
    'text-red-700',
    'text-gray-700',
    'text-purple-700',
    'text-orange-700',
    'border-blue-500',
    'border-green-500',
    'border-red-500',
    'border-gray-500',
    'border-purple-500',
    'border-orange-500',
  ],
}
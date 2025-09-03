# Frontend Architecture Guide

> **📋 Developer Reference**: Architecture patterns, conventions, and extension guidelines for the Graph Clustering Visualization frontend.

## 🏗️ Architecture Overview

This React application follows **Domain-Driven Design** principles with a **feature-based architecture** optimized for graph visualization workflows. The codebase is designed for extensibility, maintainability, and team collaboration.

### Core Design Principles

1. **🧩 Component Composition** - Small, reusable components that compose into complex UIs
2. **📦 Separation of Concerns** - Clear boundaries between UI, business logic, and data
3. **🔄 Unidirectional Data Flow** - Predictable state management with Zustand
4. **⚡ Performance First** - Optimized for large datasets and real-time updates
5. **🧪 Testability** - Every component and function designed for easy testing
6. **🚀 Scalability** - Architecture supports adding new visualization types

---

## 📁 Directory Structure Deep Dive

```
src/
├── components/          # UI Components (Presentation Layer)
│   ├── ui/             # ✨ Reusable UI primitives
│   └── [feature]/      # 🎯 Feature-specific components
├── hooks/              # 🪝 Custom React hooks (Logic Layer)
├── services/           # 🌐 External API integration
├── store/              # 📊 Global state management
├── types/              # 📝 TypeScript definitions
├── utils/              # 🛠️ Pure utility functions
├── styles/             # 🎨 Global styles and themes
└── test/               # 🧪 Testing utilities and mocks
```

### 🧩 Components Architecture

**Component Hierarchy:**
```
App (Root)
├── ErrorBoundary (Wrapper)
├── DatasetUpload (Feature)
├── ExperimentControls (Feature)
└── TimelineHierarchy (Feature)
    └── TimelineColumn[] (Composite)
        └── HierarchyLevelCard[] (Composite)
            └── MiniGraph (Primitive)
```

#### **UI Components (`components/ui/`)**

**Purpose**: Reusable, context-agnostic UI primitives

```typescript
// ✅ Good: Generic, reusable
<Button variant="primary" size="lg" onClick={handleClick}>
  Submit
</Button>

// ❌ Bad: Too specific, not reusable
<SubmitExperimentButton experimentData={data} />
```

**Rules:**
- No business logic or API calls
- Accept generic props via interfaces
- Include comprehensive prop types
- Support theming via Tailwind variants

#### **Feature Components (`components/`)**

**Purpose**: Domain-specific components that orchestrate UI primitives

```typescript
// ✅ Good: Composes UI primitives, handles feature logic
const ExperimentControls = () => {
  const { startExperiment } = useExperimentActions();
  const [params, setParams] = useState(defaultParams);
  
  return (
    <Card>
      <form onSubmit={handleSubmit}>
        <Input value={params.maxLevels} onChange={setMaxLevels} />
        <Button type="submit">Start Experiment</Button>
      </form>
    </Card>
  );
};
```

---

## 🔄 State Management Strategy

### Zustand Store Architecture

**Single Store, Multiple Slices:**
```typescript
interface VisualizationStore {
  // 📊 Data slices
  datasets: Dataset[];
  experiments: Experiment[];
  
  // 🎛️ UI state
  currentDataset: Dataset | null;
  currentViz: string;
  
  // ⚡ Actions (grouped by domain)
  addDataset: (dataset: Dataset) => void;
  addExperiment: (experiment: Experiment) => void;
  setCurrentViz: (viz: string) => void;
}
```

**State Update Patterns:**
```typescript
// ✅ Immutable updates
updateExperiment: (id, updates) => set((state) => ({
  experiments: state.experiments.map(exp => 
    exp.id === id ? { ...exp, ...updates } : exp
  )
}))

// ❌ Direct mutation
updateExperiment: (id, updates) => {
  const exp = state.experiments.find(e => e.id === id);
  exp.status = updates.status; // Mutates state!
}
```

### When to Use Local vs Global State

| Data Type | Storage | Example |
|-----------|---------|---------|
| **App-wide data** | Zustand Store | Datasets, experiments, current selection |
| **Component state** | useState | Form inputs, toggle states, local UI |
| **Derived data** | useMemo | Filtered lists, computed metrics |
| **Server state** | Custom hooks | API responses, loading states |

---

## 🪝 Custom Hooks Patterns

### Hook Categories

1. **🔄 Data Hooks** - Server state and synchronization
2. **🎮 Action Hooks** - Business logic orchestration  
3. **🛠️ Utility Hooks** - Reusable behaviors

**Example: Action Hook Pattern**
```typescript
// hooks/useExperimentActions.ts
export const useExperimentActions = () => {
  const { addExperiment, currentDataset } = useVisualizationStore();

  const startExperiment = useCallback(async (algorithm, parameters) => {
    // 1. Validation
    if (!currentDataset) throw new Error('No dataset selected');
    
    // 2. Optimistic update
    const experiment = createExperiment(algorithm, parameters);
    addExperiment(experiment);
    
    // 3. API call
    try {
      const response = await apiClient.startClustering(/*...*/);
      updateExperiment(experiment.id, { jobId: response.data.jobId });
    } catch (error) {
      updateExperiment(experiment.id, { status: 'failed', error });
    }
  }, [currentDataset, addExperiment]);

  return { startExperiment };
};
```

---

## 🌐 API Integration Architecture

### Service Layer Pattern

**Centralized API Client:**
```typescript
// services/clusteringApi.ts
export class ClusteringApiClient {
  private baseUrl = process.env.VITE_API_BASE_URL;

  async uploadDataset(name: string, files: FileSet): Promise<UploadResponse> {
    // Request/response handling with proper error boundaries
  }
}

export const apiClient = new ClusteringApiClient();
```

**Error Handling Strategy:**
```typescript
// ✅ Structured error handling
try {
  const response = await apiClient.startClustering(params);
  return response.data;
} catch (error) {
  if (error instanceof NetworkError) {
    showNotification('Connection failed. Check your network.');
  } else if (error instanceof ValidationError) {
    setFieldErrors(error.fieldErrors);
  } else {
    logError('Unexpected error', error);
    throw error; // Re-throw for error boundary
  }
}
```

### Real-time Data Patterns

**Polling with Cleanup:**
```typescript
const usePolling = (callback, interval, enabled) => {
  useEffect(() => {
    if (!enabled) return;

    const poll = async () => {
      const shouldContinue = await callback();
      if (shouldContinue) {
        timeoutId = setTimeout(poll, interval);
      }
    };

    poll();
    return () => clearTimeout(timeoutId);
  }, [callback, interval, enabled]);
};
```

---

## 🎨 Styling Architecture

### Tailwind + CSS Custom Properties

**Design System:**
```css
/* styles/global.css */
:root {
  --color-primary: #3b82f6;
  --color-node-community: #ef4444;
  --color-node-leaf: #10b981;
  --shadow-card: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
```

**Component Styling Patterns:**
```typescript
// ✅ Conditional classes with consistent patterns
const buttonClasses = [
  'px-4 py-2 rounded-md font-medium transition-colors',
  variant === 'primary' ? 'bg-blue-600 text-white hover:bg-blue-700' : '',
  disabled ? 'opacity-50 cursor-not-allowed' : '',
  className
].filter(Boolean).join(' ');

// ❌ Inline styles for dynamic values
<div style={{ width: `${progress}%` }} /> // Use CSS custom properties instead
```

### Responsive Design Strategy

**Mobile-First Approach:**
```typescript
// ✅ Progressive enhancement
<div className="
  flex flex-col          /* Mobile: stack vertically */
  md:flex-row           /* Tablet+: horizontal layout */
  lg:gap-6              /* Desktop: larger spacing */
  xl:max-w-7xl          /* Large screens: constrain width */
">
```

---

## 🧪 Testing Architecture

### Testing Strategy

1. **Unit Tests** - Pure functions and utilities
2. **Component Tests** - User interactions and rendering
3. **Integration Tests** - API clients and custom hooks
4. **E2E Tests** - Complete user workflows

**Component Testing Pattern:**
```typescript
// test/components/Button.test.tsx
describe('Button Component', () => {
  it('handles user interactions correctly', () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Click me</Button>);
    
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('applies accessibility attributes', () => {
    render(<Button disabled>Disabled</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
```

### Mock Strategies

**API Mocking:**
```typescript
// test/mocks/apiClient.ts
export const mockApiClient = {
  uploadDataset: vi.fn().mockResolvedValue({
    success: true,
    data: { datasetId: 'test-dataset-id' }
  }),
  startClustering: vi.fn().mockResolvedValue({
    success: true, 
    data: { jobId: 'test-job-id' }
  })
};
```

---

## 🚀 Extension Guidelines

### Adding New Visualization Types

**1. Create Visualization Component:**
```typescript
// components/visualizations/NetworkDashboard/NetworkDashboard.tsx
export const NetworkDashboard: React.FC<VisualizationProps> = ({ 
  experiments, 
  onExperimentSelect 
}) => {
  // Visualization-specific logic
  return <div>Network Dashboard View</div>;
};
```

**2. Register in Visualization Registry:**
```typescript
// registry/VisualizationRegistry.ts
export const VISUALIZATIONS = {
  'timeline-hierarchy': TimelineHierarchy,
  'network-dashboard': NetworkDashboard,  // Add here
  'animated-graph': AnimatedGraph,
} as const;
```

**3. Update Store Types:**
```typescript
// types/visualizations.ts
type VisualizationType = 
  | 'timeline-hierarchy' 
  | 'network-dashboard'  // Add here
  | 'animated-graph';
```

### Adding New Drill-Down Strategies

```typescript
// utils/drillDownEngine.ts
export const drillDownStrategies = {
  'most-leaf-nodes': { /* existing */ },
  'highest-centrality': {  // Add new strategy
    id: 'highest-centrality',
    name: 'Highest Centrality',
    description: 'Select nodes with highest betweenness centrality',
    selectBest: (communities, result) => {
      // Custom selection logic
      return bestCommunityId;
    }
  }
};
```

### Adding New Algorithm Support

**1. Update Types:**
```typescript
// types/visualizations.ts
type Algorithm = 'louvain' | 'scar' | 'leiden';  // Add 'leiden'
```

**2. Update UI:**
```typescript
// components/ExperimentControls.tsx
<select value={algorithm} onChange={setAlgorithm}>
  <option value="louvain">Louvain</option>
  <option value="scar">SCAR</option>
  <option value="leiden">Leiden</option>  {/* Add option */}
</select>
```

---

## 📋 Development Best Practices

### Code Organization

**✅ File Naming Conventions:**
- Components: `PascalCase.tsx` (e.g., `TimelineHierarchy.tsx`)
- Hooks: `camelCase.ts` with `use` prefix (e.g., `useExperiment.ts`)
- Utils: `camelCase.ts` (e.g., `drillDownEngine.ts`)
- Types: `camelCase.ts` (e.g., `visualizations.ts`)

**✅ Import Organization:**
```typescript
// 1. External libraries
import React, { useState, useEffect } from 'react';
import { create } from 'zustand';

// 2. Internal types
import { Experiment, Dataset } from '../types/visualizations';

// 3. Internal utilities
import { formatDuration } from '../utils/formatters';

// 4. Internal components
import { Button } from './ui/Button';
import { Card } from './ui/Card';
```

### Performance Guidelines

**✅ Optimization Patterns:**
```typescript
// Memoize expensive calculations
const processedData = useMemo(() => 
  computeHierarchyPath(experiments), [experiments]
);

// Memoize callback functions
const handleExperimentSelect = useCallback((id: string) => {
  setSelectedExperiment(id);
}, []);

// Split large components
const TimelineColumn = React.memo(({ experiment }) => {
  // Component implementation
});
```

**❌ Performance Anti-patterns:**
```typescript
// Don't create objects in render
<Component style={{ width: '100%' }} />  // Creates new object every render

// Don't use array index as key for dynamic lists
{items.map((item, index) => <Item key={index} />)}  // Use stable IDs

// Don't call hooks conditionally
if (condition) {
  const data = useData();  // Violates rules of hooks
}
```

### Error Handling Standards

**Component-Level Error Boundaries:**
```typescript
// Wrap risky components
<ErrorBoundary fallback={<GraphErrorFallback />}>
  <ComplexGraphVisualization data={data} />
</ErrorBoundary>
```

**Graceful Degradation:**
```typescript
// Always provide fallbacks for missing data
const MiniGraph = ({ coordinates, nodes }) => {
  if (!coordinates || Object.keys(coordinates).length === 0) {
    return <EmptyStateMessage />;
  }
  
  return <SVGVisualization />;
};
```

---

## 🔧 Development Workflow

### Getting Started (New Developer)

1. **Environment Setup:**
   ```bash
   git clone [repository]
   cd graph-clustering-frontend
   npm install
   npm run dev
   ```

2. **Understand the Data Flow:**
   ```
   User Upload → API Client → Zustand Store → Components → UI
   ```

3. **Start Small:**
   - Modify existing components
   - Add new UI variants
   - Extend existing hooks

4. **Follow Testing:**
   ```bash
   npm run test        # Run existing tests
   npm run test:ui     # Visual test interface
   npm run type-check  # TypeScript validation
   ```

### Feature Development Process

1. **📋 Planning:**
   - Identify which layer(s) need changes
   - Plan component composition
   - Consider state management needs

2. **🏗️ Implementation:**
   - Start with types and interfaces
   - Build from UI primitives up
   - Add business logic last

3. **🧪 Testing:**
   - Write tests as you develop
   - Test error conditions
   - Verify accessibility

4. **📚 Documentation:**
   - Update this architecture guide
   - Add inline code comments
   - Update README if needed

---

## 🎯 Architecture Decision Records (ADRs)

### Why Zustand over Redux?

**Decision**: Use Zustand for state management
**Reasoning**: 
- Simpler API with less boilerplate
- Better TypeScript integration
- Sufficient for app complexity
- Easier onboarding for new developers

### Why Tailwind over CSS Modules?

**Decision**: Use Tailwind CSS for styling
**Reasoning**:
- Faster development with utility classes
- Consistent design system
- Better collaboration (shared vocabulary)
- Easy responsive design

### Why Feature-based over Layer-based Architecture?

**Decision**: Organize by features, not technical layers
**Reasoning**:
- Features can be developed independently
- Easier to understand and modify
- Better encapsulation
- Supports team scaling

---

## 🤝 Contributing Guidelines

### Before Making Changes

1. **📖 Read this guide** completely
2. **🔍 Check existing patterns** for similar functionality
3. **💬 Discuss architecture changes** with the team
4. **✅ Run tests** to ensure nothing breaks

### Code Review Checklist

- [ ] Follows established patterns and conventions
- [ ] Includes appropriate TypeScript types
- [ ] Has test coverage for new functionality
- [ ] Handles error cases gracefully
- [ ] Follows accessibility guidelines
- [ ] Updates documentation if needed

---

**🎉 Happy coding!** This architecture is designed to grow with your needs while maintaining code quality and developer experience.
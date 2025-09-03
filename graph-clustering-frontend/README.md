# Graph Clustering Visualization Frontend

A modern React frontend for visualizing graph clustering results with an innovative timeline-based hierarchy exploration interface.

## 🎯 Features

- **Timeline Hierarchy Visualization**: Compare multiple clustering experiments side-by-side
- **Real-time Job Monitoring**: Live progress tracking for clustering operations  
- **Interactive Graph Visualization**: Mini-graphs with coordinate-based node positioning
- **Multi-Algorithm Support**: Louvain and SCAR clustering algorithms
- **Drill-down Exploration**: Navigate from root communities to leaf nodes
- **Responsive Design**: Works on desktop and mobile devices
- **Type-Safe**: Full TypeScript implementation
- **Modern Stack**: React 18, Zustand, Vite, Tailwind CSS

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Go backend server running on `localhost:8080`

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd graph-clustering-frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`

### Build for Production

```bash
# Build the application
npm run build

# Preview the build
npm run preview
```

## 📁 Project Structure

```
src/
├── components/                  # Reusable UI components
│   ├── ui/                     # Basic UI elements
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   └── Loading.tsx
│   ├── DatasetUpload.tsx       # Dataset upload functionality
│   ├── ExperimentControls.tsx  # Parameter input controls
│   ├── MiniGraph.tsx          # Graph visualization components
│   ├── HierarchyLevelCard.tsx # Hierarchy level display
│   ├── TimelineColumn.tsx     # Individual experiment column
│   ├── TimelineHierarchy.tsx  # Main timeline interface
│   └── ErrorBoundary.tsx      # Error handling
├── hooks/                      # Custom React hooks
│   └── useExperiment.ts       # Experiment management logic
├── services/                   # API integration
│   └── clusteringApi.ts       # Backend API client
├── store/                      # State management
│   └── visualizationStore.ts  # Zustand store
├── types/                      # TypeScript definitions
│   ├── api.ts                 # API response types
│   └── visualizations.ts     # App-specific types
├── utils/                      # Utility functions
│   ├── drillDownEngine.ts     # Hierarchy navigation logic
│   └── formatters.ts          # Data formatting utilities
├── styles/                     # Styling
│   └── global.css             # Global styles and Tailwind
├── test/                       # Test files
│   ├── setup.ts              # Test configuration
│   └── components/           # Component tests
└── App.tsx                     # Main application component
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
VITE_API_BASE_URL=http://localhost:8080/api/v1
VITE_APP_TITLE=Graph Clustering Visualization
```

### Backend Integration

The frontend expects a Go REST API with these endpoints:

- `POST /api/v1/datasets` - Upload dataset
- `POST /api/v1/datasets/{id}/clustering` - Start clustering job  
- `GET /api/v1/datasets/{id}/clustering/{jobId}` - Get job status
- `GET /api/v1/datasets/{id}/hierarchy?jobId={jobId}` - Get results

See the [API documentation](./docs/api.md) for detailed specifications.

## 🎨 Usage Guide

### 1. Upload Dataset

1. Click "Upload Dataset" 
2. Provide three files:
   - **Graph File**: Edge list (`source target [weight]`)
   - **Properties File**: Node types (`nodeId typeId`)  
   - **Path File**: Meta-path (`typeId` per line)
3. Give your dataset a name and upload

### 2. Create Experiments

1. Select algorithm (Louvain or SCAR)
2. Configure parameters:
   - **Max Levels**: Hierarchy depth (1-20)
   - **Max Iterations**: Optimization rounds (10-1000)
   - **Min Modularity Gain**: Convergence threshold
3. Click "Start Experiment"

### 3. Explore Results

- Each experiment appears as a **timeline column**
- **Hierarchy levels** are displayed top to bottom
- **Mini-graphs** show community structure with coordinates
- **Click columns** to select and compare
- **Drill down** through community hierarchy

## 🧪 Testing

```bash
# Run tests
npm run test

# Run tests with UI
npm run test:ui

# Type checking
npm run type-check

# Linting
npm run lint
```

### Test Structure

- **Unit tests**: Individual component testing
- **Integration tests**: API client and hooks
- **Visual tests**: Mini-graph rendering
- **End-to-end**: Full workflow testing

## 📊 Architecture Decisions

### State Management
- **Zustand** for global state (experiments, datasets)
- **React state** for local component state
- **Custom hooks** for reusable logic

### Styling Approach  
- **Tailwind CSS** for utility-first styling
- **CSS custom properties** for theming
- **Responsive design** with mobile-first approach

### API Integration
- **Fetch API** with TypeScript interfaces
- **Polling mechanism** for job status tracking
- **Error handling** with retry logic

### Performance Optimizations
- **Code splitting** by feature
- **Lazy loading** for visualization components  
- **Memoization** for expensive calculations
- **Debounced inputs** for parameter controls

## 🚨 Troubleshooting

### Common Issues

**Backend Connection Failed**
```bash
# Check if Go server is running
curl http://localhost:8080/api/v1/health

# Verify proxy configuration in vite.config.ts
```

**File Upload Errors**
- Ensure files are in correct format
- Check file size limits (backend configuration)
- Verify file encoding (UTF-8 recommended)

**Visualization Not Loading**
- Check browser console for errors
- Verify coordinates data in API response
- Ensure SVG support in browser

### Debug Mode

Set environment variable for detailed logging:
```bash
VITE_DEBUG=true npm run dev
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards

- **TypeScript**: Strict mode enabled
- **ESLint**: Follow configured rules
- **Prettier**: Code formatting
- **Tests**: Required for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- React team for the excellent framework
- Zustand for simple state management
- Tailwind CSS for utility-first styling
- Vite for lightning-fast development experience
- The graph clustering research community

---

**Built with ❤️ for graph analysis and visualization**
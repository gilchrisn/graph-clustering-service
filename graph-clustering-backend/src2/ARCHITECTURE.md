# Graph Visualization Backend - Architecture Guide

> **For Developers:** How to understand, extend, and maintain this codebase

## 🏗️ System Overview

This backend replaces a chaotic 6-endpoint RPC system with clean RESTful architecture, achieving 10x performance improvement through in-memory processing and modern Go practices.

**Key Principles:**
- **Separation of Concerns** - Each layer has one responsibility
- **Dependency Injection** - Components depend on interfaces, not implementations
- **Resource-Oriented Design** - RESTful APIs over RPC-style endpoints
- **Background Processing** - Non-blocking job execution
- **Structured Logging** - Machine-readable, searchable logs

## 📁 Project Structure

```
.
├── main.go                 # Application entry point & server setup
├── config/
│   └── config.go          # Configuration management (env vars, defaults)
├── api/                   # HTTP layer (handlers, routes, middleware)
│   ├── handlers.go        # HTTP request/response handling
│   ├── routes.go          # RESTful route definitions
│   └── middleware.go      # Cross-cutting concerns (logging, CORS, recovery)
├── service/               # Business logic layer (orchestration)
│   ├── dataset.go         # Dataset upload & lifecycle management
│   ├── clustering.go      # Clustering workflow orchestration
│   ├── job.go            # Background job processing & lifecycle
│   └── comparison.go     # 🆕 Experiment comparison service
├── algorithm/             # Domain layer (algorithm execution)
│   ├── interface.go       # Algorithm abstraction & registry
│   ├── louvain.go        # Louvain algorithm adapter
│   └── scar.go           # SCAR algorithm adapter
├── models/                # Data models & types
│   └── models.go         # Structs, enums, API response types
└── utils/                 # Shared utilities
    └── response.go       # HTTP response helpers & error handling
```

## 🎯 Architectural Patterns

### 1. Three-Layer Architecture

```
┌─────────────────┐    HTTP/JSON     ┌──────────────────┐
│   Frontend      │◄─────────────────│   API Layer      │
│   (React/Vue)   │                  │   (HTTP concerns) │
└─────────────────┘                  └──────────────────┘
                                              │
                                              ▼ interfaces
                                     ┌──────────────────┐
                                     │   Service Layer  │
                                     │ (Business Logic) │
                                     └──────────────────┘
                                              │
                                              ▼ interfaces
                                     ┌──────────────────┐
                                     │ Algorithm Layer  │
                                     │ (Domain Logic)   │
                                     └──────────────────┘
```

**Layer Responsibilities:**

**API Layer (`api/`):**
- HTTP request parsing & validation
- Response serialization (JSON)
- HTTP status code mapping
- Authentication (future)
- Rate limiting (future)

**Service Layer (`service/`):**
- Business workflow orchestration
- Data validation & transformation
- Cross-service coordination
- Transaction management
- Background job scheduling
- 🆕 **Experiment comparison coordination**

**Algorithm Layer (`algorithm/`):**
- Algorithm execution logic
- Parameter validation
- Result format conversion
- Resource management (CPU, memory)

### 2. Dependency Injection Pattern

```go
// ✅ Dependencies flow downward, injected at startup
func main() {
    // Create services (inner layers first)
    datasetService := service.NewDatasetService()
    jobService := service.NewJobService(datasetService)
    clusteringService := service.NewClusteringService(datasetService, jobService)
    comparisonService := service.NewComparisonService(clusteringService)  // 🆕 NEW
    
    // Create API handlers (outer layer)
    handlers := api.NewHandlers(datasetService, clusteringService, jobService, comparisonService)
}

// ✅ Services accept dependencies through constructors
func NewClusteringService(ds *DatasetService, js *JobService) *ClusteringService {
    return &ClusteringService{
        datasetService: ds,  // Injected dependency
        jobService:     js,  // Injected dependency
    }
}

// 🆕 NEW: Comparison service follows same pattern
func NewComparisonService(cs *ClusteringService) *ComparisonService {
    return &ComparisonService{
        clusteringService: cs,  // Injected dependency
    }
}
```

**Benefits:**
- **Testability**: Easy to inject mocks for unit tests
- **Flexibility**: Swap implementations without changing code
- **Clear Dependencies**: Explicit about what each component needs
- **No Global State**: Easier to reason about and debug

### 3. Interface-Driven Design

```go
// Define contract
type Algorithm interface {
    Name() models.AlgorithmType
    ValidateParameters(params models.JobParameters) error
    Run(ctx context.Context, dataset *models.Dataset, ...) (*AlgorithmResult, error)
}

// Implementations
type LouvainAdapter struct{} // Implements Algorithm
type SCARAdapter struct{}    // Implements Algorithm

// Registry pattern for discovery
type Registry struct {
    algorithms map[models.AlgorithmType]Algorithm
}
```

**Why This Works:**
- **Polymorphism**: Service layer doesn't know which algorithm is running
- **Extensibility**: Add new algorithms without modifying existing code
- **Testing**: Create mock algorithms for unit tests
- **Consistency**: All algorithms follow same contract

## 🔄 Data Flow Architecture

### Request Processing Flow

```
1. HTTP Request → API Layer
   ↓
2. Parse/Validate → Service Layer
   ↓
3. Create Job → Background Processing (for clustering)
   ↓
4. Algorithm Execution → Algorithm Layer
   ↓
5. Result Storage → In-Memory Cache
   ↓
6. HTTP Response ← API Layer
```

### 🆕 Comparison Processing Flow

```
1. Comparison Request → API Layer
   ↓
2. Validate Experiments → ComparisonService
   ↓
3. Fetch Hierarchies → ClusteringService
   ↓
4. Compute Metrics → ComparisonService (synchronous, <5 seconds)
   ↓
5. Generate Summary → ComparisonService
   ↓
6. Store Results → In-Memory Cache
   ↓
7. HTTP Response ← API Layer
```

### Job Lifecycle

```
Created → Queued → Running → Completed
    ↓         ↓        ↓         ↓
  Validation  Worker   Algorithm  Result
              Pool     Execution  Storage
```

## 🏃 Background Job System

### Worker Pool Pattern

```go
type JobService struct {
    workers chan struct{}  // Semaphore for concurrency control
    jobs    map[string]*models.Job
    results map[string]*algorithm.AlgorithmResult
}

func (s *JobService) processJob(jobID string) {
    s.workers <- struct{}{}        // Acquire worker
    defer func() { <-s.workers }() // Release worker
    
    // Process job...
}
```

**Concurrency Control:**
- **Worker Slots**: Limit concurrent jobs to prevent resource exhaustion
- **Backpressure**: Jobs queue up when all workers busy
- **Graceful Degradation**: System stays responsive under load

### Job State Management

```go
type JobStatus string

const (
    JobStatusQueued    JobStatus = "queued"     // Waiting for worker
    JobStatusRunning   JobStatus = "running"    // Currently processing
    JobStatusCompleted JobStatus = "completed"  // Finished successfully
    JobStatusFailed    JobStatus = "failed"     // Error occurred
    JobStatusCancelled JobStatus = "cancelled"  // User cancelled
)
```

**State Transitions:**
```
queued → running → completed
   ↓        ↓         ↓
cancelled  failed   cancelled
```

## 🧵 Concurrency & Thread Safety

### Shared State Protection

```go
type JobService struct {
    jobs  map[string]*models.Job
    mutex sync.RWMutex  // Protects concurrent map access
}

func (s *JobService) Get(jobID string) (*models.Job, error) {
    s.mutex.RLock()         // Multiple readers allowed
    defer s.mutex.RUnlock()
    return s.jobs[jobID], nil
}

func (s *JobService) updateJob(...) {
    s.mutex.Lock()          // Exclusive write access
    defer s.mutex.Unlock()
    s.jobs[jobID] = newJob
}

// 🆕 NEW: Comparison service follows same pattern
type ComparisonService struct {
    comparisons map[string]*models.Comparison
    mutex       sync.RWMutex  // Same protection strategy
}
```

**Concurrency Strategy:**
- **RWMutex**: Optimize for read-heavy workloads
- **Fine-Grained Locking**: Lock only what's necessary
- **Lock-Free When Possible**: Use channels for communication

### Context-Based Cancellation

```go
func (a *LouvainAdapter) Run(ctx context.Context, ...) (*AlgorithmResult, error) {
    // Check for cancellation throughout processing
    select {
    case <-ctx.Done():
        return nil, ctx.Err()  // Graceful cancellation
    default:
        // Continue processing
    }
    
    // Call algorithm with context
    result, err := louvain.Run(graph, config, ctx)
    return convertResult(result), err
}
```

## 📊 Error Handling Strategy

### Error Categories

1. **Validation Errors** → 400 Bad Request
2. **Resource Not Found** → 404 Not Found  
3. **Internal Processing** → 500 Internal Server Error
4. **Algorithm Failures** → Job marked as failed

### Error Propagation

```go
// Service layer returns domain errors
func (s *DatasetService) Upload(...) (*Dataset, error) {
    if err := validateFiles(files); err != nil {
        return nil, fmt.Errorf("file validation failed: %w", err)
    }
    
    if err := saveFiles(files); err != nil {
        return nil, fmt.Errorf("file storage failed: %w", err)
    }
}

// 🆕 NEW: Comparison service follows same pattern
func (s *ComparisonService) Create(...) (*Comparison, error) {
    if err := s.validateComparison(expA, expB, metrics); err != nil {
        return nil, fmt.Errorf("comparison validation failed: %w", err)
    }
    
    if hierA, err := s.clusteringService.GetHierarchy(...); err != nil {
        return nil, fmt.Errorf("failed to get hierarchy A: %w", err)
    }
}

// API layer converts to HTTP responses
func (h *Handlers) UploadDataset(w http.ResponseWriter, r *http.Request) {
    dataset, err := h.datasetService.Upload(name, files)
    if err != nil {
        // Determine appropriate HTTP status based on error type
        utils.WriteErrorResponse(w, http.StatusBadRequest, "Upload failed", err)
        return
    }
    
    utils.WriteSuccessResponse(w, "Upload successful", dataset)
}
```

### Structured Error Responses

```go
type APIResponse struct {
    Success bool        `json:"success"`
    Message string      `json:"message"`  // Human-readable
    Data    interface{} `json:"data,omitempty"`
    Error   string      `json:"error,omitempty"` // Technical details
}
```

## 📝 Logging Architecture

### Structured Logging with Zerolog

```go
log.Info().
    Str("dataset_id", datasetID).
    Str("algorithm", string(algorithmType)).
    Int64("processing_time_ms", processingTime.Milliseconds()).
    Float64("modularity", result.Modularity).
    Msg("Clustering completed successfully")

// 🆕 NEW: Comparison logging follows same pattern
log.Info().
    Str("comparison_id", comparisonID).
    Str("experiment_a", fmt.Sprintf("%s/%s", expA.DatasetID, expA.JobID)).
    Str("experiment_b", fmt.Sprintf("%s/%s", expB.DatasetID, expB.JobID)).
    Strs("metrics", metrics).
    Str("overall_similarity", result.Summary.OverallSimilarity).
    Msg("Comparison completed successfully")
```

**Log Levels:**
- **Error**: System failures requiring immediate attention
- **Warn**: Potential issues that should be investigated  
- **Info**: Important business events (job started, completed, comparison created)
- **Debug**: Detailed troubleshooting information

**Searchable Context:**
- `dataset_id`: Group logs by dataset
- `job_id`: Track job lifecycle
- `comparison_id`: Track comparison operations  
- `algorithm`: Filter by algorithm type
- `user_id`: Track user actions (future)

## 🧪 Testing Strategy

### Unit Testing Pattern

```go
func TestClusteringService_StartClustering(t *testing.T) {
    // Create mock dependencies
    mockDataset := &mockDatasetService{
        datasets: map[string]*models.Dataset{
            "test-id": {ID: "test-id", Status: models.DatasetStatusUploaded},
        },
    }
    mockJob := &mockJobService{}
    
    // Test subject with injected mocks
    service := NewClusteringService(mockDataset, mockJob)
    
    // Execute test
    job, err := service.StartClustering("test-id", models.AlgorithmLouvain, params)
    
    // Verify behavior
    assert.NoError(t, err)
    assert.Equal(t, "test-id", job.DatasetID)
    assert.Equal(t, models.AlgorithmLouvain, job.Algorithm)
}

// 🆕 NEW: Comparison service testing
func TestComparisonService_Create(t *testing.T) {
    // Create mock clustering service
    mockClustering := &mockClusteringService{
        hierarchies: map[string]*models.Hierarchy{
            "job-a": {DatasetID: "dataset-1", JobID: "job-a"},
            "job-b": {DatasetID: "dataset-1", JobID: "job-b"},
        },
    }
    
    // Test subject
    service := NewComparisonService(mockClustering)
    
    // Execute test
    comparison, err := service.Create("Test", 
        models.ExperimentRef{DatasetID: "dataset-1", JobID: "job-a"},
        models.ExperimentRef{DatasetID: "dataset-1", JobID: "job-b"},
        []string{"jaccard"}, models.ComparisonOptions{})
    
    // Verify behavior
    assert.NoError(t, err)
    assert.Equal(t, "Test", comparison.Name)
    assert.Contains(t, comparison.Metrics, "jaccard")
}
```

### Integration Testing

```go
func TestAPI_FullWorkflow(t *testing.T) {
    // Start test server
    server := setupTestServer()
    defer server.Close()
    
    // Test complete workflow
    datasetID := uploadTestDataset(t, server)
    jobIDA := startClustering(t, server, datasetID, "louvain")
    jobIDB := startClustering(t, server, datasetID, "scar")
    waitForCompletion(t, server, datasetID, jobIDA)
    waitForCompletion(t, server, datasetID, jobIDB)
    
    // 🆕 NEW: Test comparison workflow
    comparisonID := startComparison(t, server, datasetID, jobIDA, jobIDB)
    comparison := getComparison(t, server, comparisonID)
    
    // Verify end-to-end behavior
    assert.Equal(t, models.ComparisonStatusCompleted, comparison.Status)
    assert.NotNil(t, comparison.Result)
}
```

## 🚀 Extending the System

### Adding New Algorithms

1. **Create Adapter** (`algorithm/my_algorithm.go`):
```go
type MyAlgorithmAdapter struct{}

func (a *MyAlgorithmAdapter) Name() models.AlgorithmType {
    return "my-algorithm"
}

func (a *MyAlgorithmAdapter) ValidateParameters(params models.JobParameters) error {
    // Validate algorithm-specific parameters
}

func (a *MyAlgorithmAdapter) Run(ctx context.Context, dataset *models.Dataset, 
    params models.JobParameters, progressCb ProgressCallback) (*AlgorithmResult, error) {
    
    // Execute algorithm
    // Convert results to hierarchy format
    // Return AlgorithmResult
}
```

2. **Register Algorithm** (`algorithm/interface.go`):
```go
func NewRegistry() *Registry {
    registry := &Registry{algorithms: make(map[models.AlgorithmType]Algorithm)}
    
    registry.Register(NewLouvainAdapter())
    registry.Register(NewSCARAdapter())
    registry.Register(NewMyAlgorithmAdapter()) // Add here
    
    return registry
}
```

3. **Update Models** (`models/models.go`):
```go
const (
    AlgorithmLouvain    AlgorithmType = "louvain"
    AlgorithmSCAR       AlgorithmType = "scar"
    AlgorithmMyAlgorithm AlgorithmType = "my-algorithm" // Add here
)
```

### 🆕 Adding New Comparison Metrics

1. **Add Metric Method** (`service/comparison.go`):
```go
func (s *ComparisonService) computeMyMetric(hierA, hierB *models.Hierarchy) (float64, error) {
    // Implement your metric computation
    // Return value between 0-1 (1 = identical)
}
```

2. **Update Metric Validation** (`service/comparison.go`):
```go
validMetrics := map[string]bool{
    "agds":      true,
    "hmi":       true,
    "jaccard":   true,
    "ari":       true,
    "my-metric": true,  // Add here
}
```

3. **Add to Computation Loop** (`service/comparison.go`):
```go
for _, metric := range comparison.Metrics {
    switch metric {
    case "agds":
        // existing code
    case "my-metric":
        if myMetric, err := s.computeMyMetric(hierA, hierB); err == nil {
            result.MyMetric = &myMetric
        }
    }
}
```

4. **Update Result Model** (`models/models.go`):
```go
type ComparisonResult struct {
    AGDS     *float64 `json:"agds,omitempty"`
    HMI      *float64 `json:"hmi,omitempty"`
    Jaccard  *float64 `json:"jaccard,omitempty"`
    ARI      *float64 `json:"ari,omitempty"`
    MyMetric *float64 `json:"myMetric,omitempty"`  // Add here
    // ... rest of fields
}
```

### Adding New Endpoints

1. **Add Route** (`api/routes.go`):
```go
func SetupRoutes(router *mux.Router, handlers *Handlers) {
    api := router.PathPrefix("/api/v1").Subrouter()
    
    // Add new endpoint
    api.HandleFunc("/my-endpoint", handlers.MyHandler).Methods("GET")
}
```

2. **Add Handler** (`api/handlers.go`):
```go
func (h *Handlers) MyHandler(w http.ResponseWriter, r *http.Request) {
    // Handle request
    result, err := h.myService.DoSomething()
    if err != nil {
        utils.WriteErrorResponse(w, http.StatusInternalServerError, "Failed", err)
        return
    }
    
    utils.WriteSuccessResponse(w, "Success", result)
}
```

### Adding New Services

1. **Create Service** (`service/my_service.go`):
```go
type MyService struct {
    dependency *OtherService
}

func NewMyService(dependency *OtherService) *MyService {
    return &MyService{dependency: dependency}
}
```

2. **Wire Dependencies** (`main.go`):
```go
func main() {
    // Create services in dependency order
    myService := service.NewMyService(otherService)
    handlers := api.NewHandlers(..., myService)
}
```

## ⚡ Performance Considerations

### Memory Management

```go
// ✅ Use sync.Pool for frequently allocated objects
var jsonBufferPool = sync.Pool{
    New: func() interface{} {
        return bytes.NewBuffer(make([]byte, 0, 1024))
    },
}

func handleRequest(w http.ResponseWriter, r *http.Request) {
    buf := jsonBufferPool.Get().(*bytes.Buffer)
    defer jsonBufferPool.Put(buf)
    buf.Reset()
    
    // Use buffer...
}
```

### Resource Limits

```go
// Limit concurrent operations
type JobService struct {
    workers chan struct{} // Max 4 concurrent jobs
}

// Set timeouts for long operations
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
defer cancel()
```

### 🆕 Comparison Performance

```go
// Comparisons are synchronous and fast (<5 seconds)
// No worker pool needed, but could add caching for repeated comparisons
type ComparisonService struct {
    comparisons map[string]*models.Comparison
    // Future: add result cache for repeated comparisons
    // cache map[string]*models.ComparisonResult
}
```

## 🔒 Security Considerations

### Input Validation

```go
func (h *Handlers) validateUpload(files map[string]*multipart.FileHeader) error {
    for name, header := range files {
        // Check file size
        if header.Size > maxFileSize {
            return fmt.Errorf("file %s too large: %d bytes", name, header.Size)
        }
        
        // Check file type
        if !isValidFileType(header.Filename) {
            return fmt.Errorf("invalid file type: %s", header.Filename)
        }
    }
    return nil
}

// 🆕 NEW: Comparison validation
func (s *ComparisonService) validateComparison(expA, expB models.ExperimentRef, metrics []string) error {
    // Validate experiments exist and are completed
    // Validate metrics are supported
    // Validate experiment references
}
```

### Future Security Features

- **Authentication**: JWT tokens or API keys
- **Authorization**: Role-based access control  
- **Rate Limiting**: Prevent abuse
- **Input Sanitization**: Prevent injection attacks
- **HTTPS**: Encrypt data in transit

## 🚧 Deployment & Operations

### Configuration Management

```go
// Environment-based configuration
type Config struct {
    Server  ServerConfig
    Jobs    JobConfig
    Storage StorageConfig
}

// Loaded from environment variables with defaults
func Load() (*Config, error) {
    return &Config{
        Server: ServerConfig{
            Address: getEnv("SERVER_ADDRESS", ":8080"),
        },
    }
}
```

### Health Checks

```go
// Ready for Kubernetes liveness/readiness probes
func (h *Handlers) HealthCheck(w http.ResponseWriter, r *http.Request) {
    health := map[string]interface{}{
        "status":    "healthy",
        "timestamp": time.Now().Format(time.RFC3339),
        "version":   "2.0.0",
    }
    utils.WriteSuccessResponse(w, "Service is healthy", health)
}
```

### Graceful Shutdown

```go
func main() {
    server := &http.Server{...}
    
    // Handle shutdown signals
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit
    
    // Graceful shutdown with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    server.Shutdown(ctx)
}
```

## ⚠️ Common Pitfalls & Best Practices

### ❌ Don't Do This

```go
// Global variables (breaks testability)
var globalDatasetService *DatasetService

// Direct dependencies (breaks flexibility)
func ProcessJob() {
    service := &DatasetService{} // Hard-coded dependency
}

// Mixed concerns (violates single responsibility)
func HandleRequestAndProcessData(w http.ResponseWriter, r *http.Request) {
    // Parse HTTP request
    // Run algorithm
    // Save to database
    // Send email
    // Return HTTP response
    // TOO MUCH! Split into layers
}
```

### ✅ Do This Instead

```go
// Dependency injection (testable, flexible)
func NewClusteringService(ds DatasetServiceInterface) *ClusteringService {
    return &ClusteringService{datasetService: ds}
}

// 🆕 NEW: Comparison service follows same pattern
func NewComparisonService(cs *ClusteringService) *ComparisonService {
    return &ComparisonService{clusteringService: cs}
}

// Single responsibility (maintainable)
func (h *Handlers) StartClustering(w http.ResponseWriter, r *http.Request) {
    // Only HTTP concerns
    var req ClusteringRequest
    json.NewDecoder(r.Body).Decode(&req)
    
    // Delegate to service layer
    job, err := h.clusteringService.StartClustering(...)
    
    // Only HTTP response
    utils.WriteSuccessResponse(w, "Job started", job)
}

func (h *Handlers) CreateComparison(w http.ResponseWriter, r *http.Request) {
    // Only HTTP concerns
    var req models.CreateComparisonRequest
    json.NewDecoder(r.Body).Decode(&req)
    
    // Delegate to service layer
    comparison, err := h.comparisonService.Create(...)
    
    // Only HTTP response
    utils.WriteSuccessResponse(w, "Comparison started", comparison)
}
```

### Error Handling Guidelines

```go
// ✅ Wrap errors with context
return fmt.Errorf("failed to process dataset %s: %w", datasetID, err)

// 🆕 NEW: Comparison errors follow same pattern
return fmt.Errorf("failed to compute %s metric: %w", metricName, err)

// ✅ Handle errors at appropriate level
func (s *Service) DoWork() error {
    if err := s.step1(); err != nil {
        return err // Propagate up
    }
    return nil
}

func (h *Handler) HandleRequest(w http.ResponseWriter, r *http.Request) {
    if err := h.service.DoWork(); err != nil {
        utils.WriteErrorResponse(w, http.StatusInternalServerError, "Work failed", err)
        return // Handle here
    }
}
```

---

## 🎯 Development Workflow

1. **Understand the Layer**: Know which layer you're working in
2. **Follow Patterns**: Use existing patterns for consistency
3. **Test Your Code**: Write unit tests with mocked dependencies
4. **Update Documentation**: Keep README and examples current
5. **Review Changes**: Ensure changes don't break architectural principles

## 🆕 Comparison Feature Integration Points

The new comparison feature demonstrates proper architectural integration:

### ✅ **What Was Done Right:**
- **Layer Separation**: ComparisonService in service layer, handlers in API layer
- **Dependency Injection**: ComparisonService depends on ClusteringService interface
- **Error Handling**: Consistent error propagation and HTTP status mapping
- **Logging**: Structured logging following existing patterns
- **Testing**: Same unit testing patterns with mock dependencies
- **Resource Management**: In-memory storage following existing patterns

### ✅ **Architectural Benefits:**
- **Zero Breaking Changes**: Existing code unmodified
- **Reusable Components**: Leverages existing clustering infrastructure
- **Extensible**: Easy to add new metrics or comparison types
- **Maintainable**: Clear separation of concerns and dependencies

**Remember**: This architecture replaces chaos with order. Keep it clean! 🧹
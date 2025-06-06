# SCAR-based Heterogeneous Graph Clustering Service
# Makefile for build automation and testing

.PHONY: help build test validate materialize example clean fmt vet deps

# Default target
help:
	@echo "SCAR Graph Clustering Service - Available Commands:"
	@echo "=================================================="
	@echo "build         - Build the main application"
	@echo "test          - Run all tests"
	@echo "test-verbose  - Run tests with verbose output"
	@echo "benchmark     - Run performance benchmarks"
	@echo "validate      - Run validation on example data"
	@echo "materialize   - Run validation + materialization on example data"
	@echo "example       - Run comprehensive materialization example"
	@echo "test-interpretations - Test both DirectTraversal and MeetingBased"
	@echo "fmt           - Format all Go code"
	@echo "vet           - Run Go vet for code analysis"
	@echo "deps          - Download and verify dependencies"
	@echo "clean         - Clean build artifacts"
	@echo ""
	@echo "Example Usage:"
	@echo "  make validate           # Quick validation test"
	@echo "  make materialize        # Full materialization demo"
	@echo "  make test-interpretations # Compare interpretation methods"
	@echo "  make test               # Run all unit tests"

# Build the main application
build:
	@echo "Building SCAR Graph Clustering Service..."
	go build -o bin/graph-clustering-service main.go
	@echo "✅ Build complete: bin/graph-clustering-service"

# Run all tests
test:
	@echo "Running all tests..."
	go test ./pkg/validation/ ./pkg/materialization/
	@echo "✅ All tests passed"

# Run tests with verbose output
test-verbose:
	@echo "Running tests with verbose output..."
	go test -v ./pkg/validation/ ./pkg/materialization/

# Run benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	go test -bench=. ./pkg/materialization/
	@echo "✅ Benchmarks complete"

# Test coverage
coverage:
	@echo "Generating test coverage report..."
	go test -cover ./pkg/validation/ ./pkg/materialization/
	go test -coverprofile=coverage.out ./pkg/validation/ ./pkg/materialization/
	go tool cover -html=coverage.out -o coverage.html
	@echo "✅ Coverage report generated: coverage.html"

# Run validation only on example data
validate:
	@echo "Running validation on example data..."
	@mkdir -p data 2>/dev/null || true
	go run main.go validate data/graph_input.json data/meta_path.json

# Run validation + materialization on example data
materialize:
	@echo "Running materialization on example data..."
	@mkdir -p data 2>/dev/null || true
	go run main.go materialize data/graph_input.json data/meta_path.json

# Run comprehensive example
example:
	@echo "Running comprehensive materialization example..."
	@mkdir -p data 2>/dev/null || true
	go run examples/materialization_example.go data/graph_input.json data/meta_path.json

# Test both interpretation methods
test-interpretations:
	@echo "Testing both meta path interpretations..."
	@mkdir -p data 2>/dev/null || true
	go run test_interpretations.go

# Format all Go code
fmt:
	@echo "Formatting Go code..."
	go fmt ./...
	@echo "✅ Code formatted"

# Run Go vet for code analysis
vet:
	@echo "Running Go vet..."
	go vet ./...
	@echo "✅ Vet analysis complete"

# Download and verify dependencies
deps:
	@echo "Downloading and verifying dependencies..."
	go mod download
	go mod verify
	go mod tidy
	@echo "✅ Dependencies updated"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf bin/
	rm -f coverage.out coverage.html
	go clean ./...
	@echo "✅ Clean complete"

# Setup development environment
setup: deps
	@echo "Setting up development environment..."
	@mkdir -p bin data examples 2>/dev/null || true
	@echo "✅ Development environment ready"

# Run all quality checks
check: fmt vet test
	@echo "✅ All quality checks passed"

# Create example data files if they don't exist
create-examples:
	@echo "Creating example data files..."
	@mkdir -p data
	@if [ ! -f data/graph_input.json ]; then \
		echo "Creating data/graph_input.json..."; \
		echo '{\n  "nodes": {\n    "a1": {"type": "Author", "name": "Alice"},\n    "a2": {"type": "Author", "name": "Bob"},\n    "p1": {"type": "Paper", "title": "Test Paper"}\n  },\n  "edges": [\n    {"from": "a1", "to": "p1", "type": "writes", "weight": 1.0},\n    {"from": "p1", "to": "a2", "type": "writes", "weight": 1.0}\n  ]\n}' > data/graph_input.json; \
	fi
	@if [ ! -f data/meta_path.json ]; then \
		echo "Creating data/meta_path.json..."; \
		echo '{\n  "id": "author_coauthorship",\n  "node_sequence": ["Author", "Paper", "Author"],\n  "edge_sequence": ["writes", "writes"],\n  "description": "Authors connected through co-authored papers"\n}' > data/meta_path.json; \
	fi
	@echo "✅ Example data files created"

# Quick start for new users
quickstart: setup create-examples validate
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo "Try these commands:"
	@echo "  make materialize  # Run full materialization"
	@echo "  make example      # Run comprehensive demo"
	@echo "  make test         # Run all tests"

# Performance testing with larger datasets
perf-test:
	@echo "Running performance tests..."
	go test -bench=. -benchtime=10s ./pkg/materialization/
	@echo "✅ Performance tests complete"

# Memory profiling
profile-memory:
	@echo "Running memory profiling..."
	go test -bench=BenchmarkMaterialization -memprofile=mem.prof ./pkg/materialization/
	go tool pprof mem.prof
	@echo "✅ Memory profiling complete"

# CPU profiling  
profile-cpu:
	@echo "Running CPU profiling..."
	go test -bench=BenchmarkMaterialization -cpuprofile=cpu.prof ./pkg/materialization/
	go tool pprof cpu.prof
	@echo "✅ CPU profiling complete"

# Install development tools
install-tools:
	@echo "Installing development tools..."
	go install golang.org/x/tools/cmd/goimports@latest
	go install golang.org/x/lint/golint@latest
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	@echo "✅ Development tools installed"

# Lint with golangci-lint (if installed)
lint:
	@if command -v golangci-lint >/dev/null 2>&1; then \
		echo "Running golangci-lint..."; \
		golangci-lint run; \
		echo "✅ Linting complete"; \
	else \
		echo "golangci-lint not installed. Run 'make install-tools' first."; \
	fi

# Full development workflow
dev: clean setup fmt vet lint test benchmark
	@echo "✅ Full development workflow complete"

# Documentation generation (if godoc is available)
docs:
	@if command -v godoc >/dev/null 2>&1; then \
		echo "Starting documentation server..."; \
		echo "Visit http://localhost:6060/pkg/github.com/yourusername/graph-clustering-service/"; \
		godoc -http=:6060; \
	else \
		echo "godoc not available. Install with: go install golang.org/x/tools/cmd/godoc@latest"; \
	fi

# Docker build (future)
docker-build:
	@echo "Docker build not yet implemented"
	@echo "Future: docker build -t graph-clustering-service ."

# Release build with optimizations
release: clean
	@echo "Building optimized release..."
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o bin/graph-clustering-service-linux main.go
	CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -ldflags="-w -s" -o bin/graph-clustering-service-windows.exe main.go
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -ldflags="-w -s" -o bin/graph-clustering-service-macos main.go
	@echo "✅ Release builds complete in bin/"
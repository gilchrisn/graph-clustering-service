package api

import (
	"net/http"
	
	"github.com/gorilla/mux"
	
)

func SetupRoutes(router *mux.Router, handlers *Handlers) {
	// API version prefix
	api := router.PathPrefix("/api/v1").Subrouter()

	// Dataset management endpoints
	datasets := api.PathPrefix("/datasets").Subrouter()
	datasets.HandleFunc("", handlers.ListDatasets).Methods("GET")
	datasets.HandleFunc("", handlers.UploadDataset).Methods("POST")
	datasets.HandleFunc("/{datasetId}", handlers.GetDataset).Methods("GET")
	datasets.HandleFunc("/{datasetId}", handlers.DeleteDataset).Methods("DELETE")


	// Clustering endpoints - unified processing endpoint
	clustering := datasets.PathPrefix("/{datasetId}/clustering").Subrouter()
	clustering.HandleFunc("", handlers.StartClustering).Methods("POST")
	clustering.HandleFunc("/{jobId}", handlers.GetClusteringJob).Methods("GET")
	clustering.HandleFunc("/{jobId}", handlers.CancelClusteringJob).Methods("DELETE")

	// Hierarchy data endpoints
	hierarchy := datasets.PathPrefix("/{datasetId}/hierarchy").Subrouter()
	hierarchy.HandleFunc("", handlers.GetFullHierarchy).Methods("GET").Queries("jobId", "{jobId}")
	hierarchy.HandleFunc("/levels/{level:[0-9]+}", handlers.GetHierarchyLevel).Methods("GET").Queries("jobId", "{jobId}")

	// Cluster drill-down endpoints
	clusters := datasets.PathPrefix("/{datasetId}/clusters").Subrouter()
	clusters.HandleFunc("/{clusterId}/nodes", handlers.GetClusterNodes).Methods("GET").Queries("jobId", "{jobId}")

	// Job management endpoints
	jobs := api.PathPrefix("/jobs").Subrouter()
	jobs.HandleFunc("/{jobId}", handlers.GetJob).Methods("GET")
	jobs.HandleFunc("/{jobId}/cancel", handlers.CancelJob).Methods("POST")

	// ADD THESE NEW COMPARISON ENDPOINTS:
	// Comparison endpoints
	comparisons := api.PathPrefix("/comparisons").Subrouter()
	comparisons.HandleFunc("", handlers.CreateComparison).Methods("POST")
	comparisons.HandleFunc("", handlers.ListComparisons).Methods("GET")
	comparisons.HandleFunc("/{comparisonId}", handlers.GetComparison).Methods("GET")
	comparisons.HandleFunc("/{comparisonId}", handlers.DeleteComparison).Methods("DELETE")

	// Multi-comparison endpoint
	comparisons.HandleFunc("/multi", handlers.CreateMultiComparison).Methods("POST")

	// Health check endpoint
	api.HandleFunc("/health", handlers.HealthCheck).Methods("GET")

	// Algorithm info endpoint
	api.HandleFunc("/algorithms", handlers.ListAlgorithms).Methods("GET")

	api.PathPrefix("/").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			// Let CORS middleware handle it
			return
		}
	}).Methods("OPTIONS")
}
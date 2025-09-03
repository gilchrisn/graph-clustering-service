// main.go
package main

import (
	"log"
	"net/http"

	"github.com/gorilla/mux"
	"graph-clustering-backend/controllers"
    "github.com/rs/cors"
)

// CORS middleware
func enableCORS(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Set CORS headers
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
        
        // Handle preflight OPTIONS request
        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }
        
        // Continue with the next handler
        next.ServeHTTP(w, r)
    })
}

func main() {
	r := mux.NewRouter()
	
	// Original homogeneous graph routes
	r.HandleFunc("/api/upload", controllers.UploadFiles).Methods("POST")
	r.HandleFunc("/api/process", controllers.ProcessDataset).Methods("POST")

	// Heterogeneous graph routes (materialize + louvain)
	r.HandleFunc("/api/upload-heterogeneous", controllers.UploadFilesHeterogeneous).Methods("POST")
	r.HandleFunc("/api/process-heterogeneous", controllers.ProcessDatasetHeterogeneous).Methods("POST")

	// SCAR processing routes
	r.HandleFunc("/api/upload-scar", controllers.UploadFilesScar).Methods("POST")
	r.HandleFunc("/api/process-scar", controllers.ProcessDatasetScar).Methods("POST")

	// Get hierarchy data (works with all processing types via query parameter)
	r.HandleFunc("/api/hierarchy/{datasetId}/{k}", controllers.GetHierarchyData).Methods("GET")

	// Get supernode coordinates (works with all processing types via query parameter)
	r.HandleFunc("/api/coordinates/{datasetId}/{algorithmId}/{supernodeId}", controllers.GetSupernodeCoordinates).Methods("GET")

	// Get node statistics (works with all processing types via query parameter)
	r.HandleFunc("/api/node/{datasetId}/{k}/{nodeId}", controllers.GetNodeStatistics).Methods("GET")

	// Algorithm comparison route
	r.HandleFunc("/api/compare", controllers.CompareAlgorithms).Methods("POST")

    // Configure CORS
    c := cors.New(cors.Options{
        AllowedOrigins: []string{"http://localhost:3000"}, // Your frontend URL
        AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
        AllowedHeaders: []string{"*"},
        AllowCredentials: true,
    })
    
    // Apply CORS middleware
    handler := c.Handler(r)

	log.Println("Server starting on :3002")
	log.Fatal(http.ListenAndServe(":3002", handler))
}
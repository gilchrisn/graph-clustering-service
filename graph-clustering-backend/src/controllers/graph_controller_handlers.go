// controllers/graph_controller_handlers.go
package controllers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gorilla/mux"
	"graph-clustering-backend/services"
)


// Process the dataset (run Louvain+ clustering and build indices) - ORIGINAL
func ProcessDataset(w http.ResponseWriter, r *http.Request) {
	log.Println("🚀 [DEBUG] ProcessDataset endpoint called")
	w.Header().Set("Content-Type", "application/json")
	
	var req struct {
		DatasetID string `json:"datasetId"`
		K         int    `json:"k"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("❌ [DEBUG] Error decoding request body: %v", err)
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ProcessResponse{
			Success: false,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}
	
	log.Printf("🔍 [DEBUG] Process request - DatasetID: %s, K: %d", req.DatasetID, req.K)
	
	if req.DatasetID == "" || req.K == 0 {
		log.Printf("❌ [DEBUG] Missing required parameters - DatasetID: '%s', K: %d", req.DatasetID, req.K)
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ProcessResponse{
			Success: false,
			Message: "Missing required parameters",
		})
		return
	}
	
	log.Println("📞 [DEBUG] Calling services.ProcessDataset...")
	
	// Process the dataset using the service
	result, err := services.ProcessDataset(req.DatasetID, req.K)
	if err != nil {
		log.Printf("❌ [DEBUG] Error processing dataset: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ProcessResponse{
			Success: false,
			Message: "Error processing dataset",
			Error:   err.Error(),
		})
		return
	}
	
	log.Printf("✅ [DEBUG] Processing successful, result: %+v", result)
	log.Println("📤 [DEBUG] Sending successful response...")
	
	json.NewEncoder(w).Encode(ProcessResponse{
		Success: true,
		Message: "Dataset processed successfully",
		Result:  result,
	})
}

// Process heterogeneous dataset (materialize + louvain)
func ProcessDatasetHeterogeneous(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	var req struct {
		DatasetID string `json:"datasetId"`
		K         int    `json:"k"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ProcessResponse{
			Success: false,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}
	
	if req.DatasetID == "" || req.K == 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ProcessResponse{
			Success: false,
			Message: "Missing required parameters",
		})
		return
	}
	
	// Process the dataset using the heterogeneous service
	result, err := services.ProcessDatasetHeterogeneous(req.DatasetID, req.K)
	if err != nil {
		log.Printf("Error processing heterogeneous dataset: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ProcessResponse{
			Success: false,
			Message: "Error processing heterogeneous dataset",
			Error:   err.Error(),
		})
		return
	}
	
	json.NewEncoder(w).Encode(ProcessResponse{
		Success: true,
		Message: "Heterogeneous dataset processed successfully",
		Result:  result,
	})
}

// Process SCAR dataset
func ProcessDatasetScar(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	var req struct {
		DatasetID string  `json:"datasetId"`
		K         int     `json:"k"`
		NK        int     `json:"nk"`
		TH        float64 `json:"th"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ProcessResponse{
			Success: false,
			Message: "Invalid request body",
			Error:   err.Error(),
		})
		return
	}
	
	if req.DatasetID == "" || req.K == 0 || req.NK == 0 || req.TH == 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ProcessResponse{
			Success: false,
			Message: "Missing required parameters: datasetId, k, nk, th",
		})
		return
	}
	
	// The meta file should follow the naming pattern: {datasetId}_meta.dat
	metaFileName := fmt.Sprintf("%s_meta.dat", req.DatasetID)
	
	// Process the dataset using the SCAR service
	result, err := services.ProcessDatasetScar(req.DatasetID, req.K, req.NK, req.TH, metaFileName)
	if err != nil {
		log.Printf("Error processing SCAR dataset: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ProcessResponse{
			Success: false,
			Message: "Error processing SCAR dataset",
			Error:   err.Error(),
		})
		return
	}
	
	json.NewEncoder(w).Encode(ProcessResponse{
		Success: true,
		Message: "SCAR dataset processed successfully",
		Result:  result,
	})
}

// Get hierarchy data for a dataset
func GetHierarchyData(w http.ResponseWriter, r *http.Request) {
	log.Println("🚀 [DEBUG] GetHierarchyData endpoint called")
	w.Header().Set("Content-Type", "application/json")
	
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]
	kStr := vars["k"]
	processingType := r.URL.Query().Get("processingType") // 'louvain', 'scar', or 'heterogeneous'
	
	log.Printf("🔍 [DEBUG] Path parameters - datasetId: %s, k: %s", datasetID, kStr)
	log.Printf("🔍 [DEBUG] Query parameter - processingType: %s", processingType)
	
	k, err := strconv.Atoi(kStr)
	if err != nil || datasetID == "" {
		log.Printf("❌ [DEBUG] Invalid parameters - datasetID: '%s', k conversion error: %v", datasetID, err)
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(HierarchyResponse{
			Success: false,
			Message: "Missing required parameters",
		})
		return
	}
	
	if processingType == "" {
		processingType = "louvain"
		log.Printf("🔧 [DEBUG] Using default processingType: %s", processingType)
	}
	
	log.Printf("📞 [DEBUG] Calling services.GetHierarchyData(%s, %d, %s)...", datasetID, k, processingType)
	
	// Get hierarchy data using the service
	data, err := services.GetHierarchyData(datasetID, k, processingType)
	if err != nil {
		log.Printf("❌ [DEBUG] Error getting hierarchy data: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(HierarchyResponse{
			Success: false,
			Message: "Error getting hierarchy data",
			Error:   err.Error(),
		})
		return
	}
	
	log.Printf("✅ [DEBUG] Hierarchy data retrieved successfully")
	log.Printf("🔍 [DEBUG] Hierarchy keys count: %d", getMapLength(data.Hierarchy))
	log.Printf("🔍 [DEBUG] Mapping keys count: %d", getMapLength(data.Mapping))
	log.Printf("📤 [DEBUG] Sending hierarchy response...")
	
	json.NewEncoder(w).Encode(HierarchyResponse{
		Success:   true,
		Hierarchy: data.Hierarchy,
		Mapping:   data.Mapping,
	})
}

// Helper function to get map length for debug logging
func getMapLength(data interface{}) int {
	if m, ok := data.(map[string]interface{}); ok {
		return len(m)
	}
	return 0
}

// Get coordinates for a specific supernode
func GetSupernodeCoordinates(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]
	algorithmID := vars["algorithmId"]
	supernodeID := vars["supernodeId"]
	processingType := r.URL.Query().Get("processingType") // 'louvain', 'scar', or 'heterogeneous'
	
	if datasetID == "" || algorithmID == "" || supernodeID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(CoordinatesResponse{
			Success: false,
			Message: "Missing required parameters",
		})
		return
	}
	
	if processingType == "" {
		processingType = "louvain"
	}
	
	// Get supernode coordinates using the service
	coordinates, err := services.GetSupernodeCoordinates(datasetID, algorithmID, supernodeID, processingType)
	if err != nil {
		log.Printf("Error getting supernode coordinates: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(CoordinatesResponse{
			Success: false,
			Message: "Error getting supernode coordinates",
			Error:   err.Error(),
		})
		return
	}
	
	json.NewEncoder(w).Encode(CoordinatesResponse{
		Success: true,
		Nodes:   coordinates.Nodes,
		Edges:   coordinates.Edges,
	})
}

// Get statistics for a specific node
func GetNodeStatistics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	vars := mux.Vars(r)
	datasetID := vars["datasetId"]
	kStr := vars["k"]
	nodeID := vars["nodeId"]
	processingType := r.URL.Query().Get("processingType") // 'louvain', 'scar', or 'heterogeneous'
	
	k, err := strconv.Atoi(kStr)
	if err != nil || datasetID == "" || nodeID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(StatisticsResponse{
			Success: false,
			Message: "Missing required parameters",
		})
		return
	}
	
	if processingType == "" {
		processingType = "louvain"
	}
	
	// Get node statistics using the service
	statistics, err := services.GetNodeStatistics(datasetID, k, nodeID, processingType)
	if err != nil {
		log.Printf("Error getting node statistics: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(StatisticsResponse{
			Success: false,
			Message: "Error getting node statistics",
			Error:   err.Error(),
		})
		return
	}
	
	log.Printf("Node statistics:\n\n%+v", statistics)
	
	json.NewEncoder(w).Encode(StatisticsResponse{
		Success:    true,
		Statistics: statistics,
	})
}

// Compare algorithms endpoint
func CompareAlgorithms(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	log.Println("🚀 [DEBUG] CompareAlgorithms endpoint called")
	
	err := r.ParseMultipartForm(32 << 20)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ComparisonResponse{
			Success: false,
			Message: "Error parsing form",
			Error:   err.Error(),
		})
		return
	}
	
	// 🔍 Extract and validate parameters
	heterogeneousParams := r.FormValue("heterogeneous")
	scarParams := r.FormValue("scar")
	
	log.Printf("🔍 [DEBUG] Heterogeneous params: %s", heterogeneousParams)
	log.Printf("🔍 [DEBUG] SCAR params: %s", scarParams)
	
	var heteroParams, scarParamsStruct map[string]interface{}
	
	if heterogeneousParams != "" {
		if err := json.Unmarshal([]byte(heterogeneousParams), &heteroParams); err != nil {
			heteroParams = make(map[string]interface{})
		}
	} else {
		heteroParams = make(map[string]interface{})
	}
	
	if scarParams != "" {
		if err := json.Unmarshal([]byte(scarParams), &scarParamsStruct); err != nil {
			scarParamsStruct = make(map[string]interface{})
		}
	} else {
		scarParamsStruct = make(map[string]interface{})
	}
	
	// NEW: Validate required files for 3-file format (EXACT frontend field names)
	requiredFiles := []string{"graphFile", "pathFile", "propertiesFile"} // Match frontend exactly
	files := make(map[string]string)
	
	log.Println("🔍 [DEBUG] Checking for required files: graphFile, pathFile, propertiesFile")
	log.Println("🔍 [DEBUG] Available form files:")
	
	// Debug: Show ALL available files in the form
	for fieldName := range r.MultipartForm.File {
		log.Printf("🔍 [DEBUG] Available field: %s", fieldName)
	}
	
	for _, fileType := range requiredFiles {
		file, header, err := r.FormFile(fileType)
		if err != nil {
			log.Printf("❌ [DEBUG] Missing required file: %s, error: %v", fileType, err)
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(ComparisonResponse{
				Success: false,
				Message: fmt.Sprintf("Missing required file: %s", fileType),
			})
			return
		}
		defer file.Close()
		files[fileType] = header.Filename
		log.Printf("✅ [DEBUG] Found file: %s = %s", fileType, header.Filename)
	}
	
	// Extract dataset name from graph file (instead of edge file)
	graphFileName := files["graphFile"]
	datasetName := strings.TrimSuffix(graphFileName, filepath.Ext(graphFileName)) // Remove extension
	
	log.Printf("🔎 [DEBUG] Starting comparison for dataset: %s", datasetName)
	log.Printf("🔧 [DEBUG] Heterogeneous params: %+v", heteroParams)
	log.Printf("🔧 [DEBUG] SCAR params: %+v", scarParamsStruct)
	
	// 🏃 DELEGATE EVERYTHING TO SERVICE
	comparisonResult, err := services.CompareAlgorithms(
		r.MultipartForm.File,
		datasetName,
		heteroParams,
		scarParamsStruct,
	)
	if err != nil {
		log.Printf("🔥 [DEBUG] Error in algorithm comparison: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ComparisonResponse{
			Success: false,
			Message: "Failed to run algorithm comparison",
			Error:   err.Error(),
		})
		return
	}
	
	// 🏁 Return results
	response := ComparisonResponse{
		Success:    true,
		Message:    "Algorithm comparison completed successfully",
		Comparison: comparisonResult,
	}
	
	log.Println("✅ [DEBUG] Comparison completed successfully")
	json.NewEncoder(w).Encode(response)
}
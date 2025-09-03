// controllers/graph_controller_uploads.go
package controllers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Helper function to save uploaded file
func saveUploadedFile(file io.Reader, destination, filename string) error {
	err := os.MkdirAll(destination, 0755)
	if err != nil {
		return err
	}
	
	dst, err := os.Create(filepath.Join(destination, filename))
	if err != nil {
		return err
	}
	defer dst.Close()
	
	_, err = io.Copy(dst, file)
	return err
}

// Extract dataset name from filename
func extractDatasetName(filename, suffix string) string {
	base := filepath.Base(filename)
	ext := filepath.Ext(base)
	name := strings.TrimSuffix(base, ext)
	if suffix != "" {
		name = strings.TrimSuffix(name, suffix)
	}
	return name
}

// Upload edge list and attributes files (ORIGINAL HOMOGENEOUS)
func UploadFiles(w http.ResponseWriter, r *http.Request) {
	log.Println("🚀 [DEBUG] UploadFiles endpoint called")
	w.Header().Set("Content-Type", "application/json")
	
	err := r.ParseMultipartForm(32 << 20) // 32MB max memory
	if err != nil {
		log.Printf("❌ [DEBUG] Error parsing multipart form: %v", err)
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(UploadResponse{
			Success: false,
			Message: "Error parsing form",
		})
		return
	}
	
	log.Println("🔍 [DEBUG] Form parsed successfully, checking for files...")
	
	edgeListFile, edgeListHeader, err1 := r.FormFile("edgeList")
	pathFile, pathHeader, err2 := r.FormFile("path")
	propertiesFile, propertiesHeader, err3 := r.FormFile("properties")
	
	if err1 != nil || err2 != nil || err3 != nil {
		log.Printf("❌ [DEBUG] Missing files - edgeList error: %v, path error: %v, properties error: %v", err1, err2, err3)
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(UploadResponse{
			Success: false,
			Message: "Missing required files. Need: edgeList, path, and properties files",
		})
		return
	}
	defer edgeListFile.Close()
	defer pathFile.Close()
	defer propertiesFile.Close()
	
	log.Printf("✅ [DEBUG] Files received - edgeList: %s, path: %s, properties: %s", edgeListHeader.Filename, pathHeader.Filename, propertiesHeader.Filename)
	
	k, _ := strconv.Atoi(r.FormValue("k"))
	if k == 0 {
		k = 25 // Default to 25 if not provided
	}
	log.Printf("🔧 [DEBUG] Parameter k: %d", k)
	
	// Get dataset name from the edge list file
	datasetName := extractDatasetName(edgeListHeader.Filename, "")
	destination := "PPRviz-reproducibility/dataset"
	
	log.Printf("📂 [DEBUG] Dataset name: %s, destination: %s", datasetName, destination)
	
	// Save files
	log.Println("💾 [DEBUG] Saving edgeList file...")
	err = saveUploadedFile(edgeListFile, destination, edgeListHeader.Filename)
	if err != nil {
		log.Printf("❌ [DEBUG] Error saving edgeList file: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(UploadResponse{
			Success: false,
			Message: "Error uploading files",
			Error:   err.Error(),
		})
		return
	}
	
	log.Println("💾 [DEBUG] Saving path file...")
	err = saveUploadedFile(pathFile, destination, pathHeader.Filename)
	if err != nil {
		log.Printf("❌ [DEBUG] Error saving path file: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(UploadResponse{
			Success: false,
			Message: "Error uploading files",
			Error:   err.Error(),
		})
		return
	}
	
	log.Println("💾 [DEBUG] Saving properties file...")
	err = saveUploadedFile(propertiesFile, destination, propertiesHeader.Filename)
	if err != nil {
		log.Printf("❌ [DEBUG] Error saving properties file: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(UploadResponse{
			Success: false,
			Message: "Error uploading files",
			Error:   err.Error(),
		})
		return
	}
	
	log.Println("✅ [DEBUG] All three files saved successfully, sending response...")
	
	// Return success response with dataset info
	response := UploadResponse{
		Success:        true,
		Message:        "Files uploaded successfully",
		DatasetID:      datasetName,
		EdgeListPath:   filepath.Join(destination, edgeListHeader.Filename),
		PathFilePath:   filepath.Join(destination, pathHeader.Filename),
		PropertiesPath: filepath.Join(destination, propertiesHeader.Filename),
		K:              k,
	}
	
	log.Printf("📤 [DEBUG] Sending response: %+v", response)
	json.NewEncoder(w).Encode(response)
}

// Upload files for HETEROGENEOUS graph processing (same 4 files as SCAR)
func UploadFilesHeterogeneous(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	err := r.ParseMultipartForm(32 << 20)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(UploadResponse{
			Success: false,
			Message: "Error parsing form",
		})
		return
	}
	
	// Get all required files
	infoFile, infoHeader, err1 := r.FormFile("infoFile")
	linkFile, linkHeader, err2 := r.FormFile("linkFile")
	nodeFile, nodeHeader, err3 := r.FormFile("nodeFile")
	metaFile, metaHeader, err4 := r.FormFile("metaFile")
	
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(UploadResponse{
			Success: false,
			Message: "Missing required files. Need 4 files: infoFile ({datasetId}_info.dat), linkFile ({datasetId}_link.dat), nodeFile ({datasetId}_node.dat), and metaFile ({datasetId}_meta.dat)",
		})
		return
	}
	defer infoFile.Close()
	defer linkFile.Close()
	defer nodeFile.Close()
	defer metaFile.Close()
	
	k, _ := strconv.Atoi(r.FormValue("k"))
	if k == 0 {
		k = 25 // Default to 25 if not provided
	}
	
	// Extract dataset name from one of the files (assuming they follow the naming pattern)
	// Example: "amazon_info.dat" -> "amazon"
	datasetName := extractDatasetName(infoHeader.Filename, "_info.dat")
	destination := "graph-materialization/dataset"
	
	// Validate that all files follow the expected naming pattern
	expectedFiles := map[string]string{
		infoHeader.Filename: fmt.Sprintf("%s_info.dat", datasetName),
		linkHeader.Filename: fmt.Sprintf("%s_link.dat", datasetName),
		nodeHeader.Filename: fmt.Sprintf("%s_node.dat", datasetName),
		metaHeader.Filename: fmt.Sprintf("%s_meta.dat", datasetName),
	}
	
	for actual, expected := range expectedFiles {
		if actual != expected {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(UploadResponse{
				Success: false,
				Message: fmt.Sprintf("File naming mismatch. Expected %s, got %s", expected, actual),
			})
			return
		}
	}
	
	// Save files
	files := []struct {
		file   io.Reader
		header string
	}{
		{infoFile, infoHeader.Filename},
		{linkFile, linkHeader.Filename},
		{nodeFile, nodeHeader.Filename},
		{metaFile, metaHeader.Filename},
	}
	
	for _, f := range files {
		err = saveUploadedFile(f.file, destination, f.header)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(UploadResponse{
				Success: false,
				Message: "Error uploading heterogeneous files",
				Error:   err.Error(),
			})
			return
		}
	}
	
	// Return success response with dataset info
	json.NewEncoder(w).Encode(UploadResponse{
		Success:        true,
		Message:        "Heterogeneous files uploaded successfully",
		DatasetID:      datasetName,
		InfoFilePath:   filepath.Join(destination, infoHeader.Filename),
		LinkFilePath:   filepath.Join(destination, linkHeader.Filename),
		NodeFilePath:   filepath.Join(destination, nodeHeader.Filename),
		MetaFilePath:   filepath.Join(destination, metaHeader.Filename),
		K:              k,
		ProcessingType: "heterogeneous",
	})
}

// Upload files for SCAR processing (4 files with specific naming)
func UploadFilesScar(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	err := r.ParseMultipartForm(32 << 20)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(UploadResponse{
			Success: false,
			Message: "Error parsing form",
		})
		return
	}
	
	// Get all required files
	infoFile, infoHeader, err1 := r.FormFile("infoFile")
	linkFile, linkHeader, err2 := r.FormFile("linkFile")
	nodeFile, nodeHeader, err3 := r.FormFile("nodeFile")
	metaFile, metaHeader, err4 := r.FormFile("metaFile")
	
	if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(UploadResponse{
			Success: false,
			Message: "Missing required files. Need 4 files: infoFile ({datasetId}_info.dat), linkFile ({datasetId}_link.dat), nodeFile ({datasetId}_node.dat), and metaFile ({datasetId}_meta.dat)",
		})
		return
	}
	defer infoFile.Close()
	defer linkFile.Close()
	defer nodeFile.Close()
	defer metaFile.Close()
	
	k, _ := strconv.Atoi(r.FormValue("k"))
	if k == 0 {
		k = 25
	}
	nk, _ := strconv.Atoi(r.FormValue("nk"))
	if nk == 0 {
		nk = 10
	}
	th, _ := strconv.ParseFloat(r.FormValue("th"), 64)
	if th == 0 {
		th = 0.5
	}
	
	// Extract dataset name from one of the files (assuming they follow the naming pattern)
	// Example: "amazon_info.dat" -> "amazon"
	datasetName := extractDatasetName(infoHeader.Filename, "_info.dat")
	destination := "scar-main/dataset"
	
	// Validate that all files follow the expected naming pattern
	expectedFiles := map[string]string{
		infoHeader.Filename: fmt.Sprintf("%s_info.dat", datasetName),
		linkHeader.Filename: fmt.Sprintf("%s_link.dat", datasetName),
		nodeHeader.Filename: fmt.Sprintf("%s_node.dat", datasetName),
		metaHeader.Filename: fmt.Sprintf("%s_meta.dat", datasetName),
	}
	
	for actual, expected := range expectedFiles {
		if actual != expected {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(UploadResponse{
				Success: false,
				Message: fmt.Sprintf("File naming mismatch. Expected %s, got %s", expected, actual),
			})
			return
		}
	}
	
	// Save files
	files := []struct {
		file   io.Reader
		header string
	}{
		{infoFile, infoHeader.Filename},
		{linkFile, linkHeader.Filename},
		{nodeFile, nodeHeader.Filename},
		{metaFile, metaHeader.Filename},
	}
	
	for _, f := range files {
		err = saveUploadedFile(f.file, destination, f.header)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(UploadResponse{
				Success: false,
				Message: "Error uploading SCAR files",
				Error:   err.Error(),
			})
			return
		}
	}
	
	// Return success response with dataset info
	json.NewEncoder(w).Encode(UploadResponse{
		Success:        true,
		Message:        "SCAR files uploaded successfully",
		DatasetID:      datasetName,
		InfoFilePath:   filepath.Join(destination, infoHeader.Filename),
		LinkFilePath:   filepath.Join(destination, linkHeader.Filename),
		NodeFilePath:   filepath.Join(destination, nodeHeader.Filename),
		MetaFilePath:   filepath.Join(destination, metaHeader.Filename),
		K:              k,
		NK:             nk,
		TH:             th,
		ProcessingType: "scar",
	})
}
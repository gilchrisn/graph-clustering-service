package utils

import (
	"encoding/json"
	"net"
	"net/http"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"

	"graph-viz-backend/models"
)

// WriteSuccessResponse writes a successful JSON response
func WriteSuccessResponse(w http.ResponseWriter, message string, data interface{}) {
	response := models.APIResponse{
		Success: true,
		Message: message,
		Data:    data,
	}

	writeJSONResponse(w, http.StatusOK, response)
}

// WriteErrorResponse writes an error JSON response
func WriteErrorResponse(w http.ResponseWriter, statusCode int, message string, err error) {
	response := models.APIResponse{
		Success: false,
		Message: message,
	}

	if err != nil {
		response.Error = err.Error()
	}

	writeJSONResponse(w, statusCode, response)
}

// WriteValidationErrorResponse writes a validation error response
func WriteValidationErrorResponse(w http.ResponseWriter, message string, errors map[string]string) {
	response := models.APIResponse{
		Success: false,
		Message: message,
		Data:    map[string]interface{}{"validation_errors": errors},
	}

	writeJSONResponse(w, http.StatusBadRequest, response)
}

// writeJSONResponse is a helper function to write JSON responses
func writeJSONResponse(w http.ResponseWriter, statusCode int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	if err := json.NewEncoder(w).Encode(data); err != nil {
		log.Error().
			Err(err).
			Int("status_code", statusCode).
			Msg("Failed to encode JSON response")

		// Try to write a basic error response if JSON encoding fails
		if statusCode != http.StatusInternalServerError {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"success": false, "message": "Internal server error", "error": "JSON encoding failed"}`))
		}
	}
}

// ExtractPaginationParams extracts pagination parameters from request
func ExtractPaginationParams(r *http.Request) (page, limit int) {
	// Default values
	page = 1
	limit = 20

	if pageStr := r.URL.Query().Get("page"); pageStr != "" {
		if p, err := strconv.Atoi(pageStr); err == nil && p > 0 {
			page = p
		}
	}

	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 100 {
			limit = l
		}
	}

	return page, limit
}

// ValidateContentType checks if request has correct content type
func ValidateContentType(r *http.Request, expectedType string) bool {
	contentType := r.Header.Get("Content-Type")
	return strings.HasPrefix(contentType, expectedType)
}

// GetClientIP extracts the real client IP from request
func GetClientIP(r *http.Request) string {
	// Check X-Forwarded-For header first
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		// Get the first IP from the comma-separated list
		if ips := strings.Split(xff, ","); len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// Fall back to RemoteAddr
	if ip, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return ip
	}

	return r.RemoteAddr
}
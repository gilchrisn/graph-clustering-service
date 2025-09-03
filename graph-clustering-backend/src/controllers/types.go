// controllers/types.go
package controllers

// Response types for API endpoints
type UploadResponse struct {
	Success        bool    `json:"success"`
	Message        string  `json:"message"`
	DatasetID      string  `json:"datasetId,omitempty"`
	EdgeListPath   string  `json:"edgeListPath,omitempty"`
	PathFilePath   string  `json:"pathFilePath,omitempty"`
	PropertiesPath string  `json:"propertiesPath,omitempty"`
	InfoFilePath   string  `json:"infoFilePath,omitempty"`
	LinkFilePath   string  `json:"linkFilePath,omitempty"`
	NodeFilePath   string  `json:"nodeFilePath,omitempty"`
	MetaFilePath   string  `json:"metaFilePath,omitempty"`
	K              int     `json:"k,omitempty"`
	NK             int     `json:"nk,omitempty"`
	TH             float64 `json:"th,omitempty"`
	ProcessingType string  `json:"processingType,omitempty"`
	Error          string  `json:"error,omitempty"`
}

type ProcessResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Result  interface{} `json:"result,omitempty"`
	Error   string      `json:"error,omitempty"`
}

type HierarchyResponse struct {
	Success   bool        `json:"success"`
	Hierarchy interface{} `json:"hierarchy,omitempty"`
	Mapping   interface{} `json:"mapping,omitempty"`
	Message   string      `json:"message,omitempty"`
	Error     string      `json:"error,omitempty"`
}

type CoordinatesResponse struct {
	Success bool        `json:"success"`
	Nodes   interface{} `json:"nodes,omitempty"`
	Edges   interface{} `json:"edges,omitempty"`
	Message string      `json:"message,omitempty"`
	Error   string      `json:"error,omitempty"`
}

type StatisticsResponse struct {
	Success    bool        `json:"success"`
	Statistics interface{} `json:"statistics,omitempty"`
	Message    string      `json:"message,omitempty"`
	Error      string      `json:"error,omitempty"`
}

type ComparisonResponse struct {
	Success    bool        `json:"success"`
	Message    string      `json:"message"`
	Comparison interface{} `json:"comparison,omitempty"`
	Error      string      `json:"error,omitempty"`
}
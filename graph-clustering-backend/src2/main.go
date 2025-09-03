package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"graph-viz-backend/api"
	"graph-viz-backend/config"
	"graph-viz-backend/service"
)


func main() {
	// Initialize structured logging with zerolog
	zerolog.TimeFieldFormat = time.RFC3339
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339})
	
	log.Info().Msg("🚀 Starting Graph Visualization Backend v2.0")

	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load configuration")
	}

	log.Info().
		Str("address", cfg.Server.Address).
		Int("max_workers", cfg.Jobs.MaxWorkers).
		Dur("job_timeout", cfg.Jobs.JobTimeout).
		Msg("Configuration loaded")

	// Initialize services with dependency injection (follow dependency order)
	datasetService := service.NewDatasetService()
	jobService := service.NewJobService(datasetService)
	clusteringService := service.NewClusteringService(datasetService, jobService)
	comparisonService := service.NewComparisonService(clusteringService)  

	log.Info().Msg("Services initialized")

	// Initialize API handlers with all services
	handlers := api.NewHandlers(datasetService, clusteringService, jobService, comparisonService) 

	// Setup router with RESTful routes
	router := mux.NewRouter()
	api.SetupRoutes(router, handlers)

	// Add middleware stack
	router.Use(api.LoggingMiddleware)
	router.Use(api.CORSMiddleware)
	router.Use(api.RecoveryMiddleware)

	// Create HTTP server with proper timeouts
	server := &http.Server{
		Addr:         cfg.Server.Address,
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	// Start server in goroutine
	go func() {
		log.Info().
			Str("address", cfg.Server.Address).
			Msg("🌐 HTTP server starting")
		
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal().Err(err).Msg("Failed to start server")
		}
	}()

	// Wait for interrupt signal to gracefully shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info().Msg("🛑 Shutdown signal received")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	log.Info().Msg("Shutting down server...")

	if err := server.Shutdown(ctx); err != nil {
		log.Fatal().Err(err).Msg("Server forced to shutdown")
	}

	log.Info().Msg("✅ Server shutdown complete")
}
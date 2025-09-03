package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	Server ServerConfig
	Jobs   JobConfig
	Storage StorageConfig
}

type ServerConfig struct {
	Address      string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
}

type JobConfig struct {
	MaxWorkers        int
	JobTimeout        time.Duration
	CleanupInterval   time.Duration
	ResultTTL         time.Duration
}

type StorageConfig struct {
	UploadDir   string
	TempDir     string
	MaxFileSize int64
}

func Load() (*Config, error) {
	cfg := &Config{
		Server: ServerConfig{
			Address:      getEnv("SERVER_ADDRESS", ":8080"),
			ReadTimeout:  getDuration("SERVER_READ_TIMEOUT", 30*time.Second),
			WriteTimeout: getDuration("SERVER_WRITE_TIMEOUT", 30*time.Second),
		},
		Jobs: JobConfig{
			MaxWorkers:      getInt("JOB_MAX_WORKERS", 4),
			JobTimeout:      getDuration("JOB_TIMEOUT", 10*time.Minute),
			CleanupInterval: getDuration("JOB_CLEANUP_INTERVAL", 5*time.Minute),
			ResultTTL:       getDuration("JOB_RESULT_TTL", 1*time.Hour),
		},
		Storage: StorageConfig{
			UploadDir:   getEnv("UPLOAD_DIR", "./uploads"),
			TempDir:     getEnv("TEMP_DIR", "./temp"),
			MaxFileSize: getInt64("MAX_FILE_SIZE", 100*1024*1024), // 100MB
		},
	}

	return cfg, nil
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return defaultValue
}

func getInt64(key string, defaultValue int64) int64 {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.ParseInt(value, 10, 64); err == nil {
			return i
		}
	}
	return defaultValue
}

func getDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if d, err := time.ParseDuration(value); err == nil {
			return d
		}
	}
	return defaultValue
}
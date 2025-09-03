package louvain

import (
	"os"
	"runtime"
	"time"
	
	"github.com/spf13/viper"
	"github.com/rs/zerolog"
)

// Config manages algorithm configuration using Viper
type Config struct {
	v *viper.Viper
}

// NewConfig creates a new configuration with defaults
func NewConfig() *Config {
	v := viper.New()
	
	// Algorithm parameters
	v.SetDefault("algorithm.max_levels", 10)
	v.SetDefault("algorithm.max_iterations", 100)
	v.SetDefault("algorithm.min_modularity_gain", -100.0)
	v.SetDefault("algorithm.resolution", 1.0)
	v.SetDefault("algorithm.random_seed", time.Now().UnixNano())
	
	// Performance parameters
	v.SetDefault("performance.parallel", true)
	v.SetDefault("performance.chunk_size", 1000)
	v.SetDefault("performance.num_workers", runtime.NumCPU())
	
	// Logging parameters
	v.SetDefault("logging.level", "info")
	v.SetDefault("logging.progress_interval_ms", 1000)
	v.SetDefault("logging.enable_progress", true)

	v.SetDefault("output.store_graphs_at_each_level", false)
	
	return &Config{v: v}
}

// LoadFromFile loads configuration from file
func (c *Config) LoadFromFile(path string) error {
	c.v.SetConfigFile(path)
	return c.v.ReadInConfig()
}

// Getters for algorithm parameters
func (c *Config) MaxLevels() int { return c.v.GetInt("algorithm.max_levels") }
func (c *Config) MaxIterations() int { return c.v.GetInt("algorithm.max_iterations") }
func (c *Config) MinModularityGain() float64 { return c.v.GetFloat64("algorithm.min_modularity_gain") }
func (c *Config) Resolution() float64 { return c.v.GetFloat64("algorithm.resolution") }
func (c *Config) RandomSeed() int64 { return c.v.GetInt64("algorithm.random_seed") }

func (c *Config) Parallel() bool { return c.v.GetBool("performance.parallel") }
func (c *Config) ChunkSize() int { return c.v.GetInt("performance.chunk_size") }
func (c *Config) NumWorkers() int { return c.v.GetInt("performance.num_workers") }

func (c *Config) LogLevel() string { return c.v.GetString("logging.level") }
func (c *Config) ProgressIntervalMS() int { return c.v.GetInt("logging.progress_interval_ms") }
func (c *Config) EnableProgress() bool { return c.v.GetBool("logging.enable_progress") }

func (c *Config) EnableMoveTracking() bool { return c.v.GetBool("analysis.track_moves") }
func (c *Config) TrackingOutputFile() string { return c.v.GetString("analysis.output_file") }

func (c *Config) StoreGraphsAtEachLevel() bool { return c.v.GetBool("output.store_graphs_at_each_level") }

// Set allows dynamic configuration changes
func (c *Config) Set(key string, value interface{}) {
	c.v.Set(key, value)
}

// CreateLogger creates a zerolog logger based on config
func (c *Config) CreateLogger() zerolog.Logger {
	level, err := zerolog.ParseLevel(c.LogLevel())
	if err != nil {
		level = zerolog.InfoLevel
	}
	
	return zerolog.New(zerolog.ConsoleWriter{
		Out:        os.Stdout,
		TimeFormat: "15:04:05",
	}).Level(level).With().Timestamp().Str("service", "louvain").Logger()
}
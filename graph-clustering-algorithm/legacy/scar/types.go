package scar

import "math"

type UintE = uint32
type UintT = uint32

const UINT_E_MAX = math.MaxUint32

// SCARConfig holds configuration for SCAR algorithm
type SCARConfig struct {
	GraphFile    string
	PropertyFile string
	PathFile     string
	OutputFile   string
	EdgesFile    string
	Prefix       string // For output files
	K            int64
	NK           int64
	Threshold    float64
	UseLouvain   bool
	SketchOutput bool // Whether to output sketches
    WriteSketchGraph bool // Whether to write sketch graph files
    SketchGraphWeights bool // Whether to use weights in sketch graph files
    NumWorkers  int // Number of parallel workers
}

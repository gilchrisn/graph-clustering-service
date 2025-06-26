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
	K            int64
	NK           int64
	Threshold    float64
	UseLouvain   bool  // New flag to use Louvain structure
}
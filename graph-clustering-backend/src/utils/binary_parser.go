// utils/binary_parser.go
package utils

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

// ReadDoubleBinaryFile reads binary file containing doubles
// Returns array of double values
func ReadDoubleBinaryFile(filePath string) ([]float64, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("error reading binary file %s: %w", filePath, err)
	}
	
	// Each double is 8 bytes
	if len(data)%8 != 0 {
		return nil, fmt.Errorf("invalid file size for double array: %d bytes", len(data))
	}
	
	doubles := make([]float64, len(data)/8)
	
	for i := 0; i < len(doubles); i++ {
		bits := binary.LittleEndian.Uint64(data[i*8 : (i+1)*8])
		doubles[i] = math.Float64frombits(bits)
	}
	
	return doubles, nil
}

// ReadIntBinaryFile reads binary file containing integers
func ReadIntBinaryFile(filePath string) ([]int, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("error reading binary file %s: %w", filePath, err)
	}
	
	// Each int is 4 bytes
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("invalid file size for int array: %d bytes", len(data))
	}
	
	ints := make([]int, len(data)/4)
	
	for i := 0; i < len(ints); i++ {
		ints[i] = int(binary.LittleEndian.Uint32(data[i*4 : (i+1)*4]))
	}
	
	return ints, nil
}

// ReadFloatBinaryFile reads binary file containing floats
func ReadFloatBinaryFile(filePath string) ([]float32, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("error reading binary file %s: %w", filePath, err)
	}
	
	// Each float is 4 bytes
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("invalid file size for float array: %d bytes", len(data))
	}
	
	floats := make([]float32, len(data)/4)
	
	for i := 0; i < len(floats); i++ {
		bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		floats[i] = math.Float32frombits(bits)
	}
	
	return floats, nil
}
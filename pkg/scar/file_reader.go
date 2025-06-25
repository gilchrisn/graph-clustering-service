package scar

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// FileReader handles reading properties and paths from files
type FileReader struct{}

func NewFileReader() *FileReader {
	return &FileReader{}
}

func (fr *FileReader) ReadProperties(filename string, n int64) ([]UintE, error) {
	properties := make([]UintE, n)
	
	if filename == "" {
		return properties, nil
	}
	
	file, err := os.Open(filename)
	if err != nil {
		return properties, fmt.Errorf("could not open property file %s: %v", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			nodeId, err1 := strconv.ParseInt(parts[0], 10, 64)
			property, err2 := strconv.ParseInt(parts[1], 10, 32)
			if err1 == nil && err2 == nil && nodeId < n {
				properties[nodeId] = UintE(property)
			}
		}
	}
	
	return properties, scanner.Err()
}

func (fr *FileReader) ReadPath(filename string) ([]UintE, int64, error) {
	path := make([]UintE, 20)
	pathLength := int64(0)
	
	if filename == "" {
		path[0] = 0
		return path, 1, nil
	}
	
	file, err := os.Open(filename)
	if err != nil {
		path[0] = 0
		return path, 1, fmt.Errorf("could not open path file %s: %v", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		
		if label, err := strconv.ParseInt(line, 10, 32); err == nil {
			if pathLength < 20 {
				path[pathLength] = UintE(label)
				pathLength++
			}
		}
	}
	
	if pathLength == 0 {
		path[0] = 0
		pathLength = 1
	}
	
	return path, pathLength, scanner.Err()
}

// Legacy functions for backward compatibility
func ReadProperties(filename string, n int64) []UintE {
	reader := NewFileReader()
	properties, err := reader.ReadProperties(filename, n)
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
	}
	return properties
}

func ReadPath(filename string) ([]UintE, int64) {
	reader := NewFileReader()
	path, length, err := reader.ReadPath(filename)
	if err != nil {
		fmt.Printf("Warning: %v\n", err)
	}
	return path, length
}
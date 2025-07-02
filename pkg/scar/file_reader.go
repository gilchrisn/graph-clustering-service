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
		fmt.Println("Warning: No properties file specified, all nodes will have property 0")
		return properties, nil
	}
	
	file, err := os.Open(filename)
	if err != nil {
		return properties, fmt.Errorf("could not open property file %s: %v", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineNum := 0
	assignmentCount := 0
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lineNum++
		
		if line == "" || strings.HasPrefix(line, "#") {
			fmt.Printf("Line %d: %s (skipped)\n", lineNum, line)
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			nodeId, err1 := strconv.ParseInt(parts[0], 10, 64)
			property, err2 := strconv.ParseInt(parts[1], 10, 32)
			if err1 == nil && err2 == nil && nodeId < n {
				properties[nodeId] = UintE(property)
				// fmt.Printf("Line %d: Node %d → property %d\n", lineNum, nodeId, property)
				assignmentCount++
			} else {
				fmt.Printf("Line %d: %s (parse error or nodeId >= n: err1=%v, err2=%v, nodeId=%d, n=%d)\n", 
					lineNum, line, err1, err2, nodeId, n)
			}
		} else {
			fmt.Printf("Line %d: %s (insufficient fields)\n", lineNum, line)
		}
	}
	
	// fmt.Printf("Properties summary: %d assignments made\n", assignmentCount)
	// fmt.Println("Final property assignment:")
	// for i := int64(0); i < n; i++ {
	// 	fmt.Printf("Node %d: property %d\n", i, properties[i])
	// }
	
	return properties, scanner.Err()
}

func (fr *FileReader) ReadPath(filename string) ([]UintE, int64, error) {
	path := make([]UintE, 20)
	pathLength := int64(0)
	
	if filename == "" {
		fmt.Println("Warning: No path file specified, using default [0]")
		path[0] = 0
		return path, 1, nil
	}
	
	file, err := os.Open(filename)
	if err != nil {
		path[0] = 0
		return path, 1, fmt.Errorf("could not open path file %s: %v", filename, err)
	}
	defer file.Close()

	fmt.Printf("=== READING PATH FILE: %s ===\n", filename)
	scanner := bufio.NewScanner(file)
	lineNum := 0
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lineNum++
		
		if line == "" || strings.HasPrefix(line, "#") {
			fmt.Printf("Line %d: %s (skipped)\n", lineNum, line)
			continue
		}
		
		if label, err := strconv.ParseInt(line, 10, 32); err == nil {
			if pathLength < 20 {
				path[pathLength] = UintE(label)
				// fmt.Printf("Line %d: path[%d] = %d\n", lineNum, pathLength, label)
				pathLength++
			} else {
				fmt.Printf("Line %d: %s (path too long, max 20)\n", lineNum, line)
			}
		} else {
			fmt.Printf("Line %d: %s (parse error: %v)\n", lineNum, line, err)
		}
	}
	
	if pathLength == 0 {
		fmt.Println("Warning: Empty path file, using default [0]")
		path[0] = 0
		pathLength = 1
	}
	
	fmt.Printf("Path summary: length=%d, path=%v\n", pathLength, path[:pathLength])
	
	return path, pathLength, scanner.Err()
}

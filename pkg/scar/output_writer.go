package scar

import (
	"fmt"
	"os"
)

// OutputWriter handles writing results to files
type OutputWriter struct{}

func NewOutputWriter() *OutputWriter {
	return &OutputWriter{}
}

func (ow *OutputWriter) WriteResults(outputFile string, community []int64, n int64) error {
	file, err := os.Create(outputFile)
	if err != nil {
		return err
	}
	defer file.Close()
	
	for i := int64(0); i < n; i++ {
		if community[i] != -1 {
			fmt.Fprintf(file, "%d %d\n", i, community[i])
		}
	}
	
	return nil
}

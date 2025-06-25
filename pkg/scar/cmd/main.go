package main

import (
	"os"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

func main() {
	// Pass command line arguments to SCAR library
	scar.RunSCAR(os.Args[1:])
}
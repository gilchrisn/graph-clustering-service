package main

import (
	"os"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

func main() {
	scar.RunSCAR(os.Args[1:])
}
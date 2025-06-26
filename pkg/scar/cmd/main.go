package main

import (
	"os"
	"github.com/gilchrisn/graph-clustering-service/pkg/scar"
)

func main() {
	scar.RunSCAR(os.Args[1:])
}

// To run: go build -o scar.exe && .\scar.exe "test_graph.txt" -pro="properties.txt" -path="path.txt" -k=6 -nk=1 -o="communities.txt" -louvain
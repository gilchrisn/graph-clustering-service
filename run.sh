#!/usr/bin/env bash
set -euo pipefail

# 1) grab your four arguments:
GRAPH="$1"       # e.g. path/to/graph.edgelist
PROPS="$2"       # e.g. path/to/props.json
PATHFILE="$3"    # e.g. path/to/pathfile
OUTBASE="$4"     # e.g. ./pipeline_outputs

# 2) iterate powers of two
for k in 2 4 8 16 32 64 128 256 512; do
  echo
  echo "=== running with SCAR_K=$k ==="
  export SCAR_K=$k

  OUTDIR="${OUTBASE}/scar_K_${k}"
  mkdir -p "$OUTDIR"

  # 3) invoke your Go program
  go run ./simple_pipeline.go ./data/dummy_graph.txt ./data/dummy_properties.txt ./data/dummy_path.txt pipeline_output
#   ./main.exe "$GRAPH" "$PROPS" "$PATHFILE" "$OUTDIR"
done

package scar

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// SketchF handles sketch computation operations
type SketchF struct {
	sketches         []UintE
	vertexProperties []UintE
	k                int64
	nk               int64
	n                int64
	iter             UintE
	path             []UintE
}

func NewSketchF(sketches []UintE, vertexProperties []UintE, k, nk, n int64, iter UintE, path []UintE) *SketchF {
	return &SketchF{
		sketches:         sketches,
		vertexProperties: vertexProperties,
		k:                k,
		nk:               nk,
		n:                n,
		iter:             iter,
		path:             path,
	}
}

func (sf *SketchF) Update(s, d int64) bool {
	if sf.vertexProperties[d] != sf.path[sf.iter] {
		return false
	}

	for l := int64(0); l < sf.nk; l++ {
		sf.updateSketchLayer(s, d, l)
	}
	return true
}

func (sf *SketchF) updateSketchLayer(s, d, l int64) {
	sValuesStart := l*sf.n*sf.k + s*sf.k + int64(sf.iter-1)*sf.n*sf.k*sf.nk
	dValuesStart := l*sf.n*sf.k + d*sf.k + int64(sf.iter)*sf.n*sf.k*sf.nk
	
	temp := make([]UintE, sf.k)
	i, t, j := int64(0), int64(0), int64(0)

	// Merge sValues and dValues into temp array
	for t < sf.k {
		sVal, dVal := sf.getSketchValues(sValuesStart, dValuesStart, i, j)
		
		if sVal == dVal && sVal != math.MaxUint32 {
			temp[t] = sVal
			t++
			i++
			j++
		} else if i < sf.k && (j >= sf.k || sVal < dVal) {
			temp[t] = sVal
			t++
			i++
		} else if j < sf.k {
			temp[t] = dVal
			t++
			j++
		} else {
			break
		}
	}

	// Copy the smallest k values back to dValues
	for idx := int64(0); idx < sf.k && idx < t; idx++ {
		sf.sketches[dValuesStart+idx] = temp[idx]
	}
}

func (sf *SketchF) getSketchValues(sValuesStart, dValuesStart, i, j int64) (UintE, UintE) {
	sVal := UintE(math.MaxUint32)
	dVal := UintE(math.MaxUint32)
	
	if i < sf.k {
		sVal = sf.sketches[sValuesStart+i]
	}
	if j < sf.k {
		dVal = sf.sketches[dValuesStart+j]
	}
	
	return sVal, dVal
}

func (sf *SketchF) Cond(d int64) bool {
	return sf.vertexProperties[d] == sf.path[sf.iter]
}

// SketchComputer handles the computation of sketches for the graph
type SketchComputer struct {
	rng *rand.Rand
}

func NewSketchComputer() *SketchComputer {
	return &SketchComputer{
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (sc *SketchComputer) ComputeForGraph(
	G *GraphStructure,
	sketches []UintE,
	path []UintE,
	pathLength int64,
	vertexProperties []UintE,
	nodeHashValue []UintE,
	k, nk int64,
) {
	n := G.n
	firstLabel := path[0]
	frontier := sc.initializeFrontier(n, firstLabel, vertexProperties, sketches, nodeHashValue, k, nk)
	
	frontierVS := NewVertexSubsetFromArray(n, frontier)
	
	// Iterate through path
	for iter := UintE(1); iter < UintE(pathLength); iter++ {
		
		f := NewSketchF(sketches, vertexProperties, k, nk, n, iter, path)
		output := sc.edgeMap(G, frontierVS, f)
		
		// Print sketch state after this iteration
		sc.printSketchState(sketches, n, k, nk, int64(iter), pathLength)
		
		frontierVS.Del()
		frontierVS = output
	}
	
	frontierVS.Del()
}

func (sc *SketchComputer) printSketchState(sketches []UintE, n, k, nk, currentIter int64, pathLength int64) {
	for i := int64(0); i < n; i++ {
		hasNonMax := false
		for j := int64(0); j < nk; j++ {
			for ki := int64(0); ki < k; ki++ {
				idx := j*n*k + i*k + ki + currentIter*n*k*nk
				if int64(idx) < int64(len(sketches)) && sketches[idx] != math.MaxUint32 {
					hasNonMax = true
					break
				}
			}
			if hasNonMax {
				break
			}
		}
		
		if hasNonMax {
			fmt.Printf("Node %d sketches: ", i)
			for j := int64(0); j < nk; j++ {
				fmt.Printf("layer%d=[", j)
				for ki := int64(0); ki < k; ki++ {
					idx := j*n*k + i*k + ki + currentIter*n*k*nk
					if int64(idx) < int64(len(sketches)) {
						val := sketches[idx]
						if val == math.MaxUint32 {
							fmt.Printf("MAX")
						} else {
							fmt.Printf("%d", val)
						}
					} else {
						fmt.Printf("OOB")
					}
					if ki < k-1 {
						fmt.Printf(",")
					}
				}
				fmt.Printf("] ")
			}
			fmt.Printf("\n")
		}
	}
}

func (sc *SketchComputer) initializeFrontier(
	n int64,
	firstLabel UintE,
	vertexProperties []UintE,
	sketches []UintE,
	nodeHashValue []UintE,
	k, nk int64,
) []bool {
	frontier := make([]bool, n)
	
	fmt.Println("=== SKETCH INITIALIZATION ===")
	fmt.Printf("First label: %d\n", firstLabel)
	
	for i := int64(0); i < n; i++ {
		if vertexProperties[i] == firstLabel {
			fmt.Printf("Node %d (property=%d) gets sketches: ", i, vertexProperties[i])
			for j := int64(0); j < nk; j++ {
				uniformValue := uint32(1000000 + i*1000 + j) // deterministic for testing
				if uniformValue == math.MaxUint32 {
					uniformValue--
				}
				sketches[j*n*k+i*k] = uniformValue
				nodeHashValue[i*nk+j] = uniformValue + 1
				fmt.Printf("layer%d=%d ", j, uniformValue)
				frontier[i] = true
			}
			fmt.Printf("\n")
		}
	}
	
	// Print complete initial sketch state
	fmt.Println("\nInitial sketch state:")
	for i := int64(0); i < n; i++ {
		if frontier[i] {
			fmt.Printf("Node %d sketches: ", i)
			for j := int64(0); j < nk; j++ {
				fmt.Printf("layer%d=[", j)
				for ki := int64(0); ki < k; ki++ {
					val := sketches[j*n*k+i*k+ki]
					if val == math.MaxUint32 {
						fmt.Printf("MAX")
					} else {
						fmt.Printf("%d", val)
					}
					if ki < k-1 {
						fmt.Printf(",")
					}
				}
				fmt.Printf("] ")
			}
			fmt.Printf("\n")
		}
	}
	fmt.Println()
	
	return frontier
}

func (sc *SketchComputer) edgeMap(G *GraphStructure, vs *VertexSubset, f *SketchF) *VertexSubset {
	numVertices := G.n
	m := vs.NumNonzeros()
	
	if m == 0 {
		return NewVertexSubset(numVertices)
	}
	
	vs.ToSparse()
	newFrontier := make([]bool, numVertices)
	
	for i := 0; i < int(m); i++ {
		v := vs.Vtx(i)
		for _, neighbor := range G.V[v].neighbors {
			if f.Cond(neighbor) && f.Update(v, neighbor) {
				newFrontier[neighbor] = true
			}
		}
	}
	
	return NewVertexSubsetFromArray(numVertices, newFrontier)
}
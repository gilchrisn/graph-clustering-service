package scar

// VertexSubset represents a subset of vertices with efficient operations
type VertexSubset struct {
	n       int64
	d       []bool  // dense representation
	s       []int64 // sparse representation
	isDense bool
	size    int64
}

func NewVertexSubset(n int64) *VertexSubset {
	return &VertexSubset{
		n:       n,
		d:       make([]bool, n),
		isDense: true,
		size:    0,
	}
}

func NewVertexSubsetFromArray(n int64, frontier []bool) *VertexSubset {
	vs := &VertexSubset{
		n:       n,
		d:       make([]bool, n),
		isDense: true,
		size:    0,
	}
	
	copy(vs.d, frontier)
	for i := int64(0); i < n; i++ {
		if frontier[i] {
			vs.size++
		}
	}
	return vs
}

func (vs *VertexSubset) IsIn(v int64) bool {
	if vs.isDense {
		return vs.d[v]
	}
	return vs.binarySearch(v)
}

func (vs *VertexSubset) binarySearch(v int64) bool {
	left, right := 0, len(vs.s)-1
	for left <= right {
		mid := (left + right) / 2
		if vs.s[mid] == v {
			return true
		} else if vs.s[mid] < v {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return false
}

func (vs *VertexSubset) Size() int64 {
	return vs.size
}

func (vs *VertexSubset) NumRows() int64 {
	return vs.n
}

func (vs *VertexSubset) NumNonzeros() int64 {
	return vs.size
}

func (vs *VertexSubset) ToSparse() {
	if !vs.isDense {
		return
	}
	vs.s = vs.s[:0]
	for i := int64(0); i < vs.n; i++ {
		if vs.d[i] {
			vs.s = append(vs.s, i)
		}
	}
	vs.isDense = false
}

func (vs *VertexSubset) ToDense() {
	if vs.isDense {
		return
	}
	for i := int64(0); i < vs.n; i++ {
		vs.d[i] = false
	}
	for _, v := range vs.s {
		vs.d[v] = true
	}
	vs.isDense = true
}

func (vs *VertexSubset) Vtx(i int) int64 {
	if vs.isDense {
		count := 0
		for j := int64(0); j < vs.n; j++ {
			if vs.d[j] {
				if count == i {
					return j
				}
				count++
			}
		}
	}
	return vs.s[i]
}

func (vs *VertexSubset) Del() {
	vs.d = nil
	vs.s = nil
}
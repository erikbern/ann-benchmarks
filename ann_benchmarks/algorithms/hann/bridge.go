// Package main builds a c-shared library that exposes the hann HNSW index
// through a small C ABI, so the Python wrapper can drive it with ctypes.
// Go objects never cross the boundary: each index lives in a mutex-guarded
// map and is referred to by an int64 handle.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"strings"
	"sync"
	"unsafe"

	"github.com/habedi/hann/core"
	"github.com/habedi/hann/hnsw"
)

type indexEntry struct {
	index *hnsw.Index
	dim   int
	count int // running count of inserted vectors, used to assign ids
}

var (
	registry   = make(map[int64]*indexEntry)
	registryMu sync.Mutex
	nextHandle int64 = 1
)

func getEntry(handle int64) *indexEntry {
	registryMu.Lock()
	defer registryMu.Unlock()
	return registry[handle]
}

// hann_hnsw_new creates an HNSW index and returns a handle to it, or -1 on
// error. The metric string is either "euclidean" or "cosine".
//
//export hann_hnsw_new
func hann_hnsw_new(dim, m, efConstruction C.int64_t, metric *C.char) (handle C.int64_t) {
	defer func() {
		if r := recover(); r != nil {
			handle = -1
		}
	}()

	var coreMetric core.Metric
	switch strings.ToLower(C.GoString(metric)) {
	case "euclidean":
		coreMetric = core.Euclidean
	case "cosine":
		coreMetric = core.Cosine
	default:
		return -1
	}

	index, err := hnsw.New(int(dim),
		hnsw.WithM(int(m)),
		hnsw.WithEfConstruction(int(efConstruction)),
		hnsw.WithMetric(coreMetric),
	)
	if err != nil {
		return -1
	}

	registryMu.Lock()
	defer registryMu.Unlock()
	h := nextHandle
	nextHandle++
	registry[h] = &indexEntry{index: index, dim: int(dim)}
	return C.int64_t(h)
}

// hann_hnsw_add_batch adds n vectors of the given dimension, laid out
// row-major in flat, assigning ids sequentially from the running count of
// the index. It returns the number of vectors added, or -1 on error.
//
//export hann_hnsw_add_batch
func hann_hnsw_add_batch(handle C.int64_t, flat *C.float, n, dim C.int64_t) (added C.int64_t) {
	defer func() {
		if r := recover(); r != nil {
			added = -1
		}
	}()

	entry := getEntry(int64(handle))
	if entry == nil || flat == nil || n <= 0 || int(dim) != entry.dim {
		return -1
	}

	rows := int(n)
	d := int(dim)
	src := unsafe.Slice((*float32)(unsafe.Pointer(flat)), rows*d)

	// Copy every row into a fresh Go slice: the C buffer is owned by the
	// caller and must not be retained past this call.
	const chunk = 50000
	for start := 0; start < rows; start += chunk {
		end := start + chunk
		if end > rows {
			end = rows
		}
		batch := make(map[int][]float32, end-start)
		for i := start; i < end; i++ {
			row := make([]float32, d)
			copy(row, src[i*d:(i+1)*d])
			batch[entry.count+i] = row
		}
		if err := core.BulkAdd(entry.index, batch); err != nil {
			return -1
		}
	}
	entry.count += rows
	return n
}

// hann_hnsw_set_ef changes the search breadth of the index. It returns 0 on
// success and -1 on error.
//
//export hann_hnsw_set_ef
func hann_hnsw_set_ef(handle, ef C.int64_t) (status C.int64_t) {
	defer func() {
		if r := recover(); r != nil {
			status = -1
		}
	}()

	entry := getEntry(int64(handle))
	if entry == nil {
		return -1
	}
	if err := entry.index.SetEf(int(ef)); err != nil {
		return -1
	}
	return 0
}

// hann_hnsw_search searches the index for the k nearest neighbors of the
// query and writes their ids into out, which must have room for k int32
// values. It returns the number of ids written, or -1 on error.
//
//export hann_hnsw_search
func hann_hnsw_search(handle C.int64_t, query *C.float, dim, k C.int64_t, out *C.int32_t) (found C.int64_t) {
	defer func() {
		if r := recover(); r != nil {
			found = -1
		}
	}()

	entry := getEntry(int64(handle))
	if entry == nil || query == nil || out == nil || k <= 0 || int(dim) != entry.dim {
		return -1
	}

	d := int(dim)
	src := unsafe.Slice((*float32)(unsafe.Pointer(query)), d)
	q := make([]float32, d)
	copy(q, src)

	neighbors, err := entry.index.Search(q, int(k))
	if err != nil {
		return -1
	}

	dst := unsafe.Slice((*int32)(unsafe.Pointer(out)), int(k))
	written := 0
	for _, nb := range neighbors {
		if written >= int(k) {
			break
		}
		dst[written] = int32(nb.ID)
		written++
	}
	return C.int64_t(written)
}

// hann_hnsw_free releases the index behind the handle. Freeing an unknown
// handle is a no-op.
//
//export hann_hnsw_free
func hann_hnsw_free(handle C.int64_t) {
	defer func() { _ = recover() }()
	registryMu.Lock()
	defer registryMu.Unlock()
	delete(registry, int64(handle))
}

func main() {}

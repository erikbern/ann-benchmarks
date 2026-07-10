/*
 * QINGMING-ENGINE: AMD RDNA3 EDITION (Navi 31)
 * WRAPPER: PyBind11 for ANN-Benchmarks
 */

#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define TOP_K 100
#define WAVE_SIZE 32
#define THREADS_PER_BLOCK 256

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err)); \
    } \
} while (0)

__device__ inline float wave_reduce_sum(float val) {
    for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// ---------------- KERNELS (保留你的原始逻辑) ----------------

template <int DIM>
__global__ __launch_bounds__(256) 
void kernel_throughput_float4(const float4* __restrict__ queries, const float4* __restrict__ database, int n_base, int* __restrict__ out_ids, int batch_size) {
    int wave_id = threadIdx.x / WAVE_SIZE; int lane_id = threadIdx.x % WAVE_SIZE;
    int qid = blockIdx.x * (blockDim.x / WAVE_SIZE) + wave_id;
    if (qid >= batch_size) return;
    
    constexpr int ITER = (DIM / 4) / WAVE_SIZE; 
    float4 my_q[ITER];
    const float4* q_ptr = queries + qid * (DIM / 4);
    
    #pragma unroll
    for(int i=0; i<ITER; ++i) my_q[i] = q_ptr[lane_id + i*WAVE_SIZE];
    
    float best_dists[TOP_K]; int best_ids[TOP_K];
    if (lane_id == 0) for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
    
    for (int i = 0; i < n_base; ++i) {
        const float4* vec_ptr = database + i * (DIM / 4);
        float dist = 0.0f;
        #pragma unroll
        for(int k=0; k<ITER; ++k) {
            float4 v = vec_ptr[lane_id + k*WAVE_SIZE];
            dist += (my_q[k].x - v.x)*(my_q[k].x - v.x) + (my_q[k].y - v.y)*(my_q[k].y - v.y) + (my_q[k].z - v.z)*(my_q[k].z - v.z) + (my_q[k].w - v.w)*(my_q[k].w - v.w);
        }
        dist = wave_reduce_sum(dist);
        if (lane_id == 0 && dist < best_dists[TOP_K-1]) {
            int pos = TOP_K - 1; while (pos > 0 && best_dists[pos-1] > dist) { best_dists[pos] = best_dists[pos-1]; best_ids[pos] = best_ids[pos-1]; pos--; }
            best_dists[pos] = dist; best_ids[pos] = i; 
        }
    }
    if (lane_id == 0) for (int k = 0; k < TOP_K; ++k) out_ids[qid * TOP_K + k] = best_ids[k];
}

template <int DIM>
__global__ __launch_bounds__(256) 
void kernel_throughput_float(const float* __restrict__ queries, const float* __restrict__ database, int n_base, int* __restrict__ out_ids, int batch_size) {
    int wave_id = threadIdx.x / WAVE_SIZE; int lane_id = threadIdx.x % WAVE_SIZE;
    int qid = blockIdx.x * (blockDim.x / WAVE_SIZE) + wave_id;
    if (qid >= batch_size) return;
    constexpr int ITER = DIM / WAVE_SIZE; 
    float my_q[ITER];
    const float* q_ptr = queries + qid * DIM;
    #pragma unroll
    for(int i=0; i<ITER; ++i) my_q[i] = q_ptr[lane_id + i*WAVE_SIZE];
    float best_dists[TOP_K]; int best_ids[TOP_K];
    if (lane_id == 0) for(int k=0; k<TOP_K; ++k) { best_dists[k] = 1e30f; best_ids[k] = -1; }
    for (int i = 0; i < n_base; ++i) {
        const float* vec_ptr = database + i * DIM;
        float dist = 0.0f;
        #pragma unroll
        for(int k=0; k<ITER; ++k) {
            float v = vec_ptr[lane_id + k*WAVE_SIZE]; float diff = my_q[k] - v; dist += diff * diff;
        }
        dist = wave_reduce_sum(dist);
        if (lane_id == 0 && dist < best_dists[TOP_K-1]) {
            int pos = TOP_K - 1; while (pos > 0 && best_dists[pos-1] > dist) { best_dists[pos] = best_dists[pos-1]; best_ids[pos] = best_ids[pos-1]; pos--; }
            best_dists[pos] = dist; best_ids[pos] = i; 
        }
    }
    if (lane_id == 0) for (int k = 0; k < TOP_K; ++k) out_ids[qid * TOP_K + k] = best_ids[k];
}

// ---------------- ENGINE WRAPPER ----------------

class QingMingEngine {
    void *d_database; 
    void *d_queries;
    int *d_results;
    int dimension;
    int n_base;
    size_t max_q_alloc;

public:
    QingMingEngine(py::array_t<float> db_data) {
        auto buf = db_data.request();
        n_base = buf.shape[0];
        dimension = buf.shape[1];
        
        size_t db_size = n_base * dimension * sizeof(float);
        HIP_CHECK(hipMalloc(&d_database, db_size));
        HIP_CHECK(hipMemcpy(d_database, buf.ptr, db_size, hipMemcpyHostToDevice));

        max_q_alloc = 20000;
        HIP_CHECK(hipMalloc(&d_queries, max_q_alloc * dimension * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_results, max_q_alloc * TOP_K * sizeof(int)));
    }

    ~QingMingEngine() {
        hipFree(d_database);
        hipFree(d_queries);
        hipFree(d_results);
    }

    py::array_t<int> query(py::array_t<float> q_data, int k) {
        auto buf = q_data.request();
        int n_q = buf.shape[0];
        
        // 动态扩容
        if (n_q > max_q_alloc) {
            hipFree(d_queries); hipFree(d_results);
            max_q_alloc = n_q * 1.5;
            HIP_CHECK(hipMalloc(&d_queries, max_q_alloc * dimension * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_results, max_q_alloc * TOP_K * sizeof(int)));
        }

        HIP_CHECK(hipMemcpy(d_queries, buf.ptr, n_q * dimension * sizeof(float), hipMemcpyHostToDevice));

        int threads = 256;
        int blocks = (n_q * WAVE_SIZE + threads - 1) / threads;

        if (dimension == 128) {
            kernel_throughput_float4<128><<<blocks, threads>>>((float4*)d_queries, (float4*)d_database, n_base, d_results, n_q);
        } else if (dimension == 960) {
            kernel_throughput_float<960><<<blocks, threads>>>((float*)d_queries, (float*)d_database, n_base, d_results, n_q);
        } else {
            // Fallback for other dims if needed, or throw error
             throw std::runtime_error("Unsupported Dimension for AMD Kernel");
        }
        
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<int> host_res(n_q * TOP_K);
        HIP_CHECK(hipMemcpy(host_res.data(), d_results, n_q * TOP_K * sizeof(int), hipMemcpyDeviceToHost));
        
        // 截取用户需要的 Top-K
        if (k != TOP_K) {
            std::vector<int> final_res(n_q * k);
            for(int i=0; i<n_q; ++i) {
                for(int j=0; j<k; ++j) {
                    final_res[i*k + j] = host_res[i*TOP_K + j];
                }
            }
            return py::array_t<int>({n_q, k}, final_res.data());
        }

        return py::array_t<int>({n_q, TOP_K}, host_res.data());
    }
};

PYBIND11_MODULE(qingming_hip, m) {
    py::class_<QingMingEngine>(m, "QingMingEngine")
        .def(py::init<py::array_t<float>>())
        .def("query", &QingMingEngine::query);
}

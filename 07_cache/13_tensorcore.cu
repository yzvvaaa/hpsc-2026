#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

// ---- Tunables (override at compile time with -DBLOCK_TILE_K=64 etc.) ----
#ifndef BLOCK_TILE_M
#define BLOCK_TILE_M 128
#endif
#ifndef BLOCK_TILE_N
#define BLOCK_TILE_N 128
#endif
#ifndef BLOCK_TILE_K
#define BLOCK_TILE_K 32      // 32 keeps register pressure low; try 64 once verified
#endif
#ifndef WARP_TILE_M
#define WARP_TILE_M 64
#endif
#ifndef WARP_TILE_N
#define WARP_TILE_N 64
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef SMEM_PADDING
#define SMEM_PADDING 16
#endif

// Each thread computes a WARP_TILE_M x WARP_TILE_N output region (in 16x16 wmma tiles).
// SMEM is DOUBLE BUFFERED so the next K-tile can be loaded while the current one computes.
//
// SMEM layouts:
//   A: [BLOCK_TILE_K][BLOCK_TILE_M + PAD]  (K-major)  -> matrix_a col_major
//   B: [BLOCK_TILE_N][BLOCK_TILE_K + PAD]  (N-major)  -> matrix_b col_major
// The B layout is the key change vs. the [K][N] starter: the 4 contiguous-in-K values
// read from column-major global B now land contiguously in SMEM, so the store is a
// vectorized half2 (no scalar stores, no bank conflicts).

__global__ __launch_bounds__(BLOCK_SIZE)
void kernel(int dim_m, int dim_n, int dim_k, const float *d_a, const float *d_b, float *d_c) {
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;

  constexpr int sizeA = BLOCK_TILE_K * (BLOCK_TILE_M + SMEM_PADDING);  // halfs per A buffer
  constexpr int sizeB = BLOCK_TILE_N * (BLOCK_TILE_K + SMEM_PADDING);  // halfs per B buffer

  extern __shared__ half smem[];
  half* smem_A = smem;                 // 2 buffers, stride sizeA
  half* smem_B = smem + 2 * sizeA;     // 2 buffers, stride sizeB

  // ---- Accumulators (live in registers for the whole K loop) ----
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WARP_TILE_M / 16][WARP_TILE_N / 16];
  #pragma unroll
  for (int r = 0; r < WARP_TILE_M / 16; r++)
    #pragma unroll
    for (int c = 0; c < WARP_TILE_N / 16; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  const int warps_per_col = BLOCK_TILE_N / WARP_TILE_N;
  const int warp_row = warp_id / warps_per_col;
  const int warp_col = warp_id % warps_per_col;

  // ---- Global -> SMEM load descriptors ----
  constexpr int steps_a = (BLOCK_TILE_K * BLOCK_TILE_M) / (4 * BLOCK_SIZE);
  constexpr int num_cols_vec_a = BLOCK_TILE_M / 4;
  constexpr int steps_b = (BLOCK_TILE_K * BLOCK_TILE_N) / (4 * BLOCK_SIZE);
  constexpr int num_cols_vec_b = BLOCK_TILE_K / 4;   // vectorize along K

  const int offset_a_m = BLOCK_TILE_M * blockIdx.x;
  const int offset_b_n = BLOCK_TILE_N * blockIdx.y;

  // Register staging for the prefetched tile (kept as half to halve register pressure)
  half2 rA[steps_a * 2];
  half2 rB[steps_b * 2];

  // ===== PROLOGUE: load tile @ k=0 directly into buffer 0 =====
  #pragma unroll
  for (int step = 0; step < steps_a; ++step) {
    int idx = step * BLOCK_SIZE + tid;
    int j = idx / num_cols_vec_a;
    int m = (idx % num_cols_vec_a) * 4;
    float4 v = *reinterpret_cast<const float4*>(&d_a[(0 + j) * dim_m + offset_a_m + m]);
    *reinterpret_cast<half2*>(&smem_A[j * (BLOCK_TILE_M + SMEM_PADDING) + m + 0]) = __float22half2_rn(make_float2(v.x, v.y));
    *reinterpret_cast<half2*>(&smem_A[j * (BLOCK_TILE_M + SMEM_PADDING) + m + 2]) = __float22half2_rn(make_float2(v.z, v.w));
  }
  #pragma unroll
  for (int step = 0; step < steps_b; ++step) {
    int idx = step * BLOCK_SIZE + tid;
    int n = idx / num_cols_vec_b;
    int kk = (idx % num_cols_vec_b) * 4;
    float4 v = *reinterpret_cast<const float4*>(&d_b[(offset_b_n + n) * dim_k + 0 + kk]);
    *reinterpret_cast<half2*>(&smem_B[n * (BLOCK_TILE_K + SMEM_PADDING) + kk + 0]) = __float22half2_rn(make_float2(v.x, v.y));
    *reinterpret_cast<half2*>(&smem_B[n * (BLOCK_TILE_K + SMEM_PADDING) + kk + 2]) = __float22half2_rn(make_float2(v.z, v.w));
  }
  __syncthreads();

  int read_buf = 0;
  for (int k_tile = 0; k_tile < dim_k; k_tile += BLOCK_TILE_K) {
    const int next = k_tile + BLOCK_TILE_K;
    const int write_buf = read_buf ^ 1;
    const bool has_next = (next < dim_k);

    // ---- 1) Prefetch NEXT tile from global into registers (latency overlaps compute) ----
    if (has_next) {
      #pragma unroll
      for (int step = 0; step < steps_a; ++step) {
        int idx = step * BLOCK_SIZE + tid;
        int j = idx / num_cols_vec_a;
        int m = (idx % num_cols_vec_a) * 4;
        float4 v = *reinterpret_cast<const float4*>(&d_a[(next + j) * dim_m + offset_a_m + m]);
        rA[step * 2 + 0] = __float22half2_rn(make_float2(v.x, v.y));
        rA[step * 2 + 1] = __float22half2_rn(make_float2(v.z, v.w));
      }
      #pragma unroll
      for (int step = 0; step < steps_b; ++step) {
        int idx = step * BLOCK_SIZE + tid;
        int n = idx / num_cols_vec_b;
        int kk = (idx % num_cols_vec_b) * 4;
        float4 v = *reinterpret_cast<const float4*>(&d_b[(offset_b_n + n) * dim_k + next + kk]);
        rB[step * 2 + 0] = __float22half2_rn(make_float2(v.x, v.y));
        rB[step * 2 + 1] = __float22half2_rn(make_float2(v.z, v.w));
      }
    }

    // ---- 2) Compute on the current (read_buf) tile ----
    half* A_buf = smem_A + read_buf * sizeA;
    half* B_buf = smem_B + read_buf * sizeB;
    #pragma unroll
    for (int k_step = 0; k_step < BLOCK_TILE_K / 16; ++k_step) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[WARP_TILE_N / 16];

      #pragma unroll
      for (int c = 0; c < WARP_TILE_N / 16; ++c) {
        int col_idx = warp_col * WARP_TILE_N + c * 16;   // n start
        wmma::load_matrix_sync(b_frag[c],
            &B_buf[col_idx * (BLOCK_TILE_K + SMEM_PADDING) + k_step * 16],
            BLOCK_TILE_K + SMEM_PADDING);
      }
      #pragma unroll
      for (int r = 0; r < WARP_TILE_M / 16; ++r) {
        int row_idx = warp_row * WARP_TILE_M + r * 16;   // m start
        wmma::load_matrix_sync(a_frag,
            &A_buf[k_step * 16 * (BLOCK_TILE_M + SMEM_PADDING) + row_idx],
            BLOCK_TILE_M + SMEM_PADDING);
        #pragma unroll
        for (int c = 0; c < WARP_TILE_N / 16; ++c)
          wmma::mma_sync(acc[r][c], a_frag, b_frag[c], acc[r][c]);
      }
    }

    // ---- 3) Commit the prefetched registers into the OTHER (write_buf) buffer ----
    if (has_next) {
      half* A_wb = smem_A + write_buf * sizeA;
      half* B_wb = smem_B + write_buf * sizeB;
      #pragma unroll
      for (int step = 0; step < steps_a; ++step) {
        int idx = step * BLOCK_SIZE + tid;
        int j = idx / num_cols_vec_a;
        int m = (idx % num_cols_vec_a) * 4;
        *reinterpret_cast<half2*>(&A_wb[j * (BLOCK_TILE_M + SMEM_PADDING) + m + 0]) = rA[step * 2 + 0];
        *reinterpret_cast<half2*>(&A_wb[j * (BLOCK_TILE_M + SMEM_PADDING) + m + 2]) = rA[step * 2 + 1];
      }
      #pragma unroll
      for (int step = 0; step < steps_b; ++step) {
        int idx = step * BLOCK_SIZE + tid;
        int n = idx / num_cols_vec_b;
        int kk = (idx % num_cols_vec_b) * 4;
        *reinterpret_cast<half2*>(&B_wb[n * (BLOCK_TILE_K + SMEM_PADDING) + kk + 0]) = rB[step * 2 + 0];
        *reinterpret_cast<half2*>(&B_wb[n * (BLOCK_TILE_K + SMEM_PADDING) + kk + 2]) = rB[step * 2 + 1];
      }
      __syncthreads();
      read_buf = write_buf;
    }
  }

  // ---- Write back to C (column-major, ldc = dim_m) ----
  #pragma unroll
  for (int r = 0; r < WARP_TILE_M / 16; r++) {
    #pragma unroll
    for (int c = 0; c < WARP_TILE_N / 16; c++) {
      int c_m = offset_a_m + warp_row * WARP_TILE_M + r * 16;
      int c_n = offset_b_n + warp_col * WARP_TILE_N + c * 16;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
		 A, CUDA_R_32F, m,
		 B, CUDA_R_32F, k,
		 &beta,
		 C, CUDA_R_32F, m,
		 CUBLAS_COMPUTE_32F_FAST_16F,
		 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  dim3 block = dim3(BLOCK_SIZE);
  dim3 grid = dim3((m+BLOCK_TILE_M-1)/BLOCK_TILE_M, (n+BLOCK_TILE_N-1)/BLOCK_TILE_N);
  int smem_size = (2 * BLOCK_TILE_K * (BLOCK_TILE_M + SMEM_PADDING) +
                   2 * BLOCK_TILE_N * (BLOCK_TILE_K + SMEM_PADDING)) * sizeof(half);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block, smem_size >>>(m,
			      n,
			      k,
			      A,
			      B,
			      C2);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
      return 1;
    }
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}

#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

// Default configurations (highly optimized values for H100)
#ifndef BLOCK_TILE_M
#define BLOCK_TILE_M 128
#endif

#ifndef BLOCK_TILE_N
#define BLOCK_TILE_N 128
#endif

#ifndef BLOCK_TILE_K
#define BLOCK_TILE_K 64
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

__global__ __launch_bounds__(BLOCK_SIZE)
void kernel(int dim_m, int dim_n, int dim_k, const float *d_a, const float *d_b, float *d_c) {
  int tid = threadIdx.x;
  int warp_id = threadIdx.x / 32;

  extern __shared__ half smem[];
  int size_a = BLOCK_TILE_K * (BLOCK_TILE_M + SMEM_PADDING);
  
  half* smem_A = smem;
  half* smem_B = smem + size_a;

  // A warp computes a WARP_TILE_M x WARP_TILE_N sub-block of C (which is wmma tiles).
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WARP_TILE_M / 16][WARP_TILE_N / 16];
  #pragma unroll
  for (int r = 0; r < WARP_TILE_M / 16; r++)
    for (int c = 0; c < WARP_TILE_N / 16; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  const int warps_per_col = BLOCK_TILE_N / WARP_TILE_N;
  int warp_row = warp_id / warps_per_col;
  int warp_col = warp_id % warps_per_col;

  // Global load step constants
  const int steps_a = (BLOCK_TILE_K * BLOCK_TILE_M) / (4 * BLOCK_SIZE);
  const int num_cols_vec_a = BLOCK_TILE_M / 4;
  const int steps_b = (BLOCK_TILE_K * BLOCK_TILE_N) / (4 * BLOCK_SIZE);
  const int num_cols_vec_b = BLOCK_TILE_K / 4;

  int offset_a_m = BLOCK_TILE_M * blockIdx.x;
  int offset_b_n = BLOCK_TILE_N * blockIdx.y;

  for (int k = 0; k < dim_k; k += BLOCK_TILE_K) {
    // Load A tile into SMEM using vectorized half2 conversion
    #pragma unroll
    for (int step = 0; step < steps_a; ++step) {
      int idx = step * BLOCK_SIZE + tid;
      int j = idx / num_cols_vec_a;
      int m = (idx % num_cols_vec_a) * 4;
      float4 val = *reinterpret_cast<const float4*>(&d_a[(k + j) * dim_m + offset_a_m + m]);
      half2 h_xy = __float22half2_rn(make_float2(val.x, val.y));
      half2 h_zw = __float22half2_rn(make_float2(val.z, val.w));
      *reinterpret_cast<half2*>(&smem_A[j * (BLOCK_TILE_M + SMEM_PADDING) + m + 0]) = h_xy;
      *reinterpret_cast<half2*>(&smem_A[j * (BLOCK_TILE_M + SMEM_PADDING) + m + 2]) = h_zw;
    }

    // Load B tile into SMEM using vectorized half2 conversion
    #pragma unroll
    for (int step = 0; step < steps_b; ++step) {
      int idx = step * BLOCK_SIZE + tid;
      int n = idx / num_cols_vec_b;
      int j = (idx % num_cols_vec_b) * 4;
      float4 val = *reinterpret_cast<const float4*>(&d_b[(offset_b_n + n) * dim_k + k + j]);
      
      // B is loaded such that columns of the block map to rows in memory? 
      // wait, d_b is col-major or row-major?
      // B is col-major. d_b[n * dim_k + k + j]. We read 4 contiguous elements along K dimension.
      // We store it into smem_B such that we can use wmma::row_major load later?
      // Wait, in previous code:
      // smem_B[(j + 0) * (BLOCK_TILE_N + SMEM_PADDING) + n] = val.x;
      // This means we CANNOT use vectorized half2 store because the memory addresses are NOT contiguous!
      // (j+0), (j+1) are different rows in SMEM!
      // So we must store them separately.
      smem_B[(j + 0) * (BLOCK_TILE_N + SMEM_PADDING) + n] = __float2half(val.x);
      smem_B[(j + 1) * (BLOCK_TILE_N + SMEM_PADDING) + n] = __float2half(val.y);
      smem_B[(j + 2) * (BLOCK_TILE_N + SMEM_PADDING) + n] = __float2half(val.z);
      smem_B[(j + 3) * (BLOCK_TILE_N + SMEM_PADDING) + n] = __float2half(val.w);
    }

    __syncthreads();

    #pragma unroll
    for (int k_step = 0; k_step < BLOCK_TILE_K / 16; ++k_step) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WARP_TILE_N / 16];

      // Load all B fragments for this k_step
      #pragma unroll
      for (int c = 0; c < WARP_TILE_N / 16; ++c) {
        int col_idx = warp_col * WARP_TILE_N + c * 16;
        wmma::load_matrix_sync(b_frag[c], &smem_B[k_step * 16 * (BLOCK_TILE_N + SMEM_PADDING) + col_idx], BLOCK_TILE_N + SMEM_PADDING);
      }

      // Load one A fragment at a time and multiply with all B fragments
      #pragma unroll
      for (int r = 0; r < WARP_TILE_M / 16; ++r) {
        int row_idx = warp_row * WARP_TILE_M + r * 16;
        wmma::load_matrix_sync(a_frag, &smem_A[k_step * 16 * (BLOCK_TILE_M + SMEM_PADDING) + row_idx], BLOCK_TILE_M + SMEM_PADDING);
        
        #pragma unroll
        for (int c = 0; c < WARP_TILE_N / 16; ++c) {
          wmma::mma_sync(acc[r][c], a_frag, b_frag[c], acc[r][c]);
        }
      }
    }
    __syncthreads();
  }

  // Write back to global memory C
  #pragma unroll
  for (int r = 0; r < WARP_TILE_M / 16; r++) {
    #pragma unroll
    for (int c = 0; c < WARP_TILE_N / 16; c++) {
      int c_m = offset_a_m + warp_row * WARP_TILE_M + r * 16;
      int c_n = offset_b_n + warp_col * WARP_TILE_N + c * 16;
      if (c_n < dim_n && c_m < dim_m) {
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
      }
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
  int smem_size = (BLOCK_TILE_K * (BLOCK_TILE_M + SMEM_PADDING) + 
                   BLOCK_TILE_K * (BLOCK_TILE_N + SMEM_PADDING)) * sizeof(half);
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

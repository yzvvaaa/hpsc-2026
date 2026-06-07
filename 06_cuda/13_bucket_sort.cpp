#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void fillBucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void scanBucket(int *bucket, int *offset, int *tmp, int range) {
  int i = threadIdx.x;
  offset[i] = bucket[i];
  for (int j = 1; j < range; j <<= 1) {
    tmp[i] = offset[i];
    __syncthreads();
    if (i >= j) offset[i] += tmp[i-j];
    __syncthreads();
  }
  offset[i] -= bucket[i];
}

__global__ void fillKey(int *key, int *bucket, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= range) return;
  int j = offset[i];
  for (; bucket[i] > 0; bucket[i]--) {
    key[j++] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket, *offset, *tmp;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&offset, range * sizeof(int));
  cudaMallocManaged(&tmp, range * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }

  int m = 32; // block size
  fillBucket<<<(n + m - 1) / m, m>>>(key, bucket, n);
  scanBucket<<<1, range>>>(bucket, offset, tmp, range);
  fillKey<<<(range + m - 1) / m, m>>>(key, bucket, offset, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(offset);
  cudaFree(tmp);
}

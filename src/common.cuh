#pragma once

#include <cuda.h>
#include <helper_cuda.h>

/* General kernels */
__host__ __device__ float normpdf(float x) {
  return exp(-0.5f * x * x) * (1.0f / sqrt(2.0f * M_PI));
}

/* Device contants */
__constant__ int   N;
__constant__ float T, r, sigma, dt, omega, s0, k;

/* Struct to be passed to the MC method to store calculated values */
template <class T>
struct mc_results {
  T *price, *delta, *vega, *gamma;

  __host__
    void AllocateHost(const int size) {
      price = (T *) malloc(sizeof(T) * size);
      delta = (T *) malloc(sizeof(T) * size);
      vega = (T *) malloc(sizeof(T) * size);
      gamma = (T *) malloc(sizeof(T) * size);
    }

  __host__
    void AllocateDevice(const int size) {
      checkCudaErrors(cudaMalloc((void **) &price, sizeof(T) * size));
      checkCudaErrors(cudaMalloc((void **) &delta, sizeof(T) * size));
      checkCudaErrors(cudaMalloc((void **) &vega, sizeof(T) * size));
      checkCudaErrors(cudaMalloc((void **) &gamma, sizeof(T) * size));
    }

  __host__
    void CopyFromDevice(const int size, const mc_results<T> &d_results) {
      checkCudaErrors( cudaMemcpy(price, d_results.price,
            sizeof(T) * size, cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(delta, d_results.delta,
            sizeof(T) * size, cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(vega, d_results.vega,
            sizeof(T) * size, cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(gamma, d_results.gamma,
            sizeof(T) * size, cudaMemcpyDeviceToHost) );
    }

  __host__
    void ReleaseHost() {
      free(price);
      free(delta);
      free(vega);
      free(gamma);
    }

  __host__
    void ReleaseDevice() {
      checkCudaErrors(cudaFree(price));
      checkCudaErrors(cudaFree(delta));
      checkCudaErrors(cudaFree(vega));
      checkCudaErrors(cudaFree(gamma));
    }
};

#pragma once

#include <cuda.h>
#include <helper_cuda.h>

#include "common.cuh"

namespace demeter {

  /* General kernels */
  __host__ __device__ float NormPDF(float x) {
    return exp(-0.5f * x * x) * (1.0f / sqrt(2.0f * M_PI));
  }

  __host__ float NormCDF(float x) {
    return std::erfc(-x/std::sqrt(2))/2;
  }

  /* Main Monte Carlo simulation */
  template <class T>
  __global__ 
  void MCSimulation(float *d_z, MCResults<float> d_results) {
    T prod;
    // Index into random variables
    prod.ind = threadIdx.x + N*blockIdx.x*blockDim.x;

    prod.SimulatePath(N, d_z);
    prod.CalculatePayoffs(d_results);
  }

} // namespace demeter

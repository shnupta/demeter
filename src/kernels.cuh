#pragma once

#include <cuda.h>
#include <helper_cuda.h>
#include <cudaProfiler.h>

#include "common.cuh"

namespace demeter {

  /* General kernels */
  __host__ __device__ float NormPDF(float x) {
    return exp(-0.5f * x * x) * (1.0f / sqrt(2.0f * M_PI));
  }

  __host__ float NormCDF(float x) {
    return std::erfc(-x/std::sqrt(2))/2;
  }

  __global__ void TransformSobol(float *d_z, float *temp_z) {
    int desired_idx = threadIdx.x + N * blockIdx.x * blockDim.x;
    int temp_idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = 0; n < N; n++) {
      d_z[desired_idx] = temp_z[temp_idx];
      desired_idx += blockDim.x;
      temp_idx += PATHS;
    }
  }

  /* Main Monte Carlo simulation */
  template <class T>
  __global__ 
  void MCSimulation(float *d_z, float *d_path, MCResults<double> d_results,
      MCMode mode) {
    T prod;
    // Index into random variables
    prod.ind = threadIdx.x + N*blockIdx.x*blockDim.x;

    if (mode == MCMode::QUASI_BB)
      prod.SimulatePathQuasiBB(N, d_z, d_path);
    else 
      prod.SimulatePath(N, d_z); 

    if (mode == MCMode::STANDARD_AV)
      prod.CalculatePayoffs(d_results, true);
    else 
      prod.CalculatePayoffs(d_results, false);
  }

} // namespace demeter

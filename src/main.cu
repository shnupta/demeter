////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iostream>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

#include "common.cuh"
#include "kernels.cuh"
#include "product.cuh"
#include "timer.cuh"

using namespace demeter;

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

template <typename S>
void RunAndCompareMC(int npath, int timesteps, float h_T, float h_dt, float h_r,
    float h_sigma, float h_omega, float h_s0, float h_k, MCMode mode,
    LRResults<double>& lr_results) {

  if (mode == MCMode::QUASI) {
    printf("Mode = QUASI\n");
  } else if (mode == MCMode::STANDARD) {
    printf("Mode = STANDARD\n");
  } else if(mode == MCMode::QUASI_BB) {
    printf("Mode = QUASI_BB\n");
  } else if(mode == MCMode::STANDARD_AV) {
    printf("STANDARD_AV\n");
  }

  // Initalise host product and print name
  S h_prod;
  h_prod.PrintName();

  float *d_z, *d_temp_z;
  float *d_path;
  MCResults<double> h_results, d_results;
  Timer timer;

  // Copy values to GPU constants
  checkCudaErrors(cudaMemcpyToSymbol(N, &timesteps, sizeof(timesteps)));
  checkCudaErrors(cudaMemcpyToSymbol(PATHS ,&npath, sizeof(npath)));
  checkCudaErrors(cudaMemcpyToSymbol(T, &h_T, sizeof(h_T)));
  checkCudaErrors(cudaMemcpyToSymbol(r, &h_r, sizeof(h_r)));
  checkCudaErrors(cudaMemcpyToSymbol(sigma, &h_sigma, sizeof(h_sigma)));
  checkCudaErrors(cudaMemcpyToSymbol(dt, &h_dt, sizeof(h_dt)));
  checkCudaErrors(cudaMemcpyToSymbol(omega, &h_omega, sizeof(h_omega)));
  checkCudaErrors(cudaMemcpyToSymbol(s0, &h_s0, sizeof(h_s0)));
  checkCudaErrors(cudaMemcpyToSymbol(k, &h_k, sizeof(h_k)));

  // Allocate host and device results
  h_results.AllocateHost(npath);
  d_results.AllocateDevice(npath);

  // Allocate memory for random variables
  checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(float) * timesteps * npath));
  if (mode == MCMode::QUASI || mode == MCMode::QUASI_BB) {
    checkCudaErrors(cudaMalloc((void **)&d_temp_z,
          sizeof(float) * timesteps * npath));
  }
  if (mode == MCMode::QUASI_BB) {
    checkCudaErrors(cudaMalloc((void **) &d_path,
          sizeof(float) * timesteps * npath));
  }

  timer.StartDeviceTimer();

  curandGenerator_t gen;
  if (mode == MCMode::STANDARD || mode == MCMode::STANDARD_AV) {
    checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
    /* checkCudaErrors( curandGenerateNormal(gen, d_z, timesteps * npath, 0.0f, 1.0f) ); */
  } else if (mode == MCMode::QUASI || mode == MCMode::QUASI_BB) {
    checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
    checkCudaErrors(curandSetQuasiRandomGeneratorDimensions(gen, timesteps));
    /* checkCudaErrors(curandGenerateNormal(gen, d_temp_z, timesteps*npath, 0.0f, 1.0f)); */
  }

  timer.StopDeviceTimer();

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n\n",
          timer.GetDeviceElapsedTime(), timesteps*npath/(0.001*timer.GetDeviceElapsedTime()));

  // Execute kernel and time it
  const int runs = 500;
  MCResults<double> final_results;
  final_results.AllocateHost(runs);

  // Perform M independent runs of the simulation to get variance
  for (int j = 0; j < runs; ++j) {
    // Transform ordering of random variables into one that maximises memory
    // locality when threads access for path simulation
    if (mode == MCMode::QUASI || mode == MCMode::QUASI_BB) {
      checkCudaErrors(curandGenerateNormal(gen, d_temp_z, timesteps*npath, 0.0f, 1.0f));
      TransformSobol<<<npath/64, 64>>>(d_z, d_temp_z);
    } else {
      checkCudaErrors( curandGenerateNormal(gen, d_z, timesteps * npath, 0.0f, 1.0f) );
    }

    timer.StartDeviceTimer();

    MCSimulation<S> <<<npath/64, 64>>>(d_z, d_path, d_results, mode);
    getLastCudaError("MCSimulation execution failed\n");

    timer.StopDeviceTimer();


    // Copy back results

    h_results.CopyFromDevice(npath, d_results);
    h_results.CalculateStatistics(npath);

    // Transfer averages to final results struct
    final_results.price[j] = h_results.avg_price;
    final_results.delta[j] = h_results.avg_delta;
    final_results.vega[j] = h_results.avg_vega;
    final_results.gamma[j] = h_results.avg_gamma;
    final_results.lr_delta[j] = h_results.avg_lr_delta;
    final_results.lr_vega[j] = h_results.avg_lr_vega;
    final_results.lr_gamma[j] = h_results.avg_lr_gamma;
    /* h_results.PrintStatistics(true ,"GPU"); */
  }

  final_results.CalculateStatistics(runs);

  // Grab the LR results to calculate later VRFs
  if (mode == MCMode::STANDARD) {
    lr_results.delta = final_results.avg_lr_delta;
    lr_results.vega = final_results.avg_lr_vega;
    lr_results.gamma = final_results.avg_lr_gamma;
    lr_results.err_delta = final_results.err_lr_delta;
    lr_results.err_vega = final_results.err_lr_vega;
    lr_results.err_gamma = final_results.err_lr_gamma;

    printf("\nLIKELIHOOD RATIO\n| %13s | %13s | %13s | %13s | %13s | %13s |\n| %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f |\n\n",
        "delta", "err", "vega", "err", "gamma", "err", lr_results.delta,
        lr_results.err_delta, lr_results.vega, lr_results.err_vega,
        lr_results.gamma, lr_results.err_gamma);
  } 

  final_results.PrintStatistics(true, "GPU");

  printf("\nVRF for delta = %13.8f\n",
      (lr_results.err_delta * lr_results.err_delta)
      / (final_results.err_delta * final_results.err_delta));
  printf("\nVRF for vega = %13.8f\n",
      (lr_results.err_vega * lr_results.err_vega)
      / (final_results.err_vega * final_results.err_vega));
  printf("\nVRF for gamma = %13.8f\n",
      (lr_results.err_gamma * lr_results.err_gamma)
      / (final_results.err_gamma * final_results.err_gamma));

  // CPU calculation

  // Copy random variables
  float *h_z = (float *) malloc(sizeof(float) * timesteps * npath);
  float *h_temp_z = (float *) malloc(sizeof(float) * timesteps * npath);
  if (mode == MCMode::QUASI || mode == MCMode::QUASI_BB) {
    checkCudaErrors( cudaMemcpy(h_temp_z, d_temp_z, sizeof(float) * timesteps * npath, cudaMemcpyDeviceToHost) );

    // Rejig for sobol dimensions
    int i = 0, j = 0;
    while (i < npath) {
      while (j < timesteps) {
        h_z[i * timesteps + j] = h_temp_z[i + j * npath];
        j++;
      }
      i++;
      j = 0;
    }
  } else if (mode == MCMode::STANDARD || mode == MCMode::STANDARD_AV) {
    checkCudaErrors( cudaMemcpy(h_z, d_z, sizeof(float) * timesteps * npath, cudaMemcpyDeviceToHost) );
  }


  timer.StartHostTimer();
  h_prod.HostMC(npath, timesteps, h_z, h_r, h_dt, h_sigma, h_s0, h_k, h_T,
      h_omega, h_results);
  timer.StopHostTimer();

  h_results.CalculateStatistics(npath);
  /* h_results.PrintStatistics(false, "CPU"); */

  printf("\nGPU execution time (ms): %f \n", timer.GetDeviceElapsedTime());
  printf("CPU execution time (ms): %f \n", timer.GetHostElapsedTime());
  printf("Speedup factor: %fx\n", timer.GetSpeedUpFactor());
  printf("----------------------------------------------------------------------------------------------------------------------------------------\n");

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  h_results.ReleaseHost();
  d_results.ReleaseDevice();
  final_results.ReleaseHost();
  free(h_z);
  free(h_temp_z);
  checkCudaErrors( cudaFree(d_z) );
  if (mode == MCMode::QUASI || mode == MCMode::QUASI_BB) {
    checkCudaErrors( cudaFree(d_temp_z) );
  }
  if (mode == MCMode::QUASI_BB) {
    checkCudaErrors( cudaFree(d_path) );
  }
}

int main(int argc, const char **argv){
  cudaDeviceProp prop;

  // initialise card
  findCudaDevice(argc, argv);
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "   --- General Information for device ---\n";
  std::cout << "Name: " << prop.name << std::endl;
  std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "Clock rate: " << prop.clockRate << std::endl;
  std::cout << "Device copy overlap: " << (prop.deviceOverlap ? "Enabled\n" : "Disabled\n");
  std::cout << "Kernel execution timeout: " << (prop.kernelExecTimeoutEnabled ? "Enabled\n" : "Disabled\n");

  std::cout << "   --- Memory Information for device ---\n";
  std::cout << "Total glob mem: " << prop.totalGlobalMem << std::endl;
  std::cout << "Total constant mem: " << prop.totalConstMem << std::endl;
  std::cout << "Max mem pitch: " << prop.memPitch << std::endl;
  std::cout << "Texture alignment: " << prop.textureAlignment << std::endl;

  std::cout << "   --- MP Information for device ---\n";
  std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
  std::cout << "Shared mem per mp: " << prop.sharedMemPerBlock << std::endl;
  std::cout << "Registers per mp: " << prop.regsPerBlock << std::endl;
  std::cout << "Threads in ward: " << prop.warpSize << std::endl;
  std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "Max thread dimensions: (" << prop.maxThreadsDim[0] << ", " 
      << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
  std::cout << "Max grid dimensions: (" << prop.maxGridSize[0] << ", " 
      << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";

  std::cout << std::endl;
    
  int NPATH = (1 << 15);
  int h_m = 6;

  while (h_m <= 8) {
    int h_N = 1 << h_m;
    float h_T, h_r, h_sigma, h_dt, h_omega, h_s0, h_k;

    LRResults<double> lr_results;

    printf("NPATH = %13d    h_N = %13d\n\n", NPATH, h_N);


    h_T     = 1.0f;
    h_r     = 0.1f;
    h_sigma = 0.2f;
    h_dt    = h_T/h_N;
    h_omega = h_r - (h_sigma * h_sigma) / 2.0f;
    h_s0      = 100.0f;
    h_k       = 90.0f;

    // AA
    RunAndCompareMC<ArithmeticAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::STANDARD, lr_results);
    RunAndCompareMC<ArithmeticAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::STANDARD_AV, lr_results);
    RunAndCompareMC<ArithmeticAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::QUASI, lr_results);
    RunAndCompareMC<ArithmeticAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::QUASI_BB, lr_results);
    printf("\n\n\n");
    // BA
    RunAndCompareMC<BinaryAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::STANDARD, lr_results);
    RunAndCompareMC<BinaryAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::STANDARD_AV, lr_results);
    RunAndCompareMC<BinaryAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::QUASI, lr_results);
    RunAndCompareMC<BinaryAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::QUASI_BB, lr_results);
    printf("\n\n\n");
    // L
    RunAndCompareMC<Lookback<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::STANDARD, lr_results);
    RunAndCompareMC<Lookback<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::STANDARD_AV, lr_results);
    RunAndCompareMC<Lookback<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::QUASI, lr_results);
    RunAndCompareMC<Lookback<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::QUASI_BB, lr_results);
    printf("\n\n\n");


    /* NPATH <<= 1; */
    h_m += 2; // Jump to 256 timesteps

  }

  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();

}


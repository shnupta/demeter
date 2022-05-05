////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

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

enum MCMode {
  STANDARD, QUASI
};

template <typename S>
void RunAndCompareMC(int npath, int timesteps, float h_T, float h_dt, float h_r,
    float h_sigma, float h_omega, float h_s0, float h_k, MCMode mode) {

  if (mode == MCMode::QUASI) {
    printf("Mode = QUASI\n");
  } else if (mode == MCMode::STANDARD) {
    printf("Mode = STANDARD\n");
  }

  // Initalise host product and print name
  S h_prod;
  h_prod.PrintName();

  float *d_z, *d_temp_z, *d_paths;
  MCResults<float> h_results, d_results;
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
  checkCudaErrors(cudaMalloc((void **)&d_temp_z,
        sizeof(float) * timesteps * npath));
  checkCudaErrors(cudaMalloc((void **) &d_paths,
      sizeof(float) * timesteps * npath));

  timer.StartDeviceTimer();

  curandGenerator_t gen;
  if (mode == MCMode::STANDARD) {
    checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
    checkCudaErrors( curandGenerateNormal(gen, d_z, timesteps * npath, 0.0f, 1.0f) );
  } else if (mode == MCMode::QUASI) {
    checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));
    checkCudaErrors(curandSetQuasiRandomGeneratorDimensions(gen, timesteps));
    checkCudaErrors(curandGenerateNormal(gen, d_temp_z, timesteps*npath, 0.0f, 1.0f));
    // Transform ordering of random variables into one that maximises memory
    // locality when threads access for path simulation
    TransformSobol<<<npath/64, 64>>>(d_z, d_temp_z);
  }

  timer.StopDeviceTimer();

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n\n",
          timer.GetDeviceElapsedTime(), timesteps*npath/(0.001*timer.GetDeviceElapsedTime()));

  // Execute kernel and time it

  timer.StartDeviceTimer();

  MCSimulation<S> <<<npath/64, 64>>>(d_z, d_paths, d_results);
  getLastCudaError("MCSimulation execution failed\n");

  timer.StopDeviceTimer();


  // Copy back results

  h_results.CopyFromDevice(npath, d_results);
  h_results.CalculateStatistics(npath);
  h_results.PrintStatistics(true ,"GPU");

  // CPU calculation

  // Copy random variables
  float *h_z = (float *) malloc(sizeof(float) * timesteps * npath);
  float *h_temp_z = (float *) malloc(sizeof(float) * timesteps * npath);
  if (mode == MCMode::QUASI) {
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
  } else if (mode == MCMode::STANDARD) {
    checkCudaErrors( cudaMemcpy(h_z, d_z, sizeof(float) * timesteps * npath, cudaMemcpyDeviceToHost) );
  }


  timer.StartHostTimer();
  h_prod.HostMC(npath, timesteps, h_z, h_r, h_dt, h_sigma, h_s0, h_k, h_T,
      h_omega, h_results);
  timer.StopHostTimer();

  h_results.CalculateStatistics(npath);
  h_results.PrintStatistics(false, "CPU");

  printf("\nGPU execution time (ms): %f \n", timer.GetDeviceElapsedTime());
  printf("CPU execution time (ms): %f \n", timer.GetHostElapsedTime());
  printf("Speedup factor: %fx\n", timer.GetSpeedUpFactor());
  printf("----------------------------------------------------------------------------------------------------------------------------------------\n");

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  h_results.ReleaseHost();
  d_results.ReleaseDevice();
  free(h_z);
  free(h_temp_z);
  checkCudaErrors( cudaFree(d_z) );
  checkCudaErrors( cudaFree(d_temp_z) );
}

int main(int argc, const char **argv){

  // initialise card
  findCudaDevice(argc, argv);
    
  /* int NPATH=960000, h_N=300; */
  int NPATH=960000;// / 16;
  int h_N=256; // 2^8
  /* int h_N=512; // 2^9 */
  /* int h_N=16; // 2^4 */
  // TODO: Try with 16 and print first

  while (NPATH <= 960000) {

    float h_T, h_r, h_sigma, h_dt, h_omega, h_s0, h_k;

    printf("NPATH = %13d    h_N = %13d\n\n", NPATH, h_N);


    h_T     = 1.0f;
    h_r     = 0.1f;
    h_sigma = 0.2f;
    h_dt    = h_T/h_N;
    h_omega = h_r - (h_sigma * h_sigma) / 2.0f;
    h_s0      = 100.0f;
    h_k       = 100.0f;

    RunAndCompareMC<ArithmeticAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma,
        h_omega, h_s0, h_k, MCMode::QUASI);
    /* RunAndCompareMC<BinaryAsian<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma, */
    /*     h_omega, h_s0, h_k, MCMode::QUASI); */
    /* RunAndCompareMC<Lookback<float>>(NPATH, h_N, h_T, h_dt, h_r, h_sigma, */
    /*     h_omega, h_s0, h_k, MCMode::QUASI); */

    printf("\n\n\n");

    NPATH <<= 1;

  }

  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();

}


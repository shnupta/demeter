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


__global__ void TransformSobol(float *d_z, float *temp_z) {
  int desired_idx = threadIdx.x + N * blockIdx.x * blockDim.x;
  int temp_idx = threadIdx.x + blockIdx.x * blockDim.x;

  // TODO: Check for some other desired indices that it transfers correctly
  bool print = false; // desired_idx == N * 1 * blockDim.x;

  for (int n = 0; n < N; n++) {
    if (print) printf("Moving from index %d (dim %d) to %d (block %d dim %d)\n",
        temp_idx, temp_idx / PATHS, desired_idx, desired_idx / (N * blockDim.x), (desired_idx / blockDim.x) % N);

    d_z[desired_idx] = temp_z[temp_idx];
    desired_idx += blockDim.x;
    temp_idx += PATHS;
  }
}

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
    
  int NPATH=960000, h_N=300;
  /* int     NPATH=64, h_N=1; */
  float h_T, h_r, h_sigma, h_dt, h_omega, h_s0, h_k;
  float *d_z, *temp_z;
  MCResults<float> h_results, d_results;
  Timer timer;

  // initialise card

  findCudaDevice(argc, argv);

  // allocate memory on host and device

  h_results.AllocateHost(NPATH);
  d_results.AllocateDevice(NPATH);

  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*h_N*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&temp_z, sizeof(float)*h_N*NPATH) );

  // define constants and transfer to GPU

  h_T     = 1.0f;
  h_r     = 0.1f;
  h_sigma = 0.2f;
  h_dt    = h_T/h_N;
  h_omega = h_r - (h_sigma * h_sigma) / 2.0f;
  h_s0      = 100.0f;
  h_k       = 100.0f;

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(PATHS,    &NPATH,    sizeof(NPATH)) );
  checkCudaErrors( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );
  checkCudaErrors( cudaMemcpyToSymbol(omega,   &h_omega,   sizeof(h_omega)) );
  checkCudaErrors( cudaMemcpyToSymbol(s0,   &h_s0,   sizeof(h_s0)) );
  checkCudaErrors( cudaMemcpyToSymbol(k,   &h_k,   sizeof(h_k)) );

  // random number generation

  timer.StartDeviceTimer();

  curandGenerator_t gen;
  /* checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) ); */
  /* checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) ); */
  /* checkCudaErrors( curandGenerateNormal(gen, d_z, h_N*NPATH, 0.0f, 1.0f) ); */
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_QUASI_DEFAULT) );
  checkCudaErrors( curandSetQuasiRandomGeneratorDimensions(gen, h_N) );
  checkCudaErrors( curandGenerateNormal(gen, temp_z, h_N*NPATH, 0.0f, 1.0f) );

  timer.StopDeviceTimer();

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          timer.GetDeviceElapsedTime(), h_N*NPATH/(0.001*timer.GetDeviceElapsedTime()));

  TransformSobol<<<NPATH/64, 64>>>(d_z, temp_z);

  // execute kernel and time it

  printf("\n====== GPU ======\n");
  timer.StartDeviceTimer();

  MCSimulation< ArithmeticAsian<float> > <<<NPATH/64, 64>>>(d_z, d_results);
  getLastCudaError("pathcalc execution failed\n");

  timer.StopDeviceTimer();

  printf("Monte Carlo kernel execution time (ms): %f \n", timer.GetDeviceElapsedTime());

  // copy back results and compute stats

  h_results.CopyFromDevice(NPATH, d_results);
  h_results.CalculateStatistics(NPATH);
  h_results.PrintStatistics();

  // CPU calculation
  printf("\n====== CPU ======\n");

  // Copy random variables
  float *h_z = (float *) malloc(sizeof(float) * h_N * NPATH);
  float *h_temp = (float *) malloc(sizeof(float) * h_N * NPATH);
  checkCudaErrors( cudaMemcpy(h_temp, temp_z, sizeof(float) * h_N * NPATH, cudaMemcpyDeviceToHost) );

  // Rejig for sobol dimensions
  int i = 0, j = 0;
  while (i < NPATH) {
    while (j < h_N) {
      h_z[i * h_N + j] = h_temp[i + j * NPATH];
      j++;
    }
    i++;
    j = 0;
  }


  ArithmeticAsian<float> prod;
  timer.StartHostTimer();
  prod.HostMC(NPATH, h_N, h_z, h_r, h_dt, h_sigma, h_s0, h_k, h_T, h_omega, h_results);
  timer.StopHostTimer();
  printf("CPU execution time (ms): %f \n", timer.GetHostElapsedTime());

  h_results.CalculateStatistics(NPATH);
  h_results.PrintStatistics();

  printf("\nGPU speedup over serial CPU: %fx\n", timer.GetSpeedUpFactor());

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  h_results.ReleaseHost();
  d_results.ReleaseDevice();
  free(h_z);
  checkCudaErrors( cudaFree(d_z) );
  checkCudaErrors( cudaFree(temp_z) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}


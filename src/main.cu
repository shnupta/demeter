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
#include "product.cuh"

////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////

template <class T>
__global__ 
void pathcalc(float *d_z, mc_results<float> d_results) {
  T prod;
  // Index into random variables
  prod.ind = threadIdx.x + N*blockIdx.x*blockDim.x;

  prod.SimulatePath(N, d_z);
  prod.CalculatePayoffs(d_results);
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
    
  int NPATH=960000, h_N=100;
  /* int     NPATH=64, h_N=1; */
  float h_T, h_r, h_sigma, h_dt, h_omega, h_s0, h_k;
  float *d_z;
  mc_results<float> h_results, d_results;
  double sum1, sum2, deltasum, vegasum, gammasum;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_results.AllocateHost(NPATH);
  d_results.AllocateDevice(NPATH);

  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*h_N*NPATH) );

  // define constants and transfer to GPU

  h_T     = 1.0f;
  h_r     = 0.1f;
  h_sigma = 0.2f;
  h_dt    = h_T/h_N;
  h_omega = h_r - (h_sigma * h_sigma) / 2.0f;
  h_s0      = 100.0f;
  h_k       = 100.0f;

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );
  checkCudaErrors( cudaMemcpyToSymbol(omega,   &h_omega,   sizeof(h_omega)) );
  checkCudaErrors( cudaMemcpyToSymbol(s0,   &h_s0,   sizeof(h_s0)) );
  checkCudaErrors( cudaMemcpyToSymbol(k,   &h_k,   sizeof(h_k)) );

  // random number generation

  cudaEventRecord(start);

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
  checkCudaErrors( curandGenerateNormal(gen, d_z, h_N*NPATH, 0.0f, 1.0f) );
 
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, h_N*NPATH/(0.001*milli));

  // execute kernel and time it

  printf("\n====== GPU ======\n");
  cudaEventRecord(start);

  pathcalc< arithmetic_asian<float> > <<<NPATH/64, 64>>>(d_z, d_results);
  getLastCudaError("pathcalc execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Monte Carlo kernel execution time (ms): %f \n",milli);

  // copy back results

  h_results.CopyFromDevice(NPATH, d_results);

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  deltasum = 0.0;
  gammasum = 0.0;
  for (int i=0; i<NPATH; i++) {
    sum1 += h_results.price[i];
    sum2 += h_results.price[i]*h_results.price[i];
    deltasum += h_results.delta[i];
    gammasum += h_results.gamma[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	 sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );
  printf("Average delta = %13.8f\n\n", deltasum / NPATH);
  printf("Average gamma = %13.8f\n\n", gammasum / NPATH);

  // CPU calculation
  printf("====== CPU ======\n");

  // Copy random variables
  float *h_z = (float *) malloc(sizeof(float) * h_N * NPATH);
  checkCudaErrors( cudaMemcpy(h_z, d_z, sizeof(float) * h_N * NPATH, cudaMemcpyDeviceToHost) );

  arithmetic_asian<float> asian;
  auto h_start = std::chrono::steady_clock::now();
  asian.HostMC(NPATH, h_N, h_z, h_r, h_dt, h_sigma, h_s0, h_k, h_T, h_omega, h_results);
  auto h_end = std::chrono::steady_clock::now();
  float h_milli = std::chrono::duration_cast<std::chrono::milliseconds>(h_end - h_start).count();
  printf("CPU execution time (ms): %f \n", h_milli);


  sum1 = 0.0;
  sum2 = 0.0;
  deltasum = 0.0;
  gammasum = 0.0;
  for (int i=0; i<NPATH; i++) {
    sum1 += h_results.price[i];
    sum2 += h_results.price[i]*h_results.price[i];
    deltasum += h_results.delta[i];
    gammasum += h_results.gamma[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	 sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );
  printf("Average delta = %13.8f\n\n", deltasum / NPATH);
  printf("Average gamma = %13.8f\n\n", gammasum / NPATH);


  printf("\nGPU speedup over serial CPU: %fx\n", h_milli / milli);

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  h_results.ReleaseHost();
  d_results.ReleaseDevice();
  free(h_z);

  checkCudaErrors( cudaFree(d_z) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}


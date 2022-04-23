////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float T, r, sigma, dt;

////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////


__global__ void pathcalc(float *d_z, float *d_v, float *d_delta)
{
  float s1, y1, payoff, avg_s1, delta;
  int   ind;

  // move array pointers to correct position

  // version 1
  ind = threadIdx.x + N*blockIdx.x*blockDim.x;

  // version 2
  // ind = 2*N*threadIdx.x + 2*N*blockIdx.x*blockDim.x;


  // path calculation

  s1 = 100.0f;
  avg_s1 = 100.0f;
  /* printf("Initial s1 = %f\n", s1); */

  for (int n=0; n<N; n++) {
    y1   = d_z[ind];
    // version 1
    ind += blockDim.x;      // shift pointer to next element

    s1 = s1 * (1.0f + r * dt + sigma * sqrt(dt) * y1);
    avg_s1 += s1;
    /* printf("New s1 = %f\n", s1); */
  }

  avg_s1 /= N;

  // put payoff value into device array

  /* payoff = avg_s1 - 100.0f > 0.0f ? exp(-r * T) : 0.0f; */ 
  payoff = exp(-r * T) * max(s1 - 100.0f, 0.0f);
  delta = s1 - 100.0f > 0.0f ? exp(-r * T) * (s1 / 100.0f) : 0.0f;
  /* printf("delta = %f\n", delta); */

  d_v[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
  d_delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
    
  int     NPATH=960000, h_N=100;
  /* int     NPATH=64, h_N=1; */
  float   h_T, h_r, h_sigma, h_dt;
  float  *h_v, *d_v, *d_z, *h_delta, *d_delta;
  double  sum1, sum2;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float)*NPATH);
  h_delta = (float *)malloc(sizeof(float)*NPATH);

  checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*h_N*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&d_delta, sizeof(float)*h_N*NPATH) );

  // define constants and transfer to GPU

  h_T     = 0.08333333;
  h_r     = 0.03f;
  h_sigma = 0.2f;
  h_dt    = h_T/h_N;

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  checkCudaErrors( cudaMemcpyToSymbol(T,    &h_T,    sizeof(h_T)) );
  checkCudaErrors( cudaMemcpyToSymbol(r,    &h_r,    sizeof(h_r)) );
  checkCudaErrors( cudaMemcpyToSymbol(sigma,&h_sigma,sizeof(h_sigma)) );
  checkCudaErrors( cudaMemcpyToSymbol(dt,   &h_dt,   sizeof(h_dt)) );

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

  cudaEventRecord(start);

  pathcalc<<<NPATH/64, 64>>>(d_z, d_v, d_delta);
  getLastCudaError("pathcalc execution failed\n");

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("Monte Carlo kernel execution time (ms): %f \n",milli);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
                   cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaMemcpy(h_delta, d_delta, sizeof(float)*NPATH,
                   cudaMemcpyDeviceToHost) );

  // compute average

  sum1 = 0.0;
  sum2 = 0.0;
  float deltasum = 0.0;
  for (int i=0; i<NPATH; i++) {
    sum1 += h_v[i];
    sum2 += h_v[i]*h_v[i];
    deltasum += h_delta[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
	 sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );
  printf("\nAverage delta = %13.8f\n\n", deltasum / NPATH);

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_v);
  free(h_delta);
  checkCudaErrors( cudaFree(d_v) );
  checkCudaErrors( cudaFree(d_z) );
  checkCudaErrors( cudaFree(d_delta) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}


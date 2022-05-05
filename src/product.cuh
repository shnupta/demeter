#pragma once

#include <cuda.h>
#include <helper_cuda.h>

#include "common.cuh"

namespace demeter {

  template <class T>
  struct Product {
    __device__ virtual void SimulatePath(const int N, float *d_z, float *d_path) = 0;
    __device__ virtual void CalculatePayoffs(MCResults<T> &d_results) = 0;
  };

  template <class S>
  struct ArithmeticAsian : Product<S> {
    S s1, avg_s1, y1, psi_d, payoff, delta, vega, gamma; // CPW estimates
    S lr_delta;
    float z1;
    int ind, ind_zero;

    void PrintName() {
      printf("\n=== ArithmeticAsian ===\n");
    }

    /* __device__ */ 
    /*   void SimulatePath(const int N, float *d_z, float *d_path) override { */ 
    /*     s1 = s0; */ 
    /*     avg_s1 = 0.0; */ 

    /*     for (int n=0; n<N; n++) { */ 
    /*       y1   = d_z[ind]; */ 
    /*       ind += blockDim.x;      // shift pointer to next element */ 
    /*       z1 = y1; // store z1 for lr calculation */ 

    /*       s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1); */ 
    /*       avg_s1 += s1; */ 
    /*     } */ 
    /*     avg_s1 /= N; */ 
    /*   } */ 

    // TODO: should return type S
    __device__
      float wn(int n, float *d_path) {
        if (n == 0) return 0.0;
        else return d_path[ind_zero + blockDim.x * (n-1)];
      }

    __device__
      float tn(int n) {
        return dt*n;
      }

    // TODO: d_path should be of type S really
    // TODO: precompute b vector and interpolations weights
    __device__
      void SimulatePath(const int N, float *d_z, float *d_path) override {
        float a, b;
        ind_zero = ind;
        s1 = S(0.0);
        int h = N; // 2^m
        int jmax = 1;
        int m = static_cast<int>(log2f(h));

        d_path[ind_zero + blockDim.x*(h-1)] = omega * T + sigma * sqrt(T) * d_z[ind];

        for (int k = 1; k <= m; k++) { // k = 1,...,m
          int imin = h / 2;
          int i = imin;
          int l = 0, r = h;
          for (int j = 1; j <= jmax; j++) {
            ind += blockDim.x;      // shift pointer to next Zn
            a = ((tn(r) - tn(i))*wn(l, d_path) + (tn(i) - tn(l))*wn(r, d_path)) / (tn(r) - tn(l));
            b = sqrt(((tn(i) - tn(l)) * (tn(r) - tn(i))) / (tn(r) - tn(l)));
            if (i == 1) z1 = d_z[ind];
            d_path[ind_zero + blockDim.x * (i-1)] = a + sigma*b*d_z[ind];

            i += h; l += h; r += h;
          }
          jmax *= 2;
          h = imin;
        }
        avg_s1 = s0;
        for (int i = 0; i < N; ++i) {
          avg_s1 += s0 * exp(d_path[ind_zero + blockDim.x * i]);
        }
        avg_s1 /= N+1;
      }

    __device__
      void CalculatePayoffs(MCResults<S> &d_results) override {
        psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

        payoff = exp(-r * T) * max(avg_s1 - k, S(0.0));
        delta = exp(r * (dt - T)) * (avg_s1 / s0) 
          * (S(1.0) - normcdf(psi_d - sigma * sqrt(dt)));
        // TODO: Calculate vega
        vega = S(0.0);
        gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
          * NormPDF(psi_d);
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));

        d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
        d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
        /* d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega; */
        d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
        d_results.lr_delta[threadIdx.x + blockIdx.x*blockDim.x] = lr_delta;
      }

    __host__
      void HostMC(const int NPATHS, const int N, float *z, float r, float dt,
          float sigma, S s0, S k, float T, float omega, MCResults<S> &results) {
        ind = 0;
        for (int i = 0; i < NPATHS; ++i) {
          s1 = s0;
          avg_s1 = s0;
          for (int n=0; n<N; n++) {
            y1   = z[ind];
            ind++;
            s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1);
            avg_s1 += s1;
          }
          avg_s1 /= N+1;
          psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

          payoff = exp(-r * T) * max(avg_s1 - k, S(0.0));
          delta = exp(r * (dt - T)) * (avg_s1 / s0) 
            * (S(1.0) - NormCDF(psi_d - sigma * sqrt(dt)));
          // TODO: Calculate vega
          vega = S(0.0);
          gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
            * NormPDF(psi_d);

          results.price[i] = payoff;
          results.delta[i] = delta;
          /* results.vega[i] = vega; */
          results.gamma[i] = gamma;
        }
      }
  };

  /* template <class S> */
  /* struct BinaryAsian : Product<S> { */
  /*   S s1, avg_s1, y1, psi_d, payoff, delta, vega, gamma; */  
  /*   int ind; */

  /*   void PrintName() { */
  /*     printf("\n=== BinaryAsian ===\n"); */
  /*   } */

  /*   __device__ */
  /*     void SimulatePath(const int N, float *d_z) override { */
  /*       s1 = s0; */
  /*       avg_s1 = s0; */

  /*       for (int n=0; n<N; n++) { */
  /*         y1   = d_z[ind]; */
  /*         ind += blockDim.x;      // shift pointer to next element */

  /*         s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1); */
  /*         avg_s1 += s1; */
  /*       } */
  /*       avg_s1 /= N; */
  /*     } */

  /*   __device__ */
  /*     void CalculatePayoffs(MCResults<S> &d_results) override { */
  /*       psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt)); */

  /*       payoff = avg_s1 - k > S(0.0) ? exp(-r * T) : S(0.0); */
  /*       delta = (exp(-r * T) / s0 * sigma * sqrt(dt)) * NormPDF(psi_d); */
  /*       // TODO: Calculate vega */
  /*       vega = S(0.0); */
  /*       gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * NormPDF(psi_d) */
  /*         * ((psi_d / (sigma * sqrt(dt)) - S(1.0))); */

  /*       d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff; */
  /*       d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta; */
  /*       /1* d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega; *1/ */
  /*       d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma; */
  /*     } */

  /*   __host__ */
  /*     void HostMC(const int NPATHS, const int N, float *z, float r, float dt, */
  /*         float sigma, S s0, S k, float T, float omega, MCResults<S> &results) { */
  /*       ind = 0; */
  /*       for (int i = 0; i < NPATHS; ++i) { */
  /*         s1 = s0; */
  /*         avg_s1 = s0; */
  /*         for (int n=0; n<N; n++) { */
  /*           y1   = z[ind]; */
  /*           ind++; */
  /*           s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1); */
  /*           avg_s1 += s1; */
  /*         } */
  /*         avg_s1 /= N; */
  /*         psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt)); */

  /*         payoff = avg_s1 - k > S(0.0) ? exp(-r * T) : S(0.0); */
  /*         delta = (exp(-r * T) / s0 * sigma * sqrt(dt)) * NormPDF(psi_d); */
  /*         // TODO: Calculate vega */
  /*         vega = S(0.0); */
  /*         gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * NormPDF(psi_d) */
  /*         * ((psi_d / (sigma * sqrt(dt)) - S(1.0))); */

  /*         results.price[i] = payoff; */
  /*         results.delta[i] = delta; */
  /*         /1* results.vega[i] = vega; *1/ */
  /*         results.gamma[i] = gamma; */
  /*       } */
  /*     } */
  /* }; */

  /* template <class S> */
  /* struct Lookback : Product<S> { */
  /*   S s1, s_max, psi_d, y1, payoff, delta, vega, gamma; */  
  /*   int ind; */

  /*   void PrintName() { */
  /*     printf("\n=== Lookback ===\n"); */
  /*   } */
    
  /*   __device__ */
  /*     void SimulatePath(const int N, float *d_z) override { */
  /*       s1 = s0; */
  /*       s_max = s0; */

  /*       for (int n=0; n<N; n++) { */
  /*         y1   = d_z[ind]; */
  /*         ind += blockDim.x;      // shift pointer to next element */

  /*         s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1); */
  /*         if (s1 > s_max) s_max = s1; */
  /*       } */
  /*     } */

  /*   __device__ */
  /*     void CalculatePayoffs(MCResults<S> &d_results) override { */
  /*       psi_d = (log(k) - log(s_max) - omega * dt) / (sigma * sqrt(dt)); */

  /*       payoff = exp(-r * T) * max(s_max - k, 0.0f); */
  /*       delta = exp(r * (dt - T)) * (s_max / s0) */
  /*         * (1.0f - normcdf(psi_d - sigma * sqrt(dt))); */
  /*       // TODO: Calculate vega */
  /*       vega = S(0.0); */
  /*       gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt))) */
  /*         * NormPDF(psi_d); */

  /*       d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff; */
  /*       d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta; */
  /*       /1* d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega; *1/ */
  /*       d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma; */
  /*     } */

  /*   // TODO */
  /*   __host__ */
  /*     void HostMC(const int NPATHS, const int N, float *z, float r, float dt, */
  /*         float sigma, S s0, S k, float T, float omega, MCResults<S> &results) { */
  /*       ind = 0; */
  /*       for (int i = 0; i < NPATHS; ++i) { */
  /*         s1 = s0; */
  /*         s_max = s0; */
  /*         for (int n=0; n<N; n++) { */
  /*           y1   = z[ind]; */
  /*           ind++; */
  /*           s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1); */
  /*           if (s1 > s_max) s_max = s1; */
  /*         } */
  /*         psi_d = (log(k) - log(s_max) - omega * dt) / (sigma * sqrt(dt)); */

  /*         payoff = exp(-r * T) * max(s_max - k, 0.0f); */
  /*         delta = exp(r * (dt - T)) * (s_max / s0) */
  /*           * (1.0f - normcdf(psi_d - sigma * sqrt(dt))); */
  /*         // TODO: Calculate vega */
  /*         vega = S(0.0); */
  /*         gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt))) */
  /*           * NormPDF(psi_d); */

  /*         results.price[i] = payoff; */
  /*         results.delta[i] = delta; */
  /*         /1* results.vega[i] = vega; *1/ */
  /*         results.gamma[i] = gamma; */
  /*       } */
  /*     } */
  /* }; */

} // namespace demeter

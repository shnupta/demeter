#pragma once

#include <cuda.h>
#include <helper_cuda.h>

#include "common.cuh"

namespace demeter {

  template <class S>
  struct Product {
    __device__ virtual void SimulatePath(const int N, float *d_z, S *d_path) = 0;
    __device__ virtual void CalculatePayoffs(MCResults<S> &d_results) = 0;
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

    __device__ 
      void SimulatePath(const int N, float *d_z, float *d_path) override { 
        s1 = s0; 
        avg_s1 = 0.0; 

        for (int n=0; n<N; n++) { 
          y1   = d_z[ind]; 
          ind += blockDim.x;      // shift pointer to next element 
          z1 = y1; // store z1 for lr calculation 

          s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1); 
          avg_s1 += s1; 
        } 
        avg_s1 /= N; 
      } 

    __device__
      S wn(int n, S *d_path) {
        if (n == 0) return S(0.0);
        else return d_path[ind_zero + blockDim.x * (n-1)];
      }

    __device__
      float tn(int n) {
        return dt*n;
      }

    // TODO: Allow for averaging dates to be passed?
    /* __device__ */
    /*   void SimulatePath(const int N, float *d_z, S *d_path) override { */
    /*     int i; */
    /*     S a, b; */
    /*     ind_zero = ind; */
    /*     int h = N; // 2^m */
    /*     int m = static_cast<int>(log2f(h)); */

    /*     d_path[ind_zero] = d_z[ind]; */

    /*     for (int k = 1; k <= m; k++) { // k = 1,...,m */
    /*       i = (1 << k) - 1; */
    /*       for (int j = (1 << (k-1)) - 1; j >= 0; --j) { */
    /*         ind += blockDim.x; */
    /*         y1 = d_z[ind]; */
    /*         a = S(0.5) * d_path[ind_zero + j * blockDim.x]; */
    /*         b = sqrt(1.0 / (1 << (k+1))); */
    /*         d_path[ind_zero + i * blockDim.x] = a - b * y1; */
    /*         i--; */
    /*         d_path[ind_zero + i * blockDim.x] = a + b * y1; */
    /*         i--; */
    /*       } */
    /*     } */
         
    /*     s1 = s0; */
    /*     for (int k = 0; k < N; ++k) { */
    /*       s1 = s1 * (1 + r * dt + sigma * sqrt(T) * d_path[ind_zero + k * blockDim.x]); */
    /*       avg_s1 += s1; */
    /*     } */
    /*     avg_s1 /= N; */
    /*   } */

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

  template <class S>
  struct European : Product<S> {
    S s1, y1, payoff, delta, vega, gamma; // CPW estimates
    S lr_delta;
    float z1;
    int ind, ind_zero;

    void PrintName() {
      printf("\n=== European ===\n");
    }

    __device__ 
      void SimulatePath(const int N, float *d_z, float *d_path) override { 
        s1 = s0; 

        for (int n=0; n<N; n++) { 
          y1   = d_z[ind]; 
          ind += blockDim.x;      // shift pointer to next element 
          z1 = y1; // store z1 for lr calculation 

          s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1); 
        } 
      } 

    __device__
      S wn(int n, S *d_path) {
        if (n == 0) return S(0.0);
        else return d_path[ind_zero + blockDim.x * (n-1)];
      }

    __device__
      float tn(int n) {
        return dt*n;
      }

    /* __device__ */
    /*   void SimulatePath(const int N, float *d_z, S *d_path) override { */
    /*     int i; */
    /*     S a, b; */
    /*     ind_zero = ind; */
    /*     int h = N; // 2^m */
    /*     int m = static_cast<int>(log2f(h)); */

    /*     d_path[ind_zero] = d_z[ind]; */

    /*     for (int k = 1; k <= m; k++) { // k = 1,...,m */
    /*       i = (1 << k) - 1; */
    /*       for (int j = (1 << (k-1)) - 1; j >= 0; --j) { */
    /*         ind += blockDim.x; */
    /*         y1 = d_z[ind]; */
    /*         a = S(0.5) * d_path[ind_zero + j * blockDim.x]; */
    /*         b = sqrt(1.0 / (1 << (k+1))); */
    /*         d_path[ind_zero + i * blockDim.x] = a - b * y1; */
    /*         i--; */
    /*         d_path[ind_zero + i * blockDim.x] = a + b * y1; */
    /*         i--; */
    /*       } */
    /*     } */
         
    /*     s1 = s0; */
    /*     for (int k = 0; k < N; ++k) { */
    /*       s1 = s1 * (1 + r * dt + sigma * sqrt(T) * d_path[ind_zero + k * blockDim.x]); */
    /*     } */
    /*   } */

    __device__
      void CalculatePayoffs(MCResults<S> &d_results) override {
        payoff = exp(-r * T) * max(s1 - k, S(0.0));
        /* delta = exp(r * (dt - T)) * (avg_s1 / s0) */ 
        /*   * (S(1.0) - normcdf(psi_d - sigma * sqrt(dt))); */
        /* // TODO: Calculate vega */
        /* vega = S(0.0); */
        /* gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt))) */
        /*   * NormPDF(psi_d); */
        /* lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt))); */

        d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
        /* d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta; */
        /* /1* d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega; *1/ */
        /* d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma; */
        /* d_results.lr_delta[threadIdx.x + blockIdx.x*blockDim.x] = lr_delta; */
      }

    __host__
      void HostMC(const int NPATHS, const int N, float *z, float r, float dt,
          float sigma, S s0, S k, float T, float omega, MCResults<S> &results) {
        ind = 0;
        for (int i = 0; i < NPATHS; ++i) {
          s1 = s0;
          for (int n=0; n<N; n++) {
            y1   = z[ind];
            ind++;
            s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1);
          }
          payoff = exp(-r * T) * max(s1 - k, S(0.0));
          /* delta = exp(r * (dt - T)) * (avg_s1 / s0) */ 
          /*   * (S(1.0) - NormCDF(psi_d - sigma * sqrt(dt))); */
          /* // TODO: Calculate vega */
          /* vega = S(0.0); */
          /* gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt))) */
          /*   * NormPDF(psi_d); */

          results.price[i] = payoff;
          /* results.delta[i] = delta; */
          /* /1* results.vega[i] = vega; *1/ */
          /* results.gamma[i] = gamma; */
        }
      }
  };

} // namespace demeter

#pragma once

#include <cuda.h>
#include <helper_cuda.h>

#include "common.cuh"

template <class T>
struct product {
  __device__ virtual void SimulatePath(const int N, float *d_z) = 0;
  __device__ virtual void CalculatePayoffs(mc_results<T> &d_results) = 0;
};

template <class S>
struct arithmetic_asian : product<S> {
  S s1, avg_s1, y1, psi_d, payoff, delta, vega, gamma;  
  int ind;

  __device__
    void SimulatePath(const int N, float *d_z) override {
      s1 = s0;
      avg_s1 = s0;

      for (int n=0; n<N; n++) {
        y1   = d_z[ind];
        ind += blockDim.x;      // shift pointer to next element

        s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1);
        avg_s1 += s1;
      }
      avg_s1 /= N;
    }

  __device__
    void CalculatePayoffs(mc_results<S> &d_results) override {
      // TODO: Check this psi_d
      psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

      payoff = exp(-r * T) * max(avg_s1 - k, S(0.0));
      delta = exp(r * (dt - T)) * (avg_s1 / s0) 
        * (S(1.0) - normcdf(psi_d - sigma * sqrt(dt)));
      // TODO: Calculate vega
      vega = S(0.0);
      gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
        * normpdf(psi_d);

      d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
      d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
      /* d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega; */
      d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
    }

  __host__
    void HostMC(const int NPATHS, const int N, float *z, float r, float dt,
        float sigma, S s0, S k, float T, float omega, mc_results<S> &results) {
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
        avg_s1 /= N;
        psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

        payoff = avg_s1 - k > S(0.0) ? exp(-r * T) : S(0.0);
        delta = (exp(-r * T) / s0 * sigma * sqrt(dt)) * normpdf(psi_d);
        // TODO: Calculate vega
        vega = S(0.0);
        gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * normpdf(psi_d)
          * ((psi_d / (sigma * sqrt(dt)) - S(1.0)));

        results.price[i] = payoff;
        results.delta[i] = delta;
        /* results.vega[i] = vega; */
        results.gamma[i] = gamma;
      }
    }
};

template <class S>
struct binary_asian : product<S> {
  S s1, avg_s1, y1, psi_d, payoff, delta, vega, gamma;  
  int ind;

  __device__
    void SimulatePath(const int N, float *d_z) override {
      s1 = s0;
      avg_s1 = s0;

      for (int n=0; n<N; n++) {
        y1   = d_z[ind];
        ind += blockDim.x;      // shift pointer to next element

        s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1);
        avg_s1 += s1;
      }
      avg_s1 /= N;
    }

  __device__
    void CalculatePayoffs(mc_results<S> &d_results) override {
      psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

      payoff = avg_s1 - k > S(0.0) ? exp(-r * T) : S(0.0);
      delta = (exp(-r * T) / s0 * sigma * sqrt(dt)) * normpdf(psi_d);
      // TODO: Calculate vega
      vega = S(0.0);
      gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * normpdf(psi_d)
        * ((psi_d / (sigma * sqrt(dt)) - S(1.0)));

      d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
      d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
      /* d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega; */
      d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
    }
};

template <class S>
struct lookback : product<S> {
  S s1, s_max, psi_d, y1, payoff, delta, vega, gamma;  
  int ind;

  __device__
    void SimulatePath(const int N, float *d_z) override {
      s1 = s0;
      s_max = s0;

      for (int n=0; n<N; n++) {
        y1   = d_z[ind];
        ind += blockDim.x;      // shift pointer to next element

        s1 = s1 * (S(1.0) + r * dt + sigma * sqrt(dt) * y1);
        if (s1 > s_max) s_max = s1;
      }
    }

  __device__
    void CalculatePayoffs(mc_results<S> &d_results) override {
      psi_d = (log(k) - log(s_max) - omega * dt) / (sigma * sqrt(dt));

      payoff = exp(-r * T) * max(s_max - k, 0.0f);
      delta = exp(r * (dt - T)) * (s_max / s0)
        * (1.0f - normcdf(psi_d - sigma * sqrt(dt)));
      // TODO: Calculate vega
      vega = S(0.0);
      gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
        * normpdf(psi_d);

      d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
      d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
      /* d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega; */
      d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
    }
};

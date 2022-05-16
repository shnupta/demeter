#pragma once

#include <cuda.h>
#include <helper_cuda.h>

#include "common.cuh"

namespace demeter {

  template <class S>
  struct Product {
    __device__ virtual void SimulatePath(const int N, float *d_z) = 0;
    __device__ virtual void CalculatePayoffs(MCResults<S> &d_results) = 0;
  };

  template <class S>
  struct ArithmeticAsian : Product<S> {
    S s1, av_s1, s_tilde, av_s_tilde, avg_s1, av_avg_s1;
    S psi_d, av_psi_d, payoff, av_payoff, delta, av_delta, vega, av_vega,
      gamma, av_gamma;
    S vega_inner_sum, av_vega_inner_sum;
    S lr_delta, lr_vega, lr_gamma;
    S final_payoff;
    float z1, av_z1, z, av_z, W1, av_W1, W_tilde, av_W_tilde;
    int ind;

    void PrintName() {
      printf("\n=== ArithmeticAsian ===\n");
    }

    __device__ 
      void SimulatePath(const int N, float *d_z) override { 
        // Initial setup
        z   = d_z[ind]; 
        av_z = -z;
        z1 = z; // Capture z1 for lr_estimate
        av_z1 = av_z;

        // Initial path values
        W1 = sqrt(dt) * z;
        av_W1 = sqrt(dt) * av_z;
        W_tilde = W1;
        av_W_tilde = W1;
        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        av_s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (av_W_tilde - av_W1));
        s1 = s0 * exp(omega * dt + sigma * W1);
        av_s1 = s0 * exp(omega * dt + sigma * av_W1);
        
        // Set initial values required for greek estimates
        avg_s1 = s1;
        av_avg_s1 = av_s1;
        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        av_vega_inner_sum = av_s_tilde * (av_W_tilde - av_W1 - sigma * (dt - dt));
        lr_vega = ((z*z - S(1.0)) / sigma) - (z * sqrt(dt));

        // Simulate over rest of N timesteps
        for (int n = 1; n < N; n++) { 
          ind += blockDim.x;      // shift pointer to random variable
          z = d_z[ind]; 
          av_z = -z;

          // Stock dynamics
          W_tilde = W_tilde + sqrt(dt) * z;
          s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          // Antithetic path
          av_W_tilde = av_W_tilde + sqrt(dt) * av_z;
          av_s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (av_W_tilde - av_W1));
          av_s1 = av_s_tilde * exp(omega * dt + sigma * av_W1); 

          // Required for greek estimations
          vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
          av_vega_inner_sum += av_s_tilde * (av_W_tilde - av_W1 - sigma * (dt*n - dt));
          lr_vega += ((z*z - S(1.0)) / sigma) - (z * sqrt(dt));
          avg_s1 += s1; 
          av_avg_s1 += av_s1;
        } 
        avg_s1 /= N; 
        av_avg_s1 /= N;
        vega_inner_sum /= N;
        av_vega_inner_sum /= N;
      } 

    __device__
      void CalculatePayoffs(MCResults<S> &d_results) override {

        psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));
        av_psi_d = (log(k) - log(av_avg_s1) - omega * dt) / (sigma * sqrt(dt));

        // Discounted payoff
        payoff = exp(-r * T) * max(avg_s1 - k, S(0.0));
        av_payoff = exp(-r * T) * max(av_avg_s1 - k, S(0.0));
        final_payoff = 0.5 * (payoff + av_payoff);

        // CPW Delta
        delta = exp(r * (dt - T)) * (avg_s1 / s0) 
          * (S(1.0) - normcdf(psi_d - sigma * sqrt(dt)));
        av_delta = exp(r * (dt - T)) * (av_avg_s1 / s0) 
          * (S(1.0) - normcdf(av_psi_d - sigma * sqrt(dt)));

        // CPW Vega
        vega = exp(r * (dt - T)) * (S(1.0) - normcdf(psi_d - sigma*sqrt(dt)))
          * vega_inner_sum + k * exp(-r * T) * NormPDF(psi_d) * sqrt(dt);
        av_vega = exp(r * (dt - T)) * (S(1.0) - normcdf(av_psi_d - sigma*sqrt(dt)))
          * av_vega_inner_sum + k * exp(-r * T) * NormPDF(av_psi_d) * sqrt(dt);

        // CPW Gamma
        gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
          * NormPDF(psi_d);
        av_gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
          * NormPDF(av_psi_d);

        // Likelihood ratio
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - S(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
          (z1 / (s0 * s0 * sigma * sqrt(dt))));

        // Antithetic averages
        delta = 0.5 * (delta + av_delta);
        vega = 0.5 * (vega + av_vega);
        gamma = 0.5 * (gamma + av_gamma);

        // Store results in respective arrays
        d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = final_payoff;
        d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
        d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega;
        d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
        d_results.lr_delta[threadIdx.x + blockIdx.x*blockDim.x] = lr_delta;
        d_results.lr_vega[threadIdx.x + blockIdx.x*blockDim.x] = lr_vega;
        d_results.lr_gamma[threadIdx.x + blockIdx.x*blockDim.x] = lr_gamma;
      }

    __host__
      void HostMC(const int NPATHS, const int N, float *h_z, float r, float dt,
          float sigma, S s0, S k, float T, float omega, MCResults<S> &results) {
        ind = 0;

        for (int i = 0; i < NPATHS; ++i) {
          // Initial setup
          z   = h_z[ind]; 
          z1 = z; // Capture z1 for lr_estimate

          // Initial path values
          W1 = sqrt(dt) * z;
          W_tilde = W1;
          s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
          s1 = s0 * exp(omega * dt + sigma * W1);
          
          // Set initial values required for greek estimates
          avg_s1 = s1;
          vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
          lr_vega = ((z*z - S(1.0)) / sigma) - (z * sqrt(dt));

          // Simulate over rest of N timesteps
          for (int n = 1; n < N; n++) { 
            ind++;
            z = h_z[ind]; 

            // Stock dynamics
            W_tilde = W_tilde + sqrt(dt) * z;
            s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
            s1 = s_tilde * exp(omega * dt + sigma * W1); 

            // Required for greek estimations
            vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
            lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
            avg_s1 += s1; 
          } 
          avg_s1 /= N; 
          vega_inner_sum /= N;

          psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

          payoff = exp(-r * T) * max(avg_s1 - k, S(0.0));

          delta = exp(r * (dt - T)) * (avg_s1 / s0) 
            * (S(1.0) - NormCDF(psi_d - sigma * sqrt(dt)));

          vega = exp(r * (dt - T)) * (S(1.0) - NormCDF(psi_d - sigma*sqrt(dt)))
            * vega_inner_sum + k * exp(-r * T) * NormPDF(psi_d) * sqrt(dt);

          gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
            * NormPDF(psi_d);

          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - S(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
            (z1 / (s0 * s0 * sigma * sqrt(dt))));

          results.price[i] = payoff;
          results.delta[i] = delta;
          results.vega[i] = vega;
          results.gamma[i] = gamma;
          results.lr_delta[i] = lr_delta;
          results.lr_vega[i] = lr_vega;
          results.lr_gamma[i] = lr_gamma;
        }
      }
  };

  template <class S>
  struct BinaryAsian : Product<S> {
    S s1, s_tilde, avg_s1, psi_d, payoff, delta, vega, gamma;  
    S vega_inner_sum;
    S lr_delta, lr_vega, lr_gamma;
    float z, z1, W1, W_tilde;
    int ind;
    

    void PrintName() {
      printf("\n=== BinaryAsian ===\n");
    }

    __device__
      void SimulatePath(const int N, float *d_z) override {
        // Initial setup
        z   = d_z[ind]; 
        z1 = z; // Capture z1 for lr_estimate

        // Initial path values
        W1 = sqrt(dt) * z;
        W_tilde = W1;
        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);
        
        // Set initial values required for greek estimates
        avg_s1 = s1;
        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        lr_vega = ((z*z - S(1.0)) / sigma) - (z * sqrt(dt));

        // Simulate over rest of N timesteps
        for (int n = 1; n < N; n++) { 
          ind += blockDim.x;      // shift pointer to random variable
          z = d_z[ind]; 

          // Stock dynamics
          W_tilde = W_tilde + sqrt(dt) * z;
          s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          // Required for greek estimations
          vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
          lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
          avg_s1 += s1; 
        } 
        avg_s1 /= N; 
        vega_inner_sum /= N;
      }

    __device__
      void CalculatePayoffs(MCResults<S> &d_results) override {
        psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

        // Discounted payoff
        payoff = avg_s1 - k > S(0.0) ? exp(-r * T) : S(0.0);

        // CPW Delta
        delta = (exp(-r * T) / (s0 * sigma * sqrt(dt))) * NormPDF(psi_d);

        // CPW Vega
        vega = exp(-r * T) * NormPDF(psi_d) *
          ((S(1.0) / (sigma * sqrt(dt) * avg_s1)) * vega_inner_sum +
           psi_d / sigma - sqrt(dt));

        // CPW Gamma
        gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * NormPDF(psi_d)
          * ((psi_d / (sigma * sqrt(dt)) - S(1.0)));

        // LR
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - S(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
          (z1 / (s0 * s0 * sigma * sqrt(dt))));

        // Store results
        d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
        d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
        d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega;
        d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
        d_results.lr_delta[threadIdx.x + blockIdx.x*blockDim.x] = lr_delta;
        d_results.lr_vega[threadIdx.x + blockIdx.x*blockDim.x] = lr_vega;
        d_results.lr_gamma[threadIdx.x + blockIdx.x*blockDim.x] = lr_gamma;
      }

    __host__
      void HostMC(const int NPATHS, const int N, float *h_z, float r, float dt,
          float sigma, S s0, S k, float T, float omega, MCResults<S> &results) {
        ind = 0;

        for (int i = 0; i < NPATHS; ++i) {
          // Initial setup
          z   = h_z[ind]; 
          z1 = z; // Capture z1 for lr_estimate

          // Initial path values
          W1 = sqrt(dt) * z;
          W_tilde = W1;
          s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
          s1 = s0 * exp(omega * dt + sigma * W1);
          
          // Set initial values required for greek estimates
          avg_s1 = s1;
          vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
          lr_vega = ((z*z - S(1.0)) / sigma) - (z * sqrt(dt));
          
          // Simulate over rest of N timesteps
          for (int n = 1; n < N; n++) { 
            ind++;
            z = h_z[ind]; 

            // Stock dynamics
            W_tilde = W_tilde + sqrt(dt) * z;
            s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
            s1 = s_tilde * exp(omega * dt + sigma * W1); 

            // Required for greek estimations
            vega_inner_sum += s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
            lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
            avg_s1 += s1; 
          } 
          avg_s1 /= N; 
          vega_inner_sum /= N;

          psi_d = (log(k) - log(avg_s1) - omega * dt) / (sigma * sqrt(dt));

          payoff = avg_s1 - k > S(0.0) ? exp(-r * T) : S(0.0);

          delta = (exp(-r * T) / (s0 * sigma * sqrt(dt))) * NormPDF(psi_d);
          
          vega = exp(-r * T) * NormPDF(psi_d) *
            ((S(1.0) / (sigma * sqrt(dt) * avg_s1)) * vega_inner_sum +
             psi_d / sigma - sqrt(dt));

          gamma = (exp(-r * T) / (s0 * s0 * sigma * sqrt(dt))) * NormPDF(psi_d)
          * ((psi_d / (sigma * sqrt(dt)) - S(1.0)));

          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - S(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
            (z1 / (s0 * s0 * sigma * sqrt(dt))));

          results.price[i] = payoff;
          results.delta[i] = delta;
          results.vega[i] = vega;
          results.gamma[i] = gamma;
          results.lr_delta[i] = lr_delta;
          results.lr_vega[i] = lr_vega;
          results.lr_gamma[i] = lr_gamma;
        }
      }
  };

  template <class S>
  struct Lookback : Product<S> {
    S s1, s_tilde, s_max, psi_d, payoff, delta, vega, gamma;  
    S vega_inner_sum;
    S lr_delta, lr_vega, lr_gamma;
    float z, z1, W1, W_tilde;
    int ind;

    void PrintName() {
      printf("\n=== Lookback ===\n");
    }
    
    __device__
      void SimulatePath(const int N, float *d_z) override {
        // Initial setup
        z   = d_z[ind]; 
        z1 = z; // Capture z1 for lr_estimate

        // Initial path values
        W1 = sqrt(dt) * z;
        W_tilde = W1;
        s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
        s1 = s0 * exp(omega * dt + sigma * W1);
        
        // Set initial values required for greek estimates
        vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
        lr_vega = ((z*z - S(1.0)) / sigma) - (z * sqrt(dt));
        s_max = s1;

        for (int n=1; n<N; n++) {
          ind += blockDim.x;      // shift pointer to random variable
          z = d_z[ind]; 

          // Stock dynamics
          W_tilde = W_tilde + sqrt(dt) * z;
          s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
          s1 = s_tilde * exp(omega * dt + sigma * W1); 

          // Required for greek estimations
          if (s1 > s_max) {
            s_max = s1;
            vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*n - dt));
          } 
          lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
        }
      }

    __device__
      void CalculatePayoffs(MCResults<S> &d_results) override {
        psi_d = (log(k) - log(s_max) - omega * dt) / (sigma * sqrt(dt));

        // Discounted payoff
        payoff = exp(-r * T) * max(s_max - k, S(0.0));

        // CPW Delta
        delta = exp(r * (dt - T)) * (s_max / s0)
          * (1.0f - normcdf(psi_d - sigma * sqrt(dt)));

        // CPW Vega
        vega = exp(r * (dt - T)) * (S(1.0) - normcdf(psi_d - sigma*sqrt(dt)))
          * vega_inner_sum + k * exp(-r * T) * NormPDF(psi_d) * sqrt(dt);

        // CPW Gamma
        gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
          * NormPDF(psi_d);

        // LR
        lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
        lr_vega = payoff * lr_vega;
        lr_gamma = payoff * (((z1*z1 - S(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
          (z1 / (s0 * s0 * sigma * sqrt(dt))));

        // Store results
        d_results.price[threadIdx.x + blockIdx.x*blockDim.x] = payoff;
        d_results.delta[threadIdx.x + blockIdx.x*blockDim.x] = delta;
        d_results.vega[threadIdx.x + blockIdx.x*blockDim.x] = vega;
        d_results.gamma[threadIdx.x + blockIdx.x*blockDim.x] = gamma;
        d_results.lr_delta[threadIdx.x + blockIdx.x*blockDim.x] = lr_delta;
        d_results.lr_vega[threadIdx.x + blockIdx.x*blockDim.x] = lr_vega;
        d_results.lr_gamma[threadIdx.x + blockIdx.x*blockDim.x] = lr_gamma;
      }

    __host__
      void HostMC(const int NPATHS, const int N, float *h_z, float r, float dt,
          float sigma, S s0, S k, float T, float omega, MCResults<S> &results) {
        ind = 0;
        for (int i = 0; i < NPATHS; ++i) {
          // Initial setup
          z   = h_z[ind]; 
          z1 = z; // Capture z1 for lr_estimate

          // Initial path values
          W1 = sqrt(dt) * z;
          W_tilde = W1;
          s_tilde = s0 * exp(omega * sqrt(dt - dt) + sigma * (W_tilde - W1));
          s1 = s0 * exp(omega * dt + sigma * W1);
          
          // Set initial values required for greek estimates
          vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt - dt));
          lr_vega = ((z*z - S(1.0)) / sigma) - (z * sqrt(dt));
          s_max = s1;

          for (int n=0; n<N; n++) {
            ind++;
            z = h_z[ind]; 

            // Stock dynamics
            W_tilde = W_tilde + sqrt(dt) * z;
            s_tilde = s0 * exp(omega * (dt*n - dt) + sigma * (W_tilde - W1));
            s1 = s_tilde * exp(omega * dt + sigma * W1); 

            // Required for greek estimations
            if (s1 > s_max) {
              s_max = s1;
              vega_inner_sum = s_tilde * (W_tilde - W1 - sigma * (dt*n - dt)); 
            } 
            lr_vega += ((z*z - 1) / sigma) - (z * sqrt(dt));
          }

          psi_d = (log(k) - log(s_max) - omega * dt) / (sigma * sqrt(dt));

          payoff = exp(-r * T) * max(s_max - k, 0.0f);

          delta = exp(r * (dt - T)) * (s_max / s0)
            * (1.0f - normcdf(psi_d - sigma * sqrt(dt)));

          vega = exp(r * (dt - T)) * (S(1.0) - NormCDF(psi_d - sigma*sqrt(dt)))
            * vega_inner_sum + k * exp(-r * T) * NormPDF(psi_d) * sqrt(dt);

          gamma = ((k * exp(-r * T)) / (s0 * s0 * sigma * sqrt(dt)))
            * NormPDF(psi_d);

          lr_delta = payoff * (z1 / (s0 * sigma * sqrt(dt)));
          lr_vega = payoff * lr_vega;
          lr_gamma = payoff * (((z1*z1 - S(1.0)) / (s0 * s0 * sigma * sigma * dt)) - 
            (z1 / (s0 * s0 * sigma * sqrt(dt))));

          results.price[i] = payoff;
          results.delta[i] = delta;
          results.vega[i] = vega;
          results.gamma[i] = gamma;
          results.lr_delta[i] = lr_delta;
          results.lr_vega[i] = lr_vega;
          results.lr_gamma[i] = lr_gamma;
        }
      }
  };

} // namespace demeter

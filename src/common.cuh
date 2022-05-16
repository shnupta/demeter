#pragma once

namespace demeter {

  /* Device contants */
  __constant__ int   N, PATHS;
  __constant__ float T, r, sigma, dt, omega, s0, k;

  /* Struct to be passed to the MC method to store calculated values */
  template <class T>
  struct MCResults {
    T *price, *delta, *vega, *gamma;
    T *lr_delta, *lr_vega, *lr_gamma;
    double avg_price = 0.0, avg_delta = 0.0, avg_vega = 0.0, avg_gamma = 0.0;
    double err_price = 0.0, err_delta = 0.0, err_vega = 0.0, err_gamma = 0.0;
    double avg_lr_delta = 0.0, avg_lr_vega = 0.0, avg_lr_gamma = 0.0;
    double err_lr_delta = 0.0, err_lr_vega = 0.0, err_lr_gamma = 0.0;

    __host__
      void AllocateHost(const int size) {
        price = (T *) malloc(sizeof(T) * size);
        delta = (T *) malloc(sizeof(T) * size);
        vega = (T *) malloc(sizeof(T) * size);
        gamma = (T *) malloc(sizeof(T) * size);
        lr_delta = (T *) malloc(sizeof(T) * size);
        lr_vega = (T *) malloc(sizeof(T) * size);
        lr_gamma = (T *) malloc(sizeof(T) * size);
      }

    __host__
      void AllocateDevice(const int size) {
        checkCudaErrors(cudaMalloc((void **) &price, sizeof(T) * size));
        checkCudaErrors(cudaMalloc((void **) &delta, sizeof(T) * size));
        checkCudaErrors(cudaMalloc((void **) &vega, sizeof(T) * size));
        checkCudaErrors(cudaMalloc((void **) &gamma, sizeof(T) * size));
        checkCudaErrors(cudaMalloc((void **) &lr_delta, sizeof(T) * size));
        checkCudaErrors(cudaMalloc((void **) &lr_vega, sizeof(T) * size));
        checkCudaErrors(cudaMalloc((void **) &lr_gamma, sizeof(T) * size));
      }

    __host__
      void CopyFromDevice(const int size, const MCResults<T> &d_results) {
        checkCudaErrors( cudaMemcpy(price, d_results.price,
              sizeof(T) * size, cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaMemcpy(delta, d_results.delta,
              sizeof(T) * size, cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaMemcpy(vega, d_results.vega,
              sizeof(T) * size, cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaMemcpy(gamma, d_results.gamma,
              sizeof(T) * size, cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaMemcpy(lr_delta, d_results.lr_delta,
              sizeof(T) * size, cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaMemcpy(lr_vega, d_results.lr_vega,
              sizeof(T) * size, cudaMemcpyDeviceToHost) );
        checkCudaErrors( cudaMemcpy(lr_gamma, d_results.lr_gamma,
              sizeof(T) * size, cudaMemcpyDeviceToHost) );
      }

    __host__
      void ReleaseHost() {
        free(price);
        free(delta);
        free(vega);
        free(gamma);
        free(lr_delta);
        free(lr_vega);
        free(lr_gamma);
      }

    __host__
      void ReleaseDevice() {
        checkCudaErrors(cudaFree(price));
        checkCudaErrors(cudaFree(delta));
        checkCudaErrors(cudaFree(vega));
        checkCudaErrors(cudaFree(gamma));
        checkCudaErrors(cudaFree(lr_delta));
        checkCudaErrors(cudaFree(lr_vega));
        checkCudaErrors(cudaFree(lr_gamma));
      }

    __host__
      void CalculateStatistics(const int size) {
        double p_sum = 0.0, p_sum2 = 0.0, d_sum = 0.0, d_sum2 = 0.0,
               v_sum = 0.0, v_sum2 = 0.0, g_sum = 0.0, g_sum2 = 0.0;
        double lr_d_sum = 0.0, lr_d_sum2 = 0.0, lr_v_sum = 0.0,
               lr_v_sum2 = 0.0, lr_g_sum = 0.0, lr_g_sum2 = 0.0;

        for (int i = 0; i < size; ++i) {
          p_sum += price[i];
          p_sum2 += price[i] * price[i];
          d_sum += delta[i];
          d_sum2 += delta[i] * delta[i];
          v_sum += vega[i];
          v_sum2 += vega[i] * vega[i];
          g_sum += gamma[i];
          g_sum2 += gamma[i] * gamma[i];
          lr_d_sum += lr_delta[i];
          lr_d_sum2 += lr_delta[i] * lr_delta[i];
          lr_v_sum += lr_vega[i];
          lr_v_sum2 += lr_vega[i] * lr_vega[i];
          lr_g_sum += lr_gamma[i];
          lr_g_sum2 += lr_gamma[i] * lr_gamma[i];
        }

        avg_price = p_sum / size;
        avg_delta = d_sum / size;
        avg_vega = v_sum / size;
        avg_gamma = g_sum / size;
        avg_lr_delta = lr_d_sum / size;
        avg_lr_vega = lr_v_sum / size;
        avg_lr_gamma = lr_g_sum / size;

        err_price = sqrt((p_sum2 / size - (p_sum / size) * (p_sum / size)) / size);
        err_delta = sqrt((d_sum2 / size - (d_sum / size) * (d_sum / size)) / size);
        err_vega = sqrt((v_sum2 / size - (v_sum / size) * (v_sum / size)) / size);
        err_gamma = sqrt((g_sum2 / size - (g_sum / size) * (g_sum / size)) / size);
        err_lr_delta = sqrt((lr_d_sum2 / size - (lr_d_sum / size) * (lr_d_sum / size)) / size);
        err_lr_vega = sqrt((lr_v_sum2 / size - (lr_v_sum / size) * (lr_v_sum / size)) / size);
        err_lr_gamma = sqrt((lr_g_sum2 / size - (lr_g_sum / size) * (lr_g_sum / size)) / size);
      }

    __host__
      void PrintStatistics(bool print_header, const char* dev) {
        if (print_header) {
          printf("%6s | %13s | %13s | %13s | %13s | %13s | %13s | %13s | %13s | %13s | %13s | %13s | %13s | %13s | %13s |\n",
              "dev", "price", "err", "delta", "err", "vega", "err", "gamma", "err", "lr delta", "err", "lr vega", "err", "lr gamma", "err");
          printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
        }
        printf("%6s | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f | %13.8f |\n",
            dev, avg_price, err_price, avg_delta, err_delta, avg_vega, err_vega, avg_gamma, err_gamma, avg_lr_delta, err_lr_delta, avg_lr_vega, err_lr_vega, avg_lr_gamma, err_lr_gamma);
        printf("\nVRF for delta = %13.8f\n", (err_lr_delta * err_lr_delta) / (err_delta * err_delta));
        printf("VRF for vega  = %13.8f\n", (err_lr_vega * err_lr_vega) / (err_vega * err_vega));
        printf("VRF for gamma = %13.8f\n\n", (err_lr_gamma * err_lr_gamma) / (err_gamma * err_gamma));
      }
  };
} // namespace demeter

/* A timer for both GPU and CPU timing */

#pragma once

#include <cuda.h>
#include <helper_cuda.h>

#include <chrono>

namespace demeter {
  class Timer {
    public:
      Timer() {
        cudaEventCreate(&d_start);
        cudaEventCreate(&d_stop);
      }

      inline void StartDeviceTimer() {
        cudaEventRecord(d_start);
      }

      inline void StopDeviceTimer() {
        cudaEventRecord(d_stop);
        cudaEventSynchronize(d_stop);
        cudaEventElapsedTime(&d_elapsed, d_start, d_stop);
      } 

      float GetDeviceElapsedTime() {
        return d_elapsed;
      }

      inline void StartHostTimer() {
        h_start = std::chrono::steady_clock::now();
      }

      inline void StopHostTimer() {
        h_stop = std::chrono::steady_clock::now();
        h_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(h_stop - h_start).count();
      }

      float GetHostElapsedTime() {
        return h_elapsed;
      }

      float GetSpeedUpFactor() {
        return h_elapsed / d_elapsed;
      }


    private:
      cudaEvent_t d_start, d_stop;
      std::chrono::steady_clock::time_point h_start, h_stop;
      float h_elapsed, d_elapsed;

  };

} // namespace demeter

#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda.h>
#include <cuda_runtime.h>

void cuda_start_timer(cudaEvent_t* start, cudaEvent_t* stop) {
    cudaEventCreate(start);
    cudaEventCreate(stop);
    cudaEventRecord(*start);
}

float cuda_stop_timer(cudaEvent_t* start, cudaEvent_t* stop) {
    cudaEventRecord(*stop);
    cudaEventSynchronize(*stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, *start, *stop);
    return milliseconds;
}

#endif
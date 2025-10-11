#include "cuda_utils.h"
#include <iostream>
#include <cuda_runtime.h>

// Wrapper function for cudaFree
void cudaFreeWrapper(void* ptr) noexcept {
    // Can't do this due to `noexcept`
    // cudaError_t error = cudaFree(ptr);
    // if (error != cudaSuccess) {
    //     throw std::runtime_error(cudaGetErrorString(error));
    // }
    cudaFree(ptr);
}

void print_cuda(int * & ptr, int size) {
    int* h_ptr = new int[size];
    CHECK_CUDA(cudaMemcpy(h_ptr, ptr, size*sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nDevice array: ");
    for(int i=0; i<size; i++) {
        printf(" %d", h_ptr[i]);
    }
    
    delete[] h_ptr;
}


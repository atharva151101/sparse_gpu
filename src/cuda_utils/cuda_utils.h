#pragma once

void cudaFreeWrapper(void* ptr) noexcept;
void print_cuda(int * & ptr, int size);

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("cuda error\n");                                                \
        throw std::runtime_error(cudaGetErrorString(status));                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cusparse error\n");                                            \
        throw std::runtime_error(cusparseGetErrorString(status));              \
    }                                                                          \
}


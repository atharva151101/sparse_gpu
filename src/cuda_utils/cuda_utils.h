#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <sstream>

void cudaFreeWrapper(void* ptr) noexcept;
void print_cuda(int * & ptr, int size);
void print_cuda(float * & ptr, int size);

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


template<typename index_t, typename value_t>
struct SparseVector{
    index_t length;
    index_t nnz;
    index_t * indices;
    value_t * values;

    SparseVector(index_t * _indices, value_t * _values, const index_t _length, const index_t _nnz)
        : indices(_indices), values(_values), length(_length), nnz(_nnz) {}
};

template<typename index_t, typename value_t>
struct CSRMatrix{
    index_t rows;
    index_t cols;
    index_t nnz;
    index_t * row_offsets;
    index_t * col_indices;
    value_t * values;
};

template<typename index_t, typename value_t>
struct COOMatrix{
    index_t rows;
    index_t cols;
    index_t nnz;
    index_t * row_indices;
    index_t * col_indices;
    value_t * values;
};
#pragma once
#include <cstdint>
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"

void sparse_vector_fusion_test(const SparseVector<int, float> A, const SparseVector<int, float> B, const SparseVector<int, float> C, 
    int * & D_indices, float * & D_values, int * & D_nnz, int num_fused);
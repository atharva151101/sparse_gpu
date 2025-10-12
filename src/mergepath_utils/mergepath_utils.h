#pragma once
#include "../cuda_utils/cuda_utils.h"
#include <cstdint>
#include <iostream>
#include <sstream>
// mergepath_partition_find finds the balanced merge path partition boundaries for each thread
// 
// params:
//   num_tensors : number of input tensors
//   total_size : total number of non-zeros across all tensors
//   indices : indices[i] has indices array for the ith tensor
//   sizes : sizes[i] has number of non-zeros for each tensor
//   mergepath_boundary : (Must be preallocated) mergepath_boundary[i][t] has the output boundary in i^th tensor for the j^th thread
//                        boundary is inclusive, i.e. the last index that should be processed by that thread , -1 indicates no work for that tensor 
//   per_thread_work :  number of non-zeros each thread should process (this could be +-1 due to balancing)

int * sparse_vectors_balanced_mergepath(
    const uint num_vectors,
    const int total_size,
    const SparseVector<int, float> * vectors, 
    const uint num_blocks,
    const uint threads_per_block,
    const uint per_thread_work
);


#include "sparse_vector.h"
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>
#include <iostream>
#include <sstream>




// kernels to compute (a+b)*c through full fusion (l.b on a,b,c)
__global__ void sparse_vectors_full_fusion_precompute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * nnz_count
    );

__global__ void sparse_vectors_full_fusion_compute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * D_indices, float * D_values,
    int *nnz_prefix
    );


// kernels to compute (a+b)*c through partial fusion (l.b on a,b)

__global__ void sparse_vectors_partial_fusion_precompute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * nnz_count
    );

__global__ void sparse_vectors_partial_fusion_compute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * D_indices, float * D_values,
    int *nnz_prefix
    );


// kernels to compute (a+b)*c through no fusion (l.b on c)
__global__ void sparse_vectors_no_fusion_precompute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * nnz_count
    );

__global__ void sparse_vectors_no_fusion_compute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * D_indices, float * D_values,
    int *nnz_prefix
    );


// sparse_vector_fusion_test computes (A+B)*C where A,B,C are sparse vectors
// num_fused = 3 means full fusion (l.b on A,B,C), 
// num_fused = 2 means partial fusion (l.b on A,B), 
// num_fused = 1 means no fusion (l.b on C)
void sparse_vector_fusion_test(const SparseVector<int, float> A, const SparseVector<int, float> B, const SparseVector<int, float> C, 
    int * & D_indices, float * & D_values, int * & D_nnz, int num_fused) {

    int num_blocks = 1;
    int threads_per_block = 64;
    int * mergepath_boundaries;

    SparseVector<int32_t, float>* g_vectors;
    CHECK_CUDA(cudaMalloc(&g_vectors, 3 * sizeof(SparseVector<int32_t, float>)));
    

    CHECK_CUDA(cudaMemcpy(g_vectors, &A, sizeof(SparseVector<int32_t, float>), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(g_vectors + 1, &B, sizeof(SparseVector<int32_t, float>), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(g_vectors + 2, &C, sizeof(SparseVector<int32_t, float>), cudaMemcpyHostToDevice));
    
    int total_size, per_thread_work;

    //printf("Finding mergepath boundaries\n");


    if (num_fused == 3) {
        total_size = A.nnz + B.nnz + C.nnz;
        per_thread_work = (total_size)/(num_blocks*threads_per_block) + 2; // +(num_vectors-1) to account for boundary shifts due to balancing 
        
        mergepath_boundaries = sparse_vectors_balanced_mergepath(
        3,
        total_size,
        g_vectors, 
        num_blocks,
        threads_per_block,
        per_thread_work
        );
    
        CHECK_CUDA(cudaGetLastError());
    } else if (num_fused == 2) {
        total_size = A.nnz + B.nnz;
        per_thread_work = (total_size)/(num_blocks*threads_per_block) + 1; // +(num_vectors-1) to account for boundary shifts due to balancing
        
        
        mergepath_boundaries = sparse_vectors_balanced_mergepath(
        2, 
        total_size,
        g_vectors, // Passing just A and B , num_vectors = 2
        num_blocks,
        threads_per_block,
        per_thread_work
        );
        CHECK_CUDA(cudaGetLastError());
        
    } else {
        total_size = C.nnz;
        per_thread_work = (total_size)/(num_blocks*threads_per_block) + 1; 
        
        mergepath_boundaries = sparse_vectors_balanced_mergepath(
        1,
        total_size,
        g_vectors+2, // paas only C
        num_blocks,
        threads_per_block,
        per_thread_work
        );

        CHECK_CUDA(cudaGetLastError());
        //print_cuda(mergepath_boundaries, 10);

    }

    
    //printf("Starting pre-compute paas");

    int * nnz_count;
    

    CHECK_CUDA(cudaMalloc(&nnz_count, num_blocks * threads_per_block * sizeof(int)));
    CHECK_CUDA(cudaMemset(nnz_count, 0, num_blocks * threads_per_block * sizeof(int)));

    if (num_fused == 3) {
        sparse_vectors_full_fusion_precompute<<<num_blocks, threads_per_block>>>(
            g_vectors,
            mergepath_boundaries,
            per_thread_work,
            nnz_count
        );
        CHECK_CUDA(cudaGetLastError());
    } else if (num_fused == 2) {
        sparse_vectors_partial_fusion_precompute<<<num_blocks, threads_per_block>>>(
            g_vectors,
            mergepath_boundaries,
            per_thread_work,
            nnz_count
        );
        CHECK_CUDA(cudaGetLastError());
    } else {
        sparse_vectors_no_fusion_precompute<<<num_blocks, threads_per_block>>>(
            g_vectors,
            mergepath_boundaries,
            per_thread_work,
            nnz_count
        );
        CHECK_CUDA(cudaGetLastError());
    }

    //printf("Computing nnz prefix sum\n");
    int * nnz_prefix;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CHECK_CUDA(cudaMalloc((void**)&nnz_prefix, sizeof(int)*((num_blocks * threads_per_block) + 1)));
    CHECK_CUDA(cudaMemset(nnz_prefix, 0, sizeof(int)*((num_blocks * threads_per_block) + 1)));
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnz_prefix+1, num_blocks * threads_per_block));

    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nnz_count, nnz_prefix+1, num_blocks * threads_per_block));
    CHECK_CUDA( cudaGetLastError() );
    
    cudaFree(d_temp_storage);

    CHECK_CUDA(cudaMemcpy(D_nnz, nnz_prefix + num_blocks * threads_per_block, sizeof(int), cudaMemcpyDeviceToHost));

    //printf("Total nnz in output: %d\n", *D_nnz);

    CHECK_CUDA(cudaMalloc(&D_indices, (*D_nnz) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&D_values, (*D_nnz) * sizeof(float)));
    
    //printf("Starting final paas\n");
    if( num_fused == 3) {
        sparse_vectors_full_fusion_compute<<<num_blocks, threads_per_block>>>(
                g_vectors,
                mergepath_boundaries,
                per_thread_work,
                D_indices, D_values,
                nnz_prefix
            );
            CHECK_CUDA(cudaGetLastError());
    } else if (num_fused == 2) {
        sparse_vectors_partial_fusion_compute<<<num_blocks, threads_per_block>>>(
                g_vectors,
                mergepath_boundaries,
                per_thread_work,
                D_indices, D_values,
                nnz_prefix
            );
            CHECK_CUDA(cudaGetLastError());
    } else {
        sparse_vectors_no_fusion_compute<<<num_blocks, threads_per_block>>>(
                g_vectors,
                mergepath_boundaries,
                per_thread_work,
                D_indices, D_values,
                nnz_prefix
            );
            CHECK_CUDA(cudaGetLastError());
    }

    //printf("Finished Computation\n");
    cudaFree(mergepath_boundaries);

    cudaFree(nnz_count);
    cudaFree(nnz_prefix);

    cudaFree(g_vectors);
    return;
}


__global__ void sparse_vectors_full_fusion_precompute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * nnz_count
    ) {

    const SparseVector<int, float> * A = vectors;
    const SparseVector<int, float> * B = vectors + 1;
    const SparseVector<int, float> * C = vectors + 2;

    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int start_A, start_B, start_C, end_A, end_B, end_C;
    
    start_A = mergepath_partitions[0*num_threads + tid];
    start_B = mergepath_partitions[1*num_threads + tid];
    start_C = mergepath_partitions[2*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_A = mergepath_partitions[0*num_threads + tid + 1];
        end_B = mergepath_partitions[1*num_threads + tid + 1];
        end_C = mergepath_partitions[2*num_threads + tid + 1];
    } else {
        end_A = A->nnz - 1;
        end_B = B->nnz - 1;
        end_C = C->nnz - 1;
    }

    int count = 0;

    int idx_A = start_A + 1;
    int idx_B = start_B + 1;
    int idx_C = start_C + 1;

    while( (idx_A <= end_A) && (idx_B <= end_B) && (idx_C <= end_C) ) {
        int ia0 = A->indices[idx_A];
        int ib0 = B->indices[idx_B];
        int ic0 = C->indices[idx_C];
        
        int i = min(ia0, min(ib0, ic0));
        if ((ia0 == i && ib0 == i) && ic0 == i) {
            count++;
        }
        else if (ia0 == i && ic0 == i) {
            count++;
        }
        else if (ib0 == i && ic0 == i) {
            count++;
        }
        
        idx_A += (int32_t)(ia0 == i);
        idx_B += (int32_t)(ib0 == i);
        idx_C += (int32_t)(ic0 == i);
    }

    while( (idx_B <= end_B) && (idx_C <= end_C) ) {
        int ib0 = B->indices[idx_B];
        int ic0 = C->indices[idx_C];
        
        int i = min(ib0, ic0);
        
        if (ib0 == i && ic0 == i) {
            count++;
        }
        
        idx_B += (int32_t)(ib0 == i);
        idx_C += (int32_t)(ic0 == i);
        
    }

    while( (idx_A <= end_A) && (idx_C <= end_C) ) {
        int ia0 = A->indices[idx_A];
        int ic0 = C->indices[idx_C];
        
        int i = min(ia0, ic0);
        
        if (ia0 == i && ic0 == i) {
            count++;
        }
        
        idx_A += (int32_t)(ia0 == i);
        idx_C += (int32_t)(ic0 == i);
        
    }

    nnz_count[tid] = count;
    return;

}

__global__ void sparse_vectors_full_fusion_compute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * D_indices, float * D_values,
    int *nnz_prefix
    ) {
    const SparseVector<int, float> * A = vectors;
    const SparseVector<int, float> * B = vectors + 1;
    const SparseVector<int, float> * C = vectors + 2;

    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int start_A, start_B, start_C, end_A, end_B, end_C;
    
    start_A = mergepath_partitions[0*num_threads + tid];
    start_B = mergepath_partitions[1*num_threads + tid];
    start_C = mergepath_partitions[2*num_threads + tid];

    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_A = mergepath_partitions[0*num_threads + tid + 1];
        end_B = mergepath_partitions[1*num_threads + tid + 1];
        end_C = mergepath_partitions[2*num_threads + tid + 1];
    } else {
        end_A = A->nnz - 1;
        end_B = B->nnz - 1;
        end_C = C->nnz - 1;
    }

    int idx_A = start_A + 1;
    int idx_B = start_B + 1;
    int idx_C = start_C + 1;

    int idx_D = nnz_prefix[tid];

    while( (idx_A <= end_A) && (idx_B <= end_B) && (idx_C <= end_C) ) {
        int ia0 = A->indices[idx_A];
        int ib0 = B->indices[idx_B];
        int ic0 = C->indices[idx_C];
        
        int i = min(ia0, min(ib0, ic0));
        if ((ia0 == i && ib0 == i) && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = (A->values[idx_A] + B->values[idx_B]) * C->values[idx_C];
            idx_D++;

            //printf("\n new D_value : %f %f", D_values[idx_D-1], (A->values[idx_A] + B->values[idx_B]) * C->values[idx_C]);
        }
        else if (ia0 == i && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = A->values[idx_A] * C->values[idx_C];
            idx_D++;
            //printf("\n new D_value : %f %f", D_values[idx_D-1], A->values[idx_A]  * C->values[idx_C]);
        }
        else if (ib0 == i && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = B->values[idx_B] * C->values[idx_C];
            idx_D++;
            //printf("\n new D_value : %f %f", D_values[idx_D-1], B->values[idx_B]  * C->values[idx_C]);
        }
        
        idx_A += (int32_t)(ia0 == i);
        idx_B += (int32_t)(ib0 == i);
        idx_C += (int32_t)(ic0 == i);
    }

    while( (idx_B <= end_B) && (idx_C <= end_C) ) {
        int ib0 = B->indices[idx_B];
        int ic0 = C->indices[idx_C];
        
        int i = min(ib0, ic0);
        
        if (ib0 == i && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = B->values[idx_B] * C->values[idx_C];
            idx_D++;
        }
        
        idx_B += (int32_t)(ib0 == i);
        idx_C += (int32_t)(ic0 == i);
        
    }

    while( (idx_A <= end_A) && (idx_C <= end_C) ) {
        int ia0 = A->indices[idx_A];
        int ic0 = C->indices[idx_C];
        
        int i = min(ia0, ic0);
        
        if (ia0 == i && ic0 == i) {
            D_indices[idx_D] = i;
            D_values[idx_D] = A->values[idx_A] * C->values[idx_C];
            idx_D++;
        }
        
        idx_A += (int32_t)(ia0 == i);
        idx_C += (int32_t)(ic0 == i);
        
    }

    return;

}

__device__  __inline__ int locate( const SparseVector<int,float> * vect, int index) {
    int lo = 0;
    int hi = vect->nnz - 1;

    
    int mid;
    while (lo <= hi) {
        mid = lo + (hi - lo) / 2;
        if (vect->indices[mid] == index) {
            return mid;
        } else if (vect->indices[mid] < index) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1; // not found
}



__global__ void sparse_vectors_partial_fusion_precompute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * nnz_count
    ) {
    const SparseVector<int, float> * A = vectors;
    const SparseVector<int, float> * B = vectors + 1;
    const SparseVector<int, float> * C = vectors + 2;

    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int start_A, start_B, end_A, end_B;
    
    start_A = mergepath_partitions[0*num_threads + tid];
    start_B = mergepath_partitions[1*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_A = mergepath_partitions[0*num_threads + tid + 1];
        end_B = mergepath_partitions[1*num_threads + tid + 1];
    } else {
        end_A = A->nnz - 1;
        end_B = B->nnz - 1;
    }

    int count = 0;

    int idx_A = start_A + 1;
    int idx_B = start_B + 1;

    while( (idx_A <= end_A) && (idx_B <= end_B)) {
        int ia0 = A->indices[idx_A];
        int ib0 = B->indices[idx_B];
        
        int i = min(ia0, ib0);
        if (locate(C, i) != -1) {
            count++;
        }
        
        idx_A += (int32_t)(ia0 == i);
        idx_B += (int32_t)(ib0 == i);
    }

    while( (idx_B <= end_B)) {
        int ib0 = B->indices[idx_B];
        
        int i = ib0;
        
        if (locate(C, i) != -1) {
            count++;
        }
        
        idx_B += (int32_t)(ib0 == i);
    }

    while( (idx_A <= end_A)) {
        int ia0 = A->indices[idx_A];
        
        int i = ia0; 
        
        if (locate(C, i) != -1) {
            count++;
        }
        
        idx_A += (int32_t)(ia0 == i);
        
    }

    nnz_count[tid] = count;

    return;
    
}

__global__ void sparse_vectors_partial_fusion_compute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * D_indices, float * D_values,
    int *nnz_prefix
    ) {
    const SparseVector<int, float> * A = vectors;
    const SparseVector<int, float> * B = vectors + 1;
    const SparseVector<int, float> * C = vectors + 2;

    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int start_A, start_B, end_A, end_B;
    
    start_A = mergepath_partitions[0*num_threads + tid];
    start_B = mergepath_partitions[1*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_A = mergepath_partitions[0*num_threads + tid + 1];
        end_B = mergepath_partitions[1*num_threads + tid + 1];
    } else {
        end_A = A->nnz - 1;
        end_B = B->nnz - 1;
    }


    int idx_A = start_A + 1;
    int idx_B = start_B + 1;

    int idx_D = nnz_prefix[tid];

    while( (idx_A <= end_A) && (idx_B <= end_B)) {
        int ia0 = A->indices[idx_A];
        int ib0 = B->indices[idx_B];
        
        int i = min(ia0, ib0);
        int idx_C = locate(C, i);
        if (idx_C != -1) {
            D_indices[idx_D] = i;
            D_values[idx_D] = (A->values[idx_A] + B->values[idx_B]) * C->values[idx_C];
            idx_D++;
        }
        
        idx_A += (int32_t)(ia0 == i);
        idx_B += (int32_t)(ib0 == i);
    }

    while( (idx_B <= end_B)) {
        int ib0 = B->indices[idx_B];
        
        int i = ib0;
        

        int idx_C = locate(C, i);
        if (idx_C != -1) {
            D_indices[idx_D] = i;
            D_values[idx_D] = B->values[idx_B] * C->values[idx_C];
            idx_D++;
        }
        
        idx_B += (int32_t)(ib0 == i);
    }

    while( (idx_A <= end_A)) {
        int ia0 = A->indices[idx_A];
        
        int i = ia0; 
        
        int idx_C = locate(C, i);
        if (idx_C != -1) {
            D_indices[idx_D] = i;
            D_values[idx_D] = A->values[idx_A] * C->values[idx_C];
            idx_D++;
        }
        
        idx_A += (int32_t)(ia0 == i);
        
    }


    return;
    
}



__global__ void sparse_vectors_no_fusion_precompute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * nnz_count
    ) {
    const SparseVector<int, float> * A = vectors;
    const SparseVector<int, float> * B = vectors + 1;
    const SparseVector<int, float> * C = vectors + 2;


     int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int start_C, end_C;
    
    start_C = mergepath_partitions[0*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_C = mergepath_partitions[0*num_threads + tid + 1];
    } else {
        end_C = C->nnz - 1;
    }

    int idx_C = start_C + 1;

    int count = 0;

    while( (idx_C <= end_C)) {
        int ic0 = C->indices[idx_C];
        
        int i = ic0;
        
        if (locate(A, i) != -1 || locate(B, i) != -1) {
            count++;
        }
        
        idx_C += (int32_t)(ic0 == i);
    }

    nnz_count[tid] = count;
    return;
}


__global__ void sparse_vectors_no_fusion_compute(
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work,
    int * D_indices, float * D_values,
    int *nnz_prefix
    ) {
    const SparseVector<int, float> * A = vectors;
    const SparseVector<int, float> * B = vectors + 1;
    const SparseVector<int, float> * C = vectors + 2;


    int num_threads = gridDim.x * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int start_C, end_C;
    
    start_C = mergepath_partitions[0*num_threads + tid];
    
    if(tid != (gridDim.x * blockDim.x-1)) {
        end_C = mergepath_partitions[0*num_threads + tid + 1];
    } else {
        end_C = C->nnz - 1;
    }

    int idx_C = start_C + 1;

    int idx_D = nnz_prefix[tid];

    while( (idx_C <= end_C)) {
        int ic0 = C->indices[idx_C];
        
        int i = ic0;
        
        int idx_A = locate(A, i);
        int idx_B = locate(B, i);
        if (idx_A != -1 && idx_B != -1) {
            D_indices[idx_D] = i;
            D_values[idx_D] = (A->values[idx_A] + B->values[idx_B]) * C->values[idx_C];
            idx_D++;
        } else if (idx_A != -1) {
            D_indices[idx_D] = i;
            D_values[idx_D] = A->values[idx_A] * C->values[idx_C];
            idx_D++;
        } else if (idx_B != -1) {
            D_indices[idx_D] = i;
            D_values[idx_D] = B->values[idx_B] * C->values[idx_C];
            idx_D++;
        }
        
        idx_C += (int32_t)(ic0 == i);
    }
    return;
}
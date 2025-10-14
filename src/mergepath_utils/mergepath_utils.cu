#include "mergepath_utils.h"

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

# define MAX_NUM_VECTORS 10



__global__ void kern_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const int total_size,
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work);


template<typename index_t, typename value_t>
__device__  void find_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const int total_size,
    const int count,
    const SparseVector<index_t, value_t> * vectors, 
    index_t * mergepath_partitions);


template <typename index_t, typename value_t>
__device__  void balanced_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const SparseVector<index_t, value_t> * vectors, 
    index_t *mergepath_partitions);

int * sparse_vectors_balanced_mergepath(
    const uint num_vectors,
    const int total_size,
    const SparseVector<int, float> * vectors, 
    const uint num_blocks,
    const uint threads_per_block,
    const uint per_thread_work
) {
    if(num_vectors>MAX_NUM_VECTORS){
        throw std::runtime_error("Mored than MAX_NUM_VECTORS");          
    }
    int num_threads = num_blocks * threads_per_block;
    

    int * mergepath_partitions;

    CHECK_CUDA(cudaMalloc(&mergepath_partitions, num_vectors * num_threads * sizeof(int)));


    kern_mergepath_partition_sparse_vectors<<<num_blocks, threads_per_block>>>(
        num_vectors,
        total_size,
        vectors,
        mergepath_partitions,
        per_thread_work
    );
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());
    
    return mergepath_partitions;

}


// kern_mergepath_partition_sparse_vectors finds the merge path partitions for each thread
// 
// params:
//   num_vectors : number of input vectors
//   total_size : total number of non-zeros across all sparse vectors
//   vectors : array of input sparse vectors
//   mergepath_partitions : (Must be preallocated) mergepath_partitions[i*num_threads+t] is the start of the partition for i^th tensor for the j^th thread
//                        boundary is exclusive, i.e. the first index that should be processed by that t^th thread is the index after mergepath_partitions[i*num_threads+t]
//   
//   per_thread_work :  number of non-zeros each thread should process (this could be +-1 due to balancing)
__global__ void kern_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const int total_size,
    const SparseVector<int, float> * vectors, 
    int * mergepath_partitions,
    const int per_thread_work) {
         
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int count = (  (tid) * per_thread_work );


    find_mergepath_partition_sparse_vectors(
        num_vectors,
        total_size,
        count,
        vectors,
        mergepath_partitions
    );
    //__syncthreads();
    //if(tid<30 && num_vectors==3)printf("tid %d : %d %d %d\n", tid, mergepath_partitions[tid],mergepath_partitions[gridDim.x * blockDim.x+tid], mergepath_partitions[2*gridDim.x * blockDim.x+tid]);

    balanced_mergepath_partition_sparse_vectors(
        num_vectors,
        vectors,
        mergepath_partitions
    );
    return;
}

template<typename index_t, typename value_t>
__device__  void find_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const int total_size,
    const int count,
    const SparseVector<index_t, value_t> * vectors, 
    index_t * mergepath_partitions) {

    int num_threads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
    
    if(count == 0) {
        for(int i=0; i<num_vectors; i++) {
            mergepath_partitions[i*num_threads+tid] = -1;
        }
        return;
    }
   
    if(count >= total_size) {
        for(int i=0; i<num_vectors; i++) {
            mergepath_partitions[i*num_threads+tid] = vectors[i].nnz - 1;
        }
        return;
    }

    // num_vectors = 1 is just dividing a single vector equally, added just for generality in implementation
    if (num_vectors ==1) {
        index_t index = min(count-1, vectors[0].nnz-1);
        mergepath_partitions[0*num_threads+tid] = index;
        return;
    }
    else if (num_vectors ==2) {
        index_t lo = -1;
        index_t hi = min(count-1, vectors[0].nnz-1);
        index_t a_index, b_index;

        // a_index + b_index == count-2,  as (indices are 0 indexed) (if count = 1, we want a_index = 0, b_index = -1 to be the output)
        while(lo < hi ) {
            a_index = (lo + hi+1)/2;
            b_index = count - 2 - a_index;
            
            if( b_index + 1> vectors[1].nnz-1) {
                lo = a_index;
                continue;
            }

            if(vectors[0].indices[a_index] > vectors[1].indices[b_index + 1]) {
                hi = a_index-1;
            } else {
                lo = a_index;
            }
        }
       
    
        a_index = lo;

        b_index = min(count - 2 - a_index, (int)(vectors[1].nnz-1)); // min behaving wierd when one value is -ve. porbably because it is expecting uints?
        
        mergepath_partitions[0*num_threads+tid] = a_index; mergepath_partitions[1*num_threads+tid] = b_index;
        return;
    }  else {
        // General case for num_tensors > 2
        
        index_t lo = -1;
        index_t hi = min(count-1, vectors[0].nnz-1);
        index_t a_index;

        
        while(lo < hi ) {
            a_index = (lo + hi+1)/2;
            
           
            int rem_count = count - (a_index + 1);
             
            find_mergepath_partition_sparse_vectors(
                num_vectors-1,
                total_size - vectors[0].nnz,
                rem_count,
                vectors + 1,
                mergepath_partitions + num_threads
            );
        
            bool a_too_big = false;
            for(int i=1; i<num_vectors; i++) {
                
                index_t b_index = mergepath_partitions[i*num_threads+tid];

                // if the next element after b_index is less than a_index that means , a_index is too big as we can instead include more elements from b
                // and thus we need to decrease a_index
                if(b_index +1 <= vectors[i].nnz -1 &&  vectors[0].indices[a_index] > vectors[i].indices[b_index+1]) {
                    a_too_big = true;
                    break;
                }
            }

            if(a_too_big) {
                hi = a_index-1;
            } else {
                lo = a_index;
            }
            
        }

        a_index = lo;
        mergepath_partitions[0*num_threads+tid] = a_index;
        find_mergepath_partition_sparse_vectors(
                num_vectors-1,
                total_size - vectors[0].nnz,
                count - (a_index + 1),
                vectors + 1,
                mergepath_partitions + num_threads
            );
    }
};


template <typename index_t, typename value_t>
__device__  void balanced_mergepath_partition_sparse_vectors(
    const uint num_vectors,
    const SparseVector<index_t, value_t> * vectors, 
    index_t * mergepath_partitions) {
    
    if (num_vectors ==1) {
        return;
    }
   

    int num_threads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
     //__syncthreads();
     //if(tid<30)printf("starting balancing %d\n", vectors[0].nnz);
    int permutation[MAX_NUM_VECTORS];

   
    
    for(int i=0; i<num_vectors; i++) {
        permutation[i] = i;
    }
     //__syncthreads();
     //if(tid<30)printf("starting sort %d\n", vectors[0].nnz);
    // sort indices and get permutation
     // Bubble sort - not efficient but n is small so should be fine (can implement better sort later if needed)
    for(int i=0; i<num_vectors; i++) {
        for(int j=0; j<num_vectors-i-1; j++) {
            index_t index_1 = mergepath_partitions[permutation[j]*num_threads+tid];
            index_t index_2 = mergepath_partitions[permutation[j+1]*num_threads+tid];
            
            if(index_2< 0) {
                int temp = permutation[j];
                permutation[j] = permutation[j+1];
                permutation[j+1] = temp;
            } else if (index_1>=0 && index_2>=0 && 
                vectors[permutation[j]].indices[index_1] >  vectors[permutation[j+1]].indices[index_2]) {

                int temp = permutation[j];
                permutation[j] = permutation[j+1];
                permutation[j+1] = temp;
            }
        }
    }

    //if(tid<30)printf("finsihed sort\n");
    // Now arr is sorted, we can balance the partitions
    for(int i=1; i<num_vectors; i++) {
        for(int j=0; j<i; j++) {
            int vector_i = permutation[i];
            int vector_j = permutation[j];
            index_t index_i = mergepath_partitions[vector_i*num_threads+tid];
            index_t index_j = mergepath_partitions[vector_j*num_threads+tid];

            if(index_i>=0 && index_j + 1 <= vectors[vector_j].nnz - 1 && vectors[vector_i].indices[index_i] == vectors[vector_j].indices[index_j + 1]) {
                // We can shift the boundary to the right for vector_j
                mergepath_partitions[vector_j*num_threads+tid] = index_j + 1;
               
            }
            
        }
    }
    
    return;
    }





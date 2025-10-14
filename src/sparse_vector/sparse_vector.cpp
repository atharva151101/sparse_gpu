#include "sparse_vector.hpp"

CVector<int32_t, float> nb_3dmergepath_test(CVector<int32_t, float> A, CVector<int32_t, float> B, CVector<int32_t, float> C, int num_fused) {

    //printf("Start mergetest\n");
    int * D_nnz = new int;
    
    int* D_indices;
    float* D_values;

    float * D_times = new float[3];

    sparse_vector_fusion_test(
        SparseVector<int32_t, float>(A.indices.data(), A.data.data(), A.size, A.indices.shape(0)),
        SparseVector<int32_t, float>(B.indices.data(), B.data.data(), B.size, B.indices.shape(0)),
        SparseVector<int32_t, float>(C.indices.data(), C.data.data(), C.size, C.indices.shape(0) ),
        D_indices, D_values, D_nnz, num_fused, D_times);

    //printf("%f %f %f\n", D_times[0], D_times[1], D_times[2]);
    CVector<int32_t, float> D = CVector<int32_t, float>(D_indices, D_values, A.size, *D_nnz, D_times[0], D_times[1], D_times[2]);
    //print_cuda(D_values,5);
    return D;
        
}

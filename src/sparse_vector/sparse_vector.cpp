#include "sparse_vector.hpp"

CVector<int32_t, float> nb_3dmergepath_test(CVector<int32_t, float> A, CVector<int32_t, float> B, CVector<int32_t, float> C, int num_fused) {

    
    int * D_nnz = new int;
    
    int* D_indices;
    float* D_values;

    sparse_vector_fusion_test(
        SparseVector<int32_t, float>(A.indices.data(), A.data.data(), A.size, A.indices.shape(0)),
        SparseVector<int32_t, float>(B.indices.data(), B.data.data(), B.size, B.indices.shape(0)),
        SparseVector<int32_t, float>(C.indices.data(), C.data.data(), C.size, C.indices.shape(0) ),
        D_indices, D_values, D_nnz, num_fused);

    CVector<int32_t, float> D = CVector<int32_t, float>(D_indices, D_values, A.size, *D_nnz);
    //print_cuda(D_values,5);
    return D;
        
}

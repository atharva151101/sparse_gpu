#pragma once

#include "sparse_vector.h"
#include "../cuda_utils/cuda_utils.h"
#include "../mergepath_utils/mergepath_utils.h"
#include "../nb_utils.hpp"

CVector<int32_t, float> nb_3dmergepath_test(CVector<int32_t, float> A, CVector<int32_t, float> B, CVector<int32_t, float> C, int num_fused);



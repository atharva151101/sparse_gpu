import nanobind_cuda_example
import torch
import random
from parser import parse_matrix, matrix_list 
from plotter import plot, load_and_plot, plot_3, plot_2, plot_bar_graph
import numpy as np


def benchmark_coo_add(start, end, save_and_plot = True):
    df = matrix_list()
    

    nnz = []
    manual_runtime = []
    pytorch_runtime = []

    failed = []

    skip = []
    for i in range(start+1,end):
        if i in skip :
           continue
       
        print(f"Starting iteration {i}")
        A = parse_matrix(df.iloc[i-1]['name'], True).coalesce()
        B = parse_matrix(df.iloc[i]['name'], True).coalesce()

        #print(A,B)
        M = max(A.size(0),B.size(0)) 
        N = max(A.size(1),B.size(1))

        A_torch = torch.sparse_coo_tensor(A.indices(), A.values(), (M,N)).coalesce()
        B_torch = torch.sparse_coo_tensor(B.indices(), B.values(), (M,N)).coalesce()

        A_COO = nanobind_cuda_example.COO(A.indices()[0], A.indices()[1], A.values(), torch.tensor([M,N], dtype=torch.int32))
        B_COO = nanobind_cuda_example.COO(B.indices()[0], B.indices()[1], B.values(), torch.tensor([M,N], dtype=torch.int32))

        nnzA = A_COO.data.numel()
        nnzB = B_COO.data.numel()

        print(f"M {M}, N {N}, nnz {nnzA+nnzB}")

        C_pytorch, pytorch = coo_add(A_torch, B_torch, True)
        C_manual, manual = coo_add(A_COO, B_COO, False)
    
        
        ans = True

        ans = ans and torch.equal(C_pytorch.indices()[0],C_manual.row)
        ans = ans and torch.equal(C_pytorch.indices()[1],C_manual.col)
        ans = ans and torch.equal(C_pytorch.values(),C_manual.data)

        if not ans:
           print(f"FAILED at {i} for matrix {df.iloc[i-1]['name']} {df.iloc[i]['name']}")
           failed.append(i)
           diff_mask_1 = C_pytorch.indices()[0] != C_manual.row
           diff_mask_2 = C_pytorch.indices()[1] != C_manual.col
           diff_mask_3 = C_pytorch.values() != C_manual.data
           idx = -1
           if diff_mask_1.any():
             idx = torch.nonzero(diff_mask_1, as_tuple=True)[0][0].item()
           elif diff_mask_2.any():
             idx = torch.nonzero(diff_mask_2, as_tuple=True)[0][0].item()
           elif diff_mask_3.any():
             idx = torch.nonzero(diff_mask_3, as_tuple=True)[0][0].item()
           print(f"Mismatch at Index: {idx} \n rows: manual {C_manual.row[idx]} , result {C_pytorch.indices()[0][idx]}")
           print(f"cols: manual {C_manual.col[idx]} , result {C_pytorch.indices()[1][idx]}")
           print(f"values: manual {C_manual.data[idx]} , result {C_pytorch.values()[idx]}")

           print(A_COO.row[:10], B_COO.row[:10])
           print(A_COO.col[:10], B_COO.col[:10])
           print(A_COO.data[:10], B_COO.data[:10])

           print(C_manual.row[:10], C_pytorch.indices()[0][:10])

           print(C_manual.col[:10], C_pytorch.indices()[1][:10])

           print(C_manual.data[:10], C_pytorch.values()[:10])
           return
       
        nnz.append(nnzA+nnzB)
        manual_runtime.append(manual)
        #cusparse_runtime.append(cusparse)
        pytorch_runtime.append(pytorch)

    if save_and_plot:
        plot(nnz, manual_runtime, [], pytorch_runtime, f"coo_rows_nnz_{start}-{end}")
    
    print(failed)

def benchmark_csr_add(start,end, save_and_plot=True):
    df = matrix_list()
    

    nnz = []
    manual_runtime = []
    cusparse_runtime = []
    pytorch_runtime = []

    failed = []

    skip = [611]
    for i in range(start+1,end):
        if i in skip :
           continue
       
        print(f"Starting iteration {i}")
        A = parse_matrix(df.iloc[i-1]['name'])
        B = parse_matrix(df.iloc[i]['name'])
       
        M = min(A.size(0),B.size(0)) 
        N = max(A.size(1),B.size(1))


       #print(A,B)
        A_torch = torch.sparse_csr_tensor(A.crow_indices()[:M+1], A.col_indices()[:A.crow_indices()[M]], A.values()[:A.crow_indices()[M]], (M,N))
        B_torch = torch.sparse_csr_tensor(B.crow_indices()[:M+1], B.col_indices()[:B.crow_indices()[M]], B.values()[:B.crow_indices()[M]], (M,N))

        plus_row = A_torch.crow_indices() + B_torch.crow_indices()


        A_CSR = nanobind_cuda_example.CSR(A_torch.crow_indices(), A_torch.col_indices(), A_torch.values(), torch.tensor([M,N], dtype=torch.int32))
        B_CSR = nanobind_cuda_example.CSR(B_torch.crow_indices(), B_torch.col_indices(), B_torch.values(), torch.tensor([M,N], dtype=torch.int32))


        nnzA = A_CSR.data.numel()
        nnzB = B_CSR.data.numel()
        print(f"M {M}, N {N}, nnz {plus_row.max()}")

        
        C_pytorch, pytorch = torch_add(A_torch, B_torch)
        C_cusparse, cusparse = csr_add(A_CSR, B_CSR, True)
        C_manual, manual = csr_add(A_CSR, B_CSR, False)
        

        ans = True

        ans = ans and torch.equal(C_cusparse.indptr,C_manual.indptr )
        ans = ans and torch.equal(C_cusparse.indices,C_manual.indices )
        ans = ans and torch.equal(C_cusparse.data,C_manual.data )

        ans = ans and torch.equal(C_pytorch.crow_indices(),C_manual.indptr )
        ans = ans and torch.equal(C_pytorch.col_indices(),C_manual.indices )
        ans = ans and torch.equal(C_pytorch.values(),C_manual.data )

        if not ans:
           print(f"FAILED at {i} for matrix {df.iloc[i-1]['name']} {df.iloc[i]['name']}")
           failed.append(i)
           print(A_CSR.indptr[:10], B_CSR.indptr[:10])
           print(A_CSR.indices[:10], B_CSR.indices[:10])
           #C_expected =(A.to_dense() + B.to_dense()).to_sparse_csr()
           print(C_cusparse.indptr, C_manual.indptr)#, C_expected.crow_indices())
           print(C_cusparse.indices, C_manual.indices)
           print(torch.equal(C_cusparse.indptr,C_manual.indptr ))

           failure_reason(C_manual, C_cusparse)
       
        nnz.append(plus_row.max().item())
        manual_runtime.append(manual)
        cusparse_runtime.append(cusparse)
        pytorch_runtime.append(pytorch)

    if save_and_plot:
        plot(nnz, manual_runtime, cusparse_runtime, pytorch_runtime, f"torch_nnz_{start}-{end}")
    
    print(failed)

   
    


def csr_add(A_CSR, B_CSR, use_cusparse):
    torch.cuda.synchronize()
    iter = 14
    times =[]
    for i in range(iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)
        start.record()
    
        C = nanobind_cuda_example.gpu_csr_add_f32(
            A_CSR,
            B_CSR,
            use_cusparse,
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    arr = np.array(times)

    trimmed = np.sort(arr)[2:-2]
    avg = trimmed.mean()

    return C, avg

def coo_add(A_COO, B_COO, use_pytorch):
    torch.cuda.synchronize()
    iter = 14
    times =[]
    for i in range(iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)
        start.record()

        if use_pytorch:
            C = (A_COO + B_COO).coalesce()

        else:
            C = nanobind_cuda_example.gpu_coo_add_f32(
                A_COO,
                B_COO,
            )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    arr = np.array(times)

    trimmed = np.sort(arr)[2:-2]
    avg = trimmed.mean()

    return C, avg

def torch_add(A,B):
    torch.cuda.synchronize()
    iter = 14
    times =[]
    for i in range(iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)
        start.record()
    
        C = A + B
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    arr = np.array(times)

    trimmed = np.sort(arr)[2:-2]
    avg = trimmed.mean()

    return C, avg



def failure_reason(C_manual, C_result):

    if torch.equal(C_manual.indptr, C_result.indptr) is False:
        diff_mask = C_manual.indptr != C_result.indptr
        if diff_mask.any():
            idx = torch.nonzero(diff_mask, as_tuple=True)[0][0].item()
            print(f"Mismatch at index: {idx} \n indptr: manual {C_manual.indptr[idx]} , result {C_result.indptr[idx]}")

    if torch.equal(C_manual.data, C_result.data) is False:
        if torch.equal(C_manual.indices, C_result.indices) is False:
            diff_mask = C_manual.indices != C_result.indices
        diff_mask = C_manual.data != C_result.data
        if diff_mask.any():
            idx = torch.nonzero(diff_mask, as_tuple=True)[0][0].item()
            print(f"Mismatch at Index: {idx} \n indices: manual {C_manual.indices[idx]} , result {C_result.indices[idx]}")
            print(f"values: manual {C_manual.data[idx]} , result {C_result.data[idx]}")

def test_mergepath(size, length):

    # Generate random sparse vectors A, B, C
    print(length)
    a_length = int(length)#int(size * sparsity_a)
    b_length = int(length)#int(size * sparsity_b)
    c_length = int(length)#int(size * sparsity_c)
    print(a_length, b_length, c_length)
    a_index = random.sample(range(int(size)), a_length)
    a_index.sort()

    b_index = random.sample(range(int(size)), b_length)
    b_index.sort()

    c_index = random.sample(range(int(size)), c_length)
    c_index.sort()

    A_index = torch.tensor(a_index, dtype=torch.int32, device=torch.device("cuda"))
    A_data = torch.tensor([random.uniform(1, 10) for _ in range(a_length)], dtype=torch.float32, device=torch.device("cuda"))

    B_index = torch.tensor(b_index, dtype=torch.int32, device=torch.device("cuda"))
    B_data = torch.tensor([random.uniform(1, 10) for _ in range(b_length)], dtype=torch.float32, device=torch.device("cuda"))       

    C_index = torch.tensor(c_index, dtype=torch.int32, device=torch.device("cuda"))
    C_data = torch.tensor([random.uniform(1, 10) for _ in range(c_length)], dtype=torch.float32, device=torch.device("cuda"))

    # A_index = torch.tensor([0,1,5, 7, 12, 13, 20, 21], dtype=torch.int32, device=torch.device("cuda"))
    # A_data = torch.tensor([3.0]*A_index.shape[0], dtype=torch.float32, device=torch.device("cuda"))
    
    # B_index = torch.tensor([0,2,4,5 ,6,7,8,20,22,23], dtype=torch.int32, device=torch.device("cuda"))
    # B_data = torch.tensor([2.0]*B_index.shape[0], dtype=torch.float32, device=torch.device("cuda"))

    # C_index =  torch.tensor([0,1,3,7,8,20,21,22,23], dtype=torch.int32, device=torch.device("cuda"))
    # C_data = torch.tensor([1.0]*C_index.shape[0], dtype=torch.float32, device=torch.device("cuda"))

    # print("A:", A_index, A_data)
    # print("B:", B_index, B_data)
    # print("C:", C_index, C_data)
          
    A = nanobind_cuda_example.CVector(
        A_index, 
        A_data,
        size)
    
    B = nanobind_cuda_example.CVector(
        B_index,
        B_data,
        size
        )
    
    C = nanobind_cuda_example.CVector(
        C_index,
        C_data,
        size
        )

    print("Starting Benchmarks")

    iter = 14
    torch.cuda.synchronize()

    iter = 14
    times =[]
    for i in range(iter):
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True, blocking=True)
        #start.record()
        
        D_full_fusion = nanobind_cuda_example.gpu_sss_mergepath_test(A,B,C, 3)
        #end.record()
        #torch.cuda.synchronize()
        times.append([D_full_fusion.time_1,D_full_fusion.time_2, D_full_fusion.time_3])
    # print(D_full_fusion.indices, D_full_fusion.data)
    arr = np.array(times)

    result = []
    for i in range(arr.shape[1]):  # iterate over 3 columns
        vals = np.sort(arr[:, i])
        trimmed = vals[2: -2]  # remove lowest and highest `trim` values
        mean_val = trimmed.mean()
        result.append(mean_val)

    full_lb = result
    print("-----------------full done-----------------------")
    torch.cuda.synchronize()

    times =[]
    for i in range(iter):
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True, blocking=True)
        #start.record()
        D_partial_fusion = nanobind_cuda_example.gpu_sss_mergepath_test(A,B,C, 2)
        #end.record()
        #torch.cuda.synchronize()
        #times.append(start.elapsed_time(end))
        times.append([ D_partial_fusion.time_1, D_partial_fusion.time_2,  D_partial_fusion.time_3 ])

    arr = np.array(times)

    result = []
    for i in range(arr.shape[1]):  # iterate over 3 columns
        vals = np.sort(arr[:, i])
        trimmed = vals[2: -2]  # remove lowest and highest `trim` values
        mean_val = trimmed.mean()
        result.append(mean_val)
    partial_lb = result
    # print(D_partial_fusion.indices, D_partial_fusion.data)
    print("-----------------partial done------------------------")

    torch.cuda.synchronize()

    times =[]
    for i in range(iter):
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True, blocking=True)
        #start.record()
        D_nofusion = nanobind_cuda_example.gpu_sss_mergepath_test(A,B,C, 1)
        #end.record()
        #torch.cuda.synchronize()
        #times.append(start.elapsed_time(end))
        times.append([ D_partial_fusion.time_1, D_partial_fusion.time_2,  D_partial_fusion.time_3 ])
    arr = np.array(times)

    result = []
    for i in range(arr.shape[1]):  # iterate over 3 columns
        vals = np.sort(arr[:, i])
        trimmed = vals[2: -2]  # remove lowest and highest `trim` values
        mean_val = trimmed.mean()
        result.append(mean_val)
    nofusion_lb = result
    # print(D_nofusion.indices, D_nofusion.data)
    print("------------------no done-----------------------")

    ans = torch.equal(D_full_fusion.indices, D_partial_fusion.indices)
    ans = ans and torch.equal(D_full_fusion.data, D_partial_fusion.data)
    ans = ans and torch.equal(D_full_fusion.indices, D_nofusion.indices)
    ans = ans and torch.equal(D_full_fusion.data, D_nofusion.data)

    # print(D_full_fusion.data != D_partial_fusion.data)
    # print(D_full_fusion.data != D_nofusion.data)
    # print(D_nofusion.data != D_partial_fusion.data)


    return ans, full_lb, partial_lb, nofusion_lb
    

   




if __name__ == "__main__":
    
    seed = 42517
    torch.manual_seed(seed)
    random.seed(seed)
    # benchmark_csr_add(0,100)
    #print(matrix_list())
    size = 10000000
    length = np.logspace(2.5, 6.5, num=10)
    #print(length)
    full_time = []
    partial_time = []
    no_time = []
    skip = []
    failed = []

    pts = 10
    for i in range(pts):
        if i in skip:
            continue
        
        print(f"iteration {i}")
        ans, full, partial, no = test_mergepath(int(size), length[i])
        if not ans:
            failed.append(i)

        print( ans, full, partial, no)
        full_time.append(full)
        partial_time.append(partial)
        no_time.append(no)
    #print(full_time, partial_time, no_time)
    plot_bar_graph(size, length[:pts],full_time, partial_time, no_time, "bar_graph")
    # plot_2(size, length[:pts], full_time, partial_time, no_time, "mergepath_test_5")
    # print(failed)
    #benchmark_csr_add(0,100)
    #benchmark_coo_add(0,100)
    
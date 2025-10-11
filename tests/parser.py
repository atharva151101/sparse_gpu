import nanobind_cuda_example
import pandas as pd
import torch
import os

def matrix_list():
    df = pd.read_csv("tests/suitesparse_stats.csv")

    df.columns = ["name", "nnz", "percent_nnz", "total_elements", "rows", "columns"]

    # Convert numeric columns to integers/floats just in case they're strings
    df["nnz"] = df["nnz"].astype(int)
    df["total_elements"] = df["total_elements"].astype(int)
    df["rows"] = df["rows"].astype(int)
    df["columns"] = df["columns"].astype(int)
    df["percent_nnz"] = df["percent_nnz"].astype(float)

    matrix_dir = "/scratch/suitesparse/"

    df = df[df["name"].apply(lambda f: os.path.exists(os.path.join(matrix_dir, f.split('.')[0]+"/"+f)))]



    df_sorted = df.sort_values(by=["rows","columns"], ascending=[True,True])
    return df_sorted

def parse_matrix(matrix, return_coo = False):
    file = "/scratch/suitesparse/"

    matrix = file+matrix
    COO = nanobind_cuda_example.parse2D(matrix)

    row = COO.row.to(dtype=torch.long, device="cuda")
    col = COO.col.to(dtype=torch.long, device="cuda")
    indices = torch.stack([row, col], dim=0)
    values = COO.data.to(dtype=torch.float32, device="cuda")
    COO = torch.sparse_coo_tensor(indices,values,(COO.N, COO.M))
    if return_coo:
        return COO
    CSR = COO.to_sparse_csr()

    return CSR

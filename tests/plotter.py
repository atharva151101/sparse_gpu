import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def load_and_plot(file_names, plot_name):
    nnz = []
    cusparse = []
    manual = []
    pytorch = []
    dir = "benchmark_results/"
    for file in file_names:
        data = np.load(dir + file + ".npz", allow_pickle=True)
        nnz_manual = data["nnz"].tolist()
        runtime_manual = data["manual"].tolist()

        runtime_cusparse = data["cusparse"].tolist()
        runtime_pytorch = data["pytorch"].tolist()

        nnz = nnz + nnz_manual
        cusparse = cusparse + runtime_cusparse
        manual = manual + runtime_manual
        pytorch = pytorch + runtime_pytorch

    plot(nnz, manual, cusparse, pytorch, plot_name)

    
def plot(nnz, manual, cusparse, pytorch, name):
    dir = "benchmark_results/"
    np.savez(dir + name +".npz", nnz=nnz, manual=manual,
                             cusparse=cusparse, pytorch=pytorch)
    plt.figure(figsize=(8,6))

    # Manual implementation
    plt.scatter(nnz,manual, label="Manual", alpha=0.7, color="blue", marker="o", s=5)

    # Library implementation
    if len(cusparse)!=0:
        plt.scatter(nnz,cusparse, label="cusparse", alpha=0.7, color="red", marker="o", s=5)
    if len(pytorch)!=0:
        plt.scatter(nnz,pytorch, label="pytorch", alpha=0.7, color="green", marker="o", s=5)
    plt.xscale("log")
    plt.yscale("log")

    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    # Labels and formatting
    plt.xlabel("Total Non-Zeros (nnzA + nnzB)")
    plt.ylabel("Runtime (ms)")
    plt.title("Runtime vs Non-Zeros")
    plt.legend()
    plt.grid(True)

    # Save to file (PNG)
    plt.savefig(dir+name+".png", dpi=300, bbox_inches="tight")

    plt.show()

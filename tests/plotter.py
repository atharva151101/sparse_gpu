import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
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

def plot_3(nnz, lengths, full, partial, no, name):
    dir = "benchmark_results/"
    np.savez(dir + name +".npz", nnz=nnz, lengths=lengths, full=full,
                             partial=partial, no=no)
    plt.figure(figsize=(8,6))

    # Manual implementation
    plt.scatter(lengths/nnz,full, label="LB on (a,b,c)", alpha=0.7, color="blue", marker="o", s=5)

    # Library implementation
    
    plt.scatter(lengths/nnz, partial, label="LB on (a,b)", alpha=0.7, color="red", marker="o", s=5)
    plt.scatter(lengths/nnz, no, label="LB on (c)", alpha=0.7, color="green", marker="o", s=5)
    
    plt.xscale("log")
    plt.yscale("log")

    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    # Labels and formatting
    plt.xlabel("Size |a|=|b|=|c|")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Fixed %nnz , varying size , |a|=|b|=|c| (%nnz = {nnz})")
    plt.legend()
    plt.grid(True)

    # Save to file (PNG)
    plt.savefig(dir+name+".png", dpi=300, bbox_inches="tight")

    plt.show()

def plot_2(size, lengths, full, partial, no, name):
    dir = "benchmark_results/"
    np.savez(dir + name +".npz", size=size, lengths=lengths, full=full,
                             partial=partial, no=no)
    plt.figure(figsize=(8,6))

    # Manual implementation
    plt.scatter(lengths/size,full, label="LB on (a,b,c)", alpha=0.7, color="blue", marker="o", s=5)

    # Library implementation
    
    plt.scatter(lengths/size, partial, label="LB on (a,b)", alpha=0.7, color="red", marker="o", s=5)
    plt.scatter(lengths/size, no, label="LB on (c)", alpha=0.7, color="green", marker="o", s=5)
    
    plt.xscale("log")
    plt.yscale("log")

    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    # Labels and formatting
    plt.xlabel("%nnz (|a|=|b|=100*|c|)")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Fixed size , varying sparsity , |a|=|b|=100*|c| (Size = 10^8)")
    plt.legend()
    plt.grid(True)

    # Save to file (PNG)
    plt.savefig(dir+name+".png", dpi=300, bbox_inches="tight")

    plt.show()

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

def plot_bar_graph(size, lengths, full_lb, partial_lb, single_lb, name):
    dir = "benchmark_results/"

    # Plot grouped bars

    program_names = ['l.b (a,b,c)', 'l.b (a,b)', 'l.b (c)']
    stage_names = ['Merge-path', 'pre-compute', 'compute']
    color_1 = ['#c6dbef', '#6baed6', '#2171b5']
    color_2 = ['#fdd0a2','#fd8d3c','#d94801']
    color_3 = ['#c7e9c0','#74c476' ,'#238b45']
   
    for i in range(len(lengths)):
        plt.bar(lengths[i], full_lb[i][0],width=0.2*lengths[i], color=color_1[0])
        plt.bar(lengths[i], full_lb[i][1],width=0.2*lengths[i], bottom=full_lb[i][0], color=color_1[1])
        plt.bar(lengths[i], full_lb[i][2],width=0.2*lengths[i], bottom=full_lb[i][1] + full_lb[i][0], color=color_1[2])

    for i in range(len(lengths)):
        plt.bar(0.8*lengths[i], partial_lb[i][0],width=0.2*lengths[i], label=stage_names[0], color=color_2[0])
        plt.bar(0.8*lengths[i], partial_lb[i][1],width=0.2*lengths[i], bottom=partial_lb[i][0], color=color_2[1])
        plt.bar(0.8*lengths[i], partial_lb[i][2],width=0.2*lengths[i], bottom=partial_lb[i][1] + partial_lb[i][0], color=color_2[2])

    for i in range(len(lengths)):
        plt.bar(1.2*lengths[i], single_lb[i][0],width=0.2*lengths[i],color=color_3[0])
        plt.bar(1.2*lengths[i], single_lb[i][1],width=0.2*lengths[i], bottom=single_lb[i][0],  color=color_3[1])
        plt.bar(1.2*lengths[i], single_lb[i][2],width=0.2*lengths[i], bottom=single_lb[i][1] + single_lb[i][0],  color=color_3[2])

    # Set labels and title
    plt.xlabel("length |a|=|b|=|c|")
    plt.ylabel("Runtime (ms)")
    plt.title("Fixed size , varying sparsity , Size = 10^7 ")

     # representative (middle shade of each program)
    legend_patches = []
    legend_patches+=[Patch(color=c, label="l.b (a,b,c)  "+s) for c, s in zip(color_1, stage_names)]
    legend_patches+=[Patch(color=c, label="l.b (a,b)  "+s) for c, s in zip(color_2, stage_names)]
    legend_patches+=[Patch(color=c, label="l.b (c)  "+s) for c, s in zip(color_3, stage_names)]
    # Tick labels correspond to sizes
    #plt.xticks(range(len(lengths)), lengths/size)

    # Set both axes to log scale
    plt.yscale("log")
    plt.xscale("log")

    # Add legend and grid
    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))

    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    plt.legend(handles=legend_patches, title="Stages")
    plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
    plt.savefig(dir+name+".png", dpi=300, bbox_inches='tight')
import sys
from time import time
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.scan import InclusiveScanKernel
from pycuda.reduction import ReductionKernel

InteractiveShell.ast_node_interactivity = "all"
print(f'The version of PyCUDA: {pycuda.VERSION}')
print(f'The version of Python: {sys.version}')

seq = np.array([1, 2, 3, 4], dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
sum_gpu = InclusiveScanKernel(np.int32, "a+b")
print(sum_gpu(seq_gpu).get())

seq = np.array([1,100,-3,-10000, 4, 10000, 66, 14, 21], dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
max_gpu = InclusiveScanKernel(np.int32, "a > b ? a : b")
seq_max_bubble = max_gpu(seq_gpu)
print(seq_max_bubble)
print(seq_max_bubble.get()[-1])
print(np.max(seq))


a_host = np.array([1, 2, 3], dtype=np.float32)
b_host = np.array([4, 5, 6], dtype=np.float32)
print(a_host.dot(b_host))
dot_prod = ReductionKernel(np.float32, neutral="0", reduce_expr="a+b",
                           map_expr="x[i]*y[i]", arguments="float *x, float *y")
a_device = gpuarray.to_gpu(a_host)
b_device = gpuarray.to_gpu(b_host)
print(dot_prod(a_device, b_device).get())

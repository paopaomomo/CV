import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy
import numpy as np


# 使用核函数
# threadIdx:计算线程
# 第n号线程将x[n]与y[n]相加后存入z[n]。
mod = SourceModule(r"""
void __global__ add(const float *x, const float *y, float *z)
{
    const int n = threadIdx.x;
    z[n] = x[n] + y[n];
}
""")

add = mod.get_function("add")
num = 4
A = numpy.random.rand(num)
B = numpy.random.rand(num)
C = numpy.zeros(num)
A_GPU = gpuarray.to_gpu(A.astype(numpy.float32))
B_GPU = gpuarray.to_gpu(B.astype(numpy.float32))
C_GPU = gpuarray.to_gpu(B.astype(numpy.float32))
# thread=num,block=(4,1,1),grid=1,
# block(4,1,1): 几个线程？
# block(3,2,2): 几个线程？

add(A_GPU, B_GPU, C_GPU, block=(num,1,1))
# get():把数据从gpu拿下来
C = C_GPU.get()
print('A=', A)
print('B=', B)
print('C=', C)
import pdb
pdb.set_trace()

from pycuda.scan import InclusiveScanKernel
from pycuda.reduction import ReductionKernel

#seq_gpu = gpuarray.to_gpu(seq)
sum_gpu = InclusiveScanKernel(np.int32, "a+b")
#print(sum_gpu(seq_gpu).get())
print(sum_gpu(C_GPU).get())


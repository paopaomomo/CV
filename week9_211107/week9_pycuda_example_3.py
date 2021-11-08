import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy
mod = SourceModule(r"""
void __global__ add(const float *x, const float *y, float *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}
void __global__ mul(const float *x, const float *y, float *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] * y[n];
}
""")
add = mod.get_function("add")
mul = mod.get_function("mul")
num = 6
# 一个bolck计算不完
A = numpy.random.rand(num)
B = numpy.random.rand(num)
C = numpy.zeros(num)
A_GPU = gpuarray.to_gpu(A.astype(numpy.float32))
B_GPU = gpuarray.to_gpu(B.astype(numpy.float32))
C_GPU = gpuarray.to_gpu(B.astype(numpy.float32))

# 
add(A_GPU, B_GPU, C_GPU, grid=(2,), block=(4,1,1))
C = C_GPU.get()
print('A=', A)
print('B=', B)
print('C=', C)
print("*"*20)
mul(A_GPU, B_GPU, C_GPU, grid=(2,), block=(4,1,1))
C = C_GPU.get()
print('A=', A)
print('B=', B)
print('C=', C)


# 线程索引:写好核函数


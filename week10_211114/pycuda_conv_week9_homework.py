import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule('''
__global__ void convolutionGPU(float *output, float *input, float *filter, int filter_radius, int input_width, int input_height)
{
  const int filter_width = 2 * filter_radius + 1;
  int col = threadIdx.y + blockDim.y * blockIdx.y;
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  int idx = row + input_width*col;

  float sum = 0; 
  float value = 0;
  for(int i = -filter_radius; i <= filter_radius; i++)
    for(int j = -filter_radius; j <= filter_radius; j++){
      if( (col+j) < 0 ||(row+i) < 0 ||(row+i) > (input_width-1) ||(col+j) > (input_height-1) )
          value = 0;
      else        
          value = input[idx + i + j * input_width];
          sum += value * filter[(i+filter_radius) + (j+filter_radius)*filter_width];
    }
    output[idx] = sum;
}
''')

convolutionGPU = mod.get_function("convolutionGPU")


def convolution_cuda(input, filter):
    filter_radius = filter.shape[0] // 2
    output = input.copy()
    (input_height, input_width) = output.shape
    filter = np.float32(filter)
    input_height = np.int32(input_height)
    input_width = np.int32(input_width)
    filter_radius = np.int32(filter_radius)

    input_gpu = cuda.mem_alloc_like(input)
    filter_gpu = cuda.mem_alloc_like(filter)
    output_gpu = cuda.mem_alloc_like(input)
    # host to device
    cuda.memcpy_htod(input_gpu, input)
    cuda.memcpy_htod(filter_gpu, filter)

    print('start')
    convolutionGPU(output_gpu, input_gpu,
                 filter_gpu, filter_radius,
                 input_width, input_height,
                 block=(5, 1, 1), grid=(1, 5))

    cuda.memcpy_dtoh(output, output_gpu)
    return output


def test_convolution_cuda():
    # Test the convolution kernel.
    # Generate or load a test image
    input = np.array(
        [[1, 1, 1, 0, 0],
         [0, 1, 1, 1, 0],
         [0, 0, 1, 1, 1],
         [0, 0, 1, 1, 0],
         [0, 1, 1, 0, 0]
         ])
    input = np.float32(input)
    print("input:")
    print(input)

    filter = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [1, 0, 1]
                       ])

    output = input.copy()
    # not a number: nan
    output[:] = np.nan
    output = convolution_cuda(input, filter)
    print('Done running the convolution kernel!')
    print("output")
    print(output)


if __name__ == '__main__':
    test_convolution_cuda()

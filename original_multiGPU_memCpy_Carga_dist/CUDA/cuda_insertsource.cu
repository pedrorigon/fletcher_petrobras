#include "cuda_defines.h"
#include "../driver.h"
#include "cuda_insertsource.h"

__global__ void kernel_InsertSource(const float val, const int iSource,
	                            float * restrict qp, float * restrict qc)
{
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  if (ix==0)
  {
    qp[iSource]+=val;
    qc[iSource]+=val;
  }
}


void CUDA_InsertSource(const float val, const int iSource, float * restrict pc, float * restrict qc,  float * restrict pp, float * restrict qp)
{

  extern float* dev_pp[GPU_NUMBER];
  extern float* dev_pc[GPU_NUMBER];
  extern float* dev_qp[GPU_NUMBER];
  extern float* dev_qc[GPU_NUMBER];

  int num_gpus;
  CUDA_CALL(cudaGetDeviceCount(&num_gpus));
  for (int gpu = 0; gpu < 2; gpu++)
    {
        cudaDeviceProp prop;
        cudaSetDevice(gpu);
        CUDA_CALL(cudaGetDeviceProperties(&prop, gpu));
        if ((dev_pp[gpu]) && (dev_qp[gpu]))
        {
          dim3 threadsPerBlock(BSIZE_X, 1);
          dim3 numBlocks(1,1);
          kernel_InsertSource<<<numBlocks, threadsPerBlock>>> (val, iSource, dev_pc[gpu], dev_qc[gpu]);
          CUDA_CALL(cudaGetLastError());
          CUDA_CALL(cudaDeviceSynchronize());
        }
    }
}

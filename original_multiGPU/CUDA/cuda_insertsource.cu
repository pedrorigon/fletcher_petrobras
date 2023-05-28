#include "cuda_defines.h"
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

  if ((pp) && (qp))
  {
     dim3 threadsPerBlock(BSIZE_X, 1);
     dim3 numBlocks(1,1);
     kernel_InsertSource<<<numBlocks, threadsPerBlock>>> (val, iSource, pc, qc);
     CUDA_CALL(cudaGetLastError());
     CUDA_CALL(cudaDeviceSynchronize());
  }
}

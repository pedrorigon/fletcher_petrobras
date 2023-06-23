#include "cuda_defines.h"
#include "../driver.h"
#include "../map.h"
#include "cuda_insertsource.h"

__global__ void kernel_InsertSource(const float val, const int iSource,
	                            float * restrict qp, float * restrict qc, int offset, int fix_size)
{
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  if (ix==0)
  {
    qp[iSource - fix_size + offset]+=val;
    qc[iSource - fix_size + offset]+=val;
  }
}


void CUDA_InsertSource(const int sx, const int sy, const int sz, const float val, const int iSource, float * restrict pc, float * restrict qc,  float * restrict pp, float * restrict qp)
{

  extern float* dev_pp[GPU_NUMBER];
  extern float* dev_pc[GPU_NUMBER];
  extern float* dev_qp[GPU_NUMBER];
  extern float* dev_qc[GPU_NUMBER];

  int num_gpus;
  int offset;
  int fix_size;
  CUDA_CALL(cudaGetDeviceCount(&num_gpus));
  printf("Teste se entrou aqui \n");
  for (int gpu = 0; gpu < 2; gpu++)
    {
        cudaDeviceProp prop;
        cudaSetDevice(gpu);

        if (gpu == 0)
        {
            offset = 0;
            fix_size = 0;
        }
        else
        {
            offset = (ind(0,0,(sz/2)) - ind(0,0,(sz/2-4)));
            fix_size = ind(0,0,(sz/2));
        }

        printf("offset = %d - fix_size = %d - device = %d \n", offset, fix_size, gpu);


        if ((dev_pp[gpu]) && (dev_qp[gpu]))
        {
          dim3 threadsPerBlock(BSIZE_X, 1);
          dim3 numBlocks(1,1);
          kernel_InsertSource<<<numBlocks, threadsPerBlock>>> (val, iSource, dev_pc[gpu], dev_qc[gpu], offset, fix_size);
          CUDA_CALL(cudaGetLastError());
          CUDA_CALL(cudaDeviceSynchronize());
        }
    }
}

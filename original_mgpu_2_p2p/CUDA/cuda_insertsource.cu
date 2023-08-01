#include "cuda_defines.h"
#include "../driver.h"
#include "cuda_insertsource.h"

__global__ void kernel_InsertSource(const float val, const int iSource,
	                            float * restrict qp, float * restrict qc, int fix_position, int offset)
{
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  if (ix==0)
  {
    qp[iSource - fix_position + offset]+=val;
    qc[iSource - fix_position + offset]+=val;
  }
}


void CUDA_InsertSource(const float val, const int iSource, float * restrict pc, float * restrict qc,  float * restrict pp, float * restrict qp)
{
  extern Gpu gpu_map[GPU_NUMBER];
  extern float* dev_pp[GPU_NUMBER];
  extern float* dev_pc[GPU_NUMBER];
  extern float* dev_qp[GPU_NUMBER];
  extern float* dev_qc[GPU_NUMBER];

  int num_gpus;
  int teste;
  int offset = 0;
  int fix_position = 0;
  CUDA_CALL(cudaGetDeviceCount(&num_gpus));
  for (int gpu = 0; gpu < 2; gpu++)
    {
        cudaDeviceProp prop;
        cudaSetDevice(gpu);
        CUDA_CALL(cudaGetDeviceProperties(&prop, gpu));
        if ((dev_pp[gpu]) && (dev_qp[gpu]))
        {
          if(gpu != 0){
            fix_position = gpu_map[0].cpu_end_pointer;
            offset = gpu_map[1].gpu_start_pointer;
            teste = iSource - fix_position + offset;
          }else{
            teste = iSource - fix_position + offset;
          }
          dim3 threadsPerBlock(BSIZE_X, 1);
          dim3 numBlocks(1,1);
          kernel_InsertSource<<<numBlocks, threadsPerBlock>>> (val, iSource, dev_pc[gpu], dev_qc[gpu], fix_position, offset);
        }
    }
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
}

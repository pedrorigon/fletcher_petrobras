#include "cuda_defines.h"
#include "../driver.h"
#include "cuda_insertsource.h"

extern int number_gpu;

__global__ void kernel_InsertSource(const float val, const int iSource,
	                            float * restrict qp, float * restrict qc, int fix_position, int offset)
{
  const int ix=blockIdx.x * blockDim.x + threadIdx.x;
  if (ix==0)
  {
    qp[offset]+=val;
    qc[offset]+=val;
  }
}


void CUDA_InsertSource(const float val, const int iSource, float * restrict pc, float * restrict qc,  float * restrict pp, float * restrict qp)
{
  extern Gpu* gpu_map;
  extern float** dev_pp;
  extern float** dev_pc;
  extern float** dev_qp;
  extern float** dev_qc;

  int gpu_mid;
  int teste;
  int offset = 0;
  int fix_position = 0;
  //CUDA_CALL(cudaGetDeviceCount(&num_gpus));
  for (int gpu = 0; gpu < number_gpu; gpu++)
    {
        cudaDeviceProp prop;
        cudaSetDevice(gpu);
        if ((dev_pp[gpu]) && (dev_qp[gpu]))
        {
          gpu_mid = number_gpu / 2;
          if (gpu == gpu_mid){
            offset = gpu_map[2].center_position;
          } else if (gpu == (gpu_mid - 1)){
            offset = gpu_map[1].center_position;
          } else{
            offset = -1;
          }
          if(offset != -1){
            dim3 threadsPerBlock(BSIZE_X, 1);
            dim3 numBlocks(1,1);
            kernel_InsertSource<<<numBlocks, threadsPerBlock>>> (val, iSource, dev_pc[gpu], dev_qc[gpu], fix_position, offset);
          }
        }
    }
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
}

#include "hip_defines.h"
#include "../driver.h"
#include "hip_insertsource.h"

extern int number_gpu;

__global__ void kernel_InsertSource(const float val, const int iSource,
                                    float *restrict qp, float *restrict qc, int fix_position, int offset)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  if (ix == 0)
  {
    qp[offset] += val;
    qc[offset] += val;
  }
}

void HIP_InsertSource(const float val, const int iSource, float *restrict pc, float *restrict qc, float *restrict pp, float *restrict qp)
{
    extern Gpu *gpu_map;
    extern float **dev_pp;
    extern float **dev_pc;
    extern float **dev_qp;
    extern float **dev_qc;

    int gpu_mid;
    int offset = 0;
    int fix_position = 0;

    if (number_gpu != 1)
    {
        for (int gpu = 0; gpu < number_gpu; gpu++)
        {
            hipDeviceProp_t prop;
            hipSetDevice(gpu);
            if ((dev_pp[gpu]) && (dev_qp[gpu]))
            {
                gpu_mid = number_gpu / 2;
                if (gpu == gpu_mid)
                {
                    offset = gpu_map[2].center_position;
                }
                else if (gpu == (gpu_mid - 1))
                {
                    offset = gpu_map[1].center_position;
                }
                else
                {
                    offset = -1;
                }
                if (offset != -1)
                {
                    hipLaunchKernelGGL(kernel_InsertSource, dim3(1, 1), dim3(BSIZE_X, 1), 0, 0, val, iSource, dev_pc[gpu], dev_qc[gpu], fix_position, offset);
                }
            }
        }
        HIP_CALL(hipGetLastError());
        HIP_CALL(hipDeviceSynchronize());
    }
    else
    {
        hipLaunchKernelGGL(kernel_InsertSource, dim3(1, 1), dim3(BSIZE_X, 1), 0, 0, val, iSource, dev_pc[0], dev_qc[0], fix_position, iSource);
        HIP_CALL(hipGetLastError());
        HIP_CALL(hipDeviceSynchronize());
    }
}

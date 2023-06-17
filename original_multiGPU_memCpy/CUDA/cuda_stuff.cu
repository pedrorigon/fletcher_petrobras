#include "cuda_defines.h"
#include "cuda_stuff.h"
#include "../map.h"

void CUDA_Initialize(const int sx, const int sy, const int sz, const int bord,
                     float dx, float dy, float dz, float dt,
                     float *restrict ch1dxx, float *restrict ch1dyy, float *restrict ch1dzz,
                     float *restrict ch1dxy, float *restrict ch1dyz, float *restrict ch1dxz,
                     float *restrict v2px, float *restrict v2pz, float *restrict v2sz, float *restrict v2pn,
                     float *restrict vpz, float *restrict vsv, float *restrict epsilon, float *restrict delta,
                     float *restrict phi, float *restrict theta,
                     float *restrict pp, float *restrict pc, float *restrict qp, float *restrict qc)
{

   extern float* dev_ch1dxx;
   extern float* dev_ch1dyy;
   extern float* dev_ch1dzz;
   extern float* dev_ch1dxy;
   extern float* dev_ch1dyz;
   extern float* dev_ch1dxz;
   extern float* dev_v2px;
   extern float* dev_v2pz;
   extern float* dev_v2sz;
   extern float* dev_v2pn;
   extern float* dev_pp;
   extern float* dev_pc;
   extern float* dev_qp;
   extern float* dev_qc;

   int deviceCount;
   CUDA_CALL(cudaGetDeviceCount(&deviceCount));

   // Check sx, sy values
   if (sx % BSIZE_X != 0)
   {
      printf("sx(%d) must be multiple of BSIZE_X(%d)\n", sx, (int)BSIZE_X);
      exit(1);
   }
   if (sy % BSIZE_Y != 0)
   {
      printf("sy(%d) must be multiple of BSIZE_Y(%d)\n", sy, (int)BSIZE_Y);
      exit(1);
   }

   int sxsy = sx * sy; // one plan
   const size_t sxsysz = sxsy * sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   const size_t msize_vol_extra = msize_vol + 2 * sxsy * sizeof(float); // 2 extra plans for wave fields

   const int strideX = ind(1, 0, 0) - ind(0, 0, 0);
   const int strideY = ind(0, 1, 0) - ind(0, 0, 0);
   const int strideZ = ind(0, 0, 1) - ind(0, 0, 0);
/*
   CUDA_CALL(cudaMalloc(&dev_ch1dxx, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_ch1dyy, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_ch1dzz, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_ch1dxy, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_ch1dyz, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_ch1dxz, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_v2px, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_v2pz, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_v2sz, msize_vol));
   CUDA_CALL(cudaMalloc(&dev_v2pn, msize_vol));

   // Wave field arrays with an extra plan
   CUDA_CALL(cudaMalloc(&dev_pp, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_pp, 0, msize_vol_extra));
   CUDA_CALL(cudaMalloc(&dev_pc, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_pc, 0, msize_vol_extra));
   CUDA_CALL(cudaMalloc(&dev_qp, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_qp, 0, msize_vol_extra));
   CUDA_CALL(cudaMalloc(&dev_qc, msize_vol_extra));
   CUDA_CALL(cudaMemset(dev_qc, 0, msize_vol_extra));
   dev_pp+=sxsy;
   dev_pc+=sxsy;
   dev_qp+=sxsy;
   dev_qc+=sxsy;
*/
      // Cálculo do número de elementos para cada GPU
   int numElementsPerGPU = (sx * sy * sz) / deviceCount;

   // Cálculo do número de elementos para as variáveis dev_pp, dev_pc, dev_qp e dev_qc
   int numExtraElementsPerGPU = msize_vol_extra / deviceCount;

   // Cópia dos dados para cada GPU
   for (int device = 0; device < deviceCount; device++)
   {
      cudaDeviceProp deviceProp;
      CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
      printf("CUDA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
      CUDA_CALL(cudaSetDevice(device));

      // Calcula o intervalo de elementos para a GPU atual
      int gpuLower = device * numElementsPerGPU;
      int gpuUpper = (device == deviceCount - 1) ? (sx * sy * sz) : ((device + 1) * numElementsPerGPU);

      // Calcula o número de elementos para a GPU atual
      int numElements = gpuUpper - gpuLower;

      // Calcula o intervalo de elementos para as variáveis dev_pp, dev_pc, dev_qp e dev_qc
      int extraGpuLower = device * numExtraElementsPerGPU;
      int extraGpuUpper = (device == deviceCount - 1) ? msize_vol_extra : ((device + 1) * numExtraElementsPerGPU);

      // Calcula o número de elementos para as variáveis dev_pp, dev_pc, dev_qp e dev_qc
      int numExtraElements = extraGpuUpper - extraGpuLower;

      // Copia os dados da CPU para a GPU atual
     /* CUDA_CALL(cudaMemcpyAsync(dev_ch1dxx + gpuLower, ch1dxx + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dyy + gpuLower, ch1dyy + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dzz + gpuLower, ch1dzz + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dxy + gpuLower, ch1dxy + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dyz + gpuLower, ch1dyz + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dxz + gpuLower, ch1dxz + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2px + gpuLower, v2px + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2pz + gpuLower, v2pz + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2sz + gpuLower, v2sz + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2pn + gpuLower, v2pn + gpuLower, numElements * sizeof(float), cudaMemcpyHostToDevice));

      CUDA_CALL(cudaMemcpyAsync(dev_pp + extraGpuLower, pp + extraGpuLower, numExtraElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_pc + extraGpuLower, pc + extraGpuLower, numExtraElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_qp + extraGpuLower, qp + extraGpuLower, numExtraElements * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_qc + extraGpuLower, qc + extraGpuLower, numExtraElements * sizeof(float), cudaMemcpyHostToDevice));
      */

      CUDA_CALL(cudaMalloc(&dev_ch1dxx, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dyy, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dzz, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dxy, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dyz, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dxz, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_v2px, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_v2pz, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_v2sz, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_v2pn, msize_vol));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dxx, ch1dxx, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dyy, ch1dyy, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dzz, ch1dzz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dxy, ch1dxy, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dyz, ch1dyz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dxz, ch1dxz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2px, v2px, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2pz, v2pz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2sz, v2sz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2pn, v2pn, msize_vol, cudaMemcpyHostToDevice));

      // Wave field arrays with an extra plan
      CUDA_CALL(cudaMalloc(&dev_pp, msize_vol_extra));
      CUDA_CALL(cudaMemset(dev_pp, 0, msize_vol_extra));
      CUDA_CALL(cudaMalloc(&dev_pc, msize_vol_extra));
      CUDA_CALL(cudaMemset(dev_pc, 0, msize_vol_extra));
      CUDA_CALL(cudaMalloc(&dev_qp, msize_vol_extra));
      CUDA_CALL(cudaMemset(dev_qp, 0, msize_vol_extra));
      CUDA_CALL(cudaMalloc(&dev_qc, msize_vol_extra));
      CUDA_CALL(cudaMemset(dev_qc, 0, msize_vol_extra));

      dev_pp+=sxsy;
      dev_pc+=sxsy;
      dev_qp+=sxsy;
      dev_qc+=sxsy;

      CUDA_CALL(cudaDeviceSynchronize());
      CUDA_CALL(cudaGetLastError());

      printf("GPU memory usage = %ld MiB\n", 15 * msize_vol / 1024 / 1024);
      size_t freeMem, totalMem;
      CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));
      printf("GPU memory usage: %lu MiB (used) / %lu MiB (total)\n", (totalMem - freeMem) / (1024 * 1024), totalMem / (1024 * 1024));
   }
}
  

// ARTHUR - Ajustar função para receber os parametros do CUDA_Finalize.
void CUDA_Finalize(const int sx, const int sy, const int sz, const int bord,
                   float dx, float dy, float dz, float dt,
                   float *restrict ch1dxx, float *restrict ch1dyy, float *restrict ch1dzz,
                   float *restrict ch1dxy, float *restrict ch1dyz, float *restrict ch1dxz,
                   float *restrict v2px, float *restrict v2pz, float *restrict v2sz, float *restrict v2pn,
                   float *restrict vpz, float *restrict vsv, float *restrict epsilon, float *restrict delta,
                   float *restrict phi, float *restrict theta,
                   float *restrict pp, float *restrict pc, float *restrict qp, float *restrict qc)
{

   extern float* dev_ch1dxx;
   extern float* dev_ch1dyy;
   extern float* dev_ch1dzz;
   extern float* dev_ch1dxy;
   extern float* dev_ch1dyz;
   extern float* dev_ch1dxz;
   extern float* dev_v2px;
   extern float* dev_v2pz;
   extern float* dev_v2sz;
   extern float* dev_v2pn;
   extern float* dev_pp;
   extern float* dev_pc;
   extern float* dev_qp;
   extern float* dev_qc;


   int sxsy = sx * sy; // one plan
   dev_pp -= sxsy;
   dev_pc -= sxsy;
   dev_qp -= sxsy;
   dev_qc -= sxsy;

   CUDA_CALL(cudaFree(dev_ch1dxx));
   CUDA_CALL(cudaFree(dev_ch1dyy));
   CUDA_CALL(cudaFree(dev_ch1dzz));
   CUDA_CALL(cudaFree(dev_ch1dxy));
   CUDA_CALL(cudaFree(dev_ch1dyz));
   CUDA_CALL(cudaFree(dev_ch1dxz));
   CUDA_CALL(cudaFree(dev_v2px));
   CUDA_CALL(cudaFree(dev_v2pz));
   CUDA_CALL(cudaFree(dev_v2sz));
   CUDA_CALL(cudaFree(dev_v2pn));
   CUDA_CALL(cudaFree(dev_pp));
   CUDA_CALL(cudaFree(dev_pc));
   CUDA_CALL(cudaFree(dev_qp));
   CUDA_CALL(cudaFree(dev_qc));

   printf("CUDA_Finalize: SUCCESS\n");
}

void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
   // arthur: se usar memoria unificada, não precisa desta cópia.
   extern float* dev_pc;
   const size_t sxsysz = ((size_t)sx * sy) * sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   if (pc){
      CUDA_CALL(cudaMemcpyAsync(pc, dev_pc, msize_vol, cudaMemcpyDeviceToHost));
   } 
}

void CUDA_prefetch_pc(const int sx, const int sy, const int sz, float *pc)
{

   extern float* dev_pc;
   int sxsy = sx * sy; // one plan
   const size_t sxsysz = sxsy * sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   const size_t msize_vol_extra = msize_vol + 2 * sxsy * sizeof(float); // 2 extra plans for wave fields
   int deviceCount;
   CUDA_CALL(cudaGetDeviceCount(&deviceCount));

   // Cálculo do número de elementos para cada GPU
   int numElementsPerGPU = (sx * sy * sz) / deviceCount;

   // Cálculo do número de elementos para as variáveis dev_pp, dev_pc, dev_qp e dev_qc
   int numExtraElementsPerGPU = msize_vol_extra / deviceCount;

   // Cópia dos dados para cada GPU
   for (int device = 0; device < deviceCount; device++)
   {
   
      cudaDeviceProp deviceProp;
      CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
      printf("CUDA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
      CUDA_CALL(cudaSetDevice(device));

      // Calcula o intervalo de elementos para a GPU atual
      int gpuLower = device * numElementsPerGPU;
      int gpuUpper = (device == deviceCount - 1) ? (sx * sy * sz) : ((device + 1) * numElementsPerGPU);

      // Calcula o número de elementos para a GPU atual
      int numElements = gpuUpper - gpuLower;

      // Calcula o intervalo de elementos para as variáveis dev_pp, dev_pc, dev_qp e dev_qc
      int extraGpuLower = device * numExtraElementsPerGPU;
      int extraGpuUpper = (device == deviceCount - 1) ? msize_vol_extra : ((device + 1) * numExtraElementsPerGPU);

      // Calcula o número de elementos para as variáveis dev_pp, dev_pc, dev_qp e dev_qc
      int numExtraElements = extraGpuUpper - extraGpuLower;

      // Copia os dados da CPU para a GPU atual
      CUDA_CALL(cudaMemcpyAsync(pc, dev_pc, msize_vol, cudaMemcpyHostToDevice));
   }
}

void CUDA_Allocate_Model_Variables(float **restrict ch1dxx, float **restrict ch1dyy, float **restrict ch1dzz, float **restrict ch1dxy,
                                   float **restrict ch1dyz, float **restrict ch1dxz, float **restrict v2px, float **restrict v2pz, float **restrict v2sz,
                                   float **restrict v2pn, int sx, int sy, int sz)
{
   const size_t sxsysz = ((size_t)sx * sy) * sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   CUDA_CALL(cudaMallocHost(ch1dxx, msize_vol));
   CUDA_CALL(cudaMallocHost(ch1dyy, msize_vol));
   CUDA_CALL(cudaMallocHost(ch1dzz, msize_vol));
   CUDA_CALL(cudaMallocHost(ch1dxy, msize_vol));
   CUDA_CALL(cudaMallocHost(ch1dyz, msize_vol));
   CUDA_CALL(cudaMallocHost(ch1dxz, msize_vol));
   CUDA_CALL(cudaMallocHost(v2px, msize_vol));
   CUDA_CALL(cudaMallocHost(v2pz, msize_vol));
   CUDA_CALL(cudaMallocHost(v2sz, msize_vol));
   CUDA_CALL(cudaMallocHost(v2pn, msize_vol));
}

void CUDA_Allocate_main(float **restrict vpz, float **restrict vsv, float **restrict epsilon, float **restrict delta,
                        float **restrict phi, float **restrict theta, float **restrict pp, float **restrict pc, float **restrict qp,
                        float **restrict qc, int sx, int sy, int sz)
{
   int sxsy = sx * sy;
   const size_t sxsysz = ((size_t)sx * sy) * sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   const size_t msize_vol_extra = msize_vol + 2 * sx * sy * sizeof(float); // 2 extra plans for wave fields
   CUDA_CALL(cudaMallocHost(vpz, msize_vol));
   CUDA_CALL(cudaMallocHost(vsv, msize_vol));
   CUDA_CALL(cudaMallocHost(epsilon, msize_vol));
   CUDA_CALL(cudaMallocHost(delta, msize_vol));
   CUDA_CALL(cudaMallocHost(phi, msize_vol));
   CUDA_CALL(cudaMallocHost(theta, msize_vol));

   CUDA_CALL(cudaMallocHost(pp, msize_vol_extra));
   CUDA_CALL(cudaMallocHost(pc, msize_vol_extra));
   CUDA_CALL(cudaMallocHost(qp, msize_vol_extra));
   CUDA_CALL(cudaMallocHost(qc, msize_vol_extra));
   // ARTHUR - Ver se esta operação fica na CPU ou mover para a GPU.
   memset(*pp, 0, msize_vol_extra);
   memset(*pc, 0, msize_vol_extra);
   memset(*qp, 0, msize_vol_extra);
   memset(*qc, 0, msize_vol_extra);
   // pp+=sxsy;
   // pc+=sxsy;
   // qp+=sxsy;
   // qc+=sxsy;
}
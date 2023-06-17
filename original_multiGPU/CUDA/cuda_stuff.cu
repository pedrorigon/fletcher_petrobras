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

   for (int device = 0; device < deviceCount; device++)
   {
      cudaDeviceProp deviceProp;
      CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
      printf("CUDA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
      CUDA_CALL(cudaSetDevice(device));

      // arthur -- Se for realizar a cópia assíncrona com prefetch, é aqui o lugar.

      int gpuLower, gpuUpper, numElements;
      if (device == 0)
      {
         gpuLower = 0;
         gpuUpper = (sz / 2) + 5;  // +4 devido ao dercross que usar 4*stride (+ e -) 
         numElements = (sx*sy*gpuUpper) - (sx*sy*gpuLower);
      }
      else
      {
         gpuLower = (sz / 2 ) - 5;  // -4 devido ao dercross que usar 4*stride (+ e -) 
         gpuUpper = sz;
         numElements = (sx*sy*gpuUpper) - (sx*sy*gpuLower);
      }

       // Prefetch necessary arrays for the GPU

      cudaMemPrefetchAsync(&ch1dxx[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&ch1dyy[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&ch1dzz[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&ch1dxy[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&ch1dyz[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&ch1dxz[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&v2px[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&v2pz[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&v2sz[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&v2pn[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&pp[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&pc[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&qp[gpuLower], numElements * sizeof(float), device);
      cudaMemPrefetchAsync(&qc[gpuLower], numElements * sizeof(float), device);

//}
      //CUDA_CALL(cudaDeviceSynchronize());
      //CUDA_CALL(cudaGetLastError());

      printf("GPU memory usage = %ld MiB\n", 15 * msize_vol / 1024 / 1024);

      size_t freeMem, totalMem;
      CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));
      printf("GPU memory usage: %lu MiB (used) / %lu MiB (total)\n", (totalMem - freeMem) / (1024 * 1024), totalMem / (1024 * 1024));
   }
   pp += sxsy;
   pc += sxsy;
   qp += sxsy;
   qc += sxsy;
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
   int sxsy = sx * sy; // one plan
   pp -= sxsy;
   pc -= sxsy;
   qp -= sxsy;
   qc -= sxsy;

   CUDA_CALL(cudaFree(vpz));
   CUDA_CALL(cudaFree(vsv));
   CUDA_CALL(cudaFree(epsilon));
   CUDA_CALL(cudaFree(delta));
   CUDA_CALL(cudaFree(phi));
   CUDA_CALL(cudaFree(theta));
   CUDA_CALL(cudaFree(ch1dxx));
   CUDA_CALL(cudaFree(ch1dyy));
   CUDA_CALL(cudaFree(ch1dzz));
   CUDA_CALL(cudaFree(ch1dxy));
   CUDA_CALL(cudaFree(ch1dyz));
   CUDA_CALL(cudaFree(ch1dxz));
   CUDA_CALL(cudaFree(v2px));
   CUDA_CALL(cudaFree(v2pz));
   CUDA_CALL(cudaFree(v2sz));
   CUDA_CALL(cudaFree(v2pn));
   // CUDA_CALL(cudaFree(pp));
   // CUDA_CALL(cudaFree(pc));
   // CUDA_CALL(cudaFree(qp));
   // CUDA_CALL(cudaFree(qc));

   printf("CUDA_Finalize: SUCCESS\n");
}

void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
   // arthur: se usar memoria unificada, não precisa desta cópia.
   // extern float* dev_pc;
   const size_t sxsysz = ((size_t)sx * sy) * sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   // if (pc) CUDA_CALL(cudaMemcpy(pc, dev_pc, msize_vol, cudaMemcpyDeviceToHost));
   CUDA_CALL(cudaMemPrefetchAsync(pc, msize_vol, cudaCpuDeviceId));
}

void CUDA_prefetch_pc(const int sx, const int sy, const int sz, float *pc){

   int deviceCount;
   CUDA_CALL(cudaGetDeviceCount(&deviceCount));

   for (int device = 0; device < deviceCount; device++)
   {
      cudaDeviceProp deviceProp;
      CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
      printf("CUDA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
      CUDA_CALL(cudaSetDevice(device));

      // arthur -- Se for realizar a cópia assíncrona com prefetch, é aqui o lugar.

      int gpuLower, gpuUpper, numElements;
      if (device == 0)
      {
         gpuLower = 0;
         gpuUpper = (sz / 2) + 5;  // +4 devido ao dercross que usar 4*stride (+ e -) 
         numElements = (sx*sy*gpuUpper) - (sx*sy*gpuLower);
      }
      else
      {
         gpuLower = (sz / 2 ) - 5;  // -4 devido ao dercross que usar 4*stride (+ e -) 
         gpuUpper = sz;
         numElements = (sx*sy*gpuUpper) - (sx*sy*gpuLower);
      }

      cudaMemPrefetchAsync(&pc[gpuLower], numElements * sizeof(float), device);
   }
}

void CUDA_Allocate_Model_Variables(float **restrict ch1dxx, float **restrict ch1dyy, float **restrict ch1dzz, float **restrict ch1dxy,
                                   float **restrict ch1dyz, float **restrict ch1dxz, float **restrict v2px, float **restrict v2pz, float **restrict v2sz,
                                   float **restrict v2pn, int sx, int sy, int sz)
{
   const size_t sxsysz = ((size_t)sx * sy) * sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   CUDA_CALL(cudaMallocManaged(ch1dxx, msize_vol));
   CUDA_CALL(cudaMallocManaged(ch1dyy, msize_vol));
   CUDA_CALL(cudaMallocManaged(ch1dzz, msize_vol));
   CUDA_CALL(cudaMallocManaged(ch1dxy, msize_vol));
   CUDA_CALL(cudaMallocManaged(ch1dyz, msize_vol));
   CUDA_CALL(cudaMallocManaged(ch1dxz, msize_vol));
   CUDA_CALL(cudaMallocManaged(v2px, msize_vol));
   CUDA_CALL(cudaMallocManaged(v2pz, msize_vol));
   CUDA_CALL(cudaMallocManaged(v2sz, msize_vol));
   CUDA_CALL(cudaMallocManaged(v2pn, msize_vol));
}

void CUDA_Allocate_main(float **restrict vpz, float **restrict vsv, float **restrict epsilon, float **restrict delta,
                        float **restrict phi, float **restrict theta, float **restrict pp, float **restrict pc, float **restrict qp,
                        float **restrict qc, int sx, int sy, int sz)
{
   int sxsy = sx * sy;
   const size_t sxsysz = ((size_t)sx * sy) * sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   const size_t msize_vol_extra = msize_vol + 2 * sx * sy * sizeof(float); // 2 extra plans for wave fields
   CUDA_CALL(cudaMallocManaged(vpz, msize_vol));
   CUDA_CALL(cudaMallocManaged(vsv, msize_vol));
   CUDA_CALL(cudaMallocManaged(epsilon, msize_vol));
   CUDA_CALL(cudaMallocManaged(delta, msize_vol));
   CUDA_CALL(cudaMallocManaged(phi, msize_vol));
   CUDA_CALL(cudaMallocManaged(theta, msize_vol));

   CUDA_CALL(cudaMallocManaged(pp, msize_vol_extra));
   CUDA_CALL(cudaMallocManaged(pc, msize_vol_extra));
   CUDA_CALL(cudaMallocManaged(qp, msize_vol_extra));
   CUDA_CALL(cudaMallocManaged(qc, msize_vol_extra));
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
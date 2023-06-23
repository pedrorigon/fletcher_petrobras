#include "cuda_defines.h"
#include "cuda_stuff.h"
#include "../driver.h"
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

   typedef struct {
      size_t gpu_upper_x;
      size_t gpu_upper_y;
      size_t gpu_upper_z;
      size_t gpu_lower_x;
      size_t gpu_lower_y;
      size_t gpu_lower_z;
      size_t gpu_bord_size;
      size_t gpu_offset_start;
      size_t gpu_offset_end;
      size_t gpu_total_size;
      size_t cpu_lower;
      size_t cpu_upper;
      size_t cpu_bord_size_lower_inf;
      size_t cpu_bord_size_upper_inf;

   } Gpu;

   Gpu gpu_map[deviceCount];

   
   extern float* dev_ch1dxx[GPU_NUMBER];
   extern float* dev_ch1dyy[GPU_NUMBER];
   extern float* dev_ch1dzz[GPU_NUMBER];
   extern float* dev_ch1dxy[GPU_NUMBER];
   extern float* dev_ch1dyz[GPU_NUMBER];
   extern float* dev_ch1dxz[GPU_NUMBER];
   extern float* dev_v2px[GPU_NUMBER];
   extern float* dev_v2pz[GPU_NUMBER];
   extern float* dev_v2sz[GPU_NUMBER];
   extern float* dev_v2pn[GPU_NUMBER];
   extern float* dev_pp[GPU_NUMBER];
   extern float* dev_pc[GPU_NUMBER];
   extern float* dev_qp[GPU_NUMBER];
   extern float* dev_qc[GPU_NUMBER];


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

   // Cópia dos dados para cada GPU
   for (int device = 0; device < deviceCount; device++)
   {
      cudaDeviceProp deviceProp;
      CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
      printf("CUDA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
      CUDA_CALL(cudaSetDevice(device));

      CUDA_CALL(cudaMalloc(&dev_ch1dxx[device], msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dyy[device], msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dzz[device], msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dxy[device], msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dyz[device], msize_vol));
      CUDA_CALL(cudaMalloc(&dev_ch1dxz[device], msize_vol));
      CUDA_CALL(cudaMalloc(&dev_v2px[device], msize_vol));
      CUDA_CALL(cudaMalloc(&dev_v2pz[device], msize_vol));
      CUDA_CALL(cudaMalloc(&dev_v2sz[device], msize_vol));
      CUDA_CALL(cudaMalloc(&dev_v2pn[device], msize_vol));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dxx[device], ch1dxx, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dyy[device], ch1dyy, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dzz[device], ch1dzz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dxy[device], ch1dxy, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dyz[device], ch1dyz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_ch1dxz[device], ch1dxz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2px[device], v2px, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2pz[device], v2pz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2sz[device], v2sz, msize_vol, cudaMemcpyHostToDevice));
      CUDA_CALL(cudaMemcpyAsync(dev_v2pn[device], v2pn, msize_vol, cudaMemcpyHostToDevice));

      // Wave field arrays with an extra plan
      CUDA_CALL(cudaMalloc(&dev_pp[device], msize_vol));
      CUDA_CALL(cudaMemset(dev_pp[device], 0, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_pc[device], msize_vol));
      CUDA_CALL(cudaMemset(dev_pc[device], 0, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_qp[device], msize_vol));
      CUDA_CALL(cudaMemset(dev_qp[device], 0, msize_vol));
      CUDA_CALL(cudaMalloc(&dev_qc[device], msize_vol));
      CUDA_CALL(cudaMemset(dev_qc[device], 0, msize_vol));


      printf("GPU memory usage = %ld MiB\n", 15 * msize_vol / 1024 / 1024);
      size_t freeMem, totalMem;
      CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));
      printf("GPU memory usage: %lu MiB (used) / %lu MiB (total)\n", (totalMem - freeMem) / (1024 * 1024), totalMem / (1024 * 1024));
   }
   CUDA_CALL(cudaDeviceSynchronize());
   CUDA_CALL(cudaGetLastError());
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


   extern float* dev_ch1dxx[GPU_NUMBER];
   extern float* dev_ch1dyy[GPU_NUMBER];
   extern float* dev_ch1dzz[GPU_NUMBER];
   extern float* dev_ch1dxy[GPU_NUMBER];
   extern float* dev_ch1dyz[GPU_NUMBER];
   extern float* dev_ch1dxz[GPU_NUMBER];
   extern float* dev_v2px[GPU_NUMBER];
   extern float* dev_v2pz[GPU_NUMBER];
   extern float* dev_v2sz[GPU_NUMBER];
   extern float* dev_v2pn[GPU_NUMBER];
   extern float* dev_pp[GPU_NUMBER];
   extern float* dev_pc[GPU_NUMBER];
   extern float* dev_qp[GPU_NUMBER];
   extern float* dev_qc[GPU_NUMBER];
   extern float* bordSwap[GPU_NUMBER];

   int deviceCount;
   CUDA_CALL(cudaGetDeviceCount(&deviceCount));
   int sxsy = sx * sy; // one plan
   for (int device = 0; device < deviceCount; device++)
   {
      cudaDeviceProp deviceProp;
      CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
      printf("CUDA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
      CUDA_CALL(cudaSetDevice(device));

      CUDA_CALL(cudaFree(dev_ch1dxx[device]));
      CUDA_CALL(cudaFree(dev_ch1dyy[device]));
      CUDA_CALL(cudaFree(dev_ch1dzz[device]));
      CUDA_CALL(cudaFree(dev_ch1dxy[device]));
      CUDA_CALL(cudaFree(dev_ch1dyz[device]));
      CUDA_CALL(cudaFree(dev_ch1dxz[device]));
      CUDA_CALL(cudaFree(dev_v2px[device]));
      CUDA_CALL(cudaFree(dev_v2pz[device]));
      CUDA_CALL(cudaFree(dev_v2sz[device]));
      CUDA_CALL(cudaFree(dev_v2pn[device]));
      CUDA_CALL(cudaFree(dev_pp[device]));
      CUDA_CALL(cudaFree(dev_pc[device]));
      CUDA_CALL(cudaFree(dev_qp[device]));
      CUDA_CALL(cudaFree(dev_qc[device]));

   }

   printf("CUDA_Finalize: SUCCESS\n");
}

void CUDA_Update_pointers(const int sx, const int sy, const int sz, float* pc)
{
    extern float* dev_pc[GPU_NUMBER];
    int deviceCount;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    const size_t sxsysz = ((size_t)sx * sy) * sz;
    const size_t msize_vol = sxsysz * sizeof(float);
    const size_t msize_vol_half = msize_vol / 2;

    for (int device = 0; device < deviceCount; device++)
    {
        CUDA_CALL(cudaSetDevice(device));

        if (device == 0)
        {
            // Copiar a primeira metade do array dev_pc[0] --> primeira metade do array pc
            CUDA_CALL(cudaMemcpy(pc, dev_pc[0], msize_vol_half, cudaMemcpyDeviceToHost));

        }
        else
        {
            // Copiar a segunda metade do array dev_pc[device] --> segunda metade do array pc
            CUDA_CALL(cudaMemcpy(pc + (msize_vol_half / sizeof(float)), dev_pc[device] + (msize_vol_half / sizeof(float)), msize_vol_half, cudaMemcpyDeviceToHost));

        }
        CUDA_CALL(cudaDeviceSynchronize()); 
    }
}


void CUDA_prefetch_pc(const int sx, const int sy, const int sz, float *pc)
{

   extern float* dev_pc[GPU_NUMBER];
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
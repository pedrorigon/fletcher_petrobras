#include "cuda_defines.h"
#include "cuda_stuff.h"
#include "../driver.h"
#include "../map.h"

extern int number_gpu;

void CUDA_Initialize(const int sx, const int sy, const int sz, const int bord,
                     float dx, float dy, float dz, float dt,
                     float *restrict ch1dxx, float *restrict ch1dyy, float *restrict ch1dzz,
                     float *restrict ch1dxy, float *restrict ch1dyz, float *restrict ch1dxz,
                     float *restrict v2px, float *restrict v2pz, float *restrict v2sz, float *restrict v2pn,
                     float *restrict vpz, float *restrict vsv, float *restrict epsilon, float *restrict delta,
                     float *restrict phi, float *restrict theta,
                     float *restrict pp, float *restrict pc, float *restrict qp, float *restrict qc)
{
   extern Gpu *gpu_map;
   extern float **dev_ch1dxx;
   extern float **dev_ch1dyy;
   extern float **dev_ch1dzz;
   extern float **dev_ch1dxy;
   extern float **dev_ch1dyz;
   extern float **dev_ch1dxz;
   extern float **dev_v2px;
   extern float **dev_v2pz;
   extern float **dev_v2sz;
   extern float **dev_v2pn;
   extern float **dev_pp;
   extern float **dev_pc;
   extern float **dev_qp;
   extern float **dev_qc;

   int deviceCount;
   CUDA_CALL(cudaGetDeviceCount(&deviceCount));

   if (number_gpu == 4)
   {
      // GPU 0 - Important Variables
      gpu_map[0].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[0].gpu_start_pointer = 0;
      gpu_map[0].gpu_end_pointer = ind(0, 0, (sz / 4));
      gpu_map[0].gpu_size_gpu = (sx * sy * (sz / 4 + 5)) * sizeof(float);
      gpu_map[0].cpu_offset = 0;

      // GPU 1 - Important Variables
      gpu_map[1].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[1].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[1].gpu_end_pointer = ind(0, 0, (sz / 4 + 5));
      gpu_map[1].gpu_size_gpu = (sx * sy * (sz / 4 + 10)) * sizeof(float);
      gpu_map[1].cpu_offset = ind(0, 0, (sz / 4 - 5));
      gpu_map[1].center_position = ind(sx / 2, sy / 2, (sz / 4 + 5));

      // GPU 2 - Important Variables
      gpu_map[2].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[2].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[2].gpu_end_pointer = ind(0, 0, (sz / 4 + 5));
      gpu_map[2].gpu_size_gpu = (sx * sy * (sz / 4 + 10)) * sizeof(float);
      gpu_map[2].cpu_offset = ind(0, 0, (sz / 2 - 5));
      gpu_map[2].center_position = ind(sx / 2, sy / 2, 5);

      // GPU 2 - Important Variables
      gpu_map[3].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[3].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[3].gpu_end_pointer = ind(0, 0, (sz / 4 + 5));
      gpu_map[3].gpu_size_gpu = (sx * sy * (sz / 4 + 5)) * sizeof(float);
      gpu_map[3].cpu_offset = ind(0, 0, (3 * sz / 4 - 5));
   }
   else if (number_gpu == 2)
   {
      // GPU 0 - Important Variables
      gpu_map[0].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[0].gpu_start_pointer = 0;
      gpu_map[0].gpu_end_pointer = ind(0, 0, (sz / 2));
      gpu_map[0].gpu_size_gpu = (sx * sy * (sz / 2 + 5)) * sizeof(float);
      gpu_map[0].cpu_offset = 0;
      gpu_map[1].center_position = ind(sx / 2, sy / 2, (sz / 2 + 5));

      // GPU 1 - Important Variables
      gpu_map[1].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[1].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[1].gpu_end_pointer = ind(0, 0, (sz / 2 + 5));
      gpu_map[1].gpu_size_gpu = (sx * sy * (sz / 2 + 5)) * sizeof(float);
      gpu_map[1].cpu_offset = ind(0, 0, (sz / 2 - 5));
      gpu_map[1].center_position = ind(sx / 2, sy / 2, 5);
   }
   else if (number_gpu == 8)
   {
      // GPU 0 - Important Variables
      gpu_map[0].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[0].gpu_start_pointer = 0;
      gpu_map[0].gpu_end_pointer = ind(0, 0, (sz / 8));
      gpu_map[0].gpu_size_gpu = (sx * sy * (sz / 8 + 5)) * sizeof(float);
      gpu_map[0].cpu_offset = 0;

      // GPU 1 - Important Variables
      gpu_map[1].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[1].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[1].gpu_end_pointer = ind(0, 0, (sz / 8 + 5));
      gpu_map[1].gpu_size_gpu = (sx * sy * (sz / 8 + 10)) * sizeof(float);
      gpu_map[1].cpu_offset = ind(0, 0, (sz / 8 - 5));
      gpu_map[1].center_position = ind(sx / 2, sy / 2, (sz / 4 + 5));

      // GPU 2 - Important Variables
      gpu_map[2].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[2].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[2].gpu_end_pointer = ind(0, 0, (sz / 8 + 5));
      gpu_map[2].gpu_size_gpu = (sx * sy * (sz / 8 + 10)) * sizeof(float);
      gpu_map[2].cpu_offset = ind(0, 0, (sz / 4 - 5));
      gpu_map[2].center_position = ind(sx / 2, sy / 2, 5);

      // GPU 3 - Important Variables
      gpu_map[3].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[3].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[3].gpu_end_pointer = ind(0, 0, (sz / 8 + 5));
      gpu_map[3].gpu_size_gpu = (sx * sy * (sz / 8 + 10)) * sizeof(float);
      gpu_map[3].cpu_offset = ind(0, 0, ((3 * (sz / 8)) - 5));
      gpu_map[3].center_position = ind(sx / 2, sy / 2, (sz / 8 + 5));

      // GPU 4 - Important Variables
      gpu_map[4].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[4].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[4].gpu_end_pointer = ind(0, 0, (sz / 8 + 5));
      gpu_map[4].gpu_size_gpu = (sx * sy * (sz / 8 + 10)) * sizeof(float);
      gpu_map[4].cpu_offset = ind(0, 0, ((4 * (sz / 8)) - 5));
      gpu_map[4].center_position = ind(sx / 2, sy / 2, 5);

      // GPU 5 - Important Variables
      gpu_map[5].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[5].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[5].gpu_end_pointer = ind(0, 0, (sz / 8 + 5));
      gpu_map[5].gpu_size_gpu = (sx * sy * (sz / 8 + 10)) * sizeof(float);
      gpu_map[5].cpu_offset = ind(0, 0, ((5 * (sz / 8)) - 5));
      gpu_map[5].center_position = ind(sx / 2, sy / 2, 5);

      // GPU 6 - Important Variables
      gpu_map[6].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[6].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[6].gpu_end_pointer = ind(0, 0, (sz / 8 + 5));
      gpu_map[6].gpu_size_gpu = (sx * sy * (sz / 8 + 10)) * sizeof(float);
      gpu_map[6].cpu_offset = ind(0, 0, ((6 * (sz / 8)) - 5));
      gpu_map[6].center_position = ind(sx / 2, sy / 2, 5);

      // GPU 7 - Important Variables
      gpu_map[7].gpu_size_bord = sx * sy * (5) * sizeof(float);
      gpu_map[7].gpu_start_pointer = ind(0, 0, 5);
      gpu_map[7].gpu_end_pointer = ind(0, 0, (sz / 8 + 5));
      gpu_map[7].gpu_size_gpu = (sx * sy * (sz / 8 + 5)) * sizeof(float);
      gpu_map[7].cpu_offset = ind(0, 0, (7 * sz / 8 - 5));
   }

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
   for (int device = 0; device < number_gpu; device++)
   {
      cudaDeviceProp deviceProp;
      CUDA_CALL(cudaGetDeviceProperties(&deviceProp, device));
      printf("CUDA source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
      CUDA_CALL(cudaSetDevice(device));
      if (number_gpu != 1)
      {
         if (device == 0)
         {

            CUDA_CALL(cudaMalloc(&dev_ch1dxx[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dyy[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dzz[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dxy[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dyz[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dxz[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_v2px[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_v2pz[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_v2sz[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_v2pn[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dxx[0], ch1dxx, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dyy[0], ch1dyy, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dzz[0], ch1dzz, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dxy[0], ch1dxy, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dyz[0], ch1dyz, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dxz[0], ch1dxz, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_v2px[0], v2px, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_v2pz[0], v2pz, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_v2sz[0], v2sz, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_v2pn[0], v2pn, gpu_map[0].gpu_size_gpu, cudaMemcpyHostToDevice));

            // Wave field arrays with an extra plan
            CUDA_CALL(cudaMalloc(&dev_pp[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMemset(dev_pp[0], 0, gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_pc[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMemset(dev_pc[0], 0, gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_qp[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMemset(dev_qp[0], 0, gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_qc[0], gpu_map[0].gpu_size_gpu));
            CUDA_CALL(cudaMemset(dev_qc[0], 0, gpu_map[0].gpu_size_gpu));
         }
         else
         {
            CUDA_CALL(cudaMalloc(&dev_ch1dxx[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dyy[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dzz[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dxy[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dyz[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_ch1dxz[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_v2px[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_v2pz[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_v2sz[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_v2pn[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dxx[device], ch1dxx + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dyy[device], ch1dyy + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dzz[device], ch1dzz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dxy[device], ch1dxy + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dyz[device], ch1dyz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_ch1dxz[device], ch1dxz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_v2px[device], v2px + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_v2pz[device], v2pz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_v2sz[device], v2sz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpyAsync(dev_v2pn[device], v2pn + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, cudaMemcpyHostToDevice));

            // Wave field arrays with an extra plan
            CUDA_CALL(cudaMalloc(&dev_pp[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMemset(dev_pp[device], 0, gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_pc[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMemset(dev_pc[device], 0, gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_qp[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMemset(dev_qp[device], 0, gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMalloc(&dev_qc[device], gpu_map[device].gpu_size_gpu));
            CUDA_CALL(cudaMemset(dev_qc[device], 0, gpu_map[device].gpu_size_gpu));
         }
      }
      else
      {
         CUDA_CALL(cudaMalloc(&dev_ch1dxx[0], msize_vol));
         CUDA_CALL(cudaMalloc(&dev_ch1dyy[0], msize_vol));
         CUDA_CALL(cudaMalloc(&dev_ch1dzz[0], msize_vol));
         CUDA_CALL(cudaMalloc(&dev_ch1dxy[0], msize_vol));
         CUDA_CALL(cudaMalloc(&dev_ch1dyz[0], msize_vol));
         CUDA_CALL(cudaMalloc(&dev_ch1dxz[0], msize_vol));
         CUDA_CALL(cudaMalloc(&dev_v2px[0], msize_vol));
         CUDA_CALL(cudaMalloc(&dev_v2pz[0], msize_vol));
         CUDA_CALL(cudaMalloc(&dev_v2sz[0], msize_vol));
         CUDA_CALL(cudaMalloc(&dev_v2pn[0], msize_vol));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dxx[0], ch1dxx, msize_vol, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dyy[0], ch1dyy, msize_vol, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dzz[0], ch1dzz, msize_vol, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dxy[0], ch1dxy, msize_vol, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dyz[0], ch1dyz, msize_vol, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dxz[0], ch1dxz, msize_vol, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_v2px[0], v2px, msize_vol, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_v2pz[0], v2pz, msize_vol, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_v2sz[0], v2sz, msize_vol, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_v2pn[0], v2pn, msize_vol, cudaMemcpyHostToDevice));

         // Wave field arrays with an extra plan
         CUDA_CALL(cudaMalloc(&dev_pp[0], msize_vol_extra));
         CUDA_CALL(cudaMemset(dev_pp[0], 0, msize_vol_extra));
         CUDA_CALL(cudaMalloc(&dev_pc[0], msize_vol_extra));
         CUDA_CALL(cudaMemset(dev_pc[0], 0, msize_vol_extra));
         CUDA_CALL(cudaMalloc(&dev_qp[0], msize_vol_extra));
         CUDA_CALL(cudaMemset(dev_qp[0], 0, msize_vol_extra));
         CUDA_CALL(cudaMalloc(&dev_qc[0], msize_vol_extra));
         CUDA_CALL(cudaMemset(dev_qc[0], 0, msize_vol_extra));
      }
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

   extern float **dev_ch1dxx;
   extern float **dev_ch1dyy;
   extern float **dev_ch1dzz;
   extern float **dev_ch1dxy;
   extern float **dev_ch1dyz;
   extern float **dev_ch1dxz;
   extern float **dev_v2px;
   extern float **dev_v2pz;
   extern float **dev_v2sz;
   extern float **dev_v2pn;
   extern float **dev_pp;
   extern float **dev_pc;
   extern float **dev_qp;
   extern float **dev_qc;
   extern float **bordSwap;

   int deviceCount;
   CUDA_CALL(cudaGetDeviceCount(&deviceCount));
   int sxsy = sx * sy; // one plan
   for (int device = 0; device < number_gpu; device++)
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

void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
   extern float **dev_pc;
   extern Gpu *gpu_map;
   int deviceCount;
   CUDA_CALL(cudaGetDeviceCount(&deviceCount));
   const size_t sxsysz = ((size_t)sx * sy) * sz;
   const size_t msize_vol = sxsysz * sizeof(float);
   const size_t msize_vol_half = msize_vol / number_gpu;
   if (number_gpu == 1)
   {
      CUDA_CALL(cudaMemcpy(pc, dev_pc, msize_vol, cudaMemcpyDeviceToHost));
   }
   else
   {
      for (int device = 0; device < number_gpu; device++)
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
            CUDA_CALL(cudaMemcpy(pc + ((device * msize_vol_half) / sizeof(float)), dev_pc[device] + (gpu_map[device].gpu_start_pointer), msize_vol_half, cudaMemcpyDeviceToHost));
         }
         CUDA_CALL(cudaDeviceSynchronize());
      }
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

   CUDA_CALL(cudaMallocHost(pp, msize_vol));
   CUDA_CALL(cudaMallocHost(pc, msize_vol));
   CUDA_CALL(cudaMallocHost(qp, msize_vol));
   CUDA_CALL(cudaMallocHost(qc, msize_vol));
   // ARTHUR - Ver se esta operação fica na CPU ou mover para a GPU.
   memset(*pp, 0, msize_vol);
   memset(*pc, 0, msize_vol);
   memset(*qp, 0, msize_vol);
   memset(*qc, 0, msize_vol);
}

// Função auxiliar para alocar memória para um vetor de ponteiros
void allocate_pointer_array(float ***array, const char *name)
{
   *array = (float **)malloc(number_gpu * sizeof(float *));
   if (*array == NULL)
   {
      fprintf(stderr, "Memory allocation failed for %s\n", name);
      exit(EXIT_FAILURE);
   }
}

void initialize_mgpu(int gpu_number, int sz)
{
   extern Gpu *gpu_map;
   extern float **dev_ch1dxx;
   extern float **dev_ch1dyy;
   extern float **dev_ch1dzz;
   extern float **dev_ch1dxy;
   extern float **dev_ch1dyz;
   extern float **dev_ch1dxz;
   extern float **dev_v2px;
   extern float **dev_v2pz;
   extern float **dev_v2sz;
   extern float **dev_v2pn;
   extern float **dev_pp;
   extern float **dev_pc;
   extern float **dev_qp;
   extern float **dev_qc;

   gpu_map = (Gpu *)malloc(number_gpu * sizeof(Gpu));
   if (gpu_map == NULL)
   {
      fprintf(stderr, "Memory allocation failed for gpu_map\n");
      exit(EXIT_FAILURE);
   }

   // Alocar memória para cada vetor de ponteiros
   allocate_pointer_array(&dev_ch1dxx, "dev_ch1dxx");
   allocate_pointer_array(&dev_ch1dyy, "dev_ch1dyy");
   allocate_pointer_array(&dev_ch1dzz, "dev_ch1dzz");
   allocate_pointer_array(&dev_ch1dxy, "dev_ch1dxy");
   allocate_pointer_array(&dev_ch1dyz, "dev_ch1dyz");
   allocate_pointer_array(&dev_ch1dxz, "dev_ch1dxz");
   allocate_pointer_array(&dev_v2px, "dev_v2px");
   allocate_pointer_array(&dev_v2pz, "dev_v2pz");
   allocate_pointer_array(&dev_v2sz, "dev_v2sz");
   allocate_pointer_array(&dev_v2pn, "dev_v2pn");
   allocate_pointer_array(&dev_pp, "dev_pp");
   allocate_pointer_array(&dev_pc, "dev_pc");
   allocate_pointer_array(&dev_qp, "dev_qp");
   allocate_pointer_array(&dev_qc, "dev_qc");

   if (number_gpu == 4)
   {
      gpu_map[0].lower_kernel1 = sz / 4 - 5;
      gpu_map[0].upper_kernel1 = sz / 4;
      gpu_map[0].lower_kernel2 = 5;
      gpu_map[0].upper_kernel2 = sz / 4 - 5;

      gpu_map[1].lower_kernel1 = sz/4;
      gpu_map[1].upper_kernel1 = sz/4 + 5;
      gpu_map[1].lower_kernel2 = 10;
      gpu_map[1].upper_kernel2 = sz / 4;

      gpu_map[2].lower_kernel1 = sz/4;
      gpu_map[2].upper_kernel1 = sz/4 + 5;
      gpu_map[2].lower_kernel2 = 10;
      gpu_map[2].upper_kernel2 = sz / 4;

      gpu_map[3].lower_kernel1 = 5;
      gpu_map[3].upper_kernel1 = 10;
      gpu_map[3].lower_kernel2 = 10;
      gpu_map[3].upper_kernel2 = sz / 4 - 1;
   }
   else if (number_gpu == 2)
   {
      gpu_map[0].lower_kernel1 = sz / 2 - 5;
      gpu_map[0].upper_kernel1 = sz / 2;
      gpu_map[0].lower_kernel2 = 5;
      gpu_map[0].upper_kernel2 = sz / 2 - 5;

      gpu_map[1].lower_kernel1 = 5;
      gpu_map[1].upper_kernel1 = 10;
      gpu_map[1].lower_kernel2 = 10;
      gpu_map[1].upper_kernel2 = sz / 2 - 1;
   }
   else if (number_gpu == 8){
      gpu_map[0].lower_kernel1 = sz / 8 - 5;
      gpu_map[0].upper_kernel1 = sz / 8;
      gpu_map[0].lower_kernel2 = 5;
      gpu_map[0].upper_kernel2 = sz / 8 - 5;

      gpu_map[1].lower_kernel1 = sz/8;
      gpu_map[1].upper_kernel1 = sz/8 + 5;
      gpu_map[1].lower_kernel2 = 10;
      gpu_map[1].upper_kernel2 = sz / 8;

      gpu_map[2].lower_kernel1 = sz/8;
      gpu_map[2].upper_kernel1 = sz/8 + 5;
      gpu_map[2].lower_kernel2 = 10;
      gpu_map[2].upper_kernel2 = sz / 8;

      gpu_map[3].lower_kernel1 = sz/8;
      gpu_map[3].upper_kernel1 = sz/8 + 5;
      gpu_map[3].lower_kernel2 = 10;
      gpu_map[3].upper_kernel2 = sz / 8;

      gpu_map[4].lower_kernel1 = sz/8;
      gpu_map[4].upper_kernel1 = sz/8 + 5;
      gpu_map[4].lower_kernel2 = 10;
      gpu_map[4].upper_kernel2 = sz / 8;

      gpu_map[5].lower_kernel1 = sz/8;
      gpu_map[5].upper_kernel1 = sz/8 + 5;
      gpu_map[5].lower_kernel2 = 10;
      gpu_map[5].upper_kernel2 = sz / 8;

      gpu_map[6].lower_kernel1 = sz/8;
      gpu_map[6].upper_kernel1 = sz/8 + 5;
      gpu_map[6].lower_kernel2 = 10;
      gpu_map[6].upper_kernel2 = sz / 8;

      gpu_map[7].lower_kernel1 = 5;
      gpu_map[7].upper_kernel1 = 10;
      gpu_map[7].lower_kernel2 = 10;
      gpu_map[7].upper_kernel2 = sz / 8 - 1;

   }
}

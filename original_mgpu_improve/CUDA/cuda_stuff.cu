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
   extern Gpu gpu_map[GPU_NUMBER];

   // GPU 0 - Important Variables
   gpu_map[0].gpu_upper_x = sx;
   gpu_map[0].gpu_upper_y = sy;
   gpu_map[0].gpu_upper_z = sz/2;
   gpu_map[0].gpu_lower_x = 0;
   gpu_map[0].gpu_lower_y = 0;
   gpu_map[0].gpu_lower_z = 0;
   gpu_map[0].gpu_size_bord = sx*sy*(4)*sizeof(float);
   gpu_map[0].gpu_payload = sx*sy*(sz/2)*sizeof(float);
   gpu_map[0].gpu_start_pointer = 0;
   gpu_map[0].gpu_end_pointer = ind(0,0,(sz/2));
   gpu_map[0].cpu_start_pointer = 0;
   gpu_map[0].cpu_end_pointer = ind(0,0,(sz/2));
   gpu_map[0].gpu_size_gpu = (sx*sy*(sz/2 + 4)) * sizeof(float);
   gpu_map[0].cpu_offset = 0;

   // GPU 1 - Important Variables
   gpu_map[1].gpu_upper_x = 0;
   gpu_map[1].gpu_upper_y = 0;
   gpu_map[1].gpu_upper_z = sz/2 + 4;
   gpu_map[1].gpu_lower_x = 0;
   gpu_map[1].gpu_lower_y = 0;
   gpu_map[1].gpu_lower_z = 4;
   gpu_map[1].gpu_size_bord = sx*sy*(4)*sizeof(float);
   gpu_map[1].gpu_payload = sx*sy*(sz/2)*sizeof(float);
   gpu_map[1].gpu_start_pointer = ind(0,0,4);
   gpu_map[1].gpu_end_pointer = ind(0,0,(sz/2+4));
   gpu_map[1].cpu_start_pointer = ind(0,0,(sz/2));
   gpu_map[1].cpu_end_pointer = ind(0,0,(sz));
   gpu_map[1].gpu_size_gpu = (sx*sy*(sz/2 + 4)) * sizeof(float);
   gpu_map[1].cpu_offset = ind(0,0,(sz/2 - 4));
   
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
      if(device == 0){

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
      }else{
         CUDA_CALL(cudaMalloc(&dev_ch1dxx[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_ch1dyy[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_ch1dzz[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_ch1dxy[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_ch1dyz[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_ch1dxz[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_v2px[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_v2pz[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_v2sz[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_v2pn[1], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dxx[1], ch1dxx + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dyy[1], ch1dyy + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dzz[1], ch1dzz + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dxy[1], ch1dxy + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dyz[1], ch1dyz + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_ch1dxz[1], ch1dxz + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_v2px[1], v2px + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_v2pz[1], v2pz + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_v2sz[1], v2sz + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));
         CUDA_CALL(cudaMemcpyAsync(dev_v2pn[1], v2pn + gpu_map[1].cpu_offset, gpu_map[1].gpu_size_gpu, cudaMemcpyHostToDevice));

         // Wave field arrays with an extra plan
         CUDA_CALL(cudaMalloc(&dev_pp[device], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMemset(dev_pp[device], 0, gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_pc[device], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMemset(dev_pc[device], 0, gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_qp[device], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMemset(dev_qp[device], 0, gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMalloc(&dev_qc[device], gpu_map[1].gpu_size_gpu));
         CUDA_CALL(cudaMemset(dev_qc[device], 0, gpu_map[1].gpu_size_gpu));
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
    extern Gpu gpu_map[GPU_NUMBER];
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
            CUDA_CALL(cudaMemcpy(pc + (msize_vol_half / sizeof(float)), dev_pc[device] + (gpu_map[1].gpu_start_pointer), msize_vol_half, cudaMemcpyDeviceToHost));

        }
        CUDA_CALL(cudaDeviceSynchronize()); 
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
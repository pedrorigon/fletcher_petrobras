#include "hip_defines.h"
#include "hip_stuff.h"
#include "../driver.h"
#include "../map.h"

extern int number_gpu;

void HIP_Initialize(const int sx, const int sy, const int sz, const int bord,
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
   HIP_CALL(hipGetDeviceCount(&deviceCount));

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
      hipDeviceProp_t deviceProp;
      HIP_CALL(hipGetDeviceProperties(&deviceProp, device));
      printf("HIP source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
      HIP_CALL(hipSetDevice(device));
      if (number_gpu != 1)
      {
         if (device == 0)
         {
            HIP_CALL(hipMalloc(&dev_ch1dxx[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dyy[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dzz[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dxy[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dyz[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dxz[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_v2px[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_v2pz[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_v2sz[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_v2pn[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMemcpyAsync(dev_ch1dxx[0], ch1dxx, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dyy[0], ch1dyy, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dzz[0], ch1dzz, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dxy[0], ch1dxy, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dyz[0], ch1dyz, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dxz[0], ch1dxz, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_v2px[0], v2px, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_v2pz[0], v2pz, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_v2sz[0], v2sz, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_v2pn[0], v2pn, gpu_map[0].gpu_size_gpu, hipMemcpyHostToDevice));

            // Wave field arrays with an extra plan
            HIP_CALL(hipMalloc(&dev_pp[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMemset(dev_pp[0], 0, gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_pc[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMemset(dev_pc[0], 0, gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_qp[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMemset(dev_qp[0], 0, gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_qc[0], gpu_map[0].gpu_size_gpu));
            HIP_CALL(hipMemset(dev_qc[0], 0, gpu_map[0].gpu_size_gpu));
         }
         else
         {
            HIP_CALL(hipMalloc(&dev_ch1dxx[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dyy[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dzz[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dxy[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dyz[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_ch1dxz[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_v2px[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_v2pz[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_v2sz[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_v2pn[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMemcpyAsync(dev_ch1dxx[device], ch1dxx + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dyy[device], ch1dyy + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dzz[device], ch1dzz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dxy[device], ch1dxy + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dyz[device], ch1dyz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_ch1dxz[device], ch1dxz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_v2px[device], v2px + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_v2pz[device], v2pz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_v2sz[device], v2sz + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));
            HIP_CALL(hipMemcpyAsync(dev_v2pn[device], v2pn + gpu_map[device].cpu_offset, gpu_map[device].gpu_size_gpu, hipMemcpyHostToDevice));

            // Wave field arrays with an extra plan
            HIP_CALL(hipMalloc(&dev_pp[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMemset(dev_pp[device], 0, gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_pc[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMemset(dev_pc[device], 0, gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_qp[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMemset(dev_qp[device], 0, gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMalloc(&dev_qc[device], gpu_map[device].gpu_size_gpu));
            HIP_CALL(hipMemset(dev_qc[device], 0, gpu_map[device].gpu_size_gpu));
         }
      }
      else
      {
         HIP_CALL(hipMalloc(&dev_ch1dxx[0], msize_vol));
         HIP_CALL(hipMalloc(&dev_ch1dyy[0], msize_vol));
         HIP_CALL(hipMalloc(&dev_ch1dzz[0], msize_vol));
         HIP_CALL(hipMalloc(&dev_ch1dxy[0], msize_vol));
         HIP_CALL(hipMalloc(&dev_ch1dyz[0], msize_vol));
         HIP_CALL(hipMalloc(&dev_ch1dxz[0], msize_vol));
         HIP_CALL(hipMalloc(&dev_v2px[0], msize_vol));
         HIP_CALL(hipMalloc(&dev_v2pz[0], msize_vol));
         HIP_CALL(hipMalloc(&dev_v2sz[0], msize_vol));
         HIP_CALL(hipMalloc(&dev_v2pn[0], msize_vol));
         HIP_CALL(hipMemcpyAsync(dev_ch1dxx[0], ch1dxx, msize_vol, hipMemcpyHostToDevice));
         HIP_CALL(hipMemcpyAsync(dev_ch1dyy[0], ch1dyy, msize_vol, hipMemcpyHostToDevice));
         HIP_CALL(hipMemcpyAsync(dev_ch1dzz[0], ch1dzz, msize_vol, hipMemcpyHostToDevice));
         HIP_CALL(hipMemcpyAsync(dev_ch1dxy[0], ch1dxy, msize_vol, hipMemcpyHostToDevice));
         HIP_CALL(hipMemcpyAsync(dev_ch1dyz[0], ch1dyz, msize_vol, hipMemcpyHostToDevice));
         HIP_CALL(hipMemcpyAsync(dev_ch1dxz[0], ch1dxz, msize_vol, hipMemcpyHostToDevice));
         HIP_CALL(hipMemcpyAsync(dev_v2px[0], v2px, msize_vol, hipMemcpyHostToDevice));
         HIP_CALL(hipMemcpyAsync(dev_v2pz[0], v2pz, msize_vol, hipMemcpyHostToDevice));
         HIP_CALL(hipMemcpyAsync(dev_v2sz[0], v2sz, msize_vol, hipMemcpyHostToDevice));
         HIP_CALL(hipMemcpyAsync(dev_v2pn[0], v2pn, msize_vol, hipMemcpyHostToDevice));

         // Wave field arrays with an extra plan
         HIP_CALL(hipMalloc(&dev_pp[0], msize_vol_extra));
         HIP_CALL(hipMemset(dev_pp[0], 0, msize_vol_extra));
         HIP_CALL(hipMalloc(&dev_pc[0], msize_vol_extra));
         HIP_CALL(hipMemset(dev_pc[0], 0, msize_vol_extra));
         HIP_CALL(hipMalloc(&dev_qp[0], msize_vol_extra));
         HIP_CALL(hipMemset(dev_qp[0], 0, msize_vol_extra));
         HIP_CALL(hipMalloc(&dev_qc[0], msize_vol_extra));
         HIP_CALL(hipMemset(dev_qc[0], 0, msize_vol_extra));
      }
      printf("GPU memory usage = %ld MiB\n", 15 * msize_vol / 1024 / 1024);
      size_t freeMem, totalMem;
      HIP_CALL(hipMemGetInfo(&freeMem, &totalMem));
      printf("GPU memory usage: %lu MiB (used) / %lu MiB (total)\n", (totalMem - freeMem) / (1024 * 1024), totalMem / (1024 * 1024));
   }
   HIP_CALL(hipDeviceSynchronize());
   HIP_CALL(hipGetLastError());
}

void HIP_Finalize(const int sx, const int sy, const int sz, const int bord,
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
    HIP_CALL(hipGetDeviceCount(&deviceCount));
    int sxsy = sx * sy; // one plan
    for (int device = 0; device < number_gpu; device++)
    {
        hipDeviceProp_t deviceProp;
        HIP_CALL(hipGetDeviceProperties(&deviceProp, device));
        printf("HIP source using device(%d) %s with compute capability %d.%d.\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
        HIP_CALL(hipSetDevice(device));

        HIP_CALL(hipFree(dev_ch1dxx[device]));
        HIP_CALL(hipFree(dev_ch1dyy[device]));
        HIP_CALL(hipFree(dev_ch1dzz[device]));
        HIP_CALL(hipFree(dev_ch1dxy[device]));
        HIP_CALL(hipFree(dev_ch1dyz[device]));
        HIP_CALL(hipFree(dev_ch1dxz[device]));
        HIP_CALL(hipFree(dev_v2px[device]));
        HIP_CALL(hipFree(dev_v2pz[device]));
        HIP_CALL(hipFree(dev_v2sz[device]));
        HIP_CALL(hipFree(dev_v2pn[device]));
        HIP_CALL(hipFree(dev_pp[device]));
        HIP_CALL(hipFree(dev_pc[device]));
        HIP_CALL(hipFree(dev_qp[device]));
        HIP_CALL(hipFree(dev_qc[device]));
    }

    printf("HIP_Finalize: SUCCESS\n");
}


void HIP_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
    extern float **dev_pc;
    extern Gpu *gpu_map;
    int deviceCount;
    HIP_CALL(hipGetDeviceCount(&deviceCount));
    const size_t sxsysz = ((size_t)sx * sy) * sz;
    const size_t msize_vol = sxsysz * sizeof(float);
    const size_t msize_vol_half = msize_vol / number_gpu;
    if (number_gpu == 1)
    {
        HIP_CALL(hipMemcpy(pc, dev_pc, msize_vol, hipMemcpyDeviceToHost));
    }
    else
    {
        for (int device = 0; device < number_gpu; device++)
        {
            HIP_CALL(hipSetDevice(device));

            if (device == 0)
            {
                // Copy the first half of the array dev_pc[0] --> first half of the array pc
                HIP_CALL(hipMemcpy(pc, dev_pc[0], msize_vol_half, hipMemcpyDeviceToHost));
            }
            else
            {
                // Copy the second half of the array dev_pc[device] --> second half of the array pc
                HIP_CALL(hipMemcpy(pc + ((device * msize_vol_half) / sizeof(float)), dev_pc[device] + (gpu_map[device].gpu_start_pointer), msize_vol_half, hipMemcpyDeviceToHost));
            }
            HIP_CALL(hipDeviceSynchronize());
        }
    }
}


void HIP_Allocate_Model_Variables(float **restrict ch1dxx, float **restrict ch1dyy, float **restrict ch1dzz, float **restrict ch1dxy,
                                   float **restrict ch1dyz, float **restrict ch1dxz, float **restrict v2px, float **restrict v2pz, float **restrict v2sz,
                                   float **restrict v2pn, int sx, int sy, int sz)
{
    const size_t sxsysz = ((size_t)sx * sy) * sz;
    const size_t msize_vol = sxsysz * sizeof(float);
    HIP_CALL(hipHostMalloc(ch1dxx, msize_vol));
    HIP_CALL(hipHostMalloc(ch1dyy, msize_vol));
    HIP_CALL(hipHostMalloc(ch1dzz, msize_vol));
    HIP_CALL(hipHostMalloc(ch1dxy, msize_vol));
    HIP_CALL(hipHostMalloc(ch1dyz, msize_vol));
    HIP_CALL(hipHostMalloc(ch1dxz, msize_vol));
    HIP_CALL(hipHostMalloc(v2px, msize_vol));
    HIP_CALL(hipHostMalloc(v2pz, msize_vol));
    HIP_CALL(hipHostMalloc(v2sz, msize_vol));
    HIP_CALL(hipHostMalloc(v2pn, msize_vol));
}


void HIP_Allocate_main(float **restrict vpz, float **restrict vsv, float **restrict epsilon, float **restrict delta,
                       float **restrict phi, float **restrict theta, float **restrict pp, float **restrict pc, float **restrict qp,
                       float **restrict qc, int sx, int sy, int sz)
{
    int sxsy = sx * sy;
    const size_t sxsysz = ((size_t)sx * sy) * sz;
    const size_t msize_vol = sxsysz * sizeof(float);
    const size_t msize_vol_extra = msize_vol + 2 * sx * sy * sizeof(float); // 2 extra plans for wave fields
    HIP_CALL(hipHostMalloc(vpz, msize_vol));
    HIP_CALL(hipHostMalloc(vsv, msize_vol));
    HIP_CALL(hipHostMalloc(epsilon, msize_vol));
    HIP_CALL(hipHostMalloc(delta, msize_vol));
    HIP_CALL(hipHostMalloc(phi, msize_vol));
    HIP_CALL(hipHostMalloc(theta, msize_vol));

    HIP_CALL(hipHostMalloc(pp, msize_vol_extra)); // Allocate extra space for wave fields
    HIP_CALL(hipHostMalloc(pc, msize_vol_extra)); // Allocate extra space for wave fields
    HIP_CALL(hipHostMalloc(qp, msize_vol_extra)); // Allocate extra space for wave fields
    HIP_CALL(hipHostMalloc(qc, msize_vol_extra)); // Allocate extra space for wave fields
    // ARTHUR - Ver se esta operação fica na CPU ou mover para a GPU.
    memset(*pp, 0, msize_vol_extra);
    memset(*pc, 0, msize_vol_extra);
    memset(*qp, 0, msize_vol_extra);
    memset(*qc, 0, msize_vol_extra);
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

      gpu_map[1].lower_kernel1 = sz / 4;
      gpu_map[1].upper_kernel1 = sz / 4 + 5;
      gpu_map[1].lower_kernel2 = 10;
      gpu_map[1].upper_kernel2 = sz / 4;

      gpu_map[2].lower_kernel1 = sz / 4;
      gpu_map[2].upper_kernel1 = sz / 4 + 5;
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
   else if (number_gpu == 8)
   {
      gpu_map[0].lower_kernel1 = sz / 8 - 5;
      gpu_map[0].upper_kernel1 = sz / 8;
      gpu_map[0].lower_kernel2 = 5;
      gpu_map[0].upper_kernel2 = sz / 8 - 5;

      gpu_map[1].lower_kernel1 = sz / 8;
      gpu_map[1].upper_kernel1 = sz / 8 + 5;
      gpu_map[1].lower_kernel2 = 10;
      gpu_map[1].upper_kernel2 = sz / 8;

      gpu_map[2].lower_kernel1 = sz / 8;
      gpu_map[2].upper_kernel1 = sz / 8 + 5;
      gpu_map[2].lower_kernel2 = 10;
      gpu_map[2].upper_kernel2 = sz / 8;

      gpu_map[3].lower_kernel1 = sz / 8;
      gpu_map[3].upper_kernel1 = sz / 8 + 5;
      gpu_map[3].lower_kernel2 = 10;
      gpu_map[3].upper_kernel2 = sz / 8;

      gpu_map[4].lower_kernel1 = sz / 8;
      gpu_map[4].upper_kernel1 = sz / 8 + 5;
      gpu_map[4].lower_kernel2 = 10;
      gpu_map[4].upper_kernel2 = sz / 8;

      gpu_map[5].lower_kernel1 = sz / 8;
      gpu_map[5].upper_kernel1 = sz / 8 + 5;
      gpu_map[5].lower_kernel2 = 10;
      gpu_map[5].upper_kernel2 = sz / 8;

      gpu_map[6].lower_kernel1 = sz / 8;
      gpu_map[6].upper_kernel1 = sz / 8 + 5;
      gpu_map[6].lower_kernel2 = 10;
      gpu_map[6].upper_kernel2 = sz / 8;

      gpu_map[7].lower_kernel1 = 5;
      gpu_map[7].upper_kernel1 = 10;
      gpu_map[7].lower_kernel2 = 10;
      gpu_map[7].upper_kernel2 = sz / 8 - 1;
   }
}

const char* get_default_device_name() {
    static char device_name[256];
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // Pegando propriedades do device padrão
    strncpy(device_name, deviceProp.name, sizeof(device_name));
    return device_name;
}
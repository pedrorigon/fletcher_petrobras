#include "openacc_propagate.h"
#include "../derivatives.h"
#include "../map.h"
#include <openacc.h>

// Helper function to get the number of GPUs
// int get_num_gpus()
//{
//  return acc_get_num_devices(acc_device_nvidia);
//}

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time

void OPENACC_Propagate(int sx, int sy, int sz, int bord,
                       float dx, float dy, float dz, float dt, int it,
                       float *restrict pp, float *restrict pc, float *restrict qp, float *restrict qc)
{
#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP

  int num_gpus = get_num_gpus();
  printf("Number of GPUs: %d\n", num_gpus);
  int chunk_size = (sz - 2 * bord) / num_gpus;

  // Launch computation on multiple GPUs
  for (int gpu = 0; gpu < num_gpus; gpu++)
  {
    int start_z = bord + gpu * chunk_size;
    int end_z = (gpu == num_gpus - 1) ? (sz - bord) : (start_z + chunk_size);

#pragma acc set device_num(gpu)

#pragma acc kernels default(present) async(gpu)
    { // start acc

      // solve both equations in all internal grid points,
      // including absortion zone

#pragma acc loop independent
      for (int iz = start_z; iz < end_z; iz++)
      {
#pragma acc loop independent
        for (int iy = bord; iy < sy - bord; iy++)
        {
#pragma acc loop independent
          for (int ix = bord; ix < sx - bord; ix++)
          {

#define SAMPLE_LOOP
#include "../sample.h"
#undef SAMPLE_LOOP
          }
        }
      }
    } // end acc
  }

  // Wait for all GPUs to finish
  for (int gpu = 0; gpu < num_gpus; gpu++)
  {
#pragma acc wait(gpu)
  }
}

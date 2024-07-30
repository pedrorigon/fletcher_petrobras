#include "openacc_insertsource.h"
#include <openacc.h>

// Helper function to get the number of GPUs
int get_num_gpus()
{
  return acc_get_num_devices(acc_device_nvidia);
}

// InsertSource: compute and insert source value at index iSource of arrays p and q

// Function to insert source into the domain on each GPU
void OPENACC_InsertSource(float dt, int it, int iSource,
                          float *p, float *q, float src, const int sx, const int sy, const int sz)
{
  int num_gpus = get_num_gpus();
  int chunk_size = sz / num_gpus;

  // Determine which GPU should handle the source insertion
  for (int gpu = 0; gpu < num_gpus; gpu++)
  {
    int start_z = gpu * chunk_size;
    int end_z = (gpu == num_gpus - 1) ? sz : (start_z + chunk_size);

    // Check if the source is within the range of this GPU
    if (iSource >= start_z * sx * sy && iSource < end_z * sx * sy)
    {
#pragma acc set device_num(gpu)
#pragma acc parallel default(present)
      {
        p[iSource] += src;
        q[iSource] += src;
      }
    }
  }
}
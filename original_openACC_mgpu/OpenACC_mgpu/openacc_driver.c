#include "../driver.h"
#include "openacc_propagate.h"
#include "openacc_insertsource.h"
#include "../sample.h"

extern float *ch1dxx, *ch1dyy, *ch1dzz, *ch1dxy, *ch1dyz, *ch1dxz, *v2px, *v2pz, *v2sz, *v2pn;

void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
                       float dx, float dy, float dz, float dt,
                       float *restrict vpz, float *restrict vsv, float *restrict epsilon, float *restrict delta,
                       float *restrict phi, float *restrict theta,
                       float *restrict pp, float *restrict pc, float *restrict qp, float *restrict qc)
{
  int num_gpus = get_num_gpus();

  for (int gpu = 0; gpu < num_gpus; gpu++)
  {
    #pragma acc set device_num(gpu)
    #pragma acc enter data copyin(ch1dxx[0 : sx * sy * sz], ch1dyy[0 : sx * sy * sz], ch1dzz[0 : sx * sy * sz], ch1dxy[0 : sx * sy * sz], ch1dyz[0 : sx * sy * sz], ch1dxz[0 : sx * sy * sz], v2px[0 : sx * sy * sz], v2pz[0 : sx * sy * sz], v2sz[0 : sx * sy * sz], v2pn[0 : sx * sy * sz])
    #pragma acc enter data copyin(pp[0 : sx * sy * sz], pc[0 : sx * sy * sz], qp[0 : sx * sy * sz], qc[0 : sx * sy * sz])
  }
}

void DRIVER_Finalize()
{
}

void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
	int num_gpus = get_num_gpus();
	int chunk_size = sz / num_gpus;

	for (int gpu = 0; gpu < num_gpus; gpu++)
	{
		int start_z = gpu * chunk_size;
		int end_z = (gpu == num_gpus - 1) ? sz : (start_z + chunk_size);
		int size = (end_z - start_z) * sx * sy;

#pragma acc set device_num(gpu)
#pragma acc update host(pc[start_z * sx * sy : size])
	}
}

void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
					  const float dx, const float dy, const float dz, const float dt, const int it,
					  float *restrict pp, float *restrict pc, float *restrict qp, float *restrict qc)
{

	OPENACC_Propagate(sx, sy, sz, bord,
					  dx, dy, dz, dt, it,
					  pp, pc, qp, qc);
}

void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float *q, float src, const int sx, const int sy, const int sz)
{

	OPENACC_InsertSource(dt, it, iSource, p, q, src, sx, sy, sz);
}

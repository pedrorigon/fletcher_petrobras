#include "../driver.h"
#include "openacc_propagate.h"
#include "openacc_insertsource.h"
#include "../zTileSize.h"

extern float *ch1dxx, *ch1dyy, *ch1dzz, *ch1dxy, *ch1dyz, *ch1dxz, *v2px, *v2pz, *v2sz, *v2pn;

float * restrict pDx = NULL;
float * restrict pDy = NULL;
float * restrict qDx = NULL;
float * restrict qDy = NULL;

void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
		       float dx, float dy, float dz, float dt,
		       float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
		       float * restrict phi, float * restrict theta,
		       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

#pragma acc enter data copyin(ch1dxx[0:sx*sy*sz])
#pragma acc enter data copyin(ch1dyy[0:sx*sy*sz])
#pragma acc enter data copyin(ch1dzz[0:sx*sy*sz])
#pragma acc enter data copyin(ch1dxy[0:sx*sy*sz])
#pragma acc enter data copyin(ch1dyz[0:sx*sy*sz])
#pragma acc enter data copyin(ch1dxz[0:sx*sy*sz])
#pragma acc enter data copyin(  v2px[0:sx*sy*sz])
#pragma acc enter data copyin(  v2pz[0:sx*sy*sz])
#pragma acc enter data copyin(  v2sz[0:sx*sy*sz])
#pragma acc enter data copyin(  v2pn[0:sx*sy*sz])

#pragma acc enter data copyin(pp[0:sx*sy*sz])
#pragma acc enter data copyin(pc[0:sx*sy*sz])
#pragma acc enter data copyin(qp[0:sx*sy*sz])
#pragma acc enter data copyin(qc[0:sx*sy*sz])

  pDx=(float *) malloc(sx*sy*DIM_Z_TILE*sizeof(float));
#pragma acc enter data create(   pDx[0:sx*sy*DIM_Z_TILE])
  pDy=(float *) malloc(sx*sy*DIM_Z_TILE*sizeof(float));
#pragma acc enter data create(   pDy[0:sx*sy*DIM_Z_TILE])
  qDx=(float *) malloc(sx*sy*DIM_Z_TILE*sizeof(float));
#pragma acc enter data create(   qDx[0:sx*sy*DIM_Z_TILE])
  qDy=(float *) malloc(sx*sy*DIM_Z_TILE*sizeof(float));
#pragma acc enter data create(   qDy[0:sx*sy*DIM_Z_TILE])
}



void DRIVER_Finalize()
{
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{

#pragma acc update host(pc[0:sx*sy*sz])

}


void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it, 
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

	   OPENACC_PropagateDer1Der1LM (  sx,   sy,   sz,   bord,
	                              dx,   dy,   dz,   dt,   it,
	                              pp,   pc,   qp,   qc);

}


void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float*q, float src)
{

	        OPENACC_InsertSource(dt,it,iSource,p,q, src);

}


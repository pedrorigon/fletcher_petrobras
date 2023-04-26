#include "../driver.h"
#include "openmp_propagate.h"
#include "openmp_insertsource.h"
#include "../sample.h"

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

  pDx=(float *) malloc(sx*sy*sz*sizeof(float));
  pDy=(float *) malloc(sx*sy*sz*sizeof(float));
  qDx=(float *) malloc(sx*sy*sz*sizeof(float));
  qDy=(float *) malloc(sx*sy*sz*sizeof(float));
}


void DRIVER_Finalize()
{
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
}


void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it, 
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{

	OPENMP_PropagateDer1Der1HM (  sx,   sy,   sz,   bord,
                                  dx,   dy,   dz,   dt,   it,
                                  pp,   pc,   qp,   qc);

}


void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float*q, float src)
{
        OPENMP_InsertSource(dt,it,iSource,p,q,src);
}


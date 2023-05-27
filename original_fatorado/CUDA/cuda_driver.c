#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include"../driver.h"
#include"../sample.h"
#include"cuda_stuff.h"
#include"cuda_propagate.h"
#include"cuda_insertsource.h"

void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt,
               float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
               float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
               float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
               float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
               float * restrict phi, float * restrict theta,
               float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc){


#include"../precomp.h"
CUDA_Initialize(sx, sy, sz, bord, dx, dy, dz, dt, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc);

}


void DRIVER_Finalize(const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt,
               float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
               float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
               float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
               float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
               float * restrict phi, float * restrict theta,
               float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc)
{
	CUDA_Finalize(sx, sy, sz, bord, dx, dy, dz, dt, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc);
}


void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc)
{
	CUDA_Update_pointers(sx,sy,sz,pc);
}




void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
                    const float dx, const float dy, const float dz, const float dt, const int it, const float * const restrict ch1dxx,
         const float * const restrict ch1dyy, float * restrict ch1dzz, float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
         float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn, float * restrict pp, float * restrict pc,
         float * restrict qp, float * restrict qc)
{

	// CUDA_Propagate also does TimeForward
CUDA_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc);

}


void DRIVER_InsertSource(const float val, const int iSource, float * restrict pc, float * restrict qc, float * restrict pp, float * restrict qp){
	CUDA_InsertSource(val, iSource, pc, qc, pp, qp);
}

void DRIVER_Allocate_Model_Variables(float ** restrict ch1dxx, float ** restrict ch1dyy, float ** restrict ch1dzz, float ** restrict ch1dxy,
                    float ** restrict ch1dyz, float ** restrict ch1dxz, float ** restrict v2px, float ** restrict v2pz, float ** restrict v2sz,
                        float ** restrict v2pn, int sx, int sy, int sz){

 CUDA_Allocate_Model_Variables(ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, sx, sy, sz);

}


void DRIVER_Allocate_main(float ** restrict vpz, float ** restrict vsv, float ** restrict epsilon, float ** restrict delta,
                    float ** restrict phi, float ** restrict theta, float ** restrict pp, float ** restrict pc, float ** restrict qp,
                        float ** restrict qc, int sx, int sy, int sz){

CUDA_Allocate_main(vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc, sx, sy, sz);

}

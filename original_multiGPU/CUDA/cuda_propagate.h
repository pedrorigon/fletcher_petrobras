#ifndef __CUDA_PROPAGATE
#define __CUDA_PROPAGATE

#ifdef __cplusplus
extern "C" {
#endif


// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void CUDA_Propagate(const int sx, const int sy, const int sz, const int bord,
		    const float dx, const float dy, const float dz, const float dt, const int it, const float * const restrict ch1dxx, 
         const float * const restrict ch1dyy, float * restrict ch1dzz, float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
         float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn, float * restrict pp, float * restrict pc, 
         float * restrict qp, float * restrict qc);

void CUDA_SwapArrays(float **pp, float **pc, float **qp, float **qc);

#ifdef __cplusplus
}
#endif

#endif

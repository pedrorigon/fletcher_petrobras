#include "openmp_propagate.h"
#include "../derivatives.h"
#include "../map.h"


// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time

void OPENMP_PropagateDer1Der1HM(int sx, int sy, int sz, int bord,
	       float dx, float dy, float dz, float dt, int it, 
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc) {


#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP


#pragma omp parallel
  { // start omp

    // initialize first derivative of p on x

#pragma omp for nowait
    for (int iz=0; iz<2*bord; iz++) {
      for (int iy=0; iy<sy; iy++) {
#pragma omp simd
	for (int i=ind(bord,iy,iz); i<ind(sx-bord,iy,iz); i++) {
#define SAMPLE_LOOP_1
#include "../sample.h"
#undef SAMPLE_LOOP_1
	}
      }
    }

    // initialize first derivative of p on y

#pragma omp for nowait
    for (int iz=0; iz<2*bord; iz++) {
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma omp simd
	for (int i=ind(bord,iy,iz); i<ind(sx-bord,iy,iz); i++) {
#define SAMPLE_LOOP_2
#include "../sample.h"
#undef SAMPLE_LOOP_2
	}
      }
    }

    // initialize first derivative of q on x

#pragma omp for nowait
    for (int iz=0; iz<2*bord; iz++) {
      for (int iy=0; iy<sy; iy++) {
#pragma omp simd
	for (int i=ind(bord,iy,iz); i<ind(sx-bord,iy,iz); i++) {
#define SAMPLE_LOOP_3
#include "../sample.h"
#undef SAMPLE_LOOP_3
	}
      }
    }

    // initialize first derivative of q on y

#pragma omp for
    for (int iz=0; iz<2*bord; iz++) {
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma omp simd
	for (int i=ind(bord,iy,iz); i<ind(sx-bord,iy,iz); i++) {
#define SAMPLE_LOOP_4
#include "../sample.h"
#undef SAMPLE_LOOP_4
	}
      }
    }

    
    // solve both equations in all internal grid points, 
    // including absortion zone
    
    
    for (int iz=bord; iz<sz-bord; iz++) {

      // required pDx for this iz

#pragma omp for nowait
      for (int iy=0; iy<sy; iy++) {
#pragma omp simd
	for (int i=ind(bord,iy,iz+bord); i<ind(sx-bord,iy,iz+bord); i++) {
#define SAMPLE_LOOP_1
#include "../sample.h"
#undef SAMPLE_LOOP_1
	}
      }

      // required pDy for this iz

#pragma omp for nowait
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma omp simd
	for (int i=ind(bord,iy,iz+bord); i<ind(sx-bord,iy,iz+bord); i++) {
#define SAMPLE_LOOP_2
#include "../sample.h"
#undef SAMPLE_LOOP_2
	}
      }


      // required qDx for this iz

#pragma omp for nowait
      for (int iy=0; iy<sy; iy++) {
#pragma omp simd
	for (int i=ind(bord,iy,iz+bord); i<ind(sx-bord,iy,iz+bord); i++) {
#define SAMPLE_LOOP_3
#include "../sample.h"
#undef SAMPLE_LOOP_3
	}
      }

      // required qDy for this iz

#pragma omp for
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma omp simd
	for (int i=ind(bord,iy,iz+bord); i<ind(sx-bord,iy,iz+bord); i++) {
#define SAMPLE_LOOP_4
#include "../sample.h"
#undef SAMPLE_LOOP_4
	}
      }

#pragma omp for
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma omp simd
	for (int ix=bord; ix<sx-bord; ix++) {
#define SAMPLE_LOOP_5
#include "../sample.h"
#undef SAMPLE_LOOP_5
	}
      }
    }
  } // end omp
}

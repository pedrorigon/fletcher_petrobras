#include "openacc_propagate.h"
#include "../derivatives.h"
#include "../map.h"


// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time


void OPENACC_PropagateDer1Der1HM(int sx, int sy, int sz, int bord,
	       float dx, float dy, float dz, float dt, int it, 
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc) {


#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP


#pragma acc kernels default(present) 
  { // start acc

    // initialize first derivative of p on x

#pragma acc loop independent
    for (int iz=0; iz<2*bord; iz++) {
#pragma acc loop independent
      for (int iy=0; iy<sy; iy++) {
#pragma acc loop independent
	for (int i=ind(bord,iy,iz); i<ind(sx-bord,iy,iz); i++) {
#define SAMPLE_LOOP_1
#include "../sample.h"
#undef SAMPLE_LOOP_1
	}
      }
    }

    // initialize first derivative of p on y

#pragma acc loop independent
    for (int iz=0; iz<2*bord; iz++) {
#pragma acc loop independent
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma acc loop independent
	for (int i=ind(bord,iy,iz); i<ind(sx-bord,iy,iz); i++) {
#define SAMPLE_LOOP_2
#include "../sample.h"
#undef SAMPLE_LOOP_2
	}
      }
    }

    // initialize first derivative of q on x

#pragma acc loop independent
    for (int iz=0; iz<2*bord; iz++) {
#pragma acc loop independent
      for (int iy=0; iy<sy; iy++) {
#pragma acc loop independent
	for (int i=ind(bord,iy,iz); i<ind(sx-bord,iy,iz); i++) {
#define SAMPLE_LOOP_3
#include "../sample.h"
#undef SAMPLE_LOOP_3
	}
      }
    }

    // initialize first derivative of q on y

#pragma acc loop independent
    for (int iz=0; iz<2*bord; iz++) {
#pragma acc loop independent
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma acc loop independent
	for (int i=ind(bord,iy,iz); i<ind(sx-bord,iy,iz); i++) {
#define SAMPLE_LOOP_4
#include "../sample.h"
#undef SAMPLE_LOOP_4
	}
      }
    }
//  }

    
  // solve both equations in all internal grid points, 
  // including absortion zone

//#pragma acc loop independent
    for (int iz=bord; iz<sz-bord; iz++) {

      // required pDx for this iz

#pragma acc loop independent
      for (int iy=0; iy<sy; iy++) {
#pragma acc loop independent
	for (int i=ind(bord,iy,iz+bord); i<ind(sx-bord,iy,iz+bord); i++) {
#define SAMPLE_LOOP_1
#include "../sample.h"
#undef SAMPLE_LOOP_1
	}
      }

      // required pDy for this iz

#pragma acc loop independent
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma acc loop independent
	for (int i=ind(bord,iy,iz+bord); i<ind(sx-bord,iy,iz+bord); i++) {
#define SAMPLE_LOOP_2
#include "../sample.h"
#undef SAMPLE_LOOP_2
	}
      }


      // required qDx for this iz

#pragma acc loop independent
      for (int iy=0; iy<sy; iy++) {
#pragma acc loop independent
	for (int i=ind(bord,iy,iz+bord); i<ind(sx-bord,iy,iz+bord); i++) {
#define SAMPLE_LOOP_3
#include "../sample.h"
#undef SAMPLE_LOOP_3
	}
      }

      // required qDy for this iz

#pragma acc loop independent
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma acc loop independent
	for (int i=ind(bord,iy,iz+bord); i<ind(sx-bord,iy,iz+bord); i++) {
#define SAMPLE_LOOP_4
#include "../sample.h"
#undef SAMPLE_LOOP_4
	}
      }

#pragma acc loop independent
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma acc loop independent
	for (int ix=bord; ix<sx-bord; ix++) {
#define SAMPLE_LOOP_5
#include "../sample.h"
#undef SAMPLE_LOOP_5
	}
      }
    } 
  } // end acc
}

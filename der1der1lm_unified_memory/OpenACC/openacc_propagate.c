#include "openacc_propagate.h"
#include "../derivatives.h"
#include "../map.h"
#include "../zTileSize.h"

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time


void OPENACC_PropagateDer1Der1LM(int sx, int sy, int sz, int bord,
	       float dx, float dy, float dz, float dt, int it, 
	       float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc) {


#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP


#pragma acc parallel default(present) 
  { // start acc

    const int itz=0;

    // initialize first derivative of p on x

#pragma acc loop independent collapse(2)
    for (int iz=0; iz<2*bord; iz++) {
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

#pragma acc loop independent collapse(2)
    for (int iz=0; iz<2*bord; iz++) {
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

#pragma acc loop independent collapse(2)
    for (int iz=0; iz<2*bord; iz++) {
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

#pragma acc loop independent collapse(2)
    for (int iz=0; iz<2*bord; iz++) {
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma acc loop independent
	for (int i=ind(bord,iy,iz); i<ind(sx-bord,iy,iz); i++) {
#define SAMPLE_LOOP_4
#include "../sample.h"
#undef SAMPLE_LOOP_4
	}
      }
    }
  } // end acc

    
  // solve both equations in all internal grid points, 
  // including absortion zone
  
  for (int iz=bord; iz<sz-bord; iz++) {

#pragma acc parallel default(present) 
    { // start acc

#define SAMPLE_LOOP_6
#include "../sample.h"
#undef SAMPLE_LOOP_6

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
    } // end acc

#pragma acc parallel default(present) 
    { // start acc

#define SAMPLE_LOOP_6
#include "../sample.h"
#undef SAMPLE_LOOP_6

#pragma acc loop independent
      for (int iy=bord; iy<sy-bord; iy++) {
#pragma acc loop independent
	for (int ix=bord; ix<sx-bord; ix++) {
	  const int i=ind(ix,iy,iz);

#define SAMPLE_LOOP_5
#include "../sample.h"
#undef SAMPLE_LOOP_5
	}
      }
    } 
  } // end acc
}

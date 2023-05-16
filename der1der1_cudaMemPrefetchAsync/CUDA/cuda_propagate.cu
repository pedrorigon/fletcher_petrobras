#include "cuda_defines.h"
#include "cuda_propagate.h"
#include "../derivatives.h"
#include "../map.h"

__global__ void kernel_pDx_pDy(const int sx, const int sy, const int sz, const int bord,
                               const float dxinv, const float dyinv,
                               const int strideX, const int strideY,
                               float *const restrict pDx, float *const restrict qDx,
                               float *const restrict pDy, float *const restrict qDy,
                               float *restrict pp, const float *const restrict pc,
                               float *restrict qp, const float *const restrict qc)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  // first derivative at x and y at all grid points

  for (int iz = 0; iz < sz; iz++)
  {

    const int i = ind(ix, iy, iz);

#define SAMPLE_LOOP_1
#include "../sample.h"
#undef SAMPLE_LOOP_1

#define SAMPLE_LOOP_2
#include "../sample.h"
#undef SAMPLE_LOOP_2

#define SAMPLE_LOOP_3
#include "../sample.h"
#undef SAMPLE_LOOP_3

#define SAMPLE_LOOP_4
#include "../sample.h"
#undef SAMPLE_LOOP_4
  }
}

__global__ void kernel_PropagateDer1Der1ORIG(const int sx, const int sy, const int sz, const int bord,
                                             const float dt, const float dyinv, const float dzinv,
                                             const float dxxinv, const float dyyinv, const float dzzinv,
                                             const int strideX, const int strideY, const int strideZ,
                                             const float *const restrict ch1dxx,
                                             const float *const restrict ch1dyy,
                                             const float *const restrict ch1dzz,
                                             const float *const restrict ch1dxy,
                                             const float *const restrict ch1dyz,
                                             const float *const restrict ch1dxz,
                                             float *restrict pDx, float *restrict qDx,
                                             float *restrict pDy, float *restrict qDy,
                                             float *restrict v2px, float *restrict v2pz, float *restrict v2sz,
                                             float *restrict v2pn, float *restrict pp, float *restrict pc,
                                             float *restrict qp, float *restrict qc)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  // solve both equations at a single grid point

  for (int iz = bord; iz < sz - bord; iz++)
  {

#define SAMPLE_LOOP_5
#include "../sample.h"
#undef SAMPLE_LOOP_5
  }
}

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time

void CUDA_Propagate(const int sx, const int sy, const int sz, const int bord,
                    const float dx, const float dy, const float dz, const float dt, const int it,
                    float *restrict pp, float *restrict pc, float *restrict qp, float *restrict qc)
{

  extern float *dev_vpz;
  extern float *dev_vsv;
  extern float *dev_epsilon;
  extern float *dev_delta;
  extern float *dev_phi;
  extern float *dev_theta;
  extern float *dev_ch1dxx;
  extern float *dev_ch1dyy;
  extern float *dev_ch1dzz;
  extern float *dev_ch1dxy;
  extern float *dev_ch1dyz;
  extern float *dev_ch1dxz;
  extern float *dev_v2px;
  extern float *dev_v2pz;
  extern float *dev_v2sz;
  extern float *dev_v2pn;
  extern float *dev_pp;
  extern float *dev_pc;
  extern float *dev_qp;
  extern float *dev_qc;
  extern float *dev_pDx;
  extern float *dev_pDy;
  extern float *dev_qDx;
  extern float *dev_qDy;

  dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
  dim3 numBlocks(sx / threadsPerBlock.x, sy / threadsPerBlock.y);

#define SAMPLE_PRE_LOOP
#include "../sample.h"
#undef SAMPLE_PRE_LOOP

  kernel_pDx_pDy<<<numBlocks, threadsPerBlock>>>(sx, sy, sz, bord,
                                                 dxinv, dyinv,
                                                 strideX, strideY,
                                                 dev_pDx, dev_qDx,
                                                 dev_pDy, dev_qDy,
                                                 dev_pp, dev_pc,
                                                 dev_qp, dev_qc);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  kernel_PropagateDer1Der1ORIG<<<numBlocks, threadsPerBlock>>>(sx, sy, sz, bord,
                                                               dt, dyinv, dzinv,
                                                               dxxinv, dyyinv, dzzinv,
                                                               strideX, strideY, strideZ,
                                                               dev_ch1dxx, dev_ch1dyy, dev_ch1dzz,
                                                               dev_ch1dxy, dev_ch1dyz, dev_ch1dxz,
                                                               dev_pDx, dev_qDx, dev_pDy, dev_qDy,
                                                               dev_v2px, dev_v2pz, dev_v2sz, dev_v2pn,
                                                               dev_pp, dev_pc, dev_qp, dev_qc);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_SwapArrays(&dev_pp, &dev_pc, &dev_qp, &dev_qc);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());
}

// swap array pointers on time forward array propagation
void CUDA_SwapArrays(float **pp, float **pc, float **qp, float **qc)
{
  float *tmp;

  tmp = *pp;
  *pp = *pc;
  *pc = tmp;

  tmp = *qp;
  *qp = *qc;
  *qc = tmp;
}

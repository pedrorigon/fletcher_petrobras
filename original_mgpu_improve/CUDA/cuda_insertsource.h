#ifndef __CUDA_SOURCE
#define __CUDA_SOURCE

#ifdef __cplusplus
extern "C" {
#endif

void CUDA_InsertSource(const float val, const int iSource, float * restrict pc, float * restrict qc, float * restrict pp, float * restrict qp);

#ifdef __cplusplus
}
#endif

#endif

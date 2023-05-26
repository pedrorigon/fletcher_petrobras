#ifndef _CUDA_STUFF
#define _CUDA_STUFF

#ifdef __cplusplus
extern "C"
{
#endif

    void CUDA_Initialize(const int sx, const int sy, const int sz, const int bord,
                         float dx, float dy, float dz, float dt,
                         float *restrict ch1dxx, float *restrict ch1dyy, float *restrict ch1dzz,
                         float *restrict ch1dxy, float *restrict ch1dyz, float *restrict ch1dxz,
                         float *restrict v2px, float *restrict v2pz, float *restrict v2sz, float *restrict v2pn,
                         float *restrict vpz, float *restrict vsv, float *restrict epsilon, float *restrict delta,
                         float *restrict phi, float *restrict theta,
                         float *restrict pp, float *restrict pc, float *restrict qp, float *restrict qc);

    void CUDA_Finalize(float *restrict vpz, float *restrict vsv, float *restrict epsilon, float *restrict delta,
                       float *restrict, float *restrict theta);

    void CUDA_Update_pointers(const int sx, const int sy, const int sz, float *pc);

    void alloc_data(float **vpz, float **vsv, float **epsilon, float **delta, float **phi, float **theta,
                    float **pp, float **pc, float **qp, float **qc, int sx, int sy, int sz);

#ifdef __cplusplus
}
#endif
#endif

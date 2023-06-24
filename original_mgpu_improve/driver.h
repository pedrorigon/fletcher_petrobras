#ifndef __driver_h__
#define __driver_h__
#define GPU_NUMBER 2

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
      size_t gpu_upper_x;
      size_t gpu_upper_y;
      size_t gpu_upper_z;
      size_t gpu_lower_x;
      size_t gpu_lower_y;
      size_t gpu_lower_z;
      size_t gpu_size_bord;
      size_t gpu_payload;
      size_t gpu_start_pointer;
      size_t gpu_end_pointer;
      size_t cpu_lower;
      size_t cpu_upper;
      size_t cpu_start_pointer;
      size_t cpu_end_pointer;
      size_t gpu_size_gpu;
      size_t cpu_offset;
      size_t cpu_z_start_compute;
      size_t cpu_z_end_compute;
      size_t cpu_z_start_read;
      size_t cpu_z_end_read;


   } Gpu;

void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt,
               float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
               float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
               float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
               float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
               float * restrict phi, float * restrict theta, 
               float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc);


void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
		    const float dx, const float dy, const float dz, const float dt, const int it, const float * const restrict ch1dxx, 
         const float * const restrict ch1dyy, float * restrict ch1dzz, float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz, 
         float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn, float * restrict pp, float * restrict pc, 
         float * restrict qp, float * restrict qc);

void DRIVER_Finalize(const int sx, const int sy, const int sz, const int bord,
               float dx, float dy, float dz, float dt,
               float * restrict ch1dxx, float * restrict ch1dyy, float * restrict ch1dzz,
               float * restrict ch1dxy, float * restrict ch1dyz, float * restrict ch1dxz,
               float * restrict v2px, float * restrict v2pz, float * restrict v2sz, float * restrict v2pn,
               float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta,
               float * restrict phi, float * restrict theta, 
               float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc);

void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc);

void DRIVER_InsertSource(const float val, const int iSource, float * restrict pc, float * restrict qc, float * restrict pp, float * restrict qp);

void DRIVER_Allocate_Model_Variables(float ** restrict ch1dxx, float ** restrict ch1dyy, float ** restrict ch1dzz, float ** restrict ch1dxy,
		    float ** restrict ch1dyz, float ** restrict ch1dxz, float ** restrict v2px, float ** restrict v2pz, float ** restrict v2sz,
		        float ** restrict v2pn, int sx, int sy, int sz);

void DRIVER_Allocate_main(float ** restrict vpz, float ** restrict vsv, float ** restrict epsilon, float ** restrict delta,
		    float ** restrict phi, float ** restrict theta, float ** restrict pp, float ** restrict pc, float ** restrict qp,
		        float ** restrict qc, int sx, int sy, int sz);

#ifdef __cplusplus
}
#endif
#endif

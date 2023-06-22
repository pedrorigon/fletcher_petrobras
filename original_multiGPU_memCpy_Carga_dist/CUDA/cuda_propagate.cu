#include "cuda_defines.h"
#include "cuda_propagate.h"
#include "../derivatives.h"
#include "../driver.h"
#include "../map.h"

__global__ void kernel_Propagate(const int sx, const int sy, const int sz, const int bord,
                                 const float dx, const float dy, const float dz, const float dt,
                                 const int it, const float *const restrict ch1dxx,
                                 const float *const restrict ch1dyy, float *restrict ch1dzz,
                                 float *restrict ch1dxy, float *restrict ch1dyz, float *restrict ch1dxz,
                                 float *restrict v2px, float *restrict v2pz, float *restrict v2sz,
                                 float *restrict v2pn, float *restrict pp, float *restrict pc,
                                 float *restrict qp, float *restrict qc, const int lower, const int upper)
{

    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    const int strideX = ind(1, 0, 0) - ind(0, 0, 0);
    const int strideY = ind(0, 1, 0) - ind(0, 0, 0);
    const int strideZ = ind(0, 0, 1) - ind(0, 0, 0);

    const float dxxinv = 1.0f / (dx * dx);
    const float dyyinv = 1.0f / (dy * dy);
    const float dzzinv = 1.0f / (dz * dz);
    const float dxyinv = 1.0f / (dx * dy);
    const float dxzinv = 1.0f / (dx * dz);
    const float dyzinv = 1.0f / (dy * dz);

    // solve both equations in all internal grid points,
    // including absortion zone

    for (int iz = lower; iz < upper; iz++)
    {
        const int i = ind(ix, iy, iz);
        // p derivatives, H1(p) and H2(p)
        const float pxx = Der2(pc, i, strideX, dxxinv);
        const float pyy = Der2(pc, i, strideY, dyyinv);
        const float pzz = Der2(pc, i, strideZ, dzzinv);
        const float pxy = DerCross(pc, i, strideX, strideY, dxyinv);
        const float pyz = DerCross(pc, i, strideY, strideZ, dyzinv);
        const float pxz = DerCross(pc, i, strideX, strideZ, dxzinv);
        const float cpxx = ch1dxx[i] * pxx;
        const float cpyy = ch1dyy[i] * pyy;
        const float cpzz = ch1dzz[i] * pzz;
        const float cpxy = ch1dxy[i] * pxy;
        const float cpxz = ch1dxz[i] * pxz;
        const float cpyz = ch1dyz[i] * pyz;
        const float h1p = cpxx + cpyy + cpzz + cpxy + cpxz + cpyz;
        const float h2p = pxx + pyy + pzz - h1p;

        // q derivatives, H1(q) and H2(q)
        const float qxx = Der2(qc, i, strideX, dxxinv);
        const float qyy = Der2(qc, i, strideY, dyyinv);
        const float qzz = Der2(qc, i, strideZ, dzzinv);
        const float qxy = DerCross(qc, i, strideX, strideY, dxyinv);
        const float qyz = DerCross(qc, i, strideY, strideZ, dyzinv);
        const float qxz = DerCross(qc, i, strideX, strideZ, dxzinv);
        const float cqxx = ch1dxx[i] * qxx;
        const float cqyy = ch1dyy[i] * qyy;
        const float cqzz = ch1dzz[i] * qzz;
        const float cqxy = ch1dxy[i] * qxy;
        const float cqxz = ch1dxz[i] * qxz;
        const float cqyz = ch1dyz[i] * qyz;
        const float h1q = cqxx + cqyy + cqzz + cqxy + cqxz + cqyz;
        const float h2q = qxx + qyy + qzz - h1q;

        // p-q derivatives, H1(p-q) and H2(p-q)
        const float h1pmq = h1p - h1q;
        const float h2pmq = h2p - h2q;

        // rhs of p and q equations
        const float rhsp = v2px[i] * h2p + v2pz[i] * h1q + v2sz[i] * h1pmq;
        const float rhsq = v2pn[i] * h2p + v2pz[i] * h1q - v2sz[i] * h2pmq;

        // new p and q
        pp[i] = 2.0f * pc[i] - pp[i] + rhsp * dt * dt;
        qp[i] = 2.0f * qc[i] - qp[i] + rhsq * dt * dt;
    }
}

// Propagate: using Fletcher's equations, propagate waves one dt,
//            either forward or backward in time
void CUDA_Propagate(const int sx, const int sy, const int sz, const int bord,
                    const float dx, const float dy, const float dz, const float dt, const int it,
                    const float *const restrict ch1dxx, const float *const restrict ch1dyy,
                    float *restrict ch1dzz, float *restrict ch1dxy, float *restrict ch1dyz,
                    float *restrict ch1dxz, float *restrict v2px, float *restrict v2pz,
                    float *restrict v2sz, float *restrict v2pn, float *pp, float *pc,
                    float *qp, float *qc)
{

    extern float* dev_ch1dxx[GPU_NUMBER];
    extern float* dev_ch1dyy[GPU_NUMBER];
    extern float* dev_ch1dzz[GPU_NUMBER];
    extern float* dev_ch1dxy[GPU_NUMBER];
    extern float* dev_ch1dyz[GPU_NUMBER];
    extern float* dev_ch1dxz[GPU_NUMBER];
    extern float* dev_v2px[GPU_NUMBER];
    extern float* dev_v2pz[GPU_NUMBER];
    extern float* dev_v2sz[GPU_NUMBER];
    extern float* dev_v2pn[GPU_NUMBER];
    extern float* dev_pp[GPU_NUMBER];
    extern float* dev_pc[GPU_NUMBER];
    extern float* dev_qp[GPU_NUMBER];
    extern float* dev_qc[GPU_NUMBER];


    int num_gpus;
    int lower, upper;
    CUDA_CALL(cudaGetDeviceCount(&num_gpus));

    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        cudaDeviceProp prop;
        cudaSetDevice(gpu);
        CUDA_CALL(cudaGetDeviceProperties(&prop, gpu));

        if (gpu == 0)
        {
            lower = bord + 1;
            upper = sz / 2;
        }
        else
        {
            lower = sz / 2;
            upper = sz - bord - 1;
        }

        const int width = upper - lower;

        // Calcula o número de blocos e threads por bloco para a GPU atual
        dim3 threadsPerBlock(BSIZE_X, BSIZE_Y);
        dim3 numBlocks(sx / threadsPerBlock.x, sy / threadsPerBlock.y);

        // Executar o kernel no dispositivo da iteração
        kernel_Propagate<<<numBlocks, threadsPerBlock>>>(sx, sy, sz, bord, dx, dy, dz, dt, it, dev_ch1dxx[gpu], dev_ch1dyy[gpu],
                                                         dev_ch1dzz[gpu], dev_ch1dxy[gpu], dev_ch1dyz[gpu], dev_ch1dxz[gpu], dev_v2px[gpu], dev_v2pz[gpu], dev_v2sz[gpu],
                                                         dev_v2pn[gpu], dev_pp[gpu], dev_pc[gpu], dev_qp[gpu], dev_qc[gpu], lower, upper);
        
    }

    CUDA_CALL(cudaGetLastError());
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        CUDA_SwapBord(sx, sy, sz, dev_pp[gpu], dev_qp[gpu]);
    }
    CUDA_CALL(cudaDeviceSynchronize()); 
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        CUDA_SwapArrays(&dev_pp[gpu], &dev_pc[gpu], &dev_qp[gpu], &dev_qc[gpu]);
    }
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

void CUDA_SwapBord(const int sx, const int sy, const int sz, float* pc, float* qp){

    extern float* dev_pp[GPU_NUMBER];
    extern float* dev_qp[GPU_NUMBER];

    int deviceCount;
    CUDA_CALL(cudaGetDeviceCount(&deviceCount));
    const size_t sxsysz = ((size_t)sx * sy) * sz;
    const size_t msize_vol = sxsysz * sizeof(float);
    const size_t msize_vol_extra = msize_vol + 2 * sx*sy * sizeof(float); // 2 extra plans for wave fields
    const size_t msize_vol_half = msize_vol_extra / 2;
    const int size_space = (ind(0, 0 , sz/2) - ind(0, 0, (sz/2 - 4))) * sizeof(float);
    const int size_bord = ind(0, 0, (sz / 2));
    const int size_lower = ind(0,0,0);
    const int size_gpu0 = ind(0,0,(sz/2 - 4));
    const int size_gpu1 = ind(0,0,(sz/2 + 4));
    const int size_swap_gpu0 = size_bord - ind(0, 0, (sz/2 - 4));
    const int size_swap_gpu1 = ind(0,0, (sz/2 +4)) - size_bord;

    for (int device = 0; device < deviceCount; device++)
    {
        CUDA_CALL(cudaSetDevice(device));

        CUDA_CALL(cudaMemcpy(dev_pp[0] + size_bord, dev_pp[1] + size_bord, size_space, cudaMemcpyDeviceToDevice));
        CUDA_CALL(cudaMemcpy(dev_pp[1] + size_gpu0, dev_pp[0] + size_gpu0, size_space, cudaMemcpyDeviceToDevice));

        CUDA_CALL(cudaMemcpy(dev_qp[0] + size_bord, dev_qp[1] + size_bord, size_space, cudaMemcpyDeviceToDevice));
        CUDA_CALL(cudaMemcpy(dev_qp[1] + size_gpu0, dev_qp[0] + size_gpu0, size_space, cudaMemcpyDeviceToDevice));
        CUDA_CALL(cudaDeviceSynchronize()); 
    }
}
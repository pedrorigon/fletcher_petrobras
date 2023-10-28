#include "cuda_defines.h"
#include "cuda_propagate.h"
#include "../derivatives.h"
#include "../driver.h"
#include "../map.h"

cudaStream_t stream[8];
cudaStream_t swap_stream[8];
cudaStream_t compute_stream[8];
int g_peer_access_enabled = 0;
extern int number_gpu;

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
#pragma unroll
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
                    float *qp, float *qc, int bsize_x, int bsize_y)
{
    extern Gpu *gpu_map;
    extern float **dev_ch1dxx;
    extern float **dev_ch1dyy;
    extern float **dev_ch1dzz;
    extern float **dev_ch1dxy;
    extern float **dev_ch1dyz;
    extern float **dev_ch1dxz;
    extern float **dev_v2px;
    extern float **dev_v2pz;
    extern float **dev_v2sz;
    extern float **dev_v2pn;
    extern float **dev_pp;
    extern float **dev_pc;
    extern float **dev_qp;
    extern float **dev_qc;

    int lower, upper, gpu;
    // CUDA_CALL(cudaGetDeviceCount(&num_gpus));

    if (number_gpu == 1)
    {
        lower = bord + 1;
        upper = sz - bord - 1;
        gpu = 0;
    }

    for (gpu = 0; gpu < number_gpu; gpu++)
    {
        // cudaDeviceProp prop;
        cudaSetDevice(gpu);

        // Calcula o número de blocos e threads por bloco para a GPU atual
        dim3 threadsPerBlock(bsize_x, bsize_y);
        dim3 numBlocks(sx / threadsPerBlock.x, sy / threadsPerBlock.y);
        if (number_gpu != 1)
        {
            if (gpu == 0)
            {
                lower = gpu_map[0].lower_kernel1;
                upper = gpu_map[0].upper_kernel1;
            }
            else if (gpu == (number_gpu - 1))
            {
                lower = gpu_map[gpu].lower_kernel1;
                upper = gpu_map[gpu].lower_kernel1;
            }
            else
            {
                lower = gpu_map[gpu].lower_kernel1;
                upper = gpu_map[gpu].lower_kernel1;

                // Executar o kernel no dispositivo da iteração
                kernel_Propagate<<<numBlocks, threadsPerBlock, 0, compute_stream[gpu]>>>(sx, sy, sz, bord, dx, dy, dz, dt, it, dev_ch1dxx[gpu], dev_ch1dyy[gpu],
                                                                                         dev_ch1dzz[gpu], dev_ch1dxy[gpu], dev_ch1dyz[gpu], dev_ch1dxz[gpu], dev_v2px[gpu], dev_v2pz[gpu], dev_v2sz[gpu],
                                                                                         dev_v2pn[gpu], dev_pp[gpu], dev_pc[gpu], dev_qp[gpu], dev_qc[gpu], (bord + 1), (bord + 1 + 5));
            }
        }
        // Executar o kernel no dispositivo da iteração
        kernel_Propagate<<<numBlocks, threadsPerBlock>>>(sx, sy, sz, bord, dx, dy, dz, dt, it, dev_ch1dxx[gpu], dev_ch1dyy[gpu],
                                                         dev_ch1dzz[gpu], dev_ch1dxy[gpu], dev_ch1dyz[gpu], dev_ch1dxz[gpu], dev_v2px[gpu], dev_v2pz[gpu], dev_v2sz[gpu],
                                                         dev_v2pn[gpu], dev_pp[gpu], dev_pc[gpu], dev_qp[gpu], dev_qc[gpu], lower, upper);
    }

    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    if (number_gpu == 1)
    {
        CUDA_SwapArrays(&dev_pp[0], &dev_pc[0], &dev_qp[0], &dev_qc[0]);
        CUDA_CALL(cudaDeviceSynchronize());
    }
    else
    {
        CUDA_SwapBord(sx, sy, sz);
        
        for (int gpu = 0; gpu < number_gpu; gpu++)
        {
            // cudaDeviceProp prop;
            cudaSetDevice(gpu);

            if (gpu == 0)
            {
                lower = gpu_map[0].lower_kernel2;
                upper = gpu_map[0].upper_kernel2;
            }
            else if (gpu == (number_gpu - 1))
            {
                lower = gpu_map[gpu].lower_kernel2;
                upper = gpu_map[gpu].upper_kernel2;
                ;
            }
            else
            {
                lower = gpu_map[gpu].lower_kernel2;
                upper = gpu_map[gpu].upper_kernel2;
            }

            const int width = upper - lower;

            // Calcula o número de blocos e threads por bloco para a GPU atual
            dim3 threadsPerBlock(bsize_x, bsize_y);
            dim3 numBlocks(sx / threadsPerBlock.x, sy / threadsPerBlock.y);

            // Executar o kernel no dispositivo da iteração
            kernel_Propagate<<<numBlocks, threadsPerBlock, 0, stream[gpu]>>>(sx, sy, sz, bord, dx, dy, dz, dt, it, dev_ch1dxx[gpu], dev_ch1dyy[gpu],
                                                                             dev_ch1dzz[gpu], dev_ch1dxy[gpu], dev_ch1dyz[gpu], dev_ch1dxz[gpu], dev_v2px[gpu], dev_v2pz[gpu], dev_v2sz[gpu],
                                                                             dev_v2pn[gpu], dev_pp[gpu], dev_pc[gpu], dev_qp[gpu], dev_qc[gpu], lower, upper);
        }
        for (int gpu = 0; gpu < number_gpu; gpu++)
        {
            CUDA_CALL(cudaStreamSynchronize(stream[gpu]));
        }
        // CUDA_CALL(cudaStreamSynchronize(stream[0]));
        // CUDA_CALL(cudaStreamSynchronize(stream[1]));
        // CUDA_CALL(cudaStreamSynchronize(stream[2]));
        // CUDA_CALL(cudaStreamSynchronize(stream[3]));

        for (int gpu = 0; gpu < number_gpu; gpu++)
        {
            CUDA_SwapArrays(&dev_pp[gpu], &dev_pc[gpu], &dev_qp[gpu], &dev_qc[gpu]);
        }
        CUDA_CALL(cudaDeviceSynchronize());

        for (int gpu = 0; gpu < number_gpu; gpu++)
        {
            CUDA_CALL(cudaStreamSynchronize(swap_stream[gpu]));
        }

        // CUDA_CALL(cudaStreamSynchronize(swap_stream[0]));
        // CUDA_CALL(cudaStreamSynchronize(swap_stream[1]));
        // CUDA_CALL(cudaStreamSynchronize(swap_stream[2]));
        // CUDA_CALL(cudaStreamSynchronize(swap_stream[3]));
    }
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

void CUDA_SwapBord(const int sx, const int sy, const int sz)
{

    extern float **dev_pp;
    extern float **dev_qp;
    extern Gpu *gpu_map;

    // Define sizes
    const int size_gpu0 = ind(0, 0, (sz / number_gpu - 5));
    const int size_med = ind(0, 0, (sz / number_gpu));

    if (number_gpu == 8)
    {
        // GPU 0 <-> GPU 1
        CUDA_CALL(cudaSetDevice(0));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[0] + gpu_map[0].gpu_end_pointer, 0, dev_pp[1] + gpu_map[1].gpu_start_pointer, 1, gpu_map[0].gpu_size_bord, swap_stream[0]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[0] + gpu_map[0].gpu_end_pointer, 0, dev_qp[1] + gpu_map[1].gpu_start_pointer, 1, gpu_map[0].gpu_size_bord, swap_stream[0]));

        CUDA_CALL(cudaSetDevice(1));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[1], 1, dev_pp[0] + size_gpu0, 0, gpu_map[1].gpu_size_bord, swap_stream[1]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[1], 1, dev_qp[0] + size_gpu0, 0, gpu_map[1].gpu_size_bord, swap_stream[1]));

        // GPU 1 <-> GPU 2
        CUDA_CALL(cudaSetDevice(1));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[1] + gpu_map[1].gpu_end_pointer, 1, dev_pp[2] + gpu_map[2].gpu_start_pointer, 2, gpu_map[1].gpu_size_bord, swap_stream[1]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[1] + gpu_map[1].gpu_end_pointer, 1, dev_qp[2] + gpu_map[2].gpu_start_pointer, 2, gpu_map[1].gpu_size_bord, swap_stream[1]));

        CUDA_CALL(cudaSetDevice(2));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[2], 2, dev_pp[1] + size_med, 1, gpu_map[2].gpu_size_bord, swap_stream[2]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[2], 2, dev_qp[1] + size_med, 1, gpu_map[2].gpu_size_bord, swap_stream[2]));

        // GPU 2 <-> GPU 3
        CUDA_CALL(cudaSetDevice(2));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[2] + gpu_map[2].gpu_end_pointer, 2, dev_pp[3] + gpu_map[3].gpu_start_pointer, 3, gpu_map[2].gpu_size_bord, swap_stream[2]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[2] + gpu_map[2].gpu_end_pointer, 2, dev_qp[3] + gpu_map[3].gpu_start_pointer, 3, gpu_map[2].gpu_size_bord, swap_stream[2]));

        CUDA_CALL(cudaSetDevice(3));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[3], 3, dev_pp[2] + size_med, 2, gpu_map[3].gpu_size_bord, swap_stream[3]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[3], 3, dev_qp[2] + size_med, 2, gpu_map[3].gpu_size_bord, swap_stream[3]));

        // GPU 3 <-> GPU 4
        CUDA_CALL(cudaSetDevice(3));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[3] + gpu_map[3].gpu_end_pointer, 3, dev_pp[4] + gpu_map[4].gpu_start_pointer, 4, gpu_map[3].gpu_size_bord, swap_stream[3]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[3] + gpu_map[3].gpu_end_pointer, 3, dev_qp[4] + gpu_map[4].gpu_start_pointer, 4, gpu_map[3].gpu_size_bord, swap_stream[3]));

        CUDA_CALL(cudaSetDevice(4));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[4], 4, dev_pp[3] + size_med, 3, gpu_map[4].gpu_size_bord, swap_stream[4]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[4], 4, dev_qp[3] + size_med, 3, gpu_map[4].gpu_size_bord, swap_stream[4]));

        // GPU 4 <-> GPU 5
        CUDA_CALL(cudaSetDevice(4));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[4] + gpu_map[4].gpu_end_pointer, 4, dev_pp[5] + gpu_map[5].gpu_start_pointer, 5, gpu_map[4].gpu_size_bord, swap_stream[4]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[4] + gpu_map[4].gpu_end_pointer, 4, dev_qp[5] + gpu_map[5].gpu_start_pointer, 5, gpu_map[4].gpu_size_bord, swap_stream[4]));

        CUDA_CALL(cudaSetDevice(5));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[5], 5, dev_pp[4] + size_med, 4, gpu_map[5].gpu_size_bord, swap_stream[5]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[5], 5, dev_qp[4] + size_med, 4, gpu_map[5].gpu_size_bord, swap_stream[5]));

        // GPU 5 <-> GPU 6
        CUDA_CALL(cudaSetDevice(5));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[5] + gpu_map[5].gpu_end_pointer, 5, dev_pp[6] + gpu_map[6].gpu_start_pointer, 6, gpu_map[5].gpu_size_bord, swap_stream[5]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[5] + gpu_map[5].gpu_end_pointer, 5, dev_qp[6] + gpu_map[6].gpu_start_pointer, 6, gpu_map[5].gpu_size_bord, swap_stream[5]));

        CUDA_CALL(cudaSetDevice(6));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[6], 6, dev_pp[5] + size_med, 5, gpu_map[6].gpu_size_bord, swap_stream[6]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[6], 6, dev_qp[5] + size_med, 5, gpu_map[6].gpu_size_bord, swap_stream[6]));

        // GPU 6 <-> GPU 7
        CUDA_CALL(cudaSetDevice(6));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[6] + gpu_map[6].gpu_end_pointer, 6, dev_pp[7] + gpu_map[7].gpu_start_pointer, 7, gpu_map[6].gpu_size_bord, swap_stream[6]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[6] + gpu_map[6].gpu_end_pointer, 6, dev_qp[7] + gpu_map[7].gpu_start_pointer, 7, gpu_map[6].gpu_size_bord, swap_stream[6]));

        CUDA_CALL(cudaSetDevice(7));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[7], 7, dev_pp[6] + size_med, 6, gpu_map[7].gpu_size_bord, swap_stream[7]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[7], 7, dev_qp[6] + size_med, 6, gpu_map[7].gpu_size_bord, swap_stream[7]));
    }
    else if (number_gpu == 4)
    {

        // GPU 0 <-> GPU 1
        CUDA_CALL(cudaSetDevice(0));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[0] + gpu_map[0].gpu_end_pointer, 0, dev_pp[1] + gpu_map[1].gpu_start_pointer, 1, gpu_map[0].gpu_size_bord, swap_stream[0]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[0] + gpu_map[0].gpu_end_pointer, 0, dev_qp[1] + gpu_map[1].gpu_start_pointer, 1, gpu_map[0].gpu_size_bord, swap_stream[0]));

        CUDA_CALL(cudaSetDevice(1));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[1], 1, dev_pp[0] + size_gpu0, 0, gpu_map[1].gpu_size_bord, swap_stream[1]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[1], 1, dev_qp[0] + size_gpu0, 0, gpu_map[1].gpu_size_bord, swap_stream[1]));

        // GPU 1 <-> GPU 2
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[1] + gpu_map[1].gpu_end_pointer, 1, dev_pp[2] + gpu_map[2].gpu_start_pointer, 2, gpu_map[1].gpu_size_bord, swap_stream[1]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[1] + gpu_map[1].gpu_end_pointer, 1, dev_qp[2] + gpu_map[2].gpu_start_pointer, 2, gpu_map[1].gpu_size_bord, swap_stream[1]));

        CUDA_CALL(cudaSetDevice(2));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[2], 2, dev_pp[1] + size_med, 1, gpu_map[2].gpu_size_bord, swap_stream[2]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[2], 2, dev_qp[1] + size_med, 1, gpu_map[2].gpu_size_bord, swap_stream[2]));

        // GPU 2 <-> GPU 3
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[2] + gpu_map[2].gpu_end_pointer, 2, dev_pp[3] + gpu_map[3].gpu_start_pointer, 3, gpu_map[2].gpu_size_bord, swap_stream[2]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[2] + gpu_map[2].gpu_end_pointer, 2, dev_qp[3] + gpu_map[3].gpu_start_pointer, 3, gpu_map[2].gpu_size_bord, swap_stream[2]));

        CUDA_CALL(cudaSetDevice(3));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[3], 3, dev_pp[2] + size_med, 2, gpu_map[3].gpu_size_bord, swap_stream[3]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[3], 3, dev_qp[2] + size_med, 2, gpu_map[3].gpu_size_bord, swap_stream[3]));
    }
    else if (number_gpu == 2)
    {
        CUDA_CALL(cudaSetDevice(0));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[0] + gpu_map[0].gpu_end_pointer, 0, dev_pp[1] + gpu_map[1].gpu_start_pointer, 1, gpu_map[0].gpu_size_bord, swap_stream[0]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[0] + gpu_map[0].gpu_end_pointer, 0, dev_qp[1] + gpu_map[1].gpu_start_pointer, 1, gpu_map[0].gpu_size_bord, swap_stream[0]));

        CUDA_CALL(cudaSetDevice(1));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_pp[1], 1, dev_pp[0] + size_gpu0, 0, gpu_map[1].gpu_size_bord, swap_stream[1]));
        CUDA_CALL(cudaMemcpyPeerAsync(dev_qp[1], 1, dev_qp[0] + size_gpu0, 0, gpu_map[1].gpu_size_bord, swap_stream[1]));
    }
}

void InitializeStreams()
{

    extern float **dev_pp;
    extern float **dev_qp;
    extern Gpu *gpu_map;

    for (int i = 1; i < number_gpu; i++)
    {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_CALL(cudaStreamCreate(&compute_stream[i])); // Initializing compute_stream for all GPUs
    }

    for (int i = 0; i < number_gpu; i++)
    {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_CALL(cudaStreamCreate(&stream[i])); // Initializing stream for all GPUs
    }

    // Check and enable peer access if not already enabled
    if (!g_peer_access_enabled)
    {
        for (int i = 0; i < number_gpu; i++)
        {
            CUDA_CALL(cudaSetDevice(i));
            CUDA_CALL(cudaStreamCreate(&swap_stream[i]));

            for (int j = 0; j < number_gpu; j++)
            {
                if (i != j)
                {
                    int can_access;
                    CUDA_CALL(cudaDeviceCanAccessPeer(&can_access, i, j));
                    if (can_access)
                    {
                        CUDA_CALL(cudaDeviceEnablePeerAccess(j, 0));
                    }
                }
            }
        }
        g_peer_access_enabled = 1;
    }
}
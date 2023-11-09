#include "utils.h"
#include "source.h"
#include "driver.h"
#include "fletcher.h"
#include "walltime.h"
#include "model.h"
#include "HIP/hip_stuff.h"
#include "HIP/hip_propagate.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <limits.h>
#define MEGA 1.0e-6
#define GIGA 1.0e-9

#ifdef PAPI
#include "ModPAPI.h"
#endif

#define MODEL_GLOBALVARS
#include "precomp.h"
#undef MODEL_GLOBALVARS

#define POPULATION_SIZE 30
#define MAX_NUM_THREADS 64
#define MAX_MULTIPLICATION 1024
#define TOURNAMENT_SIZE 2
#define MUTATION_Y_PROBABILITY 0.2
#define MUTATION_X_PROBABILITY 0.2

extern int number_gpu;
// preciso rodar o kernel com todas as configurações da população inicial
// a aptidão de cada indivíduo será 1/tempo de exec

void ReportProblemSizeCSV(const int sx, const int sy, const int sz, const int bord, const int st, FILE *f)
{
    fprintf(f, "sx; %d; sy; %d; sz; %d; bord; %d;  st; %d; \n", sx, sy, sz, bord, st);
}

void ReportMetricsCSV(double walltime, double MSamples, long HWM, char *HWMUnit, FILE *f)
{
    fprintf(f, "walltime; %lf; MSamples; %lf; HWM;  %ld; HWMUnit;  %s;\n", walltime, MSamples, HWM, HWMUnit);
}

void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr, const int sx, const int sy, const int sz, const int bord,
           const float dx, const float dy, const float dz, const float dt, const int it, float *restrict pp, float *restrict pc, float *restrict qp, float *restrict qc,
           float *restrict vpz, float *restrict vsv, float *restrict epsilon, float *restrict delta, float *restrict phi, float *restrict theta, int mgpu_number_input)
{

    int nOut = 1;
    int size = sx * sy * sz;
    float tSim = 0.0, tOut = nOut * dtOutput;

    const long samplesPropagate = (long)(sx - 2 * (bord)) * (long)(sy - 2 * (bord)) * (long)(sz - 2 * (bord));
    const long totalSamples = samplesPropagate * (long)st;

    float *ch1dxx = NULL; // isotropy simetry deep angle
    float *ch1dyy = NULL; // isotropy simetry deep angle
    float *ch1dzz = NULL; // isotropy simetry deep angle
    float *ch1dxy = NULL; // isotropy simetry deep angle
    float *ch1dyz = NULL; // isotropy simetry deep angle
    float *ch1dxz = NULL; // isotropy simetry deep angle
    float *v2px = NULL;   // coeficient of H2(p)
    float *v2pz = NULL;   // coeficient of H1(q)
    float *v2sz = NULL;   // coeficient of H1(p-q) and H2(p-q)
    float *v2pn = NULL;   // coeficient of H2(p)

    DRIVER_Allocate_Model_Variables(&ch1dxx, &ch1dyy, &ch1dzz, &ch1dxy, &ch1dyz, &ch1dxz, &v2px, &v2pz, &v2sz, &v2pn, sx, sy, sz);

    for (int i = 0; i < size; i++)
    {
        float sinTheta = sin(theta[i]);
        float cosTheta = cos(theta[i]);
        float sin2Theta = sin(2.0 * theta[i]);
        float sinPhi = sin(phi[i]);
        float cosPhi = cos(phi[i]);
        float sin2Phi = sin(2.0 * phi[i]);
        ch1dxx[i] = sinTheta * sinTheta * cosPhi * cosPhi;
        ch1dyy[i] = sinTheta * sinTheta * sinPhi * sinPhi;
        ch1dzz[i] = cosTheta * cosTheta;
        ch1dxy[i] = sinTheta * sinTheta * sin2Phi;
        ch1dyz[i] = sin2Theta * sinPhi;
        ch1dxz[i] = sin2Theta * cosPhi;
    }
#ifdef _DUMP
    {
        const int iPrint = ind(bord + 1, bord + 1, bord + 1);
        printf("ch1dxx=%f; ch1dyy=%f; ch1dzz=%f; ch1dxy=%f; ch1dxz=%f; ch1dyz=%f\n", ch1dxx[iPrint], ch1dyy[iPrint], ch1dzz[iPrint], ch1dxy[iPrint], ch1dxz[iPrint], ch1dyz[iPrint]);
    }
#endif

    // coeficients of H1 and H2 at PDEs
    for (int i = 0; i < size; i++)
    {
        v2sz[i] = vsv[i] * vsv[i];
        v2pz[i] = vpz[i] * vpz[i];
        v2px[i] = v2pz[i] * (1.0 + 2.0 * epsilon[i]);
        v2pn[i] = v2pz[i] * (1.0 + 2.0 * delta[i]);
    }

#ifdef _DUMP
    {
        const int iPrint = ind(bord + 1, bord + 1, bord + 1);
        printf("vsv=%e; vpz=%e, v2pz=%e\n", vsv[iPrint], vpz[iPrint], v2pz[iPrint]);
        printf("v2sz=%e; v2pz=%e, v2px=%e, v2pn=%e\n", v2sz[iPrint], v2pz[iPrint], v2px[iPrint], v2pn[iPrint]);
    }
#endif

    int gpu_qtd = mgpu_number_input;
    number_gpu = mgpu_number_input;
    printf("numero de GPUS %d\n", gpu_qtd);
    printf("numero de GPUS %d\n", number_gpu);
    initialize_mgpu(gpu_qtd, sz);

    // CUDA_Initialize initialize target, allocate data etc
    DRIVER_Initialize(sx, sy, sz, bord, dx, dy, dz, dt, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz,
                      v2px, v2pz, v2sz, v2pn, vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc); // ok Arthur

    double walltime = 0.0;
    double timeIt = 0.0;
    double res = 0.0;
    int bsize_x = 16, bsize_y = 16;

    InitializeStreams();

    // Loop para executar o kernel com cada configuração e coletar os tempos de execução

    for (int it = 1; it <= st; it++)
    {
        // Calculate / obtain source value on i timestep
        float src = Source(dt, it - 1);
        DRIVER_InsertSource(src, iSource, pc, qc, pp, qp);
        printf("\nBsize_x: %d \n", bsize_x);
        printf("Bsize_y: %d \n", bsize_y);
        const double t0 = wtime();

        DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, bsize_x, bsize_y);

        timeIt = wtime() - t0;
        walltime += timeIt;

        res = (MEGA * (double)samplesPropagate) / timeIt;
        printf("\nIteracao: %d ; Bsize_x: %d ; Bsize_y: %d ; tempoExec: %lf ; MSamples: %lf \n", it, bsize_x, bsize_y, timeIt, res);

        tSim = it * dt;
        if (tSim >= tOut)
        {
            // DRIVER_Update_pointers(sx,sy,sz,pc);
            // DumpSliceFile(sx,sy,sz,pc,sPtr);
            //  CUDA_prefetch_pc(sx,sy,sz,pc);

            tOut = (++nOut) * dtOutput;

        }
    }


const double MSamples = (MEGA * (double)totalSamples) / walltime;
printf("Execution time (s) is %lf\n", walltime);
printf("MSamples/s %.0lf\n", MSamples);
fflush(stdout);

// DRIVER_Finalize deallocate data, clean-up things etc
DRIVER_Finalize(sx, sy, sz, bord, dx, dy, dz, dt, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz,
                v2px, v2pz, v2sz, v2pn, vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc);
}

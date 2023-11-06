#include "utils.h"
#include "source.h"
#include "driver.h"
#include "fletcher.h"
#include "walltime.h"
#include "model.h"
#include "CUDA/cuda_stuff.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h> // for DBL_MAX
#include <limits.h>
#include <stdbool.h>

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
           float *restrict vpz, float *restrict vsv, float *restrict epsilon, float *restrict delta, float *restrict phi, float *restrict theta)
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

#define MODEL_INITIALIZE
#include "precomp.h"
#undef MODEL_INITIALIZE

  // CUDA_Initialize initialize target, allocate data etc
  DRIVER_Initialize(sx, sy, sz, bord,
                    dx, dy, dz, dt,
                    vpz, vsv, epsilon, delta,
                    phi, theta,
                    pp, pc, qp, qc);

  double walltime = 0.0;
  double timeIt = 0.0;
  double res = 0.0;
  int bsize_x = 128, bsize_y = 4;

  for (int it = 1; it <= st; it++)
  {
    // Calculate / obtain source value on i timestep
    float src = Source(dt, it - 1);
    DRIVER_InsertSource(dt, it - 1, iSource, pc, qc, src);

    printf("\nBsize_x: %d \n", bsize_x);
    printf("Bsize_y: %d \n", bsize_y);
    const double t0 = wtime();

    DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, pp, pc, qp, qc, bsize_x, bsize_y); // ajustar parametros
    // SwapArrays(&pp, &pc, &qp, &qc);

    timeIt = wtime() - t0;
    walltime += timeIt;

    res = (MEGA * (double)samplesPropagate) / timeIt;
    printf("\nIteracao: %d ; Bsize_x: %d ; Bsize_y: %d ; tempoExec: %lf ; MSamples: %lf \n", it, bsize_x, bsize_y, timeIt, res);

    // find_optimal_block_size(sx, timeIt, &bsize_x, &bsize_y);
    // printf("tempo deu: %lf\n", timeIt);

    tSim = it * dt;
    if (tSim >= tOut)
    {
      // DRIVER_Update_pointers(sx,sy,sz,pc);
      // DumpSliceFile(sx,sy,sz,pc,sPtr);
      //  CUDA_prefetch_pc(sx,sy,sz,pc);

      tOut = (++nOut) * dtOutput;
#ifdef _DUMP
      // DRIVER_Update_pointers(sx,sy,sz,pc);
      // DumpSliceSummary(sx,sy,sz,sPtr,dt,it,pc,src);
#endif
    }
  }

  const char StringHWM[6] = "VmHWM";
  char line[256], title[12], HWMUnit[8];
  const long HWM;
  const double MSamples = (MEGA * (double)totalSamples) / walltime;

  /*FILE *fp=fopen("/proc/self/status","r");
  while (fgets(line, 256, fp) != NULL){
    if (strncmp(line, StringHWM, 5) == 0) {
      sscanf(line+6,"%ld %s", &HWM, HWMUnit);
      break;
    }
  }
  fclose(fp);
  */
  // nao vamos salvar em disco

  // Dump Execution Metrics

  printf("Execution time (s) is %lf\n", walltime);
  printf("MSamples/s %.0lf\n", MSamples);
  printf("Memory High Water Mark is %ld %s\n", HWM, HWMUnit);

  // Dump Execution Metrics in CSV

  FILE *fr = NULL;
  const char fName[] = "Report.csv";
  fr = fopen(fName, "w");

  // report problem size

  ReportProblemSizeCSV(sx, sy, sz, bord, st, fr);

  // report collected metrics

  ReportMetricsCSV(walltime, MSamples, HWM, HWMUnit, fr);

  fclose(fr);

  fflush(stdout);

  // DRIVER_Finalize deallocate data, clean-up things etc
  DRIVER_Finalize(sx, sy, sz, bord, dx, dy, dz, dt, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz,
                  v2px, v2pz, v2sz, v2pn, vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc);
}

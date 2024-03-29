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



typedef struct block_size
{
  int bsize_x;
  int bsize_y;
} block_size;

typedef struct optimal_block
{
  int bsize_x;
  int bsize_y;
  double min_time;
} optimal_block;

static optimal_block opt_block = {.bsize_x = 0, .bsize_y = 0, .min_time = DBL_MAX};
static int already_optimized = 0;
static int saved = 0;
static int block_index = 0;
static block_size sizes[] = {{2, 2}, {2, 4}, {2, 8}, {16, 16}, {16, 32}, {32, 2}, {32, 4}, {32, 8}, {32, 16}, {64, 2} ,{64, 4}, {64, 8}, {128, 2} ,{2, 16}, {2, 32}, {2, 64}, {2, 128}, {4, 2}, {4, 4}, {4, 8}, {4, 16}, {4, 32}, {4, 64}, {4, 128}, {2, 2}, {2, 4}, {2, 8}, {2, 16}, {2, 32}, {2, 64}, {2, 128}, {4, 2}, {4, 4}, {4, 8}, {4, 16}, {2, 2}, {2, 4}, {2, 8}, {2, 16}, {2, 32}, {2, 64}, {2, 128}, {4, 2}, {4, 4}, {4, 8}, {4, 16}, {4, 32}, {4, 64}, {4, 128}, {8, 2}, {8, 4}, {8, 8}, {8, 16} ,{32, 2}, {32, 4}, {32, 8}, {32, 16}, {64, 2} ,{64, 4}, {64, 8}, {128, 2} ,{2, 16}, {2, 32}, {2, 64}, {2, 128}, {4, 2}, {4, 4}, {4, 8}, {128, 4}, {4, 16}, {4, 32}, {4, 64}, {4, 128}, {2, 2}, {2, 4}, {2, 8}, {2, 16}, {2, 32}, {2, 64}, {2, 128}, {4, 2}, {4, 4}, {4, 8}, {4, 16}, {2, 2}, {2, 4}, {2, 8}, {2, 16}, {2, 32}, {2, 64}, {2, 128}, {4, 2}, {4, 4}, {4, 8}, {4, 16}, {4, 32}, {4, 64}, {4, 128}, {8, 2}, {8, 4}, {8, 8}, {8, 16}, {8, 32}, {8, 64}, {16, 2}, {16, 4}, {16, 8}, {16, 16}, {16, 32}, {32, 2}, {32, 4}, {32, 8},  {2, 64}, {2, 128}, {4, 2}, {4, 4}, {4, 8}, {4, 16}, {4, 32}, {4, 64}, {4, 128}, {2, 2}, {2, 4}, {2, 8}, {2, 16}, {2, 32}, {2, 64}, {2, 128}, {4, 2}, {4, 4}, {4, 8}, {4, 16}, {4, 32}, {4, 64}, {4, 128}, {8, 2}, {8, 4}, {8, 8}, {8, 16}, {8, 32}, {8, 64}, {16, 2}, {16, 4}, {16, 8}, {16, 16}, {16, 32}, {32, 2}, {32, 4}, {32, 8}, {32, 16}, {64, 2}, {8, 16}, {8, 32}, {8, 64}, {16, 2}, {16, 4}, {16, 8}, {16, 16}, {16, 32}, {32, 2}, {32, 4}, {32, 8}, {32, 16}, {64, 2} ,{64, 4}, {64, 8}, {128, 2}};

// Variável global para armazenar o tamanho de bloco ótimo encontrado
static int optimal_bsize_x = 0;
static int optimal_bsize_y = 0;
static int optimal_bsize_initialized = 0;
static int configMatch = 0;

char *rtrim(char *str)
{
  char *end = str + strlen(str) - 1;
  while (end > str && isspace((unsigned char)*end))
    end--;
  *(end + 1) = 0;
  return str;
}

void save_optimal_config(const char *gpu_name, int sx, optimal_block ob)
{
  FILE *file = fopen("configurations.txt", "a");
  if (file)
  {
    fprintf(file, "%s | %d | %d | %d\n", gpu_name, sx, ob.bsize_x, ob.bsize_y);
    fclose(file);
  }
  else
  {
    perror("Erro ao abrir o arquivo configurations.txt para escrita");
  }
}

int load_optimal_config(const char *gpu_name, int sx)
{
  FILE *file = fopen("configurations.txt", "r");
  char buffer[256];
  int found = 0;

  if (file)
  {
    printf("Arquivo configurations.txt aberto com sucesso.\n");

    while (fgets(buffer, sizeof(buffer), file))
    {
      char stored_device_name[128];
      int stored_sx, stored_bsize_x, stored_bsize_y;

      if (sscanf(buffer, "%127[^|] | %d | %d | %d", stored_device_name, &stored_sx, &stored_bsize_x, &stored_bsize_y) == 4)
      {
        rtrim(stored_device_name); // Adicione esta linha
        printf("Lido do arquivo: %s | %d | %d | %d\n", stored_device_name, stored_sx, stored_bsize_x, stored_bsize_y);

        if (strcmp(stored_device_name, gpu_name) == 0 && stored_sx == sx)
        {
          printf("Configuração encontrada para %s com sx=%d. Bsize_x=%d, Bsize_y=%d\n", gpu_name, sx, stored_bsize_x, stored_bsize_y);
          optimal_bsize_x = stored_bsize_x;
          optimal_bsize_y = stored_bsize_y;
          found = 1;
          break;
        }
      }
      else
      {
        printf("Formato inválido na linha: %s", buffer);
      }
    }

    fclose(file);
  }
  else
  {
    perror("Erro ao abrir o arquivo configurations.txt para leitura");
  }

  if (!found)
  {
    printf("Configuração não encontrada para: %s | %d\n", gpu_name, sx);
  }

  return found;
}

void find_optimal_block_size(int sx, double timeIt, int *bsize_x, int *bsize_y)
{
  const char *device_name = get_default_device_name();

  // Se a configuração ótima já estiver armazenada, basta usá-la diretamente.
  if (optimal_bsize_initialized)
  {
    *bsize_x = optimal_bsize_x;
    *bsize_y = optimal_bsize_y;
    return;
  }

  // Verifica se já temos a configuração ótima armazenada.
  if (!already_optimized && load_optimal_config(device_name, sx))
  {
    // Encontrou uma configuração ótima previamente armazenada. Usa-a e retorna.
    already_optimized = 1; // Marca que já otimizamos anteriormente
    block_index = 0;       // Reinicia o índice para a próxima chamada
    saved = 0;
    optimal_bsize_initialized = 1; // Marca que a configuração ótima foi inicializada
    *bsize_x = optimal_bsize_x;
    *bsize_y = optimal_bsize_y;
    return;
  }

  // Verifica o tempo atual em relação ao ótimo
  if (*bsize_x * *bsize_y < 1024 && timeIt < opt_block.min_time)
  {
    opt_block.min_time = timeIt;
    opt_block.bsize_x = *bsize_x;
    opt_block.bsize_y = *bsize_y;
    saved = 0;
  }

  // Move para o próximo tamanho de bloco
  if (block_index < sizeof(sizes) / sizeof(block_size))
  {
    *bsize_x = sizes[block_index].bsize_x;
    *bsize_y = sizes[block_index].bsize_y;
    block_index++;
  }
  else
  {
    if (!saved)
    {
      save_optimal_config(device_name, sx, opt_block);
      saved = 1;
    }

    // Usa os tamanhos ótimos encontrados.
    *bsize_x = opt_block.bsize_x;
    *bsize_y = opt_block.bsize_y;

    // Reinicia para a próxima chamada
    block_index = 0;
    already_optimized = 1;               // Marca que encontramos o tamanho ótimo
    optimal_bsize_initialized = 1;       // Marca que a configuração ótima foi inicializada
    optimal_bsize_x = opt_block.bsize_x; // Armazena a configuração ótima em variáveis globais
    optimal_bsize_y = opt_block.bsize_y;
  }
}

int find_optimal_config_for_gpu(int sx, int *bsize_x, int *bsize_y)
{

  const char *device_name = get_default_device_name();
  // Verifica se a configuração ideal já foi encontrada anteriormente
  if (optimal_bsize_initialized)
  {
    *bsize_x = optimal_bsize_x;
    *bsize_y = optimal_bsize_y;
    configMatch = 1; // Indica que a configuração ideal foi encontrada
    return 1;        // Configuração ótima encontrada
  }

  // Verifica se há uma configuração ótima para a GPU e sx específicos.
  if (!already_optimized && load_optimal_config(device_name, sx))
  {
    // Configuração ótima encontrada, configMatch foi definida como 1 na função load_optimal_config.
    // Vamos diretamente para o loop de otimização.
    *bsize_x = optimal_bsize_x;
    *bsize_y = optimal_bsize_y;
    configMatch = 1; // Indica que a configuração ideal foi encontrada
    return 1;
  }

  return 0; // Configuração ótima não encontrada
}

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

  // CUDA_Initialize initialize target, allocate data etc
  DRIVER_Initialize(sx, sy, sz, bord, dx, dy, dz, dt, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz,
                    v2px, v2pz, v2sz, v2pn, vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc); // ok Arthur

  double walltime = 0.0;
  double timeIt = 0.0;
  double res = 0.0;
  int bsize_x = 32, bsize_y = 16;

  int optBsize = find_optimal_config_for_gpu(sx, &bsize_x, &bsize_y);

  if (optBsize == 1)
  {

    for (int it = 1; it <= st; it++)
    {
      // Calculate / obtain source value on i timestep
      float src = Source(dt, it - 1);
      DRIVER_InsertSource(src, iSource, pc, qc, pp, qp);

      printf("\nBsize_x: %d \n", bsize_x);
      printf("Bsize_y: %d \n", bsize_y);
      const double t0 = wtime();

      DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, bsize_x, bsize_y); // ajustar parametros
      // SwapArrays(&pp, &pc, &qp, &qc);

      timeIt = wtime() - t0;
      walltime += timeIt;

      res = (MEGA*(double)samplesPropagate)/timeIt;
      printf("\nIteracao: %d ; Bsize_x: %d ; Bsize_y: %d ; tempoExec: %lf ; MSamples: %lf \n", it, bsize_x, bsize_y, timeIt, res);

      // find_optimal_block_size(sx, timeIt, &bsize_x, &bsize_y);
      //printf("tempo deu: %lf\n", timeIt);

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
  }
  else
  {

    for (int it = 1; it <= st; it++)
    {
      // Calculate / obtain source value on i timestep
      float src = Source(dt, it - 1);
      DRIVER_InsertSource(src, iSource, pc, qc, pp, qp);

      printf("\nBsize_x: %d \n", bsize_x);
      printf("Bsize_y: %d \n", bsize_y);
      const double t0 = wtime();

      DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, bsize_x, bsize_y); // ajustar parametros
      // SwapArrays(&pp, &pc, &qp, &qc);

      timeIt = wtime() - t0;
      walltime += timeIt;

      res = (MEGA*(double)samplesPropagate)/timeIt;
      printf("\nIteracao: %d ; Bsize_x: %d ; Bsize_y: %d ; tempoExec: %lf ; MSamples: %lf \n", it, bsize_x, bsize_y, timeIt, res);

      find_optimal_block_size(sx, timeIt, &bsize_x, &bsize_y);

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

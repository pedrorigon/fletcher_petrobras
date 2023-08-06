#include "utils.h"
#include "source.h"
#include "driver.h"
#include "fletcher.h"
#include "walltime.h"
#include "model.h"
#include "CUDA/cuda_stuff.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define POPULATION_SIZE 50
// Valores possíveis para bsize_x e bsize_y
int bsize_values[] = {2, 4, 8, 16, 32, 64, 128};
int possible_values[] = {2, 4, 8, 16, 32, 64, 128};
// Indivíduo na população
typedef struct {
    int bsize_x;
    int bsize_y;
    double timeIt;
} Individual;

Individual population[POPULATION_SIZE];
Individual best_individual;

//#define MODEL_GLOBALVARS
//ARTHUR: Transformar em variável local.

//#undef MODEL_GLOBALVARS

// Inicializa a população com indivíduos aleatórios
void initialize_population() {
    srand(time(NULL));
    for (int i = 0; i < POPULATION_SIZE; i++) {
        do {
            population[i].bsize_x = bsize_values[rand() % (sizeof(bsize_values) / sizeof(int))];
            population[i].bsize_y = bsize_values[rand() % (sizeof(bsize_values) / sizeof(int))];
        } while (population[i].bsize_x * population[i].bsize_y > 1024);

        population[i].timeIt = __DBL_MAX__;
    }
    best_individual = population[0];
}

// Realiza mutação em um indivíduo
void mutate(Individual *individual) {
    do {
        individual->bsize_x = bsize_values[rand() % (sizeof(bsize_values) / sizeof(int))];
        individual->bsize_y = bsize_values[rand() % (sizeof(bsize_values) / sizeof(int))];
    } while (individual->bsize_x * individual->bsize_y > 1024);
}

// Seleciona um indivíduo da população para cruzamento
Individual tournament_selection() {
    Individual selected;
    int selected_index = rand() % POPULATION_SIZE;
    selected = population[selected_index];
    for (int i = 0; i < 5; i++) { // Compete com 5 outros indivíduos
        int competitor_index = rand() % POPULATION_SIZE;
        if (population[competitor_index].timeIt < selected.timeIt) {
            selected = population[competitor_index];
            selected_index = competitor_index;
        }
    }
    return selected;
}

// Realiza cruzamento entre dois indivíduos
// Realiza cruzamento entre dois indivíduos
Individual crossover(Individual parent1, Individual parent2) {
    Individual offspring;
    do {
        offspring.bsize_x = (rand() < 0.5) ? parent1.bsize_x : parent2.bsize_x;
        offspring.bsize_y = (rand() < 0.5) ? parent1.bsize_y : parent2.bsize_y;
    } while (offspring.bsize_x * offspring.bsize_y >= 1024);
    offspring.timeIt = __DBL_MAX__;
    return offspring;
}


void update_bsize_values(int *bsize_x, int *bsize_y, double timeIt) {
    // Se a população não foi inicializada, inicialize
    if (best_individual.timeIt == __DBL_MAX__) {
        initialize_population();
    }

    // Atualize o tempo do indivíduo correspondente à geração atual
    int current_generation = (int) (*bsize_x == best_individual.bsize_x && *bsize_y == best_individual.bsize_y);
    population[current_generation].timeIt = timeIt;

    // Verifique se o indivíduo atual é o melhor até agora
    if (timeIt < best_individual.timeIt) {
        best_individual.bsize_x = *bsize_x;
        best_individual.bsize_y = *bsize_y;
        best_individual.timeIt = timeIt;
    }

    // Realize operações de cruzamento e mutação para gerar a próxima geração
    for (int i = 0; i < POPULATION_SIZE; i++) {
        Individual parent1 = tournament_selection();
        Individual parent2 = tournament_selection();
        population[i] = crossover(parent1, parent2);
        if ((double) rand() / RAND_MAX < 0.1) { // 10% de chance de mutação
            mutate(&population[i]);
        }
    }

    // A próxima chamada para essa função usará o primeiro indivíduo da nova geração
    *bsize_x = population[0].bsize_x;
    *bsize_y = population[0].bsize_y;
}

void ReportProblemSizeCSV(const int sx, const int sy, const int sz, const int bord, const int st, FILE *f){
  fprintf(f,"sx; %d; sy; %d; sz; %d; bord; %d;  st; %d; \n",sx, sy, sz, bord, st);
}

void ReportMetricsCSV(double walltime, double MSamples,long HWM, char *HWMUnit, FILE *f){
  fprintf(f,"walltime; %lf; MSamples; %lf; HWM;  %ld; HWMUnit;  %s;\n",walltime, MSamples, HWM, HWMUnit);
}


void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr, const int sx, const int sy, const int sz, const int bord,
           const float dx, const float dy, const float dz, const float dt, const int it, float * restrict pp, float * restrict pc, float * restrict qp, float * restrict qc,
	         float * restrict vpz, float * restrict vsv, float * restrict epsilon, float * restrict delta, float * restrict phi, float * restrict theta)
{

  int nOut=1;
  int size = sx*sy*sz;
  float tSim=0.0, tOut=nOut*dtOutput;

  const long samplesPropagate=(long)(sx-2*(bord))*(long)(sy-2*(bord))*(long)(sz-2*(bord));
  const long totalSamples=samplesPropagate*(long)st;

  float *ch1dxx=NULL;  // isotropy simetry deep angle
  float *ch1dyy=NULL;  // isotropy simetry deep angle
  float *ch1dzz=NULL;  // isotropy simetry deep angle
  float *ch1dxy=NULL;  // isotropy simetry deep angle
  float *ch1dyz=NULL;  // isotropy simetry deep angle
  float *ch1dxz=NULL;  // isotropy simetry deep angle
  float *v2px=NULL;  // coeficient of H2(p)
  float *v2pz=NULL;  // coeficient of H1(q)
  float *v2sz=NULL;  // coeficient of H1(p-q) and H2(p-q)
  float *v2pn=NULL;  // coeficient of H2(p)
  
  DRIVER_Allocate_Model_Variables(&ch1dxx, &ch1dyy, &ch1dzz, &ch1dxy, &ch1dyz, &ch1dxz, &v2px, &v2pz, &v2sz, &v2pn, sx, sy, sz);

  for (int i=0; i<size; i++) {
    float sinTheta=sin(theta[i]);
    float cosTheta=cos(theta[i]);
    float sin2Theta=sin(2.0*theta[i]);
    float sinPhi=sin(phi[i]);
    float cosPhi=cos(phi[i]);
    float sin2Phi=sin(2.0*phi[i]);
    ch1dxx[i]=sinTheta*sinTheta * cosPhi*cosPhi;
    ch1dyy[i]=sinTheta*sinTheta * sinPhi*sinPhi;
    ch1dzz[i]=cosTheta*cosTheta;
    ch1dxy[i]=sinTheta*sinTheta * sin2Phi;
    ch1dyz[i]=sin2Theta         * sinPhi;
    ch1dxz[i]=sin2Theta         * cosPhi;
  }
  #ifdef _DUMP
  {
    const int iPrint=ind(bord+1,bord+1,bord+1);
    printf("ch1dxx=%f; ch1dyy=%f; ch1dzz=%f; ch1dxy=%f; ch1dxz=%f; ch1dyz=%f\n",ch1dxx[iPrint], ch1dyy[iPrint], ch1dzz[iPrint], ch1dxy[iPrint], ch1dxz[iPrint], ch1dyz[iPrint]);
  }
  #endif

  // coeficients of H1 and H2 at PDEs
  for (int i=0; i<size; i++){
    v2sz[i]=vsv[i]*vsv[i];
    v2pz[i]=vpz[i]*vpz[i];
    v2px[i]=v2pz[i]*(1.0+2.0*epsilon[i]);
    v2pn[i]=v2pz[i]*(1.0+2.0*delta[i]);
  }
  
#ifdef _DUMP
{
  const int iPrint=ind(bord+1,bord+1,bord+1);
  printf("vsv=%e; vpz=%e, v2pz=%e\n", vsv[iPrint], vpz[iPrint], v2pz[iPrint]);
  printf("v2sz=%e; v2pz=%e, v2px=%e, v2pn=%e\n", v2sz[iPrint], v2pz[iPrint], v2px[iPrint], v2pn[iPrint]);
}
#endif

  // CUDA_Initialize initialize target, allocate data etc
DRIVER_Initialize(sx, sy, sz, bord, dx, dy, dz, dt, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, 
              v2px, v2pz, v2sz, v2pn, vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc); //ok Arthur

double walltime=0.0;
double timeIt=0.0;
int bsize_x=16, bsize_y=16;



for (int it=1; it<=st; it++) {
    // Calculate / obtain source value on i timestep
    float src = Source(dt, it-1);
    DRIVER_InsertSource(src, iSource, pc, qc, pp, qp);


    printf("\nBsize_x: %d \n", bsize_x);
    printf("Bsize_y: %d \n", bsize_y);

    const double t0=wtime();
    
    DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, bsize_x, bsize_y); //ajustar parametros
    //SwapArrays(&pp, &pc, &qp, &qc);

    timeIt=wtime()-t0;
    walltime+=timeIt;

    printf("tempo deu: %lf\n", timeIt);
    update_bsize_values(&bsize_x, &bsize_y, timeIt);

    tSim=it*dt;
    if (tSim >= tOut) {
      //DRIVER_Update_pointers(sx,sy,sz,pc);
      //DumpSliceFile(sx,sy,sz,pc,sPtr);
    //  CUDA_prefetch_pc(sx,sy,sz,pc);

      tOut=(++nOut)*dtOutput;
#ifdef _DUMP
      //DRIVER_Update_pointers(sx,sy,sz,pc);
      //DumpSliceSummary(sx,sy,sz,sPtr,dt,it,pc,src);
#endif
    }
  }




  const char StringHWM[6]="VmHWM";
  char line[256], title[12],HWMUnit[8];
  const long HWM;
  const double MSamples=(MEGA*(double)totalSamples)/walltime;
  
  /*FILE *fp=fopen("/proc/self/status","r");
  while (fgets(line, 256, fp) != NULL){
    if (strncmp(line, StringHWM, 5) == 0) {
      sscanf(line+6,"%ld %s", &HWM, HWMUnit);
      break;
    }
  }
  fclose(fp);
  */  //nao vamos salvar em disco

  // Dump Execution Metrics
  
  printf ("Execution time (s) is %lf\n", walltime);
  printf ("MSamples/s %.0lf\n", MSamples);
  printf ("Memory High Water Mark is %ld %s\n",HWM, HWMUnit);

  // Dump Execution Metrics in CSV
  
  FILE *fr=NULL;
  const char fName[]="Report.csv";
  fr=fopen(fName,"w");

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


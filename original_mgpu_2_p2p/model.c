#include "utils.h"
#include "source.h"
#include "driver.h"
#include "fletcher.h"
#include "walltime.h"
#include "model.h"
#include "CUDA/cuda_stuff.h"
#include <time.h>
#define NUM_ACTIONS 6
#define EPSILON 0.1
#define ALPHA 0.5
#define GAMMA 0.9
#define EPSILON_START 1.0
#define EPSILON_MIN 0.1
#define EPSILON_DECAY 0.05

int actions[NUM_ACTIONS] = {1, 2, 4, 8, 16, 32};
float q_table[NUM_ACTIONS][NUM_ACTIONS]= {0};

//#define MODEL_GLOBALVARS
//ARTHUR: Transformar em variável local.

//#undef MODEL_GLOBALVARS


void ReportProblemSizeCSV(const int sx, const int sy, const int sz, const int bord, const int st, FILE *f){
  fprintf(f,"sx; %d; sy; %d; sz; %d; bord; %d;  st; %d; \n",sx, sy, sz, bord, st);
}

void ReportMetricsCSV(double walltime, double MSamples,long HWM, char *HWMUnit, FILE *f){
  fprintf(f,"walltime; %lf; MSamples; %lf; HWM;  %ld; HWMUnit;  %s;\n",walltime, MSamples, HWM, HWMUnit);
}

// Função para inicializar a tabela Q
void initialize_q_table() {
    for (int i = 0; i < NUM_ACTIONS; i++) {
        for (int j = 0; j < NUM_ACTIONS; j++) {
            q_table[i][j] = 0.0;
        }
    }
}

// Função para escolher a próxima ação
int choose_action(int state, float epsilon) {
    // Escolhe uma ação aleatória com probabilidade epsilon
    if (((float) rand() / RAND_MAX) < epsilon) {
        return rand() % NUM_ACTIONS;
    }
    // Caso contrário, escolhe a ação com maior valor Q
    else {
        float max_val = -1e9;
        int max_action = -1;
        for (int action = 0; action < NUM_ACTIONS; action++) {
            if (q_table[state][action] > max_val) {
                max_val = q_table[state][action];
                max_action = action;
            }
        }
        return max_action;
    }
}


#define EPSILON_START 1.0
#define EPSILON_MIN 0.1
#define EPSILON_DECAY 0.01

void optimize_block_sizes(int iteration, double *timeIt, int *bsize_x, int *bsize_y) {
    static int old_Bsize_X = -1;
    static int old_Bsize_Y = -1;
    static double old_walltime = 0.0;
    static double epsilon = EPSILON_START;

    // Gera valores iniciais aleatórios para Bsize_X e Bsize_Y na primeira chamada da função
    if (old_Bsize_X == -1 || old_Bsize_Y == -1) {
        old_Bsize_X = actions[rand() % NUM_ACTIONS];
        old_Bsize_Y = actions[rand() % NUM_ACTIONS];
    }

    // Se não for a primeira iteração, atualiza a tabela Q com base no tempo de execução anterior
    if (iteration != 1) {
        int reward = old_walltime - *timeIt;
        int old_state = old_Bsize_X;
        int old_action = old_Bsize_Y;
        int new_state = choose_action(old_Bsize_X, epsilon);
        int new_action = choose_action(old_Bsize_Y, epsilon);

        float old_q_value = q_table[old_state][old_action];
        float max_new_q_value = -1e9;
        for (int action = 0; action < NUM_ACTIONS; action++) {
            if (q_table[new_state][action] > max_new_q_value) {
                max_new_q_value = q_table[new_state][action];
            }
        }

        q_table[old_state][old_action] = old_q_value + ALPHA * (reward + GAMMA * max_new_q_value - old_q_value);

        old_Bsize_X = new_state;
        old_Bsize_Y = new_action;
    }

    // Atualiza o tempo de execução antigo
    old_walltime = *timeIt;

    // Atualiza bsize_x e bsize_y
    *bsize_x = old_Bsize_X;
    *bsize_y = old_Bsize_Y;

    // Decaimento de epsilon
    if (epsilon > EPSILON_MIN) {
        epsilon -= EPSILON_DECAY;
    }
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
int bsize_x, bsize_y;

// Inicialize a tabela Q e o gerador de números aleatórios
initialize_q_table();
srand(time(NULL));

for (int it=1; it<=st; it++) {
    // Calculate / obtain source value on i timestep
    float src = Source(dt, it-1);
    DRIVER_InsertSource(src, iSource, pc, qc, pp, qp);

    optimize_block_sizes(it, &timeIt, &bsize_x, &bsize_y);

    printf("valor de Bsize_x usado inicialmente: %d \n", bsize_x);
    printf("valor de Bsize_y usado inicialmente: %d \n", bsize_y);

    const double t0=wtime();
    
    DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, bsize_x, bsize_y); //ajustar parametros
    //SwapArrays(&pp, &pc, &qp, &qc);

    timeIt=wtime()-t0;
    walltime+=timeIt;

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


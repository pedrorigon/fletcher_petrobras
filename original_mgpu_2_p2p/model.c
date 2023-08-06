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
#define EPSILON_MIN 0
#define EPSILON_DECAY 0.01

#define NUM_ACTIONS_X 5  // bsize_x pode ser {2, 4, 8, 16, 32}, então temos 5 ações possíveis
#define NUM_ACTIONS_Y 4  // bsize_y pode ser {2, 4, 8, 16}, então temos 4 ações possíveis


int actions_X[NUM_ACTIONS_X] = {2, 4, 8, 16, 32};
int actions_Y[NUM_ACTIONS_Y] = {2, 4, 8, 16};

float q_table_X[NUM_ACTIONS_X][NUM_ACTIONS_X]= {0};
float q_table_Y[NUM_ACTIONS_Y][NUM_ACTIONS_Y]= {0};

//#define MODEL_GLOBALVARS
//ARTHUR: Transformar em variável local.

//#undef MODEL_GLOBALVARS


void ReportProblemSizeCSV(const int sx, const int sy, const int sz, const int bord, const int st, FILE *f){
  fprintf(f,"sx; %d; sy; %d; sz; %d; bord; %d;  st; %d; \n",sx, sy, sz, bord, st);
}

void ReportMetricsCSV(double walltime, double MSamples,long HWM, char *HWMUnit, FILE *f){
  fprintf(f,"walltime; %lf; MSamples; %lf; HWM;  %ld; HWMUnit;  %s;\n",walltime, MSamples, HWM, HWMUnit);
}

// Função para inicializar as tabelas Q
void initialize_q_table() {
    // Inicializar a tabela Q para bsize_x
    for (int i = 0; i < NUM_ACTIONS_X; i++) {
        for (int j = 0; j < NUM_ACTIONS_X; j++) {
            q_table_X[i][j] = 0.0;
        }
    }
    // Inicializar a tabela Q para bsize_y
    for (int i = 0; i < NUM_ACTIONS_Y; i++) {
        for (int j = 0; j < NUM_ACTIONS_Y; j++) {
            q_table_Y[i][j] = 0.0;
        }
    }
}


// Função para escolher a próxima ação
// Função para escolher a próxima ação
int choose_action(int state, float epsilon, int actions[], int num_actions, float q_table[][num_actions]) {
    float random_number = (float)rand() / RAND_MAX;
    if (random_number < epsilon) {
        // Ação aleatória para exploração
        return actions[rand() % num_actions];
    } else {
        // Escolhe a ação com o maior valor Q para explotação
        int best_action_index = 0;
        float best_q_value = q_table[state][0];
        for (int i = 1; i < num_actions; i++) {
            if (q_table[state][i] > best_q_value) {
                best_q_value = q_table[state][i];
                best_action_index = i;
            }
        }
        return actions[best_action_index];
    }
}




void optimize_block_sizes(int iteration, double *timeIt, int *bsize_x, int *bsize_y) {
    static int old_Bsize_X = -1;
    static int old_Bsize_Y = -1;
    static double old_walltime = 0.0;
    static double epsilon = EPSILON_START;

    if (old_Bsize_X == -1 || old_Bsize_Y == -1) {
        old_Bsize_X = actions_X[rand() % NUM_ACTIONS_X];
        old_Bsize_Y = actions_Y[rand() % NUM_ACTIONS_Y];
    }

    if (iteration != 1) {
        int reward = *timeIt - old_walltime;  // Alteração feita aqui.

        // Limitar a recompensa a um intervalo [-1, 1]
        if (reward > 1) {
            reward = 1;
        } else if (reward < -1) {
            reward = -1;
        }

        int old_state_index_X = -1;
        int old_action_index_X = -1;
        for (int i = 0; i < NUM_ACTIONS_X; i++) {
            if (actions_X[i] == old_Bsize_X) old_state_index_X = i;
            if (actions_X[i] == old_Bsize_Y) old_action_index_X = i;
        }

        int new_state_X = choose_action(old_state_index_X, epsilon, actions_X, NUM_ACTIONS_X, q_table_X);
        int new_state_index_X = -1;
        for (int i = 0; i < NUM_ACTIONS_X; i++) {
            if (actions_X[i] == new_state_X) new_state_index_X = i;
        }

        float old_q_value_X = q_table_X[old_state_index_X][old_action_index_X];
        float max_new_q_value_X = -1e9;
        for (int action = 0; action < NUM_ACTIONS_X; action++) {
            if (q_table_X[new_state_index_X][action] > max_new_q_value_X) {
                max_new_q_value_X = q_table_X[new_state_index_X][action];
            }
        }

        q_table_X[old_state_index_X][old_action_index_X] = old_q_value_X + ALPHA * (reward + GAMMA * max_new_q_value_X - old_q_value_X);

        old_Bsize_X = new_state_X;

        // Repita o processo acima para Y
        int old_state_index_Y = -1;
        int old_action_index_Y = -1;
        for (int i = 0; i < NUM_ACTIONS_Y; i++) {
            if (actions_Y[i] == old_Bsize_X) old_state_index_Y = i;
            if (actions_Y[i] == old_Bsize_Y) old_action_index_Y = i;
        }

        int new_state_Y = choose_action(old_state_index_Y, epsilon, actions_Y, NUM_ACTIONS_Y, q_table_Y);
        int new_state_index_Y = -1;
        for (int i = 0; i < NUM_ACTIONS_Y; i++) {
            if (actions_Y[i] == new_state_Y) new_state_index_Y = i;
        }

        float old_q_value_Y = q_table_Y[old_state_index_Y][old_action_index_Y];
        float max_new_q_value_Y = -1e9;
        for (int action = 0; action < NUM_ACTIONS_Y; action++) {
            if (q_table_Y[new_state_index_Y][action] > max_new_q_value_Y) {
                max_new_q_value_Y = q_table_Y[new_state_index_Y][action];
            }
        }

        q_table_Y[old_state_index_Y][old_action_index_Y] = old_q_value_Y + ALPHA * (reward + GAMMA * max_new_q_value_Y - old_q_value_Y);

        old_Bsize_Y = new_state_Y;
    }

    old_walltime = *timeIt;
    *bsize_x = old_Bsize_X;
    *bsize_y = old_Bsize_Y;

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

    printf("tempo deu: %lf\n", timeIt);

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


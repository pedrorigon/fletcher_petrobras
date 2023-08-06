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

#define NUM_THREADS_X 7
#define NUM_THREADS_Y 7
#define NUM_ACTIONS (NUM_THREADS_X * NUM_THREADS_Y)

// Definição dos valores possíveis de threads por bloco em X e Y.
int valores_threads_X[NUM_THREADS_X] = {2, 4, 8, 16, 32, 64, 128};
int valores_threads_Y[NUM_THREADS_Y] = {2, 4, 8, 16, 32, 64, 128};


//#define MODEL_GLOBALVARS
//ARTHUR: Transformar em variável local.

//#undef MODEL_GLOBALVARS


void ReportProblemSizeCSV(const int sx, const int sy, const int sz, const int bord, const int st, FILE *f){
  fprintf(f,"sx; %d; sy; %d; sz; %d; bord; %d;  st; %d; \n",sx, sy, sz, bord, st);
}

void ReportMetricsCSV(double walltime, double MSamples,long HWM, char *HWMUnit, FILE *f){
  fprintf(f,"walltime; %lf; MSamples; %lf; HWM;  %ld; HWMUnit;  %s;\n",walltime, MSamples, HWM, HWMUnit);
}

// Função para verificar se a combinação de threads por bloco respeita o limite de 1024.
int verifica_limite(int threads_X, int threads_Y) {
    return (threads_X * threads_Y < 1024);
}

// Função para escolher uma ação com base na política ε-greedy.
int escolher_acao(double** Q, int estado, double epsilon) {
    if (rand() / (double)RAND_MAX < epsilon) {
        // Escolher uma ação aleatória (exploração).
        int acao;
        do {
            acao = rand() % NUM_ACTIONS;
        } while (!verifica_limite(valores_threads_X[acao / NUM_THREADS_Y], valores_threads_Y[acao % NUM_THREADS_Y]));
        return acao;
    } else {
        // Escolher a melhor ação com base na tabela Q (exploração).
        int melhor_acao = 0;
        for (int a = 1; a < NUM_ACTIONS; a++) {
            if (Q[estado][a] > Q[estado][melhor_acao] && verifica_limite(valores_threads_X[a / NUM_THREADS_Y], valores_threads_Y[a % NUM_THREADS_Y])) {
                melhor_acao = a;
            }
        }
        return melhor_acao;
    }
}


// Função para atualizar a tabela Q com base no algoritmo de Q-learning.
void atualizar_Q(double** Q, int estado, int acao, double recompensa, int proximo_estado, double taxa_aprendizado, double fator_desconto) {
    double valor_maximo_proximo_estado = Q[proximo_estado][0];
    for (int a = 1; a < NUM_ACTIONS; a++) {
        if (Q[proximo_estado][a] > valor_maximo_proximo_estado) {
            valor_maximo_proximo_estado = Q[proximo_estado][a];
        }
    }
    double novo_valor_Q = Q[estado][acao] + taxa_aprendizado * (recompensa + fator_desconto * valor_maximo_proximo_estado - Q[estado][acao]);
    Q[estado][acao] = novo_valor_Q;
}

// Função para inicializar a tabela Q
double** inicializar_tabela_Q(int num_estados) {
    double** Q = (double*)malloc(num_estados * sizeof(double));
    for (int i = 0; i < num_estados; i++) {
        Q[i] = (double*)malloc(NUM_ACTIONS * sizeof(double));
        for (int j = 0; j < NUM_ACTIONS; j++) {
            Q[i][j] = 0.0;
        }
    }
    return Q;
}

// Função para criar o espaço de estados
int** criar_espaco_estados(int num_estados) {
    int valores_threads_X[NUM_THREADS_X] = {2, 4, 8, 16, 32, 64, 128};
    int valores_threads_Y[NUM_THREADS_Y] = {2, 4, 8, 16, 32, 64, 128};
    int** estados = (int*)malloc(num_estados * sizeof(int));
    int num_combinacoes_validas = 0;
    for (int i = 0; i < NUM_THREADS_X; i++) {
        for (int j = 0; j < NUM_THREADS_Y; j++) {
            int threads_X = valores_threads_X[i];
            int threads_Y = valores_threads_Y[j];
            if (verifica_limite(threads_X, threads_Y)) {
                estados[num_combinacoes_validas] = (int*)malloc(2 * sizeof(int));
                estados[num_combinacoes_validas][0] = threads_X;
                estados[num_combinacoes_validas][1] = threads_Y;
                num_combinacoes_validas++;
            }
        }
    }
    return estados;
}

void executar_Q_learning(double** Q, int** estados, int num_combinacoes_validas, double epsilon, double taxa_aprendizado, double fator_desconto, double tempo_execucao) {
    int estado_atual = 0;
    for (int iteracao = 0; iteracao < num_combinacoes_validas; iteracao++) {
        // Definir a configuração inicial como 16x16 na primeira iteração.
        if (iteracao == 0) {
            estado_atual = num_combinacoes_validas - 1; // Última configuração (16x16).
        } else {
            // Escolher um estado atual (configuração atual) aleatoriamente.
            estado_atual = rand() % num_combinacoes_validas;
        }

        // Escolher uma ação (configuração) com base na política ε-greedy.
        int acao_escolhida = escolher_acao(Q, estado_atual, epsilon);

        // Obter os valores de threads por bloco em X e Y para a ação escolhida.
        int threads_por_bloco_X = estados[estado_atual][0];
        int threads_por_bloco_Y = estados[estado_atual][1];

        // Calcular o tempo de execução para a configuração escolhida.
        //double tempo_execucao = calcular_tempo_execucao(threads_por_bloco_X, threads_por_bloco_Y);

        // Calcular a recompensa com base no tempo de execução observado.
        double recompensa = 1.0 / tempo_execucao;

        // Atualizar a tabela Q com base no algoritmo de Q-learning.
        int proximo_estado = acao_escolhida; // Para este exemplo, o próximo estado é o mesmo que a ação escolhida.
        atualizar_Q(Q, estado_atual, acao_escolhida, recompensa, proximo_estado, taxa_aprendizado, fator_desconto);
    }
}

void liberar_memoria(double** Q, int** estados, int num_estados, int num_combinacoes_validas) {
    for (int i = 0; i < num_estados; i++) {
        free(Q[i]);
    }
    free(Q);

    for (int i = 0; i < num_combinacoes_validas; i++) {
        free(estados[i]);
    }
    free(estados);
}


// Função para criar e inicializar a tabela Q.
double** criar_tabela_Q(int num_estados, int num_acoes) {
    double** Q = (double*)malloc(num_estados * sizeof(double));
    for (int i = 0; i < num_estados; i++) {
        Q[i] = (double*)malloc(num_acoes * sizeof(double));
        for (int j = 0; j < num_acoes; j++) {
            Q[i][j] = 0.0; // Inicialização com zeros.
            // Alternativamente, você pode inicializar com valores arbitrários.
        }
    }
    return Q;
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
//initialize_q_table();
srand(time(NULL));

// Definição dos valores possíveis de threads por bloco em X e Y.
int num_estados = NUM_THREADS_X * NUM_THREADS_Y;

//CRIA estados
int** estados = criar_espaco_estados(num_estados);

// Contagem das combinações válidas
    int num_combinacoes_validas = 0;
    for (int i = 0; i < NUM_THREADS_X; i++) {
        for (int j = 0; j < NUM_THREADS_Y; j++) {
            int threads_X = valores_threads_X[i];
            int threads_Y = valores_threads_Y[j];
            if (verifica_limite(threads_X, threads_Y)) {
                num_combinacoes_validas++;
            }
        }
    }

// Inicialização da tabela Q com valores arbitrários (ou zeros).
double** Q = inicializar_tabela_Q(num_estados);



for (int it=1; it<=st; it++) {
    // Calculate / obtain source value on i timestep
    float src = Source(dt, it-1);
    DRIVER_InsertSource(src, iSource, pc, qc, pp, qp);

    //optimize_block_sizes(it, &timeIt, &bsize_x, &bsize_y);
    // Loop de iterações do Q-learning.
    double epsilon = 0.1; // Altere para o seu valor de epsilon.
    double taxa_aprendizado = 0.5; // Altere para o seu valor de taxa de aprendizado.
    double fator_desconto = 1 / timeIt; // Altere para o seu valor de fator de desconto.
    executar_Q_learning(Q, estados, num_combinacoes_validas, epsilon, taxa_aprendizado, fator_desconto, timeIt);

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


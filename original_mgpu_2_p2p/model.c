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
#include <stdbool.h>


// Definição das configurações de threads por bloco (x, y)
typedef struct
{
    int x;
    int y;
} ConfiguracaoThreads;

// Número de configurações de threads por bloco
#define NUM_CONFIGURACOES 7

// Configurações de threads por bloco (x, y)
ConfiguracaoThreads configuracoes[NUM_CONFIGURACOES] = {
    {4, 16},
    {16, 4},
    {4, 32},
    {32, 4},
    {16, 32},
    {32, 16},
    {16, 16}};

// Matriz Q para armazenar os valores de recompensa estimados para cada estado-ação
float matrizQ[NUM_CONFIGURACOES][NUM_CONFIGURACOES] = {0};

// Parâmetros do algoritmo Q-learning
#define TAXA_APRENDIZADO 0.1
#define FATOR_DESCONTO 0.9
#define NUM_EPISODIOS 7
#define EPSILON_INICIAL 0.8
#define EPSILON_MINIMO 0.05
#define DECAIMENTO_EPSILON (EPSILON_INICIAL - EPSILON_MINIMO) / NUM_EPISODIOS

// Função para escolher uma ação usando a política e-greedy
int escolher_acao(int estado, float epsilon)
{
    // Com probabilidade epsilon, escolha uma ação aleatória (exploração)
    if ((float)rand() / RAND_MAX < epsilon)
    {
        return rand() % NUM_CONFIGURACOES;
    }

    // Caso contrário, escolha a ação com maior valor Q (explotação)
    int melhor_acao = 0;
    for (int acao = 1; acao < NUM_CONFIGURACOES; ++acao)
    {
        if (matrizQ[estado][acao] > matrizQ[estado][melhor_acao])
        {
            melhor_acao = acao;
        }
    }
    return melhor_acao;
}

// Função principal do algoritmo Q-learning
void q_learning(int* num_threadsX, int* num_threadsY, double tempo_execucao)
{
    srand(time(NULL));

    for (int episodio = 0; episodio < NUM_EPISODIOS; ++episodio)
    {
        // Encontre o índice da configuração atual no vetor de configurações
        int estado_atual = -1;
        for (int i = 0; i < NUM_CONFIGURACOES; ++i)
        {
            if (configuracoes[i].x == *num_threadsX && configuracoes[i].y == *num_threadsY)
            {
                estado_atual = i;
                break;
            }
        }

        if (estado_atual == -1)
        {
            // A configuração atual não foi encontrada no vetor de configurações
            printf("Erro: Configuração inválida: (%d, %d)\n", *num_threadsX, *num_threadsY);
            return;
        }

        // Escolha a próxima ação usando a política e-greedy
        float epsilon = EPSILON_INICIAL - DECAIMENTO_EPSILON * episodio;
        if (epsilon < EPSILON_MINIMO)
        {
            epsilon = EPSILON_MINIMO;
        }
        int proxima_acao = escolher_acao(estado_atual, epsilon);
        ConfiguracaoThreads proxima_configuracao = configuracoes[proxima_acao];

        // Obtenha a recompensa com base na diferença de tempo de execução
        float recompensa = 1.0 / tempo_execucao;

        // Atualize a matriz Q com o algoritmo Q-learning
        float valor_max_q_proxima_acao = matrizQ[proxima_acao][0];
        for (int acao = 1; acao < NUM_CONFIGURACOES; ++acao)
        {
            if (matrizQ[proxima_acao][acao] > valor_max_q_proxima_acao)
            {
                valor_max_q_proxima_acao = matrizQ[proxima_acao][acao];
            }
        }

        matrizQ[estado_atual][proxima_acao] = matrizQ[estado_atual][proxima_acao] +
                                              TAXA_APRENDIZADO * (recompensa +  valor_max_q_proxima_acao -
                                                                  matrizQ[estado_atual][proxima_acao]);

        // Avance para o próximo estado
        *num_threadsX = proxima_configuracao.x;
        *num_threadsY = proxima_configuracao.y;
    }
}

// Função para encontrar a configuração ótima com base na matriz Q aprendida
ConfiguracaoThreads encontrar_configuracao_otima()
{
    int estado_atual = 0;
    int melhor_acao = 0;

    // Encontre a ação com maior valor Q para o estado inicial
    for (int acao = 0; acao < NUM_CONFIGURACOES; ++acao)
    {
        if (matrizQ[estado_atual][acao] > matrizQ[estado_atual][melhor_acao])
        {
            melhor_acao = acao;
        }
    }

    return configuracoes[melhor_acao];
}

void imprimir_matriz_Q()
{
    printf("Matriz Q:\n");
    for (int estado = 0; estado < NUM_CONFIGURACOES; ++estado)
    {
        for (int acao = 0; acao < NUM_CONFIGURACOES; ++acao)
        {
            printf("%.2f ", matrizQ[estado][acao]);
        }
        printf("\n");
    }
}

// Simulação da chamada do kernel para obter o tempo de execução
float simular_execucao_kernel(ConfiguracaoThreads configuracao)
{
    // Simule a chamada do kernel e retorne um valor aleatório como tempo de execução
    return rand() % 100 + 1; // Valor aleatório entre 1 e 100 para simulação.
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
srand(time(NULL));

// Loop para executar o kernel com cada configuração e coletar os tempos de execução

double tempos_execucao[NUM_CONFIGURACOES];
for (int it=1; it<=st; it++) {
    // Calculate / obtain source value on i timestep
    float src = Source(dt, it-1);
    DRIVER_InsertSource(src, iSource, pc, qc, pp, qp);
    
    if (it-1 >= NUM_CONFIGURACOES){
      ConfiguracaoThreads configuracao_otima = encontrar_configuracao_otima();
      int bsize_x = configuracao_otima.x;
      int bsize_y = configuracao_otima.y;
    }else{
      ConfiguracaoThreads configuracao_atual = configuracoes[it - 1];
      int bsize_x = configuracao_atual.x;
      int bsize_y = configuracao_atual.y;
    }

    printf("\nBsize_x: %d \n", bsize_x);
    printf("Bsize_y: %d \n", bsize_y);
    //update_bsize_values(&bsize_x, &bsize_y, timeIt);

    const double t0=wtime();
    
    DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, bsize_x, bsize_y); //ajustar parametros
    //SwapArrays(&pp, &pc, &qp, &qc);

    timeIt=wtime()-t0;
    walltime+=timeIt;

    q_learning(&bsize_x, &bsize_y, timeIt);

    tempos_execucao[it-1] = timeIt;


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

  imprimir_matriz_Q();
  ConfiguracaoThreads configuracao_otima = encontrar_configuracao_otima();

    printf("Configuração ótima de threads por bloco: (%d, %d)\n",
           configuracao_otima.x, configuracao_otima.y);




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


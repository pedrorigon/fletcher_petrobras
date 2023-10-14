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
#include <limits.h>

#define POPULATION_SIZE 20
#define MAX_NUM_THREADS 64
#define MAX_MULTIPLICATION 1024
#define TOURNAMENT_SIZE 2
#define MUTATION_Y_PROBABILITY 0.1
#define MUTATION_X_PROBABILITY 0.1 

typedef struct
{
    int thread_x;
    int thread_y;
    double fitness;
} Individuo;

int isPowerOfTwo(int num)
{
    return (num & (num - 1)) == 0;
}

void inicializarPopulacao(Individuo *populacao, int tamanho_populacao)
{
    srand(time(NULL));

    int indices_fixos[][2] = {{32, 4}, {32, 8}, {32, 16}, {64, 4}};
    int num_indices_fixos = sizeof(indices_fixos) / sizeof(indices_fixos[0]);

    int populacao_atual = 0;
    while (populacao_atual < num_indices_fixos)
    {
        int x = indices_fixos[populacao_atual][0];
        int y = indices_fixos[populacao_atual][1];
        if (x * y < MAX_MULTIPLICATION)
        {
            populacao[populacao_atual].thread_x = x;
            populacao[populacao_atual].thread_y = y;
            populacao[populacao_atual].fitness = 0.0;
            populacao_atual++;
        }
    }

    while (populacao_atual < tamanho_populacao)
    {
        int x = rand() % (MAX_NUM_THREADS - 1) + 2;
        int y = rand() % (MAX_NUM_THREADS - 1) + 2;
        if (x * y < MAX_MULTIPLICATION &&
            isPowerOfTwo(x) && isPowerOfTwo(y))
        {
            populacao[populacao_atual].thread_x = x;
            populacao[populacao_atual].thread_y = y;
            populacao[populacao_atual].fitness = 0.0;
            populacao_atual++;
        }
    }
}

int encontrarMelhorIndice(Individuo *populacao)
{
    int indiceMelhor = 0;
    for (int i = 1; i < POPULATION_SIZE; i++)
    {
        if (populacao[i].fitness > populacao[indiceMelhor].fitness)
        {
            indiceMelhor = i;
        }
    }
    return indiceMelhor;
}

void selecaoPorTorneio(Individuo *populacao, Individuo *pais)
{
    for (int i = 0; i < 2; i++)
    {
        Individuo vencedor = populacao[rand() % POPULATION_SIZE]; // Escolhe um indivíduo aleatório como o primeiro candidato vencedor do torneio

        for (int j = 1; j < TOURNAMENT_SIZE; j++)
        {
            Individuo concorrente = populacao[rand() % POPULATION_SIZE]; // Escolhe um indivíduo aleatório como um concorrente no torneio
            if (concorrente.fitness > vencedor.fitness)
            {
                vencedor = concorrente; // Atualiza o vencedor se o concorrente tiver um fitness melhor
            }
        }

        pais[i] = vencedor; // Adiciona o vencedor como um pai
    }
}

int gerarNovoValorAleatorio()
{
    int vetor[] = {2, 4, 8, 16, 32, 64};
    int tamanho_vetor = sizeof(vetor) / sizeof(vetor[0]);

    // Inicializa a semente para geração de números aleatórios
    srand(time(NULL));

    // Gera um número aleatório entre 0 e 5
    int indice_aleatorio = rand() % tamanho_vetor;
    int random_value = vetor[indice_aleatorio];

    return random_value;
}


void crossoverEMutacao(Individuo *pais, Individuo *filhos)
{ // Verificar se a multiplicação dos valores de thread_x * thread_y é inferior a 1024
    if ((pais[1].thread_y * pais[0].thread_x < 1024) && (pais[1].thread_x * pais[0].thread_y < 1024))
    {
        // Realizar crossover trocando os valores entre os pais
        filhos[0].thread_x = pais[1].thread_y;
        filhos[0].thread_y = pais[0].thread_x;

        filhos[1].thread_x = pais[1].thread_x;
        filhos[1].thread_y = pais[0].thread_y;
    }
    else
    {
        filhos[0].thread_x = pais[0].thread_x;
        filhos[0].thread_y = pais[0].thread_y;
        filhos[1].thread_x = pais[1].thread_x;
        filhos[1].thread_y = pais[1].thread_y;
    }

    // Realizar mutação com probabilidade MUTATION_PROBABILITY
    // for (int i = 0; i < 2; i++)
    //{
    if ((double)rand() / RAND_MAX < MUTATION_Y_PROBABILITY)
{
    int new_thread_y = gerarNovoValorAleatorio();

    // Garantir que o novo valor de thread_y é válido para mutação
    while (new_thread_y * filhos[0].thread_x >= MAX_MULTIPLICATION)
    {
        new_thread_y = gerarNovoValorAleatorio();
    }

    filhos[0].thread_y = new_thread_y;
}

if ((double)rand() / RAND_MAX < MUTATION_X_PROBABILITY)
{
    int new_thread_x = gerarNovoValorAleatorio();
   
    while (new_thread_x * filhos[0].thread_y >= MAX_MULTIPLICATION)
    {
        new_thread_x = gerarNovoValorAleatorio();
    }

    filhos[0].thread_x = new_thread_x;
}
    //}
}

Individuo *gerarNovaSubpopulacao(Individuo *populacao)
{
    Individuo *novaSubpopulacao = (Individuo *)malloc(POPULATION_SIZE * sizeof(Individuo));

    for (int i = 0; i < POPULATION_SIZE; i++)
    {
        // Seleção de pais por torneio
        Individuo pais[2];
        selecaoPorTorneio(populacao, pais);

        // Crossover e mutação para gerar filhos
        Individuo filhos[2];
        crossoverEMutacao(pais, filhos);

        if (i == 0)
        {
            // Copiar o melhor indivíduo inalterado para a nova subpopulação
            int indiceMelhor = encontrarMelhorIndice(populacao);
            novaSubpopulacao[i] = populacao[indiceMelhor];
        }
        else
        {
            novaSubpopulacao[i] = filhos[0]; // Por exemplo, você pode escolher o primeiro filho
        }
    }

    // Substituir os indivíduos antigos na população atual pelos novos
    for (int i = 0; i < POPULATION_SIZE; i++)
    {
        populacao[i] = novaSubpopulacao[i];
    }

    free(novaSubpopulacao); // Liberar memória da nova subpopulação

    return populacao; // Retornar a população atualizada
}
// preciso rodar o kernel com todas as configurações da população inicial
// a aptidão de cada indivíduo será 1/tempo de exec


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
double kernel_time=0.0;
double res=0.0;
//int bsize_x=32, bsize_y=16;

Individuo populacao[POPULATION_SIZE];

// Inicializar população
inicializarPopulacao(populacao, POPULATION_SIZE);

// Loop para executar o kernel com cada configuração e coletar os tempos de execução

for (int it=1; it<=st; it++) {
    // Calculate / obtain source value on i timestep
    float src = Source(dt, it-1);
    DRIVER_InsertSource(src, iSource, pc, qc, pp, qp);

   // printf("\nBsize_x: %d \n", bsize_x);
  //  printf("Bsize_y: %d \n", bsize_y);
    //update_bsize_values(&bsize_x, &bsize_y, timeIt);

    if (it <= POPULATION_SIZE)
        {
            printf("\nBsize_x: %d \n", populacao[it - 1].thread_x);
            printf("Bsize_y: %d \n", populacao[it - 1].thread_y);
            const double t0=wtime();
            DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, populacao[it - 1].thread_x, populacao[it - 1].thread_y);
            kernel_time=wtime()-t0;
            walltime+=kernel_time;
            res = (MEGA*(double)samplesPropagate)/kernel_time;
            populacao[it - 1].fitness = 1 / kernel_time;
            printf("\n Iteracao: %d ; Bsize_x: %d ; Bsize_y: %d ; tempoExec: %lf ; MSamples: %lf \n", it, populacao[it - 1].thread_x, populacao[it - 1].thread_y, kernel_time, res);
            // printf("Indivíduo %d: thread_x = %d, thread_y = %d, fitness = %f\n", i - 1, populacao[i - 1].thread_x, populacao[i - 1].thread_y, populacao[i - 1].fitness);
            //  printf("Indivíduo %d: thread_x = %d, thread_y = %d, fitness = %f\n", i, populacao[i - 1].thread_x, populacao[i - 1].thread_y, populacao[i - 1].fitness);
            //    inicio primeira seleção genética
            //     Gerar a nova subpopulação
            if (it == POPULATION_SIZE)
            {
                printf("ta na hora da Seleçao\n");
                Individuo *novaSubpopulacao = gerarNovaSubpopulacao(populacao);
                // Substituir a população atual pela nova subpopulação
                for (int j = 0; j < POPULATION_SIZE; j++)
                {
                    populacao[j] = novaSubpopulacao[j];
                    // printf("nova pop: %d thread x: %d thread y: %d\n", j, populacao[j].thread_x, populacao[j].thread_y);
                }
                //free(novaSubpopulacao);
            }
        }
        else if (it <= POPULATION_SIZE * 2)
        {
            printf("\nBsize_x: %d \n", populacao[it - POPULATION_SIZE - 1].thread_x);
            printf("Bsize_y: %d \n", populacao[it - POPULATION_SIZE - 1].thread_y);
            const double t1=wtime();
            DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, populacao[it - POPULATION_SIZE - 1].thread_x, populacao[it - POPULATION_SIZE - 1].thread_y);
            kernel_time=wtime()-t1;
            walltime+=kernel_time;
            populacao[it - POPULATION_SIZE - 1].fitness = 1 / kernel_time;
            res = (MEGA*(double)samplesPropagate)/kernel_time;

            printf("\n Iteracao: %d ; Bsize_x: %d ; Bsize_y: %d ; tempoExec: %lf ; MSamples: %lf \n", it, populacao[it - POPULATION_SIZE - 1].thread_x, populacao[it - POPULATION_SIZE - 1].thread_y, kernel_time, res);
            // printf("SEGUNDA GERAÇÃO \n valor indice: %d Indivíduo %d: thread_x = %d, thread_y = %d, fitness = %f\n", i, i - POPULATION_SIZE - 1, populacao[i - POPULATION_SIZE - 1].thread_x, populacao[i - POPULATION_SIZE - 1].thread_y, populacao[i - POPULATION_SIZE - 1].fitness);
            //  printf("Indivíduo %d: thread_x = %d, thread_y = %d, fitness = %f\n", i - 1, populacao[(i - 1) / 2].thread_x, populacao[(i - 1) / 2].thread_y, populacao[(i - 1) / 2].fitness);
            //  Gerar a nova subpopulação
            if (it == POPULATION_SIZE * 2)
            {
                printf("ta na hora da SEGUNDA Seleçao\n");
                Individuo *novaSubpopulacao = gerarNovaSubpopulacao(populacao);
                // Substituir a população atual pela nova subpopulação
                for (int j = 0; j < POPULATION_SIZE; j++)
                {
                    populacao[j] = novaSubpopulacao[j];
                    // printf("nova pop: %d thread x: %d thread y: %d\n", j, populacao[j].thread_x, populacao[j].thread_y);
                }
                //free(novaSubpopulacao);
            }
        }
        else if (it <= POPULATION_SIZE * 3)
        {
            printf("\nBsize_x: %d \n", populacao[it - 1 - 2 * POPULATION_SIZE].thread_x);
            printf("Bsize_y: %d \n", populacao[it - 1 - 2 * POPULATION_SIZE].thread_y);
            const double t2=wtime();
            DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, populacao[it - 1 - 2 * POPULATION_SIZE].thread_x, populacao[it - 1 - 2 * POPULATION_SIZE].thread_y); 
            kernel_time=wtime()-t2;
            walltime+=kernel_time;
            populacao[it - 1 - 2 * POPULATION_SIZE].fitness = 1 / kernel_time;
            res = (MEGA*(double)samplesPropagate)/kernel_time;
            printf("\n Iteracao: %d ; Bsize_x: %d ; Bsize_y: %d ; tempoExec: %lf ; MSamples: %lf \n", it, populacao[it - 1 - 2 * POPULATION_SIZE].thread_x, populacao[it - 1 - 2 * POPULATION_SIZE].thread_y, kernel_time, res);
            printf("TERCEIRA GERAÇÃO \n valor indice: %d Indivíduo %d: thread_x = %d, thread_y = %d, fitness = %f\n", it, it - 1 - 2 * POPULATION_SIZE, populacao[it - 1 - 2 * POPULATION_SIZE].thread_x, populacao[it - 1 - 2 * POPULATION_SIZE].thread_y, populacao[it - 1 - 2 * POPULATION_SIZE].fitness);
            // printf("Indivíduo %d: thread_x = %d, thread_y = %d, fitness = %f\n", i - 1, populacao[(i - 1) / 2].thread_x, populacao[(i - 1) / 2].thread_y, populacao[(i - 1) / 2].fitness);
            // Gerar a nova subpopulação
            if (it == POPULATION_SIZE * 3)
            {
                printf("ta na hora da TERCEIRA Seleçao\n");
                Individuo *novaSubpopulacao = gerarNovaSubpopulacao(populacao);
                // Substituir a população atual pela nova subpopulação
                for (int j = 0; j < POPULATION_SIZE; j++)
                {
                    populacao[j] = novaSubpopulacao[j];
                    // printf("nova pop: %d thread x: %d thread y: %d\n", j, populacao[j].thread_x, populacao[j].thread_y);
                }
                //free(novaSubpopulacao);
            }
        }
        else
        {
            int melhorConfig = encontrarMelhorIndice(populacao);
            printf("\nBsize_x: %d \n", populacao[melhorConfig].thread_x);
            printf("Bsize_y: %d \n", populacao[melhorConfig].thread_y);
            const double t3=wtime();
            DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, populacao[melhorConfig].thread_x, populacao[melhorConfig].thread_y); 
            kernel_time=wtime()-t3;
            walltime+=kernel_time;
            printf("MELHOR CONFIG: %d x %d", populacao[melhorConfig].thread_x, populacao[melhorConfig].thread_y);
            res = (MEGA*(double)samplesPropagate)/kernel_time;
            printf("\nIteracao: %d ; Bsize_x: %d ; Bsize_y: %d ; tempoExec: %lf ; MSamples: %lf \n", it, populacao[melhorConfig].thread_x, populacao[melhorConfig].thread_y, kernel_time, res);
        }

    //const double t0=wtime();
    
    //DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, ch1dxx, ch1dyy, ch1dzz, ch1dxy, ch1dyz, ch1dxz, v2px, v2pz, v2sz, v2pn, pp, pc, qp, qc, bsize_x, bsize_y); //ajustar parametros
    //SwapArrays(&pp, &pc, &qp, &qc);

    //timeIt=wtime()-t0;
    //walltime+=timeIt;

    //printf("tempo deu: %lf\n", timeIt);

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
  
  FILE *fp=fopen("/proc/self/status","r");
  while (fgets(line, 256, fp) != NULL){
    if (strncmp(line, StringHWM, 5) == 0) {
      sscanf(line+6,"%ld %s", &HWM, HWMUnit);
      break;
    }
  }
  fclose(fp);
  //nao vamos salvar em disco

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

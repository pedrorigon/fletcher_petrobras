#define POPULATION_SIZE 30
#define NUM_GENERATIONS 3 
#define MAX_NUM_THREADS 64
#define MAX_MULTIPLICATION 1024
#define TOURNAMENT_SIZE 2   //define numero de participantes no torneio 
#define MUTATION_Y_PROBABILITY 0.2
#define MUTATION_X_PROBABILITY 0.0 

typedef struct
{
    int thread_x;
    int thread_y;
    double fitness;
} Individuo;

//int isPowerOfTwo(int num)
//{
    //return (num & (num - 1)) == 0;
//}

void inicializarPopulacao(Individuo *populacao, int tamanho_populacao)
{
    srand(time(NULL));

    int indices_fixos[][2] = {{32, 4}, {64, 8}, {32, 16}, {64, 4}};
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

    int vetor_x[] = {16, 32, 64};
    int vetor_y[] = {2, 4, 8, 16, 32, 64};
    int tamanho_vetor_x = sizeof(vetor_x) / sizeof(vetor_x[0]);
    int tamanho_vetor_y = sizeof(vetor_y) / sizeof(vetor_y[0]);


    while (populacao_atual < tamanho_populacao)
    {
	
    	// Inicializa a semente para geração de números aleatórios
    	srand(time(NULL));

    	// Gera um número aleatório entre para os indices
    	int indice_aleatorio_x = rand() % tamanho_vetor_x;
	    int indice_aleatorio_y = rand() % tamanho_vetor_y;
       
	    int x = vetor_x[indice_aleatorio_x];
      int y = vetor_y[indice_aleatorio_y];
      if (x * y < MAX_MULTIPLICATION)
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
        Individuo vencedor = populacao[rand() % POPULATION_SIZE]; //Escolhe um indivíduo aleatório como o primeiro para ser o atual vencedor do torneio

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
{ 
    // Verificar se a multiplicação dos valores é inferior a 1024
    if ((pais[0].thread_x * pais[1].thread_y < 1024) && (pais[1].thread_x * pais[0].thread_y < 1024))
    {
        // Realizar crossover trocando os valores entre os pais
        filhos[0].thread_x = pais[0].thread_x;
        filhos[0].thread_y = pais[1].thread_y;

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
            novaSubpopulacao[i] = filhos[0]; 
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

//aqui comeca a ideia 

int initialize(int it){
  Individuo populacao[POPULATION_SIZE];
  inicializarPopulacao(populacao, POPULATION_SIZE);

  int bsize_x = populacao[it - geracao*POPULATION_SIZE - 1].thread_x;
  int bsize_y = populacao[it - geracao*POPULATION_SIZE - 1].thread_y;

  return bsize_x, bsize_y;
}

//executa o kernel com bsize x e y 
//coleta o tempo desse kernel 

void genetic_exec(int it, double kernel_time, int bsize_x, int bsize_y){

  if(it % POPULATION_SIZE != 0 && it <= POPULATION_SIZE*3){
      populacao[it - geracao*POPULATION_SIZE - 1].fitness = 1 / kernel_time; //calcula o fitness da exec anterior
      bsize_x = populacao[it - geracao*POPULATION_SIZE - 1].thread_x;    //passa config p prox exec
      bsize_y = populacao[it - geracao*POPULATION_SIZE - 1].thread_y;
  }
  else if(it % POPULATION_SIZE == 0 && it <= POPULATION_SIZE*3){
    //fazer selecao 
    geracao = geracao + 1; //fazer ponteiro para pegar o valor atualizado 
    Individuo *novaSubpopulacao = gerarNovaSubpopulacao(populacao);
    // Substituir a população atual pela nova subpopulação
    for (int j = 0; j < POPULATION_SIZE; j++){
      populacao[j] = novaSubpopulacao[j];              
    }
  }
  else{
    //já achou melhor, só segue executando 
    int melhorConfig = encontrarMelhorIndice(populacao);
    bsize_x = populacao[melhorConfig].thread_x;
    bsize_y = populacao[melhorConfig].thread_y;
  }
}






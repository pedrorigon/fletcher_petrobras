# How-to

### Compilar e fazer teste rápido (Job Interativo)

- conectar no frontend:
	```bash
	ssh gppd-hpc.inf.ufrgs.br
	```

- alocar o servidor:
	```bash
	salloc -p blaise -t 02:00:00
	```

- entrar no servidor
	```bash
	ssh blaise
	```

- se for a blaise, faça:
	```bash
	cd $SSD1
	```
- caso contrário:
	```bash
	cd $SCRATCH
	```

- clonar o repositório:
	```bash
	git clone git@gitlab.com:msserpa/fletcher-3-0-petrobras.git
	```

- entrar no diretório dos experimentos:
	```bash
	cd fletcher-3-0-petrobras/exp/
	```

- carregar a env (ela está configurada para **blaise**, se trocar o servidor altere **PGCC_GPU_SM** e **CUDA_GPU_SM**:
	```bash
	source env.sh
	```

- para compilar:
	```bash
	./compile.sh
	```

- para executar um teste rápido:
	```bash
	./test.sh
	```

- - -

### Rodar experimento completo (Job não-interativo)

- conectar no frontend:
	```bash
	ssh gppd-hpc.inf.ufrgs.br
	```

- clonar o repositório:
	```bash
	git clone git@gitlab.com:msserpa/fletcher-3-0-petrobras.git
	```

- entrar no diretório dos experimentos:
	```bash
	cd $HOME/fletcher-3-0-petrobras/exp/
	```

- executar experimentos (lembrar de alterar o env.sh quando alterar a GPU):
	```bash
	NOW=`date +"%d-%m-%Y_%H-%M-%S"`

	sbatch --partition=blaise --time=24:00:00 --output="./slurm/$NOW.out" --error="./slurm/$NOW.err" time.batch

	```

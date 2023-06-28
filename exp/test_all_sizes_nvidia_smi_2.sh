#!/usr/bin/env bash

# Função para calcular o valor crítico de t a partir do nível de confiança e do tamanho da amostra
# Cuidar a dependência do módulo scipy que deve estar instalado
# Caso não estiver instalado --> "sudo apt-get install python3-scipy"
# Precisa de sudo -- verificar se PCAD tem --> SCIPY
function get_t_critical() {
    confidence_level=$1
    sample_size=$2
    degrees_of_freedom=$(echo "$sample_size - 1" | bc)
    t_critical=$(python3 -c "import scipy.stats; print(scipy.stats.t.ppf($confidence_level + (1 - $confidence_level)/2, $degrees_of_freedom))")
    echo $t_critical
}

# Cria a pasta output, se não existir
mkdir -p output

# Verifica se já existe um arquivo com o mesmo nome e renomeia, se necessário
output_file="resultados.csv"
counter=1
while [ -e "output/$output_file" ]; do
    output_file="resultados($counter).csv"
    ((counter++))
done
output_file="output/$output_file"

# Cria o arquivo CSV e adiciona o cabeçalho
echo "Aplicativo,Tamanho,Resultado Médio,Desvio Padrão,IC Inferior,IC Superior,Potência Média,Energia" > $output_file

cd bin/

for app in *.`hostname`.x; do
    echo "---------------------------------------------------"
    echo $app
    echo "---------------------------------------------------"
    
    for size in $(seq 88 32 408); do
        echo "Size: $size"

        total_msamples=0
        num_runs=10
        msamples_values=()
        total_gpu_usage=0
 	total_gpu_usage=0
        total_time=0
        start_time=0
        end_time=0
        

        for run in $(seq 1 $num_runs); do   
            if [[ $app == *"OpenACC"* ]]; then
                echo "GPU"
                echo "---------------------------------------------------"
                unset -v ACC_NUM_CORES
                export ACC_DEVICE_TYPE=nvidia
                ./$app TTI $size $size $size 16 12.5 12.5 12.5 0.001 0.005 16 32 | grep "MSamples/s"
                export ACC_DEVICE_TYPE=host
                export ACC_NUM_CORES=$(lscpu | grep "^CPU(s):" | awk {'print $2'})
                echo "---------------------------------------------------"
                echo "CPU"
                echo "---------------------------------------------------"
            fi
            # Início da contagem de tempo
            start_time=$(date +%s.%N)
            nvidia-smi dmon -s up -d 1 -o T > "../output/${app}_$size.txt" &
            nvidia_smi_pid=$!
            
            # Executa o aplicativo e salva o resultado filtrado no arquivo CSV
            result=$( { ./$app TTI $size $size $size 16 12.5 12.5 12.5 0.001 0.005 16 32; } 2>&1 )
            kill $nvidia_smi_pid
            # Cálculo do tempo de execução
            end_time=$(date +%s.%N)
            execution_time=$(echo "$end_time - $start_time" | bc)
            msamples=$(echo "$result" | grep "MSamples/s" || true)
            echo "$msamples"
            if [[ ! -z $msamples ]]; then
                msamples_value=$(echo $msamples | awk '{print $2}')
                total_msamples=$(echo "$total_msamples + $msamples_value" | bc)
                msamples_values+=($msamples_value)
            fi

           gpu_usage=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
           total_gpu_usage=$(echo "$total_gpu_usage + $gpu_usage" | bc)
           total_time=$(echo "$total_time + $execution_time" | bc)
           
        done
        pot_gpu_media=$(echo "scale=2; $total_gpu_usage / $num_runs" | bc)
        energy_gpu_consumed=$(echo "$total_time * $pot_gpu_media" | bc)
        # Calcula a média dos resultados
        average_msamples=$(echo "scale=2; $total_msamples / $num_runs" | bc)

        # Calcula a variância dos resultados
        variance=0
        for msamples_value in "${msamples_values[@]}"; do
            diff=$(echo "$msamples_value - $average_msamples" | bc)
            squared_diff=$(echo "scale=2; $diff * $diff" | bc)
            variance=$(echo "scale=2; $variance + $squared_diff" | bc)
        done

        variance=$(echo "scale=2; $variance / $num_runs" | bc)

        # Calcula o desvio padrão dos resultados
        stddev=$(echo "scale=2; sqrt($variance)" | bc)

        # Calcula o intervalo de confiança de 95%
        confidence_level=0.95
        t_critical=$(get_t_critical $confidence_level $num_runs)
        margin_of_error=$(echo "scale=2; $t_critical * $stddev / sqrt($num_runs)" | bc)
        lower_bound=$(echo "scale=2; $average_msamples - $margin_of_error" | bc)
        upper_bound=$(echo "scale=2; $average_msamples + $margin_of_error" | bc)

        # Salva a média, o desvio padrão, o intervalo de confiança, potência média e energia no arquivo CSV
        echo "$app,$size,$average_msamples,$stddev,$lower_bound,$upper_bound,$pot_gpu_media, $total_time, $energy_gpu_consumed" >> "../$output_file"
    done
done

cd ../


#!/usr/bin/env bash

# Função para obter o número total de GPUs no sistema
function get_total_gpus() {
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l
}

# Função para calcular o valor crítico de t
function get_t_critical() {
    confidence_level=$1
    sample_size=$2
    degrees_of_freedom=$(echo "$sample_size - 1" | bc)
    t_critical=$(python3 -c "import scipy.stats; print(scipy.stats.t.ppf($confidence_level + (1 - $confidence_level)/2, $degrees_of_freedom))")
    echo $t_critical
}

# Cria a pasta output e logs, se não existirem
mkdir -p output logs

# Setup inicial para nomear o arquivo de saída 
output_file="resultados.csv"
counter=1
while [ -e "output/$output_file" ]; do
    output_file="resultados($counter).csv"
    ((counter++))
done
output_file="output/$output_file"
echo "Aplicativo,Tamanho,Resultado Médio,Desvio Padrão,IC Inferior,IC Superior" > $output_file

cd bin/

for app in *.`hostname`.x; do
    echo "---------------------------------------------------"
    echo $app
    echo "---------------------------------------------------"
    
    # Criação de logs para o nvidia-smi para cada aplicativo
    mkdir -p ../logs/$app
    log_interval=10
    total_gpus=$(get_total_gpus)
    for gpu in $(seq 0 $(($total_gpus - 1))); do
        nvidia-smi -i $gpu -lms $log_interval > ../logs/$app/gpu$gpu.txt &
    done
    sleep $log_interval
    pkill -f "nvidia-smi -i"
    
    # Processamento dos logs para calcular médias de potência e consumo de energia
    declare -A avg_power
    declare -A total_energy
    for gpu in $(seq 0 $(($total_gpus - 1))); do
        log_file="../logs/$app/gpu$gpu.txt"
        total_power=0
        count=0
        while IFS= read -r line; do
            power=$(echo $line | awk -F ' ' '{print $13}' | tr -d 'W')
            total_power=$(echo "$total_power + $power" | bc)
            ((count++))
        done < "$log_file"
        avg_power[$gpu]=$(echo "scale=2; $total_power / $count" | bc)
        total_energy[$gpu]=$(echo "scale=2; $avg_power[$gpu] * $log_interval" | bc)
    done
    
    # Gera um arquivo .csv com a média da potência e a energia consumida
    csv_file="../logs/$app/power_energy_summary.csv"
    echo "GPU, Average Power (W), Total Energy (Joule)" > $csv_file
    for gpu in $(seq 0 $(($total_gpus - 1))); do
        echo "$gpu, ${avg_power[$gpu]}, ${total_energy[$gpu]}" >> $csv_file
    done

    # Continuação do seu loop original...
    for size in $(seq 88 32 408); do
        echo "Size: $size"
        total_msamples=0
        num_runs=10
        msamples_values=()

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

            # Executa o aplicativo e salva o resultado filtrado no arquivo CSV
            result=$( { ./$app TTI $size $size $size 16 12.5 12.5 12.5 0.001 0.005 16 32; } 2>&1 )
            msamples=$(echo "$result" | grep "MSamples/s" || true)
            echo "$msamples"
            if [[ ! -z $msamples ]]; then
                msamples_value=$(echo $msamples | awk '{print $2}')
                total_msamples=$(echo "$total_msamples + $msamples_value" | bc)
                msamples_values+=($msamples_value)
            fi
        done

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

        # Salva a média, o desvio padrão e o intervalo de confiança no arquivo CSV
        echo "$app,$size,$average_msamples,$stddev,$lower_bound,$upper_bound" >> "../$output_file"
    done
done

cd ../


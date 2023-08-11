#!/usr/bin/env bash

function get_t_critical() {
    confidence_level=$1
    sample_size=$2
    degrees_of_freedom=$(echo "$sample_size - 1" | bc)
    t_critical=$(python3 -c "import scipy.stats; print(scipy.stats.t.ppf($confidence_level + (1 - $confidence_level)/2, $degrees_of_freedom))")
    echo $t_critical
}

# Cria as pastas output e logs dentro dela, se não existirem
mkdir -p output/logs

# Verifica se já existe um arquivo com o mesmo nome e renomeia, se necessário
output_file="resultados.csv"
counter=1
while [ -e "output/$output_file" ]; do
    output_file="resultados($counter).csv"
    ((counter++))
done
output_file="output/$output_file"

# Cria o arquivo CSV e adiciona o cabeçalho
echo "Aplicativo,Tamanho,Resultado Médio,Desvio Padrão,IC Inferior,IC Superior" > $output_file

cd bin/

for app in *.`hostname`.x; do
    echo "---------------------------------------------------"
    echo $app | tee -a ../output/logs/${app}_summary.txt
    echo "---------------------------------------------------"
    
    for size in $(seq 88 32 408); do
        echo "Size: $size" | tee -a ../output/logs/${app}_size_${size}.txt

        total_msamples=0
        num_runs=5
        msamples_values=()

        for run in $(seq 1 $num_runs); do
            if [[ $app == *"OpenACC"* ]]; then
                echo "GPU" | tee -a ../output/logs/${app}_size_${size}.txt
                echo "---------------------------------------------------" | tee -a ../output/logs/${app}_size_${size}.txt
                unset -v ACC_NUM_CORES
                export ACC_DEVICE_TYPE=nvidia
                ./$app TTI $size $size $size 16 12.5 12.5 12.5 0.001 2 | tee -a ../output/logs/${app}_size_${size}.txt | grep "MSamples/s"
                export ACC_DEVICE_TYPE=host
                export ACC_NUM_CORES=$(lscpu | grep "^CPU(s):" | awk {'print $2'})
                echo "---------------------------------------------------" | tee -a ../output/logs/${app}_size_${size}.txt
                echo "CPU" | tee -a ../output/logs/${app}_size_${size}.txt
                echo "---------------------------------------------------" | tee -a ../output/logs/${app}_size_${size}.txt
            fi

            result=$( { ./$app TTI $size $size $size 16 12.5 12.5 12.5 0.001 2 | tee -a ../output/logs/${app}_size_${size}.txt; } 2>&1 )
            msamples=$(echo "$result" | grep "MSamples/s" || true)
            echo "$msamples" | tee -a ../output/logs/${app}_size_${size}.txt
            if [[ ! -z $msamples ]]; then
                msamples_value=$(echo $msamples | awk '{print $2}')
                total_msamples=$(echo "$total_msamples + $msamples_value" | bc)
                msamples_values+=($msamples_value)
            fi
        done

        average_msamples=$(echo "scale=2; $total_msamples / $num_runs" | bc)
        variance=0
        for msamples_value in "${msamples_values[@]}"; do
            diff=$(echo "$msamples_value - $average_msamples" | bc)
            squared_diff=$(echo "scale=2; $diff * $diff" | bc)
            variance=$(echo "scale=2; $variance + $squared_diff" | bc)
        done

        variance=$(echo "scale=2; $variance / $num_runs" | bc)
        stddev=$(echo "scale=2; sqrt($variance)" | bc)

        confidence_level=0.95
        t_critical=$(get_t_critical $confidence_level $num_runs)
        margin_of_error=$(echo "scale=2; $t_critical * $stddev / sqrt($num_runs)" | bc)
        lower_bound=$(echo "scale=2; $average_msamples - $margin_of_error" | bc)
        upper_bound=$(echo "scale=2; $average_msamples + $margin_of_error" | bc)

        echo "$app,$size,$average_msamples,$stddev,$lower_bound,$upper_bound" >> "../$output_file"

    done

done

cd ../


#!/usr/bin/env bash

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
echo "Aplicativo,Resultado Médio,Desvio Padrão" > $output_file

cd bin/

for app in *.`hostname`.x; do
    echo "---------------------------------------------------"
    echo $app
    echo "---------------------------------------------------"
    
    total_msamples=0
    num_runs=10
    msamples_values=()

    for run in $(seq 1 $num_runs); do
        if [[ $app == *"OpenACC"* ]]; then
            echo "GPU"
            echo "---------------------------------------------------"
            unset -v ACC_NUM_CORES
            export ACC_DEVICE_TYPE=nvidia
            ./$app TTI 344 344 344 16 12.5 12.5 12.5 0.001 0.005 | grep "MSamples/s"
            export ACC_DEVICE_TYPE=host
            export ACC_NUM_CORES=`lscpu | grep "^CPU(s):" | awk {'print $2'}`
            echo "---------------------------------------------------"
            echo "CPU"
            echo "---------------------------------------------------"
        fi
        
        # Executa o aplicativo e salva o resultado filtrado no arquivo CSV
        result=$( { ./$app TTI 344 344 344 16 12.5 12.5 12.5 0.001 0.005; } 2>&1 )
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

    # Salva a média e o desvio padrão no arquivo CSV
    echo "$app,$average_msamples,$stddev" >> "../$output_file"
done

cd ../


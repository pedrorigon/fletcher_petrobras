#!/usr/bin/env bash

function get_t_critical() {
    confidence_level=$1
    sample_size=$2
    degrees_of_freedom=$(echo "$sample_size - 1" | bc)
    t_critical=$(python3 -c "import scipy.stats; print(scipy.stats.t.ppf($confidence_level + (1 - $confidence_level)/2, $degrees_of_freedom))")
    echo $t_critical
}

mkdir -p output

output_file="resultados.csv"
counter=1
while [ -e "output/$output_file" ]; do
    output_file="resultados($counter).csv"
    ((counter++))
done
output_file="output/$output_file"

echo "Aplicativo, Bsize_X, Bsize_Y, Resultado Médio, Desvio Padrão, IC Inferior, IC Superior" > $output_file

cd bin/

bsize_values=("4 4" "4 8" "8 4" "4 16" "8 8" "16 4" "4 32" "8 16" "16 8" "32 4" "8 32" "16 16" "32 8" "16 32" "32 16" "32 32")

for app in *.`hostname`.x; do
    echo "---------------------------------------------------"
    echo $app
    echo "---------------------------------------------------"

    for bsize in "${bsize_values[@]}"; do
        bsize_x=$(echo $bsize | cut -d' ' -f1)
        bsize_y=$(echo $bsize | cut -d' ' -f2)

        echo "Bsize_X = $bsize_x, Bsize_Y = $bsize_y" # Imprimir valores

        total_msamples=0
        num_runs=10
        msamples_values=()

        for run in $(seq 1 $num_runs); do
            if [[ $app == *"OpenACC"* ]]; then
                echo "GPU"
                echo "---------------------------------------------------"
                unset -v ACC_NUM_CORES
                export ACC_DEVICE_TYPE=nvidia
                ./$app TTI 376 376 376 16 12.5 12.5 12.5 0.001 0.005 $bsize_x $bsize_y | grep "MSamples/s"
                export ACC_DEVICE_TYPE=host
                export ACC_NUM_CORES=`lscpu | grep "^CPU(s):" | awk {'print $2'}`
                echo "---------------------------------------------------"
                echo "CPU"
                echo "---------------------------------------------------"
            fi
            
            result=$( { ./$app TTI 376 376 376 16 12.5 12.5 12.5 0.001 0.005 $bsize_x $bsize_y; } 2>&1 )
            msamples=$(echo "$result" | grep "MSamples/s" || true)
            echo "$msamples"
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
        echo "$app, $bsize_x, $bsize_y, $average_msamples, $stddev, $lower_bound, $upper_bound" >> "../$output_file"

    done
done

cd ../


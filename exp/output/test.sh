#!/bin/bash

app="./$1"
size=$2

# Inicializa o comando de monitoramento do nvidia-smi
nvidia-smi dmon -i 0 -s mupcvt -d 1 -o TD > log.txt &

# Executa o comando desejado
$app TTI $size $size $size 16 12.5 12.5 12.5 0.001 0.005 16 32

# Aguarda o comando finalizar
wait

# Encerra o monitoramento do nvidia-smi
pkill -f "nvidia-smi dmon"

echo "Script finalizado. Os logs foram salvos em log.txt."


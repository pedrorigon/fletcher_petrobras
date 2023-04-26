#!/usr/bin/env bash

#set -o errexit -o nounset -o pipefail -o posix

mkdir -p bin/
cd ../

for version in der1der1 der1der1_cudaMemPrefetchAsync der1der1_unified_memory original original_unified_memory der1der1hm der1der1hm_unified_memory der1der1lm der1der1lm_unified_memory; do
	cd $version
	for backend in CUDA; do
		echo "-----------------------------------------------------"
		echo "   $version - $backend"
		echo "-----------------------------------------------------"
		make clean
		make backend=$backend
	        if [[ $backend == *"OpenACC"* ]]; then
                        cp ModelagemFletcher.exe ../exp/bin/$version.$backend-CPU.`hostname`.x
                        mv ModelagemFletcher.exe ../exp/bin/$version.$backend-GPU.`hostname`.x
		else
                        mv ModelagemFletcher.exe ../exp/bin/$version.$backend.`hostname`.x
		fi
	done
	cd ..
done

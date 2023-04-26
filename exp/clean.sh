#!/usr/bin/env bash

#set -o errexit -o nounset -o pipefail -o posix

mkdir -p bin/
cd ../

for version in der1der1 original; do
	cd $version
	for backend in OpenMP CUDA; do
		echo "-----------------------------------------------------"
		echo "   $version - $backend"
		echo "-----------------------------------------------------"
		make clean
                        rm -f ModelagemFletcher.exe
	done
	cd ..
done

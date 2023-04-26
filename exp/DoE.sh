#!/bin/bash

#set -o errexit -o nounset -o pipefail -o posix

declare -a APP=("original" "der1der1")

if [ "$1" == "cpu-only" ]; then
	declare -a VERSAO=("OpenMP" "OpenACC-CPU")
else
	declare -a VERSAO=("OpenMP" "CUDA")
fi

OUTPUT=$root/DoE/`hostname | awk -F. {'print $1'}`.csv
rm -f $OUTPUT

size=0
for application in "${APP[@]}"; do
	for version in "${VERSAO[@]}"; do
		for size in `seq 24 32 504`; do
			echo "$application;$version;$size" >> $OUTPUT
			size=$((size+1))
		done
	done
done

printf "\tcreating full factorial with $size runs ...\n"

for i in `seq 1 10`; do
	shuf $OUTPUT -o $OUTPUT
done

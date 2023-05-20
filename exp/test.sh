#!/usr/bin/env bash

#set -o errexit -o nounset -o pipefail -o posix

cd bin/

for app in *.`hostname`.x; do
	echo "---------------------------------------------------"
	echo $app
	echo "---------------------------------------------------"
	
	if [[ $app == *"OpenACC"* ]]; then
		echo "GPU"
		echo "---------------------------------------------------"
		unset -v ACC_NUM_CORES
		export ACC_DEVICE_TYPE=nvidia
  		./$app TTI 88 88 88 16 12.5 12.5 12.5 0.001 0.005 | grep Samples
  		export ACC_DEVICE_TYPE=host
		export ACC_NUM_CORES=`lscpu | grep "^CPU(s):" | awk {'print $2'}`
		echo "---------------------------------------------------"
		echo "CPU"
		echo "---------------------------------------------------"
	fi
	./$app TTI 88 88 88 16 12.5 12.5 12.5 0.001 0.005 | grep Samples
done

cd ../

mkdir -p bin/
cd ../

for version in original_mgpu_4_p2p; do
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

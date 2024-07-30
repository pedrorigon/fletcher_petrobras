mkdir -p bin/
cd ../

for version in original_openACC_mgpu original original_mgpu_2_p2p_PDP_FIX original_mgpu_4_p2p_PDP_FIX; do
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
			rm ../exp/bin/$version.$backend-CPU.`hostname`.x
		else
                        mv ModelagemFletcher.exe ../exp/bin/$version.$backend.`hostname`.x
		fi
	done
	cd ..
done

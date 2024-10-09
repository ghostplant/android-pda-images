shell:
	docker run -it --rm --privileged --shm-size=1g \
		-v /usr/bin/nvidia-smi:/usr/local/nvidia/bin/nvidia-smi \
		-v $(shell dirname `find /usr/lib/ -name libnvidia-ptxjitcompiler.so` 2>/dev/null | grep -v i386 | tail -n 1)/libnvidia-ptxjitcompiler.so:/usr/local/nvidia/lib/libnvidia-ptxjitcompiler.so \
		-v $(shell dirname `find /usr/lib/ -name libcuda.so.1` 2>/dev/null | grep -v i386 | tail -n 1):/usr/local/nvidia/lib64 \
		-v /:/host -w /host/$(shell pwd) nvcr.io/nvidia/pytorch:24.01-py3 bash

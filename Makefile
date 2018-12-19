SRCS = complexmat-test.cpp complexmat.hpp dynmem.hpp

complexmat-test-gpu: $(SRCS)
	/opt/hercules-compiler/bin/clang++ $< complexmat-test.cpp -std=c++11 -Wall -Wextra -DNDEBUG -g -O3 -fno-exceptions -fno-omit-frame-pointer -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xcuda-fatbinary -L/opt/hercules-compiler/lib/hercules -Xcuda-fatbinary -lpremnotify-gpu -L/opt/hercules-compiler/lib/hercules -lpremnotify-cpu -Xcuda-fatbinary -v -Xcuda-ptxas -v -Xcuda-ptxas -maxrregcount=32 --cuda-path=/usr/local/cuda-8.0  -o $@ 2>&1 | c++filt

complexmat-test-cpu: $(SRCS)
	/opt/hercules-compiler/bin/clang++ -Rpass=inline -DNDEBUG -g -O3 -std=c++11 -Wall -Wextra -pedantic -Wno-long-long -fno-exceptions -fno-omit-frame-pointer  -rdynamic $< /opt/hercules-compiler/lib/hercules/libpremnotify-cpu.a -o $@ 2>&1 | c++filt


run-gpu: complexmat-test-gpu
	LD_LIBRARY_PATH=/opt/hercules-compiler/lib ./complexmat-test-gpu

objdump:
	objdump -dSC complexmat-test-cpu

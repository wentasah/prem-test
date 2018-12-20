SRCS = complexmat-test.cpp complexmat.hpp dynmem.hpp

CXX=/opt/hercules-compiler/bin/clang++

CXXFLAGS-CPU=-Rpass=inline -DNDEBUG -g -O3 -std=c++11 -Wall -Wextra -fno-exceptions -fno-omit-frame-pointer -rdynamic -pedantic -Wno-long-long
CXXFLAGS-GPU=-Rpass=inline -DNDEBUG -g -O3 -std=c++11 -Wall -Wextra -fno-exceptions -fno-omit-frame-pointer -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -Xcuda-fatbinary -L/opt/hercules-compiler/lib/hercules -Xcuda-fatbinary -lpremnotify-gpu -L/opt/hercules-compiler/lib/hercules -lpremnotify-cpu -Xcuda-fatbinary -v -Xcuda-ptxas -v -Xcuda-ptxas -maxrregcount=32 --cuda-path=/usr/local/cuda-8.0

LIBPREMNOTIFY-CPU=/opt/hercules-compiler/lib/hercules/libpremnotify-cpu.a
complexmat-test-gpu: $(SRCS)
	$(CXX) $< $(CXXFLAGS-GPU) -o $@ 2>&1 | c++filt

complexmat-test-cpu: $(SRCS)
	$(CXX) $(CXXFLAGS-CPU) $< $(LIBPREMNOTIFY-CPU) -o $@ 2>&1 | c++filt

complexmat-test-cpu.ll: $(SRCS)
	$(CXX) $(CXXFLAGS-CPU) $< $(LIBPREMNOTIFY-CPU) -o $@ -S -emit-llvm 2>&1 | c++filt

run-gpu: complexmat-test-gpu
	LD_LIBRARY_PATH=/opt/hercules-compiler/lib ./complexmat-test-gpu

objdump: complexmat-test-cpu
	objdump -dSC complexmat-test-cpu

clean:
	rm -f complexmat-test-cpu complexmat-test-cpu.ll complexmat-test-gpu

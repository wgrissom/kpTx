# kpTx
k-Space-Domain Parallel Transmit Pulse Design

The main computation is done in an OpenMP-multithreaded MATLAB MEX function.
This is straightforward to compile in Linux, but can be challenging on a Mac. 
The following strategy worked for MATLAB R2020a running on Mac OS Catalina, 10.15.5:
1. Install Homebrew: [https://brew.sh](https://brew.sh)
2. Use Homebrew to install LLVM and OpenMP: `brew install llvm libomp`
3. Tell MATLAB to use a custom MEX setup for C compilation, which points to your llvm compiler and includes OpenMP flags: `mex -setup:/path/to/clang_openmp_maci64.xml C` (clang_openmp_maci64.xml is included in this repo)
4. Compile using, e.g.: `mex -largeArrayDims -lmwlapack LS_fft_mex_clean_OpenMP_Calloc_ForB0.c`

The following sites were helpful in figuring this out: 
- [https://stackoverflow.com/questions/37362414/openmp-with-mex-in-matlab-on-mac](https://stackoverflow.com/questions/37362414/openmp-with-mex-in-matlab-on-mac)
- [https://stackoverflow.com/questions/43555410/enable-openmp-support-in-clang-in-mac-os-x-sierra-mojave](https://stackoverflow.com/questions/43555410/enable-openmp-support-in-clang-in-mac-os-x-sierra-mojave)

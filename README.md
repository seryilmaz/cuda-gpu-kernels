# cuda-gpu-kernels
Custom GPU kernels written in CUDA

Includes custom kernels that can perform multiple independent matrix multiply tasks concurrently on GPU. This is functionally similar
to cublasSgemmBatched but with no extra kernel launch overhead. This code performs better than cublasSgemmBatched in most cases 
when matrix sizes are <= ~256 (or more generally when kernel launch overhead is significant compared to 1 matrix multiply task). Note that
this code is suboptimal for large matrices (>=1024) and although it can reach up to 78% of peak throughput for large matrices, 
it is still ~10-15% worse than cublasSgemm for large enough matrices. On the other hand, this code is very efficient when the task consists of
many independent matrix multiply jobs. For the extreme case when the matrix sizes are ~32, this code performs ~10x better than 
cudaSgemmBatched with many matrix multiply tasks.
Many techniques used here are borrowed from V. Volkov et al., GTC, 2010 [http://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf] 
and from R.  Nath et al., Int’l J. High Performance Computing Application, 2010 [http://journals.sagepub.com/doi/abs/10.1177/1094342010385729].

Experiments are done on Quadro K2200 (compute capability 5.0).



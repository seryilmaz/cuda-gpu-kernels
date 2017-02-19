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

Experiments are done on Quadro K2200 (compute capability 5.0). Compiled with nvcc and with -std=c++11 option. CUDA version is 8.0.
OS is Ubuntu 14.04. 

Kernel usage:
Run this to go into the relevant directory:
cd ConcurrMatMul 

Run "make" (optional).
Then you can run ./multiMatrixMul to execute concurrent matrix multiply operation. It completes multiple C=AB matrix multiply tasks.
Command line arguments:

-hA : height of matrix A. Default is 128

-wA : width of matrix A. Default is 128

-wB : width of matrix B. Default is 128

-numOfTasks : number of independent matrix multiply ops you want to run concurrently. Default is 100.

-highCompute : Set this to 1 if your matrices are of size >64, and you have multiple matrix multiply tasks to fill the machine.
If you have less than 10 matrix multiply jobs and your matrices are <=64 in size, you might get better performance if you set this to 0. This options should not affect the end result but it affects performance. Default is 1. 

-blockSize : This is kernel level option. If you set highCompute to 0, this is always 32 and is ignored. Otherwise, options are: 
16, 32, 64, 96, 128. If your matrices are at least of size 128, set this to 128. If you have small matrices, setting this to small values might give better results. Default is 128.

-nIter : Number of repetitions for the job (repeats the whole job nIter times). Larger nIter is useful to have a statistically robust performance measure. If you set nIter to 30 and numOfTasks to 100, then it will execute 30 jobs in a for loop, where each
job consists of 100 independent concurrent matrix multiply operations. Default is 30. 

-checkCorrectness : Set to 1 if you would like to have the result from GPU to be compared to result from CPU. Set to 0 if you don't want the comparison. Setting this to 1 will result in longer execution. A good method is setting this to 1 for new matrix
sizes or numOfTasks when the settings are used the first time to make sure GPU gives correct results; and setting it to 0 when the same settings are used again. Default is 0.



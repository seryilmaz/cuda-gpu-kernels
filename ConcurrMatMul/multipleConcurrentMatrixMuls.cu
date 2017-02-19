//This code is produced by Burc Eryilmaz. Please include this notice if you would like to use this code.
//Minor portions of this code is taken from CUDA 8.0 samples. Some techniques used in different kernels are 
//taken from Nath et al. and Volkov et al. (see README for more details).
#include <iostream>

using namespace std;

//this constInit function is from CUDA samples.
void constInit(float* array, int arraySize){

  for(int i=0; i<arraySize; i++){
    array[i]=0.1f;
  }
}

void matMulCPU(float *A, float *B, float *C, int hA, int wA, int wB){
  for(int i=0;i<hA;i++)
    for(int j=0;j<wB;j++){
      C[i*wB+j] = 0;
      for(int k=0;k<wA;k++)
        C[i*wB+j] += A[i*wA+k]*B[k*wB+j]; 
    }
}


bool compareArrays(float* A, float* B, int arraySize){
  for(int i=0;i<arraySize;i++){
    if(abs(A[i]-B[i])/abs(max(A[i],B[i])) > 1e-4) {
      printf("A:%.9f B:%.9f i:%d\n",A[i],B[i],i);
      return 0;}
  }
  return 1;
}

//this kernel employs techniques from V. Volkov et al., GTC 2010 (see README for details).
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA0B(float **Cl, float **Al, float **Bl, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *A, *B, *C ;
    A = Al[blockIdx.z];
    B = Bl[blockIdx.z];
    C = Cl[blockIdx.z];

    // index of the first sub-matrix of A processed by the block
    int start_a = wA * BLOCK_SIZE * by;

    // index of the last sub-matrix of A processed by the block
    int end_a   = start_a + wA - 1;

    // step size used to iterate through the sub-matrices of A
    int step_a  = BLOCK_SIZE;

    // index of the first sub-matrix of B processed by the block
    int start_b = BLOCK_SIZE * bx;

    // step size used to iterate through the sub-matrices of B
    int step_b  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub[8]={0,0,0,0,0,0,0,0};
    
    // loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = start_a, b = start_b;
         a <= end_a;
         a += step_a, b += step_b)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix     
        
        #pragma unroll 8
        for(int i=0; i<8;i+=1){
          As[ty+4*i][tx] = A[a + wA * (ty+4*i) + tx];
          Bs[ty+4*i][tx] = B[b + wB * (ty+4*i) + tx]; 
        } 
 
        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        #pragma unroll 32

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {

            Csub[0] += As[ty][k] * Bs[k][tx];
            Csub[1] += As[ty+4][k] * Bs[k][tx];
            Csub[2] += As[ty+8][k] * Bs[k][tx];
            Csub[3] += As[ty+12][k] * Bs[k][tx];   
            Csub[4] += As[ty+16][k] * Bs[k][tx];
            Csub[5] += As[ty+20][k] * Bs[k][tx];
            Csub[6] += As[ty+24][k] * Bs[k][tx];
            Csub[7] += As[ty+28][k] * Bs[k][tx];                      
        }

        // synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    #pragma unroll 8
    for(int i=0; i<8;i+=1) C[c + wB * (ty+4*i) + tx] = Csub[i];
}

//This kernel employs techniques from 
//R. Nath et al., Int’l J. High Performance Computing Application, 2010
//see README for details
template <int block, int block_iter> __global__ void
matrixMulCUDA32B(float **Cl, float **Al, float **Bl, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *A, *B, *C ;
    A = Al[blockIdx.z];
    B = Bl[blockIdx.z];
    C = Cl[blockIdx.z];
    
    // Index of the first sub-matrix of A processed by the block
    //block is 64*64; we will process 16 by 16 in the inner loop of the threads in thread block
    int start_a = by * wA * block;
    int end_a = start_a + wA-1;
    int step_a = block_iter;
    int start_b = bx * block;
    int step_b = wB * block_iter;

    float Csub[2][2]={0,0,0,0};
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = start_a, b = start_b;
         a <=end_a ;
         a += step_a, b += step_b)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[block][block_iter];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[block_iter][block];



        Bs[ty][tx] = B[b + ty*wB + tx];
        Bs[ty][tx+16] = B[b + ty*wB + tx+16];
             
        As[ty][tx] = A[a + ty*wA + tx];
        As[ty+16][tx] = A[a + (ty+16)*wA + tx];
                    
        __syncthreads();
        
        #pragma unroll 
        for(int i=0;i<16;i++){              
          Csub[0][0] += As[ty][i]*Bs[i][tx];
          Csub[0][1] += As[ty][i]*Bs[i][tx+16];       
          Csub[1][0] += As[ty+16][i]*Bs[i][tx];
          Csub[1][1] += As[ty+16][i]*Bs[i][tx+16];                 
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes 16 elements
    int c = by*block*wB + block*bx;
    #pragma unroll 
    for(int i=0; i<2; i++){
      C[c+(ty+16*i)*wB+tx] = Csub[i][0]; 
      C[c+(ty+16*i)*wB+tx+16] = Csub[i][1]; 
    }
}

//This kernel employs techniques from 
//R. Nath et al., Int’l J. High Performance Computing Application, 2010
//see README for details
template <int block, int block_iter> __global__ void
matrixMulCUDA16B(float **Cl, float **Al, float **Bl, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *A, *B, *C ;
    A = Al[blockIdx.z];
    B = Bl[blockIdx.z];
    C = Cl[blockIdx.z];
    // Index of the first sub-matrix of A processed by the block
    //block is 64*64; we will process 16 by 16 in the inner loop of the threads in thread block
    int start_a = by * wA * block;
    int end_a = start_a + wA-1;
    int step_a = block_iter;
    int start_b = bx * block;
    int step_b = wB * block_iter;

    float Csub=0;
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = start_a, b = start_b;
         a <=end_a ;
         a += step_a, b += step_b)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[block][block_iter];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[block_iter][block];

        Bs[ty][tx] = B[b + ty*wB + tx];            
        As[ty][tx] = A[a + ty*wA + tx];
                   
        __syncthreads();
        
        #pragma unroll 
        for(int i=0;i<16;i++){              
          Csub += As[ty][i]*Bs[i][tx];                      
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes 16 elements
    int c = by*block*wB + block*bx;

    C[c+ty*wB+tx] = Csub; 

}

//This kernel employs techniques from 
//R. Nath et al., Int’l J. High Performance Computing Application, 2010
//see README for details
template <int block, int block_iter> __global__ void
matrixMulCUDA64B(float **Cl, float **Al, float **Bl, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *A, *B, *C ;
    A = Al[blockIdx.z];
    B = Bl[blockIdx.z];
    C = Cl[blockIdx.z]; 
    
    // Index of the first sub-matrix of A processed by the block
    //block is 64*64; we will process 16 by 16 in the inner loop of the threads in thread block
    int start_a = by * wA * block;
    int end_a = start_a + wA-1;
    int step_a = block_iter;
    int start_b = bx * block;
    int step_b = wB * block_iter;

    float Csub[4][4]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = start_a, b = start_b;
         a <=end_a ;
         a += step_a, b += step_b)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[block][block_iter];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[block_iter][block];
 

        Bs[ty][tx] = B[b + ty*wB + tx];
        Bs[ty][tx+16] = B[b + ty*wB + tx+16];
        Bs[ty][tx+32] = B[b + ty*wB + tx+32];
        Bs[ty][tx+48] = B[b + ty*wB + tx+48];
             
        As[ty][tx] = A[a + ty*wA + tx];
        As[ty+16][tx] = A[a + (ty+16)*wA + tx];
        As[ty+32][tx] = A[a + (ty+32)*wA + tx];
        As[ty+48][tx] = A[a + (ty+48)*wA + tx];  
             
        __syncthreads();
        

        #pragma unroll 
        for(int i=0;i<16;i++){              
          Csub[0][0] += As[ty][i]*Bs[i][tx];
          Csub[0][1] += As[ty][i]*Bs[i][tx+16];
          Csub[0][2] += As[ty][i]*Bs[i][tx+32];
          Csub[0][3] += As[ty][i]*Bs[i][tx+48];       
          Csub[1][0] += As[ty+16][i]*Bs[i][tx];
          Csub[1][1] += As[ty+16][i]*Bs[i][tx+16];
          Csub[1][2] += As[ty+16][i]*Bs[i][tx+32];
          Csub[1][3] += As[ty+16][i]*Bs[i][tx+48];       
          Csub[2][0] += As[ty+32][i]*Bs[i][tx];
          Csub[2][1] += As[ty+32][i]*Bs[i][tx+16];
          Csub[2][2] += As[ty+32][i]*Bs[i][tx+32];
          Csub[2][3] += As[ty+32][i]*Bs[i][tx+48];        
          Csub[3][0] += As[ty+48][i]*Bs[i][tx];
          Csub[3][1] += As[ty+48][i]*Bs[i][tx+16];
          Csub[3][2] += As[ty+48][i]*Bs[i][tx+32];
          Csub[3][3] += As[ty+48][i]*Bs[i][tx+48];              
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes 16 elements
    int c = by*block*wB + block*bx;
    #pragma unroll 
    for(int i=0; i<4; i++){
      C[c+(ty+16*i)*wB+tx] = Csub[i][0]; 
      C[c+(ty+16*i)*wB+tx+16] = Csub[i][1]; 
      C[c+(ty+16*i)*wB+tx+32] = Csub[i][2]; 
      C[c+(ty+16*i)*wB+tx+48] = Csub[i][3]; 
    }
}

//This kernel employs techniques from 
//R. Nath et al., Int’l J. High Performance Computing Application, 2010
//see README for details
template <int block, int block_iter> __global__ void
matrixMulCUDA96B(float **Cl, float **Al, float **Bl, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float *A, *B, *C ;
    A = Al[blockIdx.z];
    B = Bl[blockIdx.z];
    C = Cl[blockIdx.z];    
    // Index of the first sub-matrix of A processed by the block
    //block is 64*64; we will process 16 by 16 in the inner loop of the threads in thread block
    int start_a = by * wA * block;
    int end_a = start_a + wA-1;
    int step_a = block_iter;
    int start_b = bx * block;
    int step_b = wB * block_iter;
    float Csub[6][6]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
       
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = start_a, b = start_b;
         a <=end_a ;
         a += step_a, b += step_b)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[block][block_iter];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[block_iter][block];  
         
        Bs[ty][tx] = B[b + ty*wB + tx];
        Bs[ty][tx+16] = B[b + ty*wB + tx+16];
        Bs[ty][tx+32] = B[b + ty*wB + tx+32];
        Bs[ty][tx+48] = B[b + ty*wB + tx+48];
        Bs[ty][tx+64] = B[b + ty*wB + tx+64];
        Bs[ty][tx+80] = B[b + ty*wB + tx+80];
                    
        As[ty][tx] = A[a + ty*wA + tx];
        As[ty+16][tx] = A[a + (ty+16)*wA + tx];
        As[ty+32][tx] = A[a + (ty+32)*wA + tx];
        As[ty+48][tx] = A[a + (ty+48)*wA + tx];  
        As[ty+64][tx] = A[a + (ty+64)*wA + tx];
        As[ty+80][tx] = A[a + (ty+80)*wA + tx];  
                    
        __syncthreads();
        

        #pragma unroll 
        for(int i=0;i<16;i++){              
          Csub[0][0] += As[ty][i]*Bs[i][tx];
          Csub[0][1] += As[ty][i]*Bs[i][tx+16];
          Csub[0][2] += As[ty][i]*Bs[i][tx+32];
          Csub[0][3] += As[ty][i]*Bs[i][tx+48]; 
          Csub[0][4] += As[ty][i]*Bs[i][tx+64];
          Csub[0][5] += As[ty][i]*Bs[i][tx+80];
                                    
          Csub[1][0] += As[ty+16][i]*Bs[i][tx];
          Csub[1][1] += As[ty+16][i]*Bs[i][tx+16];
          Csub[1][2] += As[ty+16][i]*Bs[i][tx+32];
          Csub[1][3] += As[ty+16][i]*Bs[i][tx+48]; 
          Csub[1][4] += As[ty+16][i]*Bs[i][tx+64];
          Csub[1][5] += As[ty+16][i]*Bs[i][tx+80];          
                
          Csub[2][0] += As[ty+32][i]*Bs[i][tx];
          Csub[2][1] += As[ty+32][i]*Bs[i][tx+16];
          Csub[2][2] += As[ty+32][i]*Bs[i][tx+32];
          Csub[2][3] += As[ty+32][i]*Bs[i][tx+48]; 
          Csub[2][4] += As[ty+32][i]*Bs[i][tx+64];
          Csub[2][5] += As[ty+32][i]*Bs[i][tx+80];

          Csub[3][0] += As[ty+48][i]*Bs[i][tx];
          Csub[3][1] += As[ty+48][i]*Bs[i][tx+16];
          Csub[3][2] += As[ty+48][i]*Bs[i][tx+32];
          Csub[3][3] += As[ty+48][i]*Bs[i][tx+48]; 
          Csub[3][4] += As[ty+48][i]*Bs[i][tx+64];
          Csub[3][5] += As[ty+48][i]*Bs[i][tx+80];
          
          Csub[4][0] += As[ty+64][i]*Bs[i][tx];
          Csub[4][1] += As[ty+64][i]*Bs[i][tx+16];
          Csub[4][2] += As[ty+64][i]*Bs[i][tx+32];
          Csub[4][3] += As[ty+64][i]*Bs[i][tx+48]; 
          Csub[4][4] += As[ty+64][i]*Bs[i][tx+64];
          Csub[4][5] += As[ty+64][i]*Bs[i][tx+80];

          Csub[5][0] += As[ty+80][i]*Bs[i][tx];
          Csub[5][1] += As[ty+80][i]*Bs[i][tx+16];
          Csub[5][2] += As[ty+80][i]*Bs[i][tx+32];
          Csub[5][3] += As[ty+80][i]*Bs[i][tx+48]; 
          Csub[5][4] += As[ty+80][i]*Bs[i][tx+64];
          Csub[5][5] += As[ty+80][i]*Bs[i][tx+80];             
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes 16 elements
    int c = by*block*wB + block*bx;
    #pragma unroll 
    for(int i=0; i<6; i++){
      C[c+(ty+16*i)*wB+tx] = Csub[i][0]; 
      C[c+(ty+16*i)*wB+tx+16] = Csub[i][1]; 
      C[c+(ty+16*i)*wB+tx+32] = Csub[i][2]; 
      C[c+(ty+16*i)*wB+tx+48] = Csub[i][3]; 
      C[c+(ty+16*i)*wB+tx+64] = Csub[i][4]; 
      C[c+(ty+16*i)*wB+tx+80] = Csub[i][5];       
    }
}

//This kernel employs techniques from 
//R. Nath et al., Int’l J. High Performance Computing Application, 2010
//see README for details
template <int block, int block_iter> __global__ void
matrixMulCUDA128B(float **Cl, float **Al, float **Bl, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    //int bz = blockIdx.z;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float *A, *B, *C ;
    A = Al[blockIdx.z];
    B = Bl[blockIdx.z];
    C = Cl[blockIdx.z];
    
    // Index of the first sub-matrix of A processed by the block
    //block is 64*64; we will process 16 by 16 in the inner loop of the threads in thread block
    int start_a = by * wA * block;
    int end_a = start_a + wA-1;
    int step_a = block_iter;
    int start_b = bx * block;
    int step_b = wB * block_iter;
    float Csub[8][8]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
       
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = start_a, b = start_b;
         a <=end_a ;
         a += step_a, b += step_b)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[block][block_iter];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[block_iter][block];  
         
        Bs[ty][tx] = B[b + ty*wB + tx];
        Bs[ty][tx+16] = B[b + ty*wB + tx+16];
        Bs[ty][tx+32] = B[b + ty*wB + tx+32];
        Bs[ty][tx+48] = B[b + ty*wB + tx+48];
        Bs[ty][tx+64] = B[b + ty*wB + tx+64];
        Bs[ty][tx+80] = B[b + ty*wB + tx+80];
        Bs[ty][tx+96] = B[b + ty*wB + tx+96];
        Bs[ty][tx+112] = B[b + ty*wB + tx+112];        
                    
        As[ty][tx] = A[a + ty*wA + tx];
        As[ty+16][tx] = A[a + (ty+16)*wA + tx];
        As[ty+32][tx] = A[a + (ty+32)*wA + tx];
        As[ty+48][tx] = A[a + (ty+48)*wA + tx];  
        As[ty+64][tx] = A[a + (ty+64)*wA + tx];
        As[ty+80][tx] = A[a + (ty+80)*wA + tx];  
        As[ty+96][tx] = A[a + (ty+96)*wA + tx];
        As[ty+112][tx] = A[a + (ty+112)*wA + tx];                     
        __syncthreads();
        

        #pragma unroll 
        for(int i=0;i<16;i++){              
          Csub[0][0] += As[ty][i]*Bs[i][tx];
          Csub[0][1] += As[ty][i]*Bs[i][tx+16];
          Csub[0][2] += As[ty][i]*Bs[i][tx+32];
          Csub[0][3] += As[ty][i]*Bs[i][tx+48]; 
          Csub[0][4] += As[ty][i]*Bs[i][tx+64];
          Csub[0][5] += As[ty][i]*Bs[i][tx+80];
          Csub[0][6] += As[ty][i]*Bs[i][tx+96];
          Csub[0][7] += As[ty][i]*Bs[i][tx+112];
                                    
          Csub[1][0] += As[ty+16][i]*Bs[i][tx];
          Csub[1][1] += As[ty+16][i]*Bs[i][tx+16];
          Csub[1][2] += As[ty+16][i]*Bs[i][tx+32];
          Csub[1][3] += As[ty+16][i]*Bs[i][tx+48]; 
          Csub[1][4] += As[ty+16][i]*Bs[i][tx+64];
          Csub[1][5] += As[ty+16][i]*Bs[i][tx+80]; 
          Csub[1][6] += As[ty+16][i]*Bs[i][tx+96];
          Csub[1][7] += As[ty+16][i]*Bs[i][tx+112];                   
                
          Csub[2][0] += As[ty+32][i]*Bs[i][tx];
          Csub[2][1] += As[ty+32][i]*Bs[i][tx+16];
          Csub[2][2] += As[ty+32][i]*Bs[i][tx+32];
          Csub[2][3] += As[ty+32][i]*Bs[i][tx+48]; 
          Csub[2][4] += As[ty+32][i]*Bs[i][tx+64];
          Csub[2][5] += As[ty+32][i]*Bs[i][tx+80];
          Csub[2][6] += As[ty+32][i]*Bs[i][tx+96];
          Csub[2][7] += As[ty+32][i]*Bs[i][tx+112]; 
          
          Csub[3][0] += As[ty+48][i]*Bs[i][tx];
          Csub[3][1] += As[ty+48][i]*Bs[i][tx+16];
          Csub[3][2] += As[ty+48][i]*Bs[i][tx+32];
          Csub[3][3] += As[ty+48][i]*Bs[i][tx+48]; 
          Csub[3][4] += As[ty+48][i]*Bs[i][tx+64];
          Csub[3][5] += As[ty+48][i]*Bs[i][tx+80];
          Csub[3][6] += As[ty+48][i]*Bs[i][tx+96];
          Csub[3][7] += As[ty+48][i]*Bs[i][tx+112]; 
                    
          Csub[4][0] += As[ty+64][i]*Bs[i][tx];
          Csub[4][1] += As[ty+64][i]*Bs[i][tx+16];
          Csub[4][2] += As[ty+64][i]*Bs[i][tx+32];
          Csub[4][3] += As[ty+64][i]*Bs[i][tx+48]; 
          Csub[4][4] += As[ty+64][i]*Bs[i][tx+64];
          Csub[4][5] += As[ty+64][i]*Bs[i][tx+80];
          Csub[4][6] += As[ty+64][i]*Bs[i][tx+96];
          Csub[4][7] += As[ty+64][i]*Bs[i][tx+112];           

          Csub[5][0] += As[ty+80][i]*Bs[i][tx];
          Csub[5][1] += As[ty+80][i]*Bs[i][tx+16];
          Csub[5][2] += As[ty+80][i]*Bs[i][tx+32];
          Csub[5][3] += As[ty+80][i]*Bs[i][tx+48]; 
          Csub[5][4] += As[ty+80][i]*Bs[i][tx+64];
          Csub[5][5] += As[ty+80][i]*Bs[i][tx+80];             
          Csub[5][6] += As[ty+80][i]*Bs[i][tx+96];
          Csub[5][7] += As[ty+80][i]*Bs[i][tx+112]; 

          Csub[6][0] += As[ty+96][i]*Bs[i][tx];
          Csub[6][1] += As[ty+96][i]*Bs[i][tx+16];
          Csub[6][2] += As[ty+96][i]*Bs[i][tx+32];
          Csub[6][3] += As[ty+96][i]*Bs[i][tx+48]; 
          Csub[6][4] += As[ty+96][i]*Bs[i][tx+64];
          Csub[6][5] += As[ty+96][i]*Bs[i][tx+80];             
          Csub[6][6] += As[ty+96][i]*Bs[i][tx+96];
          Csub[6][7] += As[ty+96][i]*Bs[i][tx+112]; 

          Csub[7][0] += As[ty+112][i]*Bs[i][tx];
          Csub[7][1] += As[ty+112][i]*Bs[i][tx+16];
          Csub[7][2] += As[ty+112][i]*Bs[i][tx+32];
          Csub[7][3] += As[ty+112][i]*Bs[i][tx+48]; 
          Csub[7][4] += As[ty+112][i]*Bs[i][tx+64];
          Csub[7][5] += As[ty+112][i]*Bs[i][tx+80];             
          Csub[7][6] += As[ty+112][i]*Bs[i][tx+96];
          Csub[7][7] += As[ty+112][i]*Bs[i][tx+112];           

        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes 16 elements
    int c = by*block*wB + block*bx;
    #pragma unroll 
    for(int i=0; i<8; i++){
      C[c+(ty+16*i)*wB+tx] = Csub[i][0]; 
      C[c+(ty+16*i)*wB+tx+16] = Csub[i][1]; 
      C[c+(ty+16*i)*wB+tx+32] = Csub[i][2]; 
      C[c+(ty+16*i)*wB+tx+48] = Csub[i][3]; 
      C[c+(ty+16*i)*wB+tx+64] = Csub[i][4]; 
      C[c+(ty+16*i)*wB+tx+80] = Csub[i][5]; 
      C[c+(ty+16*i)*wB+tx+96] = Csub[i][6]; 
      C[c+(ty+16*i)*wB+tx+112] = Csub[i][7];       
    }
}

int matMulGPU(int hA,int wA, int wB,int num, bool highCompute, int blockSize, int nIter, bool checkCorrectness){
  //declare pointers for pointer arrays. Each pointer in those arrays will point to a float array in the host
  float **Al, **Bl, **Cl;
  //declare pointers for pointer arrays. Each pointer in those arrays will point to a float array in the device
  //d prefix is for device pointers, and hd are for host pointers. All will point to float arrays in device though.
  float **d_Al, **d_Bl, **d_Cl, **hd_Al, **hd_Bl, **hd_Cl;
  
  //Allocate pointer arrays
  Al = (float**) malloc(num*sizeof(float*));
  Bl = (float**) malloc(num*sizeof(float*));
  Cl = (float**) malloc(num*sizeof(float*));

  hd_Al = (float**) malloc(num*sizeof(float*));
  hd_Bl = (float**) malloc(num*sizeof(float*));
  hd_Cl = (float**) malloc(num*sizeof(float*));
  
  //allocate float arrays pointed by pointers in the arrays declared above
  for(int i=0;i<num;i++){
    Al[i] = (float*) malloc(wA*hA*sizeof(float));
    Bl[i] = (float*) malloc(wA*wB*sizeof(float));
    Cl[i] = (float*) malloc(hA*wB*sizeof(float));
    constInit(Al[i],wA*hA);
    constInit(Bl[i],wA*wB);
  }
  //this will be used to transfer result of first matrix multiply op to host
  float *C2;
  C2 = (float*) malloc(hA*wB*sizeof(float));
  
  //compute the result of first matrix multiply op in CPU for comparison
  if(checkCorrectness==1) matMulCPU(Al[0],Bl[0],Cl[0],hA,wA,wB);

  //allocate float arrays in the device
  for(int i=0;i<num;i++){
    cudaMalloc(&(hd_Al[i]),sizeof(float)*wA*hA);
    cudaMalloc(&hd_Bl[i],sizeof(float)*wA*wB);
    cudaMalloc(&hd_Cl[i],sizeof(float)*hA*wB);   
    cudaMemcpy(hd_Al[i], Al[i],sizeof(float)*wA*hA,cudaMemcpyHostToDevice );
    cudaMemcpy(hd_Bl[i], Bl[i],sizeof(float)*wA*wB,cudaMemcpyHostToDevice );       
  }  
  cudaMalloc(&d_Al,num*sizeof(float*));
  cudaMalloc(&d_Bl,num*sizeof(float*));
  cudaMalloc(&d_Cl,num*sizeof(float*));
  cudaMemcpy(d_Al, hd_Al, num*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Bl, hd_Bl, num*sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Cl, hd_Cl, num*sizeof(float*), cudaMemcpyHostToDevice);

  //declare and initialize threads and grid variables, according to the command line options
  dim3 threads;
  dim3 grid;
  cudaError_t error;
  if(highCompute==0){
    threads.x=32;
    threads.y=4;
    grid.x=wB / 32;
    grid.y=hA / 32;
    grid.z=num;        
    //do some warmup
    matrixMulCUDA0B<32><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
  } else{
    threads.x=16;
    threads.y=16;
    grid.x=wB / blockSize;
    grid.y=hA / blockSize;
    grid.z=num;     
    if( ( blockSize%32 != 0 && blockSize!=16 ) || blockSize > 128 || blockSize<16){
      printf("Not a valid block size. Block size needs to be positive and at most 128, and a multiple of 32 or equal to 16\n");
      return -1;
    }  
    if(wB%blockSize != 0 || hA%blockSize != 0){
      printf("Not a valid combination of matrix sizes and block size. width(B) and height(A) must be a multiple of block size, and width(A) a multiple of 16\n");
      return -1;
    } 

    //do some warmup
    if(blockSize==16) matrixMulCUDA16B<16,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
    if(blockSize==32) matrixMulCUDA32B<32,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
    if(blockSize==64) matrixMulCUDA64B<64,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
    if(blockSize==96) matrixMulCUDA96B<96,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
    if(blockSize==128) matrixMulCUDA128B<128,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
  }
  
  cudaDeviceSynchronize();
  cudaEvent_t start;
  error = cudaEventCreate(&start);
  if(error != cudaSuccess){
    fprintf(stderr, "Failed to create start event %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
  
  cudaEvent_t stop ;
  error = cudaEventCreate(&stop);
  if(error != cudaSuccess){
    fprintf(stderr, "Failed to create stop event %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
 
  error = cudaEventRecord(start, 0);
  
  if(error != cudaSuccess){
    fprintf(stderr, "Failed to record start event %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
  
  //perform computation nIter times
  for (int j = 0; j < nIter; j++)
  {
    if(highCompute==0){
      matrixMulCUDA0B<32><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
    } else{
      if(blockSize==16) matrixMulCUDA16B<16,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
      if(blockSize==32) matrixMulCUDA32B<32,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
      if(blockSize==64) matrixMulCUDA64B<64,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
      if(blockSize==96) matrixMulCUDA96B<96,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
      if(blockSize==128) matrixMulCUDA128B<128,16><<< grid, threads >>>(d_Cl, d_Al, d_Bl, wA,wB);
    }
  }  
  
  error = cudaEventRecord(stop, 0);
  if(error != cudaSuccess){
    fprintf(stderr, "Failed to record stop event %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }  
  error = cudaEventSynchronize(stop);
  float msecElapsed =0.0f;
  error = cudaEventElapsedTime(&msecElapsed, start,stop);
  
  float totalOps = 2.0f*hA*wA*wB*nIter*num;
  float Gflops = totalOps/msecElapsed*1000 * 1.0e-9f;
  
  printf("Performance: %.4f GFLOPS, total time: %.3f msec, total Ops: %.3f Gops\n", Gflops, msecElapsed,totalOps*1.0e-9f);
  
  cudaMemcpy(C2,hd_Cl[0],sizeof(float)*hA*wB,cudaMemcpyDeviceToHost);
  if(checkCorrectness==1){
    bool success = compareArrays(Cl[0],C2,hA*wB);
    if(success==1) printf("Comparison of first matrix multiply passed\n");
    else printf("Comparison first matrix multiply failed\n");  
  }
    
  free(C2);
  
  //free allocated memory for all matrices seperately
  for(int i=0;i<num;i++){
    free(Al[i]); free(Bl[i]); free(Cl[i]); 
    cudaFree(hd_Al[i]); cudaFree(hd_Bl[i]); cudaFree(hd_Cl[i]);    
  }
  //free the pointer arrays
  free(Al); free(Bl); free(Cl);
  cudaFree(d_Al); cudaFree(d_Bl); cudaFree(d_Cl);
  free(hd_Al); free(hd_Bl); free(hd_Cl); 
  return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
  int hA=128, wA=128, wB=128, num=100, blockSize=128, nIter=30;
  bool highCompute=1, checkCorrectness = 0;
  
  for(int i=1;i<argc;i++){
    if(string(argv[i]) == "-hA") hA = atoi(argv[i+1]);
    if(string(argv[i]) == "-wA") wA = atoi(argv[i+1]);
    if(string(argv[i]) == "-wB") wB = atoi(argv[i+1]);
    if(string(argv[i]) == "-numOfTasks") num = atoi(argv[i+1]);
    if(string(argv[i]) == "-highCompute") highCompute = atoi(argv[i+1]);
    if(string(argv[i]) == "-blockSize") blockSize = atoi(argv[i+1]);  
    if(string(argv[i]) == "-nIter") nIter = atoi(argv[i+1]);
    if(string(argv[i]) == "-checkCorrectness") checkCorrectness = atoi(argv[i+1]);
  }

  int succ = matMulGPU(hA,wA,wB,num, highCompute, blockSize, nIter, checkCorrectness);
  exit(succ);
}


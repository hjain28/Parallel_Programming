#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this lab
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x ; int by = blockIdx.x; 
	int tx = threadIdx.x ; int ty = threadIdx.y;

        int Row = by*TILE_WIDTH + ty;
	int Col = bx*TILE_WIDTH + tx;
	
	float Pvalue = 0;
	for (int i =0; i< (numAColumns-1)/TILE_WIDTH+1; ++i)
	{
	   if (Row < numAColumns && i*TILE_WIDTH+tx  < numAColumns){	  //Boundary Conditions 
		Mds[ty][tx] = A[Row*numAColumns  + i*TILE_WIDTH + tx];
                }
	   else {
		Mds[ty][tx] = 0.0;
                }
	   if (i*TILE_WIDTH+ty< numBColumns && Col <numBColumns) {
		Nds[ty][tx] = B[(i*TILE_WIDTH + ty)*numBColumns + Col];
		}
	   else {
		Nds[ty][tx] =0.0;
		}
		__syncthreads();
	
	   if(Row < numAColumns && Col < numBColumns) {
		for (int j =0; j<TILE_WIDTH; ++j)
		{
			Pvalue += Mds[ty][j] + Nds[j][tx];
		}
		__syncthreads();
	     }
	}
	
	if  (Row<  numAColumns && Col < numBColumns)	
	    C[Row*numBColumns + Col] =  Pvalue;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA; // A matrix on device
  float *deviceB; // B matrix on device
  float *deviceC; // C matrix on device
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C(you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
                            
  //@@ Set numCRows and numCColumns
  numCRows    = numARows ;   // set to correct value
  numCColumns = numBColumns;    // set to correct value
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows*numCColumns*sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
   cudaMalloc((void**)&deviceA, numARows*numAColumns*sizeof(float));
  cudaMalloc((void**)&deviceB, numBRows*numBColumns*sizeof(float));
  cudaMalloc((void**)&deviceC, numCRows*numCColumns*sizeof(float));
  
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
   cudaMemcpy(deviceA,hostA, numARows*numAColumns*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB, numBRows*numBColumns*sizeof(float),cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // note that TILE_WIDTH is set to 16 on line number 13. 
  
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
   cudaMemcpy(hostC,deviceC, numCRows*numCColumns*sizeof(float),cudaMemcpyHostToDevice);
 
  wbTime_stop(Copy, "Copying output memory to the CPU");
  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(&deviceA);
  cudaFree(&deviceB);
  cudaFree(&deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
using namespace std;


__global__ void histogram(int *d_bins, const int *d_in, const int BIN_COUNT)
{

    int gridDim_X = 64 ;
    int gridDim_Y = 64 ;
    int myId = threadIdx.x + (blockDim.x * blockIdx.x) + (gridDim_X * blockDim.x *blockIdx.y) + (gridDim_Y* gridDim_X* blockDim.x * blockIdx.z) ;
    int myItem = d_in[myId];
    int myBin = myItem % BIN_COUNT;
    atomicAdd(&(d_bins[myBin]), 1);
}


int main(int argc, char **argv)
{
    const int ARRAY_SIZE = 4096*4096;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
    const int BIN_COUNT = 10;
    const int BIN_BYTES = BIN_COUNT * sizeof(int);
    cout << "SEE Output.txt File for OUTPUT __________  HIMANSHU JAIN \n" ;

    // reading input file and store into a matrix.
    int  h_in[4096*4096];
    ifstream inputfile;
    inputfile.open(argv[1]);
    int i =0;
    while(inputfile)
    { inputfile >>  h_in[i];
      i++;
    }

    
    // initialize Bin with zero
    int h_bins[BIN_COUNT]; 
    for(int i = 0; i < BIN_COUNT; i++)
    {   h_bins[i] = 0;
    }
    
    // declare GPU memory pointers
    int * d_in;
    int * d_bins;
 // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_bins, BIN_BYTES);

    // transfer the arrays to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice);


    // launch the kernel
        int Block_x = 64;
        int Block_y = 64;
        int Block_z = 64 ;
        int thread_Block = 64 ;
        histogram<<< dim3(Block_x,Block_y,Block_z),thread_Block>>>(d_bins, d_in, BIN_COUNT);

    // copy back the sum from GPU
    cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    // writing to output_file
    ofstream outputfile;
    outputfile.open("output.txt");
    outputfile << "Homework 5 : CUDA Assignment (HISTOGRAM) : Himanhu Jain \n";
    for(int i = 0; i < BIN_COUNT; i++)
    {   outputfile << i << "   =>  " << h_bins[i] << "\n";
    }
    outputfile.close();

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_bins);

    return 0;
}
                                                                                                                                                 83,1          Bot


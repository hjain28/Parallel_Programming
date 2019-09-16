#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main (int argc, char*argv[])
{	int size = 512;
	int mA[size][size], mB[size][size], mC[size][size];	// 2D Matrix A,B,C
	int rank, p;						// rank - rank of Processor (0 for Mater), p - total no. of processor  
	int RPP , i,j,k, lp; 	
	double stime, etime, stime1, etime1, stime2, etime2;					// variables for timing calculations

	MPI_Init(&argc, &argv);					// MPI Commands  : Intializing,ranking processors,size of node(total no. of processors), status		
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p); 
	MPI_Status status;

	RPP = size/p;						// RPP = Rows per processor
	
  if (rank == 0)						//** Master Node : Intializing and printing A,B matrix, Sending MAtrix A to processors **//  
    {	printf("%d,%d,\n", rank , p );
		stime = MPI_Wtime();
		for(int i = 0 ; i< size; i++)					// Initialize Matrix A & B
		{	
			for(int j =0; j< size; j++)
			{
				mA[i][j] = 1;
				mB[i][j] = i; 
			}
		}
		
		for (i =1 ; i<p; i++) 						// Splitting and Sending MAtrix A 	
		{	lp = i*RPP;
			MPI_Send(&mA[lp][0], size*RPP, MPI_INT, i, 0, MPI_COMM_WORLD);
     	} 
	
		stime2 = MPI_Wtime();
		for ( i=0; i< (RPP); i++)							// Multiplying operation in master procon Matrix A and B and store to Matrix C
		{	for (j=0; j<size; j++)
			{	mC[i][j] = 0;
				for (k=0; k<size; k++)
				{	mC[i][j] = mC[i][j] + mA[i][k]*mB[k][j];
				}
			}
		}
		etime2 = MPI_Wtime();	
    }

  MPI_Bcast(mB, size*size, MPI_INT, 0, MPI_COMM_WORLD);		//** Broadcasting Matrix  B to all processor **//

  if  (rank>0) 							//** Slave Matrix :  Receiving , Multiplication and Sending Result to Master  **//
    {	
		int x;
		MPI_Recv(&mA[x][0], size*RPP, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);		// Reciveing Splitted MAtrix A

		stime1 = MPI_Wtime();
		for ( i=x; i< (RPP+x); i++)							// Multiplying operation on Matrix A and B and store to Matrix C
		{	for (j=0; j<size; j++)
			{	mC[i][j] = 0;
				for (k=0; k<size; k++)
				{	mC[i][j] = mC[i][j] + mA[i][k]*mB[k][j];
				}
			}
		}
		etime1 = MPI_Wtime();	
	
		MPI_Send(&mC[x][0], RPP*size, MPI_INT, 0, 0, MPI_COMM_WORLD);			// Sending Resultant Matrix C to Master 
    }

	if (rank == 0)	   					        //**  Master processor : Reciving Matrix C and Printing Resultant Matrix C.		   
    {	for (i =1 ; i<p; i++) 								// Receiving MAtrix C
		{	MPI_Recv(&mC[(i*RPP)][0], size*RPP, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      	}
		
		etime = MPI_Wtime();	
		
		printf("\n INPUT Matrix A \n");
		for(int i = 0 ; i< size; i++)							// Printing Matrix a	
		{	for(int j =0; j< size; j++)
			{	printf("%d, ",mA[i][j]);
			}
			printf("\n");	
		}
		printf("\n INPUT Matrix B \n");
		for(int i = 0 ; i< size; i++)							// Printing Matrix B	
		{	for(int j =0; j< size; j++)
			{	printf("%d, ",mB[i][j]);
			}
			printf("\n");	
		}
		printf("\n Final Resultant Matrix C \n");
		for(int i = 0 ; i< size; i++)							// Printing Matrix C	
		{	for(int j =0; j< size; j++)
			{	printf("%d, ",mC[i][j]);
			}
			printf("\n");	
		}
		
		printf ( "\nTotal Actual Execution time : %f" , etime-stime);
        double ct;	
		if ((etime1-stime1)>(etime2-stime2))
		{  ct = (etime1 - stime1);}
		else 
		{  ct = (etime2 -stime2);}
		
		printf ( "\nTotal Actual Computation  time1 : %f" , ct);
		printf ( "\nTotal Actual Communication  time : %f" , (etime-stime) - ct);
	}

   MPI_Finalize();						// Closing MPI commands
   return 0;
}

		



















	 

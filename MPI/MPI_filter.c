#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

int main (int argc, char*argv[])
{	int size = 256;
	int mA[size][size], mC[size-2][size-2];
	int rank, p;
	int RPP;
	int i,j;
	double sTime, eTime;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p); 
	MPI_Status status;

	RPP = size/p;

	/* Master Processor */
    if (rank == 0) 
	{	for(int i = 0 ; i< size; i++)
	        {	for(int j =0; j< size; j++)
				{	mA[i][j] = i;
				}
            }	
		printf("RANK : %d,Total No.of PROCESSOR : %d\n", rank , p );
		for (i =1 ; i<p; i++) 
		{	if ( i == (p-1))     		
				MPI_Send(&mA[(i*RPP)-1], size*(RPP+1), MPI_INT, i, 0, MPI_COMM_WORLD);
			else
				MPI_Send(&mA[(i*RPP)-1], size*(RPP+2), MPI_INT, i, 0, MPI_COMM_WORLD);
      	}
		for ( i=0; i<(RPP+1); i++)
		{	for (j=0; j<size; j++)
			{	if  (i==0 || i==RPP || j==0 || j==(size-1))
				    continue;
				else
				{	int sq_sum;
					for (int m = -1; m < 2; m++ ) 
					{	for (int n = -1; n <2; n++) 
						sq_sum = sq_sum + (mA[i+m][j+n])*(mA[i+m][j+n]) ;	 
					}
					mC[i-1][j-1] = sqrt(sq_sum);
				}
			}
		}
		
   	}
	/*slave Processors */
    else
	{	if ( rank == (p-1))
        {	int x;
			MPI_Recv(&mA[x][0], size*(RPP+1), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			for ( i=x; i<(x+RPP+1); i++)
			{	for (j=0; j<size; j++)
				{	if  (i==x || i== (x+RPP) || j==0 || j==(size-1))
				    continue;
					else
					{	int sq_sum;
						for (int m = -1; m < 2; m++ ) 
						{	for (int n = -1; n <2; n++) 
							sq_sum = sq_sum + (mA[i+m][j+n])*(mA[i+m][j+n]) ;	 
						}
						mC[i-1][j-1] = sqrt(sq_sum);
					}
				}
			}
			MPI_Send(&mC[x-1][0],(RPP-1)*size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	    }	
		else
	    {	int x;
			MPI_Recv(&mA[x][0], size*(RPP+2), MPI_INT, 0, 0, MPI_COMM_WORLD,&status);
			for ( i=x; i<(x+RPP +2); i++)
			{	for (j=0; j<size; j++)
				{	if  (i==x || i==(x+RPP+1) || j==0 || j==(size-1))
				    continue;
					else
					{	int sq_sum;
						for (int m = -1; m < 2; m++ ) 
						{	for (int n = -1; n <2; n++) 
							sq_sum = sq_sum + (mA[i+m][j+n])*(mA[i+m][j+n]) ;	 
						}
						mC[i-1][j-1] = sqrt(sq_sum);
					}
				}
			}
			MPI_Send(&mC[x-1][0], RPP*size, MPI_INT, 0, 0, MPI_COMM_WORLD);
	    }	
	}	
	
	if (rank == 0)	   					        //**  Master processor : Reciving Matrix C and Printing Resultant Matrix C.		   
    {	for (i =1 ; i<p; i++) 
		{	if ( i == (p-1))     		
				MPI_Recv(&mC[(i*RPP)-1][0], size*(RPP-1), MPI_INT, i, 0, MPI_COMM_WORLD,&status);
			else
				 MPI_Recv(&mC(i*RPP)-1][0], size*RPP, MPI_INT, i, 0, MPI_COMM_WORLD,&status);
      	}
		
		eTime = MPI_Wtime();	
		
		printf("\n Final Resultant Matrix A \n");
		for(int i = 0 ; i< size-2; i++)							// Printing INPUT Matrix A	
		{	for(int j =0; j< size-2; j++)
			{	printf("%d, ",mA[i][j]);
			}
			printf("\n");	
		}
		printf("\n Final Resultant Matrix C \n");
		for(int i = 0 ; i< size-2; i++)							// Printing FILTERD Matrix C	
		{	for(int j =0; j< size-2; j++)
			{	printf("%d, ",mC[i][j]);
			}
			printf("\n");	
		}
		
		printf( " total execution time : %f ", eTime -sTime);
    }

   	MPI_Finalize();						// Closing MPI commands
   	return 0;
}

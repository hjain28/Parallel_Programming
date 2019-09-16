 ----Readme-------

The folder himanshujain_rzhao_AES folder contains following files hierarchy:
		      Group10_AES
				|_ test&Key	  
						|_  key.txt, test.txt, tset1.txt, test2.txt, test3.txt, test4.txt
				|_ labs
						|_ code
								|_  AES_Encrypt.cpp
								|_  AES_global.cu   		
								|_  AES_shared.cu			// key & S-box in the global memory 
								|_  AES_sharedS.cu			// key in the shared memory 
								|_  AES_sharedC.cu			// key in the constant memory
								|_  AES_global_streaming.cu		// for global streaming version
								|_  AES_shared_streaming.cu             // for shared_streaming version				
						
						|_  CMakeLists.txt
						|_  LICENSE.txt
						|_  run_AES.pbs
						|_  Module
								|_ source.cmake
								|_ AES
									 |_  AES_global.cu
									 |_  AES_shared.cu
									 |_  AES_sharedS.cu
									 |_  AES_sharedC.cu
									 |_  AES_global_streaming.cu
									 |_  AES_shared_streaming.cu
									 |_  sources.cmake
 
 				|_ Readme.txt
 
 
######################################################################################################################################################
 
How to Run for HPC user and pc user : 
 
For HPC USER follow the steps :

1. mkdir build_dir
2. cd build_dir      // Now you are inside the build_dir directory
3. CC=gcc Cxx=g++    //set flags 
4. cmake3 ../labs
5. make
6. cp ../labs/run_AES.pbs .
7. cp ../test&key/* .            // To copy key file and test file. 
8. qsub run_AES.pbs		// will generate output file encrypt.txt 

for running serial version : (still in the build_dir)

9. cp ../labs/AES_Encrypt.cpp .
10. gcc -o aes AES_Encrypt.cpp
11. ./aes 		     //will generate output file encrypt_serial.txt
12. mv encrypt_serial.txt  gold.txt


for verification 

13. diff encrypt.txt gold.txt 


-***************************************************************************-

for pc user :

1. Discard step 3,4,5,6 and 8
2. cp ../labs/code/*  .
3. follow steps 7,9,10,11,12  to generate gold file.
4. nvcc  AES_global.cu
5. ./a.out                             // will generate encrypt.txt
6. diff encrypt.txt gold.txt          //(for verification)      



_***************************************************************************_

for profiling :

1. nvvp 
2. set executable(eg. ./a.out) and run.


****************************************************************************
		Important Information :

#####To run the shared version and other version. Please follow the instructions below.

For the HPC user:

1. follow the instruction in the comment part of labs/Module/AES/sources.cmake

For the PC user :

1. change in step4
	for the shared version : nvcc AES_shared.cu
	for the key in shared version : nvcc AES_sharedS.cu
	for the key in constant version : nvcc AES_sharedC.cu


###To run the different test set
1. open the code files in labs/Module/AES/*   
2. change the test file name


for the PC user :

1. change the test file name into codefile.
 

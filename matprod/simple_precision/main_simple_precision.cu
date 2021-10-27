/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include "support.h"
#include "kernel_shared_mem.cu"
#include "kernel_unshared_mem.cu"


#define HILOS 64


int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }

    int ld = n;
    float* A_h = (float*) malloc( sizeof(float)*n*n );
    for (unsigned int i=0; i < n; i++)
    {
        for (unsigned int j=0; j < n; j++)
        {
            A_h[i*ld + j] = (rand()%100)/100.00;
        }
    }

    float* B_h = (float*) malloc( sizeof(float)*n*n );
    for (unsigned int i=0; i < n; i++)
    {
        for (unsigned int j=0; j < n; j++)
        {
            B_h[i*ld + j] = (rand()%100)/100.00;
        }
    }

    float* C_h = (float*) malloc( sizeof(float)*n*n );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Matrix size = %u * %u\n", n,n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    float *A_d, *B_d, *C_d;
    cudaMalloc( (void **) &A_d, sizeof(float)*n*n);
    cudaMalloc( (void **) &B_d, sizeof(float)*n*n);
    cudaMalloc( (void **) &C_d, sizeof(float)*n*n);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    cudaMemcpy(A_d, A_h, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float)*n*n, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    dim3 num_threads(HILOS, HILOS);
    dim3 num_blocks(ceil( float(n) / num_threads.x),ceil( float(n) / num_threads.y));

    vecAddKernelShared<<<num_blocks, num_threads>>>(A_d, B_d, C_d, n);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE

    cudaMemcpy(C_h, C_d, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    //verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;

}


/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__global__ void vecAddKernelUnshared(float* A, float* B, float* C, int n) {

    /* Multiplicaciçón de matrices sin memoria compartida */

    // Índices de la matriz
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    float Csub = 0;

    /* loop que itera fila * columna de A y B respectivamente */
    for (int k = 0; k < n; k++){
        Csub += A[i*n + k] * B[k*n + j]; 
    }

    C[i*n+j] = Csub;
}


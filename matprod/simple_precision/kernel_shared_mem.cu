/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__global__ void vecAddKernelShared(float* A, float* B, float* C, int n) {

    /* Definir identificadores */
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    __constant__ int BS = blockDim.x; // blockDim.x == blockDim.y

    /* Primer índice de las matrices de cada block (submatrices) */

    int aBegin = n * BS * by; // Num_filas * BS desplaza el puntero  una
    // fila entera de submatrices (desplazamiento vertical)
    int bBegin = BS * bx; // Desplazamiento horizontal entre submatrices

    float Csub = 0; // Elemento que calcula cada hilo individualmente


    /* Bucle para recorrer las submatrices de A de manera horizontal
       y las submatrices de B de manera vertical. En cada iteración
       se desplaza a la siguiente submatriz. De este modo se puede almacenar
       memoria de cada submatriz, en su bloque correspondiente, de forma
       simultánea */
    for (int a=aBegin, b=bBegin; a<=aBegin + n - 1; a+=BS, b+=BS*n)
    {
        __shared__ float Asub[BS][BS];
        __shared__ float Bsub[BS][BS];
        // As[ty][tx] posición local de cada hilo dentro del bloque
        // las filas son las coordenadas "y" y las columnas las "x"
        Asub[ty][tx] = A[a + n * ty + tx];//n*ty+tx == i*ld + j
        Bsub[ty][tx] = B[b + n * ty + tx];

        __syncthreads(); // Sincronización para asegurar carga completa

        /* Ahora dentro de este bucle que recorre las submatrices, se
            tiene que iterar otro bucle anidado para recorrer la fila
            y columna correspondiente de cada submatriz, para que cada hilo
            realice el producto matricial que le corresponde */
        for (int k = 0; k < BS; k++)
        {
            Csub += Asub[ty][k] * Bsub[k][tx];
        }
        __syncthreads(); // Sincronización para que no se altere la memoria
        // compartida de la siguiente iteración
    }

    // c desplaza el puntero a la primera posición de la submatriz correspondiente
    int c = n * BS * by + BS * bx;//Primer termino desplazamiento vertical, 
    // Segundo término desplazamiento horizontal


    C[c + n * ty + tx] = Csub;
}
/****************************************************************************
* Nome:                  Davide
* Cognome:               Conti
* Numero di matricola:   0000806467
****************************************************************************/

/****************************************************************************
 *
 * earthquake.c - Simple 2D earthquake model
 *
 * Copyright (C) 2018 Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last updated on 2018-12-29
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * ---------------------------------------------------------------------------
 *
 * Versione di riferimento del progetto di High Performance Computing
 * 2018/2019, corso di laurea in Ingegneria e Scienze Informatiche,
 * Universita' di Bologna. Per una descrizione del modello si vedano
 * le specifiche sulla pagina del corso:
 *
 * http://moreno.marzolla.name/teaching/HPC/
 *
 * Per compilare:
 *
 * gcc -D_XOPEN_SOURCE=600 -std=c99 -Wall -Wpedantic earthquake.c -o earthquake
 *
 * (il flag -D_XOPEN_SOURCE=600 e' superfluo perche' viene settato
 * nell'header "hpc.h", ma definirlo tramite la riga di comando fa si'
 * che il programma compili correttamente anche se inavvertitamente
 * non si include "hpc.h", o per errore non lo si include come primo
 * file come necessario).
 *
 * Per eseguire il programma si puo' usare la riga di comando seguente:
 *
 * ./earthquake 100000 256 > out
 *
 * Il primo parametro indica il numero di timestep, e il secondo la
 * dimensione (lato) del dominio. L'output consiste in coppie di
 * valori numerici (100000 in questo caso) il cui significato e'
 * spiegato nella specifica del progetto.
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>     /* rand() */
#include <assert.h>

/* energia massima */
#define EMAX 4.0f
/* energia da aggiungere ad ogni timestep */
#define EDELTA 1e-4
#define BLKDIM 32
#define BLKSIZE 1024

/**
 * Restituisce un puntatore all'elemento di coordinate (i,j) del
 * dominio grid con n colonne.
 */
__device__ __host__ static inline float *IDX(float *grid, int i, int j, int n)
{
    return (grid + i*n + j);
}

/**
 * Restituisce un numero reale pseudocasuale con probabilita' uniforme
 * nell'intervallo [a, b], con a < b.
 */
float randab( float a, float b )
{
    return a + (b-a)*(rand() / (float)RAND_MAX);
}

/**
 * Inizializza il dominio grid di dimensioni n*n con valori di energia
 * scelti con probabilità uniforme nell'intervallo [fmin, fmax], con
 * fmin < fmax.
 *
 * NON PARALLELIZZARE QUESTA FUNZIONE: rand() non e' thread-safe,
 * qundi non va usata in blocchi paralleli OpenMP; inoltre la funzione
 * non si "comporta bene" con MPI (i dettagli non sono importanti, ma
 * posso spiegarli a chi e' interessato). Di conseguenza, questa
 * funzione va eseguita dalla CPU, e solo dal master (se si usa MPI).
 */
void setup( float* grid, int n, float fmin, float fmax )
{
    for (int i = 0; i < n; i++ ) {
        for (int j = 0; j < n; j++ ) {
            *IDX(grid, i, j, n) = randab(fmin, fmax);
        }
    }
}

/**
 * Somma delta a tutte le celle del dominio grid di dimensioni
 * n*n. Questa funzione realizza il passo 1 descritto nella specifica
 * del progetto.
 */
__global__ void increment_energy( float *grid, int n, float delta )
{

    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        *IDX(grid, i, j, n) += delta;
    }
}

/**
 * Restituisce il numero di celle la cui energia e' strettamente
 * maggiore di EMAX.
 */
__global__ void count_cells(float *grid, int n, int *c)
{
    __shared__ int partial_sums[BLKSIZE];
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int lindex = blockDim.x * threadIdx.y + threadIdx.x;
    const int gindex = i * n + j;
    const int numberOfElements = n * n;
    int bsize = (blockDim.x * blockDim.y) / 2;
    if (gindex < numberOfElements) {
        partial_sums[lindex] = (*IDX(grid, i, j, n)) > EMAX ? 1 : 0;
    } else {
        partial_sums[lindex] = 0;
    }
    __syncthreads();
    while (bsize > 0) {
        if (lindex < bsize) {
            partial_sums[lindex] += partial_sums[lindex + bsize];
        }
        bsize = bsize / 2;
        __syncthreads();
    }
    if (lindex == 0) {
        atomicAdd(c, partial_sums[0]);
    }
}

/**
 * Distribuisce l'energia di ogni cella a quelle adiacenti (se
 * presenti). cur denota il dominio corrente, next denota il dominio
 * che conterra' il nuovo valore delle energie. Questa funzione
 * realizza il passo 2 descritto nella specifica del progetto.
 */
__global__ void propagate_energy(float *cur, float *next, int n)
 {
     const float FDELTA = EMAX/4;
     const int i = blockIdx.y * blockDim.y + threadIdx.y;
     const int j = blockIdx.x * blockDim.x + threadIdx.x;

     if (i < n && j < n) {
         float F = *IDX(cur, i, j, n);

         /* Se l'energia del vicino di sinistra (se esiste) e'
            maggiore di EMAX, allora la cella (i,j) ricevera'
            energia addizionale FDELTA = EMAX/4 */
         if ((j > 0) && (*IDX(cur, i, j - 1, n) > EMAX)) { F += FDELTA; }
         /* Idem per il vicino di destra */
         if ((j<n-1) && (*IDX(cur, i, j + 1, n) > EMAX)) { F += FDELTA; }
         /* Idem per il vicino in alto */
         if ((i>0) && (*IDX(cur, i - 1, j, n) > EMAX)) { F += FDELTA; }
         /* Idem per il vicino in basso */
         if ((i<n-1) && (*IDX(cur, i + 1, j, n) > EMAX)) { F += FDELTA; }

         /* Si noti che il valore di F potrebbe essere ancora
            maggiore di EMAX; questo non e' un problema:
            l'eventuale eccesso verra' rilasciato al termine delle
            successive iterazioni fino a riportare il valore
            dell'energia sotto la foglia EMAX. */
         if (F > EMAX) {
             F -= EMAX;
         }

         *IDX(next, i, j, n) = F;
     }
}

/**
 * Restituisce l'energia media delle celle del dominio grid di
 * dimensioni n*n. Il dominio non viene modificato.
 */
__global__ void average_energy(float *grid, int n, float *sum)
 {
     __shared__ float partial_sums[BLKSIZE];
     const int i = blockIdx.y * blockDim.y + threadIdx.y;
     const int j = blockIdx.x * blockDim.x + threadIdx.x;
     const int lindex = blockDim.x * threadIdx.y + threadIdx.x;
     const int gindex = i * n + j;
     const int numberOfElements = n * n;
     int bsize = (blockDim.x * blockDim.y) / 2;
     if (gindex < numberOfElements) {
         partial_sums[lindex] = *IDX(grid, i, j, n);
     } else {
         partial_sums[lindex] = 0;
     }
     __syncthreads();
     while (bsize > 0) {
         if (lindex < bsize) {
             partial_sums[lindex] += partial_sums[lindex + bsize];
         }
         bsize = bsize / 2;
         __syncthreads();
     }
     if (lindex == 0) {
         atomicAdd(sum, partial_sums[0]);
     }
}

int main( int argc, char* argv[] )
{
    const int zero = 0;
    const float zero1 = 0.0f;

    float *cur, *next;
    float *d_cur, *d_next, *d_sum;
    int *d_c;
    int s, n = 256, nsteps = 2048;
    float Emean;
    int c;

    srand(19); /* Inizializzazione del generatore pseudocasuale */

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [nsteps [n]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        nsteps = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        n = atoi(argv[2]);
    }

    const size_t size = n * n * sizeof(float);
    /* Allochiamo i domini */
    cur = (float*)malloc(size); assert(cur);
    next = (float*)malloc(size); assert(next);

    /* L'energia iniziale di ciascuna cella e' scelta
       con probabilita' uniforme nell'intervallo [0, EMAX*0.1] */
    setup(cur, n, 0, EMAX*0.1);

    dim3 block(BLKDIM, BLKDIM);
    dim3 grid((n + BLKDIM - 1) / BLKDIM, ( n + BLKDIM - 1) / BLKDIM);

    cudaMalloc((void**)&d_cur, size);
    cudaMalloc((void**)&d_next, size);
    cudaMalloc((void**)&d_c, sizeof(*d_c));
    cudaMalloc((void**)&d_sum, sizeof(*d_sum));

    cudaMemcpy(d_cur, cur, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_next, next, size, cudaMemcpyHostToDevice);

    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        /* L'ordine delle istruzioni che seguono e' importante */
        increment_energy<<<grid, block>>>(d_cur, n, EDELTA);
        cudaMemcpy(d_c, &zero, sizeof(zero), cudaMemcpyHostToDevice);
        count_cells<<<grid, block>>>(d_cur, n, d_c);
        cudaMemcpy(&c, d_c, sizeof(c), cudaMemcpyDeviceToHost);
        propagate_energy<<<grid, block>>>(d_cur, d_next, n);
        cudaMemcpy(d_sum, &zero1, sizeof(zero1), cudaMemcpyHostToDevice);
        average_energy<<<grid, block>>>(d_next, n, d_sum);
        cudaMemcpy(&Emean, d_sum, sizeof(Emean), cudaMemcpyDeviceToHost);
        Emean /= (n*n);

        printf("%d %f\n", c, Emean);

        float *tmp = d_cur;
        d_cur = d_next;
        d_next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;

    double Mupdates = (((double)n)*n/1.0e6)*nsteps; /* milioni di celle aggiornate per ogni secondo di wall clock time */
    fprintf(stderr, "%s : %.4f Mupdates in %.4f seconds (%f Mupd/sec)\n", argv[0], Mupdates, elapsed, Mupdates/elapsed);

    /* Libero la memoria */
    free(cur);
    free(next);
    cudaFree(d_cur);
    cudaFree(d_next);
    cudaFree(d_c);
    cudaFree(d_sum);
    return EXIT_SUCCESS;
}

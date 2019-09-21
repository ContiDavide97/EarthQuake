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

/**
 * Restituisce un puntatore all'elemento di coordinate (i,j) del
 * dominio grid con n colonne.
 */
static inline float *IDX(float *grid, int i, int j, int n)
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
    const int widthLine = n + 2;
    const int last = n + 1;
    for (int i = 1; i <= n; i++ ) {
        /*Inizializzo a 0 ogni prima cella di ogni riga*/
        *IDX(grid, i, 0, widthLine) = 0;
        for ( int j = 1; j <= n; j++ ) {
            *IDX(grid, i, j, widthLine) = randab(fmin, fmax);
        }
        /*Inizializzo a 0 ogni ultima cella di ogni riga*/
        *IDX(grid, i, last, widthLine) = 0;
    }
    /*Inizializzo a 0 tutte le celle della prima riga e tutte le celle dell'ultima riga*/
    for (int j = 0; j < widthLine; j++) {
        *IDX(grid, 0, j, widthLine) = 0;
        *IDX(grid, last, j, widthLine) = 0;
    }
}


/**
 * Somma delta a tutte le celle del dominio grid di dimensioni
 * n*n.
 * Restituisce il numero di celle la cui energia e' strettamente
 * maggiore di EMAX.
 * Questa funzione realizza il passo 1 descritto nella specifica
 * del progetto.
 */
int increment_energy_and_count_cells( float *grid, int n, float delta)
{
    int c = 0;
    float *cell;
    const int widthLine = n + 2;
#pragma omp parallel for reduction(+:c) default(none) private(cell) shared(n, grid, delta)
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            cell = IDX(grid, i, j, widthLine);
            *cell += delta;
            if ( *cell > EMAX ) { c++; }
        }
    }
    return c;
}

/**
 * Distribuisce l'energia di ogni cella a quelle adiacenti (se
 * presenti). cur denota il dominio corrente, next denota il dominio
 * che conterra' il nuovo valore delle energie.
 * Restituisce l'energia media delle celle del dominio grid di
 * dimensioni n*n. Il dominio non viene modificato.
 * Questa funzione realizza il passo 2 descritto nella specifica del
 * progetto.
 */
float propagate_energy_and_average_energy(float *cur, float *next, int n)
{
    const float FDELTA = EMAX/4;
    const int widthLine = n + 2;
    float sum = 0.0f;
#pragma omp parallel for reduction(+:sum) default(none) shared(cur, next, n)
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            float F = *IDX(cur, i, j, widthLine);

            /* Se l'energia del vicino di sinistra (se esiste) e'
               maggiore di EMAX, allora la cella (i,j) ricevera'
               energia addizionale FDELTA = EMAX/4 */
            if ((*IDX(cur, i, j-1, widthLine)) > EMAX) { F += FDELTA; }
            /* Idem per il vicino di destra */
            if ((*IDX(cur, i, j+1, widthLine)) > EMAX) { F += FDELTA; }
            /* Idem per il vicino in alto */
            if ((*IDX(cur, i-1, j, widthLine)) > EMAX) { F += FDELTA; }
            /* Idem per il vicino in basso */
            if ((*IDX(cur, i+1, j, widthLine) > EMAX)) { F += FDELTA; }

            if (F > EMAX) {
                F -= EMAX;
            }
            sum += F;
            /* Si noti che il valore di F potrebbe essere ancora
               maggiore di EMAX; questo non e' un problema:
               l'eventuale eccesso verra' rilasciato al termine delle
               successive iterazioni fino a riportare il valore
               dell'energia sotto la soglia EMAX. */
            *IDX(next, i, j, widthLine) = F;
        }
    }
    return (sum / (n*n));
}

int main( int argc, char* argv[] )
{
    float *cur, *next;
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

    const size_t sizeWithGosthCells = (n + 2) * (n + 2) * sizeof(float);
    /* Allochiamo i domini */
    cur = (float*)malloc(sizeWithGosthCells); assert(cur);
    next = (float*)malloc(sizeWithGosthCells); assert(next);

    /* L'energia iniziale di ciascuna cella e' scelta
       con probabilita' uniforme nell'intervallo [0, EMAX*0.1] */
    setup(cur, n, 0, EMAX*0.1);

    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        /* L'ordine delle istruzioni che seguono e' importante */
        c = increment_energy_and_count_cells(cur, n, EDELTA);
        Emean = propagate_energy_and_average_energy(cur, next, n);

        printf("%d %f\n", c, Emean);

        float *tmp = cur;
        cur = next;
        next = tmp;
    }
    const double elapsed = hpc_gettime() - tstart;

    double Mupdates = (((double)n)*n/1.0e6)*nsteps; /* milioni di celle aggiornate per ogni secondo di wall clock time */
    fprintf(stderr, "%s : %.4f Mupdates in %.4f seconds (%f Mupd/sec)\n", argv[0], Mupdates, elapsed, Mupdates/elapsed);

    /* Libero la memoria */
    free(cur);
    free(next);

    return EXIT_SUCCESS;
}

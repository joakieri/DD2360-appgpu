#include <stdio.h>
#include <sys/time.h>

#define ERROR_RANGE 1e-9

typedef struct {
    float3 pos;
    float3 vel;
} Particle;

__global__ void oneTimestepGPU(Particle *p) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;  
    p[id].pos.x += p[id].vel.x;
    p[id].pos.y += p[id].vel.y;
    p[id].pos.z += p[id].vel.z;
}

void oneTimestepCPU(Particle *p, int n) {
    for (int i = 0; i < n; i++) {
	    p[i].pos.x += p[i].vel.x;
	    p[i].pos.y += p[i].vel.y;
	    p[i].pos.z += p[i].vel.z;
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

int main(int argc, char *argv[]) {
    int NUM_PARTICLES = atoi(argv[1]);
    int NUM_ITERATIONS = atoi(argv[2]);
    int BLOCK_SIZE = atoi(argv[3]);
    Particle pCPU[NUM_PARTICLES];
    Particle pCPUres[NUM_PARTICLES];
    Particle pGPUres[NUM_PARTICLES];
    Particle *pGPU;
    int nBlocks;
    double iStart, iElapsCPU, iElapsGPU;
    double error;
    int nErrors;


    //printf("Particles, Iterations, Thread block size, CPU time, GPU time, Errors\n");

    nBlocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    srand(time(0));
    for (int i = 0; i < NUM_PARTICLES; i++) {
        pCPU[i].pos.x = rand();
        pCPU[i].pos.y = rand();
        pCPU[i].pos.z = rand();
        pCPU[i].vel.x = rand();
        pCPU[i].vel.y = rand();
        pCPU[i].vel.z = rand();
	pCPUres[i] = pCPU[i];
    }
    
    cudaMalloc(&pGPU, NUM_PARTICLES * sizeof(Particle));

    // Meassure CPU performance
    iStart = cpuSecond();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        oneTimestepCPU(pCPUres, NUM_PARTICLES);
    }
    iElapsCPU = cpuSecond() - iStart;
    
    // Meassure GPU performance
    iStart = cpuSecond();
    cudaMemcpy(pGPU, pCPU, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        oneTimestepGPU<<<nBlocks, BLOCK_SIZE>>>(pGPU);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(pGPUres, pGPU, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
    iElapsGPU = cpuSecond() - iStart;
    cudaFree(pGPU);

    // Check the number of errors
    nErrors = 0;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        error = fabs(pCPUres[i].pos.x - pGPUres[i].pos.x);
        if (error > ERROR_RANGE) {
            nErrors++;
            continue;
        }

        error = fabs(pCPUres[i].pos.y - pGPUres[i].pos.y);
        if (error > ERROR_RANGE) {
            nErrors++;
            continue;
        }

        error = fabs(pCPUres[i].pos.z - pGPUres[i].pos.z);
        if (error > ERROR_RANGE) {
            nErrors++;
            continue;
        }
    }

    printf("%d %d %d %f %f %d\n", NUM_PARTICLES, NUM_ITERATIONS, BLOCK_SIZE, iElapsCPU, iElapsGPU, nErrors);
}

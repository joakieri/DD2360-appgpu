#include <stdio.h>
#include <sys/time.h>

#define BLOCK_SIZE 256
#define ERROR_RANGE 1E-9

typedef struct {
    float3 pos;
    float3 vel;
} Particle;

void oneTimeStep(Particle *p) {
    p.pos.x += p.vel.x;
    p.pos.y += p.vel.y;
    p.pos.z += p.vel.z;
}

__global__ void oneTimestepGPU(Particle *p) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    oneTimeStep(p[id]);
}

void oneTimestepCPU(Particle *p, int n) {
    for (int i = 0; i < n; i++) {
        oneTimeStep(p[i]);
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
    Particle pCPU[NUM_PARTICLES];
    Particle pGPUres[NUM_PARTICLES];
    Particle *pGPU;
    int nBlocks;
    double iStart, iElapsCPU, iElapsGPU;
    double error;
    int nErrors;
    nBlocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    srand(time(0));

    for (int i = 0; i < NUM_PARTICLES; i++) {
        pCPU[i].pos.x = rand();
        pCPU[i].pos.y = rand();
        pCPU[i].pos.z = rand();
        pCPU[i].vel.x = rand();
        pCPU[i].vel.y = rand();
        pCPU[i].vel.z = rand();
    }

    cudaMalloc(&pGPU, NUM_PARTICLES * sizeof(Particle));
    iStart = cpuSecond();

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        oneTimestepCPU(pCPU, NUM_PARTICLES);
    }

    iElapsCPU = cpuSecond() - iStart;
    iStart = cpuSecond();
    cudaMemcpy(pGPU, pCPU, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        oneTimestepGPU<<<nBlocks, BLOCK_SIZE>>>(pGPU);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(pGPUres, pGPU, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
    iElapsGPU = cpuSecond() - iStart;
    cudaFree(pGPU);
    nErrors = 0;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        error = fabs(pCPU[i].pos.x - pGPUres[i].pos.x);

        if (error > ERROR_RANGE) {
            nErrors++;
            continue;
        }

        error = fabs(pCPU[i].pos.y - pGPUres[i].pos.y);

        if (error > ERROR_RANGE) {
            nErrors++;
            continue;
        }

        error = fabs(pCPU[i].pos.z - pGPUres[i].pos.z);

        if (error > ERROR_RANGE) {
            nErrors++;
            continue;
        }
    }

    printf("%d %d %f %f %d\n", NUM_PARTICLES, NUM_ITERATIONS, iElapsCPU, iElapsGPU, nErrors);
}
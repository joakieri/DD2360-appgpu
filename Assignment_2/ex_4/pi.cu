#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include <curand.h>
#include <sys/time.h>

#define BLOCKS 100
#ifndef TPB
#define TPB 256
#endif

#ifndef FLOAT
#define FLOAT double
#endif

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

__global__ void kernel(int *block_counts, int iter, curandState *states) {
	__shared__ int counts[TPB];
	FLOAT x, y, z;
	const int id = threadIdx.x + blockDim.x * blockIdx.x;

	curand_init(id, id, 0, &states[id]);

	counts[threadIdx.x] = 0;
	for (int i = 0; i < iter; i++) {
		x = curand_uniform_double(&states[id]);
		y = curand_uniform_double(&states[id]);
		z = sqrt((x*x) + (y*y));
		if (z <= 1.0) {
			counts[threadIdx.x]++;
		}
	}
	__syncthreads();

	block_counts[blockIdx.x] = 0;
	if (threadIdx.x == 0) {
		for (int i = 0; i < TPB; i++) {
			block_counts[blockIdx.x] += counts[i];
		}
	}
}

int main(int argc, char *argv[]) {
    const int NUM_ITER = atoi(argv[1]);
    const int THREAD_ITER = NUM_ITER / (BLOCKS * TPB);
    const int REST = NUM_ITER % (BLOCKS * TPB);
    int count = 0;
    FLOAT pi;
    double iStart, iElaps;
    int host_bc[BLOCKS];
    int * device_bc;
    curandState * dev_random;

    //printf("%d\n", THREAD_ITER);

    cudaMalloc((void**)&dev_random, BLOCKS*TPB*sizeof(curandState));
    cudaMalloc(&device_bc, BLOCKS * sizeof(int));
    
    iStart = cpuSecond();
    kernel<<<BLOCKS, TPB>>>(device_bc, THREAD_ITER, dev_random);
    cudaDeviceSynchronize();
    cudaMemcpy(host_bc, device_bc, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
    iElaps = cpuSecond() - iStart;

    for (int i = 0; i < BLOCKS; i++) {
        count += host_bc[i];
    }

    pi = ((FLOAT)count / (FLOAT)(NUM_ITER - REST)) * 4.0;
    FLOAT error = M_PI - pi;
    
    printf("pi = %f, error = %f, time: %f s\n", pi, error, iElaps);
    
    return 0;
}


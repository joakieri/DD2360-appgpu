#include <stdio.h>
#include <math.h>
#include <sys/time.h>


#define ARRAY_SIZE 100000
#define THREADS_PER_BLOCK 256
#define ERROR_RANGE 1E-9

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

__global__ void saxpyGPU(float *x, float *y, float a, int n) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= n) return;
    y[id] = y[id] + x[id] * a;

    //printf("GPU: My threadId is %d and my current Saxpy value is: %f\n", id, y[id]);
}

void saxpyCPU(float *x, float *y, float a, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = y[i] + x[i] * a;
        //printf("CPU: My array index is %d and my current Saxpy value is: %f\n", i, y[i]);
    }
}

int main() {
    printf("array size - cpu time - gpu time\n");

    for (int array_size = 1000; array_size <= ARRAY_SIZE; array_size += 5000) {
        float x[array_size];
        float y1[array_size];
        float y2[array_size];
        float *xp;
        float *yp;
        float a = 2;

        int blockSize = (array_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaMalloc(&xp, array_size * sizeof(float));
        cudaMalloc(&yp, array_size * sizeof(float));

        srand(time(0));

        for (int i = 0; i < array_size; i++) {
            x[i] = (float) rand();
            y1[i] = x[i];
        }

        cudaMemcpy(xp, x, array_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(yp, y1, array_size * sizeof(float), cudaMemcpyHostToDevice);

        //printf("Starting SAXPY on the CPU...\n");

        double startTimeCPU = cpuSecond();
        saxpyCPU(x, y1, a, array_size);
        double endTimeCPU = cpuSecond();

        double cpuExecutionTime = endTimeCPU - startTimeCPU;
        //printf("Starting SAXPY on the GPU...\n");

        double startTimeGPU = cpuSecond();
        saxpyGPU<<< blockSize, THREADS_PER_BLOCK >>>(xp, yp, a, array_size);
        cudaDeviceSynchronize();
        double endTimeGPU = cpuSecond();

        double gpuExecutionTime = endTimeGPU - startTimeGPU;

        cudaMemcpy(y2, yp, array_size * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(xp);
        cudaFree(yp);

        int errorRate = 0;

        for (int i = 0; i < array_size; i++) {
            double error = fabs(y1[i] - y2[i]);

            if (error > ERROR_RANGE) {
                fprintf(stderr, "Error for index %d\n", i);
                errorRate++;
            }
        }

        if (errorRate == 0) {
            //printf("Program ran sucessfully!\nCPU execution time: %f\nGPU execution time: %f\n", cpuExecutionTime, gpuExecutionTime);
            printf("%d %f %f\n", array_size, cpuExecutionTime, gpuExecutionTime);
        } else {
            printf("Program failed, %d errors out of %d\n", errorRate, array_size);
        }
    }
}
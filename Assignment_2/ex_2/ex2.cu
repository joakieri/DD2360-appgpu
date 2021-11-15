#include <stdio.h>

#define ARRAY_SIZE 10000
#define THREADS_PER_BLOCK 256

// Define error range

__global__ void saxpyGPU(float *x, float *y, float a) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= ARRAY_SIZE) return;

    y[id] = y[id] + x[id] * a;

    printf("GPU: My threadId is %d and my current Saxpy value is: %f\n", id, y[id]);

} 

void saxpyCPU(float *x, float *y, float a) {
    for (int i = 0; i < ARRAY_SIZE; i++) {
        y[i] = y[i] + x[i] * a;
    }
}

int main() {
    float x[ARRAY_SIZE];
    float y1[ARRAY_SIZE];
    float y2[ARRAY_SIZE];
    float * xp;
    float * yp;
    float a = 9; 
    int blockSize = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaMalloc(&xp, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&yp, ARRAY_SIZE*sizeof(float));

    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = (float)i+1;
        y1[i] = x[i];
        printf("CPU: My array index is %d and my current Saxpy value is: %f\n", i, y1[i]);
    }


    cudaMemcpy(xp, x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(yp, y1, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    saxpyCPU(x, y1, a);

    saxpyGPU<<< blockSize, THREADS_PER_BLOCK >>>(xp, yp, a);

    cudaDeviceSynchronize();

    cudaMemcpy(y2, yp, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // Perform bench mark

    // for the array size

    // e = y_parallel - y_sequential

    // if e > error_range || e < -error_range then

    // print an error message
}
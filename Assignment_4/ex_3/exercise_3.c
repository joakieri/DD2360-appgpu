#include <stdio.h>
#include <sys/time.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 100
#endif
#define ERROR_MARGIN 1e-3

void saxpy(const float a, float *x, float* y) {
	#pragma acc parallel loop copyin(x[0:ARRAY_SIZE]) copy(y[0:ARRAY_SIZE]) 
        for (int i = 0; i < ARRAY_SIZE; i++) {
            y[i] += a * x[i];
        }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

int main() {
    float x[ARRAY_SIZE];
    float y[ARRAY_SIZE];
    float *y_dev = malloc(sizeof(float) * ARRAY_SIZE);
    float a = 9;

    printf("Array size: %d\n", ARRAY_SIZE);

    int i;
    for (i = 0; i < ARRAY_SIZE; i++) {
        x[i] = i+1;
        y[i] = i+1;
	y_dev[i] = i+1;
    }

    // Run CPU kernel
    printf("Computing SAXPY on the CPU...");
    double startTimeCPU = cpuSecond();

    for (i = 0; i < ARRAY_SIZE; i++) {
        y[i] += a * x[i];
    }

    double endTimeCPU = cpuSecond();
    printf("Done!\n");

    // Run GPU kernel
    printf("Computing SAXPY on the GPU...");
    double startTimeGPU = cpuSecond();

    saxpy(a, x, y_dev);

    double endTimeGPU = cpuSecond();
    printf("Done!\n");

    // Check result
    printf("Comparing the output for each implementation...");
    int e = 0;
    for (i = 0; i < ARRAY_SIZE; i++) {
        if(y[i] - y_dev[i] > ERROR_MARGIN || y[i] - y_dev[i] < -ERROR_MARGIN) {
            e = 1;
            break;
        }
    }
    if (e == 0)
        printf("Correct!\n");
    else
        printf("Incorrect!\n");

    printf("CPU time: %f, GPU time: %f\n", endTimeCPU - startTimeCPU, endTimeGPU - startTimeGPU);

    free(y_dev);

    return 0;
}

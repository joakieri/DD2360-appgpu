#include <stdio.h>
#define ARRAY_SIZE 10000
// Define error range

__global__ void saxpyGPU(float * x, float * y, float a) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ARRAY_SIZE) return;
	y[i] = y[i] + x[i] * a;
} 

void saxpyCPU(float * x, float * y, float a) {
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

	cudaMalloc(&xp, ARRAY_SIZE*sizeof(float));
	cudaMalloc(&yp, ARRAY_SIZE*sizeof(float));

	for (int i = 0; i < ARRAY_SIZE; i++) {
		x[i] = (float)i+1;
		y1[i] = x[i];
	}
	
	cudaMemcpy(xp, x, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yp, y1, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

	saxpyCPU(x, y1, a);
	saxpyGPU(xp, yp, a);
	cudaDeviceSynchronize();
	
	cudaMemcpy(y2, yp, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

	// Perform bench mark
	// for the array size
	//	e = y_parallel - y_sequential
	//	if e > error_range || e < -error_range then
	//		print an error message
}

#include <stdio.h>

__global__ void helloKernel() {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello world! My threadId is %d\n", id);
}

int main() {
	helloKernel<<<1, 256>>>();
	cudaDeviceSynchronize();
	return 0;	
}

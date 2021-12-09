__kernel
void helloWorld() {
	int id = get_global_id(0);
	int idg = get_group_id(0);
	printf("Hello world! My threadID is (%d, %d)\n", id, idg);
}

__kernel
void saxpy(__global float* a, __global float* x, __global float *y) {
	int i = get_global_id(0);
	if (i < 100000) 
		y[i] += *a * x[i];
}
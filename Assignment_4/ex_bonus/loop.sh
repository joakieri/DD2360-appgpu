for n in {16,32,64,128,256}
do
	for m in {10000,20000,30000,40000,50000,60000,70000,80000}
	do
		echo $((n)) $((m))
		nvcc -D WG_SIZE=$((n)) -D NUM_PARTICLES=$((m)) exercise_bonus.c -lOpenCL -o ex.out
		./ex.out
	done
done
for n in {100000,1000000,10000000,100000000,1000000000}
do
	./pi.out $((n))
done


for n in {16,32,64,128,256}
do
	nvcc -arch=sm_50 -D TPB=$((n)) pi.cu -o pi.out
	./pi.out $((1000000000))
done

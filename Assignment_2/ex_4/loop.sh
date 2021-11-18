nvcc -arch=sm_50 pi.cu -o pi.out

for n in {100000,1000000,10000000,100000000,1000000000}
do
	./pi.out $((n))
done

echo ""

for n in {16,32,64,128,256}
do
	nvcc -arch=sm_50 -D TPB=$((n)) pi.cu -o pi.out
	for m in {100000,1000000,10000000,100000000,1000000000}
	do
		echo $((n)) $((m))
		./pi.out $((m))
	done
done

echo ""

nvcc -arch=sm_50 -D FLOAT=float pi.cu -o pi.out

for n in {100000,1000000,10000000,100000000,1000000000}
do
	./pi.out $((n))
done

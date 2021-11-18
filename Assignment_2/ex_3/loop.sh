echo "Particles, Iterations, Thread block size, CPU time, GPU time, Errors"

echo ""
echo "--- Part 1, testing CPU ---"
echo ""

for particles in {10,100,1000,10000,100000}
do
	./ex3.out $((particles)) $((500)) $((256))
done

echo ""
echo "--- Part 2, testing GPU ---"
echo ""

for number in {1,2,4,8,16}
do
	for particles in {10,100,1000,10000,100000}
	do
		./ex3.out $((particles)) $((500)) $((16 * number))
	done
done

#!/bin/bash
for (( i=50; i <= 610; i += 20 ))
do
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  echo "doing size $i"
  ./hess.out ${i}
done

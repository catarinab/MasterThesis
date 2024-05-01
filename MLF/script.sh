#!/bin/bash
for (( t=64; t >= 64; t /= 2 ))
do
    for (( i=50; i <= 610; i += 20 ))
    do
        export OMP_NUM_THREADS=$t
        export MKL_NUM_THREADS=$t
        echo ${OMP_NUM_THREADS}
        echo "doing size $i"
        ./mlf-shared.out -k "${i}" >> "newpar/2kkresultados-${t}";
    done
    cat "newpar/2kkresultados-${t}"
done

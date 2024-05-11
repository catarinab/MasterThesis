#!/bin/bash
for (( t=32; t >= 2; t /= 2 ))
do
    export OMP_NUM_THREADS=$t
    export MKL_NUM_THREADS=$t
    echo "${OMP_NUM_THREADS}"
    echo "${MKL_NUM_THREADS}"
    for (( i=610; i <= 610; i += 20 ))
    do
        echo "doing size $i"
        ./mlf-shared.out -k "${i}" >> "newpar/2kkresultados-${t}";
    done
    cat "newpar/2kkresultados-${t}"
done
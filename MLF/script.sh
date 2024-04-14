#!/bin/bash
for (( i=50; i <= 610; i += 20 ))
do
    echo "doing size $i"
    ./test.out "${i}"; cat "krylov/${i}/cpp.txt" >> "krylov/resultados.txt"
done

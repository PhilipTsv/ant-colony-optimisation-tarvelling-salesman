#!/bin/bash
for i in {1..10}
do
   mpirun -np 12 -f ./host_file ./aco1000 ./citylist1000 >> output.txt
done

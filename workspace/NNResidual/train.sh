#!/bin/bash

for hidden in {10..700..50}
do
	for decay in 0.001 0.0001 0.00001
	do
		echo "-------hidden: ${hidden}, decay ${decay}------"
		python3 train.py --batch 256 --decay $decay --lr 1e-4 --Ntype NB --epoch 70 --dor 0.5 --hidden $hidden
	done
done
	

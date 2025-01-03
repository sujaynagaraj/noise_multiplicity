#!/usr/bin/env bash
declare -a datasets=("MNIST")
declare -a variance_types=("model" "noise")
declare -a noise_types=("class_independent")


for i in {0..0}
do
    for j in {0..1}
    do
        for k in {0..0}
        do
            
            dataset=${datasets[$i]}
            variance_type=${variance_types[$j]}
            noise_type=${noise_types[$k]}

            sbatch launch_continuous.sh $dataset $variance_type $noise_type
        
        done

    done
done

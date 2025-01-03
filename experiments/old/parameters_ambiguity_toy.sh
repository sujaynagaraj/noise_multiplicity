#!/usr/bin/env bash
declare -a noise_types=("class_independent" "class_conditional")
declare -a settings=("xor" "vanilla")


for i in {0..1}
do
    for j in {0..1}
    do

        noise_type=${noise_types[$i]}
        setting=${settings[$j]}

        sbatch launch_ambiguity_toy.sh $setting $noise_type 
    done
done

#!/usr/bin/env bash
declare -a noise_types=("class_independent" "class_conditional")


for i in {0..1}
do

    noise_type=${noise_types[$i]}
    setting=${settings[$j]}

    sbatch launch_ambiguity_toy_prob.sh $noise_type

done

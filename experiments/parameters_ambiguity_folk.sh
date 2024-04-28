#!/usr/bin/env bash
declare -a noise_types=("class_independent" "class_conditional")
declare -a uncertainty_types=("forward" "backward")
declare -a model_types=("LR" "NN")



for i in {0..1}
do
    for j in {0..1}
    do
        for k in {0..1}
        do

            noise_type=${noise_types[$i]}
            uncertainty_type=${uncertainty_types[$j]}
            model_type=${model_types[$k]}

            sbatch launch_ambiguity_folk.sh $noise_type $uncertainty_type $model_type
        done
    done

done

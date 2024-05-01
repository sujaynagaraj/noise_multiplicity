#!/usr/bin/env bash
declare -a noise_types=("class_independent" "class_conditional")
declare -a uncertainty_types=("backward_sk" "backward_torch" "forward")
declare -a model_types=("LR" "NN")
declare -a datasets=("cshock_eicu" "cshock_mimic" "support" "saps" "lungcancer")


for i in {0..0}
do
    for j in {0..2}
    do
        for k in {0..1}
        do
            for l in {0..4}
            do

                noise_type=${noise_types[$i]}
                uncertainty_type=${uncertainty_types[$j]}
                model_type=${model_types[$k]}
                dataset=${datasets[$l]}

                sbatch launch_ambiguity_real.sh $noise_type $uncertainty_type $model_type $dataset

            done
        done
    done

done

#!/usr/bin/env bash
declare -a noise_types=("class_independent" "class_conditional" "group")
declare -a model_types=("LR" "NN" "SVM")
declare -a datasets=("cshock_eicu" "cshock_mimic" "support" "saps" "lungcancer")


for i in {0..0}
do
    for j in {0..1}
    do
        for k in {0..4}
        do
            
            noise_type=${noise_types[$i]}
            model_type=${model_types[$j]}
            dataset=${datasets[$k]}

            sbatch launch_abstain.sh $noise_type $model_type $dataset


        done
    done

done

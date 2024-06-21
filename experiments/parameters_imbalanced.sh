#!/usr/bin/env bash
declare -a noise_types=("class_independent" "class_conditional")
declare -a model_types=("LR" "NN")
declare -a datasets=("lungcancer_imbalanced" "saps_imbalanced" "support_imbalanced")
declare -a noise_levels=(0.0 0.05 0.01 0.1 0.2 0.4)

for i in {0..1}
do
    for j in {0..1}
    do
        for k in {0..1}
        do
            for h in {0..5}
            do
                noise_type=${noise_types[$i]}
                model_type=${model_types[$j]}
                dataset=${datasets[$k]}
                noise_level=${noise_levels[$h]}
            
                sbatch launch_regret.sh $noise_type $model_type $dataset $noise_level
            done
        done
    done
done

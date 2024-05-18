#!/usr/bin/env bash
declare -a misspecify_type=("over" "under")
declare -a uncertainty_types=("forward" "backward")
declare -a datasets=("cshock_eicu" "cshock_mimic" "support" "saps" "lungcancer")
declare -a noise_levels=(0.0 0.1 0.2 0.3)

for i in {0..1}
do
    for j in {0..1}
    do
        for k in {0..4}
        do
            for h in {0..3}
            do
                misspecify=${misspecify_type[$i]}
                uncertainty_type=${uncertainty_types[$j]}
                dataset=${datasets[$k]}
                noise_level=${noise_levels[$h]}

                sbatch launch_misspecify.sh $misspecify $uncertainty_type $dataset $noise_level

            done
        done
    done

done

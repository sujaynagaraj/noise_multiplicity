#!/usr/bin/env bash
declare -a noise_types=("class_independent" "class_conditional" "group")
declare -a model_types=("LR" "NN")
declare -a datasets=("cshock_eicu" "cshock_mimic" "support" "saps" "lungcancer")
declare -a noise_levels=(0.0 0.05 0.01 0.1 0.2 0.4)

for i in {0..1}
do
    for j in {0..1}
    do
        for k in {0..4}
        do
            for h in {0..5}
            do
                noise_type=${noise_types[$i]}
                model_type=${model_types[$j]}
                dataset=${datasets[$k]}
                noise_level=${noise_levels[$h]}
                file_direction="/scratch/hdd001/home/snagaraj/results/metrics/$dataset/$model_type/$noise_type"

                epsilon=0.1

                # Assume file naming as forward/backward_0.0_0.1_metrics.pkl
                # This assumes both $noise_level and $epsilon are set appropriately elsewhere in your script
                #file_name=$(printf "%s_%.2g_%.2g_metrics.pkl" "forward" "$noise_level" "$epsilon")
                file_name="forward_${noise_level}_${epsilon}_metrics.pkl"
                # Check if the file exists
                if [ -f "$file_direction/$file_name" ]; then
                    echo "File found: $file_direction/$file_name"
                    
                else
                    #echo "File not found: $file_direction/$file_name"
                    sbatch launch_ambiguity_forward.sh $noise_type $model_type $dataset $noise_level
                fi
            done
        done
    done

done

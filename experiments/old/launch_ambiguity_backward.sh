#!/usr/bin/env bash
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH -a 0
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --time=20:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=/h/snagaraj/noise_multiplicity/logs/ambiguity_backward/slurm-%A.out
#SBATCH --error=/h/snagaraj/noise_multiplicity/logs/ambiguity_backward/slurm-%A.out
#SBATCH --open-mode=append

source /pkgs/anaconda3/bin/activate noisyTS

python3 -u run_ambiguity_real.py  --noise_type $1 --model_type $2 --dataset $3 --noise_level $4 --uncertainty_type "backward" --n_models 1000
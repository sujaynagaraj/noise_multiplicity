#!/usr/bin/env bash
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH -a 0
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --time=16:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=/h/snagaraj/noise_multiplicity/logs/stability/slurm-%A_%a.out
#SBATCH --error=/h/snagaraj/noise_multiplicity/logs/stability/slurm-%A_%a.out
#SBATCH --open-mode=append


source /pkgs/anaconda3/bin/activate noisyTS

python3 -u run_stability.py  --noise_type $1 --model_type $2 --dataset $3 


#!/usr/bin/env bash
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH -a 0
#SBATCH --qos=m
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=/h/snagaraj/noise_multiplicity/logs/ambiguity_toy/slurm-%A_%a.out
#SBATCH --error=/h/snagaraj/noise_multiplicity/logs/ambiguity_toy/slurm-%A_%a.out
#SBATCH --open-mode=append


source /pkgs/anaconda3/bin/activate noisyTS

python3 -u run_ambiguity_toy.py --setting $1 --noise_type $2


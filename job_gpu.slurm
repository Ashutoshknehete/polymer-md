#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4g
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nehet004@umn.edu
#SBATCH -e %j.err
#SBATCH -o %j.out

echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR

conda activate md
#python ../routine_graft.py
python ../analysis.py

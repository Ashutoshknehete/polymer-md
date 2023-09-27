#!/bin/bash -l

python project_job.py submit -o simulation --job-output="slurm_logs/slurm-%j.out" -- --time=12:00:00 --mem=8g --mail-type=NONE --mail-user=nehet004@umn.edu --partition=a100-4 --gres=gpu:a100:1

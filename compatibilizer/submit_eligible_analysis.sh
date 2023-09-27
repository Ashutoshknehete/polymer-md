python project_job.py submit -o analysis -b 10 --job-output="slurm_logs/slurm-%j.out" -- --time=2:00:00 --mem=8g --mail-type=NONE --mail-user=nehet004@umn.edu --partition=msismall

python project_job.py submit -o compute_descriptors -b 200 --job-output="slurm_logs/slurm-%j.out" -- --time=1:00:00 --mem=2g --mail-type=NONE --mail-user=nehet004@umn.edu --partition=msismall

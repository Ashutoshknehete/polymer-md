import signac

project = signac.get_project()
ctr = 0
for job in project:
    if job.isfile("struct/random.gsd") and not job.isfile("struct/prod.gsd"):
        job.clear()
        ctr+=1
print("Cleared {:d} jobs that were initialized but failed.".format(ctr))

import signac

def micellized(job):
    if "junction_peaks" in job.doc:
        return job.doc.junction_peaks > 2
    else:
        return False # can't label as micellized if it hasn't been checked!

p = signac.get_project()

i = 0
for j in p:
    i+=1
    if i%500==0:
        print(i)
    if micellized(j):
        j.clear() 

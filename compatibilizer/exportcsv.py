import os
import numpy as np
import matplotlib.pyplot as plt
from dataPackage.preprocess import AggregateDataset

def add_surfacepressure(df):
    # adds diff (gam0-gam) column

    # get the 0-cp case (need to modify this if we tweak epsilon!)
    gamma0_avg = df[df['sp.num_CP']==0]["doc.bulk_interfacial_tension_average"].values
    gamma0_var = df[df['sp.num_CP']==0]["doc.bulk_interfacial_tension_variance"].values

    num_blocks = np.sort(df["num_blocks"].unique())
    baselengths = np.sort(df["base_length"].unique())
    col_avg = np.zeros((df.shape[0],1))
    col_var = np.zeros((df.shape[0],1))
    # for each base end-block length
    for baselength in baselengths:
        if baselength==0:
            continue
        # for each number of junctions. there should now be 3 results remaining
        junctioncounts = np.sort(df[df["base_length"]==baselength]["num_junction"].unique())
        for njunc in junctioncounts:
            # for each # blocks
            for nblock in num_blocks:
                filter = (df["base_length"]==baselength) & (df["num_junction"] == njunc) & (df["num_blocks"]==nblock)
                gamma_avg = df[filter]["doc.bulk_interfacial_tension_average"].values
                gamma_var = df[filter]["doc.bulk_interfacial_tension_variance"].values
                gammadiff_avg = -(gamma_avg-gamma0_avg)
                gammadiff_var = gamma_var + gamma0_var
                idx = np.where(filter)
                col_avg[idx] = gammadiff_avg
                col_var[idx] = gammadiff_var
    
    # add columns
    df["surface_pressure_avg"] = col_avg
    df["surface_pressure_var"] = col_var

    return df

def add_reldiff(df):
    # get the 0-cp case
    num_blocks = np.sort(df["num_blocks"].unique())
    baselengths = np.sort(df["base_length"].unique())
    col_avg = np.zeros((df.shape[0],1))
    col_var = np.zeros((df.shape[0],1))
    # for each base end-block length
    for baselength in baselengths:
        if baselength==0:
            continue
        # for each number of junctions. there should now be 3 results remaining
        junctioncounts = np.sort(df[df["base_length"]==baselength]["num_junction"].unique())
        for njunc in junctioncounts:
            filter_db = (df["base_length"]==baselength) & (df["num_junction"] == njunc) & (df["num_blocks"]==2)
            idx_db = np.where(filter_db)
            sp_db_avg = df[filter_db]["surface_pressure_avg"].values
            sp_db_var = df[filter_db]["surface_pressure_var"].values
            col_avg[idx_db] = float("NaN")
            col_var[idx_db] = float("NaN")
            # for each length not db
            for nblock in num_blocks:
                if nblock == 2:
                    continue
                filter = (df["base_length"]==baselength) & (df["num_junction"] == njunc) & (df["num_blocks"]==nblock)
                sp_avg = df[filter]["surface_pressure_avg"].values
                sp_var = df[filter]["surface_pressure_var"].values
                n = sp_avg
                d = sp_db_avg
                rel_avg = n/d
                rel_var = (1/d)**2 * sp_var + (n/(d**2))**2 * sp_db_var
                idx = np.where(filter)
                col_avg[idx] = rel_avg
                col_var[idx] = rel_var
    
    # set 0 examples to 1 because limit from right is probably 1
    col_avg[np.where(df["sp.num_CP"]==0)] = float("NaN")
    col_var[np.where(df["sp.num_CP"]==0)] = float("NaN")
    # add columns
    df["rel_surface_pressure_avg"] = col_avg
    df["rel_surface_pressure_var"] = col_var

    return df

# load project
dataroot = os.getcwd()+"/aggregate/"
aggregatedata = AggregateDataset(dataroot)
df = aggregatedata.data

# add useful descriptors to dataframe
df["num_junction"] = np.array([len(row["sp.length_CP"])-1 for name,row in df.iterrows()])*df["sp.num_CP"]
df["num_beads"] = np.array([sum(row["sp.length_CP"]) for name,row in df.iterrows()]*df["sp.num_CP"]+df["sp.num_A"]*df["sp.length_A"]+df["sp.num_B"]*df["sp.length_B"])
df["base_length"] = np.array([row["sp.length_CP"][0] for name,row in df.iterrows()])
df["num_blocks"] = np.array([len(row["sp.length_CP"]) for name,row in df.iterrows()])

# add surface pressure column
df = add_surfacepressure(df)
df = add_reldiff(df)

# create normalized column with wca density variances
wcavars = np.loadtxt("densvarWCA.txt")
normvars = []
for avg,var in zip(df["doc.junctions_per_area"], df["doc.interfacial_density_variance"]):
    if avg==0: # bare systems
        normvar = np.NaN
    else:
        wcavar = wcavars[np.isclose(wcavars[:,0],avg),1][0]
        normvar = var/wcavar
    normvars.append(normvar)
df["relative_junction_density_variance"] = normvars

# export dataframe to csv
df.to_csv("data/aggregate.csv")

import os
import pickle
import numpy as np
from scipy.signal import find_peaks

from flow import FlowProject
import hoomd
import gsd.hoomd
from signac import JSONDict

from polymerMD.structure import systemspec, systemgen
from polymerMD.simtools import sim_routines
from polymerMD.analysis import trajtools, statistics
import matplotlib.pyplot as plt

from routine_functions import build_system_spec_graft
from routine_functions import build_system_spec_linear
from routine_functions import build_system_spec_mikto
from format_plots import format_plots
import json
import xlsxwriter

wb = xlsxwriter.Workbook("mydata.xlsx")
sheet1 = wb.add_worksheet('Sheet 1')  
sheet1.write(0, 1, 'N_A')
sheet1.write(0, 2, 'M_A')
sheet1.write(0, 3, 'N_B')
sheet1.write(0, 4, 'M_B')
sheet1.write(0, 5, 'N_CP_A')
sheet1.write(0, 6, 'N_CP_B')
sheet1.write(0, 7, 'M_CP')
sheet1.write(0, 8, 'n_arms')
sheet1.write(0, 9, 'rho')
sheet1.write(0, 10, 'aspect')
sheet1.write(0, 11, 'architecture')
sheet1.write(0, 12, 'interfactial_tension_avg')
sheet1.write(0, 13, 'interfactial_tension_var')
sheet1.write(0, 14, 'interfactial_tension_nsamples')

with open('parameters.json', 'r') as file:        
    parameters = json.load(file)
    
n_simulations = len(parameters["N_A"])
for i in range(n_simulations):

        # System parameters
        N_A = parameters["N_A"][i]
        M_A = parameters["M_A"][i]
        N_B = parameters["N_B"][i]
        M_B = parameters["M_B"][i]
        N_CP = parameters["N_CP"][i]
        M_CP = parameters["M_CP"][i]
        n_arms = parameters["n_arms"][i]
        rho = parameters["rho"][i]
        aspect = parameters["aspect"][i]
        architecture = parameters["architecture"][i]
        
        sheet1.write(i+1, 1, N_A)
        sheet1.write(i+1, 2, M_A)
        sheet1.write(i+1, 3, N_B)
        sheet1.write(i+1, 4, M_B)
        sheet1.write(i+1, 5, N_CP[0])
        sheet1.write(i+1, 6, N_CP[1])
        sheet1.write(i+1, 7, M_CP)
        sheet1.write(i+1, 8, n_arms)
        sheet1.write(i+1, 9, rho)
        sheet1.write(i+1, 10, aspect)
        sheet1.write(i+1, 11, architecture)
        
        if architecture=="linear":
            system = build_system_spec_linear(M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms)
        if architecture=="mikto":
            system = build_system_spec_mikto(M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms)
        if architecture=="graft":
            system = build_system_spec_graft(M_A, N_A, M_B, N_B, M_CP, N_CP, n_arms)
        
        colors = format_plots(plt)
        
        fname_prod_log = architecture+"_prod_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.log.gsd".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        # load log and structure data
        dat = gsd.hoomd.open(fname_prod_log,'rb')
        fname_prod = architecture+"_prod_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.gsd".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        snap = gsd.hoomd.open(fname_prod, 'rb')[0]
        
        # compute interfacial tension for each frame, determine average and variance
        axis=0 # fix the axis! We really don't need to generalize it. Change later if needed
        L = snap.configuration.box[axis]
        t,gammas = trajtools.interfacial_tension_global(dat,axis,L)
        gammas = np.array(gammas)
        t = np.squeeze(t)
        # compute average interfacial tension and store
        avg_gamma = np.average(gammas)
        sheet1.write(i+1, 12, avg_gamma)
        # compute variance using estimated autocorrelation time
        var_gamma = statistics.estimator_variance(gammas)
        sheet1.write(i+1, 13, var_gamma)
        # compute the number of independent samples and the average for each sample
        samples = statistics.get_independent_samples(gammas,factor=2)
        nsamples = np.shape(samples)[0]
        sheet1.write(i+1, 14, nsamples)
        fig = plt.figure(figsize=(10, 6))
        np.random.seed(0)
        mu = avg_gamma  # Mean
        sigma = np.std(samples)  # Standard deviation
        random_numbers = np.random.normal(0, 1, nsamples)
        hist, bins, _ = plt.hist(random_numbers, bins=100, density=True, color='red', alpha=0.5, label="random #s (normal gaussian dist)")
        bar_width = bins[1] - bins[0]
        total_area = np.sum(hist * bar_width)
        
        scaled_gammas = (samples - mu) / (2 * sigma**2)
        hist, bins, _ = plt.hist(scaled_gammas, bins=100, density=True, color='blue', alpha=0.5, label="scaled interfacial tension data")
        bar_width = bins[1] - bins[0]
        total_area = np.sum(hist * bar_width)

        plt.xlabel('($\gamma$-$\mu$)/$\sigma^2$')
        plt.ylabel('Normalized Probability Density')
        plt.title('Probability Density Histogram')
        plt.legend()
        plt.show()
        
        '''
        # density profiles of monomers
        # in the future we should average this over many frames! but I haven't been recording trajectories...
        snap = gsd.hoomd.open(fname_prod, 'rb')[0]
        profiles = trajtools.density_1D_monomers(snap)
        fname_density_1D_monomers = architecture+"_density1Dmonomers_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.pkl".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        with open(fname_density_1D_monomers, 'wb') as f:
            pickle.dump(profiles, f)
        with open(fname_density_1D_monomers, 'rb') as f:
            data = pickle.load(f)
        keys_list = list(data)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
        ax.plot(data[keys_list[0]][1][0:100], data[keys_list[0]][0], color=colors["black"], label=f'monomer = {keys_list[0]}')
        ax.plot(data[keys_list[1]][1][0:100], data[keys_list[1]][0], color=colors["gray"], label=f'monomer = {keys_list[1]}')
        fname_density_1D_monomers_png = architecture+"_density1Dmonomers_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.png".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        plt.legend()
        plt.xlabel(r'x')
        plt.ylabel(r'$\phi(x)$')
        plt.title("1D volume fraction of monomers")
        plt.savefig(fname_density_1D_monomers_png)
        plt.close()

        profiles = trajtools.density_1D_species(snap,system,nBins=100)
        fname_density_1D_species = architecture+"_density1Dspecies_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.pkl".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        with open(fname_density_1D_species, 'wb') as f:
            pickle.dump(profiles, f)
        with open(fname_density_1D_species, 'rb') as f:
            data = pickle.load(f)
        keys_list = list(data)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
        ax.plot(data[keys_list[0]][1][0:100], data[keys_list[0]][0], color=colors["black"], label=f'species = {keys_list[0]}')
        ax.plot(data[keys_list[1]][1][0:100], data[keys_list[1]][0], color=colors["almond"], label=f'species = {keys_list[1]}')
        ax.plot(data[keys_list[2]][1][0:100], data[keys_list[2]][0], color=colors["gray"], label=f'species = {keys_list[2]}')
        fname_density_1D_species_png = architecture+"_density1Dspecies_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.png".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        plt.legend()
        plt.xlabel(r'x')
        plt.ylabel(r'$\phi(x)$')
        plt.title("1D volume fraction of species")
        plt.savefig(fname_density_1D_species_png)
        plt.close()
        speciesRsq = trajtools.internaldistances_species(snap,system)
        fname_internal_dist_species = architecture+"_internal_dist_species_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.pkl".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        with open(fname_internal_dist_species, 'wb') as f:
            pickle.dump(speciesRsq, f)
        with open(fname_internal_dist_species, 'rb') as f:
            data = pickle.load(f)
        keys_list = list(data)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
        xaxis = data[keys_list[0]][0]
        yaxis = np.array(data[keys_list[0]][1])/np.array(data[keys_list[0]][0])
        ax.plot(xaxis, yaxis, color=colors["black"], label=f'species = {keys_list[0]}')
        xaxis = data[keys_list[1]][0]
        yaxis = np.array(data[keys_list[1]][1])/np.array(data[keys_list[1]][0])
        ax.plot(xaxis, yaxis, color=colors["gray"], label=f'species = {keys_list[1]}')
        xaxis = data[keys_list[2]][0]
        yaxis = np.array(data[keys_list[2]][1])/np.array(data[keys_list[2]][0])
        ax.plot(xaxis, yaxis, color=colors["almond"], label=f'species = {keys_list[2]}')
        plt.xscale('log')
        fname_internal_dist_species_png = architecture+"_internal_dist_species_NA={:04d}_MA={:04d}_NB={:04d}_MB={:04d}_NCP={:04d}{:04d}_MCP={:04d}_narms={:04d}.png".format(N_A,M_A,N_B,M_B,N_CP[0],N_CP[1],M_CP,n_arms)
        plt.xlabel('n')
        plt.ylabel(r'$<R_{int}$$ ^2>$')
        plt.legend()
        plt.title("Mean Squared Internal Distances")
        plt.savefig(fname_internal_dist_species_png)
        plt.close()
        '''
        
wb.close()

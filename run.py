# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:51:34 2024

@author: Scott
"""
import multiprocessing
from boxmodel import model, fb, atmocntuning
from boxmodel import (t01,t02,t03,t04,t05,t06,t07, mse_gamma_w,mse_gamma_e,mse_gamma_n,mse_gamma_s,mse_gamma_hn,mse_gamma_hs,
mse_int_w,mse_int_e,mse_int_n,mse_int_s,swlw1,swlw2,swlw3,swlw4,swlw6,swlw7,co2)
import numpy as np
from copy import deepcopy
import os

# IMPORTING BOX MODEL CODE FROM boxmodel.py
# RUNNING HERE TO ALLOW PARALLEL PROCESSING
# HAVE TO SETUP FB AND OTHER PARAMS IN boxmodel.py BEFORE RUNNING HERE
# run this file in console: python ./run.py

# In boxmodel.py, check the following before running:
#      - length of run (timesteps, second number is years)
#      - Feedback parameters (fb)
#      - Pattern feedbacks (fb_adapt_bool)
#      - Using ERA5 params for all or CMIP6 from each model (cmip6params)
#%%

runtitle = "fb.sensneg3to2.t0.ora5.w002e002.nor005sou008.hn0002hs0002.uniformforcing.F.huang.mod.AHTfixed.OHTfixed"
print(f"filename= {runtitle}")

def merge_files(num_chunks, name, output_filename):
    merged_results = []
    for i in range(num_chunks):
        chunk_filename = f"{runtitle}.chunk{i}.{name}.npy"
        chunk_results = np.load(chunk_filename, allow_pickle=True)
        merged_results.extend(chunk_results)
        os.remove(chunk_filename)  # Remove chunk file after merging

    np.save(output_filename, merged_results)

# co2=[8]



#%%

changenum=1 
for itr in range(changenum): #decide what length you want to run for any parameter suite
    
    
    for c in range(len(co2)): #for each co2 forcing
    
        # if __name__ == "__main__":
            
        num_iterations = len(fb)
        num_processors = multiprocessing.cpu_count()
        print(f"Using {num_processors} processors for {num_iterations} tasks")
        
        chunk_size = 64 # for some reason this is the max number of possible iterations at once
        num_chunks = (num_iterations + chunk_size - 1) // chunk_size  # Calculate number of chunks to get through all iterations

        
        # Assuming fb contains tuples of parameters, including the iteration index
        parameters = [(i, itr, c, t01, t02, t03, t04, t05, t06, t07, mse_gamma_w, mse_gamma_e, mse_gamma_n, mse_gamma_s, mse_gamma_hn, mse_gamma_hs, mse_int_w, mse_int_e, mse_int_n, mse_int_s, swlw1, swlw2, swlw3, swlw4, swlw6, swlw7) for i in range(num_iterations)]
        
        for chunk_index in range(num_chunks):
            start_index = chunk_index * chunk_size
            end_index = min(start_index + chunk_size, num_iterations)
            parameters_chunk = parameters[start_index:end_index]

            # Create a pool of worker processes
            with multiprocessing.Pool(processes=num_processors) as pool:
                results = pool.starmap(model, deepcopy(parameters_chunk))
    
            print(len(results[:]))
    
            
            # extract results
            allT = [ results[i][0][0] for i in range(len(results[:])) ]
            ocean = [ results[i][1][0] for i in range(len(results[:])) ]
            atmosconv = [ results[i][2][0] for i in range(len(results[:])) ]
            totfb = [ results[i][3][0] for i in range(len(results[:])) ]
            # meanaht = [ results[i][4][0] for i in range(num_iterations) ]

            np.save( f"{runtitle}.chunk{chunk_index}.allT.npy", np.asarray(allT))
            np.save( f"{runtitle}.chunk{chunk_index}.ocean.npy", np.asarray(ocean))
            np.save( f"{runtitle}.chunk{chunk_index}.atmosconv.npy", np.asarray(atmosconv))
            np.save( f"{runtitle}.chunk{chunk_index}.totfb.npy", np.asarray(totfb))

    # Merge all chunk files into one final output file
    allfiles = ['allT','ocean','atmosconv','totfb']
    for name in allfiles:
        merge_files(num_chunks,name, f"{runtitle}.{name}.npy")

    print("All results have been processed and merged")
            
#%%
# extract results
# allT = [ results[i][0][0] for i in range(num_iterations) ]
# ocean = [ results[i][1][0] for i in range(num_iterations) ]
# atmosconv = [ results[i][2][0] for i in range(num_iterations) ]
# totfb = [ results[i][3][0] for i in range(num_iterations) ]
# # meanaht = [ results[i][4][0] for i in range(num_iterations) ]

# np.save( f"{runtitle}.allT.npy", np.asarray(allT))
# np.save( f"{runtitle}.ocean.npy", np.asarray(ocean))
# np.save( f"{runtitle}.atmosconv.npy", np.asarray(atmosconv))
# np.save( f"{runtitle}.totfb.npy", np.asarray(totfb))
# np.save( f"{runtitle}.meanaht.npy", np.asarray(meanaht))

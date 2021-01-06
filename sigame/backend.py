###     Module: backend.py of SIGAME                   ###

# Import other SIGAME modules
import sigame.galaxy as gal
import sigame.auxil as aux
import sigame.global_results as glo
import sigame.Cloudy_modeling as clo

# Import other modules
from multiprocessing import Pool
import numpy as np
import pdb as pdb
import os
from argparse import Namespace
import time
import copy

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

global p
p                      =   aux.load_parameters()

def setup_SKIRT(GR):
    """ Creates input scripts for SKIRT to calculate the 
    interstellar radiation field (ISRF), 
    attenuated by dust via radiative transfer (RT).
    """

    print('\n--- Step 1: Create input scripts for the attenuated interstellar radiation field (ISRF) with SKIRT ---')

    gal_indices = np.arange(0,GR.N_gal)
 
    # Set up SKIRT models
    gal.setup_SKIRT(gal_indices)

    print('\n--------------------------------------------------------------\n')

def read_SKIRT(GR):
    """ Read output from SKIRT and create 
    new galaxy object based on SKIRT output grid
    """

    print('\n--- Step 1: Read output from SKIRT ---')

    gal_indices = np.arange(0,GR.N_gal)

    if p.N_cores > 1:
        agents = p.N_cores
        chunksize = 1
        with Pool(processes=agents) as pool:
            results = pool.map(gal.read_SKIRT, gal_indices, chunksize)
    else:
        for gal_index in gal_indices:
            print('\nNow for galaxy # %s' % gal_index)
            gal.read_SKIRT(gal_index)

def grid_gas(GR):
    """ Performs re-gridding of gas particle properties on the AMR 
    cell grid structure provided by SKIRT.

    """

    print('\n--- Step 2: Re-grid gas on cell structure ---')

    gal_indices = np.arange(GR.N_gal)

    if p.N_cores > 1:
        agents = p.N_cores
        chunksize = 1
        with Pool(processes=agents) as pool:
            results = pool.map(gal.run_grid, gal_indices, chunksize)
    else:
        for gal_index in range(GR.N_gal):
            print('\nNow for galaxy # %s' % gal_index)
            gal.run_grid(gal_index)

    print('\n--------------------------------------------------------------\n')

def setup_Cloudy_grid(GR):
    """ Creates input scripts for grid of Cloudy models
    """

    print('\n--- Step 3: Setup Cloudy grid ---')

    cloudy_library              =   clo.library(GR,verbose=True)

    # Setup Cloudy grid input
    cloudy_library.setup_grid_input(ext='')

    # Create PBS scripts
    cloudy_library.create_job_scripts(ext='')

def run_Cloudy(GR):
    """ Runs Cloudy grid models
    """

    print('\n--- Step 3: Run Cloudy ---')

    # Run Cloudy
    cloudy_library.submit_jobs(ext='')

def combine_Cloudy(GR):
    """ combines Cloudy grid output if some grids only finished partially
    """

    print('\n--- Step 3: Combine Cloudy otuput ---')

    # Combine output files from unfinished grid models
    cloudy_library.combine_output_files(ext='')

def complete_Cloudy(GR):
    """ Completes Cloudy grid if some models did not finish
    """

    print('\n--- Step 3: Complete Cloudy grid ---')

    # Restart grid models that never ran
    cloudy_library.debug_grid(ext='')

def read_Cloudy_grid(GR):
    """ Read Cloudy grid output
    """

    print('\n--- Step 3: Read Cloudy grid output ---')

    # Read Cloudy grid output
    cloudy_library.read_grids()

def sample_Cloudy_table(GR):
    """ Reads Cloudy grid output
    """

    print('\n--- Step 3: Sample Cloudy grid in look-up table ---')

    # Sample in terms of mean density (or other parameter)
    cloudy_library.sample_cloudy_table()

def interpolate(GR):
    """ Performs interpolation in Cloudy look-up tables to get line luminosities 
    for all gas particles in each galaxy.

    """

    print('\n--- Step 4: Interpolate for line luminosities ---')

    gal_indices = np.arange(GR.N_gal) #380,400)#
    
    if p.N_cores > 1:
        agents = p.N_cores
        chunksize = 1
        with Pool(processes=agents) as pool:
            results = pool.map(gal.run_interp, gal_indices, chunksize)
    else:
        for gal_index in gal_indices:
            print('\nNow for galaxy # %s' % gal_index)
            gal.run_interp(gal_index)

    # Add mass-weighted quantities to global results
    gal.add_mw_quantities()


    

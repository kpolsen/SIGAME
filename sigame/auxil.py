# coding=utf-8
"""
Module: aux
"""

import numpy as np
import numexpr as ne
import pandas as pd
import pdb as pdb
import scipy as scipy
from scipy import optimize
# import scipy.stats as stats
import scipy.integrate as integrate
from scipy.interpolate import RegularGridInterpolator,griddata
from scipy.interpolate import interp1d
import multiprocessing as mp
import os
import time
import matplotlib.pyplot as plt
import linecache as lc
import re as re
import sys as sys
import sympy as sy
import astropy as astropy
from argparse import Namespace
from astropy.cosmology import FlatLambdaCDM
from argparse import Namespace
import pickle as pickle
import copy
import subprocess as sub
import astropy.constants as c
import astropy.units as u
from scipy.spatial import cKDTree
import sigame.galaxy as gal

#===========================================================================
""" Load parameters (used by all other modules) """
#---------------------------------------------------------------------------

def load_parameters():
    # The following will be automatically filled out by __init__.py the first time you run SIGAME.
    # To change after that, just overwrite sigame_directory with your current work folder:
    
    sigame_directory        =   os.getcwd() + '/'
    
    params                  =   np.load(sigame_directory + 'temp_params.npy', allow_pickle=True)#.item()

    return(params)

#===========================================================================
""" File handling """
#---------------------------------------------------------------------------

global params
params                      =   load_parameters()

def get_file_location(**kwargs):
    """
    Finds correct location and file name for a certain file type and galaxy
    """
    import os

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if p.gal_ob_present:
        try:
            p.zred = kwargs['gal_ob'].zred
            p.galname = kwargs['gal_ob'].name
            p.gal_num = kwargs['gal_ob'].gal_num
            p.gal_index = kwargs['gal_ob'].gal_index
        except:
            # Assume gal_ob is actually a dictionary
            p.zred = kwargs['gal_ob']['zred']
            p.galname = kwargs['gal_ob']['galname']
            p.gal_num = kwargs['gal_ob']['gal_num']
            p.gal_index = kwargs['gal_ob']['gal_index']

    # Determine data path
    if p.data_type == 'rawsimgas':         path = p.d_XL_data+'/data/particle_data/sim_data/'
    if p.data_type == 'rawsimstar':         path = p.d_XL_data+'/data/particle_data/sim_data/'
    if p.data_type == 'simgas':         path = p.d_XL_data+'/data/particle_data/'
    if p.data_type == 'simstar':        path = p.d_XL_data+'/data/particle_data/'
    if p.data_type == 'cell_data':      path = p.d_XL_data+'/data/cell_data/'
    # Determine filename
    try: 
        if not os.path.exists(path): os.mkdir(path)
        filename = os.path.join(path, 'z'+'{:.2f}'.format(p.zred)+'_%i' % p.gal_num+'_'+p.sim_name+p.sim_run+'.'+p.data_type)
        if p.data_type == 'cell_data': 
            filename = os.path.join(path, 'z'+'{:.2f}'.format(p.zred)+'_%i' % p.gal_num+'_'+p.sim_name+p.sim_run+'.'+p.data_type)
    except:
        print("Need the following to create filename: gal_ob (or zred, galname and data_type)")
        raise NameError

    return(filename)

def update_dictionary(values,new_values):
    """ updates the entries to values with entries from new_values that have
    matching keys. """
    for key in values:
        if key in new_values:
            values[key]     =   new_values[key]
    return values

def save_temp_file(data, subgrid=None, **kwargs):
    """
    Stores temporary files according to their sim or ISM type and stage of processing.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    filename    =   get_file_location(**kwargs)

    #print("Saving dataframe with pickle as %s" % filename)
    data.to_pickle(filename)

    # h5store(data, 'data', filename, **kwargs)

def load_temp_file(verbose=False,**kwargs):
    """Way to load metadata with dataframe
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    filename    =   get_file_location(**kwargs)
    #print(filename)

    try:
        data = pd.read_hdf(filename)
        try:
            data            =   data['data'][0]
        except:
            data            =   data
    except:
        if 'DTM' in p.sim_run:
            try:
                if '.sim' in filename: filename = filename.replace('_DTM','')
                data            =   pd.read_pickle(filename)
                if verbose: print('Found file at '+filename)
            except:
                if verbose: print('Did not find file at '+filename)
                data            =   0
        else:
            try:
                data            =   pd.read_pickle(filename)
                if verbose: print('Found file at '+filename)
            except:
                if verbose: print('Did not find file at '+filename)
                data            =   0

    if p.verbose:
        if type(data) != int: print('Loaded file at %s' % filename)
    return data

def h5store(df, dc_name, filename, **kwargs):
    """Way to store metadata with dataframe
    """

    try:
        metadata            =   df.metadata
    except:
        metadata            =   {}

    store = pd.HDFStore(filename)
    store.put(dc_name, df)
    store.get_storer(dc_name).attrs.metadata = metadata
    store.close()

#===========================================================================
""" For galaxy.isrf class """
#---------------------------------------------------------------------------

def NIR_index(wa):
    # sum within: (https://www2.chemistry.msu.edu/faculty/reusch/virttxtjml/cnvcalc.htm)
    # 0.5 eV to 2 eV 
    wa1 = c.c.value*c.h.value/(2*u.eV.to('J'))*1e6 # microns
    wa2 = c.c.value*c.h.value/(0.5*u.eV.to('J'))*1e6 # microns
    indices = np.arange(len(wa))
    indices = indices[(wa >= wa1) & (wa <= wa2)]
    N_start = np.min(indices)
    N_stop = np.max(indices)+1
    return(N_start,N_stop)

def FUV_index(wa):
    # sum within: (https://www2.chemistry.msu.edu/faculty/reusch/virttxtjml/cnvcalc.htm)
    # 13.6 eV (= 0.0912 micron) to 6 eV (= 0.2066 micron)
    wa1 = c.c.value*c.h.value/(13.6*u.eV.to('J'))*1e6 # microns
    wa2 = c.c.value*c.h.value/(6*u.eV.to('J'))*1e6 # microns
    indices = np.arange(len(wa))
    indices = indices[(wa >= wa1) & (wa <= wa2)]
    N_start = np.min(indices)
    N_stop = np.max(indices)+1
    return(N_start,N_stop)

def UV_index(wa):
    # sum within: (https://www2.chemistry.msu.edu/faculty/reusch/virttxtjml/cnvcalc.htm)
    # 100 to 13.6 eV 
    wa1 = c.c.value*c.h.value/(100*u.eV.to('J'))*1e6 # microns
    wa2 = c.c.value*c.h.value/(13.6*u.eV.to('J'))*1e6 # microns
    indices = np.arange(len(wa))
    indices = indices[(wa >= wa1) & (wa <= wa2)]
    N_start = np.min(indices)
    N_stop = np.max(indices)+1
    return(N_start,N_stop)

def MIR_index(wavelengths):
    # sum within: 10-24 microns ??
    indices = np.arange(len(wavelengths))
    indices = indices[(wavelengths >= 10) & (wavelengths <= 24)]
    N_start = np.min(indices)
    N_stop = np.max(indices)+1
    return(N_start,N_stop)

def TIR_index(wavelengths):
    # sum within: 3-1100 microns
    # as defined in Kennicutt and Evans 2012 table 1:
    #https://www.annualreviews.org/doi/abs/10.1146/annurev-astro-081811-125610
    
    indices = np.arange(len(wavelengths))
    indices = indices[(wavelengths >= 3) & (wavelengths <= 1100)]
    N_start = np.min(indices)
    N_stop = np.max(indices)+1
    return(N_start,N_stop)

def FIR_index(wavelengths):
    # sum within: 40-500 microns
    # https://ned.ipac.caltech.edu/level5/Sanders/Sanders2.html 
    
    indices = np.arange(len(wavelengths))
    indices = indices[(wavelengths >= 40) & (wavelengths <= 500)]
    N_start = np.min(indices)
    N_stop = np.max(indices)+1
    return(N_start,N_stop)

def read_probe_wavelengths(name):
    """ Read wavelengths used for radiation field probe of each cell.
    """

    p = copy.copy(params)

    isrf_wavelengths = pd.read_csv(p.d_skirt+name+'_rfpc_wavelengths.dat',comment='#',sep=' ',engine='python',\
        names=['%i' % i for i in range(4)]) # microns
    # isrf_freq = p.clight/(isrf_wavelengths*1e-6) # Hz
    # isrf_wavelengths['bin_width'] = isrf_wavelengths['3'] - isrf_wavelengths['2']
    bin_width = isrf_wavelengths['1'].values
    wavelengths = isrf_wavelengths['0'].values
    return(wavelengths,bin_width)

def read_SED_inst_wavelengths(name):
    """ Read wavelengths used for SED instrument.
    """

    p = copy.copy(params)

    isrf_wavelengths = pd.read_csv(p.d_skirt+'%s_xy_wavelengths.dat' % name,skiprows=4,sep=' ',engine='python',\
        names=['wavelength','bin_width','left','right']) # microns
    return(isrf_wavelengths['wavelength'].values,isrf_wavelengths['bin_width'].values)

def read_map_inst_wavelengths(name):
    """ Read wavelengths used for mapping ("frame") instrument.
    """

    p = copy.copy(params)

    isrf_wavelengths = pd.read_csv(p.d_skirt+'%s_xy_map_wavelengths.dat' % name,skiprows=4,sep=' ',engine='python',\
        names=['wavelength','bin_width','left','right']) # microns
    return(isrf_wavelengths['wavelength'].values,isrf_wavelengths['bin_width'].values)


def read_probe_intensities(name,Nbins):
    """ Read flux (in W/m2/micron/sr) from radiation field probe of each cell.
    """

    p = copy.copy(params)

    isrf_intensities = pd.read_csv(p.d_skirt+name+'_rfpc_J.dat',comment='#',sep=' ',engine='python',\
            names=['%i' % (i-1) for i in range(Nbins+1)],usecols = [i+1 for i in range(Nbins)])
    return(isrf_intensities)



#===========================================================================
""" For galaxy.frag class """
#---------------------------------------------------------------------------

def parametric_PDF(n, center, width, slope):
    """ Make mass distribution from one lognormal with power-law tail"""

    od = 10.**np.linspace(-9,10,10000)
    
    # Transition point in log of overdensity
    trans = (-1*slope - 1/2) * width**2

    #od = 10.**np.linspace(np.log10(od_bins.min()),np.log10(od_bins.max()),5000)
    lognormal = PDF_MHD_func(od,width)
 
    od_crit = np.exp(trans)
    if od_crit < np.max(od):
        
        lognormal = lognormal[od <= od_crit]
        
        powerlaw = od[od > od_crit]**slope
        powerlaw = powerlaw * lognormal[-1] / powerlaw[0]
        PDF = np.append(lognormal, powerlaw)
    else:
        PDF = lognormal

    # Integrate PDF shape on raw bins
    n = 10.**n
    od_bins = n / 10.**center
    logx_bin_size = (np.log10(od_bins[1]) - np.log10(od_bins[0]))/2.
    PDF_integrated = np.zeros(len(od_bins))
    for i_PDF,logx in enumerate(np.log10(od_bins)):
        logx_bin = np.array([logx-logx_bin_size,logx+logx_bin_size])
        x_bin = 10.**logx_bin
        indices = np.where((od >= x_bin[0]) & (od <= x_bin[1]))
        PDF_integrated[i_PDF] = np.trapz(PDF[indices],np.log(od[indices]))
    
    # Normalize
    PDF_integrated = PDF_integrated / np.sum(PDF_integrated)

    return(np.log10(PDF_integrated))

def PDF_MHD_func(x,width,log=True):
    """ Density distribution in terms of mass per density bin.
    x: overdensity (density/mean(density))
    beta: gas to magnetic pressure ratio
    M: Mach number (velocity dispersion / sound speed)

    See Padoan+97, eq. 24 in Padoan+ 2011 and eq. 9a in Pallottini+ 2019
    """

    # eq. 1 in Padoan+97 p(lnx) and Burkhart+18 eq. 2:
    sigma_MHD_2 = width**2 # sigma squared
    mean_lnx = -1./2*sigma_MHD_2
    if log: p_MHD = 1/(2*np.pi*sigma_MHD_2)**0.5*np.exp(-1/2*((np.log(x) - mean_lnx)**2/sigma_MHD_2))
    if not log: p_MHD = x**(-1)/(2*np.pi*sigma_MHD_2)**0.5*np.exp(-1/2*((np.log(x) - mean_lnx)/sigma_MHD_2)**2/(2*sigma_MHD_2))

    return(p_MHD)

def lognormal_PDF(n,n_vw,Mach=10):
    """ Make mass distribution from one lognormal"""

    od_bins = n/n_vw
    #overdensity = 10.**np.linspace(-5,8,5000)
    od = 10.**np.linspace(-9,10,10000)
    
    # Add lognormal at n_vm
    forcing_parameter = 1/3 # forcing parameter, between 1/3 and 1
    width = np.log(1 + forcing_parameter**2*Mach**2) # eq. 5 in Burkhart+18
    p_dlnx = PDF_MHD_func(od,width)

    # Integrate PDF shape on raw bins
    logx_bin_size = (np.log10(od_bins[1]) - np.log10(od_bins[0]))/2.
    PDF_integrated = np.zeros(len(od_bins))
    for i_PDF,logx in enumerate(np.log10(od_bins)):
        logx_bin = np.array([logx-logx_bin_size,logx+logx_bin_size])
        x_bin = 10.**logx_bin
        indices = np.where((od >= x_bin[0]) & (od <= x_bin[1]))
        PDF_integrated[i_PDF] = np.trapz(p_dlnx[indices],x=np.log(od[indices]))

    return(PDF_integrated/np.sum(PDF_integrated))

def lognormal_powerlaw_PDF(n,ratio,n_vw,Mach=10):
    """ Make 1 or 2 lognormals with power-law tails"""
    
    overdensity_plot = n/n_vw
    overdensity = 10.**np.linspace(-5,8,5000)
    
    # Add Mach = 10 lognormal at n_vm
    width = np.log(1 + forcing_parameter**2*Mach**2) # eq. 5 in Burkhart+18
    p_dlnx = PDF_MHD_func(overdensity,width)
    # TEST: Adding a power-law tail at high densities, with critical density from Burkhart+17
    beta_0 = 20 # thermal-to-magnetic pressure of plasma, between 0.2 and 20
    forcing_parameter = 1/3 # forcing parameter, between 1/3 and 1
    Mach = 10
    overdensity_crit = ( 1 + forcing_parameter**2 * Mach**2 * beta_0 / (beta_0 + 1) )
    p_dlnx = p_dlnx[0:int(np.where(overdensity < overdensity_crit)[0][-1]+1)]
    powerlaw_slope = -1.65 # slope
    powerlaw = overdensity[overdensity >= overdensity_crit] ** powerlaw_slope 
    powerlaw = powerlaw * p_dlnx[-1] / powerlaw[0] # normalize
    PDF = np.append(p_dlnx,powerlaw)
    # Integrate PDF shape on raw bins
    lognH_bin_size = (np.log10(overdensity_plot[1]) - np.log10(overdensity_plot[0]))/2.
    PDF_integrated1 = np.zeros(len(overdensity_plot))
    for i_PDF,lognH in enumerate(np.log10(overdensity_plot)):
        lognH_bin = np.array([lognH-lognH_bin_size,lognH+lognH_bin_size])
        x_bin = 10.**lognH_bin/(n_vw)
        # Integrate shape of lognormal in this bin
        indices = np.where((overdensity >= x_bin[0]) & (overdensity <= x_bin[1]))
        PDF_integrated1[i_PDF] = np.trapz(PDF[indices],x=np.log(overdensity[indices]))
    PDF_integrated = PDF_integrated1    

    if ratio != 0:
        # Add Mach = 10 lognormal at n_vm*ratio
        overdensity_plot = n/(n_vw*ratio)
        p_dlnx = PDF_MHD_func(overdensity,Mach)
        # TEST: Adding a power-law tail at high densities, with critical density from Burkhart+17
        beta_0 = 20 # thermal-to-magnetic pressure of plasma, between 0.2 and 20
        forcing_parameter = 1/3 # forcing parameter, between 1/3 and 1
        Mach = 10
        overdensity_crit = ( 1 + forcing_parameter**2 * Mach**2 * beta_0 / (beta_0 + 1) )
        p_dlnx = p_dlnx[0:int(np.where(overdensity < overdensity_crit)[0][-1]+1)]
        powerlaw_slope = -1.65 # slope
        powerlaw = overdensity[overdensity >= overdensity_crit] ** powerlaw_slope 
        powerlaw = powerlaw * p_dlnx[-1] / powerlaw[0] # normalize
        PDF = np.append(p_dlnx,powerlaw)
        # Integrate PDF shape on raw bins
        lognH_bin_size = (np.log10(overdensity_plot[1]) - np.log10(overdensity_plot[0]))/2.
        PDF_integrated2 = np.zeros(len(overdensity_plot))
        for i_PDF,lognH in enumerate(np.log10(overdensity_plot)):
            lognH_bin = np.array([lognH-lognH_bin_size,lognH+lognH_bin_size])
            x_bin = 10.**lognH_bin/(n_vw)
            # Integrate shape of lognormal in this bin
            indices = np.where((overdensity >= x_bin[0]) & (overdensity <= x_bin[1]))
            PDF_integrated2[i_PDF] = np.trapz(PDF[indices],x=np.log(overdensity[indices]))
        
        PDF_integrated = PDF_integrated1 + PDF_integrated2
        
    return(PDF_integrated/np.sum(PDF_integrated))


def Wendland_C2_kernel(r,h):
    """ From: https://github.com/SWIFTSIM/swiftsimio/blob/master/swiftsimio/visualisation/slice.py
    Kernel implementation for swiftsimio. This is the Wendland-C2
    kernel as shown in Denhen & Aly (2012).
    Give it a radius and a kernel width (i.e. not a smoothing length, but the
    radius of compact support) and it returns the contribution to the
    density.
    """

    kernel_constant = 21.0 * 0.31830988618379067154 / 2.0
    inverse_H = 1.0 / h
    ratio = r * inverse_H

    kernel = 0.0

    if ratio < 1.0:
        one_minus_ratio = 1.0 - ratio
        one_minus_ratio_2 = one_minus_ratio * one_minus_ratio
        one_minus_ratio_4 = one_minus_ratio_2 * one_minus_ratio_2

        kernel = max(one_minus_ratio_4 * (1.0 + 4.0 * ratio), 0.0)

        kernel *= kernel_constant * inverse_H * inverse_H * inverse_H

    return kernel

def Pfunc(i,simgas1,simgas,simstar,m_gas,m_star):
    """ Calculate the surface-density-dependent term that can get mid-plane pressure in galaxy._add_P_ext()
    OBS: Need for speed!!!
    """

    # Distance to other gas particles in disk plane:
    posxy       =   ['x','y','z']
    dist1       =   np.linalg.norm(simgas1[posxy].values.astype('float')-simgas.loc[i][posxy].values.astype('float'),axis=1)
    # print(len(dist1))
    # print(len(simgas))
    m_gas1      =   m_gas[dist1 < simgas['h'][i]]
    # pressure,surf_gas,surf_star,sigma_gas,sigma_star,vel_disp_gas = [0 for j in range(0,6)]
    if len(m_gas1) >= 1:
        surf_gas    =   sum(m_gas1)/(np.pi*simgas['h'][i]**2.)
        sigma_gas   =   np.std(simgas1.loc[dist1 < simgas['h'][i]]['vz'])
        # Distance to other star particles in disk plane:
        dist2       =   np.linalg.norm(simstar[posxy].values-simgas.loc[i][posxy].values,axis=1)
        m_star1     =   m_star[dist2 < simgas['h'][i]]
        if len(m_star1) >= 1:
            surf_star   =   sum(m_star1)/(np.pi*simgas['h'][i]**2.)
            sigma_star  =   np.std(simstar.loc[dist2 < simgas['h'][i]]['vz'])
        # Total velocity dispersion of gas
        vel_disp_gas   =   simgas1['vel_disp'].values[dist1 < simgas['h'][i]]
        if len(simstar.loc[dist2 < simgas['h'][i]]) == 0: sigma_star = 0
        if sigma_star != 0: pressure_term = surf_gas*(surf_gas+(sigma_gas/sigma_star)*surf_star)
        if sigma_star == 0: pressure_term = surf_gas*(surf_gas)

    else:
        if simgas['SFR'][i] > 0:
            surf_gas    =   simgas['m'][i]/(np.pi*simgas['h'][i]**2.)
            m_star1     =   m_star[dist2 < simgas['h'][i]]
            if len(m_star1) >= 1:
                surf_star   =   sum(m_star1)/(np.pi*simgas['h'][i]**2.)

    return i, pressure_term,surf_gas,surf_star,sigma_gas,sigma_star,vel_disp_gas

#===========================================================================
""" For galaxy.interpolation class """
#---------------------------------------------------------------------------

def get_NH_from_cloudy(logZ=0):
    '''
    Purpose
    ---------
    Read grid made with different metallicities and column densities 
    to convert NIR/FUV flux ratios to column densities.
    '''

    p = copy.copy(params)

    # READ TRANSMITTED SPECTRA FROM CLOUDY GRD
    cont2 = pd.read_table(p.d_table+'cloudy/NH/grid_run_ext.cont2',skiprows=0,names=['E','I_trans','coef'])
    E = cont2.E.values
    i_shift = np.array(['########################### GRID_DELIMIT' in _ for _ in E])
    i_delims = np.arange(len(cont2))[i_shift == True]
    N_E = i_delims[0]-9 # First 9 lines are header
    cont = np.zeros([len(i_delims),N_E])
    for i,i_delim in enumerate(i_delims):
        I_trans = cont2.I_trans[i_delim-N_E:i_delim].astype(float)
        cont[i,:] = I_trans

    # READ METALLICITIES AND COLUMN DENSITIES
    out = open(p.d_table+'cloudy/NH/grid_run_ext.out','r')
    logNHs = []
    logZs = []
    start = False
    for line in out.readlines():
        if line == ' **************************************************\n': start = True
        if start:
            if 'STOP COLUMN DENSITY ' in line:
                logNHs.append(float(line.split('COLUMN DENSITY ')[1].split(' ')[0]))
            if 'METALS' in line:
                logZs.append(float(line.split(' ')[2]))
        if line == ' Writing input files has been completed.\n':
            break
    logNHs = np.array(logNHs)
    logZs = np.array(logZs)

    x = (E[i_delims[0]-N_E:i_delims[0]]).astype(float)*u.Ry.to('eV') # eV
    F_FUV_W_m2 = np.array([np.sum(cont[_,(x >= 6) & (x < 13.6)])*1e-7*1e6 for _ in range(cont.shape[0])])
    F_NIR_W_m2 = np.array([np.sum(cont[_,(x >= 0.5) & (x < 2)])*1e-7*1e6 for _ in range(cont.shape[0])])
    R_NIR_FUV = F_NIR_W_m2 / F_FUV_W_m2

    # SELECT ONE METALLICITY
    logNHs = logNHs[logZs==logZ]
    R_NIR_FUV = R_NIR_FUV[logZs==logZ]
    logR_NIR_FUV = np.log10(R_NIR_FUV)

    return(logNHs,R_NIR_FUV)

#===========================================================================
""" For global_results.global_results class """
#---------------------------------------------------------------------------

def get_lum_dist(zred):
    '''
    Purpose
    ---------
    Calculate luminosity distance for a certain redshift
    returns D_L in Mpc
    '''

    p = copy.copy(params)

    cosmo               =   FlatLambdaCDM(H0=p.hubble*100., Om0=p.omega_m, Ob0=1-p.omega_m-p.omega_lambda)

    if len(zred) > 1:
        D_L                 =   cosmo.luminosity_distance(zred).value
        zred_0              =   zred[zred == 0]
        if len(zred_0) > 0:
            D_L[zred == 0]      =   3+27.*np.random.rand(len(zred_0)) # Mpc (see Herrera-Camus+16)

    if len(zred) == 1:
        D_L                 =   cosmo.luminosity_distance(zred).value

    # ( Andromeda is rougly 0.78 Mpc from us )

    return(D_L)

def pretty_print(col_names,units,format,values):

    N = 15*len(values[0])
    string1 = '+'
    string2 = '|'
    string3 = '|'
    for i in range(len(col_names)):
        string1 += 15*'-' + '-'
        string2 += '%15s' % (col_names[i].center(15)) + '|'
        string3 += '%15s' % (units[i].center(15)) + '|'
    string1 += '+'
    print(string1)
    print(string2)
    print(string3)
    print(string1)

    for j in range(len(values[0])):
        string4 = '|'
        for i,value in enumerate(values):
            if format[i] == 's': string4 += '%15s' % (value[j].center(15)) + '|'
            if format[i] == 'e': string4 += '%15s' % ('{:.2e}'.format(value[j])) + '|'
            if format[i] == 'f': string4 += '%15s' % ('{:.2f}'.format(value[j])) + '|'
        print(string4)
    print(string1)

#===========================================================================
""" For plotting and mapping """
#---------------------------------------------------------------------------

def select_salim18(M_star,SFR):

    MS_salim = pd.read_csv('data/observations/MS/salim2018_ms_v1.dat',\
            names=['logMstar','logsSFR','logsSFR_1','logsSFR_2'],sep='   ')
    f1 = interp1d(MS_salim.logMstar,MS_salim.logsSFR_1)
    SFR_low = M_star*10.**f1(np.log10(M_star))
    f2 = interp1d(MS_salim.logMstar,MS_salim.logsSFR_2)
    SFR_high = M_star*10.**f2(np.log10(M_star))

    indices = np.arange(len(M_star))

    indices = indices[(SFR > SFR_low) & (SFR < SFR_high)]

    return indices

def convert_cell_data_to_regular_grid(res=0.5,plane='xy',**kwargs):
    """ 
    Purpose
    ---------
    Creates and stores a matrix that maps cell data to a regular grid in 2D.

    Parameters
    ----------
    gal_index: int
        Galaxy index, default: 0

    res: float
        Pixel resolution in kpc

    plane: str
        Plane to project to (xy, xz, yz)

    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    gal_ob             =   gal.galaxy(p.gal_index)
    datacube           =   gal_ob.cell_data.get_dataframe()
    datacube['m_H2']   =  datacube.m*datacube.mf_H2_grid
    datacube['m_HII']  =  datacube.m*datacube.mf_HII_grid
    datacube['m_HI']   =  datacube.m*datacube.mf_HI_grid
    properties = np.column_stack((datacube.m, datacube.m_H2, datacube.m_HI, datacube.m_HII, \
                                  datacube.m * datacube.Z, datacube.m * datacube.G0,  datacube.m * datacube.ne_mw_grid,\
                                  datacube['L_[CII]158']/6, datacube['L_[CI]610']/6,\
                                  datacube['L_[CI]370']/6, datacube['L_[OI]145']/6, datacube['L_[OI]63']/6, datacube['L_[OIII]88']/6,\
                                  datacube['L_[NII]122']/6, datacube['L_[NII]205']/6, datacube['L_CO(3-2)']/6, datacube['L_CO(2-1)']/6,\
                                  datacube['L_CO(1-0)']/6, datacube['L_[OIV]25']/6, datacube['L_[NeII]12']/6, datacube['L_[NeIII]15']/6,\
                                  datacube['L_[SIII]18']/6, datacube['L_[FeII]25']/6,\
                                  datacube['L_H2_S(1)']/6, datacube['L_H2_S(2)']/6, datacube['L_H2_S(3)']/6, datacube['L_H2_S(4)']/6,\
                                  datacube['L_H2_S(5)']/6, datacube['L_H2_S(6)']/6, datacube['L_H2_S(7)']/6))

    
    print('Max x: ',datacube.x.max())
    cellsize = datacube.cell_size                     # (Cell-sizes in kpc)                         
    X = datacube.x                                    # (x-coordinates)
    Y = datacube.y                                    # (y-coordinates)
    Z = datacube.z                                    # (z-coordinates)
    
    xmin, xmax = min(X), max(X)
    ymin, ymax = min(Y), max(Y)
    zmin, zmax = min(Z), max(Z)
    xminindex, xmaxindex = np.where(X == xmin)[0][0], np.where(X == xmax)[0][0]
    yminindex, ymaxindex = np.where(Y == ymin)[0][0], np.where(Y == ymax)[0][0]
    zminindex, zmaxindex = np.where(Z == zmin)[0][0], np.where(Z == zmax)[0][0]
    
    cellarea = cellsize * cellsize                     # kpc^2
    area = np.column_stack((cellarea, cellarea, cellarea, cellarea, cellarea, cellarea, cellarea, cellarea, \
                   cellarea, cellarea, cellarea, cellarea, cellarea, cellarea, cellarea, cellarea,\
                   cellarea, cellarea, cellarea, cellarea, cellarea, cellarea, cellarea, cellarea,\
                   cellarea, cellarea, cellarea, cellarea, cellarea, cellarea))
    celldensity =  properties / area                   # units / kpc^2
    
    
    if plane == 'xy':
        axis1, min1, max1, minindex1, maxindex1 = X, xmin, xmax, xminindex, xmaxindex
        axis2, min2, max2, minindex2, maxindex2 = Y, ymin, ymax, yminindex, ymaxindex

    elif plane == 'yz':
        axis1, min1, max1, minindex1, maxindex1 = Y, ymin, ymax, yminindex, ymaxindex
        axis2, min2, max2, minindex2, maxindex2 = Z, zmin, zmax, zminindex, zmaxindex
    
    elif plane == 'xz':
        axis1, min1, max1, minindex1, maxindex1 = X, xmin, xmax, xminindex, xmaxindex
        axis2, min2, max2, minindex2, maxindex2 = Z, zmin, zmax, zminindex, zmaxindex
        
        
    # Cell setups: 
    cellstart1 = axis1 - (cellsize / 2)
    cellstart2 = axis2 - (cellsize / 2)
    cellend1 = cellstart1 + cellsize
    cellend2 = cellstart2 + cellsize

    # Arrays containing index and x,y,z coordinate-details of each cell:
    cell1 = np.column_stack([datacube.index, cellstart1, cellend1])
    cell2 = np.column_stack([datacube.index, cellstart2, cellend2]) 
    
    # Scipy's KD-Tree method to find nearest neghbours:
    tree = cKDTree(np.c_[axis1.ravel(), axis2.ravel()])
    R = (max(cellsize)/(2**0.5)) + (res/(2**0.5)) + 0.1
    

    # While loop to go through axis2-direction:
    index = 0
    index2 = 0
    momentmapdata = []
    pixelstart2 = min2 - (cellsize[minindex2] / 2)
    while pixelstart2 <= max2 + (cellsize[maxindex2] / 2): 
        pixelend2 = pixelstart2 + res

        # While loop to go through axis1-direction:
        index1 = 0
        pixelstart1 = min1 - (cellsize[minindex1] / 2)
        while pixelstart1 <= max1 + (cellsize[maxindex1] / 2):
            pixelend1 = pixelstart1 + res

            # Central coordinates of each pixel-lines:
            pixelcentre1 = pixelstart1 + (res/2)
            pixelcentre2 = pixelstart2 + (res/2)

            # Pixel coordinates:
            pixel = [pixelstart1, pixelcentre1, pixelend1, pixelstart2, pixelcentre2, pixelend2]

            # Calling line_luminosity function for the particular pixel: 
            lineluminosity = line_luminosity(tree, cell1, cell2, pixel, celldensity, R)
            
            if np.sum(lineluminosity) > 0:
                pass
            else:
                lineluminosity = np.zeros(len(properties[0]))

            # Appending data of each pixelline for 2D projection:
            momentmapdata.append([int(index), pixelcentre1, pixelcentre2, lineluminosity])

            index += 1
            index1 += 1 
            pixelstart1 += res

        index2 += 1
        pixelstart2 += res
        
    momentmap = np.array(momentmapdata)
    momentmap = np.append(momentmap, [['pixel size', index1, index2, 0]], axis=0) # Information of image size in pixel
        
    if p.sim_type == 'sph': filename = p.d_XL_data + 'data/regular_cell_data/moment0map_%s%s_%i' % (p.sim_name,p.sim_run,p.gal_index) + '_' + plane + '_res' + str(res)
    if p.sim_type == 'amr': filename = p.d_XL_data + 'data/regular_cell_data/moment0map_' + p.sim_name + p.sim_run + '_' + plane + '_res' + str(res)
                
    np.save(filename, momentmap)
    
def line_luminosity(tree, cell1, cell2, pixel, celldensity, R):
    '''
    Purpose
    ---------
    This function returns the added up total luminosity of the pixel.
    '''
    
    # Pixel-coordinates:
    pixelstart1 = pixel[0]
    pixelcentre1 = pixel[1]
    pixelend1 = pixel[2]
    pixelstart2 = pixel[3]
    pixelcentre2 = pixel[4]
    pixelend2 = pixel[5]
    
    # 1. Query_ball_point method of KDTree to find cells within distance R:
    ii = tree.query_ball_point([pixelcentre1, pixelcentre2], r=R)
    
    # Appending coordinate values of the nearest cells gained by KD-Tree:
    ax1 = []
    ax2 = []
    for index in ii:
        ax1.append(cell1[index])
        ax2.append(cell2[index])
    
    # 2. For loop to go over each cell:
    lineluminosity = 0
    for i in range(len(ax1)):
        
        # 3. if statement to find whether the cell overlaps the pixel or not:
        if (max(pixelstart1, ax1[i][1]) - min(pixelend1, ax1[i][2]) < 0) and (max(pixelstart2, ax2[i][1]) - min(pixelend2, ax2[i][2]) < 0):
            
            # 4. Finding edge lengths of overlapping rectangle:
            edge1 = min(pixelend1, ax1[i][2])-max(pixelstart1,ax1[i][1])
            edge2 = min(pixelend2, ax2[i][2])-max(pixelstart2,ax2[i][1])
            
            # 5. Finding overlapping crosssection area and luminosity due to that overlpping cell:
            crosssection = edge1 * edge2
            pixelluminosity = crosssection * celldensity[int(ax1[i][0])]
            
            # 6. Adding up the luminosity due to each cell:
            lineluminosity += pixelluminosity

    return lineluminosity

def projection_scaling(xg,yg,zg,hg,res_kpc):
    """ Find a scaling that can bring all coordinates to [0:1]
    for compatibility with swiftsimio functions
    """

    x = xg -1.*np.min(xg); y = yg -1.*np.min(yg); z = zg -1.*np.min(zg)
    max_scale = np.max([np.max(np.abs(x)),np.max(np.abs(y)),np.max(np.abs(z))])*1.5
    print('Scaling gas positions by: %.2f' % max_scale)

    # Resulting number of pixels
    Npix = int(np.ceil(max_scale/res_kpc))
    print('Corresponds to %i pixels' % Npix)
    print('With chosen resolution of %.2f kpc' % res_kpc)
    x,y,z,h = (xg-np.mean(xg))/max_scale+0.5,(yg-np.mean(yg))/max_scale+0.5,(zg-np.mean(zg))/max_scale+0.5,hg/max_scale
    
    pix_size = max_scale/Npix

    return(x,y,h,max_scale,pix_size,Npix)

def MS_SFR_Mstar(Mstars):
    ''' Calculate SFR for given Mstars using MS at a certain redshift
    '''

    p = copy.copy(params)

    from astropy.cosmology import FlatLambdaCDM

    cosmo               =   FlatLambdaCDM(H0=p.hubble*100., Om0=p.omega_m, Ob0=1-p.omega_m-p.omega_lambda)

    age                 =   cosmo.age(p.zred).value # Gyr

    SFR_MS              =   10.**((0.84-0.026*age)*np.log10(Mstars)-(6.51-0.11*age)) # eq. 28 in Speagle+14

    return(SFR_MS)

#===========================================================================
""" Unit conversions """
#---------------------------------------------------------------------------

def K_km_s_pc2_to_Lsun(L_K_km_s_pc2,line):
    """ From https://github.com/aconley/ALMAzsearch/blob/master/ALMAzsearch/radio_units.py.
    """

    p = copy.copy(params)

    freq = p.freq[line]

    # This is L_sun * c**3 / (4 pi (1pc/1m)**2 * k_B) in SI units (kB is in J/K),
    # which shows up in the lsun to K km/s pc^2 conversion
    lsun_kkmspc2_const = c.L_sun.value * c.c.value**3 / c.k_B / (4.0 * np.pi * u.pc.to('m') ** 2)
    L_sun = L_K_km_s_pc2 / (5e-4 * lsun_kkmspc2_const * (freq*1e9) ** (-3))

    return(L_sun)

def Lsun_to_K_km_s_pc2(L_sun,line):
    """ From https://github.com/aconley/ALMAzsearch/blob/master/ALMAzsearch/radio_units.py.
    """
    
    p = copy.copy(params)

    freq = p.freq[line]

    # This is L_sun * c**3 / (4 pi (1pc/1m)**2 * k_B) in SI units (kB is in J/K),
    # which shows up in the lsun to K km/s pc^2 conversion
    lsun_kkmspc2_const = c.L_sun.value * c.c.clight**3 / c.k_B / (4.0 * np.pi * c.pc.to('m') ** 2)

    L_K_km_s_pc2 = L_sun * 5e-4 * lsun_kkmspc2_const * (freq*1e9) ** (-3)

    # Fits with eqs from Solomon 1997 ApJ 478:
    # L_K_km_s_pc2 = L_sun * 3.125e10 * freq ** (-3)

    return(L_K_km_s_pc2)

def Lsun_to_Jy_km_s(L_sun,D_L,line):
    """ From Solomon 1997 ApJ 478
    """
    
    p = copy.copy(params)

    freq = p.freq[line]

    L_Jy_km_s = L_sun * (1+p.zred)/(1.04e-3 * freq * D_L**2) 

    return(L_Jy_km_s)

def Jy_km_s_to_Lsun(L_Jy_km_s,D_L,line):
    """ From Solomon 1997 ApJ 478
    """
    
    p = copy.copy(params)

    freq = p.freq[line]

    L_sun = L_Jy_km_s * 1.04e-3 * freq * D_L**2 / (1+p.zred)

    return(L_sun)

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
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d,interp2d,RectBivariateSpline
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
from scipy.optimize import curve_fit


#===========================================================================
""" Load parameters (used by all other modules) """
#---------------------------------------------------------------------------

def load_parameters():
    # Load parameters chosen in parameters.txt and those added in params.py.
    
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
            p.gal_num = kwargs['gal_ob'].gal_num
            p.gal_index = kwargs['gal_ob'].gal_index
        except:
            # Assume gal_ob is actually a dictionary
            p.zred = kwargs['gal_ob']['zred']
            p.gal_num = kwargs['gal_ob']['gal_num']
            p.gal_index = kwargs['gal_ob']['gal_index']

    # Determine data path
    if p.data_type == 'rawsimgas':      path = p.d_XL_data+'/data/particle_data/sim_data/'
    if p.data_type == 'rawsimstar':     path = p.d_XL_data+'/data/particle_data/sim_data/'
    if p.data_type == 'simgas':         path = p.d_XL_data+'/data/particle_data/'
    if p.data_type == 'simstar':        path = p.d_XL_data+'/data/particle_data/'
    if p.data_type == 'cell_data':      
        path = p.d_XL_data+'/data/cell_data/'
        if p.w_hii: path = p.d_XL_data+'/data/cell_data/w_hii/'
    if 'd_data' in kwargs.keys():
        if kwargs['d_data'] != '':
            path = kwargs['d_data']
    # Determine filename
    try: 
        if not os.path.exists(path): os.mkdir(path)
        filename = os.path.join(path, 'z'+'{:.2f}'.format(p.zred)+'_%i' % p.gal_num+'_'+p.sim_name+p.sim_run+'.'+p.data_type)
        if p.data_type == 'cell_data': 
            filename = os.path.join(path, 'z'+'{:.2f}'.format(p.zred)+'_%i' % p.gal_num+'_'+p.sim_name+p.sim_run+'.'+p.data_type) #+p.skirt_ext
    except:
        print("Need the following to create filename: gal_ob (or zred, galname and data_type)")
        raise NameError

    #print(p.gal_index,filename)
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

def OIR_index(wa):
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

    isrf_wavelengths = pd.read_csv(p.d_skirt+name+p.skirt_ext+'_rfpc_wavelengths.dat',comment='#',sep=' ',engine='python',\
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

    isrf_wavelengths = pd.read_csv(p.d_skirt+'%s_xy_wavelengths.dat' % (name+p.skirt_ext),skiprows=4,sep=' ',engine='python',\
        names=['wavelength','bin_width','left','right']) # microns
    return(isrf_wavelengths['wavelength'].values,isrf_wavelengths['bin_width'].values)

def read_map_inst_wavelengths(name):
    """ Read wavelengths used for mapping ("frame") instrument.
    """

    p = copy.copy(params)

    isrf_wavelengths = pd.read_csv(p.d_skirt+'%s_xy_map_wavelengths.dat' % (name+p.skirt_ext),skiprows=4,sep=' ',engine='python',\
        names=['wavelength','bin_width','left','right']) # microns
    return(isrf_wavelengths['wavelength'].values,isrf_wavelengths['bin_width'].values)

def read_probe_intensities(name,Nbins):
    """ Read flux (in W/m2/micron/sr) from radiation field probe of each cell.
    """

    p = copy.copy(params)

    isrf_intensities = pd.read_csv(p.d_skirt+name+p.skirt_ext+'_rfpc_J.dat',comment='#',sep=' ',engine='python',\
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
""" For Cloudy_modeling and galaxy.interpolation class """
#---------------------------------------------------------------------------

def get_Cloudy_lines_dict():

    Cloudy_lines_dict = {\
        'O  1 145.495m' : '[OI]145',\
        'O  1 63.1679m' : '[OI]63',\
        'O  3 88.3323m' : '[OIII]88',\
        'O  4 25.8832m' : '[OIV]25',\
        'N  2 205.244m' : '[NII]205',\
        'N  2 121.767m' : '[NII]122',\
        'NE 2 12.8101m' : '[NeII]12',\
        'NE 3 15.5509m' : '[NeIII]15',\
        'C  2 157.636m' : '[CII]158',\
        'C  1 609.590m' : '[CI]610',\
        'C  1 370.269m' : '[CI]370',\
        'CO   2600.05m' : 'CO(1-0)',\
        'CO   1300.05m' : 'CO(2-1)',\
        'CO   866.727m' : 'CO(3-2)',\
        'CO   650.074m' : 'CO(4-3)',\
        'CO   325.137m' : 'CO(8-7)',\
        'S  3 18.7078m' : '[SIII]18',\
        'FE 2 25.9811m' : '[FeII]25',\
        'H2   17.0300m' : 'H2_S(1)',\
        'H2   12.2752m' : 'H2_S(2)',\
        'H2   9.66228m' : 'H2_S(3)',\
        'H2   8.02362m' : 'H2_S(4)',\
        'H2   6.90725m' : 'H2_S(5)',\
        'H2   6.10718m' : 'H2_S(6)',\
        'H2   5.50996m' : 'H2_S(7)'}

    return(Cloudy_lines_dict)

def get_NH_from_cloudy(logZ=0):
    '''
    Purpose
    ---------
    Read grid made with different metallicities and column densities 
    to convert OIR/FUV flux ratios to column densities.
    '''

    p = copy.copy(params)
    grid_ext_name = 'grid_run_ext%s' % p.grid_ext.replace('_ext','')

    # READ TRANSMITTED SPECTRA FROM CLOUDY GRD IN UNITS erg/s
    if p.grid_ext == '_ext':
        p.d_cloudy = p.d_cloudy.replace('ext/','')
    cont2 = pd.read_table(p.d_cloudy+'NH/%s.cont2' % grid_ext_name,skiprows=0,names=['E','I_trans','coef'])
    E = cont2.E.values
    i_shift = np.array(['########################### GRID_DELIMIT' in _ for _ in E])
    i_delims = np.arange(len(cont2))[i_shift == True]
    N_E = i_delims[0]-9 # First 9 lines are header
    cont = np.zeros([len(i_delims),N_E])
    for i,i_delim in enumerate(i_delims):
        I_trans = cont2.I_trans[i_delim-N_E:i_delim].astype(float)
        cont[i,:] = I_trans

    # READ METALLICITIES AND COLUMN DENSITIES
    grd = pd.read_csv(p.d_cloudy + 'NH/%s.grd' % grid_ext_name,skiprows=1,sep='\t',names=['i','Failure','Warnings','Exit code','rank','seq','logNH','gridparam'])
    logNHs = grd['logNH'].values#[grd['Exit code'] != '       failed assert']
    print('Number of NHs : %i' % len(logNHs))

    E_eV = (E[i_delims[0]-N_E:i_delims[0]]).astype(float)*u.Ry.to('eV') # eV
    E_Hz = E_eV * u.eV.to(u.J) / c.h.value

    F_FUV = np.array([scipy.integrate.simps(cont[_,(E_eV >= 6) & (E_eV < 13.6)]/E_Hz[(E_eV >= 6) & (E_eV < 13.6)]) for _ in range(cont.shape[0])])
    F_OIR = np.array([scipy.integrate.simps(cont[_,(E_eV >= 0.5) & (E_eV < 2)]/E_Hz[(E_eV >= 0.5) & (E_eV < 2)]) for _ in range(cont.shape[0])])
    R_OIR_FUV1 = F_OIR / F_FUV 
    logR_OIR_FUV1 = np.log10(R_OIR_FUV1)

    return(logNHs,R_OIR_FUV1)

def random_positions(r,pos,N):

    ra          =   np.random.rand(N)  # draw nn random numbers between 0 and 1
    ra1         =   np.random.rand(N)  # draw nn random numbers between 0 and 1
    ra2         =   np.random.rand(N)  # draw nn random numbers between 0 and 1
    ra_R        =   ra*r
    ra_phi      =   ra1*2*np.pi
    ra_theta    =   ra2*np.pi
    ra          =   [ra_R*np.sin(ra_theta)*np.cos(ra_phi),+\
                    ra_R*np.sin(ra_theta)*np.sin(ra_phi),+\
                    ra_R*np.cos(ra_theta)]
    new_pos     =   np.column_stack((pos[0]+np.array(ra)[0,:],pos[1]+np.array(ra)[1,:],pos[2]+np.array(ra)[2,:]))

    return(new_pos,ra_R)

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

def moment0_map_location(**kwargs):
    """ Returns location of moment0 map data.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)
    if p.sim_type == 'amr': location = p.d_XL_data + 'data/regular_cell_data/moment0map_%s%s' % (p.sim_name,p.sim_run) + '_' + p.plane + '_res' + str(p.res) + '.npy'
    if p.sim_type == 'sph': location = p.d_XL_data + 'data/regular_cell_data/moment0map_%s%s_%i%s' % (p.sim_name,p.sim_run,p.gal_index,p.skirt_ext) + '_' + p.plane + '_res' + str(p.res) + '.npy'
    print(location)
    return location

def select_salim18(M_star,SFR):
    """ Selects galaxies within quartile range of Salim+18 z=0 MS.
    """

    MS_salim = pd.read_csv('data/observations/MS/salim2018_ms_v1.dat',\
            names=['logMstar','logsSFR','logsSFR_1','logsSFR_2'],sep='   ')
    f1 = interp1d(MS_salim.logMstar,MS_salim.logsSFR_1,fill_value="extrapolate")
    SFR_low = M_star*10.**f1(np.log10(M_star))
    f2 = interp1d(MS_salim.logMstar,MS_salim.logsSFR_2,fill_value="extrapolate")
    SFR_high = M_star*10.**f2(np.log10(M_star))

    indices = np.arange(len(M_star))

    indices = indices[(SFR > SFR_low) & (SFR < SFR_high)]

    return indices

def distance_from_salim18(M_star,SFR):
    """ Returns distance from Salim+18 z=0 MS in SFR.
    """

    MS_salim = pd.read_csv('data/observations/MS/salim2018_ms_v1.dat',\
            names=['logMstar','logsSFR','logsSFR_1','logsSFR_2'],sep='   ')

    f1 = interp1d(MS_salim.logMstar,np.log10(10.**MS_salim.logMstar*10.**MS_salim.logsSFR),fill_value="extrapolate")
    lSFR_S18 = f1(np.log10(M_star))

    dlSFR = np.log10(SFR) - lSFR_S18

    return dlSFR


def add_FIR_lum(res=1,R=0,plane='xy',**kwargs):
    """ Add FIR luminosity per pixel from SKIRT map output
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    location = moment0_map_location(res=res,plane=plane,gal_index=p.gal_index)
    df = pd.read_pickle(location.replace('.npy',''))

    # Plot mw nH from moment0 map
    fig, ax1 = plt.subplots(figsize=(10,10))
    ax1.scatter(df.x,df.y,marker='o',c=df.nH_mw)
    ax1.set_xlim([-R,R])
    ax1.set_ylim([-R,R])
    plt.savefig(p.d_plot+'moment0/test_G%i_nH' % (p.gal_index))

    # Plot FIR lum per pixel
    isrf_ob = gal.isrf(p.gal_index)
    image_data,units,wa     =   isrf_ob._get_map_inst(orientation=plane,select=p.select)
    N_start,N_stop          =   FIR_index(wa)
    FIR_xy_image            =   image_data[N_start:N_stop,:,:].sum(axis=0)
    index1                  =   np.arange(image_data.shape[1])
    index2                  =   np.arange(image_data.shape[2])
    index1,index2           =   np.meshgrid(index1,index2)
    F_FIR_xy_image            =   np.zeros([image_data.shape[1],image_data.shape[2]])
    # Integrate to get from W/m$^2$/micron/arcsec$^2$ to W/m$^2$/arcsec$^2$
    for i1,i2 in zip(index1.flatten(),index2.flatten()):
        F_FIR_xy_image[i1,i2]            =   np.trapz(image_data[N_start:N_stop,i1,i2],x=wa[N_start:N_stop])
    x = np.linspace(-R,R,FIR_xy_image.shape[0])
    y = np.linspace(-R,R,FIR_xy_image.shape[1])
    # Integrate over sphere and solid angle and convert to solar luminosity
    L_FIR_xy_image = F_FIR_xy_image * 4 * np.pi * 4 * np.pi * (10e6 * c.pc.to('m').value)**2 / p.Lsun 
    f_F_FIR = RectBivariateSpline(x,y,F_FIR_xy_image) 
    f_L_FIR = RectBivariateSpline(x,y,L_FIR_xy_image) 
    print(L_FIR_xy_image.min(),L_FIR_xy_image.max())
    print('FIR luminosity from summing over image: %.4e' % (L_FIR_xy_image.sum()))

    # Interpolate FIR flux at pixel locations
    F_FIR_pixels = f_F_FIR(df.y.values.astype('float64'),df.x.values.astype('float64'),grid=False)
    L_FIR_pixels = f_L_FIR(df.y.values.astype('float64'),df.x.values.astype('float64'),grid=False)
    df['F_FIR'] = F_FIR_pixels.flatten()
    df['L_FIR'] = L_FIR_pixels.flatten()
    
    df.to_csv(location.replace('.npy','.csv'))
    df.to_pickle(location.replace('.npy',''))

def convert_cell_data_to_regular_grid(res=0.5,plane='xy',**kwargs):
    """ 
    This function creates and stores a matrix that maps cell data to a regular grid in 2D.

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
    #datacube = datacube[(datacube.x > -5) & (datacube.x < 5) & (datacube.y > -5) & (datacube.y < 5 ) & (datacube.z > -5) & (datacube.z < 5) ].reset_index(drop=True)
    datacube['m_H2']   =  datacube.m*datacube.mf_H2_grid
    datacube['m_HII']  =  datacube.m*datacube.mf_HII_grid
    datacube['m_HI']   =  datacube.m*datacube.mf_HI_grid
    # Only count volume of cells with ISM (threshold from Romeel)
    cell_volume = datacube['cell_volume'].values
    cell_volume[datacube.nH.values < 0.01] = 0
    datacube['cell_volume'] = cell_volume
    datacube = datacube.fillna(0)
    # CAREFUL! Order here must correspond to dictionary in param.py
    print('Min and max pressure:')
    print(datacube.P_mw.min(),datacube.P_mw.max())
    properties = np.column_stack((datacube.m, datacube.cell_volume, datacube.m_HII, datacube.m_HI, datacube.m_H2, \
                              datacube.m * datacube.Z, datacube.m * datacube.G0,\
                              # PRESSURES
                              datacube.m * datacube.P_mw,\
                              datacube.m * datacube.P_e_mw, \
                              # TEMPERATURES
                              datacube.m * datacube.Tk_mw,\
                              datacube.m_HII * datacube.Te_mw, 
                              # MASS-WEIGHTED DENSITIES
                              datacube.m * datacube.ne_mw,datacube.m_HII * datacube.ne_HII_mw,\
                              datacube.m_HI * datacube.ne_HI_mw,datacube.m_H2 * datacube.ne_H2_mw,\
                              datacube.m * datacube.nH_mw,datacube.m_HII * datacube.nH_mw,\
                              datacube.m_HI * datacube.nH_mw,datacube.m_H2 * datacube.nH_mw,\
                              # VOLUME-WEIGHTED DENSITIES
                              datacube.cell_volume * datacube.nH,datacube.cell_volume * datacube.ne_vw))

    for line in p.lines:
        properties = np.column_stack((properties,datacube['L_%s' % line]/6,\
                                  datacube['L_%s_HII' % line]/6,\
                                  datacube['L_%s_HI' % line]/6,\
                                  datacube['L_%s_H2' % line]/6))
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
    area = cellarea
    for _ in range(len(p.moment0_dict.keys())-1):
        area = np.column_stack((area,cellarea))
    celldensity =  properties / area                  # units / kpc^2
    
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
           
            lineluminosity = np.nan_to_num(lineluminosity)
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
        
    location = moment0_map_location(res=res,plane=plane,gal_index=p.gal_index)
                
    np.save(location, momentmap)

def line_luminosity(tree, cell1, cell2, pixel, celldensity, R):
    '''
    This function returns the added up total luminosity of the pixel, used by convert_cell_data_to_regular_grid.
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
            #print(celldensity.shape)
            #pixelluminosity = crosssection * celldensity[0,int(ax1[i][0])]
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
""" For reading Cloudy output """
#---------------------------------------------------------------------------

def NII_from_logne(logne):

    p = copy.copy(params)

    cloudy_NII_ratio_ne()

    fit_params = np.load(p.d_table + 'cloudy/NII/fit_params_ne_NII.npy')

    return(np.tanh(logne*fit_params[0]+fit_params[1])*fit_params[2]+fit_params[3])

def logne_from_NII(NII_ratio):

    p = copy.copy(params)

    cloudy_NII_ratio_ne()

    fit_params = np.load(p.d_table + 'cloudy/NII/fit_params_ne_NII.npy')

    return((np.arctanh(((NII_ratio-fit_params[3])/fit_params[2]))-fit_params[1])/fit_params[0])

def cloudy_NII_ratio_ne():
    "Calculate [NII] ratio as function of electron density from 1-zone Cloudy models"

    p = copy.copy(params)

    NH,FUV,Z,DTM,nH = 20,0,0,-0.4,4

    # Fixed parameters:
    grid = pd.read_pickle(p.d_table + 'cloudy/NII/grid_table_z0_ahimsa')
    grid['index'] = np.arange(len(grid))
    grid['DTM'] = np.round(grid.DTM.values*10.)/10.

    print('For a cell in Cloudy with:')
    print('log NH = %s' % NH)
    print('log Z = %s' % Z)
    print('DTM = %s' % (10.**DTM))
    print('log nH = %s' % nH)

    # Make fit from hyperbolic tangent
    print('\nA hyperbolic tangent fit can be made to the [NII]-ne relation:')
    xfit = np.arange(-3,4,0.1)
    init_vals = 1.4,-4.8,5,5
    fit_params,covar = curve_fit(tanh_func,np.log10(grid['ne'].values),\
        grid['[NII]122'].values/grid['[NII]205'].values,p0=init_vals,maxfev=10000)
    np.save(p.d_table + 'cloudy/NII/fit_params_ne_NII',fit_params)


def tanh_func(x,a,b,c,d):

    return(np.tanh(x*a+b)*c+d)

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
    lsun_kkmspc2_const = c.L_sun.value * c.c.value**3 / c.k_B.value / (4.0 * np.pi * c.pc.to('m').value ** 2)

    L_K_km_s_pc2 = L_sun * 5e-4 * lsun_kkmspc2_const * (freq*1e9) ** (-3)

    # Fits with eqs from Solomon 1997 ApJ 478:
    L_K_km_s_pc2 = L_sun * 3.125e10 * freq ** (-3)
    #print(L_sun.min() * 3.125e10 * freq ** (-3))
    #print(L_sun.max() * 3.125e10 * freq ** (-3))

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

def nm_to_eV(nm):
    
    return(c.h.value * c.c.value / (nm*1e-9) * u.J.to('eV'))

def eV_to_micron(eV):
           
    return(c.h.value * c.c.value / (eV*u.eV.to('J')) * 1e6)


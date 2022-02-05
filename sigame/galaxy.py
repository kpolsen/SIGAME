"""
Module with classes to set up the main galaxy object, 
and carry out all tasks related to galaxies.
"""

# Import other SIGAME modules
import sigame.auxil as aux
import sigame.global_results as glo
import sigame.Cloudy_modeling as clo
# import sigame.plot as plot

# Import other modules
import numpy as np
import pandas as pd
import pdb as pdb
from scipy import interpolate
from scipy import spatial
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d,interp2d
from scipy.interpolate import RegularGridInterpolator,griddata
from scipy.spatial import cKDTree
import scipy.interpolate as interp
import matplotlib.cm as cm
import multiprocessing as mp
import subprocess as sub
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import astropy as astropy
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as c
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import os as os
import copy
from swiftsimio.visualisation import projection


#===============================================================================
###  Load parameters ###
#-------------------------------------------------------------------------------

global params
params                      =   aux.load_parameters()

#===========================================================================
### Main galaxy data classes ###
#---------------------------------------------------------------------------

class galaxy:
    """An object referring to one particular galaxy.

    Parameters
    ----------
    gal_index: int
        Galaxy index, default: 0

    Examples
    --------
    >>> import galaxy as gal
    >>> gal_ob = gal.galaxy(gal_index=0)

    """

    def __init__(self, gal_index, GR=None):

        p = copy.copy(params)

        if GR is None:
            # get global results
            GR                  =   glo.global_results()

        if p.verbose: print("constructing galaxy...")

        # grab some info from global results for this galaxy
        self.gal_index      =   gal_index
        self.gal_num        =   GR.gal_num[gal_index]
        self.radius         =   GR.R_gal[gal_index]
        self.M_star         =   GR.M_star[gal_index]
        self.M_gas          =   GR.M_gas[gal_index]
        self.R_gal          =   GR.R_gal[gal_index]
        self.R_max          =   GR.R_max[gal_index]
        self.name           =   GR.galnames[gal_index]
        self.zred           =   GR.zreds[gal_index]
        self.SFR            =   GR.SFR[gal_index]
        self.Zsfr           =   GR.Zsfr[gal_index]
        self.lum_dist       =   GR.lum_dist[gal_index]
        self.alpha          =   GR.alpha[gal_index]
        self.beta           =   GR.beta[gal_index]
        self.ang_dist_kpc   =   self.lum_dist*1000./(1+self.zred)**2

        # add objects
        self.add_attr('particle_data')
        self.add_attr('cell_data')

        if p.verbose: print("galaxy %s constructed.\n" % self.name)

    def add_attr(self,attr_name,verbose=False):
        """
        Adds either particle data or cell data as attribute to a galaxy object 
        """

        if hasattr(self, attr_name):
            if verbose: print("%s already has attribute %s" % (self.name,attr_name))
        else:
            if verbose: print("Adding %s attribute to %s ..." % (attr_name,self.name) )
            if attr_name == 'particle_data': ob = particle_data(self)
            if attr_name == 'cell_data': ob = cell_data(self)
            setattr(self,attr_name,ob)

class particle_data:
    """
    An object referring to the particle data for one galaxy (gas or stars)

    .. note:: Must be added as an attribute to a galaxy object.

    :param gal_ob: Instance of galaxy class
    :type gal_ob: object
        
    :param verbose: Print a lot of info (True) or less (False) during run time of the code
    :type verbose: bool

    >>> import galaxy as gal
    >>> gal_ob = gal.galaxy(gal_index=0)
    >>> simgas = gal_ob.particle_data.get_dataframe('simgas')
    >>> simstar = gal_ob.particle_data.get_dataframe('simstar')

    """

    def __init__(self,gal_ob,**kwargs):

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        if p.verbose: print("constructing particle_data object...")

        # add galaxy
        self.gal_ob             =   gal_ob

        if p.verbose: print("particle_data object constructed for %s.\n" % gal_ob.name)


    def _add_data(self,data_type,d_data=''):
        """ 
        Adds particle data as dataframe ("df" attribute) to particle_data object.
        """

        df                      =   aux.load_temp_file(gal_ob=self.gal_ob,data_type=data_type,d_data=d_data)
        setattr(self,data_type,df)

    def get_dataframe(self,data_type,d_data=''):
        """ 
        Returns dataframe with particle data for one galaxy.
        """

        self._add_data(data_type,d_data=d_data)
        return(getattr(self,data_type))

    def save_dataframe(self,data_type):
        """ 
        Saves particle data as dataframe to file.
        """

        aux.save_temp_file(getattr(self,data_type),gal_ob=self.gal_ob,data_type=data_type)

class cell_data:
    """
    An object referring to the gas in cell data format for one galaxy

    .. note:: Is an attribute to a galaxy object.

    :param gal_ob: Instance of galaxy class
    :type gal_ob: object
        
    :param verbose: Print a lot of info (True) or less (False) during run time of the code
    :type verbose: bool

    >>> import galaxy as gal
    >>> gal_ob       =   gal.galaxy(gal_index)
    >>> cell_data    =   gal_ob.cell_data.get_dataframe()

    """

    def __init__(self,gal_ob,**kwargs):

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        # add galaxy
        self.gal_ob =   gal_ob

        if p.verbose: print("constructing cell_data object...")

    def start_dataframe(self,**kwargs):
        """
        Creates a dataframe with cell data from SKIRT output for one galaxy.
        Called by isrf.read_skirt_output().
        """

        print('Creating dataframe with SKIRT cell data output')

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        # From SKIRT
        cellprops       =   self._read_cellprops()
        x               =   cellprops['x'].values/1e3 # kpc
        y               =   cellprops['y'].values/1e3 # kpc
        z               =   cellprops['z'].values/1e3 # kpc
        # n_e              =   cellprops['n_e'].values # cm^-3
        n_dust          =   cellprops['n_dust'].values # Msun/pc3
        cell_volume     =   cellprops['V'].values # pc^3
        cell_size       =   cellprops['V'].values**(1/3)/1e3 # from cell size [kpc]

        # Put into dataframe
        index           =   np.arange(len(x))
        df              =   pd.DataFrame({'x':x,'y':y,'z':z,'n_dust':n_dust,\
                            'cell_size':cell_size,'cell_volume':cell_volume},index=index)
        self.df         =   df
       
        # save dataframe
        self.save_dataframe()

    def _get_name(self):
        """
        Creates/gets file name where cell data of one galaxy is stored.
        """

        p = copy.copy(params)

        name = '%s%s_G%i' % (p.sim_name,p.sim_run,self.gal_ob.gal_index)
   
        return name

    def save_dataframe(self,**kwargs):
        """
        Saves cell data of one galaxy to file.
        """

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        aux.save_temp_file(self.df,gal_ob=self.gal_ob,data_type='cell_data')

    def _add_data(self):
        """ 
        Adds cell data as dataframe attribute to cell_data object.
        """

        self.df         =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='cell_data')

    def get_dataframe(self):
        """ 
        Adds cell data as dataframe attribute to cell_data object and returns dataframe..
        """

        if not hasattr(self,'df'): self._add_data()
        return(getattr(self,'df'))

    def _interpolate_cloudy_cell_table(self,lookup_table,key,cell_prop): 
        """ 
        Interpolates in Cloudy look-up table for all cells in a galaxy.
        Called by self._do_interpolation_cells().
        """

        lognHs = np.unique(lookup_table.lognHs)
        lognSFRs = np.unique(lookup_table.lognSFRs)
        logNHs = np.unique(lookup_table.logNHs)
        logFUVs = np.unique(lookup_table.logFUVs)
        logZs = np.unique(lookup_table.logZs)
        logDTMs = np.unique(lookup_table.logDTMs)

        grid = np.zeros([len(lognHs),len(lognSFRs),len(logNHs),len(logFUVs),len(logZs),len(logDTMs)])
        i                           =   0
        for i1 in range(0,len(lognHs)):
            for i2 in range(0,len(lognSFRs)):
                for i3 in range(0,len(logNHs)):
                    for i4 in range(0,len(logFUVs)):
                        for i5 in range(0,len(logZs)):
                            for i6 in range(0,len(logDTMs)):
                                grid[i1,i2,i3,i4,i5,i6]    =   lookup_table[key][i]
                                i                            +=  1
        interp                      =   RegularGridInterpolator((lognHs,lognSFRs,logNHs,logFUVs,logZs,logDTMs), grid)

        try:
            result = interp(cell_prop)
        except:
            pdb.set_trace()

        return(result)

    def _do_interpolation_cells(self):
        """ 
        Interpolates in Cloudy look-up table for all cells in a galaxy.
        """

        print('\nNow interpolating in gas cells for galaxy # %i'  % self.gal_ob.gal_index)

        GR                      =   glo.global_results()
        p = copy.copy(params)

        df_or                   =   self.get_dataframe()

        if 'DTM' not in df_or.keys():
            self._add_DTM()
        self.save_dataframe()
        if 'vx' not in df_or.keys():
            self._add_velocity()
        self.save_dataframe()
        df_or                   =   self.get_dataframe()

        df_or = df_or.copy()

        # Make sure we don't go outside of grid:
        print(np.min(df.nSFR.values))
        print(np.max(df.nSFR.values))
        print(np.min(df.nH.values))
        print(np.max(df.nH.values))


        df.Z[np.isnan(df.Z)]    =   1e-6
        df.nSFR[df.nSFR == 0]    =   1e-30 
        df.nSFR[np.isnan(df.nSFR)]    =   -30 

        # IMPORTANT: Taking the log of the entire dataframe and re-naming columns
        df                      =   np.log10(df)
        df = df.rename(columns={'nH':'lognH','Z':'logZ','G0':'logFUV','nSFR':'lognSFR','DTM':'logDTM'})

        if 'NH' not in df_or.keys():
            # Convert OIR/FUV ratio to column density NH
            logNH_cl,R_OIR_FUV_cl = aux.get_NH_from_cloudy()
            dR_OIR_FUV_cl = R_OIR_FUV_cl / R_OIR_FUV_cl.min()
            R_OIR_FUV_df = df_or.R_OIR_FUV.values
            dR_OIR_FUV_df = R_OIR_FUV_df / R_OIR_FUV_df.min()
     
            interp                  =   interp1d(np.log10(R_OIR_FUV_cl),logNH_cl,fill_value='extrapolate',kind='slinear')
            df['logNH']             =   interp(np.log10(R_OIR_FUV_df))
            df['logNH'][df.logNH <= np.min(logNH_cl)] = np.min(logNH_cl)
            df['logNH'][df.logNH >= np.max(logNH_cl)] = np.max(logNH_cl)
            df['logNH'][np.isinf(R_OIR_FUV_df)] = np.max(logNH_cl)
            df_or['NH']             =   10.**df.logNH
        else:
            df['logNH'] = np.log10(df_or['NH'])

        self.df = df_or

        print('G%i - range in logNH: %.2e to %.2e' % (self.gal_ob.gal_index,np.min(df.logNH),np.max(df.logNH)))

        # Cloudy lookup table
        cloudy_library = clo.library()
        lookup_table = cloudy_library._restore_lookup_table()
        lognHs = np.unique(lookup_table.lognHs)
        lognSFRs = np.unique(lookup_table.lognSFRs)
        logNHs = np.unique(lookup_table.logNHs)
        logFUVs = np.unique(lookup_table.logFUVs)
        logZs = np.unique(lookup_table.logZs)
        logDTMs = np.unique(lookup_table.logDTMs)

        # Check DTM values for nans...
        DTM = df_or.DTM.values
        DTM[np.isnan(DTM)] = 10.**np.min(logDTMs)
        DTM[DTM == 0] = 1e-30
        df['logDTM'] = np.log10(DTM)

        # Make sure that cell data doesn't exceed look-up table values
        for _ in p.interp_params:
            df[_][df[_] <= np.min(lookup_table[_+'s'].values)] = np.min(lookup_table[_+'s'].values) + 1e-6 * np.abs(np.min(lookup_table[_+'s']))
            df[_][df[_] >= np.max(lookup_table[_+'s'].values)] = np.max(lookup_table[_+'s'].values) - 1e-6 * np.abs(np.min(lookup_table[_+'s']))

        # Cell properties used for interpolation in cloudy grid models:
        lognH = df.lognH.values; lognH[np.isnan(lognH)] = np.min(lookup_table['lognHs'].values) + 1e-6 * np.abs(np.min(lookup_table[_+'s']))
        # check for nans in lognH:
        cell_prop               =   np.column_stack((lognH,df.lognSFR.values,df.logNH.values,df.logFUV.values,df.logZ.values,df.logDTM.values))        

        # New dataframe to fill with interpolation results
        cell_prop_new           =   df_or.copy()
        
        ### LINE EMISSION
        for target in p.lines:
            cell_prop_new['L_'+target] = self._interpolate_cloudy_cell_table(lookup_table,target,cell_prop)
            # Scale by H mass of that cell (each look-up table entry is for 1e4 Msun H mass):
            line_lum   =   10.**cell_prop_new['L_'+target].values*cell_prop_new.mH.values/1e4 
            # Only count cells with actual hydrogen mass
            line_lum[df_or.nH == 0] = 0
            cell_prop_new['L_'+target]   =   line_lum
            df['L_%s_sun' % target] = np.sum(line_lum)
            print('G%i - %s: %.2e Lsun in cells' % (self.gal_ob.gal_index,target,np.sum(line_lum)))
            # Scale to 1e4 Msun gas mass, as 1e4 Msun was assumed in Cloudy_modeling.sample_cloudy_models()
            for phase in ['HII','HI','H2']:
                line_lum = self._interpolate_cloudy_cell_table(lookup_table,target+'_'+phase,cell_prop)
                cell_prop_new['L_'+target+'_'+phase]   =   10.**line_lum*cell_prop_new.mH/1e4 
                df['L_%s_%s_sun' % (target,phase)] = np.sum(line_lum)

        ### CLOUDY CELL VOLUME
        if 'cell_size_lookup' not in cell_prop_new.keys():
            V_grid = self._interpolate_cloudy_cell_table(lookup_table,'V',cell_prop)
            cell_prop_new['cell_size_lookup']      =   (V_grid)**(1/3) / 1e3 # kpc

        ### DENSE MASS FRACTIONS
        if 'mf_1e3_grid' not in cell_prop_new.keys():
            cell_prop_new['mf_1e3_grid'] = self._interpolate_cloudy_cell_table(lookup_table,'mf_1e3',cell_prop)
            cell_prop_new['mf_1e1_grid'] = self._interpolate_cloudy_cell_table(lookup_table,'mf_1e1',cell_prop)
            print('G%i - mass fraction at nH > 1e3 cm^-3: %.3e %% ' % (self.gal_ob.gal_index,np.sum(cell_prop_new.m * cell_prop_new.mf_1e3_grid)/np.sum(cell_prop_new.m)*100.))

        ### TEMPERATURE
        Te_mw = self._interpolate_cloudy_cell_table(lookup_table,'Te_mw',cell_prop)
        Tk_mw = self._interpolate_cloudy_cell_table(lookup_table,'Tk_mw',cell_prop)
        cell_prop_new['Tk_mw'] = Tk_mw

        ### HYDROGEN IONIZATION FRACTIONS AND ELECTRON DENSITIES (remove "GRID" at some point)
        cell_prop_new['mf_H2_grid'] = self._interpolate_cloudy_cell_table(lookup_table,'mf_H2',cell_prop)
        cell_prop_new['mf_HII_grid'] = self._interpolate_cloudy_cell_table(lookup_table,'mf_HII',cell_prop)
        cell_prop_new['mf_HI_grid'] = self._interpolate_cloudy_cell_table(lookup_table,'mf_HI',cell_prop)

        ### ALPHA_CO
        m_H2 = cell_prop_new['mf_H2_grid'].values*cell_prop_new['m'].values
        L_CO = cell_prop_new['L_CO(1-0)'].values
        L_CO = aux.Lsun_to_K_km_s_pc2(L_CO,'CO(1-0)')
        cell_prop_new['alpha_CO'] = m_H2/L_CO
        print('L_CO(1-0), ', L_CO.min(),L_CO.max())
        print('alpha_CO: ',cell_prop_new['alpha_CO'].min(),cell_prop_new['alpha_CO'].max())
        print('Total alpha_CO: %.4f' % (np.sum(m_H2)/np.sum(L_CO)))

        ### HYDROGEN AND ELECTRON DENSITIES
        ne_mw = self._interpolate_cloudy_cell_table(lookup_table,'ne_mw',cell_prop) # not correct w HII regions
        ne_vw = self._interpolate_cloudy_cell_table(lookup_table,'ne_vw',cell_prop) # not correct w HII regions
        nH_mw = self._interpolate_cloudy_cell_table(lookup_table,'nH_mw',cell_prop) # not correct w HII regions
        cell_prop_new['ne_mw'] = ne_mw
        cell_prop_new['ne_vw'] = ne_vw
        cell_prop_new['nH_mw'] = nH_mw
        ne_HII_mw = self._interpolate_cloudy_cell_table(lookup_table,'ne_HII_mw',cell_prop)
        nH_HII_mw = self._interpolate_cloudy_cell_table(lookup_table,'ne_HII_mw',cell_prop)
        cell_prop_new['ne_HII_mw'] = ne_HII_mw
        cell_prop_new['nH_HII_mw'] = nH_HII_mw

        ne_HI_mw = self._interpolate_cloudy_cell_table(lookup_table,'ne_HI_mw',cell_prop)
        nH_HI_mw = self._interpolate_cloudy_cell_table(lookup_table,'nH_HI_mw',cell_prop)
        cell_prop_new['ne_HI_mw'] = ne_HI_mw
        cell_prop_new['nH_HI_mw'] = nH_HI_mw

        ne_H2_mw = self._interpolate_cloudy_cell_table(lookup_table,'ne_H2_mw',cell_prop)
        nH_H2_mw = self._interpolate_cloudy_cell_table(lookup_table,'nH_H2_mw',cell_prop)
        cell_prop_new['ne_H2_mw'] = ne_H2_mw
        cell_prop_new['nH_H2_mw'] = nH_H2_mw

        df['M_H2'] = np.sum( cell_prop_new['mf_H2_grid'] * cell_prop_new['m'])
        df['M_HII'] = np.sum( cell_prop_new['mf_HII_grid'] * cell_prop_new['m'])
        df['M_HI'] = np.sum( cell_prop_new['mf_HI_grid'] * cell_prop_new['m'])

        # Only within half-mass gas radius...
        R2_gas = getattr(GR,'R2_gas')[self.gal_ob.gal_index]
        r = np.sqrt( cell_prop_new['x']**2 + cell_prop_new['y']**2 + cell_prop_new['z']**2)
        df['M_H2_R2_gas'] = np.sum(cell_prop_new['mf_H2_grid'][r < R2_gas] * cell_prop_new['m'][r < R2_gas] )
        df['M_HI_R2_gas'] = np.sum(cell_prop_new['mf_HI_grid'][r < R2_gas] * cell_prop_new['m'][r < R2_gas] )
        df['M_HII_R2_gas'] = np.sum(cell_prop_new['mf_HII_grid'][r < R2_gas] * cell_prop_new['m'][r < R2_gas] )

        ### PRESSURE
        P_e_mw = cell_prop_new['Te_mw']*cell_prop_new['ne_mw']   
        He_H = p.elements['He']['[n_i/n_H]']
        P_mw = (cell_prop_new['ne_mw'] + cell_prop_new['nH_mw']*(1+He_H)) * cell_prop_new['Tk_mw']   
        P_mw2 = cell_prop_new['ne_mw']*cell_prop_new['Te_mw'] + cell_prop_new['nH_mw']*(1+He_H) * cell_prop_new['Tk_mw']   
        cell_prop_new['P_e_mw'] =   P_e_mw
        cell_prop_new['P_mw'] =  P_mw
        cell_prop_new['P_mw2'] =  P_mw2

        self.df                 =   cell_prop_new
        self.save_dataframe()
        df = pd.DataFrame({})
        for line in p.lines:        
            L_line = cell_prop_new['L_'+line+''].values
            df['L_%s' % line] = L_line.sum()
        df.to_pickle('data/results/temp/G%i' % self.gal_ob.gal_index)

        print('done with interpolation for galaxy # %i!' % (self.gal_ob.gal_index))

    def _read_cellprops(self,**kwargs):
        """ 
        Reads cell properties from SKIRT and store in dataframe.
        """

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        df              =   pd.read_csv(p.d_XL_data + 'skirt/' + self._get_name()+'_scp_cellprops'+'.dat',skiprows=9,sep=' ',engine='python',\
            names=['i','x','y','z','V','opt_depth','n_dust','ne','nH'],\
            dtype={'i':'Int64','x':'float64','y':'float64','z':'float64','V':'float64',\
           'opt_depth':'float64','n_dust':'float64','n_e':'float64','nH':'float64'})
        return(df)    

    def _add_nH(self):
        """ 
        Calculates hydrogen density per cell from nearby fluid elements.
        """

        p = copy.copy(params)

        df                  =   self.df.copy()
        coords_cells        =   df[['x','y','z']].values
 
        # Get sim data for gas
        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='simgas')
        coords_sim          =   simgas[['x','y','z']].values

        # Find particles/cell closest to each SKIRT cell
        if p.sim_type == 'sph': indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=64)[1]
        if p.sim_type == 'amr': indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=1)[1]

        # Select/derive nH for each SKIRT cell
        if p.sim_type == 'sph':
            rho                  =   np.zeros(len(df))
            print('max nH in simgas: %.2f' % (np.max(simgas.nH)))
            for i in range(len(df)):
                # coordinates of this cell
                xyz_i = coords_cells[i]
                
                # distance to neighboring sim particles
                dx = coords_sim[indices[i]] - xyz_i
                r = np.sqrt(np.sum(dx * dx, axis=1))
                
                # properties of neighboring sim particles
                simgas_cut = simgas.iloc[indices[i]].reset_index(drop=True)
                m = simgas_cut['m']; h = simgas_cut['h']
     
                # mass-weighted average metallicity in this region, using SPH formalism:
                kernel = [aux.Wendland_C2_kernel(r1,h1) for r1,h1 in zip(r,h)]
                rho[i] = sum(m * kernel) # Msun/kpc^3, estimate of density, not from snapshot
            df['rho']   =   rho # Msun/kpc^3
            df['m']     =   rho * df['cell_volume'].values/1e9 # Msun 
            df['mH']    =   3/4 * df['m'].values # Msun (assuming He mass fraction of 25%)
            df['nH']    =   df['mH'].values * p.Msun / p.mH / (df['cell_volume'].values*p.pc2cm**3) # Msun/kpc^3 -> H/cm^-3
        if p.sim_type == 'amr':
            df['nH'] = simgas.nH.values[indices]
            df['mH'] = df['nH'].values * p.mH * (df['cell_volume'].values*p.pc2cm**3) / p.Msun # Msun/kpc^3 -> H/cm^-3
            df['m'] = 4/3 * df['mH'].values # Msun (assuming He mass fraction of 25%)
            df['rho'] = df['m'] / (df['cell_volume'].values/1e9) # Msun/kpc^3 

        print('max nH in cell_data: %.2f' % (np.max(df.nH)))
        print('min nH in cell_data: %.2f' % (np.min(df.nH)))
        print('Total mass in simgas: %.2e' % (np.sum(simgas.m)))
        print('Total mass in cell_data: %.2e' % (np.sum(df.m)))
        self.df             =   df 

    def _add_FIR_flux(self,fluxOutputStyle="Wavelength"):
        """ 
        Calculates FIR flux in each cell from SKIRT radiation field probe.

        :param fluxOutputStyle: Units of flux in each cell, defaults to "Wavelength" corresponding to (W/m2/micron)
        :type fluxOutputStyle: str, optional
        """

        p = copy.copy(params)

        self._add_data()
        df                  =   self.df

        wavelengths,bin_width = aux.read_probe_wavelengths(self._get_name())
        Nbins               =   len(wavelengths)

        N_start,N_stop      =   aux.FIR_index(wavelengths)
        if fluxOutputStyle == "Wavelength":     
            df['F_FIR_W_m2']        =   np.array([np.trapz(I_W_m2_micron[i,N_start:N_stop],x=wavelengths[N_start:N_stop]) \
                                            for i in range(len(df))])
 
        self.df             =   df

    def _add_FUV_flux(self,fluxOutputStyle="Wavelength"):
        """ 
        Calculates FUV flux in each cell from SKIRT radiation field probe.

        :param fluxOutputStyle: Units of flux in each cell, defaults to "Wavelength" corresponding to (W/m2/micron)
        :type fluxOutputStyle: str, optional
        """

        p = copy.copy(params)

        self._add_data()
        df                  =   self.df

        if ('F_UV_W_m2' not in df.keys()) | (p.ow == True):
            # Read probe wavelengths
            wavelengths,bin_width = aux.read_probe_wavelengths(self._get_name())
            Nbins               =   len(wavelengths)

            # Read probe intensities in W/m2/micron/sr
            I_W_m2_micron_sr    =   np.array(aux.read_probe_intensities(self._get_name(),Nbins))

            # Convert intensities to W/m2/micron
            I_W_m2_micron       =  I_W_m2_micron_sr * 4 * np.pi

            # Integrate intensities in OIR
            N_start,N_stop      =   aux.OIR_index(wavelengths)
            if fluxOutputStyle == "Wavelength":     
                df['F_OIR_W_m2']        =   np.array([np.trapz(I_W_m2_micron[i,N_start:N_stop],x=wavelengths[N_start:N_stop]) \
                                                for i in range(len(df))])
            
            # Integrate intensities in FUV
            N_start,N_stop      =   aux.FUV_index(wavelengths)
            if fluxOutputStyle == "Wavelength":     
                df['F_FUV_W_m2']        =   np.array([np.trapz(I_W_m2_micron[i,N_start:N_stop],x=wavelengths[N_start:N_stop]) \
                                                for i in range(len(df))])

            # Integrate intensities in UV
            N_start,N_stop      =   aux.UV_index(wavelengths)
            if fluxOutputStyle == "Wavelength":     
                df['F_UV_W_m2']        =   np.array([np.trapz(I_W_m2_micron[i,N_start:N_stop],x=wavelengths[N_start:N_stop]) \
                                                for i in range(len(df))])

        df['UV_to_FUV']     =   df['F_UV_W_m2'].values/df['F_FUV_W_m2'].values
        R_OIR_FUV_df        =   df['F_OIR_W_m2'].values/df['F_FUV_W_m2'].values
        R_OIR_FUV_min = np.min(R_OIR_FUV_df)

        # Storing OIR/FUV flux ratio here so it doesn't get overwritten
        df['R_OIR_FUV']         =   df['F_OIR_W_m2'].values/df['F_FUV_W_m2'].values

        # Normalize to Habing flux (1.6e-3 erg/cm^2/s)
        df['F_FUV_Habing']      =   df['F_FUV_W_m2'].values * 1e7 / 1e4 / 1.6e-3

        # Normalize to G0 energy density (5.29e-14 ergs/cm^3)
        # http://www.ita.uni-heidelberg.de/~rowan/ISM_lectures/galactic-rad-fields.pdf eq. 18
        df['E_FUV_ergs_cm3']    =   df['F_FUV_W_m2'].values / p.clight / 1e-7 / 1e6
        df['G0']                =   df['E_FUV_ergs_cm3'].values / 5.29e-14 # ergs/cm^3 from Peter Camps

        self.df             =   df

    def _add_DTM(self):
        """ 
        Calculates dust-to-metal ratio per cell from nearby fluid elements.
        """


        p = copy.copy(params)

        # Make sure cell data from SKIRT is loaded
        self._add_data()
        df                  =   self.df.copy()
        coords_cells        =   df[['x','y','z']].values
        
        # Get particle data for gas
        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='simgas')
        coords_sim          =   simgas[['x','y','z']].values

        # Find particles closest to each cell
        if p.sim_type == 'sph': indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=64)[1]
        if p.sim_type == 'amr': indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=1)[1]

        # Pick the nearest sim particle
        DTM                 =   np.zeros(len(df))
 
        if p.sim_type == 'sph': 
            for i in range(len(df)):
                # coordinates of this cell
                xyz_i = coords_cells[i]
                
                # distance to neighboring sim particles
                dx = coords_sim[indices[i]] - xyz_i
                r = np.sqrt(np.sum(dx * dx, axis=1))
                
                # properties of neighboring sim particles
                simgas_cut = simgas.iloc[indices[i]].reset_index(drop=True)
                DTM_cut = simgas_cut['DTM']
     
                # Just take nearest cell
                DTM[i] = DTM_cut[np.argmin(r)]

        if p.sim_type == 'amr': 
            print('Sim type is AMR: fixing DTM at 0.4')
            DTM = np.zeros(len(df)) + 0.4

        df['DTM']           =   DTM

        self.df             =   df

    def _add_velocity(self):
        """ 
        Calculates velocity (vx,vy,xz) per cell from nearby fluid elements.
        """

        p = copy.copy(params)

        # Make sure cell data from SKIRT is loaded
        self._add_data()
        df                  =   self.df.copy()
        coords_cells        =   df[['x','y','z']].values
        
        # Get particle data for gas
        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='simgas')
        coords_sim          =   simgas[['x','y','z']].values

        # Find particles closest to each cell
        if p.sim_type == 'sph': indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=64)[1]
        if p.sim_type == 'amr': indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=1)[1]

        # Pick the nearest sim particle
        vx                  =   np.zeros(len(df))
        vy                  =   np.zeros(len(df))
        vz                  =   np.zeros(len(df))
 
        if p.sim_type == 'sph': 
            for i in range(len(df)):
                # coordinates of this cell
                xyz_i = coords_cells[i]
                
                # distance to neighboring sim particles
                dx = coords_sim[indices[i]] - xyz_i
                r = np.sqrt(np.sum(dx * dx, axis=1))
                
                # properties of neighboring sim particles
                simgas_cut = simgas.iloc[indices[i]].reset_index(drop=True)
                vx_cut = simgas_cut['vx']
                vy_cut = simgas_cut['vy']
                vz_cut = simgas_cut['vz']
     
                # Just take nearest cell
                vx[i] = vx_cut[np.argmin(r)]
                vy[i] = vy_cut[np.argmin(r)]
                vz[i] = vz_cut[np.argmin(r)]

        #if p.sim_type == 'amr': 
        #    pass

        df['vx']           =   vx
        df['vy']           =   vy
        df['vz']           =   vz

        self.df             =   df

    def _add_metallicity(self):
        """ 
        Calculates metallicity per cell from nearby fluid elements.
        """

        p = copy.copy(params)

        # Make sure cell data from SKIRT is loaded
        self._add_data()
        df                  =   self.df.copy()
        coords_cells        =   df[['x','y','z']].values
        
        # Get particle data for gas
        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='simgas')
        coords_sim          =   simgas[['x','y','z']].values

        # Find particles closest to each cell
        if p.sim_type == 'sph': indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=64)[1]
        if p.sim_type == 'amr': indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=1)[1]

        Z                   =   np.zeros(len(df))
        #Z2                  =   np.zeros(len(df))
 
        if p.sim_type == 'sph': 
            for i in range(len(df)):
                # coordinates of this cell
                xyz_i = coords_cells[i]
                
                # distance to neighboring sim particles
                dx = coords_sim[indices[i]] - xyz_i
                r = np.sqrt(np.sum(dx * dx, axis=1))
                
                # properties of neighboring sim particles
                simgas_cut = simgas.iloc[indices[i]].reset_index(drop=True)
                m = simgas_cut['m']; h = simgas_cut['h']; Z_cut = simgas_cut['Z']
     
                # mass-weighted average metallicity in this region, using SPH formalism:
                kernel = [aux.Wendland_C2_kernel(r1,h1) for r1,h1 in zip(r,h)]
                dens_i = sum(m * kernel) # estimate of density, not from snapshot
                Z[i] = np.sum(m*Z_cut*kernel)/dens_i
     
                # Take nearest cell if no neighbors found within h
                if dens_i == 0:
                    Z[i] = Z_cut[np.argmin(r)]
                #Z2[i] = Z_cut[np.argmin(r)]

        if p.sim_type == 'amr': 
            Z = simgas.Z.values[indices]
            #for i in range(len(df)):
            #    Z[i] = simgas.Z.values[indices]


        df['Z'] =   Z

        self.df             =   df

    def _add_SFR_density(self):
        """ 
        Calculates SFR density per cell from nearby fluid elements.
        """

        # Make sure cell data from SKIRT is loaded
        self._add_data()
        df                  =   self.df.copy()
        coords_cells        =   df[['x','y','z']].values
        V_cells             =   df[['cell_volume']].values
        
        # Get particle data for gas
        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='simgas')
        #simgas['nSFR']      =   simgas['SFR'] / (4/3*np.pi*simgas.h.values**3)
        coords_sim          =   simgas[['x','y','z']].values

        # Find particles closest to each cell
        indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=64)[1]

        nSFR    = np.zeros(len(df))
        for i in range(len(df)):
            xyz_i = coords_cells[i]
            dx = coords_sim[indices[i]]- xyz_i
            r = np.sqrt(np.sum(dx * dx,axis=1))
            simgas_cut = simgas.iloc[indices[i][r < 0.2]].reset_index(drop=True)
            if len(simgas_cut) >= 1:
                nSFR[i] = np.sum(simgas_cut['SFR'].values/0.2**3) # Mstar/yr/kpc^3

        df['SFR_density'] =   nSFR

        df['nSFR'] =   nSFR

        self.df             =   df

#---------------------------------------------------------------------------
### FOR ISRF TASK ###
#---------------------------------------------------------------------------

def setup_SKIRT(gal_indices):
    """
    A function that creates SKIRT input, using input from isrf classs.
    Called by backend.py.

    :param gal_indices: List of galaxy indices to loop over.
    :type gal_indices: list
    """

    p = copy.copy(params)

    GR                  =   glo.global_results()

    for gal_index in gal_indices:

        print('\n- Setting up SKIRT for galaxy # %i ' % (gal_index))

        # Load gas and star particles
        isrf_ob             =   isrf(GR=GR, gal_index=gal_index)
        simgas              =   isrf_ob.particle_data.get_dataframe('simgas')
        simstar             =   isrf_ob.particle_data.get_dataframe('simstar')
     
        # Select distance to put galaxy
        isrf_ob._set_distance_to_galaxy()

        if p.sim_type == 'amr': setattr(isrf_ob,'R_max',np.abs(simgas.x.values).max())
        print(p.R_max)
     
        # Save gas data in SKIRT format
        simgas_skirt = simgas.copy()
        simgas_skirt[['x','y','z']] = simgas_skirt[['x','y','z']]*1000. # pc
        skirt_filename = p.d_skirt + '%s_gas.dat' % isrf_ob._get_name()
        if p.sim_type == 'amr':
            # https://skirt.ugent.be/skirt9/class_cell_medium.html
            header = '# Gas Cells\n'+\
                        '# Columns contain: xmin(pc) ymin(pc) zmin(pc) xmax(pc) ymax(pc) zmax(pc) M(Msun) Z(0-1)'
            simgas_skirt['Z'] = simgas_skirt['Z']*p.mf_Z1 # Asplund+09, metallicity to SKIRT is mass fraction...
            simgas_skirt['Z'] = simgas_skirt['Z'].map(lambda x: '%.6e' % x)
            # Get cell boundaries
            grid_levels = np.unique(simgas.grid_level)
            dxs = np.sort(np.unique(simgas.dx))[::-1]
            xl,yl,zl = simgas.copy().x.values, simgas.y.copy().values, simgas.z.copy().values # lower left corners
            xu,yu,zu = simgas.copy().x.values, simgas.y.copy().values, simgas.z.copy().values # upper right corners
            for grid_level,dx in zip(grid_levels,dxs):
                print('now grid level %i' % grid_level)
                print('with size: %.4f kpc' % dx)
                index = np.where(simgas.grid_level.values == grid_level)[0]
                xl[index] = xl[index] - dx/2.
                yl[index] = yl[index] - dx/2.
                zl[index] = zl[index] - dx/2.
                xu[index] = xu[index] + dx/2.
                yu[index] = yu[index] + dx/2.
                zu[index] = zu[index] + dx/2.
            df   =  pd.DataFrame({\
               'xmin':xl*1e3,\
               'ymin':yl*1e3,\
               'zmin':zl*1e3,\
               'xmax':xu*1e3,\
               'ymax':yu*1e3,\
               'zmax':zu*1e3,\
               'm':simgas.m,\
               'Z':simgas.Z * p.mf_Z1 }) 
            df.to_csv(skirt_filename,header=False,sep=' ',index=False,float_format='%.4e')
        if p.sim_type == 'sph':
            simgas_skirt[['h']] = simgas_skirt[['h']]*1000. # pc
            if 'DTM' in p.sim_run: 
                # Let's just use Z and assume as DTM ratio
                header = '# SPH Gas Particles\n'+\
                            '# Columns contain: x(pc) y(pc) z(pc) h(pc) M(Msun) Z(0-1)'
                simgas_skirt['Z'] = simgas_skirt['Z']*p.mf_Z1 # Asplund+09, metallicity to SKIRT is mass fraction...
                simgas_skirt['Z'] = simgas_skirt['Z'].map(lambda x: '%.6e' % x)
                simgas_skirt = simgas_skirt[['x','y','z','h','m','Z']]
            else:
                # Calculate mass-weighted DTM ratio (total metal mass as in Dave+19)
                simgas['DTM']     =   simgas.m_dust.values/(simgas.m_dust.values + simgas.Z.values * p.mf_Z1 * simgas.m.values)
                isrf_ob.particle_data.simgas = simgas
                isrf_ob.particle_data.save_dataframe('simgas')
                mw_DTM            =   np.sum(simgas.DTM * simgas.m.values)/np.sum(simgas.m.values) 

                for col in ['x','y','z','h','m_dust']:
                    simgas_skirt[col] = simgas_skirt[col].map(lambda x: '%.2f' % x)
                header = '# SPH Gas Particles\n'+\
                            '# Columns contain: x(pc) y(pc) z(pc) h(pc) M(Msun)'
                simgas_skirt = simgas_skirt[['x','y','z','h','m_dust']]
            simgas_skirt.to_csv(skirt_filename,header=False,index=False,sep=' ')
        def line_prepender(filename, line):
            with open(filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(line.rstrip('\r\n') + '\n' + content)
        line_prepender(skirt_filename,header)
     
        # Save star data in SKIRT format
        simstar_skirt = simstar.copy()
        # Convert current mass to initial mass for stars (table takes age in log(yr))
        df = pd.read_pickle(p.d_table+'fsps/z%i_age_mass_grid' % p.zred)
     
        # STARS
        header = '# Star Particles\n'+\
                        '# Columns contain: x(pc) y(pc) z(pc) h(pc) M(Msun) Z age(yr)'
        simstar_skirt[['x','y','z']] = simstar_skirt[['x','y','z']]*1000. # pc
        simstar_skirt['age'] = simstar_skirt['age']*1e9 # yr
        simstar_skirt['m_init'] = simstar_skirt['m'].values / np.interp(np.log10(simstar_skirt['age'].values),df['age'].values,df['mass_remaining'].values) 
        for col in ['x','y','z','m_init']:
            simstar_skirt[col] = simstar_skirt[col].map(lambda x: '%.2f' % x)
        simstar_skirt['Z'] = simstar_skirt['Z']*0.0134 # Asplund+09, metallicity to SKIRT is mass fraction...
        simstar_skirt['Z'] = simstar_skirt['Z'].map(lambda x: '%.6e' % x)
        # Simple estimate of stellar h: https://github.com/SKIRT/SKIRT9/issues/10
        m1,m2 = np.min(simstar['m']),np.max(simstar['m'])
        simstar_skirt['h'] = (simstar['m']-m1)/(m2-m1)*(300-100) + 100
        if p.sim_name == 'enzo': simstar_skirt['h'] = (simstar['m']-m1)/(m2-m1)*(500-200) + 300

        # Save old stars for Bruzual&Charlot
        skirt_filename = p.d_skirt + '%s_star_old.dat' % isrf_ob._get_name()
        simstar_skirt_old = simstar_skirt.copy()
        print('Max age: ',simstar_skirt.age.max())
        print('Min age: ',simstar_skirt.age.min())
        simstar_skirt_old['age'] = simstar_skirt_old['age'].map(lambda x: '%.6e' % x)
        simstar_skirt_old[['x','y','z','h','m_init','Z','age']].to_csv(skirt_filename,header=False,index=False,sep=' ')
        def line_prepender(filename, line):
            with open(filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(line.rstrip('\r\n') + '\n' + content)
        line_prepender(skirt_filename,header)   

        # Edit SKIRT input file
        ski_template            =   open(p.d_table+'skirt/skirt_template_%s.ski' % p.sim_type,'r')
        try:
            os.remove(p.d_skirt+"G%i.ski" % gal_index)
        except:
            pass
        ski_copy = open(p.d_skirt+'%s%s_G%i.ski' % (p.sim_name,p.sim_run,gal_index),'w')
        print('(using galaxy radius for 1/2 * image size: %.2f kpc)' % (isrf_ob.R_max))
        for line in ski_template:
     
            # Fraction of metals locked into dust
            if line.find('<massFraction>') >= 0:
                if 'DTM' in p.sim_run: line = line.replace('<massFraction>', str(0.5).format('%.4f'))
                else: line = line.replace('<massFraction>', str(1).format('%.4f'))
     
            if line.find('<Rmax>') >= 0:
                line = line.replace('<Rmax>', str(np.ceil(isrf_ob.R_max*1000.)).format('%.4f'))
     
            if line.find('<FOV>') >= 0:
                line = line.replace('<FOV>', str(np.ceil(2*isrf_ob.R_max*1000.)).format('%.4f'))
     
            if line.find('<distance>') >= 0:
                line = line.replace('<distance>', str(isrf_ob.distance).format('%.2f'))
     
            if line.find('<galaxy>') >= 0:
                line = line.replace('<galaxy>',isrf_ob._get_name())
     
            ski_copy.write(line)
     
        ski_template.close()
        ski_copy.close()
     
    # Run SKIRT
    print('Now ready to run skirt: go to %s and run SKIRT'% p.d_skirt)
    print("One way: qsub -J 0-N_gal run_skirt.sh")

def read_SKIRT(gal_index):
    """
    Reads SKIRT output, using isrf classs.
    """

    print('\nNow for galaxy # %s' % gal_index)

    isrf_obj      =   isrf(gal_index = gal_index)
    isrf_obj.setup_tasks()
    if isrf_obj.read_skirt: 
        isrf_obj.read_skirt_output()
        print('\nDone w galaxy # %s' % gal_index)
    else: 
        print('\nNothing to do for galaxy # %s' % gal_index)

class isrf(galaxy):
    """
    An class to handle all tasks related to the interstellar radiation field (ISRF) for one galaxy.
    Child class that inherits from parent class 'galaxy'.
    """

    pass  

    def setup_tasks(self):
        """
        Controls tasks to be executed, based on existing files and the overwrite [ow] parameter
        """

        p = copy.copy(params)

        self.gal_ob = dict(zred=self.zred,galname=self.name,gal_num=self.gal_num,gal_index=self.gal_index)
        # If overwriting, do all subgridding
        p.ow=True
        if p.ow:
            #print('Overwrite is ON, will execute the following tasks:')
            self.read_skirt     =   True
        # If not overwriting, check if subgridding has been done
        if not p.ow:
            #print('Overwrite is OFF, will execute the following tasks:')
            self.read_skirt     =   False

            # Check for SKIRT output files and saved ISRF
            cell_data           =   self.cell_data.get_dataframe()
            try:
                if ('G0' not in cell_data.keys()) | p.ow: self.read_skirt = True
            except:
                print('G0 is not found in cell_data output')
                self.read_skirt = True
            df = self.cell_data._read_cellprops()
            try:
                if len(df) != len(cell_data): self.read_skirt = True
            except:
                print('Did not find cell_data, reading SKIRT again')

        if self.read_skirt: print('* Read SKIRT output for galaxy # %i' % self.gal_index)
        if self.read_skirt == 0: print('Nothing! For galaxy # %i' % self.gal_index)

    def _get_name(self,method=''):
        """ Gets galaxy identifier used when running SKIRT

        :param method: String to identify specific SKIRT runs.
        :type method: str, optional
        """

        p = copy.copy(params)

        name = '%s%s%s_G%i' % (p.sim_name,p.sim_run,method,self.gal_index)
        
        return name

    def _set_distance_to_galaxy(self):
        """
        Sets distance to galaxy in Mpc (defaults to 10 Mpc for z = 0).
        """

        if self.zred == 0: 
            self.distance       =   10 
        else:
            self.distance       =   self.lum_dist 

    def read_skirt_output(self):
        """
        Reads and saves galaxy integrated luminosities from SKIRT output files 
        (SED instrument), and saves SKIRT grid for further use in SIGAME.
        """

        p = copy.copy(params)
        
        # Select distance to put galaxy
        self._set_distance_to_galaxy()

        if p.verbose: print('Adding galaxy-integrated luminosities to GR...')
        GR                      =   glo.global_results()

        if p.verbose: print('Saving new galaxy cell data object to store FUV flux...')
        self.cell_data._add_data()
        cellgas = self.cell_data.df
        if (type(cellgas) == int) | (p.ow == True): 
            print('Found no cell dataframe or "overwrite" is on, starting a new one for galaxy # %i' % self.gal_index)
            cellgas                 =   self.cell_data.start_dataframe()
        else:
            print('cell dataframe already created')
        cellgas = self.cell_data.df

        if ('G0' not in cellgas.keys()) | (p.ow == True):
            if p.verbose: print('Add FUV flux and OIR/FUV ratio to cell data object')
            self.cell_data._add_FUV_flux()
            self.cell_data.save_dataframe()
        else:
            print('FUV flux already calculated')
 
        if (GR.get_item(self.name,'L_bol_sun') != -1) | (p.ow == True):
            L_bol_sun = self._add_bol_lum()
            if p.verbose: print('Bolometric luminosity: %.4e Lsun' % self.L_bol_sun)
            GR.edit_item(self.name,'L_bol_sun',self.L_bol_sun)
 
        if (GR.get_item(self.name,'L_FUV_sun') != -1) | (p.ow == True):
            L_FUV_sun = self._add_FUV_lum()
            if p.verbose: print('FUV luminosity: %.4e Lsun' % self.L_FUV_sun)
            GR.edit_item(self.name,'L_FUV_sun',self.L_FUV_sun)
 
        if (GR.get_item(self.name,'L_TIR_sun') != -1) | (p.ow == True):
            L_TIR_sun = self._add_TIR_lum()
            if p.verbose: print('TIR luminosity: %.4e Lsun' % self.L_TIR_sun)
            GR.edit_item(self.name,'L_TIR_sun',self.L_TIR_sun)
 
        if (GR.get_item(self.name,'L_FIR_sun') != -1) | (p.ow == True):
            L_FIR_sun = self._add_FIR_lum()
            if p.verbose: print('FIR luminosity: %.4e Lsun' % self.L_FIR_sun)
            GR.edit_item(self.name,'L_FIR_sun',self.L_FIR_sun)
 
        self.cell_data.save_dataframe()

    def _add_bol_lum(self):
        """
        Calculates total emitted bolometric luminosity of galaxy from SKIRT output files (SED instrument).
        """

        p = copy.copy(params)

        self._set_distance_to_galaxy()
        
        # Read flux in W/m2/micron
        SED_inst = self._read_SED(fluxOutputStyle="Wavelength")
        F_W_m2_micron = SED_inst.F_W_m2_micron.values
        wavelengths = SED_inst.wavelength.values
 
        # Convert to solar luminosity
        F_bol_W_m2 = np.trapz(F_W_m2_micron,x=wavelengths)
        L_bol_W = F_bol_W_m2*4*np.pi*(self.distance*1e6*p.pc2m)**2
        L_bol_sun = L_bol_W/p.Lsun
        
        self.L_bol_sun = L_bol_sun
        
    def _add_FUV_lum(self):
        """
        Calculates total emitted FUV luminosity of galaxy from SKIRT output files (SED instrument).
        """
   
        p = copy.copy(params)

        self._set_distance_to_galaxy()

        # Read flux in W/m2/micron
        SED_inst = self._read_SED(fluxOutputStyle="Wavelength")
        F_W_m2_micron = SED_inst.F_W_m2_micron.values
        wavelengths = SED_inst.wavelength.values

        N_start,N_stop = aux.FUV_index(wavelengths)

        # Convert to solar luminosity
        F_FUV_W_m2 = np.trapz(F_W_m2_micron[N_start:N_stop],x=wavelengths[N_start:N_stop])
        L_FUV_W = F_FUV_W_m2*4*np.pi*(self.distance*1e6*p.pc2m)**2
        L_FUV_sun = L_FUV_W/p.Lsun
        
        self.L_FUV_sun = L_FUV_sun

    def _add_TIR_lum(self):
        """
        Calculates total emitted TIR (3-1100 microns) luminosity of galaxy from SKIRT output files (SED instrument).
        TIR as defined in Kennicutt and Evans 2012 table 1:
         https://www.annualreviews.org/doi/abs/10.1146/annurev-astro-081811-125610
        """
        
        p = copy.copy(params)
        
        self._set_distance_to_galaxy()

        # Read flux in W/m2/micron
        SED_inst = self._read_SED(fluxOutputStyle="Wavelength")
        F_W_m2_micron = SED_inst.F_W_m2_micron.values
        wavelengths = SED_inst.wavelength.values

        N_start,N_stop = aux.TIR_index(wavelengths)

        # Convert to solar luminosity
        F_TIR_W_m2 = np.trapz(F_W_m2_micron[N_start:N_stop],x=wavelengths[N_start:N_stop])
        L_TIR_W = F_TIR_W_m2*4*np.pi*(self.distance*1e6*p.pc2m)**2
        L_TIR_sun = L_TIR_W/p.Lsun
        
        self.L_TIR_sun = L_TIR_sun

    def _add_FIR_lum(self):
        """
        Calculates total emitted FIR (40-500 microns)
        FIR defined as: https://ned.ipac.caltech.edu/level5/Sanders/Sanders2.html 
        """
        
        p = copy.copy(params)
        
        self._set_distance_to_galaxy()

        # Read flux in W/m2/micron
        SED_inst = self._read_SED(fluxOutputStyle="Wavelength")
        F_W_m2_micron = SED_inst.F_W_m2_micron.values
        wavelengths = SED_inst.wavelength.values

        N_start,N_stop = aux.FIR_index(wavelengths)

        # Convert to solar luminosity
        F_FIR_W_m2 = np.trapz(F_W_m2_micron[N_start:N_stop],x=wavelengths[N_start:N_stop])
        L_FIR_W = F_FIR_W_m2*4*np.pi*(self.distance*1e6*c.pc.to('m').value)**2
        L_FIR_sun = L_FIR_W/p.Lsun
 
        print('FIR luminosity: %.2e Lsun' % L_FIR_sun)
        
        self.L_FIR_sun = L_FIR_sun

    def _read_SED(self,fluxOutputStyle="Wavelength",select=''):
        """ 
        Reads SED output from SKIRT.

        :param fluxOutputStyle: Units of flux in each cell, defaults to "Wavelength" corresponding to (W/m2/micron)
        :type fluxOutputStyle: str, optional

        :param method: String to identify specific SKIRT runs, for instance with increased photon package number.
        :type method: str, optional

        """

        p = copy.copy(params)
        
        data = pd.read_csv(p.d_skirt+'%s%s_xy_sed.dat' % (self._get_name(),method),skiprows=7,sep=' ',engine='python',\
            names=['wavelength','tot','trans','direct_primary','scattered_primary',\
           'direct_secondary','scattered_secondary'])
        data['frequency'] = p.clight/(data['wavelength']*1e-6)

        # Convert flux to W/m^2
        if fluxOutputStyle == "Wavelength":     
            data['F_W_m2_micron']   =   data.tot.values

        if fluxOutputStyle == "Neutral":        
            data['F_W_m2']          =   data.tot.values # W/m^2

        return(data)

    def _get_cut_probe(self,**kwargs):
        """ 
        Gets cross section data through galaxy of radiation field from SKIRT for visualization.

        :param orientation: From which angle to view the galaxy, either 'face-on','xy','yx','edge-on','xz','zx','yz' or 'zy', defaults to 'face-on'.
        :type orientation: str, optional
        """

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        if p.orientation in ['face-on','xy','yx']: filename = '_rfcut_J_xy'
        if p.orientation in ['edge-on','xz','zx']: filename = '_rfcut_J_xz'
        if p.orientation in ['yz','zy']: filename = '_rfcut_J_yz'

        move = sub.Popen('cp ' + p.d_skirt + '%s' % self._get_name() + filename + '.fits ' +
                   p.d_temp+self._get_name()+filename+'.fits', shell=True)
        move.wait()
        image_file = get_pkg_data_filename(p.d_temp + self._get_name() + filename + '.fits')
        image_data,hdr = fits.getdata(image_file, ext=0, header=True)
        delete = sub.Popen('rm ' + p.d_temp + self._get_name() + filename + '.fits', shell=True)

        units = hdr['BUNIT']
        return(image_data,units)

    def _get_map_inst(self,**kwargs):
        """ 
        Gets projected map of radiation field from SKIRT for visualization.

        :param orientation: From which angle to view the galaxy, either 'face-on','xy','yx','edge-on','xz','zx','yz' or 'zy', defaults to 'face-on'.
        :type orientation: str, optional
        """

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        if p.orientation in ['face-on','xy','yx']: ext = '_xy_map_total'
        if p.orientation in ['edge-on','xz','zx']: ext = '_xz_map_total'
        if p.orientation in ['yz','zy']: ext = '_yz_map_total'

        hdul = fits.open(p.d_XL_data + 'skirt/' + self._get_name() + p.select + ext + '.fits')
        wa = hdul[1].data.field(0)
        image_data = hdul[0].data
        units = hdul[0].header['BUNIT']
        print('Units of fits image: %s' % units)

        return(image_data,units,wa)

#---------------------------------------------------------------------------
### FOR RE-GRIDDING TASK ###
#---------------------------------------------------------------------------

def run_grid(gal_index):
    """
    A function that puts gas properties on a grid structure, using grid class.
    Called by backend.py.

    :param gal_index: Index of galaxy to process.
    :type gal_index: int
    """

    #print('\nNow for galaxy # %i' % gal_index)

    grid_obj      =   grid(gal_index = gal_index)
    grid_obj.setup_tasks()
    grid_obj.run()

class grid(galaxy):
    """
    An class to handle all tasks related to deriving and storing the cell information for one galaxy.
    Child class that inherits from parent class 'galaxy'.
    """

    pass  


    def setup_tasks(self):
        '''
        Controls tasks to be executed, based on existing files and the overwrite [ow] parameter
        '''

        p = copy.copy(params)

        self.gal_ob = dict(zred=self.zred,galname=self.name,gal_index=self.gal_index)
        gal_ob          =   galaxy(self.gal_index)

        # If overwriting, do all tasks
        if p.ow:
            self.add_FUV_flux           =   True
            self.add_nH                 =   True
            self.add_metallicity        =   True
            self.add_SFR_density        =   True

        # If not overwriting, check which tasks have been performed
        if not p.ow:
            self.add_nH                 =   False
            self.add_FUV_flux           =   False
            self.add_metallicity        =   False
            self.add_SFR_density        =   False

            df = gal_ob.cell_data.get_dataframe()
            if 'nH' not in df.keys():
                self.add_nH                 =   True
            if 'Z' not in df.keys():
                self.add_metallicity        =   True
            if 'nSFR' not in df.keys():
                self.add_SFR_density        =   True
            if 'G0' not in df.keys():
                self.add_FUV_flux           =   True

        if p.turb == 'T20': 
            self.add_Mach_number = False

        if self.add_FUV_flux: print('* Derive and add FUV (and other band) flux for galaxy # %i' % self.gal_index)
        if self.add_metallicity: print('* Derive and add metallicity for galaxy # %i' % self.gal_index)
        if self.add_SFR_density: print('* Derive and add SFR density for galaxy # %i' % self.gal_index)
        if self.add_FUV_flux + self.add_metallicity + self.add_SFR_density + self.add_nH == 0: 
            print('Do nothing! For galaxy # %i' % self.gal_index)

    def run(self):
        """ 
        Converts particle properties to cell grid.
        """

        p = copy.copy(params)

        self.add_f_H2 = False
        self.do_fragmentation = False

        print('Running gridding procedure for G%i' % self.gal_index)

        # Add density
        if self.add_nH:

            # Load cell data
            self.cell_data._add_data()
            self.cell_data._add_nH()
            self.cell_data.save_dataframe()
            print('done with density for galaxy # %i (gal_num = %i)' % (self.gal_index,self.gal_num))

        if self.add_FUV_flux:
            
            # Load cell data
            self.cell_data._add_data()

            # Add FUV flux to cell data
            self.cell_data._add_FUV_flux()
            self.cell_data.save_dataframe()
            print('done with photometry for galaxy # %i (gal_num = %i)' % (self.gal_index,self.gal_num))

        if self.add_SFR_density:

            # Load cell data
            self.cell_data._add_data()
        
            # Add metallicity to cell data
            self.cell_data._add_SFR_density()
            self.cell_data.save_dataframe()
            print('done with SFR density for galaxy # %i (gal_num = %i)' % (self.gal_index,self.gal_num))

        if self.add_metallicity:

            # Load cell data
            self.cell_data._add_data()
        
            # Add metallicity to cell data
            self.cell_data._add_metallicity()
            self.cell_data.save_dataframe()
            print('done with Z for galaxy # %i (gal_num = %i)' % (self.gal_index,self.gal_num))

#---------------------------------------------------------------------------
### FOR INTERPOLATION TASK ###
#---------------------------------------------------------------------------

def run_interp(gal_index):
    """
    A function that interpolates Cloudy look-up tables for 
    line luminosity and other properties for one galaxy at a time, using interpolation class.
    Called by backend.py.
    """

    interp_obj      =   interpolation(gal_index = gal_index)
    interp_obj.setup_tasks()

    if interp_obj.do_interp_cells: 
        interp_obj.run_cells()
    else:
        print('Galaxy # %i cells already interpolated!' % gal_index)

def add_to_GR(**kwargs):
    """
    Function that adds integrated and mass-weighted quantities to global results file once all galaxies have been processed.
    """

    print('Add integrated and mass-weighted quantities to GR for all galaxies')

    p = copy.copy(params)

    GR                      =   glo.global_results(verbose=True)

    cloudy_library = clo.library()
    lookup_table = cloudy_library._restore_lookup_table()
    lognHs = np.unique(lookup_table.lognHs)
    lognSFRs = np.unique(lookup_table.lognSFRs)
    logNHs = np.unique(lookup_table.logNHs)
    logFUVs = np.unique(lookup_table.logFUVs)
    logZs = np.unique(lookup_table.logZs)
    logDTMs = np.unique(lookup_table.logDTMs)

    G0_mw = np.zeros(GR.N_gal)
    nH_mw = np.zeros(GR.N_gal)
    h_min = np.zeros(GR.N_gal)
    nH_min = np.zeros(GR.N_gal)
    nH_max = np.zeros(GR.N_gal)
    M_dust_DTM = np.zeros(GR.N_gal)
    age_mw = np.zeros(GR.N_gal)
    Zstar_mw = np.zeros(GR.N_gal)
    M_1e3 = np.zeros(GR.N_gal)
    M_1e1 = np.zeros(GR.N_gal)
    nH_cell_max = np.zeros(GR.N_gal)
    cell_size_min = np.zeros(GR.N_gal)
    cell_size_max = np.zeros(GR.N_gal)
    P_HII_mw = np.zeros(GR.N_gal)
    P_mw = np.zeros(GR.N_gal)
    Pe_mw = np.zeros(GR.N_gal)
    ne_mw = np.zeros(GR.N_gal)
    ne_HII_mw = np.zeros(GR.N_gal)
    ne_HI_mw = np.zeros(GR.N_gal)
    ne_H2_mw = np.zeros(GR.N_gal)
    Te_mw = np.zeros(GR.N_gal)
    Tk_mw = np.zeros(GR.N_gal)
    lums = pd.DataFrame({})
    for line in p.lines:
        lums[line] = np.zeros(GR.N_gal)
        for phase in ['HII','HI','H2']:
            lums[line+'_'+phase] = np.zeros(GR.N_gal)
        lums[line+'_HII_region'] = np.zeros(GR.N_gal)
    print(lums)
    for i in range(GR.N_gal):
        gal_ob = galaxy(gal_index = i)
        df = gal_ob.particle_data.get_dataframe('simgas')
        nH_min[i] = np.min(df.nH.values[df.nH.values > 0])
        nH_max[i] = np.max(df.nH.values)
        df = gal_ob.cell_data.get_dataframe()
        if 'ne_HI_mw' not in df.keys():
             gal_ob.cell_data._do_interpolation()
        G0_mw[i] = np.sum(df.G0.values*df.m.values)/np.sum(df.m.values)
        nH_mw[i] = np.sum(df.nH.values*df.m.values)/np.sum(df.m.values)
        df = gal_ob.particle_data.get_dataframe('simgas')
        if p.sim_type == 'sph': h_min[i] = np.min(df.h.values)
        M_dust_DTM[i] = np.sum(df.m.values*df.Z.values*p.mf_Z1*0.5)
        # stars
        df = gal_ob.particle_data.get_dataframe('simstar')
        age_mw[i] = np.sum(df.age.values*df.m.values)/np.sum(df.m.values)
        Zstar_mw[i] = np.sum(df.Z.values*df.m.values)/np.sum(df.m.values)
        df = gal_ob.cell_data.get_dataframe()
        M_1e3[i] = np.sum(df.m.values*df.mf_1e3_grid.values)
        M_1e1[i] = np.sum(df.m.values*df.mf_1e1_grid.values)
        nH_cell_max[i] = np.max(df.nH)
        cell_size_min[i] = np.min(df.cell_size)
        cell_size_max[i] = np.max(df.cell_size)
        ne_HII_mw[i] = np.sum(df.ne_HII_mw*df.mf_HII_grid*df.m)/np.sum(df.mf_HII_grid*df.m)
        ne_HI_mw[i] = np.sum(df.ne_HI_mw*df.mf_HI_grid*df.m)/np.sum(df.mf_HI_grid*df.m)
        ne_H2_mw[i] = np.sum(df.ne_H2_mw*df.mf_H2_grid*df.m)/np.sum(df.mf_H2_grid*df.m)
        if 'ne_mw' not in df.keys():
            df['ne_mw'] = df['ne_mw_grid'].values
        ne_mw[i] = np.sum(df.ne_mw*df.m)/np.sum(df.m)
        Te_mw[i] = np.sum(df.Te_mw*df.m)/np.sum(df.m)
        Tk_mw[i] = np.sum(df.Tk_mw*df.m)/np.sum(df.m)
        P_HII_mw[i] = np.sum(df.P_e_mw*df.mf_HII_grid*df.m)/np.sum(df.mf_HII_grid*df.m)
        P_mw[i] = np.sum(df.P_mw*df.m)/np.sum(df.m)
        Pe_mw[i] = np.sum(df.P_e_mw*df.m)/np.sum(df.m)
        # Line luminosities
        for line in p.lines:
            lums[line][i] = np.sum(df['L_'+line])
            for phase in ['HII','HI','H2']:
                lums[line+'_'+phase][i] = np.sum(df['L_'+line+'_'+phase])
 
    GR.add_column('G0_mw',G0_mw)
    GR.add_column('nH_cell_mw',nH_mw)
    if p.sim_type == 'sph': GR.add_column('h_min',h_min)
    GR.add_column('nH_min',nH_min)
    GR.add_column('nH_max',nH_max)
    GR.add_column('M_dust_DTM',M_dust_DTM)
    GR.add_column('age_mw',age_mw)
    GR.add_column('Zstar_mw',Zstar_mw)
    GR.add_column('M_1e3',M_1e3)
    GR.add_column('M_1e1',M_1e1)
    GR.add_column('nH_cell_max',nH_cell_max)
    GR.add_column('cell_size_min',cell_size_min)
    GR.add_column('cell_size_max',cell_size_max)
    GR.add_column('ne_mw',ne_mw)
    GR.add_column('ne_HII_mw',ne_HII_mw)
    GR.add_column('ne_HI_mw',ne_HI_mw)
    GR.add_column('ne_H2_mw',ne_H2_mw)
    GR.add_column('Te_mw',Te_mw)
    GR.add_column('Tk_mw',Tk_mw)
    GR.add_column('P_HII_mw',P_HII_mw)
    GR.add_column('P_mw',P_mw)
    GR.add_column('Pe_mw',Pe_mw)
    for line in p.lines:
        GR.add_column('L_%s_sun' % line,lums['%s' % line])
        for phase in ['HII','HI','H2']:
            GR.add_column('L_%s_%s_sun' % (line,phase),lums['%s_%s' % (line,phase)])

class interpolation(galaxy):
    """
    An class to handle all tasks related to interpolating in the Cloudy look-up table for one galaxy.
    Child class that inherits from parent class 'galaxy'.
    """

    pass  

    def setup_tasks(self):
        '''
        Controls tasks to be executed, based on existing files and the overwrite [ow] parameter
        '''

        p = copy.copy(params)

        GR                      =   glo.global_results(verbose=True)
        
        self.gal_ob = dict(zred=self.zred,galname=self.name,gal_index=self.gal_index)

        df = self.cell_data.get_dataframe()

        self.do_interp_cells = False
        self.read_skirt = False
        if p.ow:
            self.do_interp_cells = True
            print('Overwrite is on, will run interpolation')
        if not p.ow:
            try:
                df = pd.read_pickle('data/results/temp/G%i' % self.gal_index)
                print('Cells already interpolated')
            except:
                self.do_interp_cells = True

    def run_cells(self):
        """
        Executes interpolation task for gas cells in one galaxy.
        """

        self.cell_data._do_interpolation_cells()

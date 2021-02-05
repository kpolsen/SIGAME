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
        #self.file_name      =   GR.file_name[gal_index]
        self.zred           =   GR.zreds[gal_index]
        self.SFR            =   GR.SFR[gal_index]
        self.Zsfr           =   GR.Zsfr[gal_index]
        #self.SFRsd          =   GR.SFRsd[gal_index]
        self.lum_dist       =   GR.lum_dist[gal_index]
        self.alpha          =   GR.alpha[gal_index]
        self.beta           =   GR.beta[gal_index]
        self.ang_dist_kpc   =   self.lum_dist*1000./(1+self.zred)**2

        # add objects
        self.add_attr('particle_data')
        self.add_attr('cell_data')

        if p.verbose: print("galaxy %s constructed.\n" % self.name)

    def add_attr(self,attr_name,verbose=False):
        """Creates desired attribute and adds it to galaxy object """

        if hasattr(self, attr_name):
            if verbose: print("%s already has attribute %s" % (self.name,attr_name))
        else:
            if verbose: print("Adding %s attribute to %s ..." % (attr_name,self.name) )
            if attr_name == 'particle_data': ob = particle_data(self)
            if attr_name == 'cell_data': ob = cell_data(self)
            if attr_name == 'datacube': ob = datacube(self)
            setattr(self,attr_name,ob)

class particle_data:
    """An object referring to the particle data (sim or ISM)

    .. note:: Must be added as an attribute to a galaxy object.

    Parameters
    ----------
    gal_ob : object
        Instance of galaxy class.
    silent : bool
        Parameter telling the code to do (or not) print statements.

    Examples
    --------
    >>> import galaxy as gal
    >>> gal_ob = gal.galaxy(gal_index=0)
    >>> simgas = gal_ob.particle_data.get_dataframe('sim')['gas']

    """

    def __init__(self,gal_ob,**kwargs):

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        if p.verbose: print("constructing particle_data object...")

        # self.xyz_units = xyz_units

        # add labels for spatial dimentions
        # dim     =   ['x','y','z']
        # for x in dim: setattr(self,x + 'label','%s [%s]' % (x,xyz_units))

        # add galaxy
        self.gal_ob             =   gal_ob

        # add dictionaries of names for all sim types and ISM phases
        # self.add_names()

        if p.verbose: print("particle_data object constructed for %s.\n" % gal_ob.name)


    def _add_data(self,data_type):
        """ Add particle data as dataframe ("df" attribute) to particle_data object.
        """

        df                      =   aux.load_temp_file(gal_ob=self.gal_ob,data_type=data_type)
        setattr(self,data_type,df)

    def rotate_to_xy(self,data_type='simgas'):
        """ Rotate particles to lie in the xy-plane
        """

        df                      =   getattr(self,data_type).copy()

        # Calculate mass-weighted angular momentum vector


        setattr(self,data_type,df)


    def _add_surface_density(self,data_type='simgas'):
        """ Add surface density to dataframe ("df" attribute of particle_data object).
        """

        print('Derive surface density on smoothing length scales')

        df                      =   getattr(self,data_type).copy()

        # Add surface density using swiftsimio.visualisation.projection
        res_kpc                 =   np.min(df.h)
        x,y,max_scale,pix_size,Npix = aux.projection_scaling(df.x.values, df.y.values, df.z.values, res_kpc)
        if data_type == 'simgas': h = df.h
        if data_type == 'simstar': h = 0.01 # kpc
        # Make 2D projection of mass
        map2D_m                 =   projection.scatter(x, y, df.m.values, h, Npix)

        # Interpolate in map for particle positions with radial basis functions:
        xgrid                   =   np.arange(0,max_scale,pix_size) - max_scale/2.
        ygrid                   =   np.arange(0,max_scale,pix_size) - max_scale/2.
        xx,yy                   =   np.meshgrid(xgrid, ygrid)
        f_func                  =   interp.Rbf(xx, yy, map2D_m, function='linear', smooth=0)  # default smooth=0 for interpolation
        xx                      =   f_func(df.x.values,df.y.values)
        df['Sigma_'+data_type]  =   f_func(df.x.values,df.y.values)

        setattr(self,data_type,df)

    def _add_vertical_vel_disp(self,data_type='simgas'):
        """ Add vertical velocity dispersion to dataframe ("df" attribute of particle_data object).
        """

        print('Derive vertical velocity dispersion on smoothing length scales')

        df                      =   getattr(self,data_type).copy()

        # Index data points according to tree structure
        coords                  =   df[['x','y','z']].values
        indices                 =   spatial.cKDTree(coords).query(coords,k=64)[1]        

        vel_disp_vert           =   np.zeros(len(df))
        for i in range(len(vel_rms)):

            # nearest neighbors info
            df_cut = df.iloc[indices[i]].reset_index(drop=True)
            vz = df_cut['vz']

            # Vertical vel disp 
            vel_disp_vert[i]       =   np.std(np.abs(vz))

        df['vel_disp_vert']     =   vel_disp_vert

        setattr(self,data_type,df)

    def _add_vel_disp(self,data_type='simgas'):
        """ Add velocity dispersion to dataframe ("df" attribute of particle_data object).
        """

        print('Derive velocity dispersion on smoothing length scales')

        df                      =   self.simgas.copy()

        # Index data points according to tree structure
        coords                  =   df[['x','y','z']].values
        indices                 =   spatial.cKDTree(coords).query(coords,k=64)[1]

        # Add rms of velocity
        df['v']                 =   np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
        vel_rms                 =   np.zeros(len(df))
        for i in range(len(vel_rms)):

            # particle info
            xyz_i = coords[i]; h_i = df['h'][i]; m_i = df['m'][i]; v_i = df['v'][i]

            # nearest neighbors info
            df_cut = df.iloc[indices[i]].reset_index(drop=True)
            m = df_cut['m']; h = df_cut['h']; v = df_cut['v']
            
            # distance to nearest neighbors            
            dx = coords[indices[i]] - xyz_i
            r = np.sqrt(np.sum(dx * dx, axis=1))

            # RMS deviation of particle velocities from velocity estimate, using SPH formalism:
            kernel = [aux.Wendland_C2_kernel(r1,h_i) for r1 in r]
            dens_i = sum(m * kernel) # estimate of density, not from snapshot
            vel_rms[i] = np.sqrt(np.sum(m*(v-v_i)**2*kernel)/dens_i)
            
        df['vel_disp']          =   vel_rms
        self.simgas             =   df

    def _scale_vel_disp(self):
        """ Scale velocity dispersion from smoothing length to cloud scales.
        """

        print('Scale velocity dispersion from smoothing length to cloud scales')

        # Estimate typical cloud scale based on pressure-normalized scaling 
        # of a 1e4 Msun cloud (e.g. Swinbank et al. 2011 eq 4 and 6)
        self._add_P_ext()
        simgas                  =   self.simgas.copy()
        Mgmc                    =   1e4 # Msun
        simgas.R_cloud_kpc      =   (simgas.P_ext/1e4)**(-1.0/4.0)*(Mgmc/290.0)**(1.0/2.0) / 1e3 # kpc
        vel_rms_cloud           =   simgas.vel_disp.values * (self.R_cloud_kpc / simgas.h.values)**(1/3)
        simgas.vel_disp_cloud   =   vel_rms_cloud

        self.simgas             =   simgas

    def _add_P_ext(self):
        '''Adds external pressure to galaxy and stores gas/star sim particle data files again with the new information.
        '''

        print('Add external pressure field to galaxy')

        p = copy.copy(params)

        # Make global variables for Pfunc function
        global simgas, simgas1, simstar, m_gas, m_star

        simgas                  =   self.simgas.copy()
        simstar                 =   self.simstar.copy()

        # Extract star forming gas only:
        simgas1                 =   simgas.copy()
        simgas1                 =   simgas1[simgas1['SFR'] > 0].reset_index()

        # Extract gas and star masses
        m_gas,m_star            =   simgas1['m'].values,simstar['m'].values

        print('(Multiprocessing starting up! %s cores in use)' % p.N_cores)
        pdb.set_trace()

        pool                    =   mp.Pool(processes=p.N_cores)            
        results                 =   [pool.apply_async(aux.Pfunc, args=(i, simgas1, simgas, simstar, m_gas, m_star,)) for i in range(0,len(simgas))]#len(simgas)
        pool.close()
        pool.join()

        # sort results since apply_async doesn't return results in order
        res                     =   [result.get() for result in results]
        res.sort(key=lambda x: x[0])
        print('(Multiprocessing done!)')

        # Store pressure,velocity dispersion and surface densities
        for i,key in enumerate(['pressure_term','surf_gas','surf_star','sigma_gas','sigma_star','vel_disp_gas']):

            simgas[key]     =   [res[_][i+1]  for _ in range(len(res))]
            if key == 'pressure_term': 
                pressure = np.pi/2.*p.G_grav*simgas[key]/1.65 # m^3 kg^-1 s^-2 Msun pc^-2 
                simgas['P_ext'] = pressure * p.Msun**2/p.kpc2m**4/p.kB/1e6 # K/cm^3

        # Store new simgas and simstar files
        self.simgas             =   simgas
        aux.save_temp_file(simgas,gal_ob=self.gal_ob,data_type='simgas')

        del simgas, simgas1, simstar, m_gas, m_star

        setattr(self,'P_ext_added',True)

    def get_dataframe(self,data_type):
        """ Return dataframe with particle data for particle_data object.
        """

        self._add_data(data_type)
        return(getattr(self,data_type))

    def save_dataframe(self,data_type):

        aux.save_temp_file(getattr(self,data_type),gal_ob=self.gal_ob,data_type=data_type)

class cell_data:
    """An object referring to the cell data (from SKIRT "AMR" output)

    .. note:: Must be added as an attribute to a galaxy object.

    Parameters
    ----------
    gal_ob : object
        Instance of galaxy class.
    silent : bool
        Parameter telling the code to do (or not) print statements.

    Examples
    --------
    >>> import galaxy as gal
    >>> gal_ob = gal.galaxy(gal_index=0)
    >>> gal_ob.cell_data.start_dataframe()
    >>> cell_data = gal_ob.cell_data.get_dataframe()

    """

    def __init__(self,gal_ob,**kwargs):

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        # add galaxy
        self.gal_ob =   gal_ob

        if p.verbose: print("constructing cell_data object...")

    def start_dataframe(self):

        print('Creating dataframe with SKIRT cell data output')

        p = copy.copy(params)
        
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
        """ Get galaxy identifier used when running/reading SKIRT
        """

        p = copy.copy(params)

        if p.sim_type == 'sph': name = '%s%s_G%i' % (p.sim_name,p.sim_run,self.gal_ob.gal_index)
        if p.sim_type == 'amr': name = '%s%s' % (p.sim_name,p.sim_run)
   
        return name

    def save_dataframe(self):

        aux.save_temp_file(self.df,gal_ob=self.gal_ob,data_type='cell_data')

    def _add_data(self):
        """ Add cell data as dataframe to object
        """

        self.df         =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='cell_data')

    def get_dataframe(self):

        if not hasattr(self,'df'): self._add_data()
        return(getattr(self,'df'))

    def _interpolate_cloudy_table(self,lookup_table,key,cell_prop): 

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

        #pdb.set_trace()
        try:
            result = interp(cell_prop)
        except:
            pdb.set_trace()

        return(result)

    def _do_interpolation(self):
        """ Interpolate in Cloudy look-up table for line luminosities
        """

        p = copy.copy(params)

        df_or                   =   self.get_dataframe()
        if 'DTM' not in df_or.keys():
            self._add_DTM()
        self.save_dataframe()
        if 'vx' not in df_or.keys():
            self._add_velocity()
        self.save_dataframe()
        df_or                   =   self.get_dataframe()

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
        df_or['DTM'] = DTM

        # Make sure we don't go outside of grid:
        #if p.sim_name == 'enzo':
            # TEST !!!
            #df_or['SFR_density'] = 1
            #df_or['nSFR'] = 1
        df                      =   df_or.copy()[['nH','Z','G0','R_NIR_FUV','nSFR','DTM']]
        #if p.sim_name == 'simba':
        #    print('TEST!! new nSFR values')
        #    df['SFR_density']       =   df.nSFR.values/(0.2**3)
        print(np.min(df.nSFR.values))
        print(np.max(df.nSFR.values))
        print(np.min(df.nH.values))
        print(np.max(df.nH.values))
       
        # TEST !!!
        #df.nH                   =   df.nH.values*0. + 10**(-4)
        #df.SFR_density          =   df.nH.values*0. + 10**(-2.5)
        #df.G0                   =   df.nH.values*0. + 1e-3
        #df.Z                    =   df.nH.values*0. + 1e-2
        #df.DTM                  =   df.nH.values*0. + 1e-2

        df.Z[np.isnan(df.Z)]    =   1e-6 # because something weird comes out from Z evaluation step
        df.nSFR[df.nSFR == 0]    =   1e-30 
        df.nSFR[np.isnan(df.nSFR)]    =   -30 

        # IMPORTANT: Taking the log of the entire dataframe and re-naming columns
        df                      =   np.log10(df)
        #print(np.min(df.DTM),np.max(df.DTM))
        df = df.rename(columns={'nH':'lognH','Z':'logZ','G0':'logFUV','nSFR':'lognSFR','DTM':'logDTM'})

        # Convert NIR/FUV ratio to column density NH
        logNH_cl,R_NIR_FUV_cl = aux.get_NH_from_cloudy()
        R_NIR_FUV_df = df_or.R_NIR_FUV.values


        interp                  =   interp1d(np.log10(R_NIR_FUV_cl)[8::],logNH_cl[8::],fill_value='extrapolate',kind='slinear')
        df['logNH']             =   interp(np.log10(R_NIR_FUV_df))
        df['logNH'][df.logNH <= np.min(logNH_cl)] = np.min(logNH_cl)
        df['logNH'][df.logNH >= np.max(logNH_cl)] = np.max(logNH_cl)
        df['logNH'][np.isinf(R_NIR_FUV_df)] = np.max(logNH_cl)
        df_or['NH']             =   10.**df.logNH

        print('G%i - range in logNH: %.2e to %.2e' % (self.gal_ob.gal_index,np.min(df.logNH),np.max(df.logNH)))

        # Make sure that cell data doesn't exceed look-up table values
        for _ in p.interp_params:
            df[_][df[_] <= np.min(lookup_table[_+'s'].values)] = np.min(lookup_table[_+'s'].values) + 1e-6 * np.abs(np.min(lookup_table[_+'s']))
            df[_][df[_] >= np.max(lookup_table[_+'s'].values)] = np.max(lookup_table[_+'s'].values) - 1e-6 * np.abs(np.min(lookup_table[_+'s']))

        # Cell properties used for interpolation in cloudy grid models:
        #print('Now column-stacking relevant cell data')
        cell_prop               =   np.column_stack((df.lognH.values,df.lognSFR.values,df.logNH.values,df.logFUV.values,df.logZ.values,df.logDTM.values))        

        # New dataframe to fill with interpolation results
        cell_prop_new           =   df_or.copy()
        
        ### LINE EMISSION
        #print('Now interpolating for line emission')
        for target in p.lines:
            cell_prop_new['L_'+target] = self._interpolate_cloudy_table(lookup_table,target,cell_prop)
            # Scale by H mass of that cell (each look-up table entry is for 1e4 Msun H mass):
            cell_prop_new['L_'+target]   =   10.**cell_prop_new['L_'+target].values*cell_prop_new.mH/1e4 
            # as 1e4 Msun was assumed in Cloudy_modeling.sample_cloudy_models()
            # Only count cells with actual hydrogen mass
            cell_prop_new['L_'+target].values[df_or.nH == 0] = 0
            for phase in ['HII','HI','H2']:
                line_lum = self._interpolate_cloudy_table(lookup_table,target+'_'+phase,cell_prop)
                cell_prop_new['L_'+target+'_'+phase]   =   10.**line_lum*cell_prop_new.mH/1e4 

        #print('Now interpolating for mass')
        # Add mass just in case it's not there, and scale to match original simulation mass
        GR                      =   glo.global_results()
        for target in p.lines:
            line_lum                =   np.sum(cell_prop_new['L_'+target].values)
            print('G%i - %s: %.2e Lsun' % (self.gal_ob.gal_index,target,line_lum))
            GR.edit_item(self.gal_ob.name,'L_%s_sun' % target,line_lum)
            for phase in ['HII','HI','H2']:
                line_lum                =   np.sum(cell_prop_new['L_'+target+'_'+phase].values)
                GR.edit_item(self.gal_ob.name,'L_%s_%s_sun' % (target,phase),line_lum)
            
        cell_prop_new['m']      =   (cell_prop_new.cell_size * p.kpc2cm)**3 * cell_prop_new.nH * p.mH / p.Msun # Msun
        cell_prop_new['m']      *=   np.sum(GR.M_gas[self.gal_ob.gal_index]) / np.sum(cell_prop_new['m'].values)
        cell_prop_new['mH']     =   3/4 * cell_prop_new['m'].values # Msun (assuming He mass fraction of 25%)

        ### CLOUDY CELL VOLUME
        V_grid = self._interpolate_cloudy_table(lookup_table,'V',cell_prop)
        cell_prop_new['cell_size_lookup']      =   (V_grid)**(1/3) / 1e3 # kpc

        ### DENSE MASS FRACTIONS
        cell_prop_new['mf_1e3_grid'] = self._interpolate_cloudy_table(lookup_table,'mf_1e3',cell_prop)
        cell_prop_new['mf_1e1_grid'] = self._interpolate_cloudy_table(lookup_table,'mf_1e1',cell_prop)
        print('G%i - mass fraction at nH > 1e3 cm^-3: %.3e %% ' % (self.gal_ob.gal_index,np.sum(cell_prop_new.m * cell_prop_new.mf_1e3_grid)/np.sum(cell_prop_new.m)*100.))

        ### TEMPERATURE
        cell_prop_new['Te_mw'] = self._interpolate_cloudy_table(lookup_table,'Te_mw',cell_prop)
        cell_prop_new['Tk_mw'] = self._interpolate_cloudy_table(lookup_table,'Tk_mw',cell_prop)

        ### CO-DISSOCIATING FLUX
        #cell_prop_new['F_NUV_ergs_cm2_s'] = self._interpolate_cloudy_table(lookup_table,'F_NUV_ergs_cm2_s',cell_prop)

        ### HYDROGEN IONIZATION FRACTIONS AND ELECTRON DENSITIES (remove "GRID" at some point)
        cell_prop_new['mf_H2_grid'] = self._interpolate_cloudy_table(lookup_table,'mf_H2',cell_prop)
        cell_prop_new['mf_HII_grid'] = self._interpolate_cloudy_table(lookup_table,'mf_HII',cell_prop)
        cell_prop_new['mf_HI_grid'] = self._interpolate_cloudy_table(lookup_table,'mf_HI',cell_prop)
        cell_prop_new['ne_grid'] = self._interpolate_cloudy_table(lookup_table,'ne',cell_prop)
        cell_prop_new['ne_mw'] = self._interpolate_cloudy_table(lookup_table,'ne_mw',cell_prop)
        GR.edit_item(self.gal_ob.name,'M_H2',np.sum( cell_prop_new['mf_H2_grid'] * cell_prop_new['m']  ))
        GR.edit_item(self.gal_ob.name,'M_HII',np.sum( cell_prop_new['mf_HII_grid'] * cell_prop_new['m']  ))
        GR.edit_item(self.gal_ob.name,'M_HI',np.sum( cell_prop_new['mf_HI_grid'] * cell_prop_new['m']  ))
        # Only within half-mass gas radius...
        R2_gas = getattr(GR,'R2_gas')[self.gal_ob.gal_index]
        r = np.sqrt( cell_prop_new['x']**2 + cell_prop_new['x']**2 + cell_prop_new['x']**2)
        GR.edit_item(self.gal_ob.name,'M_H2_R2_gas',np.sum( cell_prop_new['mf_H2_grid'][r < R2_gas] * cell_prop_new['m'][r < R2_gas] ))
        GR.edit_item(self.gal_ob.name,'M_HII_R2_gas',np.sum( cell_prop_new['mf_HII_grid'][r < R2_gas] * cell_prop_new['m'][r < R2_gas] ))
        GR.edit_item(self.gal_ob.name,'M_HI_R2_gas',np.sum( cell_prop_new['mf_HI_grid'][r < R2_gas] * cell_prop_new['m'][r < R2_gas] ))

        ### PRESSURE
        cell_prop_new['P_HII'] =   cell_prop_new['Te_mw']*cell_prop_new['ne_mw']   
        cell_prop_new['P_gas'] =   cell_prop_new['Tk_mw']*cell_prop_new['nH']   

        self.df                 =   cell_prop_new
        self.save_dataframe()

        ### ADD LUMINOSITIES TO GLOBAL RESULTS
        for target in p.lines:
            line_lum                =   np.sum(cell_prop_new['L_'+target].values)
            print('G%i - %s: %.2e Lsun' % (self.gal_ob.gal_index,target,line_lum))
            GR                      =   glo.global_results()
            GR.edit_item(self.gal_ob.name,'L_%s_sun' % target,line_lum)

        print('done with interpolation for galaxy # %i!' % (self.gal_ob.gal_index))

    def _read_cellprops(self):
        """ Read cell properties from SKIRT
        """

        p = copy.copy(params)
        df              =   pd.read_csv(p.d_XL_data + 'skirt/' + self._get_name()+'_scp_cellprops.dat',skiprows=9,sep=' ',engine='python',\
            names=['i','x','y','z','V','opt_depth','n_dust','ne','nH'],\
            dtype={'i':'Int64','x':'float64','y':'float64','z':'float64','V':'float64',\
           'opt_depth':'float64','n_dust':'float64','n_e':'float64','nH':'float64'})
        return(df)    

    def _add_nH(self):

        p = copy.copy(params)

        df                  =   self.df.copy()
        coords_cells        =   df[['x','y','z']].values
        #print(df.head())
        #print(np.unique(df.cell_size.values))
 
        # Get sim data for gas
        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='simgas')
        coords_sim          =   simgas[['x','y','z']].values
        #print(simgas.head())

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
            df['rho'] = df['m'] / (df['cell_volume'].values/1e9) # Msun 

        print('max nH in cell_data: %.2f' % (np.max(df.nH)))
        print('min nH in cell_data: %.2f' % (np.min(df.nH)))
        print('Total mass in simgas: %.2e' % (np.sum(simgas.m)))
        print('Total mass in cell_data: %.2e' % (np.sum(df.m)))
        self.df             =   df 

    def _add_FUV_flux(self,fluxOutputStyle="Wavelength"):
        """ Read wavelengths used for radiation field probe of each cell.
        fluxOutputStyle="Neutral": Units are lambda*F_lambda (W/m2)
        fluxOutputStyle="Frequency": Units are F_nu (W/m2/Hz)
        fluxOutputStyle="Wavelength": Units are F_lambda (W/m2/micron)
        """

        p = copy.copy(params)

        self._add_data()
        df                  =   self.df

        if ('F_UV_W_m2' not in df.keys()) | (p.ow == True):
            # Read probe wavelengths
            # print('AAA',self._get_name())
            wavelengths,bin_width = aux.read_probe_wavelengths(self._get_name())
            Nbins               =   len(wavelengths)

            # Read probe intensities in W/m2/micron/sr
            I_W_m2_micron_sr    =   np.array(aux.read_probe_intensities(self._get_name(),Nbins))

            # print(len(df))
            # print(len(I_W_m2_micron_sr[:,0]))
            # Convert intensities to W/m2/micron
            I_W_m2_micron       =  I_W_m2_micron_sr * 4 * np.pi

            # Integrate intensities in NIR
            N_start,N_stop      =   aux.NIR_index(wavelengths)
            if fluxOutputStyle == "Wavelength":     
                df['F_NIR_W_m2']        =   np.array([np.trapz(I_W_m2_micron[i,N_start:N_stop],x=wavelengths[N_start:N_stop]) \
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
        R_NIR_FUV_df        =   df['F_NIR_W_m2'].values/df['F_FUV_W_m2'].values
        # Minimum NIR/FUV flux ratio in SKIRT
        # (from a 1e4 Msun star particle, 1e6 yr old, Z=solar, see "SIGAME test plots.ipynb")
        #R_NIR_FUV_min_1e4_1e6yr_Z1 = 0.014145888831442749
        R_NIR_FUV_min = np.min(R_NIR_FUV_df)

        logNH_cl,R_NIR_FUV_cl   =   aux.get_NH_from_cloudy()

        #dR_NIR_FUV_df           =   R_NIR_FUV_df/R_NIR_FUV_min#_1e4_1e6yr_Z1
        #dR_NIR_FUV_cl           =   R_NIR_FUV_cl/np.min(R_NIR_FUV_cl) 
        #print(np.min(R_NIR_FUV_df),np.max(R_NIR_FUV_df))
        #print(np.min(R_NIR_FUV_cl),np.max(R_NIR_FUV_cl))
        #print(np.min(dR_NIR_FUV_df),np.max(dR_NIR_FUV_df))
        #print(np.min(dR_NIR_FUV_cl),np.max(dR_NIR_FUV_cl))
        #dR_NIR_FUV_df[dR_NIR_FUV_df < dR_NIR_FUV_cl.min()] = dR_NIR_FUV_cl.min() 
        #dR_NIR_FUV_df[dR_NIR_FUV_df > dR_NIR_FUV_cl.max()] = dR_NIR_FUV_cl.max() 
        #R_NIR_FUV_df[R_NIR_FUV_df < R_NIR_FUV_cl.min()] = R_NIR_FUV_cl.min() 
        #R_NIR_FUV_df[R_NIR_FUV_df > R_NIR_FUV_cl.max()] = R_NIR_FUV_cl.max() 
        #interp                  =   interp1d(np.log10(dR_NIR_FUV_cl),logNH_cl,fill_value='extrapolate')
        #interp                  =   interp1d(np.log10(dR_NIR_FUV_cl)[8::],logNH_cl[8::],fill_value='extrapolate',kind='slinear')
        #logNH                   =   interp(np.log10(dR_NIR_FUV_df))
        #df['NH']                =   10.**logNH
        #df['NH'][df.NH <= 10.**np.min(logNH_cl)] = 10.**np.min(logNH_cl)
        #df['NH'][dR_NIR_FUV_df >= np.max(dR_NIR_FUV_cl)] = 10.**np.max(logNH_cl)

        #fig, ax1 = plt.subplots(figsize=(10,10))
        #ax1.plot(np.log10(dR_NIR_FUV_cl)[8::],logNH_cl[8::],'o')
        #dR_NIR_FUV = np.arange(-2,5,0.1)
        #ax1.plot(dR_NIR_FUV,interp(dR_NIR_FUV),'--r',label='interp')
        #ax1.set_xlabel('log cloudy NIR/FUV flux ratio')
        #ax1.set_ylabel('log cloudy column density')
        #ax1.legend()
        #ax1.set_ylim([14,24])
        #plt.savefig('plots/NH_test.png')

        #print(np.min(df.NH),np.max(df.NH))

        # Storing NIR/FUV flux ratio here so it doesn't get overwritten
        df['R_NIR_FUV']         =   df['F_NIR_W_m2'].values/df['F_FUV_W_m2'].values

        # Normalize to Habing flux (1.6e-3 erg/cm^2/s)
        df['F_FUV_Habing']      =   df['F_FUV_W_m2'].values * 1e7 / 1e4 / 1.6e-3

        # Normalize to G0 energy density (5.29e-14 ergs/cm^3)
        # http://www.ita.uni-heidelberg.de/~rowan/ISM_lectures/galactic-rad-fields.pdf eq. 18
        df['E_FUV_ergs_cm3']    =   df['F_FUV_W_m2'].values / p.clight / 1e-7 / 1e6
        df['G0']                =   df['E_FUV_ergs_cm3'].values / 5.29e-14 # ergs/cm^3 from Peter Camps

        self.df             =   df

    def _add_Mach_number(self):
        """ Add Mach number to cells
        """

        p = copy.copy(params)

        self._add_data()

        df                  =   self.df

        if p.turb == '10':
            print('Adopting a Mach number of 10 for all dense gas cells (> 1 cm^-3)')
            df['Mach']          =   np.zeros(len(df)) + 10
            df['Mach'][df['nH'] < 1e-3] = 0

        if p.turb == 'HK17':
            print('Calculating Mach number based on velocity dispersion from Hayward & Krumholz 2017 prescription '+\
                'combined with sound speed for molecular gas at T_mol = 10 K')


        if p.turb == 'T20':
            print('Not calculating Mach number - will use Tress+20 as look-up table for the density PDFs')

        self.df             =   df

    def _add_vel_disp_on_cloud_scale(self):
        """ Add velocity dispersion on cloud scale to cell dataframe.
        """

        # Make sure cell data from SKIRT is loaded
        self._add_data()
        df                  =   self.df.copy()
        coords_cells        =   df[['x','y','z']].values

        # Get particle data for gas
        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='simgas')
        coords_sim          =   simgas[['x','y','z']].values

        # Find particles closest to each cell
        indices             =   spatial.cKDTree(coords_sim).query(coords_cells,k=64)[1]
        df.head()

        # Pick the nearest sim particle
        vel_rms_cloud       =   np.zeros(len(df))

        for i in range(len(df)):
            # coordinates of this cell
            xyz_i = coords_cells[i]
            
            # distance to neighbor sim particles
            dx = coords_sim[indices[i]] - xyz_i
            r = np.sqrt(np.sum(dx * dx, axis=1))
            
            # choose properties of nearest sim particle
            simgas_cut = simgas.iloc[indices[i]].reset_index(drop=True)
            vel_rms_cloud[i] = simgas_cut['vel_disp_cloud'][np.argmin(r)]

        df['vel_disp_cloud'] =   vel_rms_cloud

        self.df             =   df

    def _add_DTM(self):
        """ Add DTM to cell dataframe. 
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
            pass
            DTM = np.zeros(len(simgas)) + 0.4

        df['DTM']           =   DTM

        self.df             =   df

    def _add_velocity(self):
        """ Add velocity (vx,vy,xz) to cell dataframe. 
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
        """ Add metallicity on cloud scale to cell dataframe, 
        using kernel smoothing of nearest particles.
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
        """ Add SFR density on cloud scale to cell dataframe, 
        using kernel smoothing of nearest particles.
        """

        # Make sure cell data from SKIRT is loaded
        self._add_data()
        df                  =   self.df.copy()
        coords_cells        =   df[['x','y','z']].values
        V_cells             =   df[['cell_volume']].values
        
        # Get particle data for gas
        simgas              =   aux.load_temp_file(gal_ob=self.gal_ob,data_type='simgas')
        simgas['nSFR']      =   simgas['SFR'] / (4/3*np.pi*simgas.h.values**3)
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


        # Pick the nearest sim particle
        #SFR_density         =   np.zeros(len(df))

        #for i in range(len(df)):
        #    # coordinates of this cell
        #    xyz_i = coords_cells[i]
        #    
        #    # distance to neighboring sim particles
        #    dx = coords_sim[indices[i]] - xyz_i
        #    r = np.sqrt(np.sum(dx * dx, axis=1))
            
        #    # properties of neighboring sim particles
        #    simgas_cut = simgas.iloc[indices[i]].reset_index(drop=True)
        #    m = simgas_cut['m']; h = simgas_cut['h']; SFR_cut = simgas_cut['SFR']

        #    # mass-weighted average metallicity in this region, using SPH formalism:
        #    kernel = [aux.Wendland_C2_kernel(r1,h1) for r1,h1 in zip(r,h)]
        #    dens_i = sum(m * kernel) # estimate of density, not from snapshot
        #    SFR_density[i] = np.sum(m*SFR_cut*kernel)/dens_i / (V_cells[i]/1e9) # Msun/yr/kpc^3
        #    if dens_i == 0: 
        #        # If no gas nearby, just take mean SFR 
        #        SFR_density[i] = np.mean(SFR_cut) / (V_cells[i]/1e9) # Msun/yr/kpc^3

        #    if np.isnan(SFR_density[i]): 
        #        print(dens_i)
        #        print(V_cells[i])
        #        print(SFR_cut)
        #        print(m)
        #        print(kernel)
        #        a = asegsgjh

        df['SFR_density'] =   nSFR

        df['nSFR'] =   nSFR

        self.df             =   df

    def _add_f_H2(self):
        """ Add f_H2 on cloud scale to cell dataframe, 
        using KMT+09 prescription.
        """

        p = copy.copy(params)
        
        df                  =   self.get_dataframe()

        # Surface density of gas, scaling up/down to 100 pc
        # df['sigma_gas']     =   df.nH.values * (100*p.pc2cm)**3 * p.mH / p.Msun / 100**2 # Msun/pc**2
        df['surf_gas_or']   =   1.36*df.nH.values * (df.cell_size*p.kpc2cm)**3 * p.mH / p.Msun / (df.cell_size*1e3)**2
        clumping_factor     =   10*(df.cell_size*1e3-100)/100.; clumping_factor[df.cell_size*1e3 < 100] = 1
        print('Clumping factors span:')
        print('%.2f to %.2f' % (np.min(clumping_factor),np.max(clumping_factor)))
        print('For cell sizes:')
        print('%.2f to %.2f pc' % (np.min(df.cell_size*1e3),np.max(df.cell_size*1e3)))
        df['surf_gas']      =   df['surf_gas_or']*clumping_factor

        # Eq. 1 in Dave+16 (adopted from KMT+09)
        # xi                  =   0.77 * (1 + 3.1*df.Z**0.365)
        # s                   =   np.log(1 + 0.6 * xi + 0.01 * xi**2) / (0.0396 * df.Z * (df.sigma_gas))
        # df['f_H2']          =   1 - 0.75 * s / (1 + 0.25*s)

        # Eq. 2 in KMT+09 (The star formation law in atomic and molecular gas)
        xi                  =   0.77 * (1 + 3.1*df.Z**0.365)
        s                   =   np.log(1 + 0.6*xi) / (0.04 * df.surf_gas*df.Z)
        delta               =   0.0712 * (0.1/s + 0.675)**(-2.8)
        df['f_H2']          =   1 - (1 + (3/4*s/(1+delta))**(-5))**(-1/5)

        # Eq. 1 in Narayanan and Krumholz 2014 (A theory for the excitation of CO in star-forming galaxies)
        xi                  =   0.76 * (1 + 3.1*df.Z**0.365)
        tau                 =   0.066 * df.surf_gas * df.Z
        s                   =   np.log(1 + 0.6*xi + 0.01*xi**2) / (0.6 * tau)
        df['f_H2_NK14']     =   1 - 3/4 * s / (1+0.25*s)
        df['f_H2_NK14'][s > 2] = 0

        self.df   =   df

    def _add_sound_speed(self):
        """ Add sound speed in ideal gas
        """

        p = copy.copy(params)

        df                  =   self.get_dataframe()

        T_cold_gas          =   10 # K

        adiabatic_index     =   7/5. # diatomic molecules
        m_molecule          =   2*p.mH # H2 in kg
        df['sound_speed']   =   np.sqrt(adiabatic_index*p.kB*T_cold_gas/m_molecule)/1e3 # km/s

        self.df   =   df

#---------------------------------------------------------------------------
### FOR ISRF TASK ###
#---------------------------------------------------------------------------

def setup_SKIRT(gal_indices):
    """A function that creates SKIRT input, using isrf classs.
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
     
        # Save gas data in SKIRT format
        simgas_skirt = simgas.copy()
        # Calculate mass-weighted DTM ratio (total metal mass as in Dave+19)
        simgas['DTM']     =   simgas.m_dust.values/(simgas.m_dust.values + simgas.Z.values * p.mf_Z1 * simgas.m.values)
        isrf_ob.particle_data.simgas = simgas
        isrf_ob.particle_data.save_dataframe('simgas')
        mw_DTM            =   np.sum(simgas.DTM * simgas.m.values)/np.sum(simgas.m.values) 
        simgas_skirt[['x','y','z','h']] = simgas_skirt[['x','y','z','h']]*1000. # pc
        if 'DTM' in p.sim_run: 
            header = '# SPH Gas Particles\n'+\
                        '# Columns contain: x(pc) y(pc) z(pc) h(pc) M(Msun) Z(0-1)'
            simgas_skirt['Z'] = simgas_skirt['Z']*p.mf_Z1 # Asplund+09, metallicity to SKIRT is mass fraction...
            simgas_skirt['Z'] = simgas_skirt['Z'].map(lambda x: '%.6e' % x)
            simgas_skirt = simgas_skirt[['x','y','z','h','m','Z']]
        else:
            for col in ['x','y','z','h','m_dust']:
                simgas_skirt[col] = simgas_skirt[col].map(lambda x: '%.2f' % x)
            header = '# SPH Gas Particles\n'+\
                        '# Columns contain: x(pc) y(pc) z(pc) h(pc) M(Msun)'
            simgas_skirt = simgas_skirt[['x','y','z','h','m_dust']]
        skirt_filename = p.d_skirt + '%s_gas.dat' % isrf_ob._get_name()
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
        simstar_skirt['age'] = simstar_skirt['age'].values * 1e9 # yr
     
        # STARS
        header = '# Star Particles\n'+\
                        '# Columns contain: x(pc) y(pc) z(pc) h(pc) M(Msun) Z age(yr)'
        simstar_skirt[['x','y','z']] = simstar_skirt[['x','y','z']]*1000. # pc

    # test!!!!
        #index = np.ones(len(simstar_skirt))
        #index[(simstar_skirt.x > 0) & (simstar_skirt.y > 0)] = 0
        #print(len(simstar_skirt))
        #simstar_skirt = simstar_skirt[index == 1].reset_index(drop=True)
        #print(len(simstar_skirt))

        simstar_skirt['m_init'] = simstar_skirt['m'].values / np.interp(np.log10(simstar_skirt['age'].values),df['age'].values,df['mass_remaining'].values) 
        for col in ['x','y','z','m_init']:
            simstar_skirt[col] = simstar_skirt[col].map(lambda x: '%.2f' % x)
        simstar_skirt['Z'] = simstar_skirt['Z']*0.0134 # Asplund+09, metallicity to SKIRT is mass fraction...
        simstar_skirt['Z'] = simstar_skirt['Z'].map(lambda x: '%.6e' % x)
        # Simple estimate of stellar h: https://github.com/SKIRT/SKIRT9/issues/10
        m1,m2 = np.min(simstar['m']),np.max(simstar['m'])
        simstar_skirt['h'] = (simstar['m']-m1)/(m2-m1)*(300-100) + 100


        # Save old stars for Bruzual&Charlot
        skirt_filename = p.d_skirt + '%s_star_old_test.dat' % isrf_ob._get_name()
        simstar_skirt_old = simstar_skirt.copy()#[simstar_skirt.age.values > 10e6].reset_index(drop=True)
        simstar_skirt_old['age'] = simstar_skirt_old['age'].map(lambda x: '%.6e' % x)
        simstar_skirt_old[['x','y','z','h','m_init','Z','age']].to_csv(skirt_filename,header=False,index=False,sep=' ')
        def line_prepender(filename, line):
            with open(filename, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(line.rstrip('\r\n') + '\n' + content)
        line_prepender(skirt_filename,header)   

        # Edit SKIRT input file
        ski_template            =   open(p.d_skirt+'skirt_template.ski','r')
        try:
            os.remove(p.d_skirt+"G%i.ski" % gal_index)
        except:
            pass
        if p.sim_type == 'sph': ski_copy = open(p.d_skirt+'%s%s_G%i.ski' % (p.sim_name,p.sim_run,gal_index),'w')
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
    print('Now ready to run skirt: go to %s and run SKIRT'% p.d_skirt)#"qsub -J 0-N_gal run_skirt.sh"' 

def read_SKIRT(gal_index):
    """A function that reads SKIRT output, using isrf classs.
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
    An object that will derive and store the ISRF information for one galaxy.
    Child class that inherits from parent class 'galaxy'.
    """

    pass  

    def setup_tasks(self):
        '''Controls tasks to be executed, based on existing files and the overwrite [ow] parameter
        '''

        p = copy.copy(params)

        self.gal_ob = dict(zred=self.zred,galname=self.name,gal_num=self.gal_num,gal_index=self.gal_index)
        # If overwriting, do all subgridding
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
        """ Get galaxy identifier used when running SKIRT
        """

        p = copy.copy(params)

        if p.sim_type == 'sph': name = '%s%s%s_G%i' % (p.sim_name,p.sim_run,method,self.gal_index)
        if p.sim_type == 'amr': name = '%s%s%s' % (p.sim_name,p.sim_run,method)
        
        return name

    def _set_distance_to_galaxy(self):
        """
        Set distance to galaxy in Mpc (defaults to 10 Mpc for z = 0)
        """

        if self.zred == 0: 
            self.distance       =   10 
        else:
            self.distance       =   self.lum_dist 

    def run_skirt_on_galaxy(self):
        """
        Method to edit SKIRT input file and run SKIRT as subprocess.
        """


    def check_for_skirt_output(self):

        p = copy.copy(params)
        
        return(os.path.exists(p.d_skirt+'%s_rfpc_wavelengths.dat' % self._get_name()))

    def read_skirt_output(self):
        """Calculate and save galaxy integrated luminosities from SKIRT output files 
        (SED instrument), and save SKIRT grid for further use in SIGAME.
        """

        p = copy.copy(params)
        
        # Select distance to put galaxy
        self._set_distance_to_galaxy()

        if p.verbose: print('Adding galaxy-integrated luminosities to GR...')
        GR                      =   glo.global_results()

        if p.verbose: print('Saving new galaxy cell data object to store FUV flux...')
        self.cell_data._add_data()
        if (type(self.cell_data.df) == int) | (p.ow == True): 
            print('Found no cell data dataframe or "overwrite" is on, starting a new one for galaxy # %i' % self.gal_index)
            cellgas                 =   self.cell_data.start_dataframe()

        if p.verbose: print('Add FUV flux and UV/FUV ratio to cell data object')
        self.cell_data._add_FUV_flux()

        cellgas                 =   self.cell_data.save_dataframe()
 
        L_bol_sun = self._add_bol_lum()
        if p.verbose: print('Bolometric luminosity: %.4e Lsun' % self.L_bol_sun)
        GR.edit_item(self.name,'L_bol_sun',self.L_bol_sun)
 
        L_FUV_sun = self._add_FUV_lum()
        if p.verbose: print('FUV luminosity: %.4e Lsun' % self.L_FUV_sun)
        GR.edit_item(self.name,'L_FUV_sun',self.L_FUV_sun)
 
        L_TIR_sun = self._add_TIR_lum()
        if p.verbose: print('TIR luminosity: %.4e Lsun' % self.L_TIR_sun)
        GR.edit_item(self.name,'L_TIR_sun',self.L_TIR_sun)
 
        L_FIR_sun = self._add_FIR_lum()
        if p.verbose: print('FIR luminosity: %.4e Lsun' % self.L_FIR_sun)
        GR.edit_item(self.name,'L_FIR_sun',self.L_FIR_sun)
 
        cellgas                 =   self.cell_data.save_dataframe()

    def _add_bol_lum(self):
        """Calculate emitted bolometric luminosity from SKIRT output files (SED instrument)
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
        """Calculate emitted FUV luminosity face-on from SED

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
        """Calculate emitted total TIR (3-1100 microns) luminosity face-on from SED
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
        """Calculate emitted total FIR (3-1100 microns) luminosity face-on from SED
        FIR as defined in Kennicutt and Evans 2012 table 1:
         https://www.annualreviews.org/doi/abs/10.1146/annurev-astro-081811-125610
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
        L_FIR_W = F_FIR_W_m2*4*np.pi*(self.distance*1e6*p.pc2m)**2
        L_FIR_sun = L_FIR_W/p.Lsun
 
        print('FIR luminosity: %.2e Lsun' % L_FIR_sun)
        
        self.L_FIR_sun = L_FIR_sun

    def _read_SED(self,fluxOutputStyle="Wavelength",select=''):
        """ Read SED output from SKIRT

        Parameters
        ----------
        fluxOutputStyle: str
            Units for flux, default is "Wavelength": Units are F_mu (W/m2/micron)
            Alternatively, "Neutral": Units are lambda*F_lambda (W/m2)
            Alternatively, "Frequency": Units are F_lambda (W/m2/Hz) I think...

        """

        p = copy.copy(params)
        
        data = pd.read_csv(p.d_skirt+'%s%s_xy_sed.dat' % (self._get_name(),select),skiprows=7,sep=' ',engine='python',\
            names=['wavelength','tot','trans','direct_primary','scattered_primary',\
           'direct_secondary','scattered_secondary'])
        data['frequency'] = p.clight/(data['wavelength']*1e-6)

        # Convert flux to W/m^2
        if fluxOutputStyle == "Wavelength":     
            data['F_W_m2_micron']   =   data.tot.values

        if fluxOutputStyle == "Neutral":        
            data['F_W_m2']          =   data.tot.values # W/m^2

        # if fluxOutputStyle == "Frequency":      
            # data['F_W_m2'] = data['tot']*data['frequency'] # W/m^2

        return(data)

    def _get_cut_probe(self,**kwargs):
        """ Get cross section data of radiation field.
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
        print(p.d_temp + self._get_name() + filename + '.fits')
        image_file = get_pkg_data_filename(p.d_temp + self._get_name() + filename + '.fits')
        image_data,hdr = fits.getdata(image_file, ext=0, header=True)
        delete = sub.Popen('rm ' + p.d_temp + self._get_name() + filename + '.fits', shell=True)

        units = hdr['BUNIT']
        # print('Units: %s ' % units)
        return(image_data,units)

    def _get_map_inst(self,**kwargs):
        """ Get projected map of radiation field.
        """

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        if p.orientation in ['face-on','xy','yx']: ext = '_xy_map'
        if p.orientation in ['edge-on','xz','zx']: ext = '_xz_map'
        if p.orientation in ['yz','zy']: ext = '_yz_map'

        hdul = fits.open(p.d_XL_data + 'skirt/' + self._get_name() + p.select + '.fits')
        wa = hdul[1].data.field(0)
        image_data = hdul[0].data
        units = hdul[0].header['BUNIT']

        return(image_data,units,wa)

#---------------------------------------------------------------------------
### FOR RE-GRIDDING TASK ###
#---------------------------------------------------------------------------

def run_grid(gal_index):
    """A function that puts galaxy properties on a grid structure, using 
    grid class.
    """

    #print('\nNow for galaxy # %i' % gal_index)

    grid_obj      =   grid(gal_index = gal_index)
    grid_obj.setup_tasks()
    grid_obj.run()

class grid(galaxy):
    """
    An object that will derive and store the cell information for one galaxy.
    Child class that inherits from parent class 'galaxy'.
    """

    pass  


    def setup_tasks(self):
        '''Controls tasks to be executed, based on existing files and the overwrite [ow] parameter
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
            self.add_Mach_number        =   True

        # If not overwriting, check which tasks have been performed
        if not p.ow:
            self.add_nH                 =   False
            self.add_FUV_flux           =   False
            self.add_metallicity        =   False
            self.add_SFR_density        =   False
            self.add_Mach_number        =   False


            df = gal_ob.cell_data.get_dataframe()
            if 'nH' not in df.keys():
                self.add_nH                 =   True
            if 'Mach' not in df.keys():
                self.add_Mach_number        =   True
            if 'Z' not in df.keys():
                self.add_metallicity        =   True
            if 'nSFR' not in df.keys():
                self.add_SFR_density        =   True
            if 'G0' not in df.keys():
                self.add_FUV_flux           =   True

        if p.turb == 'T20': 
            self.add_Mach_number = False

        #self.add_FUV_flux           =   True
        #self.add_metallicity        =   True
        self.add_Mach_number        =   False
        #self.add_SFR_density        =   False

        if self.add_nH: print('* Derive gas density (nH) for new cells for galaxy # %i' % self.gal_index)
        if self.add_FUV_flux: print('* Derive and add FUV (and other band) flux for galaxy # %i' % self.gal_index)
        if self.add_metallicity: print('* Derive and add metallicity for galaxy # %i' % self.gal_index)
        if self.add_SFR_density: print('* Derive and add SFR density for galaxy # %i' % self.gal_index)
        if self.add_Mach_number: print('* Derive and add velocity dispersion (or Mach number) for galaxy # %i' % self.gal_index)
        if self.add_FUV_flux + self.add_Mach_number + self.add_metallicity + self.add_SFR_density + self.add_nH == 0: 
            print('Do nothing! For galaxy # %i' % self.gal_index)

    def run(self):
        """ Convert particle properties to cell grid
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

            #print('Read FUV and NIR flux (instead of running SKIRT read again)')
            
            # Load cell data
            self.cell_data._add_data()

            # Add metallicity to cell data
            self.cell_data._add_FUV_flux()
            self.cell_data.save_dataframe()
            print('done with photometry for galaxy # %i (gal_num = %i)' % (self.gal_index,self.gal_num))

        if self.add_SFR_density:

            #print('\n- Deriving SFR density for all cells -')

            # Load cell data
            self.cell_data._add_data()
        
            # Add metallicity to cell data
            self.cell_data._add_SFR_density()
            self.cell_data.save_dataframe()
            print('done with SFR density for galaxy # %i (gal_num = %i)' % (self.gal_index,self.gal_num))

        if self.add_metallicity:

            #print('\n- Deriving metallicity for all cells -')

            # Load cell data
            self.cell_data._add_data()
        
            # Add metallicity to cell data
            self.cell_data._add_metallicity()
            self.cell_data.save_dataframe()
            print('done with Z for galaxy # %i (gal_num = %i)' % (self.gal_index,self.gal_num))

        if self.add_Mach_number:

            print('\n- Deriving Mach number on cloud scales -')

            self.cell_data._add_data()
            self.cell_data._add_Mach_number()

        if self.add_f_H2:

            print('\n- Deriving f_H2 for all cells -')

            # Load cell data
            self.cell_data._add_data()

            # Add f_H2 to cell data
            self.cell_data._add_f_H2()
            self.cell_data.save_dataframe()

#---------------------------------------------------------------------------
### FOR INTERPOLATION TASK ###
#---------------------------------------------------------------------------

def run_interp(gal_index):
    """A function that interpolates Cloudy look-up tables for 
    line luminosity and other properties, using interpolation class.
    """

    #print('\nNow for galaxy # %s' % gal_index)

    interp_obj      =   interpolation(gal_index = gal_index)
    interp_obj.setup_tasks()
    if interp_obj.do_interp: 
        interp_obj.run()
    else:
        print('Galxy # %i already interpolated!' % gal_index)

def add_mw_quantities(**kwargs):

    print('Add mw-quantities for all galaxies')

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
    P_gas_mw = np.zeros(GR.N_gal)
    ne_mw = np.zeros(GR.N_gal)
    Te_mw = np.zeros(GR.N_gal)
    Tk_mw = np.zeros(GR.N_gal)
    for i in range(GR.N_gal):
        gal_ob = galaxy(gal_index = i)
        df = gal_ob.particle_data.get_dataframe('simgas')
        nH_min[i] = np.min(df.nH.values[df.nH.values > 0])
        nH_max[i] = np.max(df.nH.values)
        df = gal_ob.cell_data.get_dataframe()
        G0_mw[i] = np.sum(df.G0.values*df.m.values)/np.sum(df.m.values)
        nH_mw[i] = np.sum(df.nH.values*df.m.values)/np.sum(df.m.values)
        df = gal_ob.particle_data.get_dataframe('simgas')
        h_min[i] = np.min(df.h.values)
        M_dust_DTM[i] = np.sum(df.m.values*df.Z.values*p.mf_Z1*0.5)
        # stars
        df = gal_ob.particle_data.get_dataframe('simstar')
        age_mw[i] = np.sum(df.age.values*df.m.values)/np.sum(df.m.values)
        Zstar_mw[i] = np.sum(df.Z.values*df.m.values)/np.sum(df.m.values)
        # cell data
        df = gal_ob.cell_data.get_dataframe()
        df_or = df.copy()
        # Check DTM values for nans...
        DTM = df.DTM.values
        DTM[np.isnan(DTM)] = 10.**np.min(logDTMs)
        DTM[DTM == 0] = 1e-30
        df                      =   df_or.copy()[['nH','Z','G0','R_NIR_FUV','nSFR','DTM']]
        df.Z[np.isnan(df.Z)]    =   1e-6 # because something weird comes out from Z evaluation step
        df.nSFR[df.nSFR == 0]    =   1e-30 
        df.nSFR[np.isnan(df.nSFR)]    =   -30 
        # IMPORTANT: Taking the log of the entire dataframe and re-naming columns
        df                      =   np.log10(df)
        #print(np.min(df.DTM),np.max(df.DTM))
        df = df.rename(columns={'nH':'lognH','Z':'logZ','G0':'logFUV','nSFR':'lognSFR','DTM':'logDTM'})
        # Convert NIR/FUV ratio to column density NH
        logNH_cl,R_NIR_FUV_cl = aux.get_NH_from_cloudy()
        R_NIR_FUV_df = df_or.R_NIR_FUV.values
        interp                  =   interp1d(np.log10(R_NIR_FUV_cl)[8::],logNH_cl[8::],fill_value='extrapolate',kind='slinear')
        df['logNH']             =   interp(np.log10(R_NIR_FUV_df))
        df['logNH'][df.logNH <= np.min(logNH_cl)] = np.min(logNH_cl)
        df['logNH'][df.logNH >= np.max(logNH_cl)] = np.max(logNH_cl)
        df['logNH'][np.isinf(R_NIR_FUV_df)] = np.max(logNH_cl)
        for _ in p.interp_params:
            df[_][df[_] <= np.min(lookup_table[_+'s'].values)] = np.min(lookup_table[_+'s'].values) + 1e-6 * np.abs(np.min(lookup_table[_+'s']))
            df[_][df[_] >= np.max(lookup_table[_+'s'].values)] = np.max(lookup_table[_+'s'].values) - 1e-6 * np.abs(np.min(lookup_table[_+'s']))
 
        # Cell properties used for interpolation in cloudy grid models:
        #print('Now column-stacking relevant cell data')
        cell_prop               =   np.column_stack((df.lognH.values,df.lognSFR.values,df.logNH.values,df.logFUV.values,df.logZ.values,df.logDTM.values))        
        df_or['Te_mw'] = gal_ob.cell_data._interpolate_cloudy_table(lookup_table,'Te_mw',cell_prop)
        df_or['Tk_mw'] = gal_ob.cell_data._interpolate_cloudy_table(lookup_table,'Tk_mw',cell_prop)
        df_or['P_HII'] = df_or['Te_mw'].values*df_or['ne_mw_grid'].values
        df_or['P_gas'] = df_or['Tk_mw'].values*df_or['nH'].values
        gal_ob.cell_data.df                 =   df_or
        gal_ob.cell_data.save_dataframe()
        df = gal_ob.cell_data.get_dataframe()
        M_1e3[i] = np.sum(df.m.values*df.mf_1e3_grid.values)
        M_1e1[i] = np.sum(df.m.values*df.mf_1e1_grid.values)
        nH_cell_max[i] = np.max(df.nH)
        cell_size_min[i] = np.min(df.cell_size)
        cell_size_max[i] = np.max(df.cell_size)
        ne_mw[i] = np.sum(df_or.ne_mw_grid*df.mf_HII_grid*df.m)/np.sum(df.mf_HII_grid*df.m)
        Te_mw[i] = np.sum(df_or.Te_mw*df.mf_HII_grid*df.m)/np.sum(df.mf_HII_grid*df.m)
        Tk_mw[i] = np.sum(df_or.Tk_mw*df.mf_HII_grid*df.m)/np.sum(df.mf_HII_grid*df.m)
        P_HII_mw[i] = np.sum(df.P_HII*df.mf_HII_grid*df.m)/np.sum(df.mf_HII_grid*df.m)
        P_gas_mw[i] = np.sum(df.P_gas*df.m)/np.sum(df.m)

    GR.add_column('G0_mw',G0_mw)
    GR.add_column('nH_cell_mw',nH_mw)
    GR.add_column('h_min',h_min)
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
    GR.add_column('Te_mw',Te_mw)
    GR.add_column('Tk_mw',Tk_mw)
    GR.add_column('P_HII_mw',P_HII_mw)
    GR.add_column('P_gas_mw',P_gas_mw)

class interpolation(galaxy):
    """
    An object that will interpolate in cloudy look-up tables for the line luminosities.
    Child class that inherits from parent class 'galaxy'.
    """

    pass  

    def setup_tasks(self):
        '''Controls tasks to be executed, based on existing files and the overwrite [ow] parameter
        '''

        p = copy.copy(params)

        GR                      =   glo.global_results(verbose=True)
        
        self.gal_ob = dict(zred=self.zred,galname=self.name,gal_index=self.gal_index)

        df = self.cell_data.get_dataframe()

        self.do_interp = False
        self.read_skirt = False
        if p.ow:
            self.do_interp = True
        if not p.ow:
            try:
                #Check if the last line was recorded
                if (getattr(GR,'L_%s_sun' % p.lines[-1])[self.gal_index] == 0):
                #if 'vx' not in df:
                #    print('missing vx!')
                    self.do_interp = True
            except:
                print('Line luminosities not registered in GR, will run interpolation')
                self.do_interp = True
            if 'L_[CII]158' not in df.keys():
                self.do_interp = True

    def run(self):

        # Interpolate in Cloudy models
        self.cell_data._do_interpolation()

#===========================================================================
# FOR PAPER
#---------------------------------------------------------------------------

def write_table():

    p = copy.copy(params)

    GR                      =   glo.global_results(verbose=True)

    with open(p.d_data + 'tables/table_sample_prop.tex','w') as file:

        for i in range(GR.N_gal):
            file.write('%i & %.4f & %.4f & %.2f & %.2f & %.2f \\\\ \n' % (i,\
                GR.M_star_caesar[i]/1e10,\
                GR.M_gas_caesar[i]/1e10,\
                GR.SFR[i],\
                GR.Zsfr[i],\
                GR.Zmw[i]
                ))

    file.close()


    with open(p.d_data + 'tables/table_sample_lines.tex','w') as file:

        for i in range(GR.N_gal):
            file.write('%i & %.2e & %.2e & %.2e & %.2e & %.2e & %.2e & %.2e & %.2e & %.2e & %.2e & %.2e \\\\ \n' % (i,\
                getattr(GR,'L_[CII]158_sun')[i],\
                getattr(GR,'L_[CI]610_sun')[i],\
                getattr(GR,'L_[CI]370_sun')[i],\
                getattr(GR,'L_[NII]205_sun')[i],\
                getattr(GR,'L_[NII]122_sun')[i],\
                getattr(GR,'L_[OI]145_sun')[i],\
                getattr(GR,'L_[OI]63_sun')[i],\
                getattr(GR,'L_[OIII]88_sun')[i],\
                getattr(GR,'L_CO(1-0)_sun')[i],\
                getattr(GR,'L_CO(2-1)_sun')[i],\
                getattr(GR,'L_CO(3-2)_sun')[i]
                ))

    file.close()

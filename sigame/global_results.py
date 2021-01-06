"""
Module: global_results
"""

# Import other SIGAME modules
import sigame.auxil as aux

# Import other modules
import numpy as np
import pandas as pd
import pdb as pdb
import os
import glob
import copy
import sys  
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

global params
params              =   aux.load_parameters()

# Needs update: why does Zsfr turn out nan sometimes?
class global_results:
    '''An object referring to the global results of a selection of galaxies, containing global properties of as attributes.

    Example
    -------
    >>> import global_results as glo
    >>> GR = glo.global_results()
    >>> GR.print_results()
    '''

    def __init__(self, **kwargs):

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        # get global results DataFrame
        GR                      =   self._get_file(**kwargs)

        # print(GR.keys())
        # print(GR[['gal_num','SFR','M_gas','Zsfr','L_[CII]158_sun']][GR.gal_num == 9749])
        # print(GR[['gal_num','SFR','M_gas','Zsfr','L_[CII]158_sun']][GR.gal_num == 515])

        # add DF entries to global_results instance
        for key in GR:
            setattr(self,key,GR[key].values)

        setattr(self,'N_gal',len(GR['galnames']))

    def _get_file(self, **kwargs):

        p = copy.copy(params)
        for key,val in kwargs.items():
            setattr(p,key,val)

        # get filename
        filename    =   self.__get_file_location(**kwargs)

        if p.verbose: 
            print("Filename in global_results._get_file(): ")
            print(filename)

        # create file if it doesn't exist
        #f = pd.read_pickle(filename)
        if not os.path.isfile(filename):
            print("\ncould not find file at %s \n... creating Global Results file!" % filename)
            if p.sim_name == 'simba': self.__create_selection_from_simba(**kwargs)
            if p.sim_name == 'enzo': self.__create_selection_from_enzo(**kwargs)

        if p.overwrite:
            print("\n Overwriting Global Results file: %s" % filename)
            if p.sim_name == 'simba': self.__create_selection_from_simba(**kwargs)
            if p.sim_name == 'enzo': self.__create_selection_from_enzo(**kwargs)

        return pd.read_pickle(filename)

    def __get_file_location(self,**kwargs):

        p = copy.copy(params)

        for key,val in kwargs.items():
            setattr(p,key,val)

        file_location   =   p.d_data + 'results/%s_%sgals_%s%s%s' % (p.z1,p.nGal,p.sim_name,p.sim_run,p.grid_ext+p.table_ext)
        #print(file_location)
        return file_location

    def __create_selection_from_simba(self,**kwargs):

        p = copy.copy(params)

        # Open galaxy sample extracted
        galsample       =   pd.read_pickle(p.d_data+'galaxy_selection/%s_galsample_%i_%s%s' % (p.z1,p.nGal,p.sim_name,p.sim_run))
        
        # Sort according to stellar mass from caesar
        galsample       =   galsample.sort_values(['M_star_caesar'], ascending=True).reset_index(drop=True)
        nGal_or         =   len(galsample)
        # Convert sample info to dictionary
        GR              =   {}
        for key in galsample.keys():
            GR[key]         = galsample[key].values

        # Let's assume that all galaxies are at the same redshift
        GR['zreds']     =   np.zeros(len(galsample)) + p.zred

        # Add global info
        SFR             =   np.zeros(len(galsample))
        Zsfr            =   np.zeros(len(galsample))
        M_dust          =   np.zeros(len(galsample))
        M_gas           =   np.zeros(len(galsample))
        M_star          =   np.zeros(len(galsample))
        R_max           =   np.zeros(len(galsample))
        R2_gas          =   np.zeros(len(galsample))
        R2_star         =   np.zeros(len(galsample))
        SFRsd           =   np.zeros(len(galsample))
        for i in range(len(galsample)):

            # Galaxy object
            rawgalname      =   'z%.2f_' % p.zred + p.sim_run + '_%i' % galsample.gal_num[i]
            gal_ob          =   dict( galname=rawgalname, zred=p.zred, gal_index=i, gal_num=galsample.gal_num[i]) # dummy gal object
            
            # Stars
            simstar         =   aux.load_temp_file(gal_ob=gal_ob, data_type='rawsimstar', gal_ob_present=True)
            print(simstar.keys())
            M_star[i]       =   np.sum(simstar['m'].values)
            SFR[i]          =   np.sum(simstar['m'].values[simstar['age']*1e9 < 100e6])/100e6

            # Gas and dust
            simgas          =   aux.load_temp_file(gal_ob=gal_ob, data_type='rawsimgas', gal_ob_present=True)
            print(simgas.keys())

            simgas['nH']    =   0.75 * simgas['nH'].values # So gas density > H density
            M_dust[i]       =   np.sum(simgas['m_dust'].values)
            M_gas[i]        =   np.sum(simgas['m'].values)
            Zsfr[i]         =   np.sum(simgas['Z'].values*simgas['SFR'].values)/np.sum(simgas['SFR'].values)
            radii           =   np.sqrt(simgas.x.values**2 + simgas.y.values**2 + simgas.z.values**2)
            m_gas           =   simgas.m.values[np.argsort(radii)]
            radii           =   radii[np.argsort(radii)]
            m_cum           =   np.cumsum(m_gas)
            R2_gas[i]       =   radii[m_cum > 0.5*M_gas[i]][0]
            radii           =   np.sqrt(simstar.x.values**2 + simstar.y.values**2 + simstar.z.values**2)
            m_star          =   simstar.m.values[np.argsort(radii)]
            radii           =   radii[np.argsort(radii)]
            m_cum           =   np.cumsum(m_star)
            R2_star[i]      =   radii[m_cum > 0.5*M_star[i]][0]
            radii           =   np.sqrt(simstar.x.values**2 + simstar.y.values**2 + simstar.z.values**2)
            SFRsd[i]        =   np.sum(simstar['m'].values[(radii < R2_star[i]) &\
                                        (simstar['age']*1e9 < 100e6)])/100e6/(np.pi*R2_star[i]**2)
            # print(R2_gas[i],R2_star[i])

            # Max radius
            r_star = np.sqrt(simstar.x**2 + simstar.y**2 + simstar.z**2)
            r_gas = np.sqrt(simgas.x**2 + simgas.y**2 + simgas.z**2)
            R_max[i]        =   np.max(np.append(r_star,r_gas))

            # Move dataframe to particle_data folder
            aux.save_temp_file(simgas, gal_ob=gal_ob, data_type='simgas', gal_ob_present=True)
            aux.save_temp_file(simstar, gal_ob=gal_ob, data_type='simstar', gal_ob_present=True)

        GR['M_star']    =   M_star
        GR['M_gas']     =   M_gas
        GR['M_dust']    =   M_dust
        GR['SFR']       =   SFR
        GR['Zsfr']      =   Zsfr
        GR['R_max']     =   R_max
        GR['R2_gas']    =   R2_gas
        GR['R2_star']   =   R2_star
        GR['SFRsd']     =   SFRsd

        # convert dictionary to DataFrame
        GR              =   pd.DataFrame(GR)
        GR              =   GR[GR.SFR > 0].reset_index(drop=True)
        print('Final number of galaxies:')
        print(len(GR))

        print('Check min and max of SFR (<100 Myr):')
        print(np.min(GR.SFR), np.max(GR.SFR))

        # Add names
        GR['galnames']  =   ['G%i' % i for i in np.arange(len(GR))]

        for line in p.lines:
            GR = self.__set_attr(GR,'L_'+line)

        for attr in ['lum_dist', 'Zmw']:
            print('get attributes: %s' %attr)
            GR = self.__set_attr(GR,attr)

        # Save DF of global results
        filename    =   self.__get_file_location(nGal=len(GR))
        print("Filename in global results __create_file: ")
        print(filename)
        GR.to_pickle(filename)

        if len(GR) != nGal_or:
            print('Stop and update parameter.txt to reflect new nGal')
            sys.exit() 

        return GR

    def __set_attr(self,GR_int,attr):
        # Get missing attributes

        p = copy.copy(params)

        if attr == 'lum_dist':
            LD                          =   aux.get_lum_dist(GR_int['zreds'])
            GR_int['lum_dist']          =   LD

        for gal_index in range(len(GR_int.SFR)):
            gal_ob = dict(galname=GR_int['galnames'][gal_index], zred=GR_int['zreds'][gal_index], gal_index=gal_index, gal_num=GR_int['gal_num'][gal_index]) # dummy gal object

            if attr == 'Zmw':
                if gal_index == 0:
                    Zmw                     =   np.zeros(len(GR_int.SFR))
                simgas  =   aux.load_temp_file(gal_ob=gal_ob, data_type='simgas', gal_ob_present=True)
                Zmw[gal_index] = np.sum(simgas['Z']*simgas['m'])/np.sum(simgas['m'])
                GR_int['Zmw']               =    Zmw

            if attr == 'Zsfr':
                if gal_index == 0:
                    Zsfr                     =   np.zeros(GR_int['N_gal'][0])
                simgas  =   aux.load_temp_file(gal_ob=gal_ob, data_type='simgas', gal_ob_present=True)
                Zsfr[gal_index] = np.sum(simgas['Z']*simgas['SFR'])/np.sum(simgas['SFR'])
                GR_int['Zsfr']               =    Zsfr


        return(GR_int)

    def __create_selection_from_enzo(self,**kwargs):

        p = copy.copy(params)

        simgas = np.load(p.d_XL_data + 'data/sim_data/%s%s_gas.npy' % (p.sim_name,p.sim_run))
        simstar = np.load(p.d_XL_data + 'data/sim_data/%s%s_star.npy' % (p.sim_name,p.sim_run))

        M_gas = np.sum(simgas[:,6])
        Z = np.sum(simgas[:,5])
        Zmw = np.sum(Z*M_gas)/M_gas
        Zsfr = Zmw#np.sum(Z*SFR)/np.sum(SFR)
        M_star = np.sum(simstar[:,3])
        age = np.sum(simstar[:,5])
        SFR = np.sum(M_star[age <= 100e6]/100e6)
        GR = pd.DataFrame({'M_gas':np.array([M_gas]),'M_star':np.array([M_star]),'SFR':np.array([SFR]),'Zsfr':np.array([Zsfr]),'Zmw':np.array([Zmw]),'gal_num':np.array([0])})
        GR['galnames']  =   ['G0']
        GR['zreds']     =   np.array([0])
        GR['lum_dist']  =   aux.get_lum_dist(GR['zreds'])
        r = np.sqrt(simgas[:,0]**2 + simgas[:,1]**2 + simgas[:,2]**2)
        GR['R_gal']     =   np.array([np.max(r)]) # change this
        GR['R_max']     =   np.array([np.max(r)])
        GR['file_name'] =   p.sim_run
        # for now:
        GR['alpha']     = 0
        GR['beta']      = 0
        # Save DF of global results
        filename    =   self.__get_file_location(nGal=len(GR))
        print("Filename in global results __create_file: ")
        print(filename)
        GR.to_pickle(filename)

        # Save as dataframes
        simgas = pd.DataFrame({\
                    'x':simgas[:,0],\
                    'y':simgas[:,1],\
                    'z':simgas[:,2],\
                    'dx':simgas[:,3],\
                    'nH':simgas[:,4],\
                    'Z':simgas[:,5],\
                    'm':simgas[:,6],\
                    'grid_level':simgas[:,7]})
        simgas.to_pickle(p.d_XL_data + 'data/sim_data/z' + '{:.2f}'.format(p.zred) + '_%s%s.simgas' % (p.sim_name,p.sim_run))
        simstar = pd.DataFrame({\
                    'x':simstar[:,0],\
                    'y':simstar[:,1],\
                    'z':simstar[:,2],\
                    'm':simstar[:,3],\
                    'Z':simstar[:,4],\
                    'age':simstar[:,5]})
        simstar.to_pickle(p.d_XL_data + 'data/sim_data/%s_.simstar' % p.sim_run)
        simstar.to_pickle(p.d_XL_data + 'data/sim_data/z' + '{:.2f}'.format(p.zred) + '_%s%s.simstar' % (p.sim_name,p.sim_run))

        return(GR)

    def get_gal_index(self,gal_num=0):

        GR = self._get_file(overwrite=False)

        gal_nums = getattr(GR,'gal_num')
        indices = np.arange(len(gal_nums))

        return(indices[gal_nums == gal_num][0])

    def get_gal_num(self,gal_index=0):

        GR = self._get_file(overwrite=False)

        gal_nums = getattr(GR,'gal_num')
        indices = np.arange(len(gal_nums))

        return(gal_nums[indices == gal_index])

    def get_attr(self,name):

        GR = self._get_file(overwrite=False)

        value = getattr(GR,name)

        return(value)

    def edit_item(self,galname,name,value):

        p = copy.copy(params)

        GR = self._get_file(overwrite=False)

        if not hasattr(GR,name): GR[name] = np.zeros(len(GR))

        values          =   GR.copy()[name]
        values[GR['galnames'] == galname] = value
        GR[name]        =   values

        self.save_results(GR)

    def add_column(self,name,values):

        p = copy.copy(params)

        GR = self._get_file(overwrite=False)

        GR[name] = values

        self.save_results(GR)

    def save_results(self,GR):

        file_location = self.__get_file_location()

        GR.to_pickle(file_location)

    def print_header(self):

        GR = self._get_file(overwrite=False)

        print(GR.head())

    def print_all(self):

        GR = self._get_file(overwrite=False)

        print(GR)

    def print_results(self):

        p = copy.copy(params)

        # Rerun global results file creating
        GR = self._get_file(overwrite=False)     # maybe overwrite=True to ensure we get latest numbers??

        print('\n BASIC GALAXY INFO')

        aux.pretty_print(\
            ['Name','z','Mstar (caesar)','SFR','R','Lum dist'],\
            ['','','[10^10 Msun]','[Msun/yr]','[kpc]','[Mpc]'],\
            ['s','f','e','f','f','f','f'],\
            [GR.galnames, GR.zreds, GR.M_star/1e10, GR.SFR, GR.R_gal, GR.lum_dist]
            )
        #GR.SFRsd/p.SFRsd_MW, 

        try:
            print('\n ISM PROPERTIES')
            aux.pretty_print(\
                ['Name','Gas mass','Molecular mass', 'f_mol','mw-Z','SFR-Z'],\
                ['','[10^10 Msun]','[10^10 Msun]','[%]','[Zsun]','[Zsun]'],\
                ['s','e','e','f','f','f'],\
                [GR.galnames, GR.M_gas/1e10, GR.M_mol/1e10,GR.M_mol/GR.M_gas*100.,GR.Zmw,GR.Zsfr]
                )
        except:
            print('cannot print ISM properties yet')

        try:
            print('\n BROAD BAND LUMINOSITIES FROM SKIRT')
            aux.pretty_print(\
                ['Name','L_FUV','L_TIR', 'L_bol'],\
                ['','[Lsun]','[Lsun]','[Lsun]'],\
                ['s','e','e','e'],\
                [GR.galnames, GR.L_FUV_sun, GR.L_TIR_sun,GR.L_bol_sun]
                )

            print('Min max L_FUV: %.2e %.2e' % (np.min(GR.L_FUV_sun),np.max(GR.L_FUV_sun)))
            print('Min max L_TIR: %.2e %.2e' % (np.min(GR.L_TIR_sun),np.max(GR.L_TIR_sun)))
            print('Min max L_bol: %.2e %.2e' % (np.min(GR.L_bol_sun),np.max(GR.L_bol_sun)))
        except:
            print('cannot print broad band luminosities yet')

        try:
            print('\n LINE EMISSION')
            aux.pretty_print(\
                ['Name','L_[CII]158','L_CO(1-0)','L_[OI]63','L_[OIII]88'],\
                ['','[Lsun]','[Lsun]','[Lsun]','[Lsun]'],\
                ['s','e','e','e','e'],\
                [GR.galnames, GR['L_[CII]158_sun'], GR['L_CO(1-0)_sun'], GR['L_[OI]63_sun'], GR['L_[OIII]88_sun']]
                )
        except:
            print('cannot print line luminosities yet')

    def print_galaxy_properties(self,**kwargs):

        args        =   dict(gal_index=0)
        args        =   aux.update_dictionary(args,kwargs)
        for key,val in args.items():
            exec(key + '=val')

        print('\nProperties of Galaxy number %s, %s, at redshift %s' % (gal_index+1,self.galnames[gal_index],self.zreds[gal_index]))

        # Print these properties
        print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
        print('|%20s|%20s|%15s|' % ('Property'.center(20), 'Value'.center(20), 'Name in code'.center(15)))
        print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))
        print('|%20s|%20s|%15s|' % ('Redshift'.center(20), '{:.3f}'.format(self.zreds[gal_index]), 'zred'.center(15)))
        print('|%20s|%20s|%15s|' % ('Radius'.center(20), '{:.3f}'.format(np.max(self.R_gal[gal_index])), 'R_gal'.center(15)))
        print('|%20s|%20s|%15s|' % ('Stellar mass'.center(20), '{:.3e}'.format(self.M_star[gal_index]), 'M_star'.center(15)))
        print('|%20s|%20s|%15s|' % ('ISM mass'.center(20), '{:.3e}'.format(self.M_gas[gal_index]), 'M_gas'.center(15)))
        print('|%20s|%20s|%15s|' % ('Dense gas mass fraction'.center(20), '{:.3e}'.format(self.f_dense[gal_index]*100.), 'f_dense'.center(15)))
        print('|%20s|%20s|%15s|' % ('DM mass'.center(20), '{:.3e}'.format(self.M_dm[gal_index]), 'M_dm'.center(15)))
        print('|%20s|%20s|%15s|' % ('SFR'.center(20), '{:.3f}'.format(self.SFR[gal_index]), 'SFR'.center(15)))
        print('|%20s|%20s|%15s|' % ('SFR surface density'.center(20), '{:.4f}'.format(self.SFRsd[gal_index]), 'SFRsd'.center(15)))
        print('|%20s|%20s|%15s|' % ('SFR-weighted Z'.center(20), '{:.4f}'.format(self.Zsfr[gal_index]), 'Zsfr'.center(15)))
        print('+%20s+%20s+%15s+' % ((20*'-'), (20*'-'), (15*'-')))




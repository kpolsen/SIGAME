# coding=utf-8
"""
Submodule: param
"""

import linecache as lc
import numpy as np
import re
import pandas as pd
import periodictable as per
import matplotlib.colors as colors
import pdb
from argparse import Namespace
import pickle as pickle
import os as os

class read_params:
    """ Read parameters and save as object
    """

    def __init__(self,params_file):


        #===========================================================================
        """ Read from parameters_z*.txt """
        #---------------------------------------------------------------------------


        file                =   open(params_file,'r', encoding="utf8")
        lc.clearcache()

        parameter_list      =   ['nGal','ow','halos','snaps','haloIDs','z1',\
                                'hubble','omega_m','omega_lambda','omega_r',\
                                'sim_type','sim_name','sim_run','d_sim',\
                                'v_res','v_max','x_res_pc','x_max_pc','inc_dc',\
                                'target','lines',\
                                'd_XL_data','N_cores','N_param','turb','grid_ext','table_ext']

        for i,line in enumerate(file):
            for parameter in parameter_list:
                if line.find('['+parameter+']') >= 0:
                    setattr(self,parameter,re.sub('\n','',lc.getline(params_file,i+2)))
        file.close()
        if self.d_XL_data == '': self.d_XL_data = os.getcwd() + '/'
        self.d_skirt        =   self.d_XL_data + 'skirt/'
        print('will look for skirt input/output data in %s\n' % (self.d_skirt))


        #===========================================================================
        """ Set up directories """
        #---------------------------------------------------------------------------

        # Where large data is stored (alternative SIGAME_dev folder)
        self.parent         =   os.getcwd() + '/'
        self.d_data         =   os.getcwd() + '/data/'
        self.d_table        =   os.getcwd() + '/look-up-tables/'
        self.d_cloudy       =   self.d_XL_data + '/cloudy/'
        if self.grid_ext == '_ext': 
            self.d_cloudy       =   self.d_XL_data + '/cloudy/ext/'
        self.d_temp         =   os.getcwd() + '/sigame/temp/'
        self.d_plot         =   os.getcwd() + '/plots/'

        print(self.d_temp)
        print('\nwill look for code in %s%s' % (os.getcwd(), '/sigame/'))
        print('will look for sim data in %s' % (self.d_data))
        print('will look for cloudy data in %s' % (self.d_cloudy))

        #===========================================================================
        """ Convert parameters to int/float/lists C """
        #---------------------------------------------------------------------------

        self.zred          =   float(self.z1)
        self.nGal          =   int(self.nGal)
        if self.ow == 'yes': self.ow = True
        if self.ow == 'no': self.ow = False
        self.z1            =   'z'+str(int(self.z1))
        self.hubble        =   float(self.hubble)
        self.omega_m       =   float(self.omega_m)
        self.omega_lambda  =   float(self.omega_lambda)
        self.omega_r       =   float(self.omega_r)
        # self.frac_h        =   float(self.frac_h)
        # self.f_R_gal       =   float(self.f_R_gal)
        self.v_res         =   float(self.v_res)
        self.v_max         =   float(self.v_max)
        self.x_res_pc      =   float(self.x_res_pc)
        self.x_max_pc      =   float(self.x_max_pc)
        self.N_cores       =   int(self.N_cores)
        self.N_param       =   int(self.N_param)

        # Always take z-direction as Line-of-Sight
        self.los_dc = 'z'
        # By how much should the galaxy datacube be rotated from face-on around y axis?
        if hasattr(self,'inc_dc'):
            self.inc_dc         =   re.findall(r'\w+',self.inc_dc)
        if hasattr(self,'haloIDs'):
            haloIDs             =   re.findall(r'\w+',self.haloIDs)
            self.haloIDs        =   [int(haloIDs[i]) for i in range(0,len(haloIDs))]
        if hasattr(self,'lines'):
            lines               =   self.lines
            # lines               =   lines.replace('[','')
            # lines               =   lines.replace(']','')
            lines               =   lines.replace(' ','')
            lines               =   lines.replace("'","")
            lines               =   lines.split(',')
            self.lines     =   lines
        if hasattr(self,'target'):
            lines               =   self.target
            lines               =   lines.replace('[','')
            lines               =   lines.replace(']','')
            lines               =   lines.replace(' ','')
            lines               =   lines.replace("'","")
            # lines               =   lines.split(',')
            self.target         =   lines


        #===========================================================================
        """ Run options for SÍGAME """
        #---------------------------------------------------------------------------

        run_options         =   ['step1_setup_SKIRT',\
                                 'step1_read_SKIRT',\
                                 'step2_grid_gas',\
                                 'step3_setup_Cloudy_grid',\
                                 'step3_run_Cloudy',\
                                 'step3_combine_Cloudy',\
                                 'step3_complete_Cloudy',\
                                 'step3_read_Cloudy_grid',\
                                 'step3_make_Cloudy_table',\
                                 'step4_interpolate',\
                                 'step4_derived_results']

        file                =   open(params_file,'r', encoding="utf8")
        lc.clearcache()
        for i,line in enumerate(file):
            for run_option in run_options:
                if line.find(run_option) >= 0:
                    line1                   =   lc.getline(params_file,i+1)
                    setattr(self,run_option,False)
                    if line1[0:2] == '+1': setattr(self,run_option,True)

        #===========================================================================
        """ Print chosen parameters """
        #---------------------------------------------------------------------------

        print('\n' + (' Parameters chosen').center(20+10+10+10))
        print('+%20s+%10s+%15s+%50s+' % ((20*'-'), (10*'-'), (15*'-'), (50*'-')))
        print('|%20s|%10s|%15s|%50s|' % ('Parameter'.center(20), 'Value'.center(10), 'Name in code'.center(15), 'Explanation'.center(50)))
        print('+%20s+%10s+%15s+%50s+' % ((20*'-'), (10*'-'), (15*'-'), (50*'-')))
        print('|%20s|%10g|%15s|%50s|' % ('Repr. redshift'.center(20), self.zred, 'zred'.center(15),'Redshift of simulation snapshot'.center(50)))
        print('|%20s|%10g|%15s|%50s|' % ('# galaxies'.center(20), self.nGal, 'nGal'.center(15),'Number of galaxies in redshift sample'.center(50)))
        print('|%20s|%10s|%15s|%50s|' % ('Sim name'.center(20), self.sim_name, 'sim_name'.center(15),'Simulation name'.center(50)))
        print('|%20s|%10s|%15s|%50s|' % ('Sim run'.center(20), self.sim_run, 'sim_run'.center(15),'Simulation run'.center(50)))
        print('|%20s|%10s|%15s|%50s|' % ('Grid ext'.center(20), self.grid_ext, 'grid_ext'.center(15),'Cloudy grid extension'.center(50)))
        print('|%20s|%10s|%15s|%50s|' % ('Table ext'.center(20), self.table_ext, 'table_ext'.center(15),'Cloudy look-up table extension'.center(50)))

        print('+%20s+%10s+%15s+%50s+' % ((20*'-'), (10*'-'), (15*'-'), (50*'-')))

        print('\nThis is what sigame.run() is set up to do (change in parameter file):')
        if self.step1_setup_SKIRT:            print('- Generate input scripts for SKIRT radiative transfer')
        if self.step1_read_SKIRT:             print('- Read SKIRT output')
        if self.step2_grid_gas:               print('- Re-grid simulated gas data on cell structure')
        if self.step3_setup_Cloudy_grid:      print('- Generate input scripts for Cloudy ionization equilibrium models')
        if self.step3_run_Cloudy:             print('- Run Cloudy')
        if self.step3_combine_Cloudy:         print('- Combine Cloudy otuput')
        if self.step3_complete_Cloudy:        print('- Complete Cloudy grid')
        if self.step3_read_Cloudy_grid:       print('- Combine Cloudy output, debug or read Cloudy grid')
        if self.step3_make_Cloudy_table:      print('- Sample Cloudy grid in look-up table')
        if self.step4_interpolate:            print('- Interpolate in cloudy look-up table for each gas cell')
        if self.step4_derived_results:        print('- Add derived results to global results file')

        #===========================================================================
        """ Constants and variables used by SIGAME """
        #---------------------------------------------------------------------------

        self.Tkcmb       			=   2.725*(1+float(self.zred))      # CMB temperature at this redshift
        self.G_grav      			=   6.67428e-11                          # Gravitational constant [m^3 kg^-1 s^-2]
        self.clight      			=   299792458                            # Speed of light [m/s]
        self.hplanck     			=   4.135667662e-15                      # Planck constant [eV*s]
        self.Ryd                   =   13.60569253                          # [eV]
        self.eV_J                  =   1.6021766208e-19                     # [J]
        self.Habing         	    =   1.6e-3                               # ergs/cm^2/s
        self.Msun        			=   1.989e30                             # Mass of Sun [kg]
        self.Lsun        			=   3.839e26                             # Bol. luminosity of Sun [W]
        self.kB          			=   1.381e-23                            # Boltzmanns constant [J K^-1]
        self.kB_ergs          		=   1.3806e-16                           # Boltzmanns constant [ergs K^-1]
        self.b_wien                =   2.8977729e-3                         # Wien's displacement constant [m K]
        self.kpc2m       			=   3.085677580666e19                    # kpc in m
        self.pc2m        			=   3.085677580666e16                    # pc in m
        self.kpc2cm      			=   3.085677580666e21                    # kpc in cm
        self.pc2cm       			=   3.085677580666e18                    # pc in cm
        self.au          			=   per.constants.atomic_mass_constant   # atomic mass unit [kg]
        self.freq                   =   {'[CII]158':1900.5369,\
                                        '[CI]370':809.34197,\
                                        '[CI]610':492.160651,\
                                        '[OI]63':4744.774906758,\
                                        '[OI]145':2060.06909,\
                                        '[OIII]88':3393.006224818,\
                                        '[NII]122':2459.370214752,\
                                        '[NII]205':1461.132118324,\
                                        'CO(1-0)':115.2712018,\
                                        'CO(2-1)':230.5380000,\
                                        'CO(3-2)':345.7959899,\
                                        'CO(4-3)':461.0407682,\
                                        'CO(5-4)':576.2679305,\
                                        '[OIV]25':11582.51135871917,\
                                        '[NeII]12':23402.81949399302,\
                                        '[NeIII]15':19278.141972490335,\
                                        '[SIII]18':16024.998022215334,\
                                        '[FeII]25':11538.867022566405,\
                                        'H2_S(1)':17603.78496770405,\
                                        'H2_S(2)':24422.612910583943,\
                                        'H2_S(3)':31027.092777274098,\
                                        'H2_S(4)':37363.740805272435,\
                                        'H2_S(5)':43402.57816062832,\
                                        'H2_S(6)':49088.52498207028,\
                                        'H2_S(7)':54409.189540395935}

        self.f_CII                 =   1900.5369                            # frequency in GHz
        self.f_CI1                 =   492.160651                           # frequency in GHz
        self.f_CI2                 =   809.34197                            # frequency in GHz
        self.f_NII_122             =   2459.370214752                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html
        self.f_NII_205             =   1461.132118324                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html
        self.f_OI                  =   4744.774906758                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html
        self.f_OIII       			=   3393.006224818                       # frequency in GHz http://www.ipac.caltech.edu/iso/lws/atomic.html
        els         					=   ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',\
                                            'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn']
        m_elements  =   {}
        for el in els: m_elements[el] = [getattr(per,el).mass*self.au]
        m_elements  =   pd.DataFrame(m_elements)
        # solar abundances from cloudy (n_i/n_H)
        n_elements  =   pd.DataFrame(data=np.array([[1],[1e-1],[2.04e-9],[2.63e-11],\
            [6.17e-10],[2.45e-4],[8.51e-5],[4.90e-4],[3.02e-8],[1.0e-4],[2.14e-6],[3.47e-5],\
            [2.95e-6],[3.47e-5],[3.20e-7],[1.84e-5],[1.91e-7],[2.51e-6],[1.32e-7],[2.29e-6],\
            [1.48e-9],[1.05e-7],[1.00e-8],[4.68e-7],[2.88e-7],[2.82e-5],[8.32e-8],[1.78e-6],\
            [1.62e-8],[3.98e-8]]).transpose(),columns=els) # from cloudy
        # solar mass fractions used in simulations
        self.mf_solar    =   {'tot':0.0134, 'He':0.2485, 'C':2.38e-3,'N': 0.70e-3,'O': 5.79e-3,'Ne': 1.26e-3,
                            'Mg':7.14e-4,'Si': 6.17e-4, 'S':3.12e-4, 'Ca':0.65e-4,'Fe': 1.31e-3}
        elements    =   m_elements.append(n_elements,sort=True)
        elements.index  =   ['mass','[n_i/n_H]']
        self.elements 				=	elements
        self.a_C         			=   elements.loc['[n_i/n_H]','C'] # solar abundance of carbon
        self.mf_Z1                  =   0.0134                      # Asplund+09
        self.mH          			=   elements.loc['mass','H']    # Hydrogen atomic mass [kg]
        self.mC          			=   elements.loc['mass','C']    # Carbon atomic mass [kg]
        self.me          			=   9.10938215e-31              # electron mass [kg]
        self.m_p 					=	1.6726e-24
        self.mCII        			=   self.mC-self.me
        self.mCIII       			=   self.mC-2.*self.me
        self.mCO                    =   elements.loc['mass','C']+elements.loc['mass','O']
        self.mH2         			=   2.*elements.loc['mass','H'] # Molecular Hydrogen [kg]
        self.pos         			=   ['x','y','z']               # set of coordinates (will use often)
        self.posxy	    			=	['x','y']                   # set of coordinates (will use often)
        self.vpos					=	['vx','vy','vz']            # set of velocities (will use often)
        self.FUV_ISM     			=   0.6*1.6*1e-3                # local FUV field [ergs/s/cm^2]
        self.CR_ISM      			=   3e-17                       # local CR field [s^-1]
        self.SFRsd_MW               =   0.0033                      # [Msun/yr/kpc^2] https://ned.ipac.caltech.edu/level5/March15/Kennicutt/Kennicutt5.html
        self.Herschel_limits        =   dict(CII=0.13e-8)           # W/m^2/sr Croxall+17 KINGFISH

        #===========================================================================
        """ For plotting """
        #---------------------------------------------------------------------------

        self.this_work              =   'Model galaxies at z ~ '+self.z1.replace('z','')+' (this work)'
        self.sigame_label           =   r'S$\mathrm{\'I}$GAME at z$\sim$'+self.z1.replace('z','')+' (this work)'
        # datacubes -> ISM
        self.ISM_dc_phases          =   ['GMC','DNG','DIG']
        self.ISM_dc_labels          =   dict( DIG='Diffuse Ionized Gas', DNG='Diffuse Neutral Gas', GMC='Giant Molecular Clouds' )
        self.ISM_dc_colors          =   dict( DIG='b', DNG='orange', GMC='r' )
        #---------------------------------------------------------------------------
        # particle_data -> sim
        self.sim_types              =   ['gas', 'dm', 'star']
        self.sim_labels             =   dict( gas='gas', dm='dark matter', star='star', GMC='Giant Molecular Clouds')
        self.sim_colors             =   dict( gas='blue', dm='grey', star='orange', GMC='r')
        #---------------------------------------------------------------------------
        # particle_data -> ISM
        self.ISM_phases             =   ['GMC', 'dif']
        self.ISM_labels             =   dict(GMC='Giant Molecular Cloud', diff='Diffuse Gas')
        self.ISM_colors             =   dict(GMC='r', diff='b')
        #---------------------------------------------------------------------------
        self.redshift_colors        =   {0:'blue',2:'purple',6:'red'}
        # I would take a look at plot.set_mpl_params() for these
        self.galaxy_marker          =   'o'
        self.galaxy_ms              =   6
        self.galaxy_alpha           =   0.7
        self.galaxy_lw              =   2
        self.galaxy_mew             =   0
        #---------------------------------------------------------------------------
        self.color_names            =   colors.cnames
        col                         =   ['']*len(self.color_names)
        i                           =   0
        for key,value in self.color_names.items():
            col[i]         =    key
            i              +=   1
        self.col                    =   col
        self.colsel      			=   [u'fuchsia',u'darkcyan',u'indigo',u'hotpink',u'blueviolet',u'tomato',u'seagreen',\
                    					u'magenta',u'cyan',u'darkred',u'purple',u'lightgrey',\
                    					u'brown',u'orange',u'darkgreen',u'black',u'yellow',\
                    					u'darkmagenta',u'olive',u'lightsalmon',u'darkblue',\
                    					u'navajowhite',u'sage']

        self.add_default_args()

        #===========================================================================
        """ Save parameters """
        #---------------------------------------------------------------------------

        with open('temp_params.npy', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def add_default_args(self):

        default_args        =   dict(\

                                add=False,\
                                add_obs=True,\
                                angle=-120,\
                                ax=False,\

                                bins=100,\
                                box_size=100,\

                                cb=False,\
                                cell_type='',\
                                classification='spherical',\
                                cloudy_param={'ISM':2},\
                                color='k',\
                                colorbar=True,\
                                convolve=False,\
                                cmap='viridis',\

                                data='sim',\
                                dc_name='data',\
                                debug=False,\

                                ext = '',\
                                
                                flabel='Jy',\
                                figsize=(10,8),\
                                fs_labels=15,\
                                FWHM=None,\
                                format='png',\

                                galname='',\
                                gal_index=0,\
                                gal_indices=False,\
                                gal_ob_present=True,\
                                gal_ob={},\
                                grid_Z = np.arange(-2,0.51,0.5),\
                                grid_nH = np.arange(-4,7.1,1),\
                                grid_FUV = np.arange(-7,4.1,2),\
                                grid_DTM = np.arange(-2,-0.19,0.5),\
                        
                                interp_params=['lognH','lognSFR','logZ','logFUV','logNH','logDTM'],\
                                ISM_phase='',\
                                ISM_dc_phase='tot',\
                                Iunits='Jykms',\

                                keep_const={'logZ':0},\

                                legend=False,\
                                line='[CII]158',\
                                label=False,\
                                log=True,\

                                map_type='',\
                                method='caesar',\
                                min_fraction=1./1e6,\

                                N_radial_bins=30,\
                                nGals=[246,400],\

                                one_color=True,\
                                orientation='face-on',\
                                overwrite=False,\

                                pix_size_kpc=0.1,\
                                plot_gas=False,\
                                plot_stars=False,\
                                plot=True,\
                                prop = 'm',\

                                rotate=False,\
                                R_max=False,\

                                savefig=False,\
                                scale=1.0,\
                                select='',\
                                # sim_run='_100Mpc',\
                                sim_runs=['_25Mpc','_100Mpc'],\

                                target='L_CII',\
                                text=False,\

                                verbose=False,\
                                vlabel='km/s',\
                                vmax=False,\
                                vmin=False,\

                                weight='m',\

                                xlabel='x [kpc]',\
                                xlim=False,\
                                xyz_units='kpc',\

                                ylabel='y [kpc]',\
                                ylim=False,\

                                zlim=False,\
                                zred=0

                                )

        for key,val in default_args.items():
            setattr(self,key,val)

def update_params_file(new_params,verbose=False):

    import aux as aux
    params                      =   aux.load_parameters()

    for key in new_params:
        params[key]               =   new_params[key]

    np.save('SIGAME_v3/temp_params',params)

    if verbose:
        print('Updated params.npy with:')
        for key in new_params:
            print('- '+key)

    return(params)

# coding=utf-8
"""
Module: plot
"""

# Import other SIGAME modules
import sigame.global_results as glo
import sigame.auxil as aux
import sigame.galaxy as gal
import sigame.plot as plot
import sigame.Cloudy_modeling as clo

# Import other modules
# from __future__ import division
import pandas as pd
import numpy as np
import pdb as pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
# from sphviewer.tools import QuickView
from matplotlib.colors import LogNorm
import copy
import os as os
import scipy as scipy
from scipy import ndimage, misc
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
#import moviepy.editor as mpy
import astropy.constants as c
import astropy.units as u
from scipy import integrate
from astropy.cosmology import FlatLambdaCDM
import sklearn as sklearn
from sklearn.linear_model import LinearRegression
from scipy.stats import kde


#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

global params
params                      =   aux.load_parameters()


#===============================================================================
"""  Basic plotting """
#-------------------------------------------------------------------------------

def histos(**kwargs):
    '''Makes histograms of all (particles in all) galaxies in the sample on the same plot.

    Parameters
    ----------

    gal_indices : list
        List of the galaxies to be included, default: False (all galaxies)

    bins : int/float
        Number of bins, default: 100

    add : bool
        If True, add to an existing plot, default: False

    one_color : bool
        If True, use only one color for all lines, default: True

    fs_labels : int/float
        Fontsize, default: 15

    '''

    GR                      =   glo.global_results()

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # By default, do all galaxies
    gal_indices = range(len(GR.galnames))

    if p.gal_indices:
        gal_indices = p.gal_indices

    print('Check gal indices: ',gal_indices)

    # set label sizes
    mpl.rcParams['xtick.labelsize'] = p.fs_labels
    mpl.rcParams['ytick.labelsize'] = p.fs_labels

    # Ask a lot of questions!!
    data_type       =   input('For which data type? [default: cell] '+\
                        '\n sim for raw simulation data (gas/stars/dark matter)'+\
                        '\n cell for cell data (built from SKIRT output)'+\
                        '\n gmc for Giant Molecular Clouds'+\
                        '\n dng for Diffuse Neutral Gas'+\
                        '\n dig for Diffuse Ionized Gas'+\
                        '...? ')
    if data_type == '': data_type =   'cell'
    # data_type = data_type.upper()

    if data_type == 'sim':
        sim_type        =   input('\nGas or star or dark matter (dm)? [default: gas] ... ')
        if sim_type == '': sim_type =   'gas'
        sim_type        =   'sim'+sim_type

    # Start plotting (fignum = 1: first plot)
    if not p.add:
        plt.close('all')
        #plt.ion()
    redo        =   'y'
    fignum      =   1
    counter     =   1
    mindat,maxdat = np.array([]),np.array([])
    while redo == 'y':
        if fignum >1:
            quant        =   input('\nOver what quantity? [default: m]... ')
            if quant == '': quant =   'm'
        histos1     =   np.zeros([len(GR.galnames),p.bins+2])
        histos2     =   np.zeros([len(GR.galnames),p.bins+3])
        igal        =   0
        Ngal        =   0
        indices     =   []
        for gal_index in gal_indices: #TEST
            print(gal_index)
            zred,galname        =   GR.zreds[gal_index],GR.galnames[gal_index]
            gal_ob              =   gal.galaxy(GR=GR, gal_index=gal_index)

            if data_type == 'sim': dat0 = aux.load_temp_file(gal_ob=gal_ob,data_type=sim_type)
            if data_type == 'cell': dat0 = aux.load_temp_file(gal_ob=gal_ob,data_type='cell_data')

            if gal_index == 0:
                print(dat0.keys())

            #print('TEST!!!!')
            #dat0 = dat0.iloc[0:100]
            # Choose what to make histogram over and start figure
            if counter == 1:
                print('\nOver what quantity? Options:')
                keys = ''
                for key in dat0.keys(): keys = keys + key + ', '
                print(keys)

                quant           =   input('[default: m]... ')
                if quant == '': quant =   'm'

                weigh           =   input('\nMass or number-weighted (m vs n)? [default: n] ... ')
                if weigh == '': weigh =   'n'

                logx            =   input('\nLogarithmix x-axis? [default: y] ... ')
                if logx == '': logx =   'y'

                logy            =   input('\nLogarithmix y-axis? [default: y] ... ')
                if logy == '': logy =   'y'

                if p.add:
                    print('\nadding to already existing figure')
                    fig         =   plt.gcf()
                    ax1         =   fig.add_subplot(p.add[0],p.add[1],p.add[2])
                else:
                    print('\ncreating new figure')
                    fig         =   plt.figure(fignum,figsize=p.figsize)
                    ax1         =   fig.add_subplot(1,1,1)

            # Weigh the data (optional) and calculate histogram
            if quant == 'm_mol': dat0['m_mol'] = dat0['f_H2'].values*dat0['m'].values
            dat         =   dat0[quant].values.squeeze()
            print('Total number of data points: %i'  % (len(dat)))
            if weigh == 'm': w           =   dat0['m']
            if weigh == 'n': w           =   1./len(dat0)
            if data_type == 'SIM':
                if quant == 'nH': dat = dat/(mH*1000.)/1e6 # Hydrogen only per cm^3
            print(np.min(dat),np.max(dat))
            mindat = np.append(mindat, np.min(dat))
            maxdat = np.append(maxdat, np.max(dat))
            if logx == 'y':
                dat[dat == 0] = 1e-30 # to avoid crash if metallicity is zero
                dat = np.log10(dat)
                i_nan   =   np.isnan(dat)
                if weigh == 'm':  w       =   w[i_nan == False]
                dat     =   dat[i_nan == False]
                if p.xlim:  
                    w = w[(dat >= p.xlim[0]) & (dat <= p.xlim[1])]
                    dat = dat[(dat >= p.xlim[0]) & (dat <= p.xlim[1])]
            # print('min and max: %s and %s ' % (np.min(dat[dat > -100]),dat.max()))
            #if logy == 'n':
            #    if weigh == 'm':  w       =   w[dat > -10.**(20)]
            #    dat     =   dat[dat > -10.**(20)]
            #if logy == 'y':
            #    if weigh == 'm':  w       =   w[dat > -20]
            #    dat     =   dat[dat > -20]
            # force bin edges?
            if (type(p.xlim) != bool) & (gal_index == 0): p.bins = np.linspace(p.xlim[0],p.xlim[1],p.bins+1)
            if weigh == 'n':    hist        =   np.histogram(dat,bins=p.bins)
            if weigh == 'm':    hist        =   np.histogram(dat,bins=p.bins,weights=w)
            if 'f_HI' in quant:
                print('Particles are above 0.9: %s %%' % (1.*len(dat[dat > 0.9])/len(dat)*100.))
                print('Particles are below 0.1: %s %%' % (1.*len(dat[dat < 0.1])/len(dat)*100.))
            if 'f_H2' in quant:
                print('Particles are above 0.9: %s %%' % (1.*len(dat[dat > 0.9])/len(dat)*100.))
                print('Particles are below 0.1: %s %%' % (1.*len(dat[dat < 0.1])/len(dat)*100.))

            hist1            =  np.asarray(hist[0]) # histogram
            hist2            =  np.asarray(hist[1]) # bin edges
            # save bin edges for next time
            if gal_index == 0:
                p.bins = hist2
            print(hist1)
            hist1            =  hist1*1./sum(hist1)*100.
            wid              =  (hist2[1]-hist2[0])
            # add some zeros to bring histogram down
            hist2            =  np.append([hist2],[hist2.max()+wid])
            hist2            =  np.append([hist2.min()-wid],[hist2])
            hist1            =  np.append([hist1],[0])
            hist1            =  np.append([0],[hist1])
            histos1[igal,:]  =   hist1
            histos2[igal,:]  =   hist2


            if not p.one_color:
                ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='-',drawstyle='steps',color=col[igal],label='G'+str(int(igal+1)))

            igal             +=  1
            Ngal             +=  1
            indices.append(gal_index)

            counter += 1

        histos1             =   histos1[0:Ngal,:]
        histos2             =   histos2[0:Ngal,:]

        if p.one_color:

            # Plot as background the 2 sigma distribution around the mean in each bin
            minhistos1,maxhistos1,meanhistos1,sumhistos1       =   np.zeros(len(histos1[0,:])), np.zeros(len(histos1[0,:])), np.zeros(len(histos1[0,:])), np.zeros(len(histos1[0,:]))
            for i in range(0,len(histos1[0,:])):
                meanhistos1[i]     =   np.mean(histos1[:,i])
                minhistos1[i]      =   meanhistos1[i]-2.*np.std(histos1[:,i])
                maxhistos1[i]      =   meanhistos1[i]+2.*np.std(histos1[:,i])
                if logy == 'y':
                    histo            =   histos1.copy()[:,i]
                    #histo[histo == 0] = np.nan
                    loghisto            =   np.log10(histo)
                    # loghisto[np.isnan(loghisto)] = 0
                    #meanhistos1[i]      =   np.nanmean(loghisto)
                    #meanhistos1[i]      =   np.mean(loghisto[histo != 0])
                    meanhistos1[i]      =   np.median(loghisto[histo != 0])
                    minhistos1[i]       =   meanhistos1[i]-2.*np.std(loghisto[histo != 0])
                    maxhistos1[i]       =   meanhistos1[i]+2.*np.std(loghisto[histo != 0])
                    meanhistos1[i]      =   10.**meanhistos1[i]
                    sumhistos1[i]       =   np.sum(histo)
                    #if hist2[i] > -3: pdb.set_trace()
                    # if hist2[i] > -8: pdb.set_trace()
            if p.method != 'all_cells': ax1.fill_between(hist2[0:len(hist1)]+wid/2, 10.**minhistos1, 10.**maxhistos1, facecolor='lightgreen', alpha=0.5, lw=0)
            # Now plot actual histograms
            for i in range(Ngal):
                # pdb.set_trace()
                hist2           =   histos2[i,:]
                hist1           =   histos1[i,:]
                if p.method != 'all_cells': ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls='-',drawstyle='steps',color='teal',label='G'+str(int(indices[i]+1)),alpha=0.5,lw=1.5)
                # if hist1[2] > 10: pdb.set_trace()

            # Now plot mean of histograms
            if (Ngal > 1) & (p.method != 'all_cells'): ax1.plot(hist2[0:len(hist1)]+wid/2,meanhistos1,ls='-',drawstyle='steps',color='blue',lw=2)
            if p.method == 'all_cells': ax1.plot(hist2[0:len(hist1)]+wid/2,sumhistos1,ls='-',drawstyle='steps',color='blue',lw=2)
        # if logx == 'y':     ax1.set_xscale('log')

        # pdb.set_trace()
        # labels and ranges
        xl          =   getlabel(quant.replace('L_',''))
        if logx    == 'y': xl = 'log '+getlabel(quant.replace('L_',''))
        ax1.set_xlabel(xl,fontsize=p.fs_labels)
        if weigh     == 'n': ax1.set_ylabel('Number fraction [%]',fontsize=p.fs_labels)
        if weigh     == 'm': ax1.set_ylabel('Mass fraction [%]',fontsize=p.fs_labels)
        ax1.set_ylim([max(hist1)/1e4,max(hist1)*10.])

        print('Total range in data values:')
        print(mindat.min(),maxdat.max())
        print('mindat')
        print(mindat)
        print('maxdat')
        print(maxdat)
        if not p.add:
            #fig.canvas.draw()

            # axes ranges
            if p.xlim: ax1.set_xlim(p.xlim)
            if p.ylim:
                if logy == 'y':
                    ax1.set_ylim([10.**p.ylim[0],10.**p.ylim[1]])
                else:
                    ax1.set_ylim(p.ylim)
            #fig.canvas.draw()

            if logy    == 'y': ax1.set_yscale('log')

            savefig         =   input('Save figure? [default: n] ... ')
            if savefig == '': savefig = 'n'
            if savefig == 'y':
                if not os.path.exists(p.d_plot + 'histos/'):
                    os.makedirs(p.d_plot + 'histos/')
                name            =   input('Figure name? plots/histos/... ')
                if name == '':
                    name = galname + '_' + data_type + '_' + quant
                if not os.path.isdir(p.d_plot + 'histos/'): os.mkdir(p.d_plot + 'histos/')
                plt.savefig(p.d_plot + 'histos/'+name+'.png', format='png', dpi=250, facecolor='w')  

            # New figure?
            if p.add:
                redo = 'n'
            else:
                redo        =   input('plot another quantity? [default: n] ... ')
                if redo == '': redo='n'
                if redo == 'n':
                    # restoring defaults
                    mpl.rcParams['xtick.labelsize'] = u'medium'
                    mpl.rcParams['ytick.labelsize'] = u'medium'
                    # break
                fignum      +=1
                changex, changey  =   'n','n'

def map_cell_property(**kwargs):
    """ Map a cell property in 2D
    
    Parameters
    ----------
    gal_index : int/float
        A galaxy index must be passed, default: None

    sim_type : str
        A sim_type must be passed ('simgas', 'simstar'), default: ''

    prop : str
        A property to be mapped can be passed, default: 'm'

    vmin : int/float
        A min value in log typically can be passed, default: 5

    """

    GR                      =   glo.global_results()

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    counter = 0
    fignum = 1
    if p.gal_index == 'all':
        for gal_index in range(GR.N_gal):

            if counter == 0:
                fig, axes = plt.subplots(3, 3, figsize=(20,15))
                axs = [axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1],axes[1,2],axes[2,0],axes[2,1],axes[2,2]]
                counter = 9

            gal_ob                  =   gal.galaxy(GR=GR, gal_index=gal_index)
            print('Now mapping %s' % gal_ob.name)
            isrf_ob = gal.isrf(gal_index)

            # Load SKIRT output
            wavelengths,bin_width   =   aux.read_probe_wavelengths(isrf_ob.name)
            N_start,N_stop          =   aux.FUV_index(wavelengths)
            image_data,units        =   isrf_ob._get_cut_probe(orientation=p.orientation)

            # Plot
            ax1 = axs[9 - counter]
            if p.prop == 'FUV':
                # FUV_xy_image            =   np.array([np.trapz(image_data[N_start:N_stop,:,:],x=wavelengths[N_start:N_stop]) \
                #                             for i in range(len(df))])
                FUV_xy_image            =   image_data[N_start:N_stop,:,:].sum(axis=0) * 4 * np.pi
                FUV_xy_image            =   ndimage.rotate(FUV_xy_image, 0, reshape=True)
                # FUV_xy_image            =   np.fliplr(FUV_xy_image)
                FUV_xy_image[FUV_xy_image <= 0] = np.min(FUV_xy_image[FUV_xy_image > 0])
                im                      =   ax1.imshow(np.log10(FUV_xy_image),\
                    extent=[-isrf_ob.radius,isrf_ob.radius,-isrf_ob.radius,isrf_ob.radius],\
                    vmin=p.vmin,\
                    cmap='twilight')
                lab                     =   'FUV flux [W/m$^2$/micron]'

            # pdb.set_trace()

            ax1.set_xlabel('x [kpc]'); ax1.set_ylabel('y [kpc]')
            # Limit axes limits a bit to avoid area with no particles...
            ax1.set_xlim([-0.8*gal_ob.radius,0.8*gal_ob.radius])
            ax1.set_ylim([-0.8*gal_ob.radius,0.8*gal_ob.radius])
            if p.prop == 'm':
                ax1.text(0.05,0.85,'M$_{gas}$=%.2eM$_{\odot}$' % np.sum(simgas.m),\
                    fontsize=14,transform=ax1.transAxes,color='white')

            counter -= 1


            if counter == 0:
                cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, label=lab)
                # fig.colorbar(im,shrink=0.8,label=lab)

            if counter == 0 or gal_index == GR.N_gal-1:
                figname = p.d_plot + 'cell_data/map_%s_%s_gals_%s_%i.png' % (p.prop,p.z1,p.orientation,fignum)
                print('Saving in ' + figname)
                # plt.tight_layout()
                plt.savefig(figname, format='png', dpi=250, facecolor='w')
                fignum += 1
                pdb.set_trace()
    else:
        fig, ax1 = plt.subplots(figsize=(10,10))
        gal_ob                  =   gal.galaxy(GR=GR, gal_index=p.gal_index)
        simgas                  =   aux.load_temp_file(gal_ob=gal_ob,data_type='cell_data')
        print(simgas.keys())
        map2D,lab,max_scale     =   make_projection_map(simgas,prop=p.prop)

        # Plot
        Rmax = max_scale/2
        if p.log:
            map2D[map2D < 10.**p.vmin] = 10.**p.vmin/2
            map2D = np.log10(map2D)
        if not p.log: map2D[map2D < p.vmin] = p.vmin/2 #np.min(map2D[map2D > 0])
        im = ax1.imshow(map2D,\
            extent=[-Rmax,Rmax,-Rmax,Rmax],vmin=p.vmin,cmap=p.cmap)
        # Limit axes limits a bit to avoid area with no particles...
        ax1.set_xlim([-2/3*gal_ob.radius,2/3*gal_ob.radius])
        ax1.set_ylim([-2/3*gal_ob.radius,2/3*gal_ob.radius])
        fig.colorbar(im,shrink=0.8,ax=ax1,label=lab)
        ax1.set_xlabel('x [kpc]'); ax1.set_ylabel('y [kpc]')

        print('Saving in ' + p.d_plot + 'sim_data/map_%s_G%i.png' % (p.prop,p.gal_index))
        if not os.path.isdir(p.d_plot + 'cell_data/'): os.mkdir(p.d_plot + 'cell_data/')
        plt.savefig(p.d_plot + 'cell_data/map_%s_G%i.png' % (p.prop,p.gal_index), format='png', dpi=250, facecolor='w')

def map_sim_property(**kwargs):
    """ Map a simulation property in 2D
    
    .. note:: Requires swiftsimio installed

    Parameters
    ----------
    gal_index : int/float
        A galaxy index must be passed, default: None

    sim_type : str
        A sim_type must be passed ('simgas', 'simstar'), default: ''

    prop : str
        A property to be mapped can be passed, default: 'm'

    pix_size_kpc : int/float
        Size of each pixel in kpc, default: 0.1

    vmin : int/float
        A min value in log typically can be passed, default: 5


    """

    GR                      =   glo.global_results()

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    counter = 0
    fignum = 1
    if p.gal_index == 'all':

        for gal_index in GR.N_gal - np.arange(GR.N_gal) - 1:

            if counter == 0:
                fig, axes = plt.subplots(3, 3, figsize=(20,15))
                axs = [axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1],axes[1,2],axes[2,0],axes[2,1],axes[2,2]]
                counter = 9

            gal_ob                  =   gal.galaxy(GR=GR, gal_index=gal_index)
            simgas                  =   aux.load_temp_file(gal_ob=gal_ob,data_type='simgas')
            map2D,lab,max_scale     =   make_projection_map(simgas,prop=p.prop)
            if p.prop == 'm': map2D = map2D * simgas.m.sum()/np.sum(map2D) 

            # Plot
            Rmax = max_scale/2
            ax1 = axs[9 - counter]
            if p.log:
                map2D[map2D < 10.**p.vmin] = 10.**p.vmin/2
                map2D[map2D > 10.**p.vmax] = 10.**p.vmax
                map2D = np.log10(map2D)
            if not p.log:
                map2D[map2D < p.vmin] = p.vmin/2
                map2D[map2D > p.vmax] = p.vmax
            im = ax1.imshow(map2D,\
                extent=[-Rmax,Rmax,-Rmax,Rmax],vmin=p.vmin,cmap=p.cmap)
            fig.colorbar(im,shrink=0.8,ax=ax1,label=lab)
            if not p.add: ax1.set_xlabel('x [kpc]'); ax1.set_ylabel('y [kpc]')
            # Limit axes limits a bit to avoid area with no particles...
            ax1.set_xlim([-0.99*Rmax,0.99*Rmax])
            ax1.set_ylim([-0.99*Rmax,0.99*Rmax])
            if (p.prop == 'm') & (p.text == True):
                ax1.text(0.05,0.85,'M$_{gas}$=%.2eM$_{\odot}$' % np.sum(simgas.m),\
                    fontsize=14,transform=ax1.transAxes,color='white')
                ax1.text(0.05,0.75,'SFR=%.2eM$_{\odot}$/yr' % GR.SFR[gal_index],\
                    fontsize=14,transform=ax1.transAxes,color='white')

            counter -= 1

            #if counter == 0:
                # ax1 = plt.subplots(1, 1)
                #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, label=lab)
                # fig.colorbar(im,shrink=0.8,label=lab)

            if counter == 0 or gal_index == GR.N_gal-1:
                print('Saving in ' + p.d_plot + 'sim_data/map_%s_%s_gals_%i.%s' % (p.prop,p.z1,fignum,p.format))
                # plt.tight_layout()
                if not os.path.isdir(p.d_plot + 'sim_data/'): os.mkdir(p.d_plot + 'sim_data/')
                plt.savefig(p.d_plot + 'sim_data/map_%s_%s_gals_%i.%s' % (p.prop,p.z1,fignum,p.format), format=p.format, dpi=250, facecolor='w')
                fignum += 1

    else:
        if p.add:
            fig, ax1 = plt.gcf(), p.ax
        if not p.add:
            fig = plt.figure(figsize=(8,6))
            ax1 = fig.add_axes([0.1, 0.01, 0.8, 0.8]) 
            ax1.axis('equal')

        gal_ob                  =   gal.galaxy(GR=GR, gal_index=p.gal_index)
        simgas                  =   aux.load_temp_file(gal_ob=gal_ob,data_type=p.sim_type)
        if p.R_max:
            # Cut out square
            simgas = simgas[(np.abs(simgas.x) < p.R_max) & (np.abs(simgas.y) < p.R_max)]
            # Add bottom left corner
            extra_row = simgas.iloc[0] # to ensure that map gets the right size
            extra_row['x'],extra_row['y'] = -p.R_max,-p.R_max
            extra_row[p.prop] = 0
            simgas = simgas.append(extra_row).reset_index(drop=True)         
            # Add top right corner
            extra_row = simgas.iloc[0] # to ensure that map gets the right size
            extra_row['x'],extra_row['y'] = p.R_max,p.R_max
            extra_row[p.prop] = 0
            simgas = simgas.append(extra_row).reset_index(drop=True)         
        else:
            pass
        map2D,lab,max_scale     =   make_projection_map(simgas,prop=p.prop)
        if p.prop == 'm': map2D = map2D * simgas.m.sum()/np.sum(map2D) 
        print('Min and max of map: ',map2D.min(),map2D.max())
        #map2D[map2D < 1e4] = 1e6
        # Plot map
        if not p.R_max:
            p.R_max = max_scale/2
        if p.log: 
            if not p.vmax: p.vmax = np.log10(map2D).max()
            if not p.vmin: p.vmin = np.log10(map2D).max() - 4
            map2D[map2D < 10.**p.vmin] = 10.**p.vmin/2
            map2D[map2D > 10.**p.vmax] = 10.**p.vmax
            map2D = np.log10(map2D)
        else:
            if not p.vmax: p.vmax = np.max(map2D)
            if not p.vmin: p.vmin = np.min(map2D) / 1e3
            map2D[map2D < p.vmin] = p.vmin #np.min(map2D[map2D > 0])
        map2D = np.flipud(map2D)

        im = ax1.imshow(map2D,\
            extent=[-max_scale/2,max_scale/2,-max_scale/2,max_scale/2],vmin=p.vmin,vmax=p.vmax,cmap=p.cmap)
        # Limit axes limits a bit to avoid area with no particles...
        zoom = 1#/1.5
        ax1.set_xlim([-1/zoom * p.R_max,1/zoom * p.R_max])
        ax1.set_ylim([-1/zoom * p.R_max,1/zoom * p.R_max])
        if p.colorbar: 
            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im,cax=cax1,label=lab)
        if not p.add: ax1.set_xlabel('x [kpc]'); ax1.set_ylabel('y [kpc]')
        if (p.prop == 'm') & (p.text == True):
            simstar                  =   aux.load_temp_file(gal_ob=gal_ob,data_type='simstar')
            ax1.text(0.05,0.92,'M$_{star}$=%.1e M$_{\odot}$' % np.sum(simstar.m),\
                fontsize=14,transform=ax1.transAxes,color='white')
            ax1.text(0.05,0.86,'M$_{gas}$=%.1e M$_{\odot}$' % np.sum(simgas.m),\
                fontsize=14,transform=ax1.transAxes,color='white')
            ax1.text(0.05,0.80,'SFR=%.2f M$_{\odot}$/yr' % GR.SFR[p.gal_index],\
                fontsize=14,transform=ax1.transAxes,color='white')
        if p.savefig:
            if not os.path.isdir(p.d_plot + 'sim_data/'): os.mkdir(p.d_plot + 'sim_data/')    
            plt.savefig(p.d_plot + 'sim_data/map_%s_G%i.png' % (p.prop,p.gal_index), format=p.format, dpi=250, facecolor='w')

    if not p.colorbar: return(im)

def map_sim_positions(**kwargs):
    """ Simple function to map sim particle positions in 2D
    
    Parameters
    ----------
    gal_index : int/float
        A galaxy index must be passed, default: None

    sim_type : str
        A sim_type must be passed ('simgas', 'simstar'), default: ''

    prop : str
        A property to be mapped can be passed, default: 'm'

    pix_size_kpc : int/float
        Size of each pixel in kpc, default: 0.1

    vmin : int/float
        A min value in log typically can be passed, default: 5


    """

    GR                      =   glo.global_results()

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    fig, ax1 = plt.subplots(figsize=(10,10))
    # p.gal_index = np.where(GR.file_name == 'z0.00_G7169_cG29270')[0][0]
    gal_ob                  =   gal.galaxy(GR=GR, gal_index=p.gal_index)
    # print('TEST!',gal_ob.file_name,p.gal_index)
    simdata                 =   aux.load_temp_file(gal_ob=gal_ob,data_type=p.sim_type)

    # Plot
    print(simdata.head())
    ax1.plot(simdata.x,simdata.y,'o',ms=2,mew=2)

    print(gal_ob.radius)
    # Limit axes limits a bit to avoid area with no particles...
    # ax1.set_xlim([-2/3*gal_ob.radius,2/3*gal_ob.radius])make_projec
    # ax1.set_ylim([-2/3*gal_ob.radius,2/3*gal_ob.radius])
    ax1.set_xlabel('x [kpc]'); ax1.set_ylabel('y [kpc]')

def make_projection_map(simgas,**kwargs):
    """ Make projection map with swiftsimio: https://github.com/SWIFTSIM/swiftsimio

    Parameters
    ----------
    simgas : pandas dataframe
        Simulation particle data for one galaxy must be passed

    prop : str
        A property to be mapped can be passed, default: 'm'

    pix_size_kpc : int/float
        Size of each pixel in kpc, default: 0.1

    vmin : int/float
        A min value in log typically can be passed, default: 5

    Returns
    -------
    map2D : 2D numpy array of values on regular grid
    lab : colorbar label
    max_scale : size of image [kpc]

    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    from swiftsimio.visualisation import projection

    kernel_gamma = 1.936492 # For Wendland-C2 cubic kernel used in swiftsimio/visualisation/slice.py


    xg,yg,zg = simgas.y.values, simgas.x.values, simgas.z.values
    # vxg,vyg,vzg = simgas.vx.values, simgas.vy.values, simgas.vz.values
    hg,mg = simgas.h.values/kernel_gamma,simgas.m.values
    # Zg,dmg,f_H2g,SFRg = simgas.Z.values,simgas.m_dust.values,simgas.f_H2.values,simgas.SFR.values

    # Find a scaling that can bring all coordinates to [0:1]
    x = xg -1.*np.min(xg); y = yg -1.*np.min(yg); z = zg -1.*np.min(zg)
    max_scale = np.max([np.max(np.abs(x)),np.max(np.abs(y))])*p.scale
    print('Scaling gas positions by: %.2f' % max_scale)
    # Resulting number of pixels
    Npix = int(np.ceil(max_scale/p.pix_size_kpc))
    # print('Corresponds to %i pixels' % Npix)
    x,y,z,h = (xg-np.mean(xg))/max_scale+0.5,(yg-np.mean(yg))/max_scale+0.5,(zg-np.mean(zg))/max_scale+0.5,hg/max_scale

    pix_size = max_scale/Npix

    # Render 2D maps
    map2D_m = projection.scatter(x, y, mg, h, Npix)
    map2D_m[map2D_m == 0] = np.min(map2D_m[map2D_m > 0])
    if p.prop == 'm':
        map2D = map2D_m
        lab = 'log($\Sigma_{gas}$) [M$_{\odot}$/kpc$^2$]'
    if p.prop =='Z':
        map2D = projection.scatter(x, y, mg*Zg, h, Npix) / map2D_m
        lab = 'log(Z) [Z$_{\odot}$]'
    if p.prop =='f_H2':
        map2D = projection.scatter(x, y, mg*f_H2g, h, Npix) / map2D_m
        lab = 'log(f$_{H2}$)'
    if p.prop =='SFR':
        map2D = projection.scatter(x, y, mg*SFRg, h, Npix) / map2D_m
        lab = 'log(M$_{\odot}$/yr/kpc$^2$)'
    if p.prop == 'vy_H2':
        map2D_mH2 = projection.scatter(x, y, mg*f_H2g, h, Npix)
        map2D = projection.scatter(x, y, vyg*mg*f_H2g, h, Npix) / map2D_mH2
        lab = r'$\langle$ v$_{y}\rangle_{mw,H2}$ [km/s])'
    if p.prop == 'vz_H2':
        map2D_mH2 = projection.scatter(x, y, mg*f_H2g, h, Npix)
        map2D = projection.scatter(x, y, vzg*mg*f_H2g, h, Npix) / map2D_mH2
        lab = r'$\langle$ v$_{z}\rangle_{mw,H2}$ [km/s])'

    return(map2D,lab,max_scale)

def stamps(d_data='',**kwargs):
    """ Map a simulation property in 2D for each galaxy and save in separate figures.
    """
    GR                      =   glo.global_results()
    if p.gal_index == 'all':
        gal_indices             =   np.arange(GR.N_gal)
    else:
        gal_indices             =   p.gal_index

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    for gal_index in gal_indices:

        fig, ax = plt.subplots(figsize=(8,8))

        gal_ob                  =   gal.galaxy(GR=GR, gal_index=gal_index)
        simgas                  =   gal_ob.particle_data.get_dataframe('simgas',d_data=d_data)
        map2D,lab,max_scale     =   make_projection_map(simgas,prop=p.prop,pix_size_kpc=p.pix_size_kpc,scale=1.5)

        # Plot
        ax.set_facecolor("black")
        Rmax = max_scale/2
        if p.log:
            map2D[map2D < 10.**p.vmin] = 10.**p.vmin/2
            map2D[map2D > 10.**p.vmax] = 10.**p.vmax
            map2D = np.log10(map2D)
        if not p.log:
            map2D[map2D < p.vmin] = p.vmin/2
            map2D[map2D > p.vmax] = p.vmax
        im = ax.imshow(map2D,\
            extent=[-Rmax,Rmax,-Rmax,Rmax],vmin=p.vmin,vmax=p.vmax,cmap=p.cmap)
        Rmax = p.R_max
        ax.set_xlim([-Rmax,Rmax])
        ax.set_ylim([-Rmax,Rmax])
        ax.text(0.05,0.05,'G%i' % gal_index,\
                fontsize=55,transform=ax.transAxes,color='white')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        #plt.gca().set_axis_off()
        #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        #    hspace = 0, wspace = 0)
        #plt.margins(0,0)
        #plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #plt.gca().yaxis.set_major_locator(plt.NullLocator())
        if not os.path.isdir(p.d_plot + 'sim_data/stamps/'): os.mkdir(p.d_plot + 'sim_data/stamps/')    
        plt.savefig(p.d_plot + 'sim_data/stamps/%s%s_G%i.png' % (p.sim_name,p.sim_run,gal_index),\
                 bbox_inches = 'tight', pad_inches = 0)

def stamp_collection(d_data='',**kwargs):
    """ Map a simulation property in 2D for each galaxy and save in combined figures.
    
    .. note:: Requires swiftsimio installed

    Parameters
    ----------
    gal_index : int/float
        A galaxy index must be passed, default: None

    sim_type : str
        A sim_type must be passed ('simgas', 'simstar'), default: ''

    prop : str
        A property to be mapped can be passed, default: 'm'

    pix_size_kpc : int/float
        Size of each pixel in kpc, default: 0.1

    vmin : int/float
        A min value in log typically can be passed, default: 5


    """

    GR                      =   glo.global_results()

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # Because some of the 25Mpc galaxies are HUGE
    if p.gal_index == 'all':
        gal_indices             =   np.arange(GR.N_gal)
        gal_indices             =   gal_indices[GR.R_max < 200.]
        print(len(gal_indices))
    else: 
        gal_indices             =   p.gal_index


    print('TEST!!')
    gal_indices = [91,93,124,117,121,130,135,136,139,143,146,147,152,154,164,166,167,168,171,173,174,175,186,189,192,203,211,213,214,226,222,223,226,228,233,236]

    N_stamps_1 = 8
    N_stamps_2 = 6

    #zoom = 1.5

    counter = N_stamps_1 * N_stamps_2
    fignum = 0
    plotnum = 0

    for gal_index in gal_indices:

        if counter == N_stamps_1 * N_stamps_2:
            print('Creating new figure')
            fig, axes = plt.subplots(figsize=(20,20))
            # fig,(axs,cax) = plt.subplots(ncols=2,figsize = (20,30),\
                  # gridspec_kw={"width_ratios":[1, 0.05]})
            gs1 = mpl.gridspec.GridSpec(N_stamps_1, N_stamps_2,left=0.05,top=0.95,bottom=0.05,right=0.82)

        gal_ob                  =   gal.galaxy(GR=GR, gal_index=gal_index)
        #simgas                  =   aux.load_temp_file(gal_ob=gal_ob,data_type=p.sim_type)
        simgas                  =   gal_ob.particle_data.get_dataframe('simgas',d_data=d_data)
        map2D,lab,max_scale     =   make_projection_map(simgas,prop=p.prop,pix_size_kpc=p.pix_size_kpc,scale=1.5)

        # Plot
        ax1 = plt.subplot(gs1[N_stamps_1*N_stamps_2 - counter])
        ax1.set_facecolor("black")
        Rmax = max_scale/2
        # ax1 = axs[5*8 - counter]
        if p.log:
            map2D[map2D < 10.**p.vmin] = 10.**p.vmin/2
            map2D[map2D > 10.**p.vmax] = 10.**p.vmax
            map2D = np.log10(map2D)
        if not p.log:
            map2D[map2D < p.vmin] = p.vmin/2
            map2D[map2D > p.vmax] = p.vmax
        im = ax1.imshow(map2D,\
            extent=[-Rmax,Rmax,-Rmax,Rmax],vmin=p.vmin,vmax=p.vmax,cmap=p.cmap)
        Rmax = p.R_max
        ax1.set_xlim([-Rmax,Rmax])
        ax1.set_ylim([-Rmax,Rmax])
        ax1.text(0.05,0.05,'G%i' % gal_index,\
                fontsize=14,transform=ax1.transAxes,color='white')
        if p.prop == 'm':
            ax1.text(0.05,0.85,'M$_{gas}$=%.2eM$_{\odot}$' % np.sum(simgas.m),\
                fontsize=14,transform=ax1.transAxes,color='white')
            ax1.text(0.05,0.75,'SFR=%.2eM$_{\odot}$/yr' % GR.SFR[gal_index],\
                fontsize=14,transform=ax1.transAxes,color='white')
            ax1.text(0.05,0.65,'# gas particles: %i' % (len(simgas)),\
                fontsize=14,transform=ax1.transAxes,color='white')

        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

        counter -= 1
        plotnum += 1

        print(gal_index, counter)
        if counter == 0 or gal_index == gal_indices[-1]:
            gs1.update(wspace=0.0, hspace=0.0)
            axes.set_xlabel('x [kpc]'); axes.set_ylabel('y [kpc]')
            cbar_ax = fig.add_axes([0.85, 0.06, 0.02, 0.85])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(label=lab,size=20)
            cbar.ax.tick_params(labelsize=14)
            print('Saving in ' + p.d_plot + 'sim_data/%s%s_map_%s_%s_gals_%i.png' % (p.sim_name,p.sim_run,p.prop,p.z1,fignum))
            # plt.tight_layout()
            if not os.path.isdir(p.d_plot + 'sim_data/'): os.mkdir(p.d_plot + 'sim_data/')    
            plt.savefig(p.d_plot + 'sim_data/%s%s_map_%s_%s_gals_%i.png' % (p.sim_name,p.sim_run,p.prop,p.z1,fignum), format='png', dpi=250, facecolor='w')
            counter = N_stamps_1 * N_stamps_2
            fignum += 1
            plt.close('all')

#---------------------------------------------------------------------------
### SIM CHEKS ###
#---------------------------------------------------------------------------

def Main_Sequence(**kwargs):
    """ Plots main sequence of galaxy selection, comparing with full simulation volume and observations.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    method              =   p.method
    fig,ax              =   plt.subplots(figsize = (8,6))
 
    # Plot all galaxies in simulation volume
    try:
        df_all              =   pd.read_pickle(p.d_data + 'galaxy_selection/z0_all_galaxies%s' % p.sim_runs[0])
        print('%i galaxies in Simba-%s' % (len(df_all),p.sim_runs[0]))
        df_all1 = df_all[(df_all['SFR_'+method] > 0) & (df_all['SFR_'+method] != 1)]
        hb = ax.hexbin(df_all1['M_star_'+method],df_all1['SFR_'+method],bins='log',xscale='log',yscale='log',\
                             cmap='binary',lw=0,gridsize=70)
        df_all              =   pd.read_pickle(p.d_data + 'galaxy_selection/z0_all_galaxies%s' % p.sim_runs[1])
        print('%i galaxies in Simba-%s' % (len(df_all),p.sim_runs[1]))
        df_all2 = df_all[df_all['SFR_'+method] > 0]
        df_all = df_all1.append(df_all2, ignore_index=True)
        hb = ax.hexbin(df_all['M_star_'+method],df_all['SFR_'+method],bins='log',xscale='log',yscale='log',\
                                cmap='binary',lw=0,gridsize=(50,70))
    except:
        print('Missing file to plot all galaxies in Simba-%s' % (p.sim_runs[0]))

    # Plot 25 Mpc box? 
    if p.select == '_25Mpc':
        GR                  =   glo.global_results(sim_run='_25Mpc',nGal=240,grid_ext='_ext_ism_BPASS')
        M_star,SFR,Zsfr = getattr(GR,'M_star'),getattr(GR,'SFR'),getattr(GR,'Zsfr')
        ax.plot(1,1,'^',color='forestgreen',label='Simba-25 galaxy sample',ms=10)
        sc = ax.scatter(M_star,SFR,\
            marker='^',s=50,alpha=0.8,c=np.log10(Zsfr),vmin=np.log10(0.01),vmax=np.log10(2),cmap='summer',zorder=10)

    # Plot current sample
    GR                  =   glo.global_results()
    M_star,SFR,Zsfr = getattr(GR,'M_star'),getattr(GR,'SFR'),getattr(GR,'Zsfr')
    if p.select == '_MS':
        indices = aux.select_salim18(GR.M_star,GR.SFR)
        M_star = M_star[indices]
        SFR = SFR[indices]
        Zsfr = Zsfr[indices]
        print('With MS selection criteria: only %i galaxies' % (len(M_star)))
    ax.plot(1,1,'o',color='forestgreen',label='Simba-100 galaxy sample',ms=10)
    print(len(M_star))
    print(np.max(Zsfr))
    sc = ax.scatter(M_star,SFR,\
            marker='o',s=20,alpha=0.8,c=np.log10(Zsfr),vmin=np.log10(0.01),vmax=np.log10(2),cmap='summer',zorder=10)

    # Plot observations
    if p.zred == 0:
        MS_salim = pd.read_csv('data/observations/MS/salim2018_ms_v1.dat',\
                names=['logMstar','logsSFR','logsSFR_1','logsSFR_2'],sep='   ')
        ax.fill_between(10.**MS_salim.logMstar,10.**MS_salim.logMstar*10.**MS_salim.logsSFR_1,\
                 10.**MS_salim.logMstar*10.**MS_salim.logsSFR_2,color='royalblue',alpha=0.3)
        ax.plot(10.**MS_salim.logMstar,10.**MS_salim.logMstar*10.**MS_salim.logsSFR,\
                '--',color='mediumblue',label='[Salim+18] SF MS')
        # MS_salim = pd.read_csv('data/observations/MS/salim2018_ms_v2.dat',names=['logMstar','logsSFR'],sep='   ')
        # ax.plot(10.**MS_salim.logMstar,10.**MS_salim.logMstar*10.**MS_salim.logsSFR,'--',label='[Salim+18] SF MS')
        cosmo = FlatLambdaCDM(H0=0.68*100 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
        t = cosmo.age(0).value
        fit_speagle = 10.**((0.84-0.026*t)*np.log10(ax.get_xlim())-(6.51-0.11*t))
        #ax.fill_between(ax.get_xlim(),10.**(np.log10(fit_speagle)-0.3),\
        #    10.**(np.log10(fit_speagle)+0.3),alpha=0.2,color='grey')
        fit_speagle = 10.**((0.84-0.026*t)*np.log10(ax.get_xlim())-(6.51-0.11*t))
        # Convert from Kroupa to Chabrier:  https://ned.ipac.caltech.edu/level5/March14/Madau/Madau3.html
        #ax.plot(ax.get_xlim(),fit_speagle*0.63/0.67,':',color='grey',label='[Speagle+14] "mixed" fit')
  
    ax.set_ylabel('SFR [M$_{\odot}$/yr]')
    ax.set_xlabel('M$_*$ [M$_{\odot}$]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e7,1e12])
    ax.set_ylim([10**(-2),1e2])
 
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r'log $\langle$Z$\rangle_{\rm{SFR}}$ [Z$_{\odot}$]')
    handles,labels = ax.get_legend_handles_labels()
    try:
        handles = [handles[_] for _ in [1,0,2]]#np.flip(handles)
        labels = [labels[_] for _ in [1,0,2]]#np.flip(labels)
    except:
        handles = [handles[_] for _ in [1,0]]#np.flip(handles)
        labels = [labels[_] for _ in [1,0]]#np.flip(labels)
    ax.legend(handles,labels,fontsize=12)
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'sim_data/'): os.mkdir(p.d_plot + 'sim_data/')    
        plt.savefig('plots/sim_data/SFR_Mstar_%s_%s%s' % (method,p.sim_name,p.sim_run),dpi=250,facecolor='w')


def sSFR_hist(**kwargs):
    """ Compare sSFR of full simulation volume and selection.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    method              =   p.method
    fig,ax              =   plt.subplots(figsize = (8,4))
    xlim                =   [-4,0.5] 

    # Plot 25 Mpc box? 
    if p.select == '_25Mpc':
        GR                  =   glo.global_results(sim_run='_25Mpc',nGal=240,grid_ext='_ext_ism_BPASS')
        M_star,SFR,Zsfr = getattr(GR,'M_star'),getattr(GR,'SFR'),getattr(GR,'Zsfr')
        sSFR = SFR/M_star
        ax.hist(np.log10(1e9*sSFR),bins=50,color='deepskyblue',alpha=1,label='Simba-25 galaxy sample',zorder=10)

    # Plot current sample
    GR                  =   glo.global_results()
    M_star,SFR,Zsfr = getattr(GR,'M_star'),getattr(GR,'SFR'),getattr(GR,'Zsfr')
    sSFR = SFR/M_star
    ax.hist(np.log10(1e9*sSFR),bins=50,color='green',alpha=0.5,label='Simba-100 galaxy sample',zorder=10)

    # Plot all galaxies in simulation volume
    df_all              =   pd.read_pickle(p.d_data + 'galaxy_selection/z0_all_galaxies%s' % p.sim_runs[0])
    print('%i galaxies in Simba-%s' % (len(df_all),p.sim_runs[0]))
    df_all1 = df_all[(df_all['SFR_'+method] > 0) & (df_all['SFR_'+method] != 1)]
    df_all              =   pd.read_pickle(p.d_data + 'galaxy_selection/z0_all_galaxies%s' % p.sim_runs[1])
    print('%i galaxies in Simba-%s' % (len(df_all),p.sim_runs[1]))
    df_all2 = df_all[df_all['SFR_'+method] > 0]
    df_all = df_all1.append(df_all2, ignore_index=True)

    sSFR = df_all['SFR_'+method].values/df_all['M_star_'+method].values
    ax2 = ax.twinx()
    sSFR = sSFR[(df_all['SFR_'+method] > 0) & (df_all['M_star_'+method] > 0)].astype('float')
    #ax2.hist(np.log10(1e9*sSFR[sSFR > 10**xlim[0]]),bins=100,color='grey',alpha=0.5,label='All SF galaxies in Simba-25 and Simba-100',zorder=10)
    ax2.hist(np.log10(1e9*sSFR[sSFR > 10.**xlim[0]/1e9]),fill=False,bins=100,histtype='stepfilled',fc=None,ec='k',alpha=0.8,label='All SF galaxies in Simba-25 and Simba-100',zorder=10)

    ax.set_xlabel(r'log sSFR [Gyr$^{-1}$]')
    ax.set_ylabel('Number of selected galaxies')
    ax2.set_ylabel('Number of galaxies in Simba')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim([0.8,5e1])
 
    handles,labels = ax.get_legend_handles_labels()
    handles2,labels2 = ax2.get_legend_handles_labels()
    handles = np.append(handles,handles2)
    labels = np.append(labels,labels2)
    #ax.legend(fontsize=12)
    ax.legend(handles,labels,fontsize=11)
    plt.tight_layout()
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'sim_data/'): os.mkdir(p.d_plot + 'sim_data/')    
        plt.savefig('plots/sim_data/sSFR_%s_%s%s' % (method,p.sim_name,p.sim_run),dpi=250,facecolor='w')



def Mstar_function(**kwargs):
    """ Plots stellar mass function
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if not p.xlim:
        p.xlim          =   np.array([1e10,1e13])

    df_all              =   pd.read_pickle(p.d_data + 'galaxy_selection/all_z0_galaxies')
    Mstar               =   df_all['M_star_caesar'].values

    logM_star           =   np.log10(Mstar)
    dM                  =   0.25
    N_gal               =   len(np.where((Mstar > Mstar.min()) & (Mstar < (Mstar.min() + dM)))[0])
    logM_star_bin       =   np.arange(logM_star.min(), logM_star.max(), dM)
    logM_star_bin_c     =   logM_star_bin[0:-1] + (logM_star_bin[1]-logM_star_bin[0])/2

    N_gal_array         =   np.zeros(len(logM_star_bin)-1)

    # Number of galaxies in each stellar mass bin
    for i in range(len(logM_star_bin)-1):
        N_gal_array[i] = len(np.where((logM_star > logM_star_bin[i]) & (logM_star < (logM_star_bin[i+1])))[0])

    # Corresponding volume density of galaxies
    n_gal_array = N_gal_array / (p.box_size)**3 # number of galaxies per Mpc^3

    fig, ax = plt.subplots()
    hb = ax.plot(logM_star_bin_c, np.log10(n_gal_array))
    ax.set_ylabel('$\log\Phi$ [Mpc$^{-3}$]')
    ax.set_xlabel('log Stellar Mass [M$_{\odot}$]')
    ax.set_ylim([-7,0.2])
    plt.tight_layout()
    plt.show()

def gas_dust_ratio(**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)


    GR                  =   glo.global_results()

    GD_ratio            =   getattr(GR,'M_gas')/getattr(GR,'M_dust')
    DG_ratio            =   1./GD_ratio

    print('Min and max G/D ratio: %.2e %.2e' % (GD_ratio.min(),GD_ratio.max()))
    print('Min and max D/G ratio: %.2e %.2e' % (DG_ratio.min(),DG_ratio.max()))
    print('Median G/D ratio: %.2e' % (np.median(GD_ratio)))
    print('Number of galaxies with G/D < 162 for solar ISM abudances [Zubko+2004]: %.2i' % (len(GD_ratio[GD_ratio < 162])))


#---------------------------------------------------------------------------
### FOR ISRF TASK ###
#---------------------------------------------------------------------------

def star_map(**kwargs):
    '''Plots map of stars, indicating their age and mass

    Parameters
    ----------
    gal_index : int
        Galaxy index, default: 0

    '''

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()
    gal_ob              =   gal.galaxy(GR=GR, gal_index=p.gal_index)

    simstar             =   aux.load_temp_file(gal_ob=gal_ob,data_type='simstar')
    # simstar             =   simstar[np.abs(simstar.z) < 2].reset_index(drop=True)
    if p.sim_type == 'sph': simstar.age             =   simstar.age*1e9
    simstar = simstar[simstar.age < 10**p.vmax] # only younger than 1 Gyr
    if not p.vmax: p.vmax = np.max(simstar.age)
    if not p.vmin: p.vmin = np.min(simstar.age)
    simstar             =   simstar.sort_values('age',ascending=False)
    m                   =   np.log10(simstar.m.values)

    if p.sim_type == 'sph': m                   =   (m - m.min())/(m.max() - m.min()) * 300 + 50
    if p.sim_type == 'amr': m                   =   (m - m.min())/(m.max() - m.min()) * 100 + 25

    if p.add:
        ax1                 =   p.ax
    else:
        # fig, ax1            =   plt.subplots(figsize=(7.3, 6  ))
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8]) 

    print('Range in stellar age [Myr]: ',np.min(simstar.age/1e6),np.max(simstar.age/1e6))

    sc = ax1.scatter(simstar.x,simstar.y,s=m,c=np.log10(simstar.age),alpha=0.6,cmap='jet',vmin=p.vmin,vmax=p.vmax)
    print(p.vmin,p.vmax)
    if p.colorbar: plt.colorbar(sc,shrink=0.6,ax=ax1,label='log stellar age [yr]')
    # ax1.axis('equal')
    ax1.set_aspect('equal', 'box')

    if p.R_max:
        ax1.set_xlim([-p.R_max,p.R_max]); ax1.set_ylim([-p.R_max,p.R_max])
    else:
        ax1.set_xlim([-gal_ob.R_max,gal_ob.R_max]); ax1.set_ylim([-gal_ob.R_max,gal_ob.R_max])

    if not p.add: ax1.set_xlabel('x [kpc]'); ax1.set_ylabel('y [kpc]')


    if p.savefig:
        if not os.path.isdir(p.d_plot + 'sim_data/'): os.mkdir(p.d_plot + 'sim_data/')    
        plt.savefig(p.d_plot + 'sim_data/star_map_G%i.png' % (p.gal_index), format=p.format, dpi=250, facecolor='w')

    if not p.colorbar: return(sc)

def FUV_map(**kwargs):
    '''Plots FUV projected map from SKIRT output for selected galaxies

    Parameters
    ----------
    gal_index : int
        Galaxy index, default: None (= all galaxies)
    '''

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()
    
    # Use one or all galaxies
    if p.gal_index == None:
        p.gal_index             =   np.arange(GR.N_gal)
    else:
        p.gal_index             =   [p.gal_index]

    for gal_index in p.gal_index:

        if p.add:
            fig, ax1                =   plt.gcf(), p.ax
        if not p.add:
            fig = plt.figure(figsize=(8,6))
            ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8]) 
            ax1.axis('equal')

        isrf_ob = gal.isrf(gal_index)
        gal_ob          =   gal.galaxy(gal_index)
        simgas = gal_ob.particle_data.get_dataframe('simgas')

        # Load SKIRT output
        image_data,units,wa     =   isrf_ob._get_map_inst(orientation=p.orientation,select=p.select)
        N_start,N_stop          =   aux.FUV_index(wa)
        #FUV_xy_image            =   image_data[N_start:N_stop,:,:].sum(axis=0) # wrong, see email from Peter Camps
        index1                  =   np.arange(image_data.shape[1])
        index2                  =   np.arange(image_data.shape[2])
        index1,index2           =   np.meshgrid(index1,index2)
        F_FUV_xy_image          =   np.zeros([image_data.shape[1],image_data.shape[2]])
        # Integrate to get from W/m$^2$/micron/arcsec$^2$ to W/m$^2$/arcsec$^2$
        for i1,i2 in zip(index1.flatten(),index2.flatten()):
            F_FUV_xy_image[i1,i2]            =   np.trapz(image_data[N_start:N_stop,i1,i2],x=wa[N_start:N_stop])
        # Integrate over sphere and solid angle and convert to solar luminosity
        L_FUV_xy_image = F_FUV_xy_image * 4 * np.pi * 4 * np.pi * (10e6 * c.pc.to('m').value)**2 / p.Lsun 

        # Plot image
        FUV_xy_image            =   np.flipud(F_FUV_xy_image)
        #FUV_xy_image            =   np.rot90(FUV_xy_image)
        FUV_xy_image[FUV_xy_image <= 0] = np.min(FUV_xy_image[FUV_xy_image > 0])
        print('Min max FUV flux:')
        print(FUV_xy_image.max())
        print(FUV_xy_image.min())

        R = np.max(np.abs(np.array([simgas.x,simgas.y,simgas.z])))
        im                      =   ax1.imshow(np.log10(FUV_xy_image),\
            extent=[-R,R,-R,R],vmin=np.max(np.log10(FUV_xy_image))-8,\
            cmap='twilight')
        lab                     =   'FUV flux [W/m$^2$/arcsec$^2$]'

        if p.R_max:
            ax1.set_xlim([-p.R_max,p.R_max])
            ax1.set_ylim([-p.R_max,p.R_max])

        if not p.add: ax1.set_xlabel('x [kpc]'); ax1.set_ylabel('y [kpc]')
        if p.colorbar: fig.colorbar(im,shrink=0.8,ax=ax1,label=lab)

        if p.savefig:
            if not os.path.isdir(p.d_plot + 'cell_data/'): os.mkdir(p.d_plot + 'cell_data/')    
            plt.savefig(p.d_plot + 'cell_data/FUV_map_%s%s_%s.png' % (isrf_ob._get_name(),p.select,p.orientation), format=p.format, dpi=250, facecolor='w')
        
    if not p.colorbar: return(im)

def FUV_fluxes(**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()
    N_star              =   GR.N_star[p.gal_index]

    if p.add:
        fig, ax1                =   plt.gcf(), p.ax
    if not p.add:
        fig                     =   plt.figure(figsize=(8, 6  ))
        ax1 = fig.add_axes([0.2, 0.15, 0.78, 0.8]) 


    colors = ['forestgreen','orchid','orange','cyan']
    for i,select in enumerate(p.select):
        print('Now %s' % select)

        isrf_ob = gal.isrf(p.gal_index)

        wavelengths,bin_width = aux.read_probe_wavelengths(isrf_ob._get_name()+select)
        Nbins               =   len(wavelengths)

        # Read probe intensities in W/m2/micron/sr
        I_W_m2_micron_sr    =   np.array(aux.read_probe_intensities(isrf_ob._get_name()+select,Nbins))

        # Convert intensities to W/m2/micron
        I_W_m2_micron       =  I_W_m2_micron_sr * 4 * np.pi

        # Integrate intensities in FUV
        print('Do integration')
        N_start,N_stop      =   aux.FUV_index(wavelengths)
        F_FUV_W_m2          =   np.array([np.trapz(I_W_m2_micron[i,N_start:N_stop],x=wavelengths[N_start:N_stop]) \
                                        for i in range(len(I_W_m2_micron))])

        # Normalize to G0 energy density (5.29e-14 ergs/cm^3)
        # http://www.ita.uni-heidelberg.de/~rowan/ISM_lectures/galactic-rad-fields.pdf eq. 18
        E_FUV_ergs_cm3      =   F_FUV_W_m2 / p.clight / 1e-7 / 1e6
        G0                  =   E_FUV_ergs_cm3 / 5.29e-14 # ergs/cm^3 from Peter Camps

        # df               =    isrf_ob.cell_data.get_dataframe()
        hist             =   np.histogram(np.log10(G0[G0 > 0]),bins=p.bins)#,weights=df.m.values[G0 > 0])
        hist1            =  np.asarray(hist[0]) # histogram
        hist2            =  np.asarray(hist[1]) # bin edges
        hist1            =  hist1*1./sum(hist1)*100.
        wid              =  (hist2[1]-hist2[0])
        # add some zeros to bring histogram down
        hist2            =  np.append([hist2],[hist2.max()+wid])
        hist2            =  np.append([hist2.min()-wid],[hist2])
        hist1            =  np.append([hist1],[0])
        hist1            =  np.append([0],[hist1])
        labels           =  {'_1e6':'10$^6$ packets: %.1e packets/source' % (1e6/N_star),\
                               '_1e7':'10$^7$ packets: %.1e packets/source' % (1e7/N_star),\
                               '_1e8':'10$^8$ packets: %.1e packets/source' % (1e8/N_star),\
                               '_1e9':'10$^9$ packets: %.1e packets/source' % (1e9/N_star)}
        ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,color=colors[i],drawstyle='steps',label=labels[select])

    ax1.set_yscale("log")
    ax1.set_xlim([-20,5])
    ax1.set_ylim([1e-3,1e3])
    ax1.set_xlabel(getlabel('lG0'))
    # ax1.set_ylabel('Fraction of mass')
    ax1.set_ylabel('Number of cells')
    ax1.legend(loc='upper left')

    if p.savefig:
        plt.tight_layout()
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/FUV_fluxes_comp_hii.png', format=p.format, dpi=250, facecolor='w')
        #plt.savefig(p.d_plot + 'luminosity/FUV_fluxes_comp.png', format=p.format, dpi=250, facecolor='w')

def FUV_lums(**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()
    colors = ['forestgreen','orchid','orange','cyan']

    L_FUVs = np.zeros(len(p.select))
    for i,select in enumerate(p.select):

        isrf_ob = gal.isrf(p.gal_index)
        isrf_ob._set_distance_to_galaxy()

        # Read flux in W/m2/micron
        SED_inst = isrf_ob._read_SED(fluxOutputStyle="Wavelength",select=select)
        F_W_m2_micron = SED_inst.F_W_m2_micron.values
        wavelengths = SED_inst.wavelength.values
        N_start,N_stop = aux.FUV_index(wavelengths)

        # Convert to solar luminosity
        F_FUV_W_m2 = np.trapz(F_W_m2_micron[N_start:N_stop],x=wavelengths[N_start:N_stop])
        L_FUV_W = F_FUV_W_m2*4*np.pi*(isrf_ob.distance*1e6*p.pc2m)**2
        L_FUVs[i] = L_FUV_W/p.Lsun

    if p.add:
        fig, ax1                =   plt.gcf(), p.ax
    if not p.add:
        fig                     =   plt.figure(figsize=(8, 6  ))
        ax1 = fig.add_axes([0.2, 0.15, 0.78, 0.8]) 

    ax1.plot(np.arange(len(p.select))+1,L_FUVs)
    [ax1.plot(_+1,L_FUVs[_],'x',ms=8,mew=3,color=colors[_]) for _ in range(len(L_FUVs))]
    ax1.set_xticks(np.arange(len(p.select))+1)
    ax1.set_xticklabels([_.replace('_','') for _ in p.select])
    ax1.set_ylim(np.min(L_FUVs)/1.1,np.max(L_FUVs)*1.1)
    ax1.set_ylabel(r'L$_{\rm FUV}$ [L$_{\odot}$]')
    ax1.set_xlabel('Number of photon packets')
    ax1.set_title('%.2f%% change in luminosity' % ((np.max(L_FUVs)-np.min(L_FUVs))/L_FUVs[0]*100.))
    ax1.set_yscale("log")

    if p.savefig:
        plt.tight_layout()
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/FUV_lums_comp.png', format=p.format, dpi=250, facecolor='w')
        
def FUV_crosssec(**kwargs):
    '''Plots FUV cross-section from SKIRT output for selected galaxies

    Parameters
    ----------
    gal_index : int
        Galaxy index, default: None (= all galaxies)

    '''

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    if p.gal_index == None:
        # Use all galaxies
        p.gal_index             =   np.arange(GR.N_gal)
    else:
        p.gal_index             =   [p.gal_index]

    for gal_index in p.gal_index:

        isrf_ob = gal.isrf(gal_index)

        # Load SKIRT output
        wavelengths,bin_width   =   aux.read_probe_wavelengths(isrf_ob._get_name())
        N_start,N_stop          =   aux.FUV_index(wavelengths)
        image_data,units        =   isrf_ob._get_cut_probe(orientation=p.orientation)

        # FUV_xy_image            =   image_data[N_start:N_stop,:,:].sum(axis=0) * 4 * np.pi
        FUV_image_W_m2          =   image_data.copy()
        for i in range(N_start,N_stop):
            FUV_image_W_m2[i,:,:]       =   FUV_image_W_m2[i,:,:]*bin_width[i]
        FUV_image_W_m2          =   FUV_image_W_m2[N_start:N_stop,:,:].sum(axis=0) * 4 * np.pi
        FUV_image_G0            =   FUV_image_W_m2 / c.c.value / 1e-7 / 1e6 / 5.29e-14 # like in _add_FUV_flux
        # FUV_image_G0            =   ndimage.rotate(FUV_image_G0, 90, reshape=True)
        # FUV_xy_image            =   np.fliplr(FUV_xy_image)
        FUV_image_G0            =   np.flipud(FUV_image_G0)
        FUV_image_G0[FUV_image_G0 <= 0] = np.min(FUV_image_G0[FUV_image_G0 > 0])
        if p.add:
            fig, ax1                =   plt.gcf(), p.ax
        if not p.add:
            fig, ax1                =   plt.subplots(figsize=(8, 6  ))
        im                      =   ax1.imshow(np.log10(FUV_image_G0),\
            extent=[-isrf_ob.R_max,isrf_ob.R_max,-isrf_ob.R_max,isrf_ob.R_max],\
            vmin=p.vmin,vmax=p.vmax,\
            cmap=p.cmap)

        # Add stars - not sure about orientation...
        if p.plot_stars:
            gal_ob                  =   dict(zred=isrf_ob.zred,galname=isrf_ob.name,gal_index=isrf_ob.gal_index)
            simstar                 =   aux.load_temp_file(gal_ob=gal_ob,data_type='simstar')
            simstar                 =   simstar.copy()[np.abs(simstar['z']) < 1].reset_index(drop=True)

            # Rotate by 90 degrees
            rot90                   =   -1.*np.array([[np.cos(np.pi/2.),-np.sin(np.pi/2.)],[np.sin(np.pi/2.),np.cos(np.pi/2.)]])
            xy                      =   np.dot(rot90,np.array([simstar.x.values,simstar.y.values]))
            ax1.plot(xy[0,:],xy[1,:],'o',ms=3,mew=0,color='orange',alpha=0.7)

        # Add gas
        if p.plot_gas:
            gal_ob                  =   dict(zred=isrf_ob.zred,galname=isrf_ob.name,gal_index=isrf_ob.gal_index)
            simgas                  =   aux.load_temp_file(gal_ob=gal_ob,data_type='simgas')

            # Rotate by 90 degrees
            rot90                   =   -1.*np.array([[np.cos(np.pi/2.),-np.sin(np.pi/2.)],[np.sin(np.pi/2.),np.cos(np.pi/2.)]])
            xy                      =   np.dot(rot90,np.array([simgas.x.values,simgas.y.values]))
            # xy = np.array([simgas.x.values,simgas.y.values])
            ax1.plot(xy[1,:],xy[0,:],'o',ms=4.,color='magenta',alpha=0.7)

        ax1.set_xlabel('x [kpc]'); ax1.set_ylabel('y [kpc]')
        ax1.set_xlim([-isrf_ob.R_max,isrf_ob.R_max]); ax1.set_ylim([-isrf_ob.R_max,isrf_ob.R_max])
        if p.colorbar: fig.colorbar(im,shrink=0.6,ax=ax1,label='log FUV flux [G0]')

        if p.savefig:
            if not os.path.isdir(p.d_plot + 'cell_data/'): os.mkdir(p.d_plot + 'cell_data/')    
            plt.savefig(p.d_plot + 'cell_data/FUV_crosssec_%s.png' % (isrf_ob._get_name()), format=p.format, dpi=250, facecolor='w')
    
    if not p.colorbar: return(im)

def L_TIR_SFR(**kwargs):
    '''Plots L_TIR vs SFR from SKIRT output for all galaxies

    Parameters
    ----------
    gal_index : int
        Galaxy index, default: 0

    '''

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    #data points from SKIRT:
    SFR                 =   GR.SFR
    L_TIR_sun           =   GR.L_TIR_sun
    print(len(L_TIR_sun[L_TIR_sun > 0]))

    #SFR-L_TIR equation given in Kennicutt & Evans 2012 (using Murphy+2011):
    #logSFR = logL_TIR - logC
    logC                =   43.41
    L_TIR_test          =   10.**np.array([42, 48])
    SFR_test            =   0.94*(10.**(np.log10(L_TIR_test) - logC))
    #This L_TIR - SFR relation comes from Murphy+2011 who use a Kroupa IMF, but our MUFASA/SIMBA sims have a Chabrier IMF.
    #So SFR in the formula must be corrected with a factor 0.63/0.67=0.94

    L_TIR_test_sun      =   L_TIR_test/ (p.Lsun * 1e7)

    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15
    fig,ax = plt.subplots()
    ax.loglog(SFR_test, L_TIR_test_sun,label='Kennicutt & Evans 2012')
    ax.scatter(SFR, L_TIR_sun, label='Simba z=0')
    ax.set_ylabel("L$_{\mathrm{TIR}}$ (L$_{\odot}$)",fontsize=15)
    ax.set_xlabel(getlabel('SFR'),fontsize=15)
    l = ax.legend(fontsize=15)

    if p.xlim: ax.set_xlim(p.xlim)
    if p.ylim: ax.set_ylim(p.ylim)

    if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
    plt.savefig(p.d_plot + 'luminosity/L_TIR_SFR.png', format='png', dpi=250, facecolor='w')

def all_skirt_spectra(**kwargs):
    '''Plots all spectra from SKIRT

    Parameters
    ----------
    gal_index : int
        Galaxy index, default: 0

    '''

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()
    gal_ob              =   gal.galaxy(GR=GR, gal_index=p.gal_index)
    # gal_ob.name = gal_ob.name+'_no_Z' # test to set Z=0
    cell_data           =   aux.load_temp_file(gal_ob=gal_ob,data_type='cell_data')

    # Wavelengths and flux
    wa,bin_width        =   aux.read_probe_wavelengths(gal_ob.cell_data._get_name())
    N_wa                =   len(wa)
    print('Number of wavelengths: ',N_wa)

    # Flux per cell
    data                =   np.array(aux.read_probe_intensities(gal_ob.cell_data._get_name(),N_wa))
    print('Number of cells: ',len(data))

    print(cell_data.keys())
    UV_to_FUV           =   cell_data['UV_to_FUV'].values
    R_NIR_FUV           =   cell_data['R_NIR_FUV'].values
    print('min max UV/FUV ratio: ',np.min(UV_to_FUV),np.max(UV_to_FUV))
    l_UV_to_FUV         =   np.log10(UV_to_FUV)
    l_R_NIR_FUV         =   np.log10(R_NIR_FUV)

    # Convert wavelengths to energy
    E                   =   c.h.value*c.c.value/(wa*1e-6)*u.J.to('eV')

    # Normalize all spectra at lowest energy
    data1               =   data/cell_data['F_FUV_W_m2'].values.reshape(len(data),1)

    # Bin in terms of UV-to-FUV ratio
    N_UV                =   10
    UV_to_FUV_bins      =   np.linspace(l_UV_to_FUV.min(),l_UV_to_FUV.max(),N_UV+1)
    NIR_to_FUV_bins     =   np.linspace(l_R_NIR_FUV.min(),l_R_NIR_FUV.max(),N_UV+1)

    binned_spectra      =   np.zeros([N_UV,N_wa])
    for i_bin in range(N_UV):
        #index = np.where((l_UV_to_FUV >= UV_to_FUV_bins[i_bin]) & (l_UV_to_FUV < UV_to_FUV_bins[i_bin+1]))[0]
        index = np.where((l_R_NIR_FUV >= NIR_to_FUV_bins[i_bin]) & (l_R_NIR_FUV < NIR_to_FUV_bins[i_bin+1]))[0]
        print(len(index))
        mean_spectrum   =   np.mean(data1[index,:],axis=0)
        # Convert W/m2/micron to W/m2
        binned_spectra[i_bin,:] = integrate.cumtrapz(mean_spectrum,x=wa, initial=0)

    fig,ax = plt.subplots(figsize=(10,8))
    ax.set_xlabel('E [eV]')
    ax.set_ylabel(r'Intensity in W/m$^2$, normalized to 1 at 0.1 eV')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1e-2,1e3])
    ax.set_ylim([1e-10,1e2])
    cmap = plt.get_cmap('gnuplot2')
    colors = [cmap(i) for i in np.linspace(0, 1, N_UV+1)]
    for i in range(N_UV):#len(data)):
        # Normalize shapes to 1 at 13.6 eV
        norm = binned_spectra[i,np.argmin(np.abs(E-0.1))]
        #ax.plot(E,binned_spectra[i,:]/norm,label='UV/FUV flux ratio: %.2e-%2.e' % (10**UV_to_FUV_bins[i],10**UV_to_FUV_bins[i+1]),color=colors[i])
        #ax.plot(E,binned_spectra[i,:]/norm,'x',mew=1.5,color=colors[i])
        ax.plot(E,binned_spectra[i,:]/norm,label='NIR/FUV flux ratio: %.2e-%2.e' % (10**NIR_to_FUV_bins[i],10**NIR_to_FUV_bins[i+1]),color=colors[i])
        ax.plot(E,binned_spectra[i,:]/norm,'x',mew=1.5,color=colors[i])
    #     print(data1[i][0:10])
    #     s = asegs
    ax.plot([13.6,13.6],ax.get_ylim(),'--k')
    ax.legend()
    if not os.path.isdir(p.d_plot + 'cell_data/'): os.mkdir(p.d_plot + 'cell_data/')    
    plt.savefig(p.d_plot + 'cell_data/skirt_spectra_%s.png' % (gal_ob.name), format='png', dpi=250, facecolor='w')

    fig,ax = plt.subplots(figsize=(10,8))
    ax.hist(np.log10(R_NIR_FUV[R_NIR_FUV > 0]),bins=200)
    ax.set_xlabel(r'log NIR-to-FUV flux ratio')
    ax.set_ylabel(r'log fraction of cells')
    ax.set_yscale('log')
    plt.savefig(p.d_plot + 'cell_data/skirt_R_NIR_FUV_%s_no_Z.png' % (gal_ob.name), format='png', dpi=250, facecolor='w')


    fig,ax = plt.subplots(figsize=(10,8))
    ax.hist(np.log10(UV_to_FUV[UV_to_FUV > 0]),bins=200)
    ax.set_ylabel(r'log UV-to-FUV flux ratio')
    ax.set_yscale('log')
    plt.savefig(p.d_plot + 'cell_data/skirt_UV_to_FUV_%s.png' % (gal_ob.name), format='png', dpi=250, facecolor='w')

    fig,ax = plt.subplots(figsize=(10,8))
    ax.hist(np.log10(R_NIR_FUV[R_NIR_FUV > 0]),bins=200,weights=cell_data.m[R_NIR_FUV > 0])
    ax.set_xlabel(r'log NIR-to-FUV flux ratio')
    ax.set_ylabel(r'log mass-weighted fraction of cells')
    ax.set_yscale('log')
    plt.savefig(p.d_plot + 'cell_data/skirt_R_NIR_FUV_%s_mw.png' % (gal_ob.name), format='png', dpi=250, facecolor='w')


    pdb.set_trace()

#---------------------------------------------------------------------------
### FOR FRAGMENTATION TASK ###
#---------------------------------------------------------------------------

def three_PDF_plots(res=200,table_exts=[''],**kwargs):
    """ Plot total galactic PDF

    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)
    GR                      =   glo.global_results()

    fig, axs = plt.subplots(3, sharex='col',\
                figsize=(8,15),facecolor='w',\
                gridspec_kw={'hspace': 0, 'wspace': 0})

    # First print cell data distribution
    i = 0
    for gal_index in zip(p.gal_index):
        ax1                     =   axs[i]
        gal_ob                  =   gal.galaxy(GR=GR, gal_index=gal_index)
        df                      =   gal_ob.cell_data.get_dataframe()
        lognH                   =   np.log10(df.nH)
        hist                    =   np.histogram(lognH[df.nH.values > 0],bins=200,weights=df.m[df.nH.values > 0])
        hist1                   =   np.asarray(hist[0]) # histogram
        hist2                   =   np.asarray(hist[1]) # bin edges
        hist1                   =   hist1*1./sum(hist1)
        ax1.plot(hist2[0:len(hist1)],hist1,drawstyle='steps',ls='-',lw=1.5,\
             alpha=0.7,color=p.color[0],label='Original cell distribution')
        
        for table_ext,ls,color in zip(table_exts,['--',':'],p.color[1::]):
            if '_M10' in table_ext: lab = 'Mach = 10'
            if '_arepoPDF_M51' in table_ext: lab = 'AREPO parametrized PDF'
            PDF(gal_index,color=color,table_ext=table_ext,ls=ls,res=200,add=True,ax=ax1,label=lab,ow=p.ow)
        
        if i == 0: ax1.legend(loc='upper right',fontsize=12)
        if i == 2: ax1.set_xlabel(getlabel('lnH'))
        ax1.set_ylabel('Mass fraction per bin')

        i += 1

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'cell_data/PDFs/'): os.mkdir(p.d_plot + 'cell_data/PDFs/')    
        plt.savefig(p.d_plot + 'cell_data/PDFs/simple_PDF_%s%s%s_x3.png' % (p.sim_name,p.sim_run,p.table_ext), format='png', dpi=250, facecolor='w')

def PDF(gal_index,**kwargs):
    """ Plot total galactic PDF

    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # PDF PLACEHOLDER
    lognHs              =   np.linspace(-5,8,200)
    total_PDF           =   np.zeros(len(lognHs))

    # READ CELL DATA
    gal_ob              =   gal.galaxy(gal_index)
    df                  =   gal_ob.cell_data.get_dataframe()

    bins                =   50

    # READ FIT PARAMS OF PDF
    if '_arepoPDF' in p.table_ext:
        fit_params_SFR = np.load(p.d_table+'fragment/PDFs%s_%ipc.npy' % (p.table_ext,p.res),allow_pickle=True).item()
        fit_params = fit_params_SFR['fit_params']

        # OPTIONAL : SELECT PART OF FITS
        # fit_params_SFR['SFR_bins'] = fit_params_SFR['SFR_bins'][0:-2]
        # fit_params = fit_params[:,0:-2,:]
        # fit_params_collapse = fit_params_collapse[:,0:-2,:]

        fit_lognH_bins = fit_params_SFR['n_vw_bins'] # log
        fit_nSFR_bins = fit_params_SFR['SFR_bins'] # log
        fit_lognH_bins_c = fit_lognH_bins[0:-1] + (fit_lognH_bins[-1]-fit_lognH_bins[-2])/2
        fit_nSFR_bins_c = fit_nSFR_bins[0:-1] + (fit_nSFR_bins[-1]-fit_nSFR_bins[-2])/2
        lognSFR_bins        =   fit_nSFR_bins#np.linspace(fit_nSFR_bins.min(),fit_nSFR_bins.max(),bins)
        print('log nH bins:')
        print(fit_lognH_bins_c)
        print('log SFR bins:')
        print(fit_nSFR_bins_c)
    if '_arepoPDF' not in p.table_ext:
        lognSFR_bins        =   np.linspace(-10,1,bins)

    # BIN CELL DATA TO REDUCE COMPUTATION TIME
    lognH_bins          =   np.linspace(-8,2,bins)
    lognH_bins_c        =   lognH_bins[0:-1] + (lognH_bins[1] - lognH_bins[0])/2
    lognSFR_bins_c      =   lognSFR_bins[0:-1] + (lognSFR_bins[1] - lognSFR_bins[0])/2

    # ADD THIS LOWER VALUE TO INCLUDE ALL CELLS (except density = 0)
    lognH_bins[0]       =   -30
    lognSFR_bins[0]     =   -30
    lognSFR_bins[-1]     =   10

    df.SFR_density[df.SFR_density <= 10.**lognSFR_bins.min()] = 10.**(lognSFR_bins.min()+1)
    df.SFR_density[np.isnan(df.SFR_density)] = 10.**(lognSFR_bins.min()+1)

    if not p.add:
        fig                 =   plt.figure(figsize=(15,6))
        ax                  =   fig.add_subplot(1,2,1)

    print('Number of cells: ',len(df))
    if p.ow == False:
        try:
            PDF = pd.read_pickle(p.d_XL_data + 'data/cell_data/PDFs/%s%s_%s%s_%s' % (p.sim_name,p.sim_run,gal_ob.name,p.table_ext,p.res))
            total_PDF = PDF['total_PDF'].values
            lognHs = PDF['lognHs'].values
        except:
            p.ow = True
    if p.ow == True:
        print('Re-calculating PDF')
        i = 0
        poly1 = 0
        N_cells = 0
     
        for i_lognH in range(len(lognH_bins)-1):
            for i_lognSFR in range(len(lognSFR_bins)-1):
     
                df_cut                  =   df[(df.nH >= 10**(lognH_bins[i_lognH])) & \
                                            (df.nH < 10**(lognH_bins[i_lognH+1]))].reset_index(drop=True)
                if i_lognSFR > 0:
                    # (for the first bin in nSFR, doesn't matter if cell has no nSFR)
                    df_cut                  =   df_cut[(df_cut.SFR_density >= 10**(lognSFR_bins[i_lognSFR])) & \
                                                (df_cut.SFR_density < 10**(lognSFR_bins[i_lognSFR+1]))].reset_index(drop=True)
                N_cells += len(df_cut)
                lognH_mean, lognSFR     =   lognH_bins_c[i_lognH], lognSFR_bins_c[i_lognSFR]
     
                if '_arepoPDF' in p.table_ext:
                    # print(lognH_mean,lognSFR,len(df_cut))
                    if (lognH_bins[i_lognH] >= fit_lognH_bins[0]):
                        print(lognH_bins[i_lognH],len(df_cut))
                        i_fit_lognH_bins    =   np.argmin(np.abs(fit_lognH_bins_c - lognH_mean))
                        i_fit_lognSFR_bins  =   np.argmin(np.abs(fit_nSFR_bins_c - lognSFR))
                        fit_params_1        =   fit_params[i_fit_lognH_bins,i_fit_lognSFR_bins,:]
                        print(lognH_mean,lognSFR,fit_params_1)
     
                        if np.sum(fit_params_1) != 0:
                            PDF_integrated      =   10.**aux.parametric_PDF(lognHs,lognH_mean,fit_params_1[1],fit_params_1[2])
                            if fit_params_1[2] == -1.5:
                                PDF_integrated      =   10.**aux.parametric_PDF(lognHs,fit_params_1[0],fit_params_1[1],fit_params_1[2])
                            poly1 += 1
     
                        if np.sum(fit_params_1) == 0:
                            print('uhoh',lognH_mean,lognSFR)
                            PDF_integrated = aux.lognormal_PDF(10.**lognHs,10.**lognH_mean,Mach=1)
     
                    if (lognH_mean < fit_lognH_bins[0]):
                        PDF_integrated = aux.lognormal_PDF(10.**lognHs,10.**lognH_mean,Mach=10)
                        PDF_integrated[np.isnan(PDF_integrated)] = 0
                    if (lognH_mean < -4):
                        PDF_integrated = aux.lognormal_PDF(10.**lognHs,10.**lognH_mean,Mach=1)
                        PDF_integrated[np.isnan(PDF_integrated)] = 0
     
                if p.table_ext == '_M10':
                    PDF_integrated = aux.lognormal_PDF(10.**lognHs,10.**lognH_mean,Mach=10)
                    PDF_integrated[np.isnan(PDF_integrated)] = 0
     
                # Add to total PDF, weigthed by the mass of that cell
                total_PDF               +=  PDF_integrated * np.sum(df_cut.m)/np.sum(df.m)
                if not p.add: ax.plot(10.**lognHs,PDF_integrated * np.sum(df_cut.m)/np.sum(df.m),color='grey',lw=1,alpha=0.3)
                if np.isnan(np.sum(total_PDF)):
                    print(np.sum(df_cut.m)/np.sum(df.m),PDF_integrated)
                    pdb.set_trace()
                i += 1
                # if i == 10: pdb.set_trace()
     
        print('Total number of cells processed: ',N_cells)
        print('Total number of bins: ',bins**2)
        print('Number of bins with parametric PDFs: %i' % (poly1))
        total_PDF = total_PDF / np.sum(total_PDF)
        PDF = pd.DataFrame({'lognHs':lognHs,'total_PDF':total_PDF})
        PDF.to_pickle(p.d_XL_data + 'data/cell_data/PDFs/%s%s_%s%s_%s' % (p.sim_name,p.sim_run,gal_ob.name,p.table_ext,p.res))

    print('TEST!!!')
    total_PDF = total_PDF[(lognHs >= -4) & (lognHs <= 7)]
    lognHs = lognHs[(lognHs >= -4) & (lognHs <= 7)]
    total_PDF               =  total_PDF / np.sum(total_PDF)
    if not p.add:
        # First figure: One panel of individual binned PDFs and one panel of total PDF
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(getlabel('lnH'))
        ax.set_ylabel('dM/dlognH')
        ax.set_ylim([1e-12,1e-1])
        ax.set_xlim([1e-4,1e7])
     
        ax2 = fig.add_subplot(1,2,2)
        ax2.plot(10.**lognHs,total_PDF)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(getlabel('lnH'))
        ax2.set_ylabel('dM/dlognH')
        ax2.set_ylim([1e-4,1e-1])
        ax2.set_xlim([1e-4,1e5])
     
        if not os.path.isdir(p.d_plot + 'cell_data/PDFs/'): os.mkdir(p.d_plot + 'cell_data/PDFs/')    
        plt.savefig(p.d_plot + 'cell_data/PDFs/PDF_%s%s_%s.png' % (gal_ob.name,p.table_ext,p.res), format='png', dpi=250, facecolor='w')

    labels = {'_M10':'Mach = 10','_arepoPDF_M51':'AREPO-M51 parametrized PDF','_arepoPDF_CMZ':'AREPO-CMZ parametrized PDF'}

    # New figure: One panel of PDF and cumulative mass function (optional)
    if p.add:
        ax1 = p.ax#plt.gca()
    else:
        fig,ax1                 =   plt.subplots(figsize=(8,6))
    ax1.plot(lognHs,total_PDF,ls=p.ls,lw=2.5,color=p.color,label=labels[p.table_ext])
    ax1.set_yscale('log')
    if not p.add:
        ax1.set_xlabel('log nH [cm$^{-3}$]')
        ax1.set_ylabel('Mass fraction per bin')
    ax1.set_xlim([-4,7])
    ax1.set_ylim([1e-4,1e-1])
    ax1.grid(axis='x')
    #if p.add: ax1.legend()
    if not p.add:
        ax2 = ax1.twinx()
        ax2.plot(lognHs,np.cumsum(total_PDF),'--')
        ax2.grid(axis='y')
        ax2.set_ylim([0,1])
        ax2.set_ylabel('Cumulative mass fraction')
        ax2.text(0.4,0.1,'Mass fraction at nH > 1e3: %.1f %%' % (100*np.sum(total_PDF[lognHs >= 3])),\
                 transform=ax1.transAxes,fontsize=15,bbox=dict(facecolor='white', alpha=0.7))
    if not os.path.isdir(p.d_plot + 'cell_data/PDFs'): os.mkdir(p.d_plot + 'cell_data/PDFs')    
    if not p.add: plt.savefig(p.d_plot + 'cell_data/PDFs/simple_PDF_%s%s_%s.png' % (gal_ob.name,p.table_ext,p.res), format='png', dpi=250, facecolor='w')

    # pdb.set_trace()

def cell_properties(**kwargs):
    """ Plot the following for all cells in SKIRT output structure:
    - nH
    - cell size
    - cell mass
    - FUV flux
    Properties for look-up table interpolation:
    - Z
    - nH
    - Mach number
    - FUV flux
    And derived properties:
    - Z
    - Sigma_gas
    - f_H2
    - vel disp
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    mpl.rcParams['ytick.labelsize'] = 13
    mpl.rcParams['xtick.labelsize'] = 13

    # Cloudy lookup table
    cloudy_library = clo.library()
    lookup_table = cloudy_library._restore_lookup_table()
    lognHs = np.unique(lookup_table.lognHs)
    Machs = np.unique(lookup_table.Machs)
    logZs = np.unique(lookup_table.logZs)
    logFUVs = np.unique(lookup_table.logFUVs)

    color,alpha = 'teal',0.7
    weigh       =   'm'
    for gal_index in p.gal_indices:

        # Load sim and cell data
        gal_ob          =   gal.galaxy(gal_index)
        simgas          =   gal_ob.particle_data.get_dataframe('simgas')
        df              =   gal_ob.cell_data.get_dataframe()

        ################### From SKIRT only

        # Set figure up
        fig         =   plt.figure(figsize = (13,13))
        plt.title('Cell properties from SKIRT')
        plt.axis('off')

        # nH
        ax1         =   fig.add_subplot(2,2,1)
        ax2 = ax1.twinx()
        if weigh == 'm':
            hist        =   np.histogram(np.log10(simgas.nH[simgas.nH > 0]),bins=p.bins,weights=simgas.m[simgas.nH > 0])
            hist1            =  np.asarray(hist[0])
            hist2            =  np.asarray(hist[1])
            hist1            =  hist1*1./sum(hist1)*100.
            wid              =  (hist2[1]-hist2[0])
            ax2.plot(hist2[0:len(hist2)-1]+wid/2,hist1,ls='-',drawstyle='steps',\
                color='k',label='Original from simulations',alpha=0.7,lw=1)
        else:
            ax2.hist(np.log10(simgas.nH[simgas.nH > 0]),color='grey',\
                bins=100,label='Original from simulations',alpha=0.7,lw=1)
        if weigh == 'm':
            hist            =   np.histogram(np.log10(df.nH[df.nH > 0]),bins=p.bins,weights=df.m[df.nH > 0])
            hist1            =  np.asarray(hist[0])
            hist2            =  np.asarray(hist[1])
            hist1            =  hist1*1./sum(hist1)*100.
            wid              =  (hist2[1]-hist2[0])
            ax1.plot(hist2[0:len(hist2)-1]+wid/2,hist1,ls='-',drawstyle='steps',\
                color=color,label='From SKIRT cells',alpha=0.7,lw=1)
            ax1.set_ylim([1e-6,200])
            ax1.set_ylabel('Mass fraction [%]')
        else:
            ax1.hist(np.log10(df.nH[df.nH > 0]),color=color,bins=100,label='From SKIRT cells',alpha=0.7,lw=1)
        ax1.set_xlabel('log('+getlabel('nH')+')')
        ax1.legend(loc='center left'); ax2.legend(loc='upper left')
        ax1.set_yscale('log'); ax2.set_yscale('log')


        # Cell size
        ax1         =   fig.add_subplot(2,2,2)
        ax1.hist(np.log10(df.cell_size*1e3),color=color,bins=100,label='From SKIRT cells',alpha=0.7,lw=1)
        ax1.set_xlabel('log(Cell size [pc])')

        # Cell mass
        # df['m']     =   df.nH.values * (df.cell_size*p.kpc2cm)**3 * p.mH / p.Msun # Msun
        ax1         =   fig.add_subplot(2,2,3)
        ax1.hist(np.log10(df.m[df.m > 0]),color=color,bins=100,label='From SKIRT cells',alpha=0.7,lw=1)
        ax1.set_xlabel('log(Cell mass [M$_{\odot}$])')
        ax1.legend(loc='center left'); ax2.legend(loc='upper left')
        ax2 = ax1.twinx()
        ax2.hist(np.log10(simgas.m[simgas.m > 0]),color='grey',\
            bins=100,label='Original from simulations',alpha=0.7,lw=1)
        ax2.set_ylim([0,500])

        # FUV flux
        ax1         =   fig.add_subplot(2,2,4)
        ax1.hist(np.log10(df.F_FUV_W_m2[df.F_FUV_W_m2 > 0]),color=color,bins=100,label='From SKIRT cells',alpha=0.7,lw=1)
        ax1.set_xlabel('log('+getlabel('F_FUV_W_m2')+')')

        if not os.path.exists(p.d_plot + 'cell_data'): os.mkdir(p.d_plot + 'cell_data')

        plt.tight_layout()

        # plt.savefig(p.d_plot + 'cell_data/properties_SKIRT.png', format='png', dpi=250, facecolor='w')


        ################### Cell properties for look-up table interpolation

        # Set figure up
        fig         =   plt.figure(figsize = (13,13))
        plt.title('Cell properties for look-up table interpolation')
        plt.axis('off')

        # nH
        ax1         =   fig.add_subplot(2,2,1)
        # plot lookup table in the back
        for lognH in lognHs:
            ax1.plot([lognH,lognH],[1e-10,1e3],'-.',lw=1,color='grey')
        ax2 = ax1.twinx()
        if weigh == 'm':
            hist             =  np.histogram(np.log10(simgas.nH[simgas.nH > 0]),bins=p.bins,weights=simgas.m[simgas.nH > 0])
            hist1            =  np.asarray(hist[0])
            hist2            =  np.asarray(hist[1])
            hist1            =  hist1*1./sum(hist1)*100.
            wid              =  (hist2[1]-hist2[0])
            ax2.plot(hist2[0:len(hist2)-1]+wid/2,hist1,ls='-',drawstyle='steps',\
                color='k',label='Original from simulations',alpha=0.7,lw=2)
        else:
            ax2.hist(np.log10(simgas.nH[simgas.nH > 0]),color='k',\
                bins=100,label='Original from simulations',alpha=0.7,lw=2)
        if weigh == 'm':
            print('Max nH: %.2f' % (np.max(df.nH)))
            hist             =  np.histogram(np.log10(df.nH[df.nH > 0]),bins=p.bins,weights=df.m[df.nH > 0])
            hist1            =  np.asarray(hist[0])
            hist2            =  np.asarray(hist[1])
            hist1            =  hist1*1./sum(hist1)*100.
            wid              =  (hist2[1]-hist2[0])
            ax1.plot(hist2[0:len(hist2)-1]+wid/2,hist1,ls='-',drawstyle='steps',\
                color=color,label='From SKIRT cells',alpha=0.7,lw=2)
            ax1.set_ylim([1e-6,200])
            ax1.set_ylabel('Mass fraction [%]')
        else:
            ax1.hist(np.log10(df.nH[df.nH > 0]),color=color,bins=100,label='From SKIRT cells',alpha=0.7,lw=1)
        ax1.set_xlabel('log('+getlabel('nH')+')')
        ax1.legend(loc='center left'); ax2.legend(loc='upper left')
        ax1.set_yscale('log'); ax2.set_yscale('log')
        ax1.set_xlim([np.min(lognHs)-1,np.max(lognHs)+1])
        # ax1         =   fig.add_subplot(2,2,1)
        # ax2 = ax1.twinx()
        # ax2.hist(np.log10(simgas.nH[simgas.nH > 0]),color='grey',\
        #     bins=100,label='Original from simulations',alpha=0.7,lw=1)
        # ax1.hist(np.log10(df.nH[df.nH > 0]),color=color,bins=100,\
        #     label='From SKIRT cells',alpha=0.7,lw=1)
        # ax1.set_xlabel('log('+getlabel('nH')+')')
        # ax1.set_yscale('log'); ax2.set_yscale('log')
        # ax1.legend(loc='center left'); ax2.legend(loc='upper left')

        # Z
        ax1         =   fig.add_subplot(2,2,2)
        # plot lookup table in the back
        for logZ in logZs:
            ax1.plot([logZ,logZ],[1e-10,1e5],'-.',lw=1,color='grey')
        ax2 = ax1.twinx()
        ax2.hist(np.log10(simgas.Z[simgas.Z > 0]),color='grey',\
            bins=100,label='Original from simulations',alpha=0.7,lw=1)
        ax1.hist(np.log10(df.Z[df.Z > 0]),color='teal',\
            bins=100,label='Evaluated at cell centers',alpha=0.7,lw=1)
        ax1.set_xlabel('log('+getlabel('Z')+')')
        ax1.legend(loc='center left'); ax2.legend(loc='upper left')
        ax1.set_ylim([1,1e4]); ax1.set_yscale('log')

        # Mach number
        ax1         =   fig.add_subplot(2,2,3)
        # plot lookup table in the back
        for Mach in Machs:
            ax1.plot([Mach,Mach],[1e-10,1e5],'-.',lw=1,color='grey')
        # ax1.hist(np.log10(simgas.Mach),color='grey',\
        #     bins=100,label='Original from simulations',alpha=0.7,lw=1)
        ax1.hist(df.Mach,color=color,bins=100,\
            label='From SKIRT cells',alpha=0.7,lw=1)
        ax1.set_xlabel(getlabel('Mach'))
        ax1.set_ylim([1,1e6]); ax1.set_yscale('log')

        # FUV flux
        ax1         =   fig.add_subplot(2,2,4)
        # plot lookup table in the back
        for logFUV in logFUVs:
            ax1.plot([logFUV,logFUV],[1e-10,1e5],'-.',lw=1,color='grey')
        ax1.hist(np.log10(df.G0[df.G0 > 1e-6]),density=True,weights=df.m.values[df.G0 > 1e-6],color=color,bins=100,\
            label='From SKIRT cells',alpha=0.7,lw=1)
        # ax1.hist(np.log10(100*df['F_FUV_Habing'][df['F_FUV_Habing'] > 1e-6]),color=color,bins=100,label='From SKIRT cells',alpha=0.7,lw=1)
        ax1.set_xlabel('log('+getlabel('G0')+')')
        # ax1.set_xlabel('log('+getlabel('F_FUV_Habing')+')')
        ax1.set_ylim([1e-6,2]); ax1.set_yscale('log')

        plt.tight_layout()

        if not os.path.isdir(p.d_plot + 'cell_data/'): os.mkdir(p.d_plot + 'cell_data/')    
        plt.savefig(p.d_plot + 'cell_data/%s_for_Cloudy.png' % gal_ob.name, format='png', dpi=250, facecolor='w')

        ################### Derived in post-process

        # # Set figure up
        # fig         =   plt.figure(figsize = (13,13))
        # plt.title('Cell properties derived in post-process of SKIRT')
        # plt.axis('off')

        # # Z
        # ax1         =   fig.add_subplot(2,2,1)
        # ax1.hist(np.log10(simgas.Z[simgas.Z > 0]),color='grey',\
        #     bins=100,label='Original from simulations',alpha=0.7,lw=1)
        # print(np.min(df.Z))
        # print(np.max(df.Z))

        # ax1.hist(np.log10(df.Z[df.Z > 0]),color='teal',\
        #     bins=100,label='Evaluated at cell centers',alpha=0.7,lw=1)
        # ax1.set_xlabel('log('+getlabel('Z')+')')
        # ax1.legend()

        # # Gas surface density
        # ax1         =   fig.add_subplot(2,2,2)
        # ax1.hist(np.log10(df.surf_gas_or[df.surf_gas_or > 0]),color='grey',\
        #     bins=100,label='From cell densities and sizes alone',alpha=0.7,lw=1)
        # ax1.hist(np.log10(df.surf_gas[df.surf_gas > 0]),color='teal',\
        #     bins=100,label='Boosted for cell sizes > 100 pc',alpha=0.7,lw=1)
        # ax1.set_xlabel('log('+getlabel('surf_gas')+')')
        # ax1.legend()

        # # f_H2
        # ax1         =   fig.add_subplot(2,2,3)
        # # ax1.hist(np.log10(df.f_H2_NK14[df.f_H2_NK14 > 0]),color='grey',\
        # #     bins=100,label='NK+14',alpha=0.7,lw=1)
        # ax1.hist(np.log10(simgas.f_H2[simgas.f_H2 > 0]),color='grey',\
        #     bins=100,label='Original from simulations',alpha=0.7,lw=1)
        # ax1.hist(np.log10(df.f_H2[df.f_H2 > 0]),color='teal',\
        #     bins=100,label='KMT+09',alpha=0.7,lw=1)
        # ax1.set_xlabel('log('+getlabel('f_H2')+')')
        # ax1.legend()
        # # ax1.set_xlim([-2.5,0])
        # print('Max f_H2 and number with f_H2 > 0 in sims:')
        # print(np.max(simgas.f_H2))
        # print(len(simgas.f_H2[simgas.f_H2 > 0]))


        # # Velocity dispersion on cloud scales
        # ax1         =   fig.add_subplot(2,2,4)
        # ax1.hist(np.log10(simgas.vel_disp[simgas.vel_disp > 0]),color='grey',\
        #     bins=100,label='From simulation particles',alpha=0.7,lw=1)
        # ax1.hist(np.log10(df.vel_disp_cloud[df.vel_disp_cloud > 0]),color='teal',\
        #     bins=100,label='Scaled down and evaluated at cell centers',alpha=0.7,lw=1)
        # ax1.legend(fontsize=12)

        # ax1.set_xlabel('log('+getlabel('vel_disp_gas')+')')

        # plt.savefig(p.d_plot + 'cell_data/properties_SKIRT_post.png', format='png', dpi=250, facecolor='w')

#---------------------------------------------------------------------------
### FOR CLOUDY MODELING ###
#---------------------------------------------------------------------------

def NH_function(**kwargs):
    """ Plots the NH(R_NIR_FUV) function used in galaxy.py get_NH_from_cloudy() 
    """
    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    logNH_cl,R_NIR_FUV_cl = aux.get_NH_from_cloudy()

    interp                  =   interp1d(np.log10(R_NIR_FUV_cl)[8::],logNH_cl[8::],fill_value='extrapolate',kind='slinear')     
    fig = plt.figure(figsize=(6,6.15))
    ax1 = fig.add_axes([0.2, 0.2, 0.79, 0.79]) 

    ax1.plot(np.log10(R_NIR_FUV_cl),logNH_cl,'o',ms=5)
    R_fit = np.arange(np.log10(R_NIR_FUV_cl).min(),np.log10(R_NIR_FUV_cl).max(),0.1)
    print(R_fit)
    print(interp(R_fit))
    ax1.plot(R_fit,interp(R_fit),'--r')

    ax1.set_xlabel('log cloudy NIR/FUV ratio')
    ax1.set_ylabel('log cloudy N$_H$ [cm${-2}$] ')
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'cloudy/'): os.mkdir(p.d_plot + 'cloudy/')    
        plt.savefig(p.d_plot + 'cloudy/NH_function.png', format=p.format, dpi=250)



def BPASS_LM_grid(**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    BPASS_LM_grid_params = pd.read_pickle(p.d_table + 'cloudy/' + 'BPASS_LM_params')
    BPASS_LM_grid = np.load(p.d_table + 'cloudy/' + 'BPASS_LM_grid_bol.npy')
    Zs = np.unique(BPASS_LM_grid_params['logZ'])
    ages = np.unique(BPASS_LM_grid_params['logAge'])

    fig,ax = plt.subplots(figsize=(13,7))
    cmap = plt.get_cmap('gnuplot2')
    colors0 = [cmap(i) for i in np.linspace(0, 1, len(Zs)+1)]
    for i in range(len(Zs)):
        ax.plot(ages,BPASS_LM_grid[:,i],color=colors0[i],label='Z: %.3f Z$_{\odot}$' % (10.**Zs[i]))

    ax.plot([np.log10(5e9),np.log10(5e9)],ax.get_ylim(),'--',label='Sun age')
    index_sun = np.argmin(np.abs(ages - np.log10(5e9)))
    ax.plot([ax.get_xlim()[0],ages[index_sun]],[BPASS_LM_grid[index_sun,2],BPASS_LM_grid[index_sun,2]],'--')


    ax.legend()
    ax.set_xlabel('log stellar age [yr]')
    ax.set_ylabel('log bolometric L/M ratio')

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig('plots/look-up/BPASS_LM_grid',dpi=200)

def BPASS_spectra(**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    filename = 'spectra-sin-imf_chab300.z020.dat'

    spec = pd.read_csv('../code/c17.02/stellar_SEDs/BPASSv2.2.1_chab300/' + filename,delim_whitespace=True,names=['AA'] + list(np.arange(51).astype(str)))
    spec['eV'] = c.c.value/(1e-10*np.array(spec.AA.values))*c.h.value*u.J.to(u.eV)
    xlim_eV = [0.1,1e5]
    spec = spec[(spec['eV'] > xlim_eV[0]) & (spec['eV'] <= xlim_eV[1])].reset_index(drop=True)
    print(spec.head())

    fig,ax = plt.subplots(figsize=(13,7))
    ax2 = ax.twiny()
    cmap = plt.get_cmap('gnuplot2')
    colors0 = [cmap(i) for i in np.linspace(0, 1, 8+2)]
    j = 0
    for i in [0,5,10,15,20,30,40,50]:
        ax.plot(spec.AA.values, spec['%i' % i],color=colors0[j],label='age: %.2e yr' % (10**(6+0.1*(i))))
        j += 1
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([spec.AA.min(),spec.AA.max()])

    # Add eV scale
    xlim_eV = c.c.value/(1e-10*np.array(ax.get_xlim()))*c.h.value*u.J.to(u.eV)
    ax2.plot([1e6],[1e10])
    print('aaaa',xlim_eV[::-1])
    ax2.set_xscale('log')
    ax2.set_xlim(xlim_eV[::-1])
    ax2.invert_xaxis()
    print(ax.get_xlim())
    print(ax2.get_xlim())
    ax2.plot([6,6],ax.get_ylim(),'--')
    ax2.plot([13.6,13.6],ax.get_ylim(),'--')
    print(ax2.get_xlim())
  
    ax.legend()
    ax.set_xlabel('$\AA$')
    ax.set_ylabel('Lsun/$\AA$')
    ax.set_ylim([1e-4,1e8])


    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig('plots/look-up/BPASS_spectra',dpi=200)

def attenuated_spectra(**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    filename = 'grid_run_ext_ism_BPASS'

    grd = pd.read_csv(p.d_cloudy + 'NH/%s.grd' % filename,skiprows=1,sep='\t',names=['i','Failure','Warnings','Exit code','rank','seq','logNH','gridparam'])
    exts = grd['logNH'].values#[grd['Exit code'] != '       failed assert']

    print(exts)

    cont2 = pd.read_table(p.d_cloudy + 'NH/%s.cont2' % filename,skiprows=0,names=['E','I_trans','coef'])
    E = cont2.E.values
    i_shift = np.array(['########################### GRID_DELIMIT' in _ for _ in E])
    i_delims = np.arange(len(cont2))[i_shift == True]
    N_E = i_delims[0]-9 # First 9 lines are header
    cont = np.zeros([len(i_delims),N_E])
    for i,i_delim in enumerate(i_delims):
        I_trans = cont2.I_trans[i_delim-N_E:i_delim].astype(float)
        cont[i,:] = I_trans
    cont_cut = cont#[Zs == 0,:]
    fig,ax = plt.subplots(figsize=(13,7))
    ax.set_ylim([1e-16,1e2])
    NIR = aux.nm_to_eV(np.array([2,0.5])*1e3)
    ax.fill_between(NIR,[ax.get_ylim()[1],ax.get_ylim()[1]],color='red',alpha=0.3,label='OIR band used here')
    NUV = aux.nm_to_eV(np.array([400,300]))
    # ax.fill_between(NUV,[ax.get_ylim()[1],ax.get_ylim()[1]],color='blue',alpha=0.1,label='NUV')
    MUV = aux.nm_to_eV(np.array([300,200]))
    # ax.fill_between([MUV[0],6],[ax.get_ylim()[1],ax.get_ylim()[1]],color='green',alpha=0.1,label='MUV')
    FUV = np.array([6,13.6])
    ax.fill_between(FUV,[ax.get_ylim()[1],ax.get_ylim()[1]],color='teal',alpha=0.3,label='FUV band used here')
    x = (E[i_delims[0]-N_E:i_delims[0]]).astype(float)*u.Ry.to('eV') # eV
    cmap = plt.get_cmap('gnuplot2')
    colors0 = [cmap(i) for i in np.linspace(0, 1, cont_cut.shape[0]+2)]
    logNHs,R_NIR_FUV = aux.get_NH_from_cloudy()
    i = 0
    for ext in exts:
        #ax.plot(x,cont_cut[i,:],c=colors0[i],alpha=0.7,lw=2.5,label='logR$_{\mathrm{NIR/FUV}}$ = %.1f (logN$_\mathrm{H}$ = %.1f)' % (np.log10(R_NIR_FUV[i]),ext))
        ax.plot(x,cont_cut[i,:],c=colors0[i],alpha=0.7,lw=2.5,label='logR$_{\mathrm{OIR/FUV}}$ = %.2f' % (np.log10(R_NIR_FUV[i])))# (logN$_\mathrm{H}$ = %.1f)' % (,ext))
        i += 1
    ax.legend(loc='upper right',fontsize=13)
    #dext = exts[1]-exts[0]
    #m = ax.scatter([1e9,1e9],[1e9,1e9],c=[exts.min(),exts.max() + dext*2],cmap='gnuplot2')
    #plt.colorbar(m,label='log N$_{\mathrm{H}}$ [cm$^{-2}$]')
    #dR = R_NIR_FUV[1]-R_NIR_FUV[0]
    #m = ax.scatter([1e9,1e9],[1e9,1e9],c=[R_NIR_FUV.min(),R_NIR_FUV.max() + dext*2],cmap='gnuplot2')
    #plt.colorbar(m,label='NIR/FUV flux ratio')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([1e-1,1e5])
    ax.set_xlabel('E [eV]')
    ax.set_ylabel('I [ergs/s/cm$^2$]')
    ax.plot([13.6,13.6],ax.get_ylim(),'--k')
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig('plots/look-up/attenuated_spectra_BPASS',dpi=200)

def attenuated_spectrum_for_Raga(**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    filename = 'for_Raga'

    fixed_logNH = 21

    # CONTINUUM
    #cont2 = pd.read_table(p.d_cloudy + 'NH/%s.cont2' % filename,skiprows=10,names=['E','I_trans','coef'])
    #E = cont2.E.values
    #x = E.astype(float)*u.Ry.to('J') 
    #cont = cont2.I_trans.astype(float)
    #wa_cont = c.c.value / (x/c.h.value) *1e10 # J -> AA
    cont = pd.read_table(p.d_cloudy + 'NH/%s.cont' % filename,skiprows=10,names=['E','I_inc','I_trans','DiffOut','net trans','reflc','total','reflin','outlin','lineID','cont','nLine coef'])
    #print(cont.head())
    #print(cont.describe())
    wa_cont = c.c.value / (cont.E.values*u.Ry.to('J')/c.h.value) *1e10 # Rydberg -> AA
    r = cont['net trans'].values/cont['outlin'].values
    print((r[cont['outlin'] > 0]).min())
    cont_or = cont['net trans'].values # ergs/cm^s/s 
    print(cont['net trans'])
    #print(cont['net trans'].values[0:200])
    #print(cont['outlin'].values[0:200]) # ergs/cm^s/s 
    cont_line_boost = cont['net trans'].values + 1e2*cont['outlin'].values # ergs/cm^s/s 

    # Convert to ergs/s/cm2/AA
    dAA = np.abs(np.diff(wa_cont))
    cont_or = cont_or[0:-1]/dAA
    cont_line_boost = cont_line_boost[0:-1]/dAA
    print(cont_line_boost)

    # LINES
    #lines = pd.read_table(p.d_cloudy + 'NH/%s.lines' % filename,skiprows=352,names=['E','ID','I_int','I_em','type'],sep='\t')
    #lines = lines.iloc[0:1190-352-1].reset_index(drop=True)
    #x = lines.E.values.astype(float)*u.Ry.to('J') # eV
    #wa_lines = c.c.value / (x/c.h.value) *1e10 # J -> AA
    #F_lines = 10.**lines['I_em'].values
    #print(F_lines.max())
    #for wa,F_line in zip(wa_lines,F_lines):
    #    i = np.argmin(np.abs(wa_cont-wa))
    #    print(i,cont[i],F_line)
    #    cont[i] += F_line

    #print(x[(wa > 3000) & (wa < 6000)])
    #print(wa[(wa > 3000) & (wa < 6000)])
    fig,ax = plt.subplots(figsize=(13,7))
    ax.plot(wa_cont[0:-1],cont_line_boost,c='b',alpha=0.7,lw=1.5,label='Continuum + line boost')
    ax.plot(wa_cont[0:-1],cont_or,'r--',alpha=0.7,lw=1,label='Original continuum + lines')
    ax.legend(loc='upper right',fontsize=13)
    ax.set_yscale('log')
    ax.set_xlim([30,6000])
    ax.set_ylim([1e35,1e41])
    ax.set_xlabel('$\lambda$ [$\AA$]')
    ax.set_ylabel('I [ergs/s/$\AA$]')
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig('plots/look-up/attenuated_spectrum_Z1_NH%i' % fixed_logNH,dpi=200)



def emission_vs_depth(filename,**kwargs):
    """ Plot line emission vs depth for multi-cell Cloudy models in the NH folder
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if p.add:
        ax = plt.gca(); c = 'r'
    if not p.add:
        fig,ax = plt.subplots(figsize=(8,6)); c = 'b'
    st_cols = ['depth','[CII]158','[OI]63','CO(1-0)','CO(2-1)','CO(3-2)']
    st = pd.read_csv(p.d_cloudy.replace('ext/','') + 'NH/' + filename + '.str',sep='\t',skiprows=1,names=st_cols)
    dx = np.append(0,np.diff(st.depth))
    pc2cm = u.parsec.to(u.cm)
    # Derive mass-luminosity ratio
    import astropy.constants as c
    M = 1e3 * c.m_p.value * st.depth.values.max() / u.M_sun.to(u.kg) 
    cloudy_lin_header = ['#lineslist','C  1 609.590m','C  1 370.269m','C  2 157.636m','O  1 63.1679m','O  1 145.495m','O  3 88.3323m','N  2 205.244m','N  2 121.767m','CO   2600.05m','CO   1300.05m','CO   866.727m','CO   650.074m','CO   325.137m','H2   17.0300m','H2   12.2752m','H2   9.66228m','H2   8.02362m','H2   6.90725m','H2   6.10718m','H2   5.50996m','O  4 25.8832m','NE 2 12.8101m','NE 3 15.5509m','S  3 18.7078m','FE 2 25.9811m']
    cloudy_lin = pd.read_csv(p.d_cloudy.replace('ext/','') + 'NH/' + filename + '.lin',\
        sep='\t',names=cloudy_lin_header,comment='#').reset_index(drop=True)
    Cloudy_lines_dict = aux.get_Cloudy_lines_dict()
    cloudy_lin = cloudy_lin.rename(columns=Cloudy_lines_dict)
    L = cloudy_lin['CO(1-0)'][0] * u.erg.to('J') / c.L_sun.value
    print(L,M)
    ax.plot(st.depth/pc2cm,dx*st['CO(1-0)'],'-',color='m',label='CO(1-0): %.2e Lsun/Msun' % (L/M))
    L = cloudy_lin['[OI]63'][0] * u.erg.to('J') / c.L_sun.value
    print(L,M)
    ax.plot(st.depth/pc2cm,dx*st['[OI]63'],'g--',label='[OI]63: %.2e Lsun/Msun' % (L/M))
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel('Depth [pc]')
    ax.set_ylabel('Intensity [ergs/s/cm^2]')
    ax.legend()
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig('plots/look-up/emission_%s' % filename,dpi=200)

def mass_luminosity_ratio(filename,**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    cont1 = pd.read_table(p.d_cloudy + 'tests/%s.txt' % filename,sep=' ',skiprows=0,names=['E','I_inc'])
    E = cont1.E.values
    E_eV = E.astype(float)*u.Ry.to('eV')
    print(E_eV.min(),E_eV.max())
    E_Hz = E_eV * u.eV.to(u.J) / c.h.value
    I_erg_s_cm2_Hz = cont1.I_inc.values 
    Lbol_sun =  4 * np.pi * (0.1*u.parsec.to(u.cm))**2 * scipy.integrate.simps(I_erg_s_cm2_Hz,E_Hz) * 1e-7 / c.L_sun.value 
    #Lbol_sun = scipy.integrate.simps(I_erg_s_Hz,E_Hz) * 1e-7 / c.L_sun.value 
    #print(c.L_sun.value)
    #Lbol_sun =  np.sum(I_erg_s) * 1e-7 / c.L_sun.value 
    print(Lbol_sun)

    M       =  1e6 # from input file 
    print(M)

    print('Luminosity to mass ratio: %.2f Lsun/Msun' % (Lbol_sun/M))


#---------------------------------------------------------------------------
### FOR INTERPOLATION TASK ###
#---------------------------------------------------------------------------

def sim_params(x,y,**kwargs):
    GR                      =   glo.global_results()

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # SELECT GALAXIES
    rand_gal_index = np.random.randint(0, GR.N_gal, size=(p.bins))
    if p.bins == GR.N_gal: rand_gal_index = np.arange(GR.N_gal)
    if p.gal_index: 
        rand_gal_index = [p.gal_index]
    print(rand_gal_index)
    xs = np.array([])
    ys = np.array([])
    m_tot,m_encomp,m_y0 = 0,0,0
    for gal_index in rand_gal_index:
        print(gal_index)
        gal_ob              =   gal.galaxy(gal_index)
        df                  =   gal_ob.particle_data.get_dataframe('simgas')
        x1                  =   df[x].values
        y1                  =   df[y].values
        print(np.max(x1))
        print(np.max(y1))
        x1[x1 <= p.xlim[0]] = p.xlim[0]
        y1[y1 <= p.ylim[0]] = p.ylim[0]
        m_tot               +=   np.sum(df.m.values)
        m_encomp            +=   np.sum(df.m[(x1>=p.xlim[0]) & (y1>=p.ylim[0])].values)
        m_y0                +=   np.sum(df.m[(y1 == 0)].values)
        ys                  =   np.append(ys,y1[(x1>=p.xlim[0]) & (y1>=p.ylim[0])])
        xs                  =   np.append(xs,x1[(x1>=p.xlim[0]) & (y1>=p.ylim[0])])
    print('Min max of %s:' % x)
    print(xs.min(),xs.max())
    print('Min max of %s:' % y)
    print(ys.min(),ys.max())
    fig,ax = plt.subplots(figsize=(10,8))
    hb = ax.hexbin(xs,ys,xscale='log',yscale='log',bins='log',mincnt=1,lw=None,gridsize=50,cmap='inferno')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of cells in %i galaxies' % len(rand_gal_index))
    ax.set_xlabel(getlabel(x))
    ax.set_ylabel(getlabel(y))
    print('Total gas mass fraction encompassed: %.4f%%' % (m_encomp/m_tot*100))
    print('Total gas mass fraction with y = 0: %.4f%%' % (m_y0/m_tot*100))
    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)

    if p.select == 'bin':
        
        binned_x = np.linspace(np.min(np.log10(xs)),np.max(np.log10(xs)),30)
        binned_x_c = binned_x[0:-1] + (binned_x[1]-binned_x[0])/2
        binned_y = binned_x_c*0.
        print(binned_x)
        for i in range(len(binned_x) -1):
            binned_y[i] = np.median(np.log10(ys)[(xs >= 10**binned_x[i]) & (xs <= 10**binned_x[i+1]) & (ys > 2*p.ylim[0])])
        ax.plot(10**binned_x_c,10**binned_y,color='green',lw=4)
        print(binned_y)
    if not os.path.isdir(p.d_plot + 'sim_data/'): os.mkdir(p.d_plot + 'sim_data/')    
    plt.savefig('plots/sim_data/%s%s_sim_params_%s_%s_%s.png' % (p.sim_name,p.sim_run,p.z1,x,y),dpi=250, facecolor='w')

def cell_params(x,y,**kwargs):
    """ Plot contour map of cell properties for comparison with Cloudy look-up table parameters.

    Parameters
    ----------

    cloudy_param : dict
        Dictionary with the cloudy parameter name as key and value to be kept fixed as value.

    line : str
        Line name whos luminosity will be plotted in the z direction.


    """

    GR                      =   glo.global_results()

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    cloudy_library = clo.library()
    lookup_table = cloudy_library._restore_lookup_table()
    lookup_table['logG0s'] = lookup_table['logFUVs']
    if x == 'NH': 
        x_cloudy,R_NIR_FUV_cl = aux.get_NH_from_cloudy()
    else:
        x_cloudy = np.unique(lookup_table['log'+x+'s'])
    if y == 'NH': 
        y_cloudy,R_NIR_FUV_cl = aux.get_NH_from_cloudy()
    else:
        y_cloudy = np.unique(lookup_table['log'+y+'s'])

    if not p.ylim:
        p.ylim = [1e-3,30]
    if not p.xlim:
        p.xlim = [1e-7,1e3]
 
    # SELECT GALAXIES
    rand_gal_index = np.random.randint(0, GR.N_gal, size=(p.bins))
    if p.bins == GR.N_gal: rand_gal_index = np.arange(GR.N_gal)
    if p.gal_index: 
        rand_gal_index = [p.gal_index]
    print(rand_gal_index)
    xs = np.array([])
    ys = np.array([])
    m_tot,m_encomp,m_y0 = 0,0,0
    for gal_index in rand_gal_index:
        print(gal_index)
        gal_ob              =   gal.galaxy(gal_index)
        df                  =   gal_ob.cell_data.get_dataframe()
        df['nSFR']          =   df.nSFR.values#/(0.2**3)
        #df['nSFR']          =   df['SFR_density']
        #df['NH']            =   10.**df['NH']
        x1                  =   df[x].values
        y1                  =   df[y].values
        x1[x1 <= p.xlim[0]] = p.xlim[0]
        y1[y1 <= p.ylim[0]] = p.ylim[0]
        m_tot               +=   np.sum(df.m.values)
        m_encomp            +=   np.sum(df.m[(x1>=p.xlim[0]) & (y1>=p.ylim[0])].values)
        m_y0                +=   np.sum(df.m[(y1 == 0)].values)
        #print(x,x1.min(),x1.max())
        #print(y,y1.min(),y1.max())
        ys                  =   np.append(ys,y1[(x1>=p.xlim[0]) & (y1>=p.ylim[0])])
        xs                  =   np.append(xs,x1[(x1>=p.xlim[0]) & (y1>=p.ylim[0])])
    print('Min max of %s:' % x)
    print(xs.min(),xs.max())
    print('Min max of %s:' % y)
    print(ys.min(),ys.max())
    fig,ax = plt.subplots(figsize=(10,8))
    hb = ax.hexbin(xs,ys,xscale='log',yscale='log',bins='log',mincnt=1,lw=None,gridsize=50,cmap='inferno')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Number of cells in %i galaxies' % len(rand_gal_index))
    ax.set_xlabel(getlabel(x))
    ax.set_ylabel(getlabel(y))
    print('Total gas mass fraction encompassed: %.4f%%' % (m_encomp/m_tot*100))
    print('Total gas mass fraction with y = 0: %.4f%%' % (m_y0/m_tot*100))
    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)
    # Overplot Cloudy grid params
    print(x,x_cloudy)
    print(y,y_cloudy)
    for x1 in x_cloudy:
        ax.plot([10**x1,10**x1],ax.get_ylim(),'-',color='white',alpha=0.7)
        ax.plot([10**x1,10**x1],ax.get_ylim(),'--k',alpha=0.7)
    for y1 in y_cloudy:
        ax.plot(ax.get_xlim(),[10.**y1,10.**y1],'-',color='white',alpha=0.7)
        ax.plot(ax.get_xlim(),[10.**y1,10.**y1],'--k',alpha=0.7)

    if not os.path.isdir(p.d_plot + 'cell_data/'): os.mkdir(p.d_plot + 'cell_data/')    
    plt.savefig('plots/cell_data/%s%s_cell_params_%s_%s_%s.png' % (p.sim_name,p.sim_run,p.z1,x,y),dpi=250, facecolor='w')

def cloudy_table_scatter(x_index='lognHs',y_index='lognSFRs',**kwargs):
    """ Plot a scatter plot with Cloudy look-up tables.

    Parameters
    ----------

    keep_const : dict
        Dictionary with the cloudy parameter name as key and value to be kept fixed as value.

    line : str
        Line name whos luminosity will be plotted in the z direction.


    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    cloudy_library = clo.library()
    lookup_table = cloudy_library._restore_lookup_table()
    lookup_table['lognSFRs'] = np.round(lookup_table['lognSFRs']*10.)/10.

    fig,ax = plt.subplots(figsize=(8,5))

    key_const1 = list(p.keep_const.keys())[0]
    value_const1 = list(p.keep_const.values())[0]
    print('%s table values:' % key_const1)
    print(np.unique(lookup_table[key_const1]))
    print('kept fixed at %f' % value_const1)
    lookup_table_cut = lookup_table[(lookup_table[key_const1] == value_const1)]

    try: 
        key_const2 = list(p.keep_const.keys())[1]
        value_const2 = list(p.keep_const.values())[1]
        print('%s table values:' % key_const2)
        print(np.unique(lookup_table[key_const2]))
        print('kept fixed at %f' % value_const2)
        lookup_table_cut = lookup_table_cut[(lookup_table_cut[key_const2] == value_const2)]
        print('2 fixed parameters')
    except:
        pass
    try: 
        key_const3 = list(p.keep_const.keys())[2]
        value_const3 = list(p.keep_const.values())[2]
        print('%s table values:' % key_const3)
        print(np.unique(lookup_table[key_const3]))
        print('kept fixed at %f' % value_const3)
        lookup_table_cut = lookup_table_cut[(lookup_table_cut[key_const3] == value_const3)]
        print('3 fixed parameters')
    except:
        pass
    try: 
        key_const4 = list(p.keep_const.keys())[3]
        value_const4 = list(p.keep_const.values())[3]
        print('%s table values:' % key_const4)
        print(np.unique(lookup_table[key_const4]))
        print('kept fixed at %f' % value_const4)
        lookup_table_cut = lookup_table_cut[(lookup_table_cut[key_const4] == value_const4)]
        print('4 fixed parameters')
    except:
        pass

    x, y = lookup_table_cut[x_index].values, lookup_table_cut[y_index].values
    print(x.min(),x.max())

    if p.line == '[CII]158_CO(1-0)':
        line_lum = 10.**lookup_table_cut['[CII]158'].values / 10.**lookup_table_cut['CO(1-0)'].values
        line_lum = np.log10(line_lum)
    if p.line == 'alpha_CO':
        line_lum = 1e4 / aux.Lsun_to_K_km_s_pc2(10.**lookup_table_cut['CO(1-0)'].values,'CO(1-0)') 
    try:
        line_lum = lookup_table_cut[p.line].values
    except:
        pass

    lum = line_lum
    vmin = np.min(lum)
    vmax = np.max(lum)
    if p.ylim:
        vmin = p.ylim[0]
        vmax = p.ylim[1]
    #lum[lum < vmin] = vmin
    #lum[lum > vmax] = vmax
    if p.log: 
        lum = np.log10(lum)
        vmin,vmax = np.log10(vmin),np.log10(vmax)

    print('Highest and lowest value to be mapped:', np.min(lum), np.max(lum))
    print(vmin,vmax)
    print(p.zlim)

    sc = ax.scatter(x, lum, marker='o', c=y, cmap="jet", alpha=0.8)

    translate_labels = {'lognHs':'lnH','logNHs':'lNH','logFUVs':'lG0','logZs':'lZ','lognSFRs':'lSFR_density'}
    plt.colorbar(sc,label=getlabel(translate_labels[y_index]))
    ax.set_xlabel(getlabel(translate_labels[x_index]))
    ax.set_ylabel('\n\n' + p.line)
    plt.tight_layout()
 
    if p.ylim: ax.set_ylim(p.ylim)

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig(p.d_plot + 'look-up/cloudy_table%s_%s_scatter.png' % (p.grid_ext,p.line), format='png', dpi=300)  



def cloudy_table_map(x_index='lognHs',y_index='lognSFRs',**kwargs):
    """ Plot a 2D map in Cloudy look-up tables.


    Parameters
    ----------

    keep_const : dict
        Dictionary with the cloudy parameter name as key and value to be kept fixed as value.

    line : str
        Line name whos luminosity will be plotted in the z direction.


    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    cloudy_library = clo.library()
    lookup_table = cloudy_library._restore_lookup_table()
    print(lookup_table.nH_mw.min())
    print(lookup_table.nH_mw.max())

    fig,ax = plt.subplots(figsize=(8,5))

    key_const1, key_const2, key_const3, key_const4 = list(p.keep_const.keys())[0],list(p.keep_const.keys())[1],list(p.keep_const.keys())[2],list(p.keep_const.keys())[3]
    value_const1, value_const2, value_const3, value_const4 = list(p.keep_const.values())[0],list(p.keep_const.values())[1],list(p.keep_const.values())[2],list(p.keep_const.values())[3]

    # for key, value in p.cloudy_param.items():
    #     key = key
    #     value = value

    # cloudy_parameters = np.array(['logNHs','lognHs','lognSFRs','logZs','logFUVs'])

    # x_index = cloudy_parameters[(cloudy_parameters != key) & (cloudy_parameters != 'Machs')][0]
    # y_index = cloudy_parameters[(cloudy_parameters != key) & (cloudy_parameters != 'Machs')][1]

    print('%s table values:' % key_const1)
    print(np.unique(lookup_table[key_const1]))
    print('kept fixed at %f' % value_const1)

    print('%s table values:' % key_const2)
    print(np.unique(lookup_table[key_const2]))
    print('kept fixed at %f' % value_const2)

    print('%s table values:' % key_const3)
    lookup_table[key_const3] = np.round(lookup_table[key_const3]*10.)/10.
    print(np.unique(lookup_table[key_const3]))
    print('kept fixed at %f' % value_const3)

    print('%s table values:' % key_const4)
    print(np.unique(lookup_table[key_const4]))
    print('kept fixed at %f' % value_const4)

    lookup_table_cut = lookup_table[(lookup_table[key_const1] == value_const1) & \
                            (lookup_table[key_const2] == value_const2) & \
                            (lookup_table[key_const3] == value_const3) & \
                            (lookup_table[key_const4] == value_const4)]
    x, y = lookup_table_cut[x_index].values, lookup_table_cut[y_index].values

    X, Y = np.meshgrid(np.unique(x), np.unique(y))
    print(lookup_table_cut.nH_mw.min())
    print(lookup_table_cut.nH_mw.max())


    if p.line == '[CII]158_CO(1-0)':
        line_lum = 10.**lookup_table_cut['[CII]158'].values / 10.**lookup_table_cut['CO(1-0)'].values
        line_lum = np.log10(line_lum)
    if p.line == 'alpha_CO':
        line_lum = 1e4 / aux.Lsun_to_K_km_s_pc2(10.**lookup_table_cut['CO(1-0)'].values,'CO(1-0)') 
    try:
        line_lum = lookup_table_cut[p.line].values
    except:
        pass

    lum = line_lum.reshape([len(np.unique(x)), len(np.unique(y))]).T

    vmin = np.min(lum)
    vmax = np.max(lum)
    print(vmin,vmax)
    if p.zlim:
        vmin = p.zlim[0]
        vmax = p.zlim[1]
    lum[lum < vmin] = vmin
    lum[lum > vmax] = vmax
    if p.log: 
        print('AAAA')
        lum = np.log10(lum)
        vmin,vmax = np.log10(vmin),np.log10(vmax)

    print('Highest and lowest value to be mapped:', np.min(lum), np.max(lum))
    print(vmin,vmax)

    cf = ax.contourf(X,Y, lum, cmap="jet", vmin=vmin, vmax=vmax, levels=30, lw=0, rstride=1, cstride=1,alpha=0.8)
    if getlabel(p.line) == '':
        if p.log: plt.colorbar(cf,label='log '+p.line)
        if not p.log: plt.colorbar(cf,label=p.line)
    else: 
        plt.colorbar(cf,label=getlabel(p.line))
      
    # Show where grid points are, but only where lum > 0
    failed_models = lookup_table_cut['fail'].values
    ax.plot(x[failed_models == 0],y[failed_models == 0],'x',ms=5,mew=2,color='w')

    translate_labels = {'lognHs':'lnH','logNHs':'lNH','logFUVs':'lG0','logZs':'lZ','lognSFRs':'lSFR_density'}
    ax.set_xlabel(getlabel(translate_labels[x_index]))
    ax.set_ylabel('\n\n' + getlabel(translate_labels[y_index]))
    if p.ylim: ax.set_ylim(p.ylim)
    if p.xlim: ax.set_xlim(p.xlim)
    plt.tight_layout()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig(p.d_plot + 'look-up/cloudy_table%s_%s.png' % (p.grid_ext,p.line), format='png', dpi=300)  

def cloudy_grid_ISM_phases(**kwargs):
    """ Study line emission from different ISM phases in Cloudy grid

    Parameters
    ----------
    line : str
        Name of the line to be investigated.
    ISM_phase : str
        default : 'HII'
    
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    cloudy_library = clo.library()
    print('Grid extension: ',p.grid_ext)
    model_number_matrix,grid_table = cloudy_library._restore_grid_table(grid_ext=p.grid_ext)
    grid_table = grid_table.reset_index(drop=True)
    f = grid_table['f_%s' % p.ISM_phase].values.astype(float)
    line_lum = grid_table[p.line].values.astype(float)
    print('Strange 1-zone models with low f_HII but high [NII]:')
    print(grid_table[['f_H2','f_HI','f_HII','nH','Z',p.line,'Exit code']].iloc[(f < 0.025) & (line_lum > 30000)])
    print(grid_table[['f_H2','f_HI','f_HII','nH',p.line,'Exit code']].iloc[(f < 0.025) & (line_lum > 30000)].nH.min())
    print(len(grid_table[['f_H2','f_HI','f_HII','nH',p.line,'Exit code']].iloc[(f < 0.025) & (line_lum > 30000)]))

    # Now only take part actually used by table
    if 'ext' not in p.grid_ext: grid_table= grid_table[grid_table['nH'] <= 2].reset_index(drop=True)
    if 'ext' in p.grid_ext: grid_table= grid_table[grid_table['nH'] > 2].reset_index(drop=True)

    f = grid_table['f_%s' % p.ISM_phase].values.astype(float)
    line_lum = grid_table[p.line].values.astype(float)
    Z = grid_table['Z'].values.astype(float)
    test = grid_table['f_H2'].values.astype(float) + \
           grid_table['f_HI'].values.astype(float) + \
           grid_table['f_HII'].values.astype(float)
    grid_table = grid_table.iloc[test !=0].reset_index(drop=True)
    f = f[test !=0]
    line_lum = line_lum[test !=0]
    Z = Z[test !=0]

    fig,ax = plt.subplots(figsize=(8,5))
    sc = ax.scatter(f,line_lum,c=Z)
    ax.set_xlabel('%s ISM fraction' % p.ISM_phase); ax.set_ylabel('%s luminosity' % p.line)
    plt.colorbar(sc,label='Z')
    plt.tight_layout()
    plt.savefig(p.d_plot + 'look-up/ISM_phases_%s_grid%s' % (p.line,p.grid_ext),dpi=250)



def cloudy_grid_map(**kwargs):
    """ Plot a 2D contour map of Cloudy grid models.
    
    Parameters
    ----------
    cloudy_param : dict
        Dictionary with {keys, values} where key = cloudy parameter and value is the value it will be fixed at.
        E.g.: cloudy_param={'FUV':2,'NH':19}
    line : str
        Name of the line to be plotted in the z direction.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    cloudy_library = clo.library()
    model_number_matrix,grid_table = cloudy_library._restore_grid_table(grid_ext=p.grid_ext)
    # print(len(grid_table))
    # print(len(grid_table)/len(np.unique(grid_table.nH)))

    grid_table = grid_table.fillna(-10)
    grid_table['DTM'] =  np.round(grid_table['DTM'] * 10.) / 10.
    grid_table['NH'] =  np.round(grid_table['NH'] * 10.) / 10.

    # print(grid_table.DTM[np.isnan(grid_table['DTM'])])
    # print(grid_table.NH[np.isnan(grid_table['NH'])])
    # print(grid_table.FUV[np.isnan(grid_table['FUV'])])
    # print(grid_table.nH[np.isnan(grid_table.nH)])
    # print(grid_table.Z[np.isnan(grid_table.Z)])

    print('nHs: ',np.unique(grid_table.nH))
    print('DTMs: ',np.unique(grid_table.DTM))
    print('FUVs: ',np.unique(grid_table.FUV))
    print('NHs: ',np.unique(grid_table.NH))
    print('Zs: ',np.unique(grid_table.Z))

    fig,ax = plt.subplots(figsize=(8,5))

    key1, key2, key3 = list(p.cloudy_param.keys())[0],list(p.cloudy_param.keys())[1],list(p.cloudy_param.keys())[2]
    value1, value2, value3 = list(p.cloudy_param.values())[0],list(p.cloudy_param.values())[1],list(p.cloudy_param.values())[2]

    # Decide on what goes on x and y axis
    cloudy_parameters = np.array(['NH','FUV','nH','Z','DTM'])
    x_index = cloudy_parameters[(cloudy_parameters != key1) &\
                                (cloudy_parameters != key2) &\
                                (cloudy_parameters != key3)][0]
    y_index = cloudy_parameters[(cloudy_parameters != key1) &\
                                (cloudy_parameters != key2) &\
                                (cloudy_parameters != key3)][1]
    print(x_index,y_index)
    # Cut in grid table
    grid_table_cut = grid_table.iloc[np.where((grid_table[key1].values == value1) & \
                                              (grid_table[key2].values == value2) & \
                                              (grid_table[key3].values == value3))[0]]

    x, y = grid_table_cut[x_index].values, grid_table_cut[y_index].values
    X, Y = np.meshgrid(np.unique(grid_table_cut[x_index].values), np.unique(grid_table_cut[y_index].values))

    # Plot line ratio?
    if '_' in p.line:
        L1 = grid_table_cut[p.line.split('_')[0]].values
        L2 = grid_table_cut[p.line.split('_')[1]].values
        L2[L2 == 0] = 1e9
        line_lum = (L1/L2).astype(float)
        vmin = np.min(np.log10(line_lum[L2 < 1e9]))

    else:
        line_lum = grid_table_cut[p.line].values.astype(float)
        vmin = np.min(np.log10(line_lum[line_lum > 0]))


    # ########## Patching the grid !!
    # line_lum[np.isnan(line_lum)] = -1 # what are these?
    # # 0 values: not sure if we have any?
    # # Negative numbers: missing grid point
    # i_missing = np.where(line_lum <= 0)[0]
    # line_lum[line_lum == 0] = np.min(line_lum[line_lum > 0])
    # while len(i_missing) > 0:
    #     print(i_missing)
    #     lum = np.log10(line_lum)
    #     for i in i_missing:
    #         # print(lum[i-1],lum[i+1])
    #         try: 
    #             lum[i] = (lum[i-1] + lum[i+1])/ 2
    #         except:
    #             pass
    #         # print('he',np.isnan(lum[i]))
    #         if np.isnan(lum[i]):
    #             try:
    #                 lum[i] = lum[i-1]  
    #             except:
    #                 pass
    #         if np.isnan(lum[i]):
    #             try:
    #                 lum[i] = lum[i+1] 
    #             except:
    #                 pass           
    #         line_lum[i] = 10.**lum[i]
    #         # print(i,lum[i])
    #     i_missing = np.where(line_lum < 0)[0]
    # ########## End of patching
    
    lum = np.log10(line_lum)
    lum = lum.reshape([len(np.unique(x)), len(np.unique(y))]).T


    # pdb.set_trace()
    print(p.zlim)
    if p.zlim:
        print(p.zlim)
        lum[lum < p.zlim[0]] = p.zlim[0]
        lum[lum > p.zlim[1]] = p.zlim[1]
        cf = ax.contourf(X,Y, lum, cmap="jet", vmin=p.zlim[0], vmax=p.zlim[1], lw=0, rstride=1, cstride=1,alpha=0.8, levels=20)
    else:
        cf = ax.contourf(X,Y, lum, cmap="jet", vmin=vmin, lw=0, rstride=1, cstride=1,alpha=0.8)
    # print(lum)
    ax.set_xlabel('\n\n' + getlabel('l'+x_index))
    ax.set_ylabel('\n\n' + getlabel('l'+y_index))

    ax.set_xlim([np.min(X),np.max(X)])
    ax.set_ylim([np.min(Y),np.max(Y)])

    plt.colorbar(cf)

    plt.tight_layout()
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig(p.d_plot + 'look-up/cloudy_grid_map_%s%s%s.%s' % (p.line, p.grid_ext, p.ext, p.format), format=p.format, dpi=300)  

def cloudy_grid_surface(**kwargs):
    """ Plot a 3D surface in Cloudy grid models.

    Parameters
    ----------

    cloudy_param : dict
        Dictionary with {keys, values} where key = cloudy parameter and value is the value it will be fixed at.
        E.g.: cloudy_param={'FUV':2,'NH':19}

    line : str
        Name of the line to be plotted in the z direction.


    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    cloudy_library = clo.library()
    model_number_matrix,grid_table = cloudy_library._restore_grid_table(grid_ext=p.grid_ext)

    fig = plt.figure(figsize=(10,7))
    ax = plt.axes(projection='3d')

    key1, key2 = list(p.cloudy_param.keys())[0],list(p.cloudy_param.keys())[1]
    value1, value2 = list(p.cloudy_param.values())[0],list(p.cloudy_param.values())[1]

    # Decide on what goes on x and y axis
    cloudy_parameters = np.array(['NH','FUV','hden','Z'])
    x_index = cloudy_parameters[(cloudy_parameters != key1) &\
                                (cloudy_parameters != key2)][0]
    y_index = cloudy_parameters[(cloudy_parameters != key1) &\
                                (cloudy_parameters != key2)][1]

    # Cut in grid table
    grid_table_cut = grid_table.iloc[np.where((grid_table[key1].values == value1) & \
                                              (grid_table[key2].values == value2))[0]]
    x, y = grid_table_cut[x_index].values, grid_table_cut[y_index].values
    X, Y = np.meshgrid(np.unique(grid_table_cut[x_index].values), np.unique(grid_table_cut[y_index].values))

    # Plot line ratio?
    if '_' in p.line:
        L1 = grid_table_cut[p.line.split('_')[0]].values
        L2 = grid_table_cut[p.line.split('_')[1]].values
        L2[L2 == 0] = 1e9
        line_lum = (L1/L2).astype(float)
        vmin = np.min(np.log10(line_lum[L2 < 1e9]))

    else:
        line_lum = grid_table_cut[p.line].values.astype(float)
        vmin = np.min(np.log10(line_lum[line_lum > 0]))

    lum = np.log10(line_lum)
    lum = lum.reshape([len(np.unique(x)), len(np.unique(y))]).T

    # ########## Patching the grid !!
    # line_lum[np.isnan(line_lum)] = -1 # what are these?
    # # 0 values: not sure if we have any?
    # line_lum[line_lum == 0] = np.min(line_lum[line_lum > 0])
    # # Negative numbers: missing grid point
    # i_missing = np.where(line_lum < 0)[0]
    # while len(i_missing) > 0:
    #     lum = np.log10(line_lum)
    #     for i in i_missing:
    #         # print(lum[i-1],lum[i+1])
    #         try: 
    #             lum[i] = (lum[i-1] + lum[i+1])/ 2
    #         except:
    #             pass
    #         # print('he',np.isnan(lum[i]))
    #         if np.isnan(lum[i]):
    #             try:
    #                 lum[i] = lum[i-1]  
    #             except:
    #                 pass
    #         if np.isnan(lum[i]):
    #             try:
    #                 lum[i] = lum[i+1] 
    #             except:
    #                 pass           
    #         line_lum[i] = 10.**lum[i]
    #         # print(i,lum[i])
    #     i_missing = np.where(line_lum < 0)[0]
    # ########## End of patching


    # pdb.set_trace()
    ax.plot_surface(X, Y, lum, cmap="autumn_r", vmin=vmin, lw=0, rstride=1, cstride=1,alpha=0.8)

    ax.set_xlabel('\n\n' + getlabel('l'+x_index))
    ax.set_ylabel('\n\n' + getlabel('l'+y_index))

    try:
        ax.set_zlabel('\n\n' + getlabel('l%s' % p.line))
    except:
        ax.set_zlabel('\n\n log ' + p.line.replace('_','/'))


    ax.scatter(x[line_lum > 10.**vmin],y[line_lum > 10.**vmin],np.log10(line_lum[line_lum > 10.**vmin]),\
            'o',c=np.log10(line_lum[line_lum > 10.**vmin]),cmap='autumn_r',s=50)

    # print(x)
    # print(line_lum)
    ax.view_init(30, p.angle)

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig(p.d_plot + 'look-up/cloudy_grid_%s.%s' % (p.line, p.format), format=p.format, dpi=300)  
    # pdb.set_trace()

def line_per_nH_bin(lines=[],**kwargs):

    """ Plot histogram distributions of mass fractions of ISM phases per cell
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()
    gal_num             =   GR.get_gal_num(gal_index=p.gal_index)

    # Manually read in cell data
    sim_run = 'simba_100Mpc'
    df1      =   p.d_XL_data + 'data/cell_data/comp/z0.00_%i_%s%s%s.cell_data' % (gal_num,sim_run,'_ext_ism_BPASS','_arepoPDF_M51')
    df2      =   p.d_XL_data + 'data/cell_data/comp/z0.00_%i_%s%s%s.cell_data' % (gal_num,sim_run,'_ext_ism_BPASS','_arepoPDF_CMZ')
    df3      =   p.d_XL_data + 'data/cell_data/comp/z0.00_%i_%s%s%s.cell_data' % (gal_num,sim_run,'_ism_BPASS','_arepoPDF_M51')
    df4      =   p.d_XL_data + 'data/cell_data/comp/z0.00_%i_%s%s%s.cell_data' % (gal_num,sim_run,'_ext_ism_BPASS','_M10')

    #fig,ax = plt.subplots(figsize=(8,6))
    fig, axs = plt.subplots(len(lines), sharex='col',\
                figsize=(6,15),facecolor='w',\
                gridspec_kw={'hspace': 0, 'wspace': 0})

    for i,ax in enumerate(axs):

        print('\n',lines[i])
        #for df_name,ls,col,lab in zip([df1,df2,df3,df4],['-','--','-.',':'],['cyan','blue','brown','orange'],['25Mpc_arepoPDF_M51','25Mpc_arepoPDF_CMZ','25Mpc_arepoPDF_no_ext','25Mpc_M10']):
        for df_name,ls,col,lab in zip([df1,df3],['-','-.'],['cyan','brown'],['100Mpc_arepoPDF_M51','100Mpc_arepoPDF_no_ext']):
            cell_data       =   pd.read_pickle(df_name)
            nH_mw           =   cell_data['nH_mw'].values
            print(nH_mw[nH_mw > 0].min(),nH_mw.max())
            lnH_mw          =   np.log10(nH_mw[nH_mw > 0])
            nH_axis         =   np.linspace(lnH_mw.min(),lnH_mw.max(),40)
            line = cell_data['L_%s' % lines[i]].values
            print('%s: %.4f' % (df_name,np.log10(np.sum(line))))
            dline = np.zeros(len(nH_axis)-1)
            for j in range(len(nH_axis)-1):
                dline[j]           =    np.sum(line[(nH_mw > 10**nH_axis[j]) & (nH_mw < 10**(nH_axis[j+1]))])
         
            ax.plot(nH_axis[0:-1],dline,drawstyle='steps',color=col,ls=ls,label=lab)
            ax.set_ylabel('log ' + getlabel(lines[i]))
            ax.set_yscale('log')
            if lab == '25Mpc_arepoPDF_M51': ax.set_ylim([np.max(dline)/1e8,np.max(dline)*10.])
            #ax.set_xlim([1e-1,1e10])
        if i == 3: 
            ax.set_xlabel('log '+getlabel('nHmw'))
        if 'CO' in lines[i]:
            ax.set_ylim([1e-10,ax.get_ylim()[1]])
    plt.legend()
    plt.tight_layout()
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'cell_data/'): os.mkdir(p.d_plot + 'cell_data/')    
        plt.savefig(p.d_plot + 'cell_data/nH_mw_G%i_%s.png' % (p.gal_index,p.ext), format='png', dpi=300, facecolor='w')  


def ISM_fractions_per_cell(**kwargs):
    """ Plot histogram distriutions of mass fractions of ISM phases per cell
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # Load sim and cell data
    gal_ob          =   gal.galaxy(p.gal_index)
    cell_data       =   gal_ob.cell_data.get_dataframe()

    if p.add:
        ax = plt.gca()
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    x = ax.hist(np.log10(cell_data['mf_H2_grid']),bins=100,alpha=0.6,label='H2',color='r')
    x = ax.hist(np.log10(cell_data['mf_HI_grid']),bins=100,alpha=0.6,label='HI',color='orange')
    x = ax.hist(np.log10(cell_data['mf_HII_grid']),bins=100,alpha=0.4,label='HII',color='b')
    ax.set_xlabel('log Mass fraction per cell')
    ax.set_ylabel('# cells')
    ax.set_yscale('log')
    ax.legend()

    print('Mass fraction of cells with >90\% ionized gas:')
    cell_data1 = cell_data[cell_data['mf_HII_grid'] > 0.9]
    print('%.4f %%' % (100.*np.sum(cell_data1['mf_HII_grid']*cell_data1['m'])/np.sum(cell_data['m'])))

    print('Mass fraction of total ionized gas:')
    print('%.4f %%' % (100.*np.sum(cell_data['mf_HII_grid']*cell_data['m'])/np.sum(cell_data['m'])))

    print('---')

    print('Mass fraction of cells with >90\% atomic gas:')
    cell_data1 = cell_data[cell_data['mf_HI_grid'] > 0.9]
    print('%.4f %%' % (100.*np.sum(cell_data1['mf_HI_grid']*cell_data1['m'])/np.sum(cell_data['m'])))

    print('Mass fraction of total atomic gas:')
    print('%.4f %%' % (100.*np.sum(cell_data['mf_HI_grid']*cell_data['m'])/np.sum(cell_data['m'])))

    print('---')

    print('Mass fraction of cells with >90\% molecular gas:')
    cell_data1 = cell_data[cell_data['mf_H2_grid'] > 0.9]
    print('%.4f %%' % (100.*np.sum(cell_data1['mf_H2_grid']*cell_data1['m'])/np.sum(cell_data['m'])))

    print('Mass fraction of total molecular gas:')
    print('%.4f %%' % (100.*np.sum(cell_data['mf_H2_grid']*cell_data['m'])/np.sum(cell_data['m'])))

    if p.xlim: ax.set_xlim(p.xlim)
    if p.ylim: ax.set_ylim(p.ylim)
    
def cloudy_hii_radial_hyd(props=['ne'],cols=['b'],model_name = '',**kwargs):
    """ Plot radial profile of something in HII region models
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if p.add:
        ax = plt.gca()
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    for prop,col in zip(props,cols):
        hyd = pd.read_table(p.d_cloudy_test + '%s.hyd' % model_name,skiprows=1,\
                names=['depth','Te','nH','ne','HI/H','HII/H','H2/H','H2+/H','H3+/H','H-/H'])
        if '/' in prop:
            try:
                ax2.plot(hyd['depth']/c.pc.to('cm'),hyd[prop],'o',ms=5,color=col)
                ax2.plot(hyd['depth']/c.pc.to('cm'),hyd[prop],'-',lw=1,color=col,label=prop)
            except:
                print('no twinx axis, starting a new one')
                ax2 = ax.twinx()
                ax2.plot(hyd['depth']/c.pc.to('cm'),hyd[prop],'o',ms=5,color=col)
                ax2.plot(hyd['depth']/c.pc.to('cm'),hyd[prop],'-',lw=1,color=col,label=prop)
        else:
            ax.plot(hyd['depth']/c.pc.to('cm'),hyd[prop],'o',ms=5,color=col)
            ax.plot(hyd['depth']/c.pc.to('cm'),hyd[prop],'-',lw=1,color=col,label=prop)

    ax.set_xlabel('Depth [pc]')
    ax.set_ylabel('cm$^{-3}$')
    ax.legend()
    ax2.legend()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'cloudy/'): os.mkdir(p.d_plot + 'cloudy/')    
        plt.savefig(p.d_plot + 'cloudy/%s_hyd.png' % (model_name), format='png', dpi=300, facecolor='w')  

def cloudy_hii_radial_lins(props=['[CII]158'],cols=['b'],ls=['-'],model_name = '',i_model = 0,NIIratio=False,**kwargs):
    """ Plot radial profile of something in HII region models
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if p.add:
        ax = plt.gca()
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    names = ['depth','[CI]610','[CI]370','[CII]158','[OI]63','[OI]145','[OIII]88','[NII]205','[NII]122','CO(1-0)','CO(2-1)','CO(3-2)','CO(4-3)','CO(5-4)']

    for prop,col,ls1 in zip(props,cols,ls):
        lins = pd.read_table(p.d_cloudy_test + '%s.lins' % model_name,skiprows=1,names=names)
        # print(lins['depth'].values)
        # print(lins[prop].values)
        # ax.plot(lins['depth'].values/c.pc.to('cm'),lins[prop].values,'o',ms=5,color=col)
        ax.plot(lins['depth'].values/c.pc.to('cm'),lins[prop].values,'-',lw=2,ls=ls1,color=col,label=prop)

    ax.set_xlabel('Depth [pc]')
    ax.set_ylabel('ergs/s integrated over sphere')
    ax.legend()
    ax.set_yscale('log')

    names = ['it','[CI]610','[CI]370','[CII]158','[OI]63','[OI]145','[OIII]88','[NII]205','[NII]122','CO(1-0)','CO(2-1)','CO(3-2)','CO(4-3)','CO(5-4)']



    if NIIratio:
        ax2 = ax.twinx()
        ax2.plot(lins['depth'].values/c.pc.to('cm'),lins['[NII]122'].values/lins['[NII]205'].values,'o',ms=5,color='k')
        ax2.plot(lins['depth'].values/c.pc.to('cm'),lins['[NII]122'].values/lins['[NII]205'].values,'-',lw=1,ls='-',color='k',label='NII ratio')
        ax2.set_ylim([0,13])
        cloudy_library = clo.library()
        hii_grid = cloudy_library._restore_hii_region_grid()
        print('NII ratio integrated: %.4f' % (10.**hii_grid['[NII]122'][i_model] / 10.**hii_grid['[NII]205'][i_model]))
        print('NII luminosities: ',10.**hii_grid['[NII]122'][i_model],10.**hii_grid['[NII]205'][i_model])

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'cloudy/'): os.mkdir(p.d_plot + 'cloudy/')    
        plt.savefig(p.d_plot + 'cloudy/%s_lins.png' % (model_name), format='png', dpi=300, facecolor='w')  

def cloudy_hii_NII_ratio(**kwargs):
    """ Plot radial profile of something in HII region models
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if p.add:
        ax = plt.gca()
    else:
        fig,ax = plt.subplots(figsize=(10,8))

    cloudy_library = clo.library()
    hii_grid = cloudy_library._restore_hii_region_grid()

    hii_grid['NII_ratio'] = 10.**hii_grid['[NII]122'] / 10.**hii_grid['[NII]205']

    sc = ax.scatter(np.log10(hii_grid['ne']),hii_grid['NII_ratio'],marker='o',c=hii_grid['lognH'])#hii_grid['[NII]122'])#np.log10(ages_grid))
    ax.set_ylim([0.1,10])
    ax.set_xlim([-2,4.2])
    plt.colorbar(sc,label='[NII]122 Luminosity')#log age')
    ax.set_xlabel('log ' + getlabel('ne'))
    # ax.set_ylabel('[NII]122/205')
    ax.set_ylabel('log nH')
    ax.set_yscale('log')
    for axis in [ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    plt.yticks(np.arange(np.ceil(ax.get_ylim()[0]), np.round(ax.get_ylim()[1])+1, 1.0))

    # print('NII ratio integrated: %.4f' % (lin['[NII]122']/lin['[NII]205']))

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'cloudy/'): os.mkdir(p.d_plot + 'cloudy/')    
        plt.savefig(p.d_plot + 'cloudy/hii_region_NII_ratios.png', format='png', dpi=300, facecolor='w')  


#---------------------------------------------------------------------------
### CELL DATA DIAGNOSTIC PLOTS
#---------------------------------------------------------------------------

def cell_data_diag(xval,yval,**kwargs):
    """ Plot one quantity against another in scatter/hexbin plot
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    gal_ob          =   gal.galaxy(p.gal_index)
    cell_data       =   gal_ob.cell_data.get_dataframe()
    cell_data['alpha_CO'] = cell_data.m.values / aux.Lsun_to_K_km_s_pc2(cell_data['L_CO(1-0)'].values,'CO(1-0)') 
    cell_data['m_H2'] = cell_data.m.values * cell_data.mf_H2_grid.values
    if (xval == 'L_CO(1-0)') | (yval == 'L_CO(1-0)'): cell_data['L_CO(1-0)'] = aux.Lsun_to_K_km_s_pc2(cell_data['L_CO(1-0)'].values,'CO(1-0)')
    if p.xlim: cell_data = cell_data[(cell_data[xval] > p.xlim[0]) & (cell_data[xval] < p.xlim[1])]
    if p.ylim: cell_data = cell_data[(cell_data[yval] > p.ylim[0]) & (cell_data[yval] < p.ylim[1])]
    print(xval,cell_data[xval].values.min(),cell_data[xval].values.max())
    print(yval,cell_data[yval].values.min(),cell_data[yval].values.max())
    x = cell_data[xval].values
    y = cell_data[yval].values
    x[x == 0] = np.min(x[x > 0])
    y[y == 0] = np.min(y[y > 0])
    fig,ax = plt.subplots(figsize=(16,7),facecolor='w')
    if p.hexbin:
        hx = ax.hexbin(cell_data[xval].values,cell_data[yval].values,bins='log',cmap='inferno',mincnt=1,gridsize=100)
    else:
        if p.color != 'k':
            sc = ax.scatter(cell_data[xval].values,cell_data[yval].values,c=np.log10(cell_data[p.color]),s=5,alpha=0.6,vmax=p.vmax)
            plt.colorbar(sc,label=p.color)
        else:
            ax.scatter(cell_data[xval].values,cell_data[yval].values,s=5,alpha=0.6)
    ax.set_xlabel(xval)
    ax.set_ylabel(yval)
    xlim = np.array(ax.get_xlim())
    ax.plot(xlim,xlim,'--g',lw=2,label='1-to-1 relation')
    #ax.plot(xlim,xlim*4,'--g',lw=2,label='alpha_C0 = 4')
    #ax.plot(xlim,xlim*0.18,'--r',lw=2,label='alpha_C0 = 0.18')
    ax.legend()
    if p.log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if p.xlim:
        ax.set_xlim(p.xlim)
    if p.ylim:
        ax.set_ylim(p.ylim)

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'cell_data/'): os.mkdir(p.d_plot + 'cell_data/')
        if not p.color: p.color = ''    
        plt.savefig(p.d_plot + 'cell_data/G%i_%s_%s_%s' % (p.gal_index,xval,yval,p.color),dpi=250, facecolor='w')


#---------------------------------------------------------------------------
### LINE LUMINOSITY ###
#---------------------------------------------------------------------------

def compare_runs(names,labnames,**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    d_results = p.d_data + 'results/'

    lines = ['[NII]205','[NII]122','[OIII]88','[CII]158','[OI]63','CO(1-0)','CO(2-1)','CO(3-2)']
    crit = [r'$n_{crit}$ [cm$^{-3}$]:','48','310','510','2,400',r'4.7$\times10^{5}$','650','6,200',r'2.2$\times10^{4}$']


    fig,ax = plt.subplots(figsize=(16,7),facecolor='w')
    barWidth = 1/len(lines)
    colors = ['orange','purple','brown','forestgreen','deepskyblue']
    std_obs = np.zeros(len(lines))
    from sklearn.linear_model import LinearRegression
    for iGR,name in enumerate(names):
        print(name)
        devs_median = []
        devs_mean = []
        GR = pd.read_pickle(d_results + name)
        # if name == 'z0_400gals_abun_abun':
        #     GR = GR[0:100]
        #     for line in lines:
        #         GR = GR.drop('L_' + line + '_sun', axis = 1)
        #     GR.rename(columns={\
        #         'L_NII_205_sun':'L_[NII]205_sun',\
        #         'L_NII_122_sun':'L_[NII]122_sun',\
        #         'L_OIII_88_sun':'L_[OIII]88_sun',\
        #         'L_CII_sun':'L_[CII]158_sun',\
        #         'L_CO10_sun':'L_CO(1-0)_sun',\
        #         'L_CO21_sun':'L_CO(2-1)_sun',\
        #         'L_CO32_sun':'L_CO(3-2)_sun',\
        #         'L_OI_sun':'L_[OI]63_sun'}, inplace = True)
        #     GR.to_pickle(d_results + name)
        #     GR = pd.read_pickle(d_results + name)
        for iline,line in enumerate(lines):
            L_test = GR['L_H2_S(7)_sun'].values
            if iline == 0: print(names[iGR],'L_H2_S(7)_sun > 0 : ',len(L_test[L_test > 0]))        
            L_sim = GR['L_'+line+'_sun'].values
            SFR_sim = GR['SFR'].values
            M_star_sim = GR['M_star'].values
            if p.select == '_MS':
                indices = aux.select_salim18(M_star_sim,SFR_sim)
                L_sim,SFR_sim = L_sim[indices],SFR_sim[indices]
            SFR_sim = SFR_sim[L_sim > 0] 
            L_sim = L_sim[L_sim > 0] 
            L_obs,SFR_obs,fit,std = add_line_SFR_obs(line,[1e6,1e6],ax,plot=False,select=p.select)
            L_obs1 = L_obs[(L_obs > 0) & (SFR_obs > 0)]
            SFR_obs1 = SFR_obs[(L_obs > 0) & (SFR_obs > 0)]
            print('Number of observed galaxies: %i' % (len(L_obs)))
            std_obs[iline] = std


            # Log-linear fit
            # fit = LinearRegression().fit(np.log10(SFR_obs1).reshape(-1, 1),np.log10(L_obs1).reshape(-1, 1))

            # Deviation sim - obs
            # dev = np.mean(np.log10(L_sim) - fit.predict(np.log10(SFR_sim.reshape(-1, 1))))
            devs = np.log10(L_sim) - fit.predict(np.log10(SFR_sim.reshape(-1, 1))).flatten()
            dev = np.median(np.log10(L_sim) - fit.predict(np.log10(SFR_sim.reshape(-1, 1))))
            dev_spread = np.quantile(devs, 0.75) - np.quantile(devs, 0.25)
            devs_median.append(np.median(devs))
            devs_mean.append(np.mean(devs))
          
            if line == '[OI]63':
                if iGR == 1:
                    devs1 = devs
                if iGR == 3:
                    pass
                    # print('Mean reduction in [OI]63: ',np.mean(devs1-devs))
            # Add as a bar
            # ax.bar(iline + iGR*barWidth, dev, color=colors[iGR], width=barWidth, edgecolor='white',\
                   # alpha=0.7)
            
            # Add as a box plot
            data = np.concatenate(([dev_spread], [dev])) 
            # print(fit.predict(np.log10(SFR_sim.reshape(-1, 1))).flatten().shape)
            # ax.boxplot(data,positions=[iline + iGR*barWidth],widths=[barWidth]) #color=colors[iGR],alpha=0.7,
            bplot = ax.boxplot([devs],patch_artist=True,whis=1.5,positions=[iline + iGR*barWidth],widths=[barWidth]) #color=colors[iGR],alpha=0.7,

            # print(bplot['boxes'])
            bplot['boxes'][0].set_facecolor(colors[iGR])
            # for patch, color in zip(bplot['boxes'], colors):
            #         patch.set_facecolor(colors[iGR])
            if iline == 0:
                ax.bar(iline + -100*barWidth, dev, color=colors[iGR], width=barWidth, edgecolor='white',\
                   alpha=0.7,label=labnames[names[iGR]])
        print('\n %s, median devs:' % name)
        for iline,line in enumerate(lines):
            print(line,' median dev: %.3f' % (devs_median[iline]))
        print('\n %s, mean devs:' % name)
        for iline,line in enumerate(lines):
            print(line,' mean dev: %.3f' % (devs_mean[iline]))
  

    # ax.plot(np.arange(len(lines)),-1.*std_obs,'-',color='grey',lw=3,alpha=0.6,\
    #         label='1-$\sigma$ spread in observed relation')
    # ax.plot(np.arange(len(lines)),std_obs,'-',color='grey',lw=3,alpha=0.6)
    #xarray = np.arange(len(lines))*1.00000
    #xarray[0] = xarray[0] - 0.5
    #xarray[-1] = xarray[-1] + 0.5
    #ax.fill_between(xarray,-1.*std_obs,std_obs,color='grey',alpha=0.8)
    #ax.fill_between(xarray,-2.*std_obs,2*std_obs,color='grey',alpha=0.4)
    for iline in range(len(lines)):
        print(iline,barWidth)
        ax.bar(iline+2*barWidth,4.*std_obs[iline],width=0.8*len(lines)*barWidth,bottom=-2.*std_obs[iline],color='grey',alpha=0.4)
        ax.bar(iline+2*barWidth,2.*std_obs[iline],width=0.8*len(lines)*barWidth,bottom=-1.*std_obs[iline],color='grey',alpha=0.8)


    ax.set_xlim([-1,len(lines)+1])
    ax.plot(ax.get_xlim(),[0,0],'--k')
    ax.plot(ax.get_xlim(),[1,1],':k',lw=1)
    ax.plot(ax.get_xlim(),[-1,-1],':k',lw=1)
    ax.set_ylabel('$\Delta L_{x}$ [dex]')
    ax.set_xticks(np.arange(len(lines)))
    ax.set_xticklabels(lines)
    ax.legend(loc='lower right')

    ax2 = ax.twiny()
    ax2.set_xlim([-1,len(lines)+1])
    ax2.set_xticks(np.arange(len(lines)+1)-1)
    ax2.set_xticklabels(crit)
    ax2.set_ylim([-7.5,4.5])
 
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/std_runs%s' % p.select,dpi=250, facecolor='w')

def compare_CII_w_models(**kwargs):
    """ Plot line - SFR relation with other models
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if not p.xlim: p.xlim = [-4,2]
    if not p.ylim: p.ylim = [4,10]

    fig,ax = plt.subplots(figsize=(8,6))
    ax.set_ylim(p.ylim)
    ax.set_xlim(p.xlim)

    # SIGAME Simba-100 ext ON (default run)
    GR                  =   glo.global_results(sim_run=p.sim_runs[1],nGal=p.nGals[1],grid_ext=p.grid_exts[1])
    L_line              =   getattr(GR,'L_[CII]158_sun')
    SFR                 =   getattr(GR,'SFR')
    # Phantom points to close contour lines...
    L_line              =   np.append(L_line,np.array([1e8,10**9.3]))
    SFR                 =   np.append(SFR,np.array([0.1,10**0.85]))
    lL_line             =   np.log10(L_line)
    lSFR                =   np.log10(SFR)
    lSFR                =   lSFR[L_line > 0]
    lL_line             =   lL_line[L_line > 0]
    # ax.plot(np.log10(SFR),np.log10(L_line),'o',ms=4,color='midnightblue',\
    #     alpha=0.7,label='SIGAME with Simba-100',zorder=10)
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 100
    k = kde.gaussian_kde(np.column_stack([lSFR,lL_line]).T)
    x, y = np.mgrid[lSFR.min():lSFR.max():nbins*1j,4:lL_line.max():nbins*1j]
    z = k(np.vstack([x.flatten(), y.flatten()]))
    CS = ax.contour(x, y, z.reshape(x.shape),colors='forestgreen',levels=6,zorder=10)
    CS.collections[0].set_label('SIGAME 100Mpc_arepoPDF')

    # Select only MS galaxies
    L_line              =   getattr(GR,'L_[CII]158_sun')
    indices = aux.select_salim18(GR.M_star[L_line > 0],GR.SFR[L_line > 0])
    print('With MS selection criteria: only %i galaxies' % (len(L_line)))
    lSFR = lSFR[indices]
    lL_line = lL_line[indices]
    lSFR                =   np.append(lSFR,np.array([-1,0.85,1.5]))
    lL_line             =   np.append(lL_line,np.array([8,9.3,8.4]))
    nbins = 100
    k = kde.gaussian_kde(np.column_stack([lSFR,lL_line]).T)
    x, y = np.mgrid[lSFR.min():lSFR.max():nbins*1j,4:lL_line.max():nbins*1j]
    z = k(np.vstack([x.flatten(), y.flatten()]))
    CS = ax.contour(x, y, z.reshape(x.shape),colors='purple',linestyles='dashed',levels=6,zorder=10)
    CS.collections[0].set_label('SIGAME 100Mpc_arepoPDF (MS)')


    # SIGAME Simba-100 ext OFF
    # GR                  =   glo.global_results(sim_run=p.sim_runs[1],nGal=p.nGals[1],grid_ext=p.grid_exts[0])
    # L_line              =   getattr(GR,'L_[CII]158_sun')
    # SFR                 =   getattr(GR,'SFR')
    # # Phantom points to close contour lines...
    # L_line              =   np.append(L_line,np.array([10.**9.2,1e8]))
    # SFR                 =   np.append(SFR,np.array([0.9,0.1]))
    # lL_line             =   np.log10(L_line)
    # lSFR                =   np.log10(SFR)
    # lSFR                =   lSFR[L_line > 0]
    # lL_line             =   lL_line[L_line > 0]
    # # ax.plot(np.log10(SFR),np.log10(L_line),'o',ms=4,color='midnightblue',\
    # #     alpha=0.7,label='SIGAME with Simba-100',zorder=10)
    # # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    # nbins = 100
    # k = kde.gaussian_kde(np.column_stack([lSFR,\
    #                                      lL_line]).T)

    # x, y = np.mgrid[lSFR.min():lSFR.max():nbins*1j, \
    #            4:lL_line.max():nbins*1j]
    # z = k(np.vstack([x.flatten(), y.flatten()]))
    # CS = ax.contour(x, y, z.reshape(x.shape),colors='brown',levels=6,linestyles='dotted')
    # CS.collections[0].set_label('SIGAME 100Mpc_arepoPDF_no_ext')

    # SIGAME Simba-25
    # GR                  =   glo.global_results(sim_run=p.sim_runs[0],nGal=p.nGals[0])
    # L_line              =   getattr(GR,'L_[CII]158_sun')
    # SFR                 =   getattr(GR,'SFR')
    # ax.plot(np.log10(SFR),np.log10(L_line),'^',ms=6,color='darkorchid',alpha=0.7,label='SIGAME with Simba-25')

    # Observations in background
    add_line_SFR_obs('[CII]158',L_line,ax,plot_fit=False)

    # Popping 2019
    G19 = pd.read_csv(p.d_data + 'models/Popping2019.csv',skiprows=1,sep=' ',\
        names=['logSFR', 'logLCII', 'log LCII 14th percentile', 'log LCII 86th percentile'])
    ax.plot(G19.logSFR,G19.logLCII,'k-',label='Popping 2019',alpha=0.8)
    ax.fill_between(G19.logSFR,G19['log LCII 14th percentile'].values,\
                               G19['log LCII 86th percentile'].values,color='grey',alpha=0.4)


    # Padilla 2020
    P20 = pd.read_csv(p.d_data + 'models/DataFig6.csv',skiprows=1,sep=',',\
        names=['IDSim','GalID','SFR','LCIITotal'])
    P20['logLCII'] = np.log10(P20['LCIITotal'])
    P20['logSFR'] = np.log10(P20['SFR'])


    colors = ['cyan','orange','midnightblue']
    nbins = 100
    IDSims = ['Ref25','Recal25','Ref100']
    for i,IDSim in enumerate(IDSims):
        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        k = kde.gaussian_kde(np.column_stack([P20.logSFR.values[P20.IDSim == IDSim],\
                                             P20.logLCII.values[P20.IDSim == IDSim]]).T)

        xP20, yP20 = np.mgrid[P20.logSFR.min():P20.logSFR.max():nbins*1j, \
                   4:P20.logLCII.max():nbins*1j]
        zP20 = k(np.vstack([xP20.flatten(), yP20.flatten()]))
        # To remove weird contour line:
        zP20.reshape(xP20.shape)[(xP20 > -1) & (yP20 < 5.5)] = 1e-5
        zP20.reshape(xP20.shape)[(xP20 < -3)] = 1e-5
        zP20.reshape(xP20.shape)[(xP20 < -2) & (yP20 > 8)] = 1e-5
        zP20.reshape(xP20.shape)[(xP20 > 1) & (yP20 < 7)] = 1e-5
        CS = ax.contour(xP20, yP20, zP20.reshape(xP20.shape),colors=colors[i],levels=5)
        CS.collections[0].set_label('Ramos Padilla 2020 '+IDSim)


    ax.set_xlabel('log '+getlabel('SFR'))
    ax.set_ylabel('log '+getlabel('[CII]158'))
    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[_] for _ in [8,9,10,11,12,7,0,1,2,3,4,5,6]]
    labels = [labels[_] for _ in [8,9,10,11,12,7,0,1,2,3,4,5,6]]
    plt.legend(handles,labels,fontsize=9,loc='upper left')
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/CII_SFR_w_models.png', format='png', dpi=300)  

def resolution_test(names,**kwargs):
    """ Find and compare similar galaxies in Simba-25 and Simba-100
    by selecting closest pairs in M_star, M_gas, SFR, and Z
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    d_results = p.d_data + 'results/'

    lines = ['[NII]205','[NII]122','[OIII]88','[CII]158','[OI]63','CO(1-0)','CO(2-1)','CO(3-2)']

    # Select parameters
    GR1 = pd.read_pickle(d_results + names[0])
    GR2 = pd.read_pickle(d_results + names[1])
    GR2 = GR2[GR2['L_[NII]205_sun'] * GR2['L_[NII]122_sun'] * GR2['L_[OIII]88_sun'] * GR2['L_[CII]158_sun'] * \
            GR2['L_[OI]63_sun'] * GR2['L_CO(1-0)_sun'] * \
            GR2['L_CO(2-1)_sun'] * GR2['L_CO(3-2)_sun'] != 0].reset_index(drop=True)
    
    # What is the lowest M_star and M_gas for Simba-100 with 1000 gas particles?
    print(GR1['N_gas'].min())
    print(GR1['N_gas'].max())

    M_star_min100 = np.min(GR1['M_star'][GR1['N_gas'] >= 500])
    M_gas_min100 = np.min(GR1['M_gas'][GR1['N_gas'] >= 500])
    M_star_min25 = np.min(GR2['M_star'][GR2['N_gas'] >= 500])
    M_gas_min25 = np.min(GR2['M_gas'][GR2['N_gas'] >= 500])
    print(GR1['M_gas'].min())
    print(GR2['M_gas'].min())

    props = ['M_star','M_gas','SFR','Zsfr']
    x1 = np.log10(np.column_stack([GR1.M_star,GR1.M_gas,GR1.SFR,GR1.Zsfr]))
    x2 = np.log10(np.column_stack([GR2.M_star,GR2.M_gas,GR2.SFR,GR2.Zsfr]))
    n_param = len(props)

    # Calculate differences:
    _ = 0
    xdiff_raw = np.zeros([len(GR1)*len(GR2),n_param])

    for i in range(len(GR1)):
        for j in range(len(GR2)):
            xdiff_raw[_,:] = x1[i,:] - x2[j,:]
            _ += 1

    # Store values in same shape
    xdiff_GR1_values = np.zeros([len(GR1),len(GR2),n_param])
    for iprop in range(n_param):
        for i in range(len(GR1)):
            # print(len((x1[i,iprop]*len(GR2)).flatten()))
            # print(xdiff_GR1_values[i,:,iprop])
            xdiff_GR1_values[i,:,iprop] = np.zeros(len(GR2)) + x1[i,iprop]
    xdiff_GR2_values = np.zeros([len(GR1),len(GR2),n_param])
    for iprop in range(n_param):
        for i in range(len(GR2)):
            xdiff_GR2_values[:,i,iprop] = np.zeros(len(GR1)) + x2[i,iprop]

    # Normalize parameters
    xdiff = xdiff_raw*0.
    for i in range(n_param):
        xdiff[:,i] = xdiff_raw[:,i] / (np.max(xdiff_raw[:,i]) - np.min(xdiff_raw[:,i]))

    # Shortest distance in 3D
    xdist = np.sqrt(np.sum(xdiff * xdiff, axis=1)).reshape([len(GR1),len(GR2)])
    best_fit_in_GR2 = np.zeros(len(GR1)).astype(int)
    min_xdist = np.zeros(len(GR1))
    for i in range(len(GR1)):
        # j = np.argmin(xdist[i,:])
        best_fit_in_GR2[i] = np.argmin(xdist[i,:])
        min_xdist[i] = np.min(xdist[i,:])

    # Look at line luminoisities
    cmap = plt.get_cmap('gist_rainbow_r')
    cmap = plt.get_cmap('brg')
    colors = [cmap(i) for i in np.linspace(0, 1, len(lines))]
    fig, axs = plt.subplots(nrows=1, ncols=n_param, \
        figsize=(22,5),facecolor='w',\
        gridspec_kw={'hspace': 0, 'wspace': 0.35})
    for iprop,prop in enumerate(props):
        print(prop)
        grid_points = np.append(x1[:,iprop],x2[:,iprop])
        grid_points = np.linspace(np.min(grid_points),np.max(grid_points),10)
        grid_points_c = grid_points[1::] - (grid_points[1]-grid_points[0])/2.
        i_GR1 = np.zeros(len(grid_points_c)).astype(int)-1
        i_GR2 = np.zeros(len(grid_points_c)).astype(int)-1
        spread = np.zeros([len(grid_points_c),len(lines)])
        minspread = np.zeros([len(grid_points_c),len(lines)])
        maxspread = np.zeros([len(grid_points_c),len(lines)])
        for i in range(len(grid_points_c)):
            min_xdist_cut = min_xdist[(x1[:,iprop] >= grid_points[i]) & (x1[:,iprop] <= grid_points[i+1])]
            indices = np.where([(x1[:,iprop] >= grid_points[i]) & (x1[:,iprop] <= grid_points[i+1])])[1]
            # print(grid_points[i],grid_points[i+1],x1.min(),x1.max(),indices)
            # print(grid_points_c[i],len(indices))
            print(len(min_xdist_cut))
            try:
                i_GR1[i] = indices[int(np.argmin(min_xdist_cut))]
                i_GR2[i] = best_fit_in_GR2[i_GR1[i]]
                # Spread
                for iline,line in enumerate(lines):
                    i_GR1s = indices#[np.where((min_xdist_cut > np.quantile(min_xdist_cut,0.25)) & \
                        #(min_xdist_cut < np.quantile(min_xdist_cut,0.75)))]
                    i_GR2s = best_fit_in_GR2[i_GR1s]
                    # minspread[i,iline] = np.min(np.log10(GR1['L_'+line+'_sun'].values[i_GR1s]) - \
                    #     np.log10(GR2['L_'+line+'_sun'].values[i_GR2s]))
                    # maxspread[i,iline] = np.max(np.log10(GR1['L_'+line+'_sun'].values[i_GR1s]) - \
                    #     np.log10(GR2['L_'+line+'_sun'].values[i_GR2s]))
                    minspread[i,iline] = np.quantile(np.log10(GR1['L_'+line+'_sun'].values[i_GR1s]) - \
                        np.log10(GR2['L_'+line+'_sun'].values[i_GR2s]),0.25)
                    maxspread[i,iline] = np.quantile(np.log10(GR1['L_'+line+'_sun'].values[i_GR1s]) - \
                        np.log10(GR2['L_'+line+'_sun'].values[i_GR2s]),0.75)
                    spread[i,iline] = 1.*np.std(np.log10(GR1['L_'+line+'_sun'].values[i_GR1s]) - \
                        np.log10(GR2['L_'+line+'_sun'].values[i_GR2s]))
            except:
                pass
        i_GR1_cut = i_GR1[(i_GR1 > -1) & (i_GR2 > -1)]
        i_GR2_cut = i_GR2[(i_GR1 > -1) & (i_GR2 > -1)]
        spread_cut = spread[(i_GR1 > -1) & (i_GR2 > -1)]
        minspread_cut = minspread[(i_GR1 > -1) & (i_GR2 > -1)]
        maxspread_cut = maxspread[(i_GR1 > -1) & (i_GR2 > -1)]

        grid_points_c = grid_points_c[(i_GR1 > -1) & (i_GR2 > -1)]

        ax = axs[iprop]
        ax.set_ylim([-1.5,1.5])
        if iprop == 0:
            ax.set_ylim([-2.5,1.5])
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.plot([np.min(grid_points_c),np.max(grid_points_c)],[0,0],'--k',lw=1.5)
        for iline,line in enumerate(lines):
            delta_dex = np.log10(GR1['L_'+line+'_sun'].values[i_GR1_cut]) - np.log10(GR2['L_'+line+'_sun'].values[i_GR2_cut])
            ax.plot(grid_points_c,delta_dex,marker='o',ms=5,lw=2,color=colors[iline],label=line,zorder=10)
            ax.fill_between(grid_points_c,minspread_cut[:,iline],maxspread_cut[:,iline],\
                color=colors[iline],alpha=0.3)

        if prop == 'M_gas':
            ax.plot(2*[np.log10(M_gas_min100)],ax.get_ylim(),'--k',lw=1.5)
            ax.text(1.005*np.log10(M_gas_min100),ax.get_ylim()[1]-0.3,'min(M$_{gas}$[N$_{gas}>500$]\nin Simba-100)',fontsize=10)

        ax.set_ylabel(r'$\Delta$ luminosity [dex]')# + getlabel(line))
        if prop == 'M_star':
            ax.legend(fontsize=10)
        ax.set_xlabel('log '+getlabel(prop))

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/compare_Simba100_w_Simba25.png', format='png', dpi=300)  

def compare_SIGAME_runs(names,**kwargs):
    """ Compare SIGAME runs, similar to resolution_test() above.
    in terms of line luminosities, M_star, M_gas, SFR, and Z
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    d_results = p.d_data + 'results/'

    lines = ['[NII]205','[NII]122','[OIII]88','[CII]158','[OI]63','CO(1-0)','CO(2-1)','CO(3-2)']

    # Select parameters
    GR1 = pd.read_pickle(d_results + names[0])
    GR2 = pd.read_pickle(d_results + names[1])
    
    # Quick check as function of Z
    fig, ax = plt.subplots(facecolor='w',figsize=(8,6))
    Zsfr = GR1['Zsfr'].values
    dex = np.log10(GR1['L_[NII]122_sun'].values) - np.log10(GR2['L_[NII]122_sun'].values)
    ax.plot(Zsfr,dex,'o')
    plt.savefig(p.d_plot+'luminosity/test_comp_Z',dpi=250)
    GR1['dex'] = dex
    print(GR1.sort_values('Zsfr')[['dex','gal_num','Zsfr']])

    props = ['M_star','M_gas','SFR','Zsfr']
    x1 = np.log10(np.column_stack([GR1.M_star,GR1.M_gas,GR1.SFR,GR1.Zsfr]))
    x2 = np.log10(np.column_stack([GR2.M_star,GR2.M_gas,GR2.SFR,GR2.Zsfr]))
    n_param = len(props)

    # Set up plot
    cmap = plt.get_cmap('gist_rainbow_r')
    cmap = plt.get_cmap('brg')
    colors = [cmap(i) for i in np.linspace(0, 1, len(lines))]
    fig, axs = plt.subplots(nrows=1, ncols=n_param, \
        figsize=(22,5),facecolor='w',\
        gridspec_kw={'hspace': 0, 'wspace': 0.35,'left':0.05,'right':0.95})

    bins = 8

    # For bins in props, calcualte mean and std of difference between the two model results
    for iprop,prop in enumerate(props):
        ax = axs[iprop]
        data = np.append(GR1[prop].values,GR2[prop].values)
        grid_points = np.linspace(np.log10(np.min(data)),np.log10(np.max(data)),bins)
        if prop == 'M_star': grid_points = np.linspace(np.log10(np.min(data)),np.log10(3.5e11),bins)
        if prop == 'SFR': grid_points = np.linspace(np.log10(np.min(data)),np.log10(30),bins)
        grid_points_c = grid_points[1::] - (grid_points[1]-grid_points[0])/2.
        ax.plot([np.min(grid_points_c),np.max(grid_points_c)],[0,0],'--k',lw=1.5)
        for iline,line in enumerate(lines):
            meandex = np.zeros([len(grid_points_c)])
            mediandex = np.zeros([len(grid_points_c)])
            mindex = np.zeros([len(grid_points_c)])
            maxdex = np.zeros([len(grid_points_c)])
            mindex1 = np.zeros([len(grid_points_c)])
            maxdex1 = np.zeros([len(grid_points_c)])
            line_GR1 = GR1['L_'+line+'_sun'].values
            line_GR2 = GR2['L_'+line+'_sun'].values
            GR1cut = GR1[(line_GR1 > 0) & (line_GR2 > 0)].reset_index(drop=True)
            GR2cut = GR2[(line_GR1 > 0) & (line_GR2 > 0)].reset_index(drop=True)
            line_GR1 = GR1cut['L_'+line+'_sun'].values
            line_GR2 = GR2cut['L_'+line+'_sun'].values
            devs = np.log10(line_GR1) - np.log10(line_GR2)
            for i in range(len(grid_points_c)):
                select = np.argwhere((GR1cut[prop].values >= 10.**grid_points[i]) & (GR1cut[prop].values < 10.**grid_points[i+1]))
                if iline == 0: print(len(select))
                if len(select) >= 1:
                    meandex[i] = np.mean(devs[select])
                    mediandex[i] = np.median(devs[select])
                    mindex[i] = np.mean(devs[select]) - np.std(devs[select]) 
                    maxdex[i] = np.mean(devs[select]) + np.std(devs[select]) 
                    mindex1[i] = np.quantile(devs[select],0.25)
                    maxdex1[i] = np.quantile(devs[select],0.75)
            grid_points_plot = grid_points_c[meandex != 0]
            mindex = mindex[meandex != 0]
            mindex1 = mindex1[meandex != 0]
            maxdex = maxdex[meandex != 0]
            maxdex1 = maxdex1[meandex != 0]
            mediandex = mediandex[meandex != 0]
            meandex = meandex[meandex != 0]
            ax.plot(grid_points_plot,mediandex,marker='o',ms=5,lw=2,color=colors[iline],label=line,zorder=10)
            ax.fill_between(grid_points_plot,mindex1,maxdex1,\
                color=colors[iline],alpha=0.3)
            if iline == 0: 
                global_ymin = np.min(mindex)
            else:
                if np.min(mindex) < global_ymin: global_ymin = np.min(mindex)

        ax.set_ylim([-1.5,1.5])
        if p.ylim: ax.set_ylim([global_ymin-0.2,p.ylim[1]])

        ax.set_ylabel(r'Median $\Delta$ luminosity [dex]')# + getlabel(line))
        if prop == 'M_star':
            ax.set_xlim([ax.get_xlim()[0],ax.get_xlim()[1]+0.3])
            ax.legend(fontsize=9,loc='upper right')
        ax.set_xlabel('log '+getlabel(prop))

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/compare_%s_w_%s.png' % (names[0],names[1]), format='png', dpi=300) # .eps for paper!

def ISM_lums(line,**kwargs):
    """A comparison of contributions to line luminosity from different ISM phases.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    L_tot = getattr(GR,'L_%s_sun' % line)
    L_H2 = getattr(GR,'L_%s_H2_sun' % line)
    L_HI = getattr(GR,'L_%s_HI_sun' % line)
    L_HII = getattr(GR,'L_%s_HII_sun' % line)
    fig, ax                =   plt.subplots(figsize=(8, 6  ),facecolor='w')
    x=ax.hist(np.log10(L_tot[L_tot > 0]),bins=100,alpha=0.6,label='tot',color='grey')
    x=ax.hist(np.log10(L_H2[L_H2 > 0]),bins=100,alpha=0.6,label='H2',color='r')
    x=ax.hist(np.log10(L_HI[L_HI > 0]),bins=100,alpha=0.8,label='HI',color='orange')
    x=ax.hist(np.log10(L_HII[L_HII > 0]),bins=100,alpha=0.5,label='HII',color='b')
    ax.set_xlabel(getlabel('l'+line))
    ax.set_ylabel('Number of galaxies')
    plt.legend()
    if p.savefig:
        plt.savefig(p.d_plot + 'luminosity/ISM_lums_%s.png' % line)

def ISM_effs(line,**kwargs):
    """A comparison of line luminosity efficiencies for different ISM phases.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    L_tot = getattr(GR,'L_%s_sun' % line)
    L_H2 = getattr(GR,'L_%s_H2_sun' % line)
    L_HI = getattr(GR,'L_%s_HI_sun' % line)
    L_HII = getattr(GR,'L_%s_HII_sun' % line)
    M_HII = getattr(GR,'M_HII')
    M_HI = getattr(GR,'M_HI')
    M_H2 = getattr(GR,'M_H2')
    M_gas = getattr(GR,'M_gas')

    fig, ax                =   plt.subplots(figsize=(8, 6  ),facecolor='w')
    x=ax.hist(np.log10(L_tot[L_tot > 0]/M_gas[L_tot > 0]),bins=100,alpha=0.6,label='tot',color='grey')
    x=ax.hist(np.log10(L_H2[L_H2 > 0]/M_H2[L_H2 > 0]),bins=100,alpha=0.6,label='H2',color='r')
    x=ax.hist(np.log10(L_HI[L_HI > 0]/M_HI[L_HI > 0]),bins=100,alpha=0.8,label='HI',color='orange')
    x=ax.hist(np.log10(L_HII[L_HII > 0]/M_HII[L_HII > 0]),bins=100,alpha=0.5,label='HII',color='b')
    ax.set_xlabel('log efficiency of %s [L$_{\odot}$/M$_{\odot}$]' % line)
    ax.set_ylabel('Number of galaxies')
    plt.legend()
    if p.savefig:
        plt.savefig(p.d_plot + 'luminosity/ISM_effs_%s.png' % line)

def map_line(**kwargs):
    """ Map surface brightness of one line.

    Parameters
    ----------

    line : str
        Line name whos luminosity will be plotted in the z direction.

    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

def line_SFR_array(lines,**kwargs):
    """ Plot line luminosity (in Lsun) against SFR for a selection of lines, 
    in subplots with common x axis
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    fig, axs = plt.subplots(len(lines), sharex='col',\
                figsize=(6,15),facecolor='w',\
                gridspec_kw={'hspace': 0, 'wspace': 0})

    for i,ax in enumerate(axs):

        #line_SFR(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[0],nGal=p.nGals[0],add_obs=p.add_obs,MS=p.MS,add=True,cb=True)
        line_SFR(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[1],nGal=p.nGals[1],add_obs=p.add_obs,MS=p.MS,add=True,cb=True)
        #line_SFR(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[0],nGal=p.nGals[0],add_obs=False,add=True,cb=False)

        # Only 1 galaxy
        #line_SFR(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[0],nGal=p.nGals[0],add_obs=True,add=True,cb=False)


    plt.tight_layout()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/lines_SFR_array_%s%s%s_%s%s_%s.png' % (p.ext,p.grid_ext,p.table_ext,p.sim_name,p.sim_run,p.select), format='png', dpi=300)  

def dline_dSFR_array(lines,**kwargs):
    """ Deviation from observed line-SFR relation vs distance from MS
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    fig, axs = plt.subplots(len(lines), sharex='col',\
                figsize=(6,15),facecolor='w',\
                gridspec_kw={'hspace': 0, 'wspace': 0})

    for i,ax in enumerate(axs):

        dline_dSFR(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[1],nGal=p.nGals[1],add_obs=p.add_obs,MS=p.MS,add=True,cb=True)
        dline_dSFR(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[0],nGal=p.nGals[0],add_obs=False,add=True,cb=False)

    plt.tight_layout()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/dlines_dSFR_array_%s%s%s_%s%s_%s.png' % (p.ext,p.grid_ext,p.table_ext,p.sim_name,p.sim_run,p.select), format='png', dpi=300)  

def dline_Mgas_array(lines,**kwargs):
    """ Deviation from observed line-SFR relation vs gas mass
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    fig, axs = plt.subplots(len(lines), sharex='col',\
                figsize=(6,15),facecolor='w',\
                gridspec_kw={'hspace': 0, 'wspace': 0})

    for i,ax in enumerate(axs):

        dline_Mgas(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[1],nGal=p.nGals[1],add=True,cb=True)
        dline_Mgas(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[0],nGal=p.nGals[0],add=True,cb=False)

    plt.tight_layout()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/dlines_Mgas_array_%s%s%s_%s%s_%s.png' % (p.ext,p.grid_ext,p.table_ext,p.sim_name,p.sim_run,p.select), format='png', dpi=300)  

def dline_tdepl_array(lines,**kwargs):
    """ Deviation from observed line-SFR relation vs gas depletion time scale
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    fig, axs = plt.subplots(len(lines), sharex='col',\
                figsize=(6,15),facecolor='w',\
                gridspec_kw={'hspace': 0, 'wspace': 0})

    for i,ax in enumerate(axs):

        dline_tdepl(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[1],nGal=p.nGals[1],add=True,cb=True)
        dline_tdepl(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[0],nGal=p.nGals[0],add=True,cb=False)

    plt.tight_layout()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/dlines_tdepl_array_%s%s%s_%s%s_%s.png' % (p.ext,p.grid_ext,p.table_ext,p.sim_name,p.sim_run,p.select), format='png', dpi=300)  


def line_FIR_array(lines,**kwargs):
    """ Plot line luminosity (in Lsun) against FIR luminosity for a selection of lines, 
    in subplots with common x axis
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    fig, axs = plt.subplots(len(lines), sharex='col',\
                figsize=(6,15),\
                gridspec_kw={'hspace': 0, 'wspace': 0})

    for i,ax in enumerate(axs):

        line_FIR(line=lines[i],ax=ax,add=True)

    plt.tight_layout()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/lines_FIR_array_%s%s%s_%s_%s.png' % (p.ext,p.grid_ext,p.table_ext,p.sim_name,p.sim_run), format='png', dpi=300)  

def line_sSFR_array(lines,**kwargs):
    """ Plot line luminosity (in Lsun) against sSFR for a selection of lines, 
    in subplots with common x axis
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    fig, axs = plt.subplots(len(lines), sharex='col',\
                figsize=(6,15),\
                gridspec_kw={'hspace': 0, 'wspace': 0})

    for i,ax in enumerate(axs):

        line_sSFR(line=lines[i],ax=ax,select=p.select,add=True)

    plt.tight_layout()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/lines_sSFR_array_%s%s%s_%s%s_%s.png' % (p.ext,p.grid_ext,p.table_ext,p.sim_name,p.sim_run,p.select), format='png', dpi=300)  

def line_SFR(**kwargs):
    """ Plot line luminosity (in Lsun) against SFR
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if p.line == 'CO(2-1)': p.select = 'Zsfr'

    GR                  =   glo.global_results(sim_run=p.sim_run,nGal=p.nGal)
    
    marker              =   'o'
    if p.sim_run == p.sim_runs[0]: marker = '^'

    L_line              =   getattr(GR,'L_'+p.line+'_sun')#[380:400]#[0:100]
    SFR                 =   getattr(GR,'SFR')#[380:400]#[0:100]
    M_star              =   getattr(GR,'M_star')#[380:400]#[0:100]
    # G0_mw               =   getattr(GR,'F_FUV_mw')#[0:100]
    Zsfr                =   getattr(GR,'Zsfr')#[380:400]#[0:100]
    R_gas               =   getattr(GR,'R2_gas')#[380:400]#[0:100]
    M_H2                =   getattr(GR,'M_H2_R2_gas')#[380:400]#[0:100]
    # if 'CO' in p.line: p.select = 'Sigma_M_H2'

    # Take only MS galaxies?
    if p.MS == True:
        indices = aux.select_salim18(GR.M_star,GR.SFR)
        L_line = L_line[indices]
        SFR = SFR[indices]
        M_star = M_star[indices]
        Zsfr = Zsfr[indices]
        R_gas = R_gas[indices]
        M_H2 = M_H2[indices]
        print('With MS selection criteria: only %i galaxies' % (len(L_line)))

    # Just selection of galaxies
    #SFR = SFR[0:10]
    #Zsfr = Zsfr[0:10]
    #R_gas = R_gas[0:10]
    #M_H2 = M_H2[0:10]
    #L_line = L_line[0:10]
    #M_star = M_star[0:10]

    SFR = SFR[L_line > 0]
    M_star = M_star[L_line > 0]
    Zsfr = Zsfr[L_line > 0]
    R_gas = R_gas[L_line > 0]
    M_H2 = M_H2[L_line > 0]
    L_line = L_line[L_line > 0]
    print('%i data points ' % (len(L_line)))

    lSFR = np.log10(SFR)
    lL_line = np.log10(L_line)


    # plt.plot(np.log10(M_star),np.log10(SFR),'o')
    # s = aseg

    labs                =   {'_100Mpc_M10':'Mach=10 power-law',\
                            '_100Mpc_arepoPDF_CMZ':'SIGAME v3',\
                            '_25Mpc_arepoPDF_M51':'SIGAME v3 (Simba-25)',\
                            '_100Mpc_arepoPDF_M51':'SIGAME v3 (Simba-100)'}
    lab                 =   labs[p.sim_run+p.table_ext]

    if p.add:
        ax = p.ax
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    if p.select == 'Sigma_M_H2':
        Sigma_M_H2 = M_H2/(np.pi*R_gas**2)/1e6 # per pc^-2
        m = ax.scatter(lSFR[np.argsort(Sigma_M_H2)],lL_line[np.argsort(Sigma_M_H2)],marker=marker,s=14,\
                    c=np.log10(Sigma_M_H2[np.argsort(Sigma_M_H2)]),vmin=-2.5,vmax=2.2,label=lab,alpha=0.5,zorder=10)
        p.vmin = np.log10(Sigma_M_H2.min())
        p.vmax = np.log10(Sigma_M_H2.max())
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label=r'log $\Sigma_{H2}$ [M$_{\odot}$/pc$^2$]',size=15)
    if p.select == 'M_star':
        m = ax.scatter(lSFR[np.argsort(M_star)],lL_line[np.argsort(M_star)],marker=marker,s=8,\
                   c=np.log10(M_star[np.argsort(M_star)]),vmin=-2.5,vmax=2.2,label=lab,alpha=0.5,zorder=10)
        # Just one galaxy
        # m = ax.scatter(lSFR,lL_line,marker=marker,s=15,\
                   # c=np.log10(Sigma_M_H2),vmin=-2.5,vmax=2.2,label=lab,alpha=1,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label=r'log $M_{star}$ [M$_{\odot}$]',size=15)
    if p.select == 'Zsfr':
        print('min and max Zsfr in sims: ',Zsfr.min(),Zsfr.max())
        p.vmin = np.log10(0.01)
        p.vmax = np.log10(3)
        m = ax.scatter(lSFR,lL_line,marker=marker,s=20,\
                   c=np.log10(Zsfr),label=lab,alpha=0.6,zorder=10,vmin=p.vmin,vmax=p.vmax)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label=r'log $\langle Z\rangle_{\mathrm{SFR}}$ [Z$_{\odot}$]',size=15)
    if p.select == 'F_FUV_mw':
        m = ax.scatter(lSFR,lL_line,marker=marker,s=20,\
                   c=np.log10(G0_mw),label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label='log ' + getlabel('G0'),size=15)
    if p.select == 'f_HII':
        f_HII[f_HII == 0] = np.min(f_HII[f_HII > 0])
        m = ax.scatter(lSFR[np.argsort(f_HII)],lL_line[np.argsort(f_HII)],marker=marker,s=20,\
                   c=np.log10(f_HII[np.argsort(f_HII)]),label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label='log HII region fraction',size=15)


    # Label galaxies?
    # for i in range(len(SFR)):
    #     if SFR[i] > 0:
    #         ax.text(SFR[i],L_line[i],'G%i' % GR.gal_num[i],fontsize=7)

    if p.add_obs:
        if (p.select == 'Zsfr') | (p.select == 'Sigma_M_H2'): 
            add_line_SFR_obs(p.line,L_line,ax,select=p.select,vmin=p.vmin,vmax=p.vmax)
        else:
            add_line_SFR_obs(p.line,L_line,ax,select=p.select)

    ax.set_xlabel('log ' + getlabel('SFR'))
    ax.set_ylabel('log ' + getlabel(p.line))
    handles,labels = ax.get_legend_handles_labels()
    handles = np.flip(handles)
    labels = np.flip(labels)
    if ('CO' in p.line) | ('[OI]' in p.line): 
        ax.legend(handles,labels,loc='upper left',fontsize=7,frameon=True,framealpha=0.5)
    else:
        ax.legend(handles,labels,loc='lower right',fontsize=7,frameon=True,framealpha=0.5)
    if not p.xlim: p.xlim = np.array([-3,4])
    if not p.ylim: 
        p.ylim = [np.median(lL_line) - 5,np.median(lL_line) + 3]
        if p.line == '[OI]63': p.ylim = [np.median(lL_line) - 5,np.median(lL_line) + 4]
        if 'CO' in p.line: p.ylim = [np.median(lL_line) - 4,np.median(lL_line) + 4]

    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)
    ax.grid(ls='--')

    if p.savefig & (not p.add):
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/%s_SFR.png' % p.line, format='png', dpi=300)  

def dline_dSFR(**kwargs):
    """ Plot deviation from observed line-SFR relation vs distance from MS
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results(sim_run=p.sim_run,nGal=p.nGal)
    
    marker              =   'o'
    if p.sim_run == p.sim_runs[0]: marker = '^'

    L_line              =   getattr(GR,'L_'+p.line+'_sun')#[380:400]#[0:100]
    SFR                 =   getattr(GR,'SFR')#[380:400]#[0:100]
    M_star              =   getattr(GR,'M_star')#[380:400]#[0:100]
    Zsfr                =   getattr(GR,'Zsfr')#[380:400]#[0:100]
    R_gas               =   getattr(GR,'R2_gas')#[380:400]#[0:100]
    M_H2                =   getattr(GR,'M_H2_R2_gas')#[380:400]#[0:100]

    SFR = SFR[L_line > 0]
    M_star = M_star[L_line > 0]
    Zsfr = Zsfr[L_line > 0]
    R_gas = R_gas[L_line > 0]
    M_H2 = M_H2[L_line > 0]
    L_line = L_line[L_line > 0]
    print('%i data points ' % (len(L_line)))

    # Distance from MS
    dlSFR = aux.distance_from_salim18(GR.M_star,GR.SFR)

    if p.add:
        ax = p.ax
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    # Distance from observed relation
    L_obs,SFR_obs,fit,std = add_line_SFR_obs(p.line,[1e6,1e6],ax,plot=False,select=p.select)
    ldL_line = np.log10(L_line) - fit.predict(np.log10(SFR.reshape(-1, 1))).flatten()

    labs                =   {'_M10':'Mach=10 power-law',\
                            '_arepoPDF_ext':'AREPO parametric PDF with extinction',\
                            '_arepoPDF':'SIGAME v3',\
                            '_arepoPDF_CMZ':'SIGAME v3',\
                            '_arepoPDF_M51':'SIGAME v3'}
    lab                 =   labs[p.table_ext]


    ax.text(0.05,0.9,p.line,transform=ax.transAxes,fontsize=13)
    ax.set_xlabel('log SFR - log SFR$_{MS,Salim+18}$')
    ax.set_ylabel('log L - log L$_{obs}$(SFR)')
    if not p.xlim: p.xlim = np.array([-3,3])
    if not p.ylim: 
        p.ylim = [np.median(ldL_line) - 4,np.median(ldL_line) + 3]
        # if p.line == '[OI]63': p.ylim = [np.median(ldL_line) - 5,np.median(ldL_line) + 4]
        # if 'CO' in p.line: p.ylim = [np.median(ldL_line) - 4,np.median(ldL_line) + 4]

    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)
    ax.plot([0,0],ax.get_ylim(),'--k',lw=1)
    ax.plot(ax.get_xlim(),[0,0],'--k',lw=1)

    if p.select == 'Sigma_M_H2':
        Sigma_M_H2 = M_H2/(np.pi*R_gas**2)/1e6 # per pc^-2
        m = ax.scatter(dlSFR[np.argsort(Sigma_M_H2)],ldL_line[np.argsort(Sigma_M_H2)],marker=marker,s=14,\
                    c=np.log10(Sigma_M_H2[np.argsort(Sigma_M_H2)]),vmin=-2.5,vmax=2.2,label=lab,alpha=0.5,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label=r'log $\Sigma_{H2}$ [M$_{\odot}$/pc$^2$]',size=15)

def dline_Mgas(**kwargs):
    """ Plot deviation from observed line-SFR relation vs gas mass
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results(sim_run=p.sim_run,nGal=p.nGal)
    
    marker              =   'o'
    if p.sim_run == p.sim_runs[0]: marker = '^'

    L_line              =   getattr(GR,'L_'+p.line+'_sun')#[380:400]#[0:100]
    SFR                 =   getattr(GR,'SFR')#[380:400]#[0:100]
    M_star              =   getattr(GR,'M_star')#[380:400]#[0:100]
    Zsfr                =   getattr(GR,'Zsfr')#[380:400]#[0:100]
    R_gas               =   getattr(GR,'R2_gas')#[380:400]#[0:100]
    M_H2                =   getattr(GR,'M_H2_R2_gas')#[380:400]#[0:100]
    M_gas               =   getattr(GR,'M_gas')#[380:400]#[0:100]

    SFR = SFR[L_line > 0]
    M_star = M_star[L_line > 0]
    Zsfr = Zsfr[L_line > 0]
    R_gas = R_gas[L_line > 0]
    M_H2 = M_H2[L_line > 0]
    M_gas = M_gas[L_line > 0]
    L_line = L_line[L_line > 0]
    print('%i data points ' % (len(L_line)))

    if p.add:
        ax = p.ax
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    # Distance from observed relation
    L_obs,SFR_obs,fit,std = add_line_SFR_obs(p.line,[1e6,1e6],ax,plot=False,select=p.select)
    ldL_line = np.log10(L_line) - fit.predict(np.log10(SFR.reshape(-1, 1))).flatten()

    labs                =   {'_M10':'Mach=10 power-law',\
                            '_arepoPDF_ext':'AREPO parametric PDF with extinction',\
                            '_arepoPDF':'SIGAME v3',\
                            '_arepoPDF_CMZ':'SIGAME v3',\
                            '_arepoPDF_M51':'SIGAME v3'}
    lab                 =   labs[p.table_ext]


    ax.text(0.05,0.9,p.line,transform=ax.transAxes,fontsize=13)
    ax.set_xlabel(getlabel('M_gas'))
    ax.set_ylabel('log L - log L$_{obs}$(SFR)')
    if not p.xlim: p.xlim = np.array([8,11])
    if not p.ylim: 
        p.ylim = [np.median(ldL_line) - 4,np.median(ldL_line) + 3]
    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)
    ax.plot([0,0],ax.get_ylim(),'--k',lw=1)
    ax.plot(ax.get_xlim(),[0,0],'--k',lw=1)

    if p.select == 'Sigma_M_H2':
        Sigma_M_H2 = M_H2/(np.pi*R_gas**2)/1e6 # per pc^-2
        m = ax.scatter(np.log10(M_gas)[np.argsort(Sigma_M_H2)],ldL_line[np.argsort(Sigma_M_H2)],marker=marker,s=14,\
                    c=np.log10(Sigma_M_H2[np.argsort(Sigma_M_H2)]),vmin=-2.5,vmax=2.2,label=lab,alpha=0.5,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label=r'log $\Sigma_{H2}$ [M$_{\odot}$/pc$^2$]',size=15)

    print(np.log10(M_gas.min()),np.log10(M_gas.max()))



def dline_tdepl(**kwargs):
    """ Plot deviation from observed line-SFR relation vs gas depletion time scale
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results(sim_run=p.sim_run,nGal=p.nGal)
    
    marker              =   'o'
    if p.sim_run == p.sim_runs[0]: marker = '^'

    L_line              =   getattr(GR,'L_'+p.line+'_sun')#[380:400]#[0:100]
    SFR                 =   getattr(GR,'SFR')#[380:400]#[0:100]
    M_star              =   getattr(GR,'M_star')#[380:400]#[0:100]
    M_gas               =   getattr(GR,'M_gas')#[380:400]#[0:100]
    Zsfr                =   getattr(GR,'Zsfr')#[380:400]#[0:100]
    R_gas               =   getattr(GR,'R2_gas')#[380:400]#[0:100]
    M_H2                =   getattr(GR,'M_H2_R2_gas')#[380:400]#[0:100]
    M_HII               =   getattr(GR,'M_HII_R2_gas')#[380:400]#[0:100]
    M_HI                =   getattr(GR,'M_HI_R2_gas')#[380:400]#[0:100]
    G0_mw               =   getattr(GR,'G0_mw')#[380:400]#[0:100]
    nH_mw               =   getattr(GR,'nH_mw')#[380:400]#[0:100]
    tdepl               =   getattr(GR,'M_gas') / getattr(GR,'SFR') / 1e9 # Gyr

    SFR = SFR[L_line > 0]
    M_star = M_star[L_line > 0]
    M_gas = M_gas[L_line > 0]
    Zsfr = Zsfr[L_line > 0]
    R_gas = R_gas[L_line > 0]
    M_H2 = M_H2[L_line > 0]
    M_HII = M_HII[L_line > 0]
    M_HI = M_HI[L_line > 0]
    G0_mw = G0_mw[L_line > 0]
    nH_mw = nH_mw[L_line > 0]
    tdepl = tdepl[L_line > 0]
    L_line = L_line[L_line > 0]
    print('%i data points ' % (len(L_line)))

    if p.add:
        ax = p.ax
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    # Distance from observed relation
    L_obs,SFR_obs,fit,std = add_line_SFR_obs(p.line,[1e6,1e6],ax,plot=False,select=p.select)
    ldL_line = np.log10(L_line) - fit.predict(np.log10(SFR.reshape(-1, 1))).flatten()

    labs                =   {'_M10':'Mach=10 power-law',\
                            '_arepoPDF_ext':'AREPO parametric PDF with extinction',\
                            '_arepoPDF':'SIGAME v3',\
                            '_arepoPDF_CMZ':'SIGAME v3',\
                            '_arepoPDF_M51':'SIGAME v3'}
    lab                 =   labs[p.table_ext]


    ax.set_xlabel(r'log $\tau$ [Gyr]')
    ax.set_ylabel('log L - log L$_{obs}$(SFR)')
    if not p.xlim: p.xlim = np.array([0,2.5])
    if not p.ylim: p.ylim = [np.median(ldL_line) - 4,np.median(ldL_line) + 3]

    ax.text(0.05,0.9,p.line,transform=ax.transAxes,fontsize=13)
    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)
    ax.plot([0,0],ax.get_ylim(),'--k',lw=1)
    ax.plot(ax.get_xlim(),[0,0],'--k',lw=1)

    if p.select == 'Sigma_M_H2':
        Sigma_M_H2 = M_H2/(np.pi*R_gas**2)/1e6 # per pc^-2
        m = ax.scatter(np.log10(tdepl)[np.argsort(Sigma_M_H2)],ldL_line[np.argsort(Sigma_M_H2)],marker=marker,s=14,\
                    c=np.log10(Sigma_M_H2[np.argsort(Sigma_M_H2)]),vmin=-2.5,vmax=2.2,label=lab,alpha=0.5,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label=r'log $\Sigma_{H2}$ [M$_{\odot}$/pc$^2$]',size=15)
    if p.select == 'F_FUV_mw':
        m = ax.scatter(np.log10(tdepl)[np.argsort(G0_mw)],ldL_line[np.argsort(G0_mw)],marker=marker,s=20,\
                   c=np.log10(G0_mw[np.argsort(G0_mw)]),label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label='log ' + getlabel('G0_mw'),size=15)
    if p.select == 'nH_mw':
        m = ax.scatter(np.log10(tdepl)[np.argsort(nH_mw)],ldL_line[np.argsort(nH_mw)],marker=marker,s=20,\
                   c=np.log10(nH_mw[np.argsort(nH_mw)]),label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label='log ' + getlabel('nH_mw'),size=15)
    if p.select == 'f_H2':
        f_H2 = M_H2/M_gas
        m = ax.scatter(np.log10(tdepl)[np.argsort(f_H2)],ldL_line[np.argsort(f_H2)],marker=marker,s=20,\
                   c=f_H2[np.argsort(f_H2)],label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label='H2 gas mass fraction',size=15)
    if p.select == 'f_HII':
        f_HII = M_HII/M_gas
        m = ax.scatter(np.log10(tdepl)[np.argsort(f_HII)],ldL_line[np.argsort(f_HII)],marker=marker,s=20,\
                   c=f_HII[np.argsort(f_HII)],label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label='HII gas mass fraction',size=15)
    if p.select == 'f_HI':
        f_HI = M_HI/M_gas
        m = ax.scatter(np.log10(tdepl)[np.argsort(f_HI)],ldL_line[np.argsort(f_HI)],marker=marker,s=20,\
                   c=f_HI[np.argsort(f_HI)],label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label='HI gas mass fraction',size=15)
    if p.select == 'M_H2':
        m = ax.scatter(np.log10(tdepl)[np.argsort(M_H2)],ldL_line[np.argsort(M_H2)],marker=marker,s=20,\
                   c=np.log10(M_H2[np.argsort(M_H2)]),label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label='log ' + getlabel('M_H2'),size=15)





    print(tdepl.min(),tdepl.max())

def line_sSFR(**kwargs):
    """ Plot line luminosity (in Lsun) against sSFR
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()
    
    L_line              =   getattr(GR,'L_'+p.line+'_sun')#[0:100]
    SFR                 =   getattr(GR,'SFR')#[0:100]
    Zsfr                =   getattr(GR,'Zsfr')#[0:100]
    R_gas               =   getattr(GR,'R2_gas')#[0:100]
    M_H2                =   getattr(GR,'M_H2_R2_gas')#[0:100]
    M_star              =   getattr(GR,'M_star')#[0:100]

    # Take only MS galaxies?
    if p.select == '_MS':
        indices = aux.select_salim18(GR.M_star,GR.SFR)
        L_line = L_line[indices]
        SFR = SFR[indices]
        Zsfr = Zsfr[indices]
        print('With MS selection criteria: only %i galaxies' % (len(L_line)))

    SFR = SFR[L_line > 0]
    Zsfr = Zsfr[L_line > 0]
    R_gas = R_gas[L_line > 0]
    M_H2 = M_H2[L_line > 0]
    M_star = M_star[L_line > 0]
    sSFR = SFR/M_star
    L_line = L_line[L_line > 0]

    print('%i data points ' % (len(L_line)))

    labs                =   {'_M10':'Mach=10 power-law',\
                            '_arepoPDF_dim':'AREPO parametric PDF with extinction',\
                            '_arepoPDF':'AREPO parametric PDF'}
    lab                 =   labs[p.table_ext]

    if p.add:
        ax = p.ax
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    if p.select == 'Sigma_M_H2':
        Sigma_M_H2 = M_H2/(np.pi*R_gas**2)
        m = ax.scatter(sSFR[np.argsort(Sigma_M_H2)],L_line[np.argsort(Sigma_M_H2)],marker='o',s=20,\
                   c=np.log10(Sigma_M_H2[np.argsort(Sigma_M_H2)]),vmin=3.5,label=lab,alpha=0.6,zorder=10)
        cbar = plt.colorbar(m,ax=ax)
        cbar.set_label(label=r'log $\Sigma_{H2}$ [M$_{\odot}$/kpc$^2$]',size=15)
    else:
        m = ax.scatter(sSFR,L_line,marker='o',s=20,\
                   c=Zsfr,label=lab,alpha=0.6,zorder=10)
        cbar = plt.colorbar(m,ax=ax)
        cbar.set_label(label=r'$\langle Z\rangle_{\mathrm{SFR}}$ [Z$_{\odot}$]',size=15)

    if p.add_obs:
        add_line_sSFR_obs(p.line,L_line,ax,select=p.select)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(getlabel('sSFR'))
    ax.set_ylabel(getlabel(p.line))
    handles,labels = ax.get_legend_handles_labels()
    handles = np.flip(handles)
    labels = np.flip(labels)
    # ax.legend(handles,labels,loc='upper left',fontsize=7)
    ax.legend(handles,labels,loc='lower right',fontsize=7,frameon=True,framealpha=0.5)    
    print(np.min(sSFR),np.max(sSFR))
    if not p.xlim: p.xlim = 10.**np.array([-13,-7])
    if not p.ylim: 
        p.ylim = [np.median(L_line)/1e6,np.median(L_line)*1e4]
    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)
    ax.grid(ls='--')

    if p.savefig & (not p.add):
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/%s_sSFR.png' % p.line, format='png', dpi=300)  

def line_FIR(**kwargs):
    """ Plot line luminosity (in Lsun) against FIR luminosity
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    L_line              =   getattr(GR,'L_'+p.line+'_sun')
    # L_FIR               =   getattr(GR,'L_FIR_sun')

    labs                =   {'':'SIGAME default: lognormal + power-law, Mach=10',\
                            '_arepoPDF_dim':'AREPO parametric PDF with extinction',\
                            '_arepoPDF':'SIGAME parametric PDF'}
    lab                 =   labs[p.table_ext]

    if p.add:
        ax = p.ax
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    n = 388
    m = ax.scatter(GR.L_FIR_sun[0:n],L_line[0:n],marker='o',s=20,\
               c=GR.Zsfr[0:n],label=lab,alpha=0.6,zorder=10)

    if p.add_obs:
        add_line_FIR_obs(p.line,ax)

    m = ax.scatter(GR.L_FIR_sun/1e6,L_line/1e6,marker='o',s=30,c=GR.Zsfr)
    cbar = plt.colorbar(m,ax=ax)
    cbar.set_label(label='$Z_{\mathrm{SFR}}$ [Z$_{\odot}$]',size=15)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(getlabel('L_FIR'))
    ax.set_ylabel(getlabel(p.line))
    handles,labels = ax.get_legend_handles_labels()
    handles = np.flip(handles)
    labels = np.flip(labels)
    ax.legend(handles,labels,loc='upper left',fontsize=7)
    if not p.xlim: p.xlim = 10.**np.array([6,13])
    if not p.ylim: 
        p.ylim = [np.median(L_line)/1e4,np.median(L_line)*1e4]
    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)
    ax.grid(ls='--')

    if p.savefig & (not p.add):
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/%s_FIR.png' % p.line, format='png', dpi=300)  

    # plt.close('all')

def line_Mgas(**kwargs):
    """ Plot line luminosity (in K km/s pc^2) against total ISM gas mass
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    L_line              =   getattr(GR,'L_'+p.line+'_sun')
    L_line              =   aux.Lsun_to_K_km_s_pc2(L_line,p.line)
    M_gas               =   getattr(GR,'M_gas')

    # Plot
    fig,ax = plt.subplots(figsize=(8,6))
    ax.plot(np.log10(M_gas),np.log10(L_line),'x',label='Simba galaxies')
    ax.set_xlabel(getlabel('lM_ISM'))
    ax.set_ylabel('log(L$_{\mathrm{%s}}$ [K km$\,s^{-1}$ pc$^2$])' % p.line)

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/%s_Mgas.png' % p.line, format='png', dpi=300)  

    # plt.close('all')

def line_Mstar(**kwargs):
    """ Plot line luminosity (in K km/s pc^2) against total stellar mass
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    L_line              =   getattr(GR,'L_'+p.line+'_sun')

    L_line              =   aux.Lsun_to_K_km_s_pc2(L_line,p.line)
    M_star              =   getattr(GR,'M_star')

    # Plot
    fig,ax = plt.subplots(figsize=(8,6))
    ax.plot(np.log10(M_star),np.log10(L_line),'x',label='Simba galaxies')
    ax.set_xlabel(getlabel('lM_star'))
    ax.set_ylabel('log(L$_{\mathrm{%s}}$ [K km$\,s^{-1}$ pc$^2$])' % p.line)

    if p.ylim:
        ax1 = plt.gca()
        ax1.set_ylim(p.ylim)

    if p.xlim:
        ax1 = plt.gca()
        ax1.set_xlim(p.xlim)

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/%s_Mstar.png' % p.line, format='png', dpi=300)  

def add_line_sSFR_obs(line,L_line,ax,**kwargs):
    """ Add observed galaxies as datapoints and relations if possible to line-sSFR plot

    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # --- Observations compiled in Observations.ipynb ---

    L_obs = np.array([])
    sSFR_obs = np.array([])

    if p.plot: print('\nObserved galaxies with %s:' % line)

    # Cormier et al. 2015 and Madden et al. 2013
    df = pd.read_pickle('data/observations/DGS_Cormier_2015') 
    try:
        df = pd.read_pickle('data/observations/DGS_Cormier_2015')
        if p.plot: 
            ax.errorbar(10.**df.sSFR,df['L_'+line],yerr=df['e_'+line], elinewidth=1,marker='s',ms=5,mew=0,\
                color='grey',alpha=0.8,lw=0)
            # ax.plot(10.**df.sSFR,df['L_'+line],'s',ms=5,mew=0,color='grey',alpha=0.8,label='Cormier+15 (dwarfs)')
            L_ul = df['L_'+line][df['L_'+line] < 0]
            if len(L_ul) > 0:
                ax.plot(10.**df.sSFR[df['L_'+line] < 0],L_ul,'s',ms=5,mew=0,color='grey',alpha=0.8)
                ax.errorbar(10.**df.sSFR[df['L_'+line] < 0],L_ul, elinewidth=1,\
                    uplims=np.ones(len(L_ul)),yerr=0.3,color='grey',alpha=0.8,lw=0)
                #-1.*L_ul - 10.**(np.log10(-1.*L_ul)-0.3)
        L_obs = np.append(L_obs,df['L_'+line].values)
        # print(df['L_'+line].values)
        sSFR_obs = np.append(sSFR_obs,df.sSFR.values)
        if p.plot: print('%i galaxies from Cormier+15 with positiv flux' % (len(df['L_'+line].values[df['L_'+line].values > 0])))
        # print('min SFR: ',np.min(df.SFR.values[df.sizes < 47]))
    except:
        pass

def add_line_SFR_obs(line,L_line,ax,plot_fit=True,**kwargs):
    """ Add observed galaxies as datapoints and relations if possible to line-SFR plot

    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # --- Observations compiled in Observations.ipynb ---

    L_obs = np.array([])
    SFR_obs = np.array([])

    if p.plot: print('\nObserved galaxies with %s:' % line)

    c = 'dimgrey'
    a = 0.8
    mew = 1

    # Kamenetzky et al. 2016
    df = pd.read_pickle('data/observations/AHIMSA_sample_lit')
    df = df[(df.sizes < 47) & (df.SFR > 1e-4) & (df[line+ '_Lsun'].values > 0)] 
    try:
        if p.plot: ax.plot(np.log10(df.SFR[(df.sizes < 47) & (df.SFR > 1e-4)]),\
            np.log10(df[line + '_Lsun'][(df.sizes < 47) & (df.SFR > 1e-4)]),'>',ms=6,fillstyle='none',mew=mew,\
        color=c,alpha=a,label='Mixed type galaxies [Kamenetzky+16]')
        lo_err = np.array(np.log10(df[line+ '_Lsun'].values)-np.log10(df[line+ '_Lsun'].values-df['e_'+line+ '_Lsun'].values))
        up_err = np.array(np.log10(df[line+ '_Lsun'].values+df['e_'+line+ '_Lsun'].values)-np.log10(df[line+ '_Lsun'].values))
        lo_err[df[line+ '_Lsun'].values == 0] = 0
        up_err[df[line+ '_Lsun'].values == 0] = 0
        # ax.errorbar(np.log10(df.SFR),\
        #     np.log10(df[line+ '_Lsun']),\
        #     yerr=np.column_stack([lo_err,up_err]).T,\
        #     elinewidth=1,marker='>',ms=6,mew=1,fillstyle='none',\
        #     color='grey',alpha=0.8,lw=0,label='Mixed z~0 sample [Kamenetzky+16]')
        L_obs = np.append(L_obs,df[line + '_Lsun'].values)
        SFR_obs = np.append(SFR_obs,df.SFR.values)
        if p.plot: print('%i galaxies from Kamenetzky+16 ' % (len(L_obs)))
    except:
        pass

    # print('min SFR: ',np.min(df.SFR.values[df.sizes < 47]))

    # Brauher et al. 2008
    try:
        df = pd.read_pickle('data/observations/Brauher_2008')
        if p.plot: 
            # lo_err = np.array(np.log10(df['L_'+line].values)-np.log10(df['L_'+line].values-df['e_'+line].values))
            # up_err = np.array(np.log10(df['L_'+line].values+df['e_'+line].values)-np.log10(df['L_'+line].values))
            # print(lo_err)
            # print(df['e_'+line].values/df['L_'+line])
            # ax.errorbar(np.log10(df.SFR),np.log10(df['L_'+line]),\
            #     yerr=np.column_stack([lo_err,up_err]).T,\
            #     elinewidth=1,marker='o',ms=7,mew=1,fillstyle='none',\
            #     color='grey',alpha=0.8,lw=0,label='MS/SB galaxies [Brauher+08]')
            ax.plot(np.log10(df.SFR),np.log10(df['L_%s' % line]),'o',fillstyle='none',ms=4,mew=mew,color=c,\
                alpha=a,label='MS/SB galaxies [Brauher+08]')
            L_ul = np.log10(df['L_%s' % line][df['f_'+line] == -1])
            if len(L_ul) > 0:
                # ax.plot(df.SFR[df['f_'+line] == -1],L_ul,'o',zorder=0,ms=7,mew=1,color='grey',alpha=0.8)
                ax.errorbar(np.log10(df.SFR[df['f_'+line] == -1]),L_ul,capsize=3,color=c,alpha=a,elinewidth=1,\
                    uplims=np.ones(len(L_ul)),\
                    yerr=0.3,lw=0)
        if p.plot: print('%i galaxies from Brauher+08 ' % (len(df)))
        L =df['L_%s' % line].values
        SFR =df['SFR'].values
        L = L[SFR > 0]
        SFR = SFR[SFR > 0]
        L_obs = np.append(L_obs,L)
        SFR_obs = np.append(SFR_obs,SFR)
        # print('min SFR: ',np.min(df.SFR))
    except:
        pass

    if p.select != '_MS':
        # Cormier et al. 2015
        try:
            df = pd.read_pickle('data/observations/DGS_Cormier_2015')
            if p.plot: 
                # try: 
                #     lo_err = np.array(np.log10(df['L_'+line].values)-np.log10(df['L_'+line].values-df['e_'+line].values))
                #     up_err = np.array(np.log10(df['L_'+line].values+df['e_'+line].values)-np.log10(df['L_'+line].values))
                #     ax.errorbar(df.SFR,np.log10(df['L_'+line]),\
                #         yerr=np.column_stack([lo_err,up_err]).T,\
                #         elinewidth=1,marker='x',ms=7,mew=mew,\
                #         color=c,alpha=a,lw=0,label='Dwarf galaxies [Cormier+15]')
                # except:
                ax.plot(df.SFR,np.log10(df['L_%s' % line]),'x',zorder=0,ms=7,mew=mew,color=c,alpha=a,\
                    label='Dwarf galaxies [Cormier+15]')
                L_ul = np.log10(-1.*df['L_'+line][df['L_'+line] < 0])
                if len(L_ul) > 0:
                    ax.plot(df.SFR[df['L_'+line] < 0],L_ul,'x',zorder=0,ms=7,mew=mew,color=c,alpha=a)
                    ax.errorbar(df.SFR[df['L_'+line] < 0],L_ul,capsize=3,color=c,alpha=a,elinewidth=1,\
                        uplims=np.ones(len(L_ul)),\
                        yerr=0.3,lw=0)
                    # np.log10(-1.*L_ul - 10.**(np.log10(-1.*L_ul)-0.3))
            if p.plot: print('%i galaxies from Cormier+15 ' % (len(df)))
            L_obs = np.append(L_obs,df['L_%s' % line].values)
            SFR_obs = np.append(SFR_obs,10.**df.SFR.values)
        except:
            pass

        # Schruba et al. 2012
        #try:
        if (line == 'CO(1-0)') | (line == 'CO(2-1)'):
            df = pd.read_pickle('data/observations/Schruba_2012')
            if p.plot: 
                if line == 'CO(1-0)': label = 'Mixed type galaxies [Schruba+12]'
                if line == 'CO(2-1)': label = 'Dwarf galaxies [Schruba+12]'
                f_ul = df['f_%s' % line].values
                L = df['L_%s' % line].values
                SFR = df['SFR'].values
                L_obs = np.append(L_obs,L[L > 0])
                SFR_obs = np.append(SFR_obs,SFR[L > 0])
                Z = df['Z'].values
                if line == 'CO(2-1)': 
                    SFR = SFR[L>0]
                    f_ul = f_ul[L>0]
                    Z = Z[L>0]
                    L = L[L>0]
                    print('Schruba min max Z: ',Z.min(),Z.max())
                    M_H2 = 1.8e9 * SFR # from S12 paper
                    area = np.array([1.33,1.79,1.75,7.74,11.47,12.37,26.69,83.85,12.23,39.40,19.21,7.78,14.75,59.54,31.19,39.19]) # kpc2
                    Sigma_M_H2 = M_H2 / (area*1000*1000)
                    if p.select == 'Zsfr': 
                        ax.scatter(np.log10(SFR[L > 0]),np.log10(L[L > 0]),marker='*',zorder=0,facecolors='none',s=30,\
                            linewidth=mew,c=np.log10(Z),alpha=a,label=label,vmin=p.vmin,vmax=p.vmax)
                    else:
                        ax.scatter(np.log10(SFR[L > 0]),np.log10(L[L > 0]),marker='*',zorder=0,facecolors='none',s=30,\
                            linewidth=mew,c=np.log10(Sigma_M_H2),alpha=a,label=label,vmin=p.vmin,vmax=p.vmax)
                if line == 'CO(1-0)': 
                    ax.plot(np.log10(SFR[L > 0]),np.log10(L[L > 0]),'*',zorder=0,fillstyle='none',ms=7,mew=mew,color=c,alpha=a,\
                        label=label)
                if len(f_ul) > 0:
                    # ax.plot(np.log10(SFR[f_ul == 1]),np.log10(L[f_ul == 1]),'*',zorder=0,fillstyle='none',ms=7,mew=mew,color=c,alpha=a)
                    ax.errorbar(np.log10(SFR[f_ul == 1]),np.log10(L[f_ul == 1]),capsize=3,fillstyle='none',color=c,alpha=a,elinewidth=1,\
                        uplims=np.ones(len(L[f_ul == 1])),\
                        yerr=0.3,lw=0)
            if p.plot: print('%i galaxies from Schruba+12 ' % (len(df)))
        #except:
        #    pass

        # Accurso et al. 2017
        try:
            df = pd.read_pickle('data/observations/xCOLD_GASS_Accurso_2017')
            df = df.loc[np.argwhere(df['L_CO(1-0)'].values > 0).flatten()]
            if p.plot: ax.plot(np.log10(df['SFR']),df['L_%s' % line], 'd', zorder=0,ms=7,fillstyle='none',mew=mew,color=c,alpha=a,label='COLD GASS [Accurso+17]') #c=np.log10(A17['Z']), 
            L_obs = np.append(L_obs,10.**df['L_%s' % line].values)
            if p.plot: print('%i galaxies from Accurso+17 ' % (len(df)))
            SFR_obs = np.append(SFR_obs,df.SFR.values)
        except:
            pass

        # Vanzi et al. 2009
        if line == 'CO(3-2)':
            df = pd.read_pickle('data/observations/Vanzi_2009')
            df = df.loc[np.argwhere(df['L_CO(3-2)'].values > 0).flatten()]
            if p.plot: ax.plot(np.log10(df['SFR']),np.log10(df['L_%s' % line]), 'D', zorder=0,ms=7,fillstyle='none',mew=mew,\
                color=c,alpha=a,label='Dwarf galaxies [Vanzi+09]') #c=np.log10(A17['Z']), 
            L_obs = np.append(L_obs,df['L_%s' % line].values)
            if p.plot: print('%i galaxies from Vanzi+09 ' % (len(df)))
            SFR_obs = np.append(SFR_obs,df.SFR.values)
        # except:
        #     pass


        # Diaz-Santos et al. 2013
        try:
            df = pd.read_pickle('data/observations/Diaz-Santos_2013')
            if p.plot: ax.plot(np.log10(df.SFR),np.log10(df['L_%s' % line]),'^',ms=6,zorder=0,fillstyle='none',mew=mew,color=c,alpha=a,label='LIRGs [Diaz-Santos+13]')
            if p.plot: print('%i galaxies from Diaz-Santos+17 ' % (len(df)))
            L_obs = np.append(L_obs,df['L_%s' % line].values)
            SFR_obs = np.append(SFR_obs,df.SFR.values)
            # print('min SFR: ',np.min(df.SFR))
        except:
            pass
        # Farrah et al. 2013
        # try:
        #     df = pd.read_pickle('data/observations/Farrah_2013')
        #     if p.plot: ax.plot(df.SFR,df['L_%s' % line],'<',fillstyle='none',mew=1,color='grey',alpha=0.8,label='Farrah+13 (ULIRGs)')
        #     if p.plot: print('%i galaxies from Farrah+13 ' % (len(df)))
        #     L_obs = np.append(L_obs,df['L_%s' % line].values)
        #     SFR_obs = np.append(SFR_obs,df.SFR.values)
        # except:
        #     pass
        # Zhao et al. 2016
        try:
            df = pd.read_pickle('data/observations/Zhao_2016')
            if p.plot: ax.plot(np.log10(df.SFR),np.log10(df['L_%s' % line]),'<',ms=6,fillstyle='none',mew=mew,color=c,alpha=a,label='GOALS (U)LIRGs [Zhao+16]')
            if p.plot: print('%i galaxies from Zhao+16 ' % (len(df)))
            L_obs = np.append(L_obs,df['L_%s' % line].values)
            SFR_obs = np.append(SFR_obs,df.SFR.values)
            # print('min SFR: ',np.min(df.SFR))
        except:
            pass

    if line in ['[CII]158','[OI]63','[OIII]88']:
        # De Looze 2014 relation
        if np.min(L_line) == 0 : L_line[L_line == 0] = 1e-30
        if p.plot: print(np.min(np.log10(L_line)),np.max(np.log10(L_line)))
        logL_delooze        =   np.arange(np.min(np.log10(L_line)) - 3,np.max(np.log10(L_line)) + 3)

        if line == '[CII]158':
            logSFR_delooze_DGS  =   -5.73 + 0.8 * logL_delooze
            logSFR_delooze_SBG  =   -7.06 + 1.0 * logL_delooze

        if line == '[OI]63':
            logSFR_delooze_DGS  =   -6.23 + 0.91 * logL_delooze
            logSFR_delooze_SBG  =   -6.05 + 0.89 * logL_delooze

        if line == '[OIII]88':
            logSFR_delooze_DGS  =   -6.71 + 0.92 * logL_delooze
            logSFR_delooze_SBG  =   -3.89 + 0.69 * logL_delooze

        if p.plot: ax.plot(logSFR_delooze_DGS,logL_delooze,'--',color='grey',alpha=0.7,\
            label='Local dwarf galaxies [de Looze+ 2014]')
        if p.plot: ax.plot(logSFR_delooze_SBG,logL_delooze,':',color='grey',alpha=0.7,\
            label='Local SB galaxies [de Looze+ 2014]')
    # print(SFR_obs)
    logSFR = np.arange(np.min(np.log10(SFR_obs[SFR_obs > 0])) - 3,np.max(np.log10(SFR_obs[SFR_obs > 0])) + 3)
    # fit = np.polyfit(np.log10(SFR_obs[(L_obs > 0) & (SFR_obs > 0)]),\
    #     np.log10(L_obs[(L_obs > 0) & (SFR_obs > 0)]),1)
    # pfit = np.poly1d(fit)
    # L_fit = 10.**pfit(logSFR)

    # Make log-linear fit to SFR-binned luminosities
    SFRs = SFR_obs[(L_obs > 0) & (SFR_obs > 0)]
    Ls = L_obs[(L_obs > 0) & (SFR_obs > 0)]
    SFR_axis = np.linspace(np.log10(SFRs.min()),np.log10(SFRs.max()),20)
    SFR_bins = SFR_axis[0:-1] + (SFR_axis[1]-SFR_axis[0])/2.
    Ls_binned = np.zeros(len(SFR_axis)-1)
    for i in range(len(Ls_binned)):
        Ls1 = Ls[(SFRs >= 10.**SFR_axis[i]) & (SFRs <= 10.**SFR_axis[i+1])]
        Ls_binned[i] = np.mean(np.log10(Ls1))
    SFR_bins = SFR_bins[Ls_binned > 0]
    Ls_binned = Ls_binned[Ls_binned > 0]
    # ax.plot(10.**SFR_bins,10.**Ls_binned,'x',color='orange',mew=3)
    fit = LinearRegression().fit(SFR_bins.reshape(-1, 1),\
        Ls_binned.reshape(-1, 1))
    L_fit = 10.**fit.predict(logSFR.reshape(-1, 1))
    if p.plot & plot_fit: ax.plot(logSFR,np.log10(L_fit),'--k',lw=1.5,zorder=0)

    # print(line)
    # print(np.log10(L_obs[(L_obs > 0) & (SFR_obs > 0)]))
    # print(fit.predict(SFR_obs[(L_obs > 0) & (SFR_obs > 0)].reshape(-1, 1)).flatten())

    std = np.std(np.log10(L_obs[(L_obs > 0) & (SFR_obs > 0)]) - \
        fit.predict(np.log10(SFR_obs[(L_obs > 0) & (SFR_obs > 0)]).reshape(-1, 1)).flatten())


    # Read literature data from AHIMSA project
    # obsdf       =   pd.read_pickle(p.d_data+'observations/sample_lit')
    # print(obsdf.keys())
    # print(L_obs)
    # print(SFR_obs)

    if not p.plot: 
        return(L_obs.flatten(),SFR_obs.flatten(),fit,std)

def add_line_FIR_obs(line,ax,**kwargs):
    """ Add observed galaxies as datapoints and relations if possible to line-FIR plot

    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # --- Observations compiled in Observations.ipynb ---

    print('\nObserved galaxies with %s:' % line)
    # Cormier et al. 2015 (line luminosity compiled in Zanella+08)
    try:
        df = pd.read_pickle('data/observations/Zanella2018_Cormier_2015')
        ax.plot(df.L_IR/1.4,df['L_%s' % line],'o',zorder=0,fillstyle='none',mew=1,color='grey',alpha=0.8,label='Cormier+15 (dwarfs)')
        print('%i galaxies from Cormier+15 ' % (len(df)))
    except:
        pass
    # Diaz-Santos et al. 2013
    try:
        df = pd.read_pickle('data/observations/Diaz-Santos_2013')
        ax.plot(df.L_FIR,df['L_%s' % line],'^',zorder=0,fillstyle='none',mew=1,color='grey',alpha=0.8,label='Diaz-Santos+13 (LIRGs)')
        print('%i galaxies from Diaz-Santos+17 ' % (len(df)))
    except:
        pass
    # Brauher et al. 2008
    try:
        df = pd.read_pickle('data/observations/Brauher_2008')
        ax.plot(df.L_FIR,df['L_%s' % line],'s',fillstyle='none',mew=1,color='grey',alpha=0.8,label='Brauher+08 (MS/SB)')
        print('%i galaxies from Brauher+08 ' % (len(df)))
        L =df['L_%s' % line].values
        F =df['L_FIR'].values
        print(np.min(L[L > 0]))
        print(np.min(F[F > 0]))
    except:
        pass
    # Farrah et al. 2013
    try:
        df = pd.read_pickle('data/observations/Farrah_2013')
        ax.plot(df.L_IR,df['L_%s' % line],'<',fillstyle='none',mew=1,color='grey',alpha=0.8,label='Farrah+13 (ULIRGs)')
        print('%i galaxies from Farrah+13 ' % (len(df)))
    except:
        pass
    # Kamenetzky et al. 2016
    try:
        df = pd.read_pickle('data/observations/AHIMSA_sample_lit')
        print('# of K16 galaxies with major axis < 47 arcsec: ',len(df.log_L_FIR[df.sizes < 47]))
        ax.plot(10.**df.log_L_FIR[df.sizes < 47],df[line + '_Lsun'][df.sizes < 47],'>',fillstyle='none',mew=1,color='grey',alpha=0.8,label='Kamenetzky+16 mixed')
        print('%i galaxies from Kamenetzky+16 ' % (len(df)))
    except:
        pass

def SED(**kwargs):
    """ SED (Powderday) + line emission
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()
    gal_num             =   getattr(GR,'gal_num')[p.gal_index]

    # Look for powderday SED
    found = True

    # Load MIRI filter curves
    MIRI                =   pd.read_csv('look-up-tables/observations/ImPCE_TN-00072-ATC-Iss2.txt',sep='\t',skiprows=2,\
                                names=['Wave','F560W','F770W','F1000W','F1280W','F1130W','F1500W','F1800W','F2100W','F2550W'])

    # gal_num=12
    # file_location = p.d_data + 'pd_data/%s/sed_%i' % (p.sim_run,gal_num)
    # pd_data = pickle.load(open(file_location,'rb'), encoding='latin1')
    # wav = np.array(pd_data[0,:])[0]
    # flux = np.array(pd_data[1,:])[0]
    try:
        file_location = p.d_data + 'pd_data/%s/sed_%i' % (p.sim_run,gal_num)
        pd_data = pickle.load(open(file_location,'rb'), encoding='latin1')
        wav = np.array(pd_data[0,:])[0]
        flux = np.array(pd_data[1,:])[0]
        print('Found powderday output found for gal_index = %i (gal_num = %i)!' % (p.gal_index,gal_num))
    except:
        print('No powderday output found for gal_index = %i (gal_num = %i)!' % (p.gal_index,gal_num))
        found = False

    if p.select == 'AGN': 
        try:
            file_location = p.d_data + 'pd_data/%s/sed_%i_agn' % (p.sim_run,gal_num)
            pd_data = pickle.load(open(file_location,'rb'), encoding='latin1')
            wav_agn = np.array(pd_data[0,:])[0]
            flux_agn = np.array(pd_data[1,:])[0]
        except:
            if found: print('no AGN spectra for gal_num %i' % gal_num)

    if found == True:
        wav_lines = []
        tot_flux = [] 
        if p.select == 'AGN': tot_flux_agn = [] 
        for line in p.lines:

            L_line = getattr(GR,'L_%s_sun' % line)[p.gal_index]
            D_L = getattr(GR,'lum_dist')[p.gal_index]

            L_line_Jy_km_s = aux.Lsun_to_Jy_km_s(L_line,D_L,line)

            freq = p.freq[line]

            wav_line = c.c.value / (freq*1e9) * 1e6 # microns

            if wav_line < np.max(wav):
                flux[np.argmin(np.abs(wav-wav_line))] += L_line_Jy_km_s

            if p.select == 'AGN': 
                try: 
                    flux_agn[np.argmin(np.abs(wav-wav_line))] += L_line_Jy_km_s
                except:
                    pass

            wav_lines += [wav_line]
            tot_flux += [flux[np.argmin(np.abs(wav-wav_line))]]

            if p.select == 'AGN': 
                try: 
                    tot_flux_agn += [flux_agn[np.argmin(np.abs(wav-wav_line))]]
                except:
                    pass

        fig,ax = plt.subplots(figsize=(12,6))
        # Show MIRI band
        ax.fill_between([5,28],[1e10,1e10],color='forestgreen',alpha=0.4)
        ax.loglog(wav,flux,'-',lw=2,label='Modeled spectrum\nof $z=0$ simulated galaxy')
        try: 
            ax.loglog(wav,flux_agn,'-',color='r',lw=2,label='with AGN')
        except:
            pass
        ax.set_xlabel(r'$\lambda$ [$\mu$m]')
        ax.set_ylabel('Flux (mJy)')
        ax.set_ylim([np.max(flux)*5/1e5,np.max(flux)*5.5])
        ax.set_xlim(1,10**3.1)

        cmap = plt.get_cmap('gist_rainbow_r')
        cmap = plt.get_cmap('brg')
        tot_flux = np.array(tot_flux)[wav_lines < np.max(wav)]
        line_names = np.array(p.lines)[wav_lines < np.max(wav)]
        wav_lines = np.array(wav_lines)[wav_lines < np.max(wav)]
        tot_flux = tot_flux[wav_lines.argsort()]
        line_names = line_names[wav_lines.argsort()]
        wav_lines = wav_lines[wav_lines.argsort()]
        colors = [cmap(i) for i in np.linspace(0, 1, len(wav_lines))]
        for i in range(len(wav_lines)):
            print(line_names[i],wav_lines[i])
            ax.plot(wav_lines[i],tot_flux[i],'x',mew=2,ms=5,color=colors[i])#,label=line_names[i])
            # ax.text(wav_lines[i]*0.8,tot_flux[i],line_names[i],fontsize=10,color=colors[i])
            if line_names[i] in ['H2_S(1)','[NeII]12','[FeII]25','[OI]63','[CII]158','[CI]370','[CI]610','CO(3-2)']:
                ax.text(wav_lines[i]*0.8,tot_flux[i]*3.5,line_names[i],fontsize=10,color=colors[i])
                ax.plot([wav_lines[i],wav_lines[i]],[tot_flux[i],tot_flux[i]*3],'--',lw=1,color=colors[i])
            if line_names[i] in ['H2_S(6)','H2_S(4)','H2_S(6)','[NII]122','[NII]205','[SIII]18']:
                ax.text(wav_lines[i]*0.8,tot_flux[i]*6.5,line_names[i],fontsize=10,color=colors[i])
                ax.plot([wav_lines[i],wav_lines[i]],[tot_flux[i],tot_flux[i]*6],'--',lw=1,color=colors[i])
            if line_names[i] in ['[OIV]25','[OIII]88']:
                ax.text(wav_lines[i]*0.8,tot_flux[i]/4.,line_names[i],fontsize=10,color=colors[i])
                ax.plot([wav_lines[i],wav_lines[i]],[tot_flux[i],tot_flux[i]/3],'--',lw=1,color=colors[i])
            if line_names[i] in ['[NeIII]15']:
                ax.text(wav_lines[i]*0.8,tot_flux[i]/6.5,line_names[i],fontsize=10,color=colors[i])
                ax.plot([wav_lines[i],wav_lines[i]],[tot_flux[i],tot_flux[i]/5],'--',lw=1,color=colors[i])
            if line_names[i] in ['[OI]145','H2_S(5)','H2_S(3)','H2_S(2)','H2_S(7)']:
                ax.text(wav_lines[i]*0.8,tot_flux[i]/9.,line_names[i],fontsize=10,color=colors[i])
                ax.plot([wav_lines[i],wav_lines[i]],[tot_flux[i],tot_flux[i]/7],'--',lw=1,color=colors[i])

        ax.legend(fontsize=13,fancybox=True, framealpha=0.5)

        print(MIRI.head())
        for f in MIRI.keys():
            if f != 'Wave':
                ax.fill_between(MIRI['Wave'].values,MIRI[f].values*1e5,alpha=0.6)
        ax.text(30,1e4,'JWST/MIRI filter curves',fontsize=15,color='steelblue')

        if p.savefig:
            if not os.path.isdir(p.d_plot + 'SEDs/'): os.mkdir(p.d_plot + 'SEDs/')    
            plt.savefig(p.d_plot + 'SEDs/sed_%s%s_%i.png' % (p.sim_name,p.sim_run,p.gal_index), format='png', dpi=300)  

        # plt.close('all')

def AGN_SB_diagnostic(**kwargs):
    """ Make a diagnostic plot like in Fernandez-Ontiveros 2016
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    fig,ax = plt.subplots()
    L_CII = getattr(GR,'L_[CII]158_sun')

    # x = getattr(GR,'L_[CII]158_sun')/getattr(GR,'L_[NII]122_sun')
    x = getattr(GR,'L_[OIV]25_sun')/getattr(GR,'L_[OIII]88_sun')
    y = getattr(GR,'L_[NeIII]15_sun')/getattr(GR,'L_[NeII]12_sun')
    sc = ax.scatter(x,y,marker='o',s=3,alpha=0.6,c=np.log10(getattr(GR,'SFR')))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([8e-5,200])
    ax.set_ylim([0.02,150])
    # ax.set_xlabel('[CII]$_{158}$/[NII]$_{122}$')
    plt.colorbar(sc,label='log(SFR)')
    ax.set_xlabel('[OIV]$_{25.9}$/[OIII]$_{88}$')
    ax.set_ylabel('[NeIII]$_{15.6}$/[NeII]$_{12.8}$')
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/ratio_%s' % ratio_name,dpi=300)

def CII_vs_CO(**kwargs):
    """ Make a diagnostic plot like in Fernandez-Ontiveros 2016
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results(sim_run=p.sim_runs[1],nGal=p.nGals[1],grid_ext=p.grid_exts[1])
    fig,ax1 = plt.subplots()
    L_CII = getattr(GR,'L_[CII]158_sun')
    L_CO = getattr(GR,'L_CO(1-0)_sun')
    Zsfr = getattr(GR,'Zsfr')
    lL_CO, lL_CII = np.log10(L_CO), np.log10(L_CII) 
    lL_CO, lL_CII = lL_CO[(L_CO > 0) & (L_CII > 0)], lL_CII[(L_CO > 0) & (L_CII > 0)]
    sc = ax1.scatter(np.log10(L_CO)-10, np.log10(L_CII)-10, marker='o', c=np.log10(Zsfr), cmap='viridis', zorder=10,\
        vmin=np.log10(0.05), vmax=np.log10(3.1), \
        s=10, alpha=0.8)#, label='SIGAME 100Mpc_arepoPDF')
    # print('Min Zsfr in Simba sample: ',np.min(Zsfr))
    # print('indices with L_CO < 1e0:')
    nbins = 100
    k = kde.gaussian_kde(np.column_stack([lL_CO,lL_CII]).T)
    x, y = np.mgrid[lL_CO.min():lL_CO.max():nbins*1j, \
               4:lL_CII.max():nbins*1j]
    z = k(np.vstack([x.flatten(), y.flatten()]))
    CS = ax1.contour(x, y, z.reshape(x.shape),colors='forestgreen',levels=8,zorder=10)
    CS.collections[0].set_label('SIGAME 100Mpc_arepoPDF')

    GR                  =   glo.global_results(sim_run=p.sim_runs[0],nGal=p.nGals[0],grid_ext=p.grid_exts[1])
    L_CII = getattr(GR,'L_[CII]158_sun')
    L_CO = getattr(GR,'L_CO(1-0)_sun')
    Zsfr = getattr(GR,'Zsfr')
    lL_CO, lL_CII = np.log10(L_CO), np.log10(L_CII) 
    lL_CO, lL_CII = lL_CO[(L_CO > 0) & (L_CII > 0)], lL_CII[(L_CO > 0) & (L_CII > 0)]
    lL_CO               =   np.append(lL_CO,np.array([6.1,5]))
    lL_CII              =   np.append(lL_CII,np.array([8.9,9.7]))
    # ax1.scatter(np.log10(L_CO), np.log10(L_CII), marker='^', c=np.log10(Zsfr), cmap='viridis', zorder=10,\
    #     vmin=np.log10(0.05), vmax=np.log10(3.1), \
    #     s=10, alpha=0.8, label='SIGAME 25Mpc_arepoPDF')
    nbins = 100
    k = kde.gaussian_kde(np.column_stack([lL_CO,lL_CII]).T)
    x, y = np.mgrid[lL_CO.min():lL_CO.max():nbins*1j, \
               4:lL_CII.max():nbins*1j]
    z = k(np.vstack([x.flatten(), y.flatten()]))
    CS = ax1.contour(x, y, z.reshape(x.shape),colors='deepskyblue',linestyles='dotted',levels=6)
    CS.collections[0].set_label('SIGAME 25Mpc_arepoPDF')

    GR                  =   glo.global_results(sim_run=p.sim_runs[1],nGal=p.nGals[1],grid_ext=p.grid_exts[0])
    L_CII = getattr(GR,'L_[CII]158_sun')
    L_CO = getattr(GR,'L_CO(1-0)_sun')
    Zsfr = getattr(GR,'Zsfr')
    lL_CO, lL_CII = np.log10(L_CO), np.log10(L_CII) 
    lL_CO, lL_CII = lL_CO[(L_CO > 0) & (L_CII > 0)], lL_CII[(L_CO > 0) & (L_CII > 0)]
    lL_CO               =   np.append(lL_CO,np.array([-2.2,4.7]))
    lL_CII              =   np.append(lL_CII,np.array([8,9.3]))
    # ax1.scatter(np.log10(L_CO), np.log10(L_CII), marker='^', c=np.log10(Zsfr), cmap='viridis', zorder=10,\
    #     vmin=np.log10(0.05), vmax=np.log10(3.1), \
    #     s=10, alpha=0.8, label='SIGAME v3 Simba-%s' % (p.sim_runs[0].replace('_','').replace('Mpc','')))
    nbins = 100
    k = kde.gaussian_kde(np.column_stack([lL_CO,lL_CII]).T)
    x, y = np.mgrid[lL_CO.min():lL_CO.max():nbins*1j, \
               4:lL_CII.max():nbins*1j]
    z = k(np.vstack([x.flatten(), y.flatten()]))
    CS = ax1.contour(x, y, z.reshape(x.shape),colors='brown',levels=8,zorder=5,linestyles='dashed')
    CS.collections[0].set_label('SIGAME 100Mpc_arepoPDF_no_ext')

    # Observations
    K16 = pd.read_pickle('data/observations/AHIMSA_sample_lit')
    K16_LCII = K16['[CII]158_Lsun']
    K16_LCO = K16['CO(1-0)_Lsun']
    ax1.plot(np.log10(K16_LCO), np.log10(K16_LCII), '>', color='grey', ms=6, fillstyle='none',alpha=0.8, mew=1,zorder=0,\
        label='Mixed type galaxies [Kamenetzky+16]')

    C15 = pd.read_pickle('data/observations/DGS_Cormier_2015')
    C15_LCII = C15['L_[CII]158']
    C15_LCO = C15['L_CO(1-0)']
    C15_Z = C15['Z']
    # L_ul = C15['L_[CII]158'][(C15['L_[CII]158'] < 0) & (C15['L_CO(1-0)'] > 0)]
    # if len(L_ul) > 0:
    #     ax1.plot(np.log10(C15['L_CO(1-0)'][C15['L_[CII]158'] < 0]),np.log10(-1.*L_ul),'s',ms=5,mew=0,color='grey',alpha=0.8)
    #     ax1.errorbar(np.log10(C15['L_CO(1-0)'][C15['L_[CII]158'] < 0]),np.log10(-1.*L_ul), elinewidth=1,\
    #       uplims=np.ones(len(L_ul)),yerr=np.ones(len(L_ul))*1,color='grey',alpha=0.8,lw=0)
    ax1.scatter(np.log10(C15_LCO), np.log10(C15_LCII), marker='+', c=np.log10(C15_Z), cmap='viridis', zorder=0,\
        vmin=np.log10(0.05), vmax=np.log10(3.1),\
        s=100, lw=3, alpha=0.8, label='Dwarf galaxies [Cormier+15]')

    A17 = pd.read_pickle('data/observations/xCOLD_GASS_Accurso_2017')
    A17 = A17.loc[np.argwhere(A17['L_CO(1-0)'].values > 0).flatten()]
    ax1.scatter(A17['L_CO(1-0)'],A17['L_[CII]158'], marker='d', c=np.log10(A17['Z']), cmap='viridis', zorder=0,\
        vmin=np.log10(0.05), vmax=np.log10(3.1),\
        s=50, lw=0, alpha=0.8, label='COLD GASS [Accurso+17]') #c=np.log10(A17['Z']), 

    CII_obs = np.log10(np.append(K16_LCII.values,C15_LCII.values))
    CO_obs = np.log10(np.append(K16_LCO.values,C15_LCO.values))
    CII_obs = np.append(CII_obs,A17['L_[CII]158'].values)
    CO_obs = np.append(CO_obs,A17['L_CO(1-0)'].values)
    index = np.argwhere((CII_obs > 0) & (CO_obs > 0)).flatten()
    CII_obs = CII_obs[index]
    CO_obs = CO_obs[index]

    x = np.linspace(0, 7, 100)
    fit = LinearRegression().fit(CO_obs.reshape(-1, 1),\
        CII_obs.reshape(-1, 1))
    L_fit = fit.predict(x.reshape(-1, 1))
    ax1.plot(x,  L_fit, color='black', linestyle='--', label='Log-linear fit to observations')

    ax1.set_ylabel('log ' + getlabel('[CII]158'))
    ax1.set_xlabel('log ' + getlabel('CO(1-0)'))
    plt.colorbar(sc,label=r'log $\langle$Z$\rangle_{\rm SFR}$ [Z$_{\rm \odot}$]')

    handles, labels = ax1.get_legend_handles_labels()
    print(labels) #   labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # handles = [handles[_] for _ in [2,4,3,5,0,6,7,1]]
    # labels = [labels[_] for _ in [2,4,3,5,0,6,7,1]]
    handles = [handles[_] for _ in [2,4,3,5,6,0,1]]
    labels = [labels[_] for _ in [2,4,3,5,6,0,1]]
    plt.legend(handles,labels,loc='lower left',fontsize=10.,frameon=True)

    ax1.set_xlim([-3,6.2])
    ax1.set_ylim([4,10])

    if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
    plt.savefig(p.d_plot + 'luminosity/CO_vs_CII%s%s.png' % (p.grid_ext,p.table_ext), dpi=300)

def morph_CII(**kwargs):
    """ Display galaxy morphology in CII-SFR diagram
    """


    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()
    
    L_line              =   np.log10(getattr(GR,'L_'+p.line+'_sun'))
    SFR                 =   np.log10(getattr(GR,'SFR'))

    fig,ax = plt.subplots(figsize=(20,16))

    for i in range(len(L_line)):
        
        im = mpimg.imread('plots/sim_data/stamps/%s%s_G%i.png' % (p.sim_name,p.sim_run,i))
        imbox = OffsetImage(im, zoom=0.02)
        ab = AnnotationBbox(imbox, (SFR[i],L_line[i]), pad=0, frameon=False)
        ax.add_artist(ab)

    ax.set_xlabel('log ' + getlabel('SFR'))
    ax.set_ylabel('log ' + getlabel(p.line))

    if not p.xlim: p.xlim = np.array([-3,4])
    if not p.ylim: 
        p.ylim = [np.median(L_line)-6,np.median(L_line)+4]
    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)

    if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
    plt.savefig(p.d_plot + 'luminosity/morph_CII_%s%s' % (p.sim_name,p.sim_run),dpi=350)

#---------------------------------------------------------------------------
### LINE RATIOS ###
#---------------------------------------------------------------------------

def NII_ratio_ne(**kwargs):
    """ Simple global [NII] ratio against n_e plot
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    line1,line2 = '[NII]122','[NII]205'
    L_line1 = getattr(GR,'L_'+line1+'_sun')
    L_line2 = getattr(GR,'L_'+line2+'_sun')
    # Get ratio where the two samples overlap:
    ratio = L_line1 / L_line2
    ne_mw = getattr(GR,'ne_mw')[ratio != 0]
    ratio = ratio[ratio != 0]
    label = '%s / %s' % (line1,line2)

    fig,ax = plt.subplots(figsize=(10,8))
    ax.set_xlabel('log ' + getlabel('ne'))
    ax.set_ylabel(label)
    ax.plot(np.log10(ne_mw), ratio, 'o', color='grey', alpha=0.7) 
    xs = np.arange(ax.get_xlim()[0],ax.get_xlim()[1],0.1)
    ax.plot(xs,aux.NII_from_logne(xs),'-b')

    if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
    plt.savefig(p.d_plot + 'luminosity/NII_ratio_ne_%s%s' % (p.sim_name,p.sim_run),dpi=300)


def line_ratio(ratio_name,**kwargs):
    """ Make a histogram of some line luminosity ratio
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    if ratio_name == 'NII':
        line1,line2 = '[NII]122','[NII]205'
        L_line1 = getattr(GR,'L_'+line1+'_sun')
        L_line2 = getattr(GR,'L_'+line2+'_sun')
        # Get ratio where the two samples overlap:
        ratio = L_line1 / L_line2
        ratio = ratio[ratio != 0]
        label = '%s / %s' % (line1,line2)

    if ratio_name == 'OICII':
        line1,line2 = '[OI]63','[CII]'
        L_line1 = getattr(GR,'L_'+line1+'_sun')
        L_line2 = getattr(GR,'L_'+line2+'_sun')
        # Get ratio where the two samples overlap:
        ratio = L_line1 / L_line2
        ratio = ratio[ratio > 1e-2]
        ratio = np.log10(ratio[ratio != 0])
        label = 'log %s / %s' % (line1,line2)

    fig,ax = plt.subplots(figsize=(10,8))
    h = ax.hist(ratio,bins=10,color='orange')

    ax.set_xlabel(label,fontsize=15)
    if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
    plt.savefig(p.d_plot + 'luminosity/ratio_%s' % ratio_name,dpi=300)

def line_ratio_per_pixel_w_hist(ratio_name='NII',quant='ne',res=0.5, plane='xy',**kwargs):
    """ Plot line ratio against another quantity per pixel in moment0 map, with vertical and horizontal histograms of distributions.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)
    gal_ob          =   gal.galaxy(p.gal_index)

    location = aux.moment0_map_location(res=res,plane=plane,gal_index=p.gal_index)

    # Load sim and cell data
    try:
        moment_0_map = np.load(location, allow_pickle=True)
        print('Found stored momentmap data')
    except:
        print('Did not find stored momentmap data - creating')
        aux.convert_cell_data_to_regular_grid(gal_index=p.gal_index)
        moment_0_map = np.load(location, allow_pickle=True)

    if ratio_name == 'NII':
        line1,line2 = '[NII]122','[NII]205'

    index1 = p.moment0_dict['L_%s' % line1]
    index2 = p.moment0_dict['L_%s' % line2]
    index3 = p.moment0_dict['m']
    index4 = p.moment0_dict['ne_mw']
    index_test = p.moment0_dict['m_HII_regions']
    line1 = []
    line2 = []
    ne_mw = []
    test = []
    m = []
    dataset= np.array(moment_0_map[:,3])
    for row in dataset:
        try:
            line1.append(row[index1])
            line2.append(row[index2])
            test.append(row[index_test])
            m.append(row[index3])
            ne_mw.append(row[index4])
        except:
            print(row)
    m = np.array(m)    
    ne_mw = np.array(ne_mw) / m 
    line1 = np.array(line1)
    line2 = np.array(line2)
    test = np.array(test)
    print(test.sum())
    ne_mw = np.array(ne_mw[line2 > 0])
    ratio = line1[line2 > 0]/line2[line2 > 0]

    x=ne_mw
    x[x==0] = np.min(x[x>0])
    fig = plt.figure(figsize=(8,8))
    gs = plt.GridSpec(2, 2, hspace = 0, wspace = 0,width_ratios = [5, 2.3], height_ratios = [2.3, 5],left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax = plt.subplot(gs[1, 0])
    
    ax_xDistribution = plt.subplot(gs[0, 0],sharex=ax)
    ax_yDistribution = plt.subplot(gs[1, 1],sharey=ax)
    #ax.scatter(np.log10(x), ratio, cmap='PuBu_r',alpha=0.3)
    ax.hexbin(np.log10(x), ratio, cmap='inferno',bins='log',mincnt=1,gridsize=60)
    
    #ax.set_xlim([1e-16,1e-1])
    ax.set_xlabel(r'$\langle$n$_e\rangle_{mw}$ [cm$^{-3}$]')
    ax.set_ylabel('[NII]122/[NII]205')
    ed_dist= ax_xDistribution.hist(np.log10(x),bins=100,align='mid',color='red',alpha=0.65)
    
    Ratio_dist= ax_yDistribution.hist(ratio,bins=100,orientation='horizontal',align='mid',color='red',alpha=0.65)
    plt.setp(ax_xDistribution.get_xticklabels(), visible=False)
    plt.setp(ax_yDistribution.get_yticklabels(), visible=False)

    # Overplot theoretical expression for [NII]-ne
    if ratio_name == 'NII':
        NII_theory = aux.NII_from_logne(np.linspace(np.log10(x).min(),np.log10(x).max(),20))
        ax.plot(np.linspace(np.log10(x).min(),np.log10(x).max(),20),NII_theory,color='grey',lw=2,alpha=0.5,label='theory')
    ax.legend()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'physics/'): os.mkdir(p.d_plot + 'physics/')    
        plt.savefig(p.d_plot + 'physics/%s_%s_G%i' % (ratio_name,quant,p.gal_index),dpi=300)

def line_ratio_per_pixel(ratio_name='NII',quant='ne',res=0.5, plane='xy',**kwargs):
    """ Plot line ratio against another quantity per pixel in moment0 map.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)
    gal_ob          =   gal.galaxy(p.gal_index)
    
    location = aux.moment0_map_location(res=res,plane=plane,gal_index=p.gal_index)
    # Load sim and cell data
    try:
        moment_0_map = np.load(location, allow_pickle=True)
        print('Found stored momentmap data')
    except:
        print('Did not find stored momentmap data - creating')
        aux.convert_cell_data_to_regular_grid(gal_index=p.gal_index)
        moment_0_map = np.load(location, allow_pickle=True)

    # for Malhar to fill int
    
    indexes = moment_0_map[-1]
    index1, index2 = int(indexes[1]), int(indexes[2])
    index1, index2
    moment_0_map =moment_0_map[:-1]
    x = moment_0_map[:,1]
    y = moment_0_map[:,2]
    
    if ratio_name == 'NII':
        line1,line2 = '[NII]122','[NII]205'
    datasrt=(['x','y','z','m','L_[NII]122','L_[NII]205','DTM','mf_H2_grid', 'mf_HII_grid', 'mf_HI_grid', 'ne_grid','ne_mw_grid','L_[CII]158', 'L_[CI]610',
       'L_[CI]370', 'L_[OI]145', 'L_[OI]63', 'L_[OIII]88', 'L_[NII]122',
       'L_[NII]205', 'L_CO(3-2)', 'L_CO(2-1)', 'L_CO(1-0)', 'L_[OIV]25'])
    dataset= np.array(moment_0_map[:,3])
    line1 = []
    line2 = []
    ne_mw_grid=[]
    m=[]

    for row in dataset:
        m.append(row[3])
        line1.append(row[4])
        line2.append(row[5])
        ne_mw_grid.append(row[11])
    m=np.array(m)    
    line1 = np.array(line1)
    line2 = np.array(line2)
    ne_mw_grid=np.array(ne_mw_grid)
    np.vectorize(line1)
    np.vectorize(line2)
    ratio=line1/line2
    
    
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    ig,ax = plt.subplots(figsize=(8,5))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    #ax.hexbin((ne_mw_grid/m),ratio,xscale='log',gridsize=150,cmap='inferno')
    ax_scatter.scatter((ne_mw_grid/m),ratio)
    
    ax.set_xscale('log')
    binwidth = 0.25
    lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-lim, lim))

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    
    ax.set_xlim([1e-16,1e-6])
    ax.set_xlabel(r'$\langle$n$_e\rangle_{mw}$ [cm$^{-3}$]')
    ax.set_ylabel('[NII]122/[NII]205')
    ax.axis('auto')

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'physics/'): os.mkdir(p.d_plot + 'physics/')    
        plt.savefig(p.d_plot + 'physics/%s_%s_G%i' % (ratio_name,quant,p.gal_index),dpi=300)

def line_ratio_per_pixel_AHIMSA(ratio_name='NII',quant='ne',phase='all',res=0.5, plane='xy',col='grey',add=False,**kwargs):
    """ Plot NII line ratio vs n_e for specific phase using moment0map pixels
    """

    fig = plt.figure(constrained_layout=True,figsize=(15,10))
    gs = fig.add_gridspec(3,2)

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)
    gal_ob          =   gal.galaxy(p.gal_index)
    
    location = aux.moment0_map_location(res=res,plane=plane,gal_index=p.gal_index)
    # Load sim and cell data
    try:
        moment_0_map = np.load(location, allow_pickle=True)
        print('Found stored momentmap data')
    except:
        print('Did not find stored momentmap data - creating')
        aux.convert_cell_data_to_regular_grid(gal_index=p.gal_index)
        moment_0_map = np.load(location, allow_pickle=True)

    cols = {'HII':'blue','HI':'orange','H2':'red'}
    for i,phase in enumerate(['HII','HI','H2']):
        ax = fig.add_subplot(gs[i,0])

        dataset= np.array(moment_0_map[:,3])

        # Do all pixels for comparison
        index1 = p.moment0_dict['L_[NII]122']
        index2 = p.moment0_dict['L_[NII]205']
        index3 = p.moment0_dict['m']
        index4 = p.moment0_dict['ne_mw']
        line1 = []
        line2 = []
        ne_mw = []
        m = []
        for row in dataset:
            try:
                line1.append(row[index1])
                line2.append(row[index2])
                m.append(row[index3])
                ne_mw.append(row[index4])
            except:
                print(row)
        m = np.array(m)    
        ne_mw = np.array(ne_mw)    
        line1 = np.array(line1)
        line2 = np.array(line2)
        # print(line2)
        ne_mw = np.array(ne_mw[line2 > 0])
        ratio = line1[line2 > 0]/line2[line2 > 0]
        ax.scatter(np.log10(ne_mw/m[line2 > 0]),ratio,color=col,label= 'All gas in moment0map pixels',alpha=0.3)

        # Now for one ISM phase
        index1 = p.moment0_dict['L_[NII]122_%s' % phase]
        index2 = p.moment0_dict['L_[NII]205_%s' % phase]
        index3 = p.moment0_dict['m_%s' % phase]
        index4 = p.moment0_dict['ne_%s_mw' % phase]
        line1 = []
        line2 = []
        ne_mw = []
        m_phase = []
        for row in dataset:
            try:
                line1.append(row[index1])
                line2.append(row[index2])
                m_phase.append(row[index3])
                ne_mw.append(row[index4])
            except:
                print(row)
        m_phase = np.array(m_phase)    
        ne_mw = np.array(ne_mw)    
        line1 = np.array(line1)
        line2 = np.array(line2)
        ne_mw = np.array(ne_mw[line2 > 0])
        ratio = line1[line2 > 0]/line2[line2 > 0]
        ax.scatter(np.log10(ne_mw/m[line2 > 0]),ratio,color=cols[phase],label= '%s gas in moment0map pixels' % phase,alpha=0.3)

        xs = np.arange(ax.get_xlim()[0],ax.get_xlim()[1],0.1)
        ax.plot(xs,aux.NII_from_logne(xs),'-b')
        if p.xlim: ax.set_xlim(p.xlim)
        if p.ylim: ax.set_ylim(p.ylim)
        ax.set_xlabel('log ' + getlabel('ne'))
        ax.set_ylabel(getlabel('NIIratio'))
        ax.legend()

    ax = fig.add_subplot(gs[:,-1])
    ax.set_ylabel('y [kpc]')
    ax.set_xlabel('x [kpc]')
    map_sim_property(prop='m',vmin=9,vmax=12.9,add=True,log=True,sim_type='simgas',ax=ax,**kwargs)
    plt.subplots_adjust(hspace = 0, wspace = 0.2)
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'lineratios/NII/'): os.mkdir(p.d_plot + 'lineratios/NII/')    
        plt.savefig(p.d_plot+'lineratios/NII/%s%s_G%i_NII_ne' % (p.sim_name,p.sim_run,p.gal_index)+'.png', dpi=200)
        

def line_ratio_per_cell(ratio_name,**kwargs):
    """ Make a histogram of line ratios per cell in ONE galaxy
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # Load sim and cell data
    gal_ob          =   gal.galaxy(p.gal_index)
    cell_data       =   gal_ob.cell_data.get_dataframe()

    if ratio_name == 'NII':
        xlim = [0.2,1.4]
        line1,line2 = '[NII]122','[NII]205'
        L_line1 = cell_data['L_'+line1].values
        L_line2 = cell_data['L_'+line2].values
        # Get ratio where the two samples overlap:
        ratio = L_line1 / L_line2
        if p.weight == 'm': weights = cell_data.m.values[(ratio >= xlim[0]) & (ratio <= xlim[1])]
        if p.weight == 'light': weights = cell_data['L_'+line2].values[(ratio >= xlim[0]) & (ratio <= xlim[1])]
        ratio = ratio[(ratio >= xlim[0]) & (ratio <= xlim[1])]
        label = '%s / %s' % (line1,line2)

    if ratio_name == 'OICII':
        xlim = [0,1]
        line1,line2 = '[OI]63','[CII]'
        L_line1 = cell_data['L_'+line1].values
        L_line2 = cell_data['L_'+line2].values
        # Get ratio where the two samples overlap:
        ratio = L_line1 / L_line2
        ratio = ratio[ratio > 1e-2]
        if p.weight == 'm': weights = cell_data.m.values[(ratio >= xlim[0]) & (ratio <= xlim[1])]
        if p.weight == 'light': weights = cell_data['L_'+line1].values[(ratio >= xlim[0]) & (ratio <= xlim[1])]
        ratio = np.log10(ratio[ratio != 0])
        label = 'log %s / %s' % (line1,line2)

    if p.add:
        ax = plt.gca()
    else:
        fig,ax = plt.subplots(figsize=(8,6))
    if p.weight == '':
        h = ax.hist(ratio,bins=p.bins,color=p.color,alpha=0.6,density=True)
        ax.set_ylabel('Number of cells',fontsize=15)
    if p.weight != '':
        h = ax.hist(ratio,bins=p.bins,color=p.color,alpha=0.6,weights=weights,density=True)
        if p.weight == 'm':
            ax.set_title('Mass-weighted distribution',fontsize=15)
        if p.weight == 'light':
            ax.set_title('Luminosity-weighted (%s) distribution' % line2,fontsize=15)

    # Overplot global line ratio
    GR              =   glo.global_results()
    ratio           =   getattr(GR,'L_'+line1+'_sun')[p.gal_index] / getattr(GR,'L_'+line2+'_sun')[p.gal_index]
    # print(ratio)
    # print(np.sum(cell_data['L_'+line1].values)/np.sum(cell_data['L_'+line2].values))
    ax.plot([ratio,ratio],ax.get_ylim(),'--',c=p.color)

    # ax.legend()
    ax.set_xlim(xlim)
    # ax.set_yscale('log')
    ax.set_xlabel(label,fontsize=15)
    ax.set_ylabel('Density of cells',fontsize=15)
    if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
    plt.savefig(p.d_plot + 'luminosity/res_ratio_%s_%s' % (ratio_name,p.weight),dpi=300)

#---------------------------------------------------------------------------
### MAPS ###
#---------------------------------------------------------------------------

def moment0_map(gal_index,quant='m', res=0.5, plane='xy', units='Jy', **kwargs):
    """
    Purpose
    ---------
    Makes moment0 map of a specific quantity.

    Parameters
    ----------
    gal_index: int
        Galaxy index, default: 0

    quant: str
        The quantity to be mapped, default: 'm' (mass)
        
    res: float
        Pixel resolution in kpc

    plane: str
        Plane to project to (xy, xz, yz)
        
    units: str
        Units in which the maps will be created (Jy, L_0), default: 'Jy'
    """
    
    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)
    #print('TEST, fixing R_max = 60')
    p.gal_index = gal_index
    # p.R_max = 60
    
    location = aux.moment0_map_location(res=res,plane=plane,gal_index=p.gal_index)
        
    # Getting matrix with projected emmision values: 
    #momentmap = np.load(location, allow_pickle=True)
    #pdb.set_trace()
    if p.ow:
        print('Overwrite is on - creating')
        aux.convert_cell_data_to_regular_grid(res=res, plane=plane, gal_index=p.gal_index)
        momentmap = np.load(location, allow_pickle=True)
    #try:
    #    momentmap = np.load(location, allow_pickle=True)
    #    print('Found stored momentmap data for %i' % p.gal_index)
    #    print(location)
    #    if p.ow:
    #        print('But overwrite is on - creating')
    #        aux.convert_cell_data_to_regular_grid(res=res, plane=plane, gal_index=p.gal_index)
    #        momentmap = np.load(location, allow_pickle=True)
    #except:
    #    print('Did not find stored momentmap data for %i - creating' % p.gal_index)
    #    aux.convert_cell_data_to_regular_grid(res=res, plane=plane, gal_index=p.gal_index)
    #    momentmap = np.load(location, allow_pickle=True)
    
    n = momentmap[-1]
    momentmap = momentmap[:-1]
    indexes = momentmap[-1]
    index1, index2 = int(indexes[1]), int(indexes[2])

    # Getting the desired quantity to create the momentmap:
    dictionary = p.moment0_dict

    num = dictionary[quant]
    lumus = np.array(momentmap[:,3])
    lum = []
    mass = []
    metal = []
    for prop in lumus:
        if (quant == 'Z') | (quant == 'G0') | (quant == 'ne_mw') | (quant == 'Te_mw') | (quant == 'Tk_mw'):
            lum.append(prop[num]/prop[0])
        else:
            lum.append(prop[num])
    lum = np.array(lum)

    if 'L_' in quant:
        print('Sum over %s image: %.2f Lsun' % (quant,np.sum(lum)*6))
        print('Or: %.2f K km/s pc^2' % (aux.Lsun_to_K_km_s_pc2(np.sum(lum)*6,quant.replace('L_',''))))
        print('Or: %.2f K km/s pc^2' % (aux.Lsun_to_K_km_s_pc2(1.8e8,quant.replace('L_',''))))
        lum = lum / (res**2)
    
    # Converting to Jy*km/s / kpc^2 units:
    if units == 'Jy':
        if 'L_' in quant:
            quant_name = quant.replace('L_','')
            frequencies = p.freq
        
            z = p.zred
            D = 10   # Mpc (Luminosity Distance)
            freq = frequencies[quant_name]
        
            lum = lum*(1+z) / (1.04e-3 * D**2 * freq)
            # Soloman et al. 1997
   
    # Creating momentmaps:
    ax1,ax2 = momentmap[:, 1], momentmap[:, 2]
    
    nrows, ncols = int(n[1]), int(n[2])
    grid = lum.reshape((nrows, ncols))
    #pdb.set_trace()
    # grid = np.flipud(grid)
    # normal = mpl.colors.Normalize(vmin = min(lum), vmax = max(lum))

    # Setting 0 values to something very low
    print(len(lum))
    print(len(lum[lum == 0]))
    grid[grid == 0] = 1e-30
    grid[np.isnan(grid)] = 1e-30

    # Default min,max values
    if p.log: grid = np.log10(grid)
    if (not p.vmin) : 
        p.vmin = np.max(grid)/1e5
        if p.log: p.vmin = np.max(grid) - 5
    if (not p.vmax) : 
        p.vmax = 5*np.max(grid)
        if p.log: p.vmax = np.max(grid)
    
    if quant == 'Z':
        p.vmin = 0.05
        p.vmax = 3

    if p.add:
        fig,ax = plt.gcf(),p.ax
    else:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_axes([0.15, 0.15, 0.8, 0.8]) 
        ax.axis('equal')

    if not p.R_max:
        gal_ob = gal.galaxy(p.gal_index)
        p.R_max = gal_ob.R_max
    grid = np.flipud(grid)    
    if p.rotate:
        grid = np.rot90(grid)
        grid = np.rot90(grid)
    gal_ob          =   gal.galaxy(p.gal_index)
    #cell_data       =   gal_ob.cell_data.get_dataframe()
    #extent          =   np.max(np.abs(cell_data[['x','y','z']].values))
    if p.R_max:
        extent = 1*p.R_max
    else:
        extent = 50
    cs = ax.imshow(grid, extent=(-extent, extent, -extent, extent),\
                vmin=p.vmin, vmax=p.vmax, interpolation='nearest', cmap=p.cmap)
    print(extent)
    # Add half-light radius
    x_axis = np.linspace(-extent,extent,grid.shape[0])
    y_axis = np.linspace(-extent,extent,grid.shape[1])
    x,y = np.meshgrid(x_axis,y_axis)
    r = np.sqrt(x**2 + y**2)
    r_bins = np.linspace(0,r.max(),200)
    L_bins = np.zeros(len(r_bins)-1)
    l0 = 0
    for i in range(len(r_bins)-1):
        L_bins[i] = np.sum(10.**grid[(r < r_bins[i+1])])
    R_half = r_bins[1::][L_bins >= 0.5*L_bins.max()][0]
    print('R_half: ',R_half)
    circle = plt.Circle((0,0),R_half,ec='green',fc=None,fill=False,lw=3,ls='--')
    ax.add_patch(circle)

    #if p.R_max: extent = p.R_max
    print(p.R_max,extent)
    ax.set_xlim([-1.1*extent,1.1*extent])
    ax.set_ylim([-1.1*extent,1.1*extent])


    if num == 0:
        #plt.title('mass density')
        labels = 'log surface density (M$_{\odot}$ / kpc$^2$)'        
    if 'L_' in quant:
        #plt.title(quant + ' density')
        if units == 'Jy':
            labels = 'Jy${\cdot}$km/s / kpc$^2$'
        else:
            labels = 'log surface brightness density (L$_{\odot}$ / kpc$^2$)'
    if quant == 'Z': 
        labels = 'log Z (Z$_{\odot}$)'
    if quant == 'FUV': 
        labels = 'log FUV flux (G$_{0}$)'

    if not p.add: plt.xlabel(plane[0]+' [kpc]')
    if not p.add: plt.ylabel(plane[1]+' [kpc]')

    formatter = mpl.ticker.LogFormatterExponent(10, labelOnlyBase=False, minor_thresholds=(100,20))
    if p.legend: 
        if not p.label: labels = ''
    cbar = fig.colorbar(cs, label=labels, pad=0, shrink=0.85)#0.5)#
    
    if p.savefig:
        plt.tight_layout()
        if not os.path.isdir(p.d_plot + 'moment0/'): os.mkdir(p.d_plot + 'moment0/')    
        plt.savefig(p.d_plot + 'moment0/moment0_%i_%s%s' % (p.gal_index,p.sim_name,p.sim_run) + '_' + plane + '_res' + str(res) +'_'+ quant.replace('(','').replace(')','') + '.png',dpi=500)

def line_ratio_map(quant1='L_[NII]122', quant2='L_[NII]205', ContourFunct='ne_mw', res=0.5, plane='xy', units='Jy', **kwargs):
    """
    Purpose 
    -------
    Makes line ratio map of a specific quantity.

    Parameters
    ----------
    gal_index: int
        Galaxy index, default: 0

    quant1: str
        The first line of the line ratio, default: 'L_[NII]122'
       
    quant2: str
        The second line of the line ratio, default: 'L_[NII]205'

    res: float
        Pixel resolution in kpc

    plane: str
        Plane to project to (xy, xz, yz)
        
    units: str
        Units in which the maps will be created (Jy, L_0), default: 'Jy'
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)
    GR =   glo.global_results()
    location = aux.moment0_map_location(res=res,plane=plane,gal_index=p.gal_index)

    try:
        momentmap = np.load(location, allow_pickle=True)
        print('Found stored momentmap data')
    except:
        print('Did not find stored momentmap data - creating')
        aux.convert_cell_data_to_regular_grid(res=res, plane=plane, gal_index=p.gal_index)
        momentmap = np.load(location, allow_pickle=True)

    indexes = momentmap[-1]
    index1, index2 = int(indexes[1]), int(indexes[2])
    
    momentmap = momentmap[:-1] 
    
    dictionary = p.moment0_dict
    
    num1=dictionary[quant1]
    num2=dictionary[quant2]
    num =dictionary[ContourFunct]
    x = momentmap[:,1]
    y = momentmap[:,2]
    lumus = np.array(momentmap[:,3])
    
    line1=[]
    line2=[]
    Contour_Function=[]
    m=[]
    for row in lumus:
        
        line1.append(row[num1])
        line2.append(row[num2])
     
          
        if ContourFunct == 'ne_mw':
            if row[dictionary['m']] == 0:
            
                Contour_Function.append(0)
            else:
                Contour_Function.append(row[num]/row[dictionary['m']])
        else:
            Contour_Function.append(row[num])  
   
    line1 = np.array(line1)
    line2 = np.array(line2)
    Contour_Function = np.array(Contour_Function)
    
    ratio = np.divide(line1, line2, out=np.zeros_like(line1), where=line2!=0)
    
    ratio = ratio.reshape(index1, index2)
    x = x.reshape(index1, index2)
    y = y.reshape(index1, index2)
    line1 = line1.reshape(index1, index2)
    line2 = line2.reshape(index1, index2)
    #pdb.set_trace()
    Contour_Function=Contour_Function.reshape(index1,index2)
    
    ratio[ratio==0] = np.min(ratio[ratio>0])
    Contour_Function[Contour_Function==0] = 1e-30


    if p.add:
        fig,ax = plt.gcf(),p.ax #plot already available 
    else:
        fig, ax = plt.subplots(figsize=(10,8))
        plt.subplots_adjust(left=0.1,bottom=0.2,right=0.8)
    
    if p.log: cs = ax.pcolormesh(x, y, np.log10(ratio), cmap=plt.cm.viridis, vmin=np.log10(ratio).max()-1.5, shading='auto')
    if not p.log: cs = ax.pcolormesh(x, y, ratio, cmap=plt.cm.viridis, vmin=ratio.max()/100, shading='auto')

    if not p.add:

        ax.set_title('Line Ratio map of ' + quant1.replace('L_','') + "/" + quant2.replace('L_',''))
        ax.set_xlabel('x [kpc]')
        ax.set_ylabel('y [kpc]')
        levels = np.arange(np.min(np.log10(Contour_Function[Contour_Function > 1e-30])).round(), np.max(np.log10(Contour_Function)).round(), 1)
        cr=ax.contour(x,y,np.log10(Contour_Function),cmap=plt.cm.plasma, levels=levels)

    if p.add:labels:''
    cbaxes=fig.add_axes([.15, 0.09, 0.6, 0.027])
    cbar=fig.colorbar(cr,cax=cbaxes,orientation='horizontal', label= 'log '+ getlabel(ContourFunct))
    cbaxes2 = fig.add_axes([0.82, 0.24, 0.027, 0.6])
    if p.log: fig.colorbar(cs, cax=cbaxes2, label= 'log ' + quant1.replace('L_','') + " / " + quant2.replace('L_','') )
    if not p.log: fig.colorbar(cs, cax=cbaxes2, label= quant1.replace('L_','') + " / " + quant2.replace('L_','') )
    if p.R_max:
        ax.set_xlim([-p.R_max,p.R_max])
        ax.set_ylim([-p.R_max,p.R_max])
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'lineratios/'): os.mkdir(p.d_plot + 'lineratios/')    
        plt.savefig(p.d_plot+'lineratios/map_%s%s_%i_%s_%s' % (p.sim_name,p.sim_run,p.gal_index,quant1.replace('L_',''),quant2.replace('L_',''))+ '_' + plane + '_res' + str(res) +'.png', facecolor='w', dpi=500)
        
def three_moment0_maps(gal_indices,lines,**kwargs):
    """ Make moment0 panels for 3 selected lines of 3 galaxies
    """
    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    fig = plt.figure(figsize=(17,14),constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=3, ncols=3, wspace=0.05, hspace=0.02)

    if not p.R_max: p.R_max = [60.]*3

    rotate = False
    for row_i,gal_index in enumerate(gal_indices):
        line_i = 0
        for quant in lines:
            legend = True
            label = False
            if quant == lines[-1]: label = True
            #if line_i == len(lines)-1: legend = True
            ax1 = fig.add_subplot(gs1[row_i,line_i])
            moment0_map(gal_index=gal_index,cmap=p.cmap,quant=quant,add=True,ax=ax1,R_max=p.R_max[row_i],legend=legend,label=label)
            # Make a size indicator
            ax1.set_xlim([-p.R_max[row_i],p.R_max[row_i]]); ax1.set_ylim([-p.R_max[row_i],p.R_max[row_i]])
            ax1.plot([p.R_max[row_i]*(1-0.35),p.R_max[row_i]*(1-0.35)+10],[p.R_max[row_i]*(-1+0.15),p.R_max[row_i]*(-1+0.15)],lw=4,color='white')
            ax1.text(p.R_max[row_i]*(1-0.45),p.R_max[row_i]*(-1+0.25),'10 kpc',color='white',fontsize=14)
            # Remove axes ticks
            ax1.tick_params(axis='x',which='both',labelbottom=False,bottom=False,top=False)
            ax1.tick_params(axis='y',which='both',labelleft=False,bottom=False,top=False)     
            line_i += 1
            ax1.text(p.R_max[row_i]*(-1+0.15),p.R_max[row_i]*(1-0.2),quant.replace('L_',''),color='white',fontsize=18)
        # s = segs

    gs1.update(top=0.98,bottom=0.02,left=0.02,right=0.93)
    #fig.text(0.97,0.5, 'log surface brightness density (Jy${\cdot}$km/s / kpc$^2$)', va='center', ha='center', fontsize=22, rotation='vertical')
    plt.tight_layout()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'pretty/'): os.mkdir(p.d_plot + 'pretty/')
        plt.savefig('plots/pretty/moment0_maps.png',format='png',dpi=200)

def three_mass_FUV_maps(gal_indices,**kwargs):
    """ Make panels of 3 galaxies
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    fig = plt.figure(figsize=(15,14.5),constrained_layout=False)
    gs1 = fig.add_gridspec(nrows=3, ncols=3, wspace=0.0, hspace=0.0)

    rotate = False
    for row_i,gal_index in enumerate(gal_indices):

        R_max = p.R_max[row_i]

        ax1 = fig.add_subplot(gs1[row_i, 0])
        m = map_sim_property(add=True,ax=ax1,gal_index=gal_index, \
                         prop='m',R_max=R_max,vmin=0,vmax=7,\
                        pix_size_kpc=0.5,sim_type='simgas',cmap='viridis',log=True,colorbar=False,rotate=rotate,text=p.text)
        frame = plt.gca()
        #if row_i != 2: frame.axes.get_xaxis().set_visible(False)
        if row_i != 2: ax1.set_xlabel('')
        if row_i == 0:
            cbaxes = fig.add_axes([0.05, 0.93, 0.25, 0.01]) 
            cb = plt.colorbar(m, orientation='horizontal', cax = cbaxes)
            cbaxes.xaxis.set_ticks_position('top')
            cb.ax.set_title("log $\Sigma_{\mathrm{gas}}$ [M$_{\odot}$ kpc$^{-2}$]")
        # Make a size indicator
        #ax1.plot([p.R_max-15,p.R_max-5],[-p.R_max+5,-p.R_max+5],lw=4,color='white')
        #ax1.text(p.R_max-16,-p.R_max+7,'10 kpc',color='white',fontsize=12)
        # Remove axes ticks
        ax1.tick_params(axis='x',which='both',labelbottom=False)
        ax1.tick_params(axis='y',which='both',labelleft=False)

        ax1 = fig.add_subplot(gs1[row_i, 1])
        m = star_map(add=True,ax=ax1,R_max=R_max,vmin=6,vmax=9,\
            gal_index=gal_index,colorbar=False,rotate=rotate)
        frame = plt.gca()
        #if row_i != 2: frame.axes.get_xaxis().set_visible(False)
        if row_i != 2: ax1.set_xlabel('')
        frame.axes.get_yaxis().set_visible(False)
        if row_i == 0:
            cbaxes = fig.add_axes([0.375, 0.93, 0.25, 0.01]) 
            cb = plt.colorbar(m, orientation='horizontal', cax = cbaxes) 
            cbaxes.xaxis.set_ticks_position('top')
            cb.ax.set_title("log stellar age [yr]")
        # Make a size indicator
        #ax1.plot([p.R_max-15,p.R_max-5],[-p.R_max+5,-p.R_max+5],lw=4,color='k')
        #ax1.text(p.R_max-16,-p.R_max+7,'10 kpc',color='k',fontsize=12)
        # Remove axes ticks
        ax1.tick_params(axis='x',which='both',labelbottom=False)
        ax1.tick_params(axis='y',which='both',labelleft=False)

        ax1 = fig.add_subplot(gs1[row_i, 2])
        m = FUV_map(add=True,ax=ax1,gal_index=gal_index,R_max=R_max,vmin=-10,vmax=3,select=p.select,cmap='twilight',colorbar=False,rotate=rotate)
        frame = plt.gca()
        #if row_i != 2: frame.axes.get_xaxis().set_visible(False)
        if row_i != 2: ax1.set_xlabel('')
        frame.axes.get_yaxis().set_visible(False)
        if row_i == 0:
            cbaxes = fig.add_axes([0.69, 0.93, 0.25, 0.01]) 
            cb = plt.colorbar(m, orientation='horizontal', cax = cbaxes) 
            cbaxes.xaxis.set_ticks_position('top')
            cb.ax.set_title('FUV flux [W/m$^2$/arcsec$^2$]')
        # Make a size indicator
        # if row_i == 2:
            # print('Adding size indicator')
            #ax1.text(p.R_max-16,-p.R_max+7,'10 kpc',color='w',fontsize=12)
            #ax1.plot([p.R_max-15,p.R_max-5],[-p.R_max+5,-p.R_max+5],lw=4,color='w')
        # else:
        ax1.text(R_max-16,-R_max+7,'10 kpc',color='k',fontsize=12)
        ax1.plot([R_max-15,R_max-5],[-R_max+5,-R_max+5],lw=4,color='k')
        # Remove axes ticks
        ax1.tick_params(axis='x',which='both',labelbottom=False)
        ax1.tick_params(axis='y',which='both',labelleft=False)

        # s = segs
    gs1.update(top=0.92,bottom=0.02,left=0.02,right=0.98)

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'pretty/'): os.mkdir(p.d_plot + 'pretty/')
        plt.savefig('plots/pretty/mass_FUV_maps_%s%s.png' % (p.sim_name,p.sim_run),format='png',dpi=200)

#---------------------------------------------------------------------------
### CELL DATA PHYSICAL PROPERTIES ###
#---------------------------------------------------------------------------

def Te_ne_P_panel(**kwargs):
    """ Make histrograms of Te, ne and pressure for ionized ISM
    """

    GR                      =   glo.global_results()
    gal_indices             =   np.arange(GR.N_gal)

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    for gal_index in gal_indices:
        fig = plt.figure(figsize=(15,7),constrained_layout=False)
        gal_ob                  =   gal.galaxy(GR=GR, gal_index=gal_index)
        cell_data               =   gal_ob.cell_data.get_dataframe()

        gs1 = fig.add_gridspec(nrows=1, ncols=3, wspace=0.0, hspace=0.0)

        ax = fig.add_subplot(gs1[0,0])
        h           =   np.histogram(np.log10(cell_data.Te_mw),bins=100)
        bin_size    =   (h[1][1]-h[1][0])/2
        ax.fill_between(h[1][0:-1] + bin_size,h[0],color='orange', step='pre',alpha=0.6,label='G%i' % gal_index)
        ax.set_xlabel('log mass-weighted T$_{e}$ per cell')
        ax.set_ylabel('Mass fraction')

        ax = fig.add_subplot(gs1[0,1])
        h           =   np.histogram(np.log10(cell_data.ne_mw_grid),bins=100)
        bin_size    =   (h[1][1]-h[1][0])/2
        ax.fill_between(h[1][0:-1] + bin_size,h[0],color='orange', step='pre',alpha=0.6,label='G%i' % gal_index)
        ax.set_xlabel('log mass-weighted n$_{e}$ per cell')
        ax.set_ylabel('Mass fraction')

        ax = fig.add_subplot(gs1[0,2])
        h           =   np.histogram(np.log10(cell_data.P_HII),bins=100)
        bin_size    =   (h[1][1]-h[1][0])/2
        ax.fill_between(h[1][0:-1] + bin_size,h[0],color='orange', step='pre',alpha=0.6,label='G%i' % gal_index)
        ax.set_xlabel('log mass-weighted P$_{HII}$ per cell')
        ax.set_ylabel('Mass fraction')

        plt.tight_layout()
        if p.savefig:
            if not os.path.isdir(p.d_plot + 'cell_data/pressure/'): os.mkdir(p.d_plot + 'cell_data/pressure/')
            plt.savefig(p.d_plot + 'cell_data/pressure/G%i' % gal_index, dpi=250, facecolor='w')
            plt.close()

#---------------------------------------------------------------------------
### MOVIES ###
#---------------------------------------------------------------------------

def movie(**kwargs):
    """ Make movie rotating around galaxy

    See http://zulko.github.io/blog/2014/11/29/data-animations-with-python-and-moviepy/

    """

    print("let's make a movie!")

    GR                      =   glo.global_results()

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    gal_ob                  =   gal.galaxy(GR=GR, gal_index=p.gal_index)
    celldata                =   gal_ob.cell_data.get_dataframe()

    # Set up grid
    known_points = np.array([celldata.x.values, celldata.y.values, celldata.z.values]).T
    values = celldata[p.prop].values
    values[values == 0] = 1e-6
    values = np.log10(values)
    X, Y, Z = np.meshgrid(np.arange(-gal_ob.radius,gal_ob.radius), np.arange(-gal_ob.radius,gal_ob.radius), np.arange(-gal_ob.radius,gal_ob.radius))

    grid = griddata(known_points, values, (X, Y, Z))

    # MAKE A FIGURE WITH MAYAVI

    duration = 1 # duration of the animation in seconds (it will loop)

    print('Now setting up figure')

    fig = mlab.figure(size=(200, 200), bgcolor=(1,1,1))
    mlab.contour3d(grid, contours=10, transparent=True, figure=fig)

    # ANIMATE THE FIGURE WITH MOVIEPY, WRITE AN ANIMATED GIF

    mlab.view(azimuth= 360, distance=200) # starting at this camera angle

    duration = 4 # duration of the animation in seconds (it will loop)

    def make_frame(t):
        """ Generates and returns the frame for time t. """
        mlab.view(azimuth= 100*t/duration, distance=100) # roll camera angle
        f = mlab.gcf()
        f.scene._lift()
        return mlab.screenshot(antialiased=True) # return a RGB image

    animation = mpy.VideoClip(make_frame, duration=duration)#.resize(0.5)
    # Video generation takes 10 seconds, GIF generation takes 25s
    animation.write_videofile("plots/movies/test.mp4", fps=20)
    # animation.write_gif("wireframe.gif", fps=20)

#---------------------------------------------------------------------------
### EXTRA FUNCTIONS ###
#---------------------------------------------------------------------------

def getlabel(foo):
    '''Gets axis labels for plots
    '''

    if foo == 'z': return 'z'
    if foo == 'x': return 'x position [kpc]'
    if foo == 'y': return 'y position [kpc]'
    if foo == 'z': return 'y position [kpc]'
    if foo == 'vx': return 'v$_x$ [km s$^{-1}$]'
    if foo == 'vy': return 'v$_y$ [km s$^{-1}$]'
    if foo == 'vz': return 'v$_z$ [km s$^{-1}$]'
    if foo == 'DTM': return 'DTM ratio'
    if foo == 'lDTM': return 'log DTM ratio'
    if foo == 'nH': return '$n_{\mathrm{H}}$ [cm$^{-3}$]'
    if foo == 'lnH': return 'log($n_{\mathrm{H}}$ [cm$^{-3}$])'
    if foo == 'NH': return '$N_{\mathrm{H}}$ [cm$^{-2}$]'
    if foo == 'lNH': return 'log($N_{\mathrm{H}}$ [cm$^{-2}$])'
    if foo == 'lhden': return 'log($n_{\mathrm{H}}$ [cm$^{-3}$])'
    if foo == 'nHmw': return r'$\langle n_{\mathrm{H}}\rangle_{\mathrm{mass}}$'+' [cm$^{-3}$]'
    if foo == 'nH_mw': return r'$\langle n_{\mathrm{H}}\rangle_{\mathrm{mass}}$'+' [cm$^{-3}$]'
    if foo == 'nH_pdr': return 'H density of PDR gas [cm$^{-3}$]'
    if foo == 'R_pdr': return 'Size of PDR gas [pc]'
    if foo == 'Rgmc': return 'R$_{\mathrm{GMC}}$ [pc]'
    if foo == 'lRgmc': return 'log(R$_{\mathrm{GMC}}$ [pc])'
    if foo == 'f_HI': return 'f$_{\mathrm{[HI]}}$'
    # if foo == 'f_HI1': return 'f$_{\mathrm{[HI]}}$ before'
    if foo == 'f_H2': return 'f$_{\mathrm{H2}}$'
    if foo == 'f_neu': return 'f$_{\mathrm{neu}}$'
    if foo == 'Tk': return '$T_{\mathrm{k}}$ [K]'
    if foo == 'Z': return '$Z$ [Z$_{\odot}$]'
    if foo == 'Z_star': return '$Z_{star}$ [Z$_{\odot}$]'
    if foo == 'lZ': return 'log($Z$ [Z$_{\odot}$])'
    if foo == 'Zmw': return r"$\langle Z'\rangle_{\mathrm{mass}}$"
    if foo == 'Zsfr': return r"$\langle Z'\rangle_{\mathrm{SFR}}$"
    if foo == 'Zstar': return r"$\langle Z'\rangle_{\mathrm{stars}}$"
    if foo == 'lZ_star': return 'log ' + r"$Z_{\mathrm{star}}$ [Z$_{\odot}$]"
    if foo == 'lZsfr': return r"log($\langle Z'\rangle_{\mathrm{SFR}}$ [$Z_{\odot}$])"
    if foo == 'SFR': return 'SFR [M$_{\odot}$yr$^{-1}$]'
    if foo == 'SFR_density': return 'SFR density [M$_{\odot}$yr$^{-1}$/kpc$^{3}$]'
    if foo == 'nSFR': return 'SFR density [M$_{\odot}$yr$^{-1}$/kpc$^{3}$]'
    if foo == 'lSFR_density': return 'log SFR density [M$_{\odot}$yr$^{-1}$/kpc$^{-3}$]'
    if foo == 'lSFR': return 'log(SFR [M$_{\odot}$yr$^{-1}$])'
    if foo == 'sSFR': return 'sSFR [yr$^{-1}$]'
    if foo == 'SFRsd': return '$\Sigma$$_{\mathrm{SFR}}$ [M$_{\odot}$/yr/kpc$^{2}$]'
    if foo == 'lSFRsd': return 'log($\Sigma$$_{\mathrm{SFR}}$ [M$_{\odot}$/yr kpc$^{-2}$])'
    if foo == 'h': return 'Smoothing length $h$ [kpc]'
    if foo == 'm': return 'Total mass [M$_{\odot}$]'
    if foo == 'cell_volume': return 'Cell volume [pc$^3$]'
    if foo == 'lm': return 'log(Total mass [M$_{\odot}$])'
    if foo == 'Ne': return 'Electron fraction'
    if foo == 'ne': return 'n$_{e}$ [cm$^{-3}$]'
    if foo == 'ne_mw': return 'n$_{e}$ [cm$^{-3}$]'
    if foo == 'Mgmc': return '$m_{\mathrm{GMC}}$ [M$_{\odot}$]'
    if foo == 'm_mol': return '$m_{\mathrm{mol}}$ [M$_{\odot}$]'
    if foo == 'm_dust': return '$m_{\mathrm{dust}}$ [M$_{\odot}$]'
    if foo == 'M_dust': return 'M$_{\mathrm{dust}}$ [M$_{\odot}$]'
    if foo == 'M_star': return 'M$_{\mathrm{*}}$ [M$_{\odot}$]'
    if foo == 'M_gas': return 'M$_{\mathrm{gas}}$ [M$_{\odot}$]'
    if foo == 'M_ISM': return 'M$_{\mathrm{ISM}}$ [M$_{\odot}$]'
    if foo == 'lM_ISM': return 'log(M$_{\mathrm{ISM}}$ [M$_{\odot}$])'
    if foo == 'G0': return "FUV flux [G$_{0}$]"
    if foo == 'G0_mw': return r"$\langle$F$_{\mathrm{FUV}}\rangle_{\mathrm{mass}}$ [G$_{0}$]"
    if foo == 'lG0': return "log G$_{0}$ [G$_{0}$]"
    if foo == 'CR': return "$\zeta_{\mathrm{CR}}$ [s$^{-1}$]"
    if foo == 'P_ext': return "$P_{\mathrm{ext}}$ [K cm$^{-3}$]"
    if foo == 'lP_ext': return "log($P_{\mathrm{ext}}$ [K cm$^{-3}$])"
    if foo == 'lP_extmw': return r"log($\langle P_{\mathrm{ext}}\rangle_{\mathrm{mass}}$)"
    if foo == 'age': return "Age [Gyr]"
    if foo == 'lage': return "log(Age [Gyr])"
    if foo == 'C': return "C mass fraction I think?"
    if foo == 'O': return "O mass fraction I think?"
    if foo == 'Si': return "Si mass fraction I think?"
    if foo == 'Fe': return "Fe mass fraction I think?"
    if foo == 'FUV': return "G$_0$ [0.6 Habing]"
    if foo == 'lFUV': return "log(G$_0$ [0.6 Habing])"
    if foo == 'FUVmw': return r"$\langle$G$_{\mathrm{0}}\rangle_{\mathrm{mass}}$ [0.6 Habing]"
    if foo == 'FUV_amb': return "G$_0$ (ambient) [0.6 Habing]"
    if foo == 'nH_DNG': return "H density of DNG [cm$^{-3}$]"
    if foo == 'dr_DNG': return "Thickness of DNG layer [pc]"
    if foo == 'm_DIG': return "m$_{\mathrm{DIG}}$ [M$_{\odot}$]"
    if foo == 'nH_DIG': return "n$_{\mathrm{H,DIG}}$ [cm$^{-3}$]"
    if foo == 'R': return "$R$ [kpc]"
    if foo == 'vel_disp_gas': return r"$\sigma_{\mathrm{v}}$ of gas [km s$^{-1}$]"
    if foo == 'vel_disp_cloud': return r"$\sigma_{\mathrm{v}}$ on cloud scales [km s$^{-1}$]"
    if foo == 'sigma_gas': return r"$\sigma_{\mathrm{v,\perp}}$ of gas [km s$^{-1}$]"
    if foo == 'sigma_star': return r"$\sigma_{\mathrm{v,\perp}}$ of star [km s$^{-1}$]"
    if foo == 'surf_gas': return "$\Sigma_{\mathrm{gas}}$ [M$_{\odot}$ pc$^{-2}$]"
    if foo == 'surf_star': return "$\Sigma_{\mathrm{*}}$ [M$_{\odot}$ kpc$^{-2}$]"
    if foo == 'S_CII': return 'S$_{\mathrm{[CII]}}$ [mJy]'
    if foo == 'x_e': return 'Electron fraction [H$^{-1}$]'
    if foo == 'f_CII': return '(mass of carbon in CII state)/(mass of carbon in CIII state) [%]'
    if foo == 'f_ion': return 'Ionized gas mass fraction [%]'
    if foo == 'f_neu': return 'Neutral gas mass fraction [%]'
    if foo == 'f_gas': return 'Gas mass fraction M$_{\mathrm{gas}}$/(M$_{\mathrm{gas}}$+M$_{\mathrm{*}}$) [%]'
    if foo == 'f_CII_neu': return 'f_${CII,neutral}$ [%]'
    if foo == 'F_FUV_W_m2_mi': return 'FUV flux [W/m$^2/\mu$m]'
    if foo == 'F_FUV_W_m2': return 'FUV flux [W/m$^2$]'
    if foo == 'F_NIR_W_m2': return 'NIR flux [W/m$^2$]'
    if foo == 'F_FUV_Habing': return 'FUV flux [Habing]'
    if foo == 'Mach': return 'Mach number'

    if foo == '[CII]158': return 'L$_{\mathrm{[CII]}}$ [L$_{\odot}$]'
    if foo == 'l[CII]158': return 'log(L$_{\mathrm{[CII]}}$ [L$_{\odot}$])'
    if foo == '[OI]63': return 'L$_{\mathrm{[OI]}\,63\mu\mathrm{m}}$ [L$_{\odot}$]'
    if foo == 'l[OI]63': return 'log(L$_{\mathrm{[OI]}\,63\mu\mathrm{m}}$ [L$_{\odot}$])'
    if foo == '[OI]145': return 'L$_{\mathrm{[OI]}\,145\mu\mathrm{m}}$ [L$_{\odot}$]'
    if foo == 'l[OI]145': return 'log(L$_{\mathrm{[OI]}\,145\mu\mathrm{m}}$ [L$_{\odot}$])'
    if foo == '[OIII]88': return 'L$_{\mathrm{[OIII]}\,88\mu\mathrm{m}}$ [L$_{\odot}$]'
    if foo == 'l[OIII]88': return 'log(L$_{\mathrm{[OIII]}\,88\mu\mathrm{m}}$ [L$_{\odot}$])'
    if foo == '[NII]122': return 'L$_{\mathrm{[NII]122}}$ [L$_{\odot}$]'
    if foo == 'l[NII]122': return 'log(L$_{\mathrm{[NII]122}}$ [L$_{\odot}$])'
    if foo == '[NII]205': return 'L$_{\mathrm{[NII]205}}$ [L$_{\odot}$]'
    if foo == 'l[NII]205': return 'log(L$_{\mathrm{[NII]205}}$ [L$_{\odot}$])'
    if foo == 'CO(1-0)': return 'L$_{\mathrm{CO(1-0)}}$ [L$_{\odot}$]'
    if foo == 'CO(2-1)': return 'L$_{\mathrm{CO(2-1)}}$ [L$_{\odot}$]'
    if foo == 'CO(3-2)': return 'L$_{\mathrm{CO(3-2)}}$ [L$_{\odot}$]'    
    if foo == 'lCO(1-0)': return 'log(L$_{\mathrm{CO(1-0)}}$ [L$_{\odot}$])'
    if foo == 'lCO(2-1)': return 'log(L$_{\mathrm{CO(2-1)}}$ [L$_{\odot}$])'
    if foo == 'lCO(2-3)': return 'log(L$_{\mathrm{CO(3-2)}}$ [L$_{\odot}$])'

    if foo == 'NIIratio': return '[NII]122/[NII]205'
    if foo == 'NIIratio_HI': return '[NII]122/[NII]205 from HI gas'
    if foo == 'NIIratio_HII': return '[NII]122/[NII]205 from HII gas'

    if foo == 'R_NIR_FUV': return 'NIR/FUV flux ratio'
    if foo == 'lR_NIR_FUV': return 'log NIR/FUV flux ratio'

    if foo == 'L_FIR': return 'L$_{\mathrm{FIR}}$ [L$_{\odot}$]'


    if foo == 'cell_size': return 'Cell size [kpc])'
    if foo == 'lcell_size': return 'log Cell size [kpc])'
    
    # If you got to here, nothing was found
    return('')

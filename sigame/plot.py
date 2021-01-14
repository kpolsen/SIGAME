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
import pandas as pd
import numpy as np
import pdb as pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from mpl_toolkits import mplot3d
import pickle
from matplotlib.colors import LogNorm
import copy
import os as os
from scipy import ndimage, misc
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
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

def simple_plot(**kwargs):
    '''A function to standardize all plots

    Plots that can be created this way:
        - 1. errorbar plot (with x and/or y errorbars)
        - 2. line
        - 3. histogram
        - 4. markers
        - 5. bar
        - 6. hexbin
        - 7. contour plot
        - 8. scatter plot
        - 9. hatched/filled region

    The '1' below can be replaced by '2', '3', '4' etc for several plotting options in the same figure.


    Parameters
    ----------

    add : bool
        If True add to existing axis object, otherwise new figure+axes will be created, default: False

    fig : int
        Figure number, default: 0

    figsize : tuple
        Figure size, default: (8,6)

    figname : str
        If defined, a figure will be saved with this path+name, default = not defined (no figure saved)

    figtype : str
        Format of figure, default: 'png'

    figres : float
        The dpi of the saved figure, default: 1000

    fontsize : float
        Fontsize of tick labels, default: 15

    x1 : list or numpy array
        x values

    y1 : list or numpy array
        y values

    xr : list
        x range

    yr : list
        y range

    xlog : bool
        If True: x-axis will be in log units

    ylog : bool
        If True: y-axis will be in log units

    fill1 : str
        If defined, markers will be used ('y' for filled markers, 'n' for open markers), default: 'y'

    ls1 : str
        Linestyle, default: 'None' (does markers by default)

    ds1 : str
        Drawstyle, use for 'steps', default: 'None'

    ma1 : str
        Marker type, default: 'x'

    ms1 : int/float
        Marker size, default: 5

    mew1 : int/float
        Marker edge width, default: 2

    col1 : str
        Color of markers/lines, default: 'k'

    ecol1 : str
        Edgecolor, default: 'k'

    lab1 : str
        Label for x1,y1 points, default: ''

    alpha1 : float
        Transparency fraction, default: 1.1

    dashes1 : str
        Custom-made dashes/dots, default = ''

    legend : bool
        Whether to plot legend or not, default: False

    leg_fs : float
        Legend fontsize, default: not defined (same as fontsize for labels/tick marks)

    legloc : str or list of coordinates
        Legend location, default: 'best'

    cmap1 : str
        Colormap for contour plots, default: 'viridis'

    xlab : str
        x axis label, default: no label

    ylab : str
        y axis label, default: no label

    title : str
        Plot title, default: no title

    xticks: bool
        Whether to put x ticks or not, default: True

    yticks: bool
        Whether to put y ticks or not, default: True

    lab_to_tick : int/float
        If axis labels should be larger than tick marks, say how much here, default: 1.0

    lex1,uex1: list or numpy array
        Lower and upper errorbars on x1, default: None, **options**:
        If an element in uex1 is 0 that element will be plotted as upper limit in x

    ley1,uey1: list or numpy array
        lower and upper errorbars on y1, default: None, **options**:
        If an element in uey1 is 0 that element will be plotted as upper limit in y

    histo1 : bool
        Whether to make a histogram of x1 values, default: False

    histo_real1 : bool
        Whether to use real values for histogram or percentages on y axis, default: False

    bins1 : int
        Number of bins in histogram, default: 100

    weights1 : list or numpy array
        weights to histogram, default: np.ones(len(x1))

    hexbin1 : bool
        If True, will make hexbin contour plot, default: False

    contour_type1 : str
        If defined, will make contour plot, default: not defined, **options**:
        plain: use contourf on colors alone, optionally with contour levels only (no filling),
        hexbin: use hexbin on colors alone,
        median: use contourf on median of colors,
        mean: use contourf on mean of colors,
        sum: use contourf on sum of colors

    barwidth1 : float
        If defined, will make bar plot with this barwidth

    scatter_color1 : list or numpy array
        If defined, will make scatter plot with this color, default: not defined (will not do scatter plot)

    colormin1 : float
        Minimum value for colorbar in scatter plot, default: not defined (will use all values in scatter_color1)

    lab_colorbar : str
        Label for colorbar un scatter plot, default: not defined (will not make colorbar)

    hatchstyle1 : str
        If defined, make hatched filled region, default: not defined (will not make hatched region), **options**:
        if set to '', fill with one color,
        otherwise, use '/' '//' '///'' etc.

    text : str
        If defined, add text to figure with this string, default: not defined (no text added)

    textloc : list
        Must be specified in normalized axis units, default: [0.1,0.9]

    textbox : bool
        If True, put a box around text, default: False

    fontsize_text : int/float
        Fontsize of text on plot, default: 0/7 * fontsize

    grid : bool
        Turn on grid lines, default: False

    SC_return : bool
        Return scatter plot object or not, default = False

    '''

    # Set fontsize
    if mpl.rcParams['ytick.labelsize'] == 'medium':
        fontsize            =   15
    if 'fontsize' in kwargs:
        fontsize            =   kwargs['fontsize']
        mpl.rcParams['ytick.labelsize'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
    else:
        fontsize            =   15
    lab_to_tick         =   1.
    if 'lab_to_tick' in kwargs: lab_to_tick = kwargs['lab_to_tick']
    textcol             =   'black'
    if 'textcol' in kwargs: textcol = kwargs['textcol']
    fontsize_text       =   fontsize*0.7

    # Get axes object
    if 'add' in kwargs:
        ax1                 =   plt.gca()
    else:
        fig                 =   0                                       # default figure number
        if 'fig' in kwargs: fig = kwargs['fig']
        figsize             =   (8,6)                                   # slightly smaller figure size than default
        if 'figsize' in kwargs: figsize = kwargs['figsize']
        fig,ax1             =   plt.subplots(figsize=figsize)
    # if kwargs.has_key('aspect'): ax1.set_aspect(kwargs['aspect'])

    if 'plot_margin' in kwargs:
        plt.subplots_adjust(left=kwargs['plot_margin'], right=1-kwargs['plot_margin'], top=1-kwargs['plot_margin'], bottom=kwargs['plot_margin'])
        # pdb.set_trace()

    # Default line and marker settings
    ls0                 =   'None'              # do markers by default
    ds0                 =   'default'              # do line by default
    lw0                 =   2                   # linewidth
    ma0                 =   'x'                 # marker type
    ms0                 =   5                   # marker size
    mew0                =   2                   # marker edge width
    col0                =   'k'                 # color
    ecol0               =   'k'                 # color
    lab0                =   ''                  # label
    fill0               =   'y'
    alpha0              =   1.0
    dashes0             =   ''
    only_one_colorbar   =   1
    legend              =   False
    cmap0               =   'viridis'
    bins0               =   100
    zorder0             =   100
    fillstyle0          =   'full'

    # Set axis settings
    xlab,ylab           =   '',''
    if 'xlab' in kwargs:
        ax1.set_xlabel(kwargs['xlab'],fontsize=fontsize*lab_to_tick)
    if 'ylab' in kwargs:
        ax1.set_ylabel(kwargs['ylab'],fontsize=fontsize*lab_to_tick)
    if 'title' in kwargs:
        ax1.set_title(kwargs['title'],fontsize=fontsize*lab_to_tick)
    if 'histo' in kwargs:
        ax1.set_ylabel('Number fraction [%]',fontsize=fontsize*lab_to_tick)

    # Set aspect here before colorbar
    # if kwargs.has_key('aspect'):
    #     ax1.set_aspect(kwargs['aspect'])

    # Add lines/points to plot
    for i in range(1,20):
        done            =   'n'
        if 'x'+str(i) in kwargs:
            if 'x'+str(i) in kwargs: x = kwargs['x'+str(i)]
            if 'y'+str(i) in kwargs: y = kwargs['y'+str(i)]
            # If no x values, make them up
            if not 'x'+str(i) in kwargs: x = np.arange(len(y))+1
            ls              =   ls0
            ds              =   ds0
            lw              =   lw0
            mew             =   mew0
            ma              =   ma0
            col             =   col0
            mfc             =   col0
            ecol            =   ecol0
            ms              =   ms0
            lab             =   lab0
            fill            =   fill0
            alpha           =   alpha0
            cmap            =   cmap0
            bins            =   bins0
            zorder          =   zorder0
            fillstyle       =   fillstyle0
            if 'lw'+str(i) in kwargs: lw = kwargs['lw'+str(i)]
            if 'lw' in kwargs: lw = kwargs['lw'] # or there is a general keyword for ALL lines...
            if 'mew'+str(i) in kwargs: mew = kwargs['mew'+str(i)]
            if 'ma'+str(i) in kwargs: ma = kwargs['ma'+str(i)]
            if 'ms'+str(i) in kwargs: ms = kwargs['ms'+str(i)]
            if 'col'+str(i) in kwargs: col, mfc = kwargs['col'+str(i)], kwargs['col'+str(i)]
            if 'ecol'+str(i) in kwargs: ecol = kwargs['ecol'+str(i)]
            if 'lab'+str(i) in kwargs: lab = kwargs['lab'+str(i)]
            if 'lab'+str(i) in kwargs: legend = 'on' # do make legend
            legend          =   False
            if 'legend' in kwargs: legend = kwargs['legend']
            if 'ls'+str(i) in kwargs: ls = kwargs['ls'+str(i)]
            if 'ds'+str(i) in kwargs: ds = kwargs['ds'+str(i)]
            if 'alpha'+str(i) in kwargs: alpha = kwargs['alpha'+str(i)]
            if 'cmap'+str(i) in kwargs: cmap = kwargs['cmap'+str(i)]
            if 'zorder'+str(i) in kwargs: zorder = kwargs['zorder'+str(i)]
            if 'fill'+str(i) in kwargs:
                fill                = kwargs['fill'+str(i)]
                if kwargs['fill'+str(i)] == 'y': fillstyle  = 'full'
                if kwargs['fill'+str(i)] == 'n': fillstyle, mfc, alpha  = 'none', 'None', None


            # ----------------------------------------------
            # 1. Errorbar plot
            # Errorbars/arrows in x AND y direction
            if 'lex'+str(i) in kwargs:
                if 'ley'+str(i) in kwargs:
                    for x1,y1,lex,uex,ley,uey in zip(x,y,kwargs['lex'+str(i)],kwargs['uex'+str(i)],kwargs['ley'+str(i)],kwargs['uey'+str(i)]):
                        ax1.errorbar(x1,y1,color=col,ls="None",fillstyle=fillstyle,xerr=[[lex],[uex]],yerr=[[ley],[uey]],elinewidth=lw,capsize=0,\
                            capthick=0,marker=kwargs['ma'+str(i)],label=kwargs['lab'+str(i)])
            # Errorbars/arrows in x direction
            if 'lex'+str(i) in kwargs:
                # print('>> Adding x errorbars!')
                for x1,y1,lex,uex in zip(x,y,kwargs['lex'+str(i)],kwargs['uex'+str(i)]):
                    if uex > 0: # not upper limit, plot errobars
                        ax1.errorbar(x1,y1,color=col,ls="None",fillstyle=fillstyle,xerr=[[lex],[uex]],elinewidth=lw,capsize=0,\
                            capthick=0,marker=kwargs['ma'+str(i)])
                    if uex == 0: # upper limit, plot arrows
                        ax1.errorbar(x1,y1,color=col,xerr=lex,\
                           xuplims=True,ls="None",fillstyle=fillstyle,linewidth=lw,mew=0,capthick=lw*2)
            # Errorbars/arrows in y direction
            if 'ley'+str(i) in kwargs:
                # print('>> Adding y errorbars!')
                for x1,y1,ley,uey in zip(x,y,kwargs['ley'+str(i)],kwargs['uey'+str(i)]):
                    if uey > 0: # not upper limit, plot errorbars
                        ax1.errorbar(x1,y1,color=col,ls='None',fillstyle=fillstyle,yerr=[[ley],[uey]],elinewidth=lw,\
                            capsize=0,capthick=0,marker=ma)
                    if uey == 0: # upper limit, plot arrows
                        ax1.errorbar(x1,y1,color=col,yerr=ley,\
                           uplims=True,ls="None",fillstyle=fillstyle,linewidth=lw,mew=0,capthick=lw*2)
                    continue

            # ----------------------------------------------
            # 2. Line connecting the dots
            if 'y'+str(i) in kwargs:

                if type(kwargs['y'+str(i)]) == str: y = ax1.get_ylim()
                if 'dashes'+str(i) in kwargs:
                    # print('>> Line plot!')
                    ax1.plot(x,y,ls=ls,ds=ds,color=col,lw=lw,label=lab,dashes=kwargs['dashes'+str(i)],alpha=alpha,zorder=zorder)
                    continue
                else:
                    if 'ls'+str(i) in kwargs:
                        # print(ls,ds,col,lw,zorder)
                        # print('>> Line plot!')
                        ax1.plot(x,y,ls=ls,ds=ds,color=col,lw=lw,label=lab,alpha=alpha,zorder=zorder)
                        continue

            # ----------------------------------------------
            # 3. Histogram
            if 'histo'+str(i) in kwargs:
                # print('>> Histogram!')
                if ls == 'None': ls = '-'
                weights             =   np.ones(len(x))
                if 'bins'+str(i) in kwargs: bins = kwargs['bins'+str(i)]
                if 'weights'+str(i) in kwargs: weights = 'weights'+str(i) in kwargs
                if 'histo_real'+str(i) in kwargs:
                    make_histo(x,bins,col,lab,percent=False,weights=weights,lw=lw,ls=ls,drawstyle='steps')
                else:
                    make_histo(x,bins,col,lab,percent=True,weights=weights,lw=lw,ls=ls,drawstyle='steps')
                continue

            # ----------------------------------------------
            # 4. Marker plot
            if 'fill'+str(i) in kwargs:
                # print('>> Marker plot!')
                ax1.plot(x,y,linestyle='None',color=col,marker=ma,mew=mew,ms=ms,fillstyle=fillstyle,alpha=alpha,markerfacecolor=mfc,zorder=zorder,label=lab)
                continue

            # ----------------------------------------------
            # 5. Bar plot
            if 'barwidth'+str(i) in kwargs:
                # print('>> Bar plot!')
                plt.bar(x,y,width=kwargs['barwidth'+str(i)],color=col,alpha=alpha)
                continue

            # ----------------------------------------------
            # 6. Hexbin contour bin
            if 'hexbin'+str(i) in kwargs:
                # print('>> Hexbin contour plot!')
                bins                =   300
                if 'bins'+str(i) in kwargs: bins = kwargs['bins'+str(i)]
                if 'alpha'+str(i) in kwargs: alpha = kwargs['alpha'+str(i)]
                if 'col'+str(i) in kwargs:
                    colors          =   kwargs['col'+str(i)]
                    CS              =   ax1.hexbin(x, y, C=colors, gridsize=bins, cmap=cmap, alpha=alpha)
                else:
                    CS              =   ax1.hexbin(x, y, gridsize=bins, cmap=cmap, alpha=alpha)
                continue

            # ----------------------------------------------
            # 7. Contour map

            if 'contour_type'+str(i) in kwargs:
                CS                  =   make_contour(i,fontsize,kwargs=kwargs)

            if 'colorbar'+str(i) in kwargs:
                if kwargs['colorbar'+str(i)]:
                    if only_one_colorbar == 1: pad = 0
                    if only_one_colorbar < 0: pad = 0.03
                    ax2 = ax1.twinx()
                    ax2.get_xaxis().set_visible(False)
                    ax2.get_yaxis().set_visible(False)
                    divider = make_axes_locatable(ax2)
                    cax = divider.append_axes("right", size="5%", pad=pad)
                    cbar                    =   plt.colorbar(CS,cax=cax)
                    cbar.set_label(label=kwargs['lab_colorbar'+str(i)],size=fontsize-5)   # colorbar in it's own axis
                    cbar.ax.tick_params(labelsize=fontsize-5)
                    only_one_colorbar       =   -1
                    plt.axes(ax1)

            # ----------------------------------------------
            # 8. Scatter plot (colored according to a third parameter)
            if 'scatter_color'+str(i) in kwargs:
                # print('>> Scatter plot!')
                SC              =   ax1.scatter(x,y,marker=ma,lw=mew,s=ms,c=kwargs['scatter_color'+str(i)],cmap=cmap,alpha=alpha,label=lab,edgecolor=ecol,zorder=zorder)
                if 'colormin'+str(i) in kwargs: SC.set_clim(kwargs['colormin'+str(i)],max(kwargs['scatter_color'+str(i)]))
                if 'lab_colorbar' in kwargs:
                    if only_one_colorbar > 0:
                        cbar                    =   plt.colorbar(SC,pad=0)
                        cbar.set_label(label=kwargs['lab_colorbar'],size=fontsize-2)   # colorbar in it's own axis
                        cbar.ax.tick_params(labelsize=fontsize-2)
                        only_one_colorbar       =   -1
                continue

            # ----------------------------------------------
            # 8. Filled region
            if 'hatchstyle'+str(i) in kwargs:
                # print('>> Fill a region!')
                from matplotlib.patches import Ellipse, Polygon
                if kwargs['hatchstyle'+str(i)] != '': ax1.add_patch(Polygon([[x[0],y[0]],[x[0],y[1]],[x[1],y[1]],[x[1],y[0]]],closed=True,fill=False,hatch=kwargs['hatchstyle'+str(i)],color=col),zorder=zorder)
                if kwargs['hatchstyle'+str(i)] == '': ax1.fill_between(x,y[0],y[1],facecolor=col,color=col,alpha=alpha,lw=0,zorder=zorder)
                continue

    # Log or not log?
    if 'xlog' in kwargs:
        if kwargs['xlog']: ax1.set_xscale('log')
    if 'ylog' in kwargs:
        if kwargs['ylog']: ax1.set_yscale('log')

    # Legend
    if legend:
        legloc          =   'best'
        if 'legloc' in kwargs: legloc = kwargs['legloc']
        frameon         =   not 'frameon' in kwargs or kwargs['frameon']          # if "not" true that frameon is set, take frameon to kwargs['frameon'], otherwise always frameon=True
        handles1, labels1     =   ax1.get_legend_handles_labels()
        leg_fs          =   fontsize#int(fontsize*0.7)
        if 'leg_fs' in kwargs: leg_fs = kwargs['leg_fs']
        leg = ax1.legend(loc=legloc,fontsize=leg_fs,numpoints=1,scatterpoints = 1,frameon=frameon)
        leg.set_zorder(zorder)

    # Add text to plot
    if 'text' in kwargs:
        textloc             =   [0.1,0.95]
        if 'textloc' in kwargs: textloc = kwargs['textloc']
        fontsize_text       =   fontsize
        if 'textfs' in kwargs: fontsize_text=kwargs['textfs']
        if 'textbox' in kwargs:
            ax1.text(textloc[0],textloc[1],kwargs['text'][0],\
                transform=ax1.transAxes,verticalalignment='top', horizontalalignment='right',fontsize=fontsize_text,\
                bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=1'))
        else:
            if len(kwargs['text']) == 1:
                ax1.text(textloc[0],textloc[1],kwargs['text'][0],color=textcol,\
                    transform=ax1.transAxes,verticalalignment='top', horizontalalignment='left',fontsize=fontsize_text)
            if len(kwargs['text']) > 1:
                for l,t in zip(textloc,kwargs['text']):
                    ax1.text(l[0],l[1],t,color='black',\
                        verticalalignment='top', horizontalalignment='left',fontsize=fontsize_text)

    if 'grid' in kwargs: ax1.grid()

    if 'xticks' in kwargs:
        if kwargs['xticks']:
            ax1.set_xticks(kwargs['xticks'])
            ax1.set_xticklabels(str(_) for _ in kwargs['xticks'])
        else:
            ax1.set_xticks(kwargs['xticks'])
            ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if 'xticklabels' in kwargs: ax1.set_xticklabels(kwargs['xticklabels'])

    if 'yticks' in kwargs:
        if kwargs['yticks']:
            ax1.set_yticks(kwargs['yticks'])
            ax1.set_yticklabels(str(_) for _ in kwargs['yticks'])
        else:
            ax1.set_yticks(kwargs['yticks'])
            ax1.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if 'yticklabels' in kwargs: ax1.set_yticklabels(kwargs['yticklabels'])

    if 'xr' in kwargs: ax1.set_xlim(kwargs['xr'])
    if 'yr' in kwargs: ax1.set_ylim(kwargs['yr'])

    # plt.tight_layout()

    # Save plot if figure name is supplied
    if 'figres' in kwargs:
        dpi = kwargs['figres']
    else:
        dpi = 1000
    if 'figname' in kwargs:
        figname = kwargs['figname']
        figtype = 'png'
        if 'figtype' in kwargs: figtype = kwargs['figtype']
        plt.savefig(figname+'.'+figtype, format=figtype, dpi=dpi) # .eps for paper!

    if 'show' in kwargs:
        plt.show(block=False)


    # restoring defaults
    # mpl.rcParams['xtick.labelsize'] = u'medium'
    # mpl.rcParams['ytick.labelsize'] = u'medium'

    if 'fig_return' in kwargs:
        return fig

def make_map(ob):

    set_mpl_params()

    fig =   plt.figure(figsize=(10,10))
    ax  =   fig.add_subplot(111)
    im  =   ax.contourf(ob.X, ob.Y, ob.Z, ob.Ncontours, cmap=ob.cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(ob.title)
    ax.set_xlabel(ob.xlabel)
    ax.set_ylabel(ob.ylabel)

def make_histo(x,bins,col,lab,percent=True,weights=1,lw=1,ls='-',drawstyle='steps'):
    '''
    Function to make a histogram (called by simple_plot)
    '''

    ax1             =   plt.gca()
    hist            =   np.histogram(x,bins,weights=weights)
    hist1           =   np.asarray(hist[0])
    hist2           =   np.asarray(hist[1])
    if percent: hist1           =   hist1*1./sum(hist1)*100.
    wid             =   (hist2[1]-hist2[0])
    # add some zeros to bring histogram down
    hist2           =   np.append([hist2],[hist2.max()+wid])
    hist2           =   np.append([hist2.min()-wid],[hist2])
    hist1           =   np.append([hist1],[0])
    hist1           =   np.append([0],[hist1])
    # plot it!
    ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,ls=ls,ds=ds,color=col,label=lab,lw=lw)
    if percent == True: ax1.set_ylabel('Fraction [%]')
    if percent == False: ax1.set_ylabel('Number')

def make_contour(i,fontsize,kwargs):
    '''Makes contour plot (called by simple_plot)

    Parameters
    ----------
    contour_type: str
        Method used to create contour map (see simple_plot)

    '''

    # print('Contour plot!')

    ax1                 =   plt.gca()

    linecol0            =   'k'
    cmap0               =   'viridis'
    alpha0              =   1.1
    nlev0               =   10
    only_one_colorbar   =   1

    # Put on regular grid!
    if 'y'+str(i) in kwargs:

        y               =   kwargs['y'+str(i)]
        x               =   kwargs['x'+str(i)]
        colors          =   kwargs['col'+str(i)]
        linecol         =   linecol0
        if 'linecol'+str(i) in kwargs: linecol = kwargs['linecol'+str(i)]
        cmap            =   cmap0
        if 'cmap'+str(i) in kwargs: cmap = kwargs['cmap'+str(i)]
        alpha           =   alpha0
        if 'alpha'+str(i) in kwargs: alpha = kwargs['alpha'+str(i)]
        nlev            =   nlev0
        if 'nlev'+str(i) in kwargs: nlev = kwargs['nlev'+str(i)]

        if kwargs['contour_type'+str(i)] == 'plain':

            if cmap == 'none':
                print('no cmap')
                CS = ax1.contour(x,y,colors, nlev, colors=linecol)
                plt.clabel(CS, fontsize=9, inline=1)

            if 'colormin'+str(i) in kwargs:
                # print('Colormap with a minimum value')
                CS = ax1.contourf(x,y,colors, nlev, cmap=cmap)
                ax1.contourf(x,y,colors, levels=kwargs['colormin'+str(i)], colors='k')

            else:
                if 'alpha'+str(i) in kwargs:
                    print('with alpha')
                    CS = ax1.contourf(x,y,colors, nlev, cmap=cmap, alpha=kwargs['alpha'+str(i)])
                if not 'alpha'+str(i) in kwargs:
                    # print('without alpha')
                    CS = ax1.contourf(x,y,colors, nlev, cmap=cmap)#, lw=0, antialiased=True)

        if kwargs['contour_type'+str(i)] == 'hexbin':
            CS              =   ax1.hexbin(x, y, C=colors, cmap=cmap)

        if kwargs['contour_type'+str(i)] == 'mesh':
            CS              =   ax1.pcolormesh(x,y,colors, cmap=cmap)

        if kwargs['contour_type'+str(i)] in ['median','mean','sum']:
            gridx           =   np.arange(min(x),max(x),kwargs['dx'+str(i)])
            gridy           =   np.arange(min(y),max(y),kwargs['dy'+str(i)])
            lx,ly           =   len(gridx),len(gridy)
            gridx1          =   np.append(gridx,max(gridx)+kwargs['dx'+str(i)])
            gridy1          =   np.append(gridy,max(gridy)+kwargs['dy'+str(i)])
            z               =   np.zeros([lx,ly])+min(colors)
            for i1 in range(0,lx):
                for i2 in range(0,ly):
                    # pdb.set_trace()
                    colors1         =   colors[(x > gridx1[i1]) & (x < gridx1[i1+1]) & (y > gridy1[i2]) & (y < gridy1[i2+1])]
                    if len(colors1) > 1:
                        if kwargs['contour_type'+str(i)] == 'median': z[i1,i2]       =   np.median(colors1)
                        if kwargs['contour_type'+str(i)] == 'mean': z[i1,i2]         =   np.mean(colors1)
                        if kwargs['contour_type'+str(i)] == 'sum': z[i1,i2]          =   np.sum(colors1)
            if 'nlev'+str(i) in kwargs: nlev0 = kwargs['nlev'+str(i)]
            CS               =   ax1.contourf(gridx, gridy, z.T, nlev0, cmap=cmap)
            mpl.rcParams['contour.negative_linestyle'] = 'solid'
            # CS               =   ax1.contour(gridx, gridy, z.T, 5, colors='k')
            # plt.clabel(CS, inline=1, fontsize=10)
            if 'colormin'+str(i) in kwargs: CS.set_clim(kwargs['colormin'+str(i)],max(z.reshape(lx*ly,1)))
            if 'colormin'+str(i) in kwargs: print(kwargs['colormin'+str(i)])
            CS.cmap.set_under('k')

    # plt.subplots_adjust(left=0.13, right=0.94, bottom=0.14, top=0.95)

    return CS

def histos(**kwargs):
    '''Makes histograms of all (particles in all) galaxies in the sample on the same plot.

    Parameters
    ---------

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
        gal_indices = p.gal_index

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
        plt.ion()
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
            # print('min and max: %s and %s ' % (np.min(dat[dat > -100]),dat.max()))
            if logy == 'n':
                if weigh == 'm':  w       =   w[dat > -10.**(20)]
                dat     =   dat[dat > -10.**(20)]
            if logy == 'y':
                if weigh == 'm':  w       =   w[dat > -20]
                dat     =   dat[dat > -20]
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
        xl          =   getlabel(quant)
        if logx    == 'y': xl = 'log '+getlabel(quant)
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
                plt.savefig(p.d_plot + 'histos/'+name+'.png', format='png', dpi=250) # .eps for paper!

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
    ---------
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
                plt.savefig(figname, format='png', dpi=250)
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
        plt.savefig(p.d_plot + 'cell_data/map_%s_G%i.png' % (p.prop,p.gal_index), format='png', dpi=250)

def map_sim_property(**kwargs):
    """ Map a simulation property in 2D
    
    .. note:: Requires swiftsimio installed

    Parameters
    ---------
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
                plt.savefig(p.d_plot + 'sim_data/map_%s_%s_gals_%i.%s' % (p.prop,p.z1,fignum,p.format), format=p.format, dpi=250)
                fignum += 1

    else:
        if p.add:
            fig, ax1 = plt.gcf(), p.ax
        if not p.add:
            fig = plt.figure(figsize=(6,6))
            ax1 = fig.add_axes([0.1, 0.01, 0.8, 0.8]) 
            ax1.axis('equal')

        gal_ob                  =   gal.galaxy(GR=GR, gal_index=p.gal_index)
        simgas                  =   aux.load_temp_file(gal_ob=gal_ob,data_type=p.sim_type)
        # index = np.ones(len(simgas))
        # index[(simgas.x.values > 0) & (simgas.y.values > 0)] = 0
        # simgas = simgas[index == 1].reset_index(drop=True)
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
        # Plot map
        if not p.R_max:
            p.R_max = max_scale/2
        if not p.vmax:
            p.vmax = np.max(map2D)
        if p.log: 
            map2D[map2D < 10.**p.vmin] = 10.**p.vmin/2
            map2D[map2D > 10.**p.vmax] = 10.**p.vmax
            map2D = np.log10(map2D)
        if not p.log: map2D[map2D < p.vmin] = p.vmin/2 #np.min(map2D[map2D > 0])
        #map2D            =   ndimage.rotate(map2D, 90, reshape=True)
        map2D = np.flipud(map2D)

        im = ax1.imshow(map2D,\
            extent=[-max_scale/2,max_scale/2,-max_scale/2,max_scale/2],vmin=p.vmin,vmax=p.vmax,cmap=p.cmap)
        # Limit axes limits a bit to avoid area with no particles...
        zoom = 1#/1.5
        ax1.set_xlim([-1/zoom * p.R_max,1/zoom * p.R_max])
        ax1.set_ylim([-1/zoom * p.R_max,1/zoom * p.R_max])
        if p.colorbar: fig.colorbar(im,shrink=0.8,ax=ax1,label=lab)
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
            plt.savefig(p.d_plot + 'sim_data/map_%s_G%i.png' % (p.prop,p.gal_index), format=p.format, dpi=250)

    if not p.colorbar: return(im)

def map_sim_positions(**kwargs):
    """ Simple function to map sim particle positions in 2D
    
    Parameters
    ---------
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

def stamps(**kwargs):
    """ Map a simulation property in 2D for each galaxy and save in separate figures.
    """
    GR                      =   glo.global_results()
    gal_indices             =   np.arange(GR.N_gal)

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    for gal_index in gal_indices:

        fig, ax = plt.subplots(figsize=(8,8))

        gal_ob                  =   gal.galaxy(GR=GR, gal_index=gal_index)
        simgas                  =   gal_ob.particle_data.get_dataframe('simgas')
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

def stamp_collection(**kwargs):
    """ Map a simulation property in 2D for each galaxy and save in combined figures.
    
    .. note:: Requires swiftsimio installed

    Parameters
    ---------
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
        simgas                  =   gal_ob.particle_data.get_dataframe('simgas')
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
            plt.savefig(p.d_plot + 'sim_data/%s%s_map_%s_%s_gals_%i.png' % (p.sim_name,p.sim_run,p.prop,p.z1,fignum), format='png', dpi=250)
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
    ax.set_xlim([1e7,1e12])
    ax.set_ylim([10**(-2),1e2])
  
    # Plot 25 Mpc box? 
    if p.select == '_25Mpc':
        GR = pd.read_pickle(p.d_data + 'results/z0_246gals_simba_25Mpc_arepoPDF')
        sc = ax.scatter(GR.M_star.values,GR.SFR.values,\
                marker='^',s=20,alpha=0.8,c=GR.Zsfr,label='Simba-25 galaxy sample',zorder=10)
        print(GR.SFR.values.min(),GR.M_star.values.min())

    # Plot current sample
    GR                  =   glo.global_results()
    M_star,SFR,Zsfr = getattr(GR,'M_star'),getattr(GR,'SFR'),getattr(GR,'Zsfr')
    if p.select == '_MS':
        indices = aux.select_salim18(GR.M_star,GR.SFR)
        M_star = M_star[indices]
        SFR = SFR[indices]
        Zsfr = Zsfr[indices]
        print('With MS selection criteria: only %i galaxies' % (len(M_star)))
    sc = ax.scatter(M_star,SFR,\
            marker='o',s=20,alpha=0.8,c=Zsfr,label='Simba-100 galaxy sample',zorder=10)

    # Plot all galaxies in simulation volume
    df_all              =   pd.read_pickle(p.d_data + 'galaxy_selection/z0_all_galaxies%s' % p.sim_runs[0])
    df_all1 = df_all[(df_all['SFR_'+method] > 0) & (df_all['SFR_'+method] != 1)]
    # hb = ax.hexbin(df_all['M_star_'+method],df_all['SFR_'+method],xscale='log',yscale='log',\
    #                         cmap='binary',lw=0,gridsize=70)
    df_all              =   pd.read_pickle(p.d_data + 'galaxy_selection/z0_all_galaxies%s' % p.sim_runs[1])
    df_all2 = df_all[df_all['SFR_'+method] > 0]
    df_all = df_all1.append(df_all2, ignore_index=True)

    hb = ax.hexbin(df_all2['M_star_'+method],df_all2['SFR_'+method],xscale='log',yscale='log',\
                            cmap='binary',lw=0,gridsize=(50,70))

    # Plot observations
    if p.zred == 0:
        MS_salim = pd.read_csv('data/observations/MS/salim2018_ms_v1.dat',\
                names=['logMstar','logsSFR','logsSFR_1','logsSFR_2'],sep='   ')
        ax.fill_between(10.**MS_salim.logMstar,10.**MS_salim.logMstar*10.**MS_salim.logsSFR_1,\
                 10.**MS_salim.logMstar*10.**MS_salim.logsSFR_2,color='teal',alpha=0.3)
        ax.plot(10.**MS_salim.logMstar,10.**MS_salim.logMstar*10.**MS_salim.logsSFR,\
                '--',color='teal',label='[Salim+18] SF MS')
        # MS_salim = pd.read_csv('data/observations/MS/salim2018_ms_v2.dat',names=['logMstar','logsSFR'],sep='   ')
        # ax.plot(10.**MS_salim.logMstar,10.**MS_salim.logMstar*10.**MS_salim.logsSFR,'--',label='[Salim+18] SF MS')
        cosmo = FlatLambdaCDM(H0=0.68*100 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
        t = cosmo.age(0).value
        fit_speagle = 10.**((0.84-0.026*t)*np.log10(ax.get_xlim())-(6.51-0.11*t))
        ax.fill_between(ax.get_xlim(),10.**(np.log10(fit_speagle)-0.3),\
            10.**(np.log10(fit_speagle)+0.3),alpha=0.2,color='grey')
        fit_speagle = 10.**((0.84-0.026*t)*np.log10(ax.get_xlim())-(6.51-0.11*t))
        # Convert from Kroupa to Chabrier:  https://ned.ipac.caltech.edu/level5/March14/Madau/Madau3.html
        ax.plot(ax.get_xlim(),fit_speagle*0.63/0.67,':',color='grey',label='[Speagle+14] "mixed" fit')
  


    ax.set_ylabel('SFR [M$_{\odot}$/yr]')
    ax.set_xlabel('M$_*$ [M$_{\odot}$]')
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r'$\langle$Z$\rangle_{\rm{SFR}}$ [Z$_{\odot}$]')
    handles,labels = ax.get_legend_handles_labels()
    handles = np.flip(handles)
    labels = np.flip(labels)
    ax.legend(handles,labels,fontsize=12)
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'sim_data/'): os.mkdir(p.d_plot + 'sim_data/')    
        plt.savefig('plots/sim_data/SFR_Mstar_%s_%s%s' % (method,p.sim_name,p.sim_run),dpi=200)

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

def Mstar_SFR(**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if not p.xlim:
        p.xlim              =   np.array([1e10,1e13])

    GR                  =   glo.global_results()

    Mstar               =   getattr(GR,'M_star')
    SFR                 =   getattr(GR,'SFR')
    SFR_MS              =   aux.MS_SFR_Mstar(p.xlim)

    plot.simple_plot(x1=Mstar,y1=SFR,ma1='x',fill1=True,xlog=True,ylog=True,lab1='Simba galaxies',\
        xlim=p.xlim,ylab=getlabel('lSFR'),xlab='log '+getlabel('M_star'))

    plot.simple_plot(add=True,x1=p.xlim,y1=SFR_MS,ls1='--',lab1='Speagle+14')

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'sim_data/'): os.mkdir(p.d_plot + 'sim_data/')    
        plt.savefig(p.d_plot + 'sim_data/Mstar_SFR_%s.png' % (p.z1), format='png', dpi=250)

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
    simstar             =   simstar[simstar.age < 10**p.vmax/1e9] # only younger than 1 Gyr
    simstar             =   simstar.sort_values('age',ascending=False)
    m                   =   np.log10(simstar.m.values)

    m                   =   (m - m.min())/(m.max() - m.min()) * 300 + 50

    if p.add:
        ax1                 =   p.ax
    else:
        # fig, ax1            =   plt.subplots(figsize=(7.3, 6  ))
        fig = plt.figure(figsize=(6,6.15))
        ax1 = fig.add_axes([0.1, 0.01, 0.8, 0.8]) 

    print('Range in stellar age [Myr]: ',np.min(simstar.age*1e3),np.max(simstar.age*1e3))

    sc = ax1.scatter(simstar.x,simstar.y,s=m,c=np.log10(simstar.age*1e9),alpha=0.6,cmap='jet',vmin=p.vmin,vmax=p.vmax)
    # ax1.plot(simstar.x,simstar.y,'o',ms=1)
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
        plt.savefig(p.d_plot + 'sim_data/star_map_G%i.png' % (p.gal_index), format=p.format, dpi=250)

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
            fig = plt.figure(figsize=(6,6.15))
            ax1 = fig.add_axes([0.1, 0.01, 0.8, 0.8]) 
            ax1.axis('equal')

        isrf_ob = gal.isrf(gal_index)

        # Load SKIRT output
        image_data,units        =   isrf_ob._get_map_inst(orientation=p.orientation,select=p.select)
        wa,bin_width            =   aux.read_map_inst_wavelengths(isrf_ob._get_name()+p.select)
        N_start,N_stop          =   aux.FUV_index(wa)
        FUV_xy_image            =   image_data[N_start:N_stop,:,:].sum(axis=0)

        # Plot image
        FUV_xy_image            =   np.flipud(FUV_xy_image)
        #FUV_xy_image            =   np.rot90(FUV_xy_image)

        FUV_xy_image[FUV_xy_image <= 0] = np.min(FUV_xy_image[FUV_xy_image > 0])
        R = isrf_ob.R_max
        im                      =   ax1.imshow(np.log10(FUV_xy_image),\
            extent=[-R,R,-R,R],\
            cmap='twilight')
        lab                     =   'FUV flux [W/m$^2$/micron/arcsec$^2$]'

        if p.R_max:
            ax1.set_xlim([-p.R_max,p.R_max])
            ax1.set_ylim([-p.R_max,p.R_max])

        if not p.add: ax1.set_xlabel('x [kpc]'); ax1.set_ylabel('y [kpc]')
        if p.colorbar: fig.colorbar(im,shrink=0.8,ax=ax1,label=lab)

        if p.savefig:
            if not os.path.isdir(p.d_plot + 'cell_data/'): os.mkdir(p.d_plot + 'cell_data/')    
            plt.savefig(p.d_plot + 'cell_data/FUV_map_%s%s.png' % (isrf_ob._get_name(),p.select), format=p.format, dpi=250)
        
    if not p.colorbar: return(im)

def FUV_fluxes(**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    if p.add:
        fig, ax1                =   plt.gcf(), p.ax
    if not p.add:
        fig, ax1                =   plt.subplots(figsize=(8, 6  ))

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
        labels           =  {'_1e6':'10$^6$ packets',\
                               '_1e7':'10$^7$ packets',\
                               '_1e8':'10$^8$ packets',\
                               '_1e9':'10$^9$ packets'}
        ax1.plot(hist2[0:len(hist1)]+wid/2,hist1,color=colors[i],drawstyle='steps',label=labels[select])

    ax1.set_yscale("log")
    ax1.set_xlim([-20,5])
    ax1.set_xlabel(getlabel('lG0'))
    # ax1.set_ylabel('Fraction of mass')
    ax1.set_ylabel('Number of cells')
    ax1.legend(loc='upper left')

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/FUV_fluxes_comp.png', format=p.format, dpi=250)

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
        fig, ax1                =   plt.subplots(figsize=(8, 6  ))

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
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/FUV_lums_comp.png', format=p.format, dpi=250)
        
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
            plt.savefig(p.d_plot + 'cell_data/FUV_crosssec_%s.png' % (isrf_ob._get_name()), format=p.format, dpi=250)
    
    if not p.colorbar: return(im)

def skirt_SED(**kwargs): 
    '''Plots SED from SKIRT output for selected galaxies

    Parameters
    ----------
    gal_index : int
        Galaxy index, default: 0

    '''

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results()

    if p.gal_index == None:
        # Use all galaxies
        p.gal_index = np.arange(len(GR))
    else:
        p.gal_index = [p.gal_index]

    add = False
    for gal_index in p.gal_index:

        isrf_ob = gal.isrf(gal_index)

        SED = isrf_ob._read_SED(fluxOutputStyle='Wavelength')

        if add:
            plot.simple_plot(x1=SED['wavelength'].values,y1=SED['F_W_m2'].values,ls1='-')
        else:
            plot.simple_plot(x1=SED['wavelength'].values,y1=SED['F_W_m2'].values,ls1='-',lab1='Total SED, face-on',\
                ylog=True,xlog=True,ylab=r'$\nu$F$_{\nu}$ [W/m$^2$]',xlab='Wavelength [$\mu m$]')

        ax1 = plt.gca()
        ylim = ax1.get_ylim()
        ax1.fill_between([0.0912,0.207],[ylim[0],ylim[0]],[ylim[1],ylim[1]],color='b',alpha=0.5,label='FUV range')
        ax1.set_ylim(ylim);

        ax1.fill_between([3,1000],[ylim[0],ylim[0]],[ylim[1],ylim[1]],color='r',alpha=0.5,label='IR range')
        ax1.set_ylim(ylim);

        ax1.legend(loc='lower left')

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
    plt.savefig(p.d_plot + 'luminosity/L_TIR_SFR.png', format='png', dpi=250)

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
    plt.savefig(p.d_plot + 'cell_data/skirt_spectra_%s.png' % (gal_ob.name), format='png', dpi=250)

    fig,ax = plt.subplots(figsize=(10,8))
    ax.hist(np.log10(R_NIR_FUV[R_NIR_FUV > 0]),bins=200)
    ax.set_xlabel(r'log NIR-to-FUV flux ratio')
    ax.set_ylabel(r'log fraction of cells')
    ax.set_yscale('log')
    plt.savefig(p.d_plot + 'cell_data/skirt_R_NIR_FUV_%s_no_Z.png' % (gal_ob.name), format='png', dpi=250)


    fig,ax = plt.subplots(figsize=(10,8))
    ax.hist(np.log10(UV_to_FUV[UV_to_FUV > 0]),bins=200)
    ax.set_ylabel(r'log UV-to-FUV flux ratio')
    ax.set_yscale('log')
    plt.savefig(p.d_plot + 'cell_data/skirt_UV_to_FUV_%s.png' % (gal_ob.name), format='png', dpi=250)

    fig,ax = plt.subplots(figsize=(10,8))
    ax.hist(np.log10(R_NIR_FUV[R_NIR_FUV > 0]),bins=200,weights=cell_data.m[R_NIR_FUV > 0])
    ax.set_xlabel(r'log NIR-to-FUV flux ratio')
    ax.set_ylabel(r'log mass-weighted fraction of cells')
    ax.set_yscale('log')
    plt.savefig(p.d_plot + 'cell_data/skirt_R_NIR_FUV_%s_mw.png' % (gal_ob.name), format='png', dpi=250)


    pdb.set_trace()

#---------------------------------------------------------------------------
### FOR FRAGMENTATION TASK ###
#---------------------------------------------------------------------------

def three_PDF_plots(res='M51_200pc',table_exts=[''],**kwargs):
    """ Plot total galactic PDF

    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)
    GR                      =   glo.global_results()

    fig,ax1                 =   plt.subplots(figsize=(8,6))

    # First print cell data distribution

    for gal_index,color in zip(p.gal_index,p.color):
        gal_ob                  =   gal.galaxy(GR=GR, gal_index=gal_index)
        df                      =   gal_ob.cell_data.get_dataframe()
        lognH                   =   np.log10(df.nH)
        hist                    =   np.histogram(lognH[df.nH.values > 0],bins=200,weights=df.m[df.nH.values > 0])
        hist1                   =   np.asarray(hist[0]) # histogram
        hist2                   =   np.asarray(hist[1]) # bin edges
        hist1                   =   hist1*1./sum(hist1)
        ax1.plot(hist2[0:len(hist1)],hist1,drawstyle='steps',ls='-',lw=1.5,\
             alpha=0.7,color=color,label='Original cell distribution')
        
        for table_ext,ls in zip(table_exts,['--',':']):
            PDF(gal_index,color=color,table_ext=table_ext,ls=ls,res='M51_200pc',add=True)
        
    ax1.legend(loc='upper right',fontsize=10)

    if not os.path.isdir(p.d_plot + 'cell_data/PDFs/'): os.mkdir(p.d_plot + 'cell_data/PDFs/')    
    plt.savefig(p.d_plot + 'cell_data/PDFs/simple_PDF_%s%s%s_x3.png' % (p.sim_name,p.sim_run,p.table_ext), format='png', dpi=250)

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
        fit_params_SFR = np.load(p.d_table+'fragment/%s_w_sinks%s.npy' % (p.res,p.table_ext),allow_pickle=True).item()
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
    #print('Number of cells with nSFR >= fit nSFR bins:')
    #print(len(df[df.SFR_density >= 10**lognSFR_bins.min()]))
    #print('Number of cells with nH >= fit nH bins (some are 0):')
    #print(len(df[df.nH >= 10**lognH_bins.min()]))
    #print('Number of cells with both:')
    #print(len(df[(df.SFR_density >= 10**lognSFR_bins.min()) & (df.nH >= 10**lognH_bins.min())]))

    # TEST!!!
    # df['nH'] = df.nH.values*0 + 10**(-1)
    # res = 'M51_200pc'; df = pd.read_pickle('data/high_res_data/output/results_%s' % res)
    # df['nH'] = df.n_vw.values
    # df['m'] = df.n_vw.values
    # print(len(df))
    # print(len(df[(df.nH > 0)]))
    # print(len(df[(df.nH > 0) & (df.SFR_density > 0)]))
    # nSFR = df.SFR_density.values
    # nSFR[nSFR == 0 ] = 1e-10
    # df['SFR_density'] = nSFR

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
        plt.savefig(p.d_plot + 'cell_data/PDFs/PDF_%s%s_%s.png' % (gal_ob.name,p.table_ext,res), format='png', dpi=250)

    labels = {'_M10':'Mach = 10','_arepoPDF':'AREPO parametrized PDF'}

    # New figure: One panel of PDF and cumulative mass function (optional)
    if p.add:
        ax1 = plt.gca()
    else:
        fig,ax1                 =   plt.subplots(figsize=(8,6))
    ax1.plot(lognHs,total_PDF,ls=p.ls,lw=2.5,color=p.color,label='G%i - ' % gal_index + labels[p.table_ext])
    ax1.set_yscale('log')
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
    if not p.add: plt.savefig(p.d_plot + 'cell_data/PDFs/simple_PDF_%s%s_%s.png' % (gal_ob.name,p.table_ext,res), format='png', dpi=250)

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

        # plt.savefig(p.d_plot + 'cell_data/properties_SKIRT.png', format='png', dpi=250)


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
        plt.savefig(p.d_plot + 'cell_data/%s_for_Cloudy.png' % gal_ob.name, format='png', dpi=250)

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

        # plt.savefig(p.d_plot + 'cell_data/properties_SKIRT_post.png', format='png', dpi=250)

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
    plt.savefig('plots/sim_data/%s%s_sim_params_%s_%s_%s.png' % (p.sim_name,p.sim_run,p.z1,x,y),dpi=250)

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
        df['nSFR']          =   df.nSFR.values/(0.2**3)
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
    plt.savefig('plots/cell_data/%s%s_cell_params_%s_%s_%s.png' % (p.sim_name,p.sim_run,p.z1,x,y),dpi=250)

def cloudy_table_map(x_index='lognHs',y_index='lognSFRs',**kwargs):
    """ Plot a 2D map in Cloudy look-up tables.


    Parameters
    ----------

    cloudy_param : dict
        Dictionary with the cloudy parameter name as key and value to be kept fixed as value.

    line : str
        Line name whos luminosity will be plotted in the z direction.


    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    cloudy_library = clo.library()
    lookup_table = cloudy_library._restore_lookup_table()

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

    if p.line == '[CII]158_CO(1-0)':
        line_lum = 10.**lookup_table_cut['[CII]158'].values / 10.**lookup_table_cut['CO(1-0)'].values
        line_lum = np.log10(line_lum)
    else:
        line_lum = lookup_table_cut[p.line].values

    lum = line_lum.reshape([len(np.unique(x)), len(np.unique(y))]).T
    # pdb.set_trace()

    vmin = np.min(lum)
    if p.zlim:
        vmin = p.zlim[0]
    lum[lum < vmin] = vmin

    print('Highest %s and lowest %s' % (x_index,y_index))
    print('%s lum: %.2e Lsun' % (p.line,10.**lum[np.where(np.unique(y) == np.min(y))[0],\
                                       np.where(np.unique(x) == np.max(x))[0]]))
    # print(np.max(lum))

    # ax.plot_surface(X, Y, lum, cmap="autumn_r", lw=0, rstride=1, cstride=1,alpha=0.8,vmin=vmin)
    cf = ax.contourf(X,Y, lum, cmap="jet", vmin=vmin, levels=20, lw=0, rstride=1, cstride=1,alpha=0.8)
    plt.colorbar(cf,label=getlabel(p.line))

    translate_labels = {'lognHs':'lnH','logNHs':'lNH','logFUVs':'lG0','logZs':'lZ','lognSFRs':'lSFR_density'}
    ax.set_xlabel('\n\n' + getlabel(translate_labels[x_index]))
    ax.set_ylabel('\n\n' + getlabel(translate_labels[y_index]))

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'look-up/'): os.mkdir(p.d_plot + 'look-up/')    
        plt.savefig(p.d_plot + 'look-up/cloudy_table%s_%s.png' % (p.grid_ext,p.line), format='png', dpi=300) # .eps for paper!

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
        plt.savefig(p.d_plot + 'look-up/cloudy_grid_map_%s%s.%s' % (p.line, p.grid_ext, p.format), format=p.format, dpi=300) # .eps for paper!
    # pdb.set_trace()

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
        plt.savefig(p.d_plot + 'look-up/cloudy_grid_%s.%s' % (p.line, p.format), format=p.format, dpi=300) # .eps for paper!
    # pdb.set_trace()

#---------------------------------------------------------------------------
### LINE LUMINOSITY ###
#---------------------------------------------------------------------------

def compare_runs(names,labnames,**kwargs):

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    d_results = p.d_data + 'results/'

    lines = ['[NII]205','[NII]122','[OIII]88','[CII]158','[OI]63','CO(1-0)','CO(2-1)','CO(3-2)']
    crit = [r'$\rho_{crit}$ [cm$^{-3}$]:','48','310','510','650','2,400','6,200',r'2.2$\times10^{4}$',r'4.7$\times10^{5}$']


    fig,ax = plt.subplots(figsize=(16,7))
    barWidth = 1/len(lines)
    colors = ['orange','purple','forestgreen','cyan','brown']
    std_obs = np.zeros(len(lines))
    from sklearn.linear_model import LinearRegression
    for iGR,name in enumerate(names):
        GR = pd.read_pickle(d_results + name)
        # if name == 'z0_388gals_abun_abun':
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
            L_sim = GR['L_'+line+'_sun'].values
            SFR_sim = GR['SFR'].values
            M_star_sim = GR['M_star'].values
            if p.select == '_MS':
                indices = aux.select_salim18(M_star_sim,SFR_sim)
                L_sim,SFR_sim = L_sim[indices],SFR_sim[indices]
            SFR_sim = SFR_sim[L_sim > 0] 
            L_sim = L_sim[L_sim > 0] 
            if iline == 0: print(names[iGR],len(L_sim))        
            L_obs,SFR_obs,fit,std = add_line_SFR_obs(line,[1e6,1e6],ax,plot=False,select=p.select)
            L_obs1 = L_obs[(L_obs > 0) & (SFR_obs > 0)]
            SFR_obs1 = SFR_obs[(L_obs > 0) & (SFR_obs > 0)]
            std_obs[iline] = std

            # Log-linear fit
            # fit = LinearRegression().fit(np.log10(SFR_obs1).reshape(-1, 1),np.log10(L_obs1).reshape(-1, 1))

            # Deviation sim - obs
            # dev = np.mean(np.log10(L_sim) - fit.predict(np.log10(SFR_sim.reshape(-1, 1))))
            devs = np.log10(L_sim) - fit.predict(np.log10(SFR_sim.reshape(-1, 1))).flatten()
            dev = np.median(np.log10(L_sim) - fit.predict(np.log10(SFR_sim.reshape(-1, 1))))
            dev_spread = np.quantile(devs, 0.75) - np.quantile(devs, 0.25)
            print(line,' mean dev: ',np.mean(devs))
            print(line,' median dev: ',np.median(devs))
            
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


    # ax.plot(np.arange(len(lines)),-1.*std_obs,'-',color='grey',lw=3,alpha=0.6,\
    #         label='1-$\sigma$ spread in observed relation')
    # ax.plot(np.arange(len(lines)),std_obs,'-',color='grey',lw=3,alpha=0.6)
    xarray = np.arange(len(lines))*1.00000
    xarray[0] = xarray[0] - 0.5
    xarray[-1] = xarray[-1] + 0.5
    print(xarray)
    ax.fill_between(xarray,-1.*std_obs,std_obs,color='grey',alpha=0.8)
    ax.fill_between(xarray,-2.*std_obs,2*std_obs,color='grey',alpha=0.4)

    ax.set_xlim([-1,len(lines)+1])
    ax.plot(ax.get_xlim(),[0,0],'--k')
    ax.set_ylabel('$\Delta L_{x}$ [dex]')
    ax.set_xticks(np.arange(len(lines)))
    ax.set_xticklabels(lines)
    ax.legend()

    ax2 = ax.twiny()
    ax2.set_xlim([-1,len(lines)+1])
    ax2.set_xticks(np.arange(len(lines)+1)-1)
    ax2.set_xticklabels(crit)
    ax2.set_ylim([-7.5,4])
 
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/std_runs%s' % p.select,dpi=250)

def compare_CII_w_models(**kwargs):
    """ Plot line - SFR relation with other models
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    if not p.xlim: p.xlim = [-4,2]
    if not p.ylim: p.ylim = [4,9.5]

    fig,ax = plt.subplots(figsize=(8,6))
    ax.set_ylim(p.ylim)
    ax.set_xlim(p.xlim)


    # SIGAME Simba-100
    GR                  =   glo.global_results(sim_run=p.sim_runs[1],nGal=p.nGals[1])
    L_line              =   getattr(GR,'L_[CII]158_sun')
    lL_line             =   np.log10(L_line)
    SFR                 =   getattr(GR,'SFR')
    lSFR                =   np.log10(SFR)
    lSFR                =   lSFR[L_line > 0]
    lL_line             =   lL_line[L_line > 0]

    # ax.plot(np.log10(SFR),np.log10(L_line),'o',ms=4,color='midnightblue',\
    #     alpha=0.7,label='SIGAME with Simba-100',zorder=10)
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins = 100
    k = kde.gaussian_kde(np.column_stack([lSFR,\
                                         lL_line]).T)

    x, y = np.mgrid[lSFR.min():lSFR.max():nbins*1j, \
               4:lL_line.max():nbins*1j]
    z = k(np.vstack([x.flatten(), y.flatten()]))
    # To remove weird contour line:
    CS = ax.contour(x, y, z.reshape(x.shape),colors='midnightblue',levels=6)
    CS.collections[0].set_label('SIGAME with Simba-100')
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
    ax.plot(G19.logSFR,G19.logLCII,'k-',label='Popping+19',alpha=0.8)
    ax.fill_between(G19.logSFR,G19['log LCII 14th percentile'].values,\
                               G19['log LCII 86th percentile'].values,color='grey',alpha=0.4)


    # Padilla 2020
    P20 = pd.read_csv(p.d_data + 'models/DataFig6.csv',skiprows=1,sep=',',\
        names=['IDSim','GalID','SFR','LCIITotal'])
    P20['logLCII'] = np.log10(P20['LCIITotal'])
    P20['logSFR'] = np.log10(P20['SFR'])


    colors = ['darkred','orange','springgreen']
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
        CS = ax.contour(xP20, yP20, zP20.reshape(xP20.shape),colors=colors[i],levels=5)
        CS.collections[0].set_label('Padilla+20 '+IDSim)


    ax.set_xlabel('log '+getlabel('SFR'))
    ax.set_ylabel('log '+getlabel('[CII]158'))
    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[_] for _ in [8,7,9,10,11,0,1,2,3,4,5,6]]
    labels = [labels[_] for _ in [8,7,9,10,11,0,1,2,3,4,5,6]]
    plt.legend(handles,labels,fontsize=9,loc='upper left')
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/CII_SFR_w_models.png', format='png', dpi=300) # .eps for paper!

def resolution_test(names,labnames,**kwargs):
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

    props = ['M_star','M_gas','SFR']#,'Zsfr']
    x1 = np.log10(np.column_stack([GR1.M_star,GR1.M_gas,GR1.SFR]))#,GR1.Zsfr]))
    x2 = np.log10(np.column_stack([GR2.M_star,GR2.M_gas,GR2.SFR]))#,GR2.Zsfr]))
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

    # Look at properties
    fig, axs = plt.subplots(nrows=1, ncols=n_param, \
        figsize=(20,5),\
        gridspec_kw={'hspace': 0, 'wspace': 0.35})
    for iprop,prop in enumerate(props):
        grid_points = np.append(x1[:,iprop],x2[:,iprop])
        grid_points = np.linspace(np.min(grid_points),np.max(grid_points),6)
        grid_points_c = grid_points[1::] - (grid_points[1]-grid_points[0])/2.
        i_GR1 = np.zeros(len(grid_points_c)).astype(int)
        i_GR2 = np.zeros(len(grid_points_c)).astype(int)
        for i in range(len(grid_points_c)):
            min_xdist_cut = min_xdist[(x1[:,iprop] >= grid_points[i]) & (x1[:,iprop] <= grid_points[i+1])]
            try:
                i_GR1[i] = int(np.argmin(min_xdist_cut))
                i_GR2[i] = best_fit_in_GR2[i_GR1[i]]
            except:
                pass

        ax = axs[iprop]
        ax.plot(xdiff_GR1_values[:,:,iprop].flatten(),xdiff_raw[:,iprop],'o',ms=1,color='pink',alpha=0.4)
        ax.plot(xdiff_GR2_values[:,:,iprop].flatten(),xdiff_raw[:,iprop],'o',ms=1,color='cyan',alpha=0.3)
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.plot(1e6,1e6,'o',color='pink',alpha=0.9,ms=5,label='simba-100')
        ax.plot(1e6,1e6,'o',color='cyan',alpha=0.9,ms=5,label='simba-25')
        ax.plot(grid_points_c,x1[i_GR1,iprop]-x2[i_GR2,iprop],'o',ms=5)
        ax.set_ylabel('$\Delta$ ' + getlabel(prop))# + getlabel(line))
        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(getlabel(prop))

    # Look at line luminoisities
    cmap = plt.get_cmap('gist_rainbow_r')
    cmap = plt.get_cmap('brg')
    colors = [cmap(i) for i in np.linspace(0, 1, len(lines))]
    fig, axs = plt.subplots(nrows=1, ncols=n_param, \
        figsize=(20,5),\
        gridspec_kw={'hspace': 0, 'wspace': 0.35})
    for iprop,prop in enumerate(props):
        grid_points = np.append(x1[:,iprop],x2[:,iprop])
        grid_points = np.linspace(np.min(grid_points),np.max(grid_points),6)
        grid_points_c = grid_points[1::] - (grid_points[1]-grid_points[0])/2.
        i_GR1 = np.zeros(len(grid_points_c)).astype(int)
        i_GR2 = np.zeros(len(grid_points_c)).astype(int)
        for i in range(len(grid_points_c)):
            min_xdist_cut = min_xdist[(x1[:,iprop] >= grid_points[i]) & (x1[:,iprop] <= grid_points[i+1])]
            try:
                i_GR1[i] = int(np.argmin(min_xdist_cut))
                i_GR2[i] = best_fit_in_GR2[i_GR1[i]]
            except:
                pass

        ax = axs[iprop]
        xlim = ax.get_xlim(); ylim = ax.get_ylim()

        for iline,line in enumerate(lines):
            # print('\n %s' % line)
            # print(GR1['L_'+line+'_sun'].values[i_GR1])
            # print(GR2['L_'+line+'_sun'].values[i_GR2])
            # print((GR1['L_'+line+'_sun'].values[i_GR1]-GR2['L_'+line+'_sun'].values[i_GR2])/GR1['L_'+line+'_sun'].values[i_GR1])
            delta_percent = 100.*(GR1['L_'+line+'_sun'].values[i_GR1]-GR2['L_'+line+'_sun'].values[i_GR2])/GR1['L_'+line+'_sun'].values[i_GR1]
            delta_dex = np.log10(GR1['L_'+line+'_sun'].values[i_GR1]) - np.log10(GR2['L_'+line+'_sun'].values[i_GR2])
            # print(delta_percent)
            ax.plot(grid_points_c,delta_dex,marker='o',ms=5,lw=2,color=colors[iline],label=line)
        ax.set_ylabel(r'$\Delta$ luminosity [dex]')# + getlabel(line))
        ax.legend(fontsize=10)
        ax.set_xlabel('log '+getlabel(prop))

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/compare_Simba100_w_Simba25.png', format='png', dpi=300) # .eps for paper!


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
                figsize=(6,15),\
                gridspec_kw={'hspace': 0, 'wspace': 0})

    for i,ax in enumerate(axs):

        line_SFR(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[1],nGal=p.nGals[1],add_obs=p.add_obs,add=True,cb=True)
        line_SFR(line=lines[i],ax=ax,select=p.select,sim_run=p.sim_runs[0],nGal=p.nGals[0],add_obs=False,add=True,cb=False)


    plt.tight_layout()

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/lines_SFR_array_%s%s%s_%s%s_%s.png' % (p.ext,p.grid_ext,p.table_ext,p.sim_name,p.sim_run,p.select), format='png', dpi=300) # .eps for paper!

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
        plt.savefig(p.d_plot + 'luminosity/lines_FIR_array_%s%s%s_%s_%s.png' % (p.ext,p.grid_ext,p.table_ext,p.sim_name,p.sim_run), format='png', dpi=300) # .eps for paper!

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
        plt.savefig(p.d_plot + 'luminosity/lines_sSFR_array_%s%s%s_%s%s_%s.png' % (p.ext,p.grid_ext,p.table_ext,p.sim_name,p.sim_run,p.select), format='png', dpi=300) # .eps for paper!

def line_SFR(**kwargs):
    """ Plot line luminosity (in Lsun) against SFR
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    GR                  =   glo.global_results(sim_run=p.sim_run,nGal=p.nGal)
    
    marker              =   'o'
    if p.sim_run == p.sim_runs[0]: marker = '^'

    L_line              =   getattr(GR,'L_'+p.line+'_sun')#[380:400]#[0:100]
    SFR                 =   getattr(GR,'SFR')#[380:400]#[0:100]
    # G0_mw               =   getattr(GR,'F_FUV_mw')#[0:100]
    Zsfr                =   getattr(GR,'Zsfr')#[380:400]#[0:100]
    R_gas               =   getattr(GR,'R2_gas')#[380:400]#[0:100]
    M_H2                =   getattr(GR,'M_H2_R2_gas')#[380:400]#[0:100]
    print(len(L_line[L_line > 0]))
    # if 'CO' in p.line: p.select = 'Sigma_M_H2'

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
    # G0_mw = G0_mw[L_line > 0]
    L_line = L_line[L_line > 0]
    lSFR = np.log10(SFR)
    lL_line = np.log10(L_line)


    # print('%i data points ' % (len(L_line)))

    labs                =   {'_M10':'Mach=10 power-law',\
                            '_arepoPDF_ext':'AREPO parametric PDF with extinction',\
                            '_arepoPDF':'SIGAME v3'}
    lab                 =   labs[p.table_ext]

    if p.add:
        ax = p.ax
    else:
        fig,ax = plt.subplots(figsize=(8,6))

    if p.select == 'Sigma_M_H2':
        Sigma_M_H2 = M_H2/(np.pi*R_gas**2)/1e6 # per pc^-2
        print(M_H2.min(),Sigma_M_H2.min())
        print(M_H2.max(),Sigma_M_H2.max())

        m = ax.scatter(lSFR[np.argsort(Sigma_M_H2)],lL_line[np.argsort(Sigma_M_H2)],marker=marker,s=20,\
                   c=np.log10(Sigma_M_H2[np.argsort(Sigma_M_H2)]),vmin=-2.5,vmax=2.2,label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label=r'log $\Sigma_{H2}$ [M$_{\odot}$/pc$^2$]',size=15)
    if p.select == 'Zsfr':
        m = ax.scatter(lSFR,lL_line,marker=marker,s=20,\
                   c=Zsfr,label=lab,alpha=0.6,zorder=10,vmin=0.01,vmax=3)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label=r'$\langle Z\rangle_{\mathrm{SFR}}$ [Z$_{\odot}$]',size=15)
    if p.select == 'F_FUV_mw':
        m = ax.scatter(lSFR,lL_line,marker=marker,s=20,\
                   c=np.log10(G0_mw),label=lab,alpha=0.6,zorder=10)
        if p.cb:
            cbar = plt.colorbar(m,ax=ax)
            cbar.set_label(label='log ' + getlabel('G0'),size=15)

    # Label galaxies?
    # for i in range(len(SFR)):
    #     if SFR[i] > 0:
    #         ax.text(SFR[i],L_line[i],'G%i' % GR.gal_num[i],fontsize=7)

    if p.add_obs:
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
        p.ylim = [np.median(lL_line) - 6,np.median(lL_line) + 4]
        if p.line == '[OI]63': p.ylim = [np.median(lL_line) - 5,np.median(lL_line) + 4]
    ax.set_xlim(p.xlim)
    ax.set_ylim(p.ylim)
    ax.grid(ls='--')

    if p.savefig & (not p.add):
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/%s_SFR.png' % p.line, format='png', dpi=300) # .eps for paper!

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
        plt.savefig(p.d_plot + 'luminosity/%s_sSFR.png' % p.line, format='png', dpi=300) # .eps for paper!

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
        plt.savefig(p.d_plot + 'luminosity/%s_FIR.png' % p.line, format='png', dpi=300) # .eps for paper!

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
    plot.simple_plot(x1=M_gas,y1=L_line,ma1='x',fill1=True,xlog=True,ylog=True,lab1='Simba galaxies',\
        xlab=getlabel('lM_ISM'),\
        ylab='log(L$_{\mathrm{%s}}$ [K km$\,s^{-1}$ pc$^2$])' % p.line)

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/%s_Mgas.png' % p.line, format='png', dpi=300) # .eps for paper!

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
    M_gas               =   getattr(GR,'M_gas')

    # Plot
    plot.simple_plot(x1=M_gas,y1=L_line,ma1='x',fill1=True,xlog=True,ylog=True,lab1='Simba galaxies',\
        xlab=getlabel('lM_ISM'),\
        ylab='log(L$_{\mathrm{%s}}$ [K km$\,s^{-1}$ pc$^2$])' % p.line)

    if p.ylim:
        ax1 = plt.gca()
        ax1.set_ylim(p.ylim)

    if p.xlim:
        ax1 = plt.gca()
        ax1.set_xlim(p.xlim)

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
        plt.savefig(p.d_plot + 'luminosity/%s_Mstar.png' % p.line, format='png', dpi=300) # .eps for paper!

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
            ax.plot(10.**df.sSFR,df['L_'+line],'s',ms=5,mew=0,color='grey',alpha=0.8,label='Cormier+15 (dwarfs)')
            L_ul = df['L_'+line][df['L_'+line] < 0]
            if len(L_ul) > 0:
                ax.plot(10.**df.sSFR[df['L_'+line] < 0],-1.*L_ul,'s',ms=5,mew=0,color='grey',alpha=0.8)
                ax.errorbar(10.**df.sSFR[df['L_'+line] < 0],-1.*L_ul, elinewidth=1,\
                    uplims=np.ones(len(L_ul)),yerr=-1.*L_ul - 10.**(np.log10(-1.*L_ul)-0.3),color='grey',alpha=0.8,lw=0)
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

    # Kamenetzky et al. 2016
    try:
        df = pd.read_pickle('data/observations/AHIMSA_sample_lit')
        if p.plot: ax.plot(np.log10(df.SFR[(df.sizes < 47) & (df.SFR > 1e-4)]),\
            np.log10(df[line + '_Lsun'][(df.sizes < 47) & (df.SFR > 1e-4)]),'>',ms=6,fillstyle='none',mew=1,color='grey',alpha=0.8,label='Mixed z~0 sample [Kamenetzky+16]')
        L_obs = np.append(L_obs,df[line + '_Lsun'].values[(df.sizes < 47) & (df.SFR > 1e-4)])
        SFR_obs = np.append(SFR_obs,df.SFR.values[(df.sizes < 47) & (df.SFR > 1e-4)])
        if p.plot: print('%i galaxies from Kamenetzky+16 ' % (len(L_obs)))
        # print('min SFR: ',np.min(df.SFR.values[df.sizes < 47]))
    except:
        pass
    # Brauher et al. 2008
    try:
        df = pd.read_pickle('data/observations/Brauher_2008')
        if p.plot: ax.plot(np.log10(df.SFR),np.log10(df['L_%s' % line]),'o',fillstyle='none',ms=7,mew=1,color='grey',alpha=0.8,label='MS/SB galaxies [Brauher+08]')
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
                ax.plot(df.SFR,np.log10(df['L_%s' % line]),'+',zorder=0,ms=7,mew=2,color='grey',alpha=0.8,label='Dwarf galaxies [Cormier+15]')
                L_ul = df['L_'+line][df['L_'+line] < 0]
                if len(L_ul) > 0:
                    ax.plot(df.SFR[df['L_'+line] < 0],np.log10(-1.*L_ul),'+',zorder=0,ms=7,mew=2,color='grey',alpha=0.8)
                    ax.errorbar(df.SFR[df['L_'+line] < 0],np.log10(-1.*L_ul),color='grey',alpha=0.8,elinewidth=1,\
                        uplims=np.ones(len(L_ul)),\
                        yerr=np.log10(-1.*L_ul - 10.**(np.log10(-1.*L_ul)-0.3)),lw=0)
            if p.plot: print('%i galaxies from Cormier+15 ' % (len(df)))
            L_obs = np.append(L_obs,df['L_%s' % line].values)
            SFR_obs = np.append(SFR_obs,10.**df.SFR.values)
        except:
            pass

        # Accurso et al. 2017
        try:
            df = pd.read_pickle('data/observations/xCOLD_GASS_Accurso_2017')
            df = df.loc[np.argwhere(df['L_CO(1-0)'].values > 0).flatten()]
            if p.plot: ax.plot(np.log10(df['SFR']),df['L_%s' % line], 'd', zorder=0,ms=7,fillstyle='none',mew=1,color='grey',alpha=0.8,label='COLD GASS [Accurso+17]') #c=np.log10(A17['Z']), 
            L_obs = np.append(L_obs,10.**df['L_%s' % line].values)
            if p.plot: print('%i galaxies from Accurso+17 ' % (len(df)))
            SFR_obs = np.append(SFR_obs,df.SFR.values)
        except:
            pass

        # Diaz-Santos et al. 2013
        try:
            df = pd.read_pickle('data/observations/Diaz-Santos_2013')
            if p.plot: ax.plot(np.log10(df.SFR),np.log10(df['L_%s' % line]),'^',ms=6,zorder=0,fillstyle='none',mew=1,color='grey',alpha=0.8,label='LIRGs [Diaz-Santos+13]')
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
            if p.plot: ax.plot(np.log10(df.SFR),np.log10(df['L_%s' % line]),'<',ms=6,fillstyle='none',mew=1,color='grey',alpha=0.8,label='GOALS (U)LIRGs [Zhao+16]')
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

    logSFR = np.arange(np.min(np.log10(SFR_obs[SFR_obs > 0])) - 3,np.max(np.log10(SFR_obs[SFR_obs > 0])) + 3)
    # fit = np.polyfit(np.log10(SFR_obs[(L_obs > 0) & (SFR_obs > 0)]),\
    #     np.log10(L_obs[(L_obs > 0) & (SFR_obs > 0)]),1)
    # pfit = np.poly1d(fit)
    # L_fit = 10.**pfit(logSFR)

    # Make log-linear fit to SFR-binned luminosities
    SFRs = SFR_obs[(L_obs > 0) & (SFR_obs > 0)]
    Ls = L_obs[(L_obs > 0) & (SFR_obs > 0)]
    SFR_axis = np.linspace(np.log10(SFRs.min()),np.log10(SFRs.max()),7)
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
    if p.plot & plot_fit: ax.plot(logSFR,np.log10(L_fit),'--k',lw=2,zorder=0)

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
            plt.savefig(p.d_plot + 'SEDs/sed_%s%s_%i.png' % (p.sim_name,p.sim_run,p.gal_index), format='png', dpi=300) # .eps for paper!

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

    # GR                  =   glo.global_results()
    GR                  =   glo.global_results(sim_run=p.sim_runs[1],nGal=p.nGals[1])

    fig,ax1 = plt.subplots()
    L_CII = getattr(GR,'L_[CII]158_sun')
    L_CO = getattr(GR,'L_CO(1-0)_sun')
    Zsfr = getattr(GR,'Zsfr')
    sc = ax1.scatter(np.log10(L_CO), np.log10(L_CII), marker='o', c=np.log10(Zsfr), cmap='viridis', zorder=0,\
        vmin=np.log10(0.05), vmax=np.log10(3.1), \
        s=10, alpha=0.8, label='SIGAME v3 Simba-%s' % (p.sim_runs[1]))
    print('Min Zsfr in Simba sample: ',np.min(Zsfr))
    print('indices with L_CO < 1e0:')
    print(np.argwhere(L_CO < 1e0).flatten())
    print(L_CO[np.argwhere(L_CO < 1e0).flatten()])
    print(np.argwhere(L_CO > 1e4).flatten())

    GR                  =   glo.global_results(sim_run=p.sim_runs[0],nGal=p.nGals[0])

    fig,ax1 = plt.subplots()
    L_CII = getattr(GR,'L_[CII]158_sun')
    L_CO = getattr(GR,'L_CO(1-0)_sun')
    Zsfr = getattr(GR,'Zsfr')
    sc = ax1.scatter(np.log10(L_CO), np.log10(L_CII), marker='>', c=np.log10(Zsfr), cmap='viridis', zorder=0,\
        vmin=np.log10(0.05), vmax=np.log10(3.1), \
        s=10, alpha=0.8, label='SIGAME v3 Simba-%s' % (p.sim_runs[0]))

    # Observations
    K16 = pd.read_pickle('data/observations/AHIMSA_sample_lit')
    K16_LCII = K16['[CII]158_Lsun']
    K16_LCO = K16['CO(1-0)_Lsun']
    ax1.plot(np.log10(K16_LCO), np.log10(K16_LCII), '>', color='grey', ms=6, fillstyle='none',alpha=0.8, mew=1,zorder=0,\
        label='Mixed z~0 sample [Kamenetzky+16]')

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
    # ax.legend(handles, labels)
    handles = [handles[_] for _ in [2,0,3,4,1]]
    labels = [labels[_] for _ in [2,0,3,4,1]]
    plt.legend(handles,labels,loc='lower left',fontsize=10.5,frameon=True)

    ax1.set_xlim([-4,6.2])
    ax1.set_ylim([4,10])

    if not os.path.isdir(p.d_plot + 'luminosity/'): os.mkdir(p.d_plot + 'luminosity/')    
    plt.savefig(p.d_plot + 'luminosity/CO_vs_CII.png', dpi=300)

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

def line_ratio_per_pixel(ratio_name='NII',quant='ne',**kwargs):
    """ Plot line ratio against another quantity per pixel in moment0 map.
    """

    p = copy.copy(params)
    for key,val in kwargs.items():
        setattr(p,key,val)

    # Load sim and cell data
    gal_ob          =   gal.galaxy(p.gal_index)

    # for Malhar to fill int
    if ratio_name == 'NII':
        line1,line2 = '[NII]122','[NII]205'

    if not os.path.isdir(p.d_plot + 'physics/'): os.mkdir(p.d_plot + 'physics/')    
    plt.savefig(p.d_plot + 'physics/%s_%s_G%i' % (ratio_name,quant,p.gal_index),dpi=300)

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
    
    if p.sim_type == 'amr': location = p.d_XL_data + '/data/regular_cell_data/moment0map_%s%s' % (p.sim_name,p.sim_run) + '_' + plane + '_res' + str(res) + '.npy'
    if p.sim_type == 'sph': location = p.d_XL_data + '/data/regular_cell_data/moment0map_%s%s_%i' % (p.sim_name,p.sim_run,p.gal_index) + '_' + plane + '_res' + str(res) + '.npy'
        
    # Getting matrix with projected emmision values: 
    #momentmap = np.load(location, allow_pickle=True)
    #pdb.set_trace()
    try:
        momentmap = np.load(location, allow_pickle=True)
        print('Found stored momentmap data')
        print(location)
    except:
        print('Did not find stored momentmap data - creating')
        aux.convert_cell_data_to_regular_grid(res=res, plane=plane, gal_index=p.gal_index)
        momentmap = np.load(location, allow_pickle=True)
    
    n = momentmap[-1]
    momentmap = momentmap[:-1]
    
    # Getting the desired quantity to creat the momentmap:
    dictionary = {'m':0 ,'m_H2':1 ,'m_HI':2 ,'m_HII':3 ,'Z':4 , 'FUV':5, 'ne_mw':6, 'L_[CII]158':7 , 'L_[CI]610':8 , 'L_[CI]370':9 , 'L_[OI]145':10 , 'L_[OI]63':11 , 'L_[OIII]88':12 ,\
                  'L_[NII]122':13 , 'L_[NII]205':14 , 'L_CO(3-2)':15 , 'L_CO(2-1)':16 , 'L_CO(1-0)':17,'L_[OIV]25':18,'L_[NeII]12':19,'L_[NeIII]15':20,\
                  'L_[SIII]18':21, 'L_[FeII]25':22, 'L_H2_S(1)':23, 'L_H2_S(2)':24, 'L_H2_S(3)':25, 'L_H2_S(4)':26, 'L_H2_S(5)':27, 'L_H2_S(6)':28, 'L_H2_S(7)':29 }

    num = dictionary[quant]
    lumus = np.array(momentmap[:,3])
    lum = []
    mass = []
    metal = []
    for prop in lumus:
        if num == 1:
            lum.append(prop[num]/prop[0])
        else:
            lum.append(prop[num])
    lum = np.array(lum)

    if num != 1:
        lum = lum / (res**2)
    
    # Converting to Jy*km/s / kpc^2 units:
    if units == 'Jy':
        if num > 1:
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
    # grid = np.flipud(grid)
    # normal = mpl.colors.Normalize(vmin = min(lum), vmax = max(lum))

    # Setting 0 values to something very low
    grid[grid == 0] = 1e-30
    grid[np.isnan(grid)] = 1e-30

    # Default min,max values
    if not p.vmin: p.vmin = np.max(grid)/1e4
    if not p.vmax: p.vmax = 5*np.max(grid)
    
    if quant == 'Z':
        p.vmin = 0.05
        p.vmax = 3

    if p.add:
        fig,ax = plt.gcf(),p.ax
    else:
        fig = plt.figure(figsize=(6,6.15))
        ax = fig.add_axes([0.1, 0.01, 0.8, 0.8]) 
        ax.axis('equal')

    if not p.R_max:
        gal_ob = gal.galaxy(p.gal_index)
        p.R_max = gal_ob.R_max
    grid = np.flipud(grid)    
    if p.rotate:
        grid = np.rot90(grid)
        grid = np.rot90(grid)
    gal_ob          =   gal.galaxy(p.gal_index)
    cell_data       =   gal_ob.cell_data.get_dataframe()
    coords_sim      =   cell_data[['x','y','z']].values
    r = np.sqrt(np.sum(coords_sim * coords_sim, axis=1))
    cs = ax.imshow(grid, extent=(-np.max(r), np.max(r), -np.max(r), np.max(r)), norm=LogNorm(), \
                vmin=p.vmin, vmax=p.vmax, interpolation='nearest', cmap=p.cmap)
    ax.set_xlim([-p.R_max,p.R_max])
    ax.set_ylim([-p.R_max,p.R_max])

    if num == 0:
        #plt.title('mass density')
        labels = 'log surface density (M$_{\odot}$ / kpc$^2$)'        
    if num > 5: 
        #plt.title(quant + ' density')
        if units == 'Jy':
            labels = 'Jy${\cdot}$km/s / kpc$^2$'
        else:
            labels = 'log surface brightness density (L$_{\odot}$ / kpc$^2$)'
    if num == 4: 
        labels = 'log Z (Z$_{\odot}$)'
    if num == 5: 
        labels = 'log FUV flux (G$_{0}$)'

    if not p.add: plt.xlabel(plane[0].upper()+' (kpc)')
    if not p.add: plt.ylabel(plane[1].upper()+' (kpc)')

    formatter = mpl.ticker.LogFormatterExponent(10, labelOnlyBase=False, minor_thresholds=(100,20))
    if p.legend: 
        if not p.label: labels = ''
    print(p.legend,labels)
    cbar = fig.colorbar(cs, cmap=p.cmap, label=labels, pad=0, shrink=0.85)#0.5)#
    
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'moment0/'): os.mkdir(p.d_plot + 'moment0/')    
        plt.savefig(p.d_plot + 'moment0/moment0_%i%s%s' % (p.gal_index,p.sim_name,p.sim_run) + '_' + plane + '_res' + str(res) +'_'+ quant.replace('(','').replace(')','') + '.png',dpi=500)

    #if p.add: return(cbar)

def line_ratio_map(quant1='', quant2='', res=0.5, plane='xy', units='Jy', **kwargs):
    """
    Purpose 
    -------
    Makes line ratio map of a specific quantity.

    Parameters
    ----------
    gal_index: int
        Galaxy index, default: 0

    quant1: str
        The first line of the line ratio, default: ''
       
    quant2: str
        The second line of the line ratio, default: ''

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

    if p.sim_type == 'amr': location = p.d_XL_data + '/data/regular_cell_data/moment0map_%s%s' % (p.sim_name,p.sim_run) + '_' + plane + '_res' + str(res) + '.npy'
    if p.sim_type == 'sph': location = p.d_XL_data + '/data/regular_cell_data/moment0map_%s%s_%i' % (p.sim_name,p.sim_run,p.gal_index) + '_' + plane + '_res' + str(res) + '.npy'

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

    dictionary = {'m':0 ,'m_H2':1 ,'m_HI':2 ,'m_HII':3 ,'Z':4 , 'FUV':5, 'ne_mw':6, 'L_[CII]158':7 , 'L_[CI]610':8 , 'L_[CI]370':9 , 'L_[OI]145':10 , 'L_[OI]63':11 , 'L_[OIII]88':12 ,\
                  'L_[NII]122':13 , 'L_[NII]205':14 , 'L_CO(3-2)':15 , 'L_CO(2-1)':16 , 'L_CO(1-0)':17,'L_[OIV]25':18,'L_[NeII]12':19,'L_[NeIII]15':20,\
                  'L_[SIII]18':21, 'L_[FeII]25':22, 'L_H2_S(1)':23, 'L_H2_S(2)':24, 'L_H2_S(3)':25, 'L_H2_S(4)':26, 'L_H2_S(5)':27, 'L_H2_S(6)':28, 'L_H2_S(7)':29 }

    num1=dictionary[quant1]
    num2=dictionary[quant2]
    x = momentmap[:,1]
    y = momentmap[:,2]
    lumus = np.array(momentmap[:,3])
    line1=[]
    line2=[]
    for row in lumus:
        line1.append(row[num1])
        line2.append(row[num2])
    line1 = np.array(line1)
    line2 = np.array(line2)

    ratio = np.divide(line1, line2, out=np.zeros_like(line1), where=line2!=0)
    
    ratio = ratio.reshape(index1, index2)
    x = x.reshape(index1, index2)
    y = y.reshape(index1, index2)

    if p.add:
        fig,ax = plt.gcf(),p.ax #plot already available 
    else:
        fig, ax = plt.subplots(figsize=(10,8))
    cs = plt.pcolormesh(x, y, ratio, cmap=plt.cm.viridis)
    if not p.add:
        plt.title('Line Ratio map of ' + quant1 + "/" + quant2)
        plt.xlabel('X [kpc]')
        plt.ylabel('Y [kpc]')
    if p.add:labels:''
    fig.colorbar(cs, cmap=plt.cm.viridis, label= quant1 + " / " + quant2 +" [ L$_{\odot}$/ L$_{\odot}$]" )
    if p.R_max:
        ax.set_xlim([-p.R_max,p.R_max])
        ax.set_ylim([-p.R_max,p.R_max])
    if p.savefig:
        if not os.path.isdir(p.d_plot + 'lineratios/'): os.mkdir(p.d_plot + 'lineratios/')    
        plt.savefig(p.d_plot+'lineratios/%s%s_%i_%s_%s' % (p.sim_name,p.sim_run,p.gal_index,quant1.replace('L_',''),quant2.replace('L_',''))+ '_' + plane + '_res' + str(res) +'.png', dpi=500)
        
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
            ax1.plot([p.R_max[row_i]-19,p.R_max[row_i]-9],[-p.R_max[row_i]+8,-p.R_max[row_i]+8],lw=4,color='white')
            ax1.text(p.R_max[row_i]-20,-p.R_max[row_i]+10,'10 kpc',color='white',fontsize=12)
            # Remove axes ticks
            ax1.tick_params(axis='x',which='both',labelbottom=False,bottom=False,top=False)
            ax1.tick_params(axis='y',which='both',labelleft=False,bottom=False,top=False)     
            line_i += 1
            ax1.text(-p.R_max[row_i]+4,p.R_max[row_i]-8,quant.replace('L_',''),color='white',fontsize=18)
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

        ax1 = fig.add_subplot(gs1[row_i, 0])
        m = map_sim_property(add=True,ax=ax1,gal_index=gal_index,prop='m',R_max=p.R_max,vmin=9,vmax=12.9,\
                        pix_size_kpc=0.5,sim_type='simgas',cmap='viridis',log=True,colorbar=False,rotate=rotate,text=p.text)
        frame = plt.gca()
        #if row_i != 2: frame.axes.get_xaxis().set_visible(False)
        if row_i != 2: ax1.set_xlabel('')
        if row_i == 0:
            cbaxes = fig.add_axes([0.05, 0.93, 0.25, 0.01]) 
            cb = plt.colorbar(m, orientation='horizontal', cax = cbaxes)
            cbaxes.xaxis.set_ticks_position('top')
            cb.ax.set_title("$\Sigma_{\mathrm{gas}}$ [M$_{\odot}$ pc$^{-2}$]")
        # Make a size indicator
        #ax1.plot([p.R_max-15,p.R_max-5],[-p.R_max+5,-p.R_max+5],lw=4,color='white')
        #ax1.text(p.R_max-16,-p.R_max+7,'10 kpc',color='white',fontsize=12)
        # Remove axes ticks
        ax1.tick_params(axis='x',which='both',labelbottom=False)
        ax1.tick_params(axis='y',which='both',labelleft=False)

        ax1 = fig.add_subplot(gs1[row_i, 1])
        m = star_map(add=True,ax=ax1,R_max=p.R_max,vmin=6,vmax=9,\
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
        m =     FUV_map(add=True,ax=ax1,gal_index=gal_index,R_max=p.R_max,vmin=-10,vmax=3,select=p.select,cmap='twilight',colorbar=False,rotate=rotate)
        frame = plt.gca()
        #if row_i != 2: frame.axes.get_xaxis().set_visible(False)
        if row_i != 2: ax1.set_xlabel('')
        frame.axes.get_yaxis().set_visible(False)
        if row_i == 0:
            cbaxes = fig.add_axes([0.69, 0.93, 0.25, 0.01]) 
            cb = plt.colorbar(m, orientation='horizontal', cax = cbaxes) 
            cbaxes.xaxis.set_ticks_position('top')
            cb.ax.set_title('FUV flux [W/m$^2$/micron/arcsec$^2$]')
        # Make a size indicator
        if row_i == 2:
            print('Aaa')
            #ax1.text(p.R_max-16,-p.R_max+7,'10 kpc',color='w',fontsize=12)
            #ax1.plot([p.R_max-15,p.R_max-5],[-p.R_max+5,-p.R_max+5],lw=4,color='w')
        else:
            ax1.text(p.R_max-16,-p.R_max+7,'10 kpc',color='k',fontsize=12)
            ax1.plot([p.R_max-15,p.R_max-5],[-p.R_max+5,-p.R_max+5],lw=4,color='k')
        # Remove axes ticks
        ax1.tick_params(axis='x',which='both',labelbottom=False)
        ax1.tick_params(axis='y',which='both',labelleft=False)

        # s = segs
    gs1.update(top=0.92,bottom=0.02,left=0.02,right=0.98)

    if p.savefig:
        if not os.path.isdir(p.d_plot + 'pretty/'): os.mkdir(p.d_plot + 'pretty/')
        plt.savefig('plots/pretty/mass_FUV_maps_%s%s.png' % (p.sim_name,p.sim_run),format='png',dpi=200)
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
    if foo == 'lZ': return 'log($Z$ [Z$_{\odot}$])'
    if foo == 'Zmw': return r"$\langle Z'\rangle_{\mathrm{mass}}$"
    if foo == 'Zsfr': return r"$\langle Z'\rangle_{\mathrm{SFR}}$"
    if foo == 'Zstar': return r"$\langle Z'\rangle_{\mathrm{stars}}$"
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
    if foo == 'Mgmc': return '$m_{\mathrm{GMC}}$ [M$_{\odot}$]'
    if foo == 'm_mol': return '$m_{\mathrm{mol}}$ [M$_{\odot}$]'
    if foo == 'm_dust': return '$m_{\mathrm{dust}}$ [M$_{\odot}$]'
    if foo == 'M_dust': return 'M$_{\mathrm{dust}}$ [M$_{\odot}$]'
    if foo == 'M_star': return 'M$_{\mathrm{*}}$ [M$_{\odot}$]'
    if foo == 'M_gas': return 'M$_{\mathrm{gas}}$ [M$_{\odot}$]'
    if foo == 'M_ISM': return 'M$_{\mathrm{ISM}}$ [M$_{\odot}$]'
    if foo == 'lM_ISM': return 'log(M$_{\mathrm{ISM}}$ [M$_{\odot}$])'
    if foo == 'G0': return "FUV flux [G$_{0}$]"
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

    if foo == 'R_NIR_FUV': return 'NIR/FUV flux ratio'
    if foo == 'lR_NIR_FUV': return 'log NIR/FUV flux ratio'

    if foo == 'L_FIR': return 'L$_{\mathrm{FIR}}$ [L$_{\odot}$]'


    if foo == 'cell_size': return 'Cell size [kpc])'
    if foo == 'lcell_size': return 'log Cell size [kpc])'

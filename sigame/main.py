###     Module: main.py of SIGAME                   ###

# Import other SIGAME modules
import sigame.backend as backend
import sigame.global_results as glo
import sigame.auxil as aux
import sigame.plot as plot
import pdb

#===============================================================================
"""  Load parameters """
#-------------------------------------------------------------------------------

global p
p                      =   aux.load_parameters()

#===============================================================================
"""  Initialize a global result object (GR) """
#-------------------------------------------------------------------------------

def run():
    '''Main controller that determines what tasks will be carried out
    '''
    
    GR                      =   glo.global_results(verbose=True)

    print('\n** This is the main controller running SIGAME for the given galaxy sample **')
    print('(Number of galaxies in selection: %s )' % GR.N_gal)
    if p.ow:
        print('OBS: Overwrite is ON, will overwrite any existing files')
    if not p.ow:
        print('OBS: Overwrite is OFF, will not overwrite any existing files')

    if p.step1_setup_SKIRT:            backend.setup_SKIRT(GR)
    if p.step1_read_SKIRT:             backend.read_SKIRT(GR)
    if p.step2_grid_gas:               backend.grid_gas(GR)
    if p.step3_setup_Cloudy_grid:      backend.setup_Cloudy_grid(GR)
    if p.step3_run_Cloudy:             backend.run_Cloudy(GR)
    if p.step3_combine_Cloudy:         backend.combine_Cloudy(GR)
    if p.step3_complete_Cloudy:        backend.complete_Cloudy(GR)
    if p.step3_read_Cloudy_grid:       backend.read_Cloudy_grid(GR)
    if p.step3_make_Cloudy_table:      backend.make_Cloudy_table(GR)
    if p.step4_interpolate:            backend.interpolate(GR)

def print_results():
    '''Print main global results
    '''

    GR                      =   glo.global_results(verbose=True)
    GR.print_results()




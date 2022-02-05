__all__ = ["main","param","plot"]

import os
import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 or a more recent version is required.")

import matplotlib
import socket, getpass
host = socket.gethostname()
username = getpass.getuser()
import numpy as np
import pandas as pd
import pdb
import re
import subprocess as sub
import matplotlib.pyplot as plt

print('' + '\n' +
      '     =======================================================  ' + '\n' +
      '     ' + '\n' +
      '     .oOOOo.  ooOoOOo  .oOOOo.     Oo    Oo      oO o.OOoOoo  ' + '\n' +
      '     o     o     O    .O     o    o  O   O O    o o  O        ' + '\n' +
      '     O.          o    o          O    o  o  o  O  O  o        ' + '\n' +
      '      `OOoo.     O    O         oOooOoOo O   Oo   O  ooOO     ' + '\n' +
      '           `O    o    O   .oOOo o      O O        o  O        ' + '\n' +
      '            o    O    o.      O O      o o        O  o        ' + '\n' +
      '     O.    .O    O     O.    oO o      O o        O  O        ' + '\n' +
      "      `oooO'  ooOOoOo   `OooO'  O.     O O        o ooOooOoO  " + '\n' +
      '     ' + '\n' +  
      '     =======================================================  ' + '\n' +
      '      SImulator of GAlaxy Millimeter/submillimeter Emission' + '\n' +
      '---- A code to simulate the far-IR emission lines of the ISM  ----' + '\n' +
      '------------- in galaxies from hydrodynamical codes --------------' + '\n' +
      '----- for the interpretation and prediction of observations. -----' + '\n' +
      '---- Contact: Karen Olsen, kpolsen (at) protonmail.com (2021) ----' + '\n' +
      '' + '\n')
# style from http://www.kammerl.de/ascii/AsciiSignature.php
# (alternatives: epic, roman, blocks, varsity, pepples, soft, standard, starwars)

################ Some instructions ######################

# Remember to select and edit a parameter file:
params_file = os.getcwd() + '/parameters.txt'

try:
    f = open(params_file, encoding="utf8")
    print('Reading parameter file: [' + params_file + '] ... ')
except IOError:
    sys.exit('Could not read parameter file: [' + params_file + '] ... ')

# Create parameter object
from .param import *
p = read_params(params_file)

# Matplotlib style file
plt.style.use(p.parent + 'sigame/pretty.mplstyle')

print('\n--------------------------------------------------------------')

# Import main SIGAME modules
from sigame.main import *

# Check that modules are of the right version
# aux.check_version(np,[1,12,0])

print('\nReady to continue!\n')

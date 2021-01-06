## SÍGAME: SImulator of GAlaxy Millimeter/submillimeter Emission (v3)

###
Jump straigt to the auto-generated code documentation:
https://kpolsen.github.io/SIGAME_dev/ (not updated from v2 yet)

### Description
This is a code to simulate the emission lines of the ISM in galaxies from hydrodynamical codes for the interpretation and prediction of observations.

### Obtaining the SÍGAME code
Clone this repository to an empty folder on your computer by giving the following command in the terminal:
``` 
git clone https://github.com/kpolsen/SIGAME.git
```
For updating from the master branch:
``` 
git pull origin master
```
Before making any changes to the code, please switch to your own personal branch with a name of your choice:
```
git branch -b NAME-OF-YOUR-BRANCH
```
For making a pull request with all your recent changes:
``` 
git add .
git commit -m "commmit message"
git push --set-upstream origin NAME-OF-YOUR-BRANCH
git request-pull origin/master NAME-OF-YOUR-BRANCH
```
where NAME-OF-YOUR-BRANCH is the name of your branch.

### Running SÍGAME

All modules of SÍGAME are found in the sigame/ directory and loaded into python with:
``` 
import sigame as si
```
Importing sigame will ask you what redshift you're working at and who you are to set up a path for external large files not tracked by github. To change (or add) the path for your user, go into parameters.txt and edit the relevant part:
```
Directory for large data [d_XL_data]
PATH_TO_LARGE_DATA
```
Depending on the chosen redshift, a specific paramter file while be loaded and must be supplied with the general parameters for SÍGAME (redshift, resolution of datacubes etc.). See 'parameters_z0.txt' for an example, copy to parameters.txt and modify as needed. Whenever you change the parameter file, remember to reload sigame in order to get those changes into the code:
``` 
reload(si)
```
I also recommend changing the default jupyter notebook settings to automatically reload everything each time you use a SÍGAME module in that notebook. This can be done by adding the following two lines to the beginning of your notebook:
```
%load_ext autoreload
%autoreload 2
```
OR adding the following two lines to ~/.ipython/profile_default/ipython_config.py (create this file if it doesn't exit):
```
c.InteractiveShellApp.extensions = ['autoreload'] 
c.InteractiveShellApp.exec_lines = ['%autoreload 2']

```
With SÍGAME imported, you run the code with the command:
``` 
si.run()
```
which will call the program run() in the sigame/backend.py module. What will be executed depends on what you select at the bottom of the parameter file. For example setting:
```
BACKEND TASKS
+1 step2_ISRF_RT
-1 step3_grid_gas
-1 step4_Cloudy_tables
-1 step5_interpolate
```
will only generate the SKIRT input files to calculate the interstellar radiation field (ISRF).

### Collaborators (active in developing the code)
Karen Pardos Olsen, karenolsen (at) arizona.edu

### References
  - 2020: Leung, T. K. D., Olsen, K. P., Somerville, R. S., Davé, R., Greve, T. R., Hayward, C. C., Narayanan, D. Popping, P.: "Predictions of the L_[CII]–SFR and [CII] Luminosity Function at the Epoch of Reionization", ApJ 905 102, [IOP link](https://iopscience.iop.org/article/10.3847/1538-4357/abc25e)
  - 2018: Olsen, K. P., Greve, T. R., Narayanan, D., Thompson, R., Davé, R., Niebla Rios, L., Stawinsi, S.: "Erratum: SIGAME Simulations of the [CII], [OI], and [OIII] Line Emission from Star-forming Galaxies at z~6 (2018)", ApJ 857 2, [ADS link](http://adsabs.harvard.edu/abs/2018ApJ...857..148O)
  - 2017: Olsen, K. P., Greve, T. R., Narayanan, D., Thompson, R., Davé, R., Niebla Rios, L., Stawinsi, S.: "SIGAME Simulations of the [CII], [OI], and [OIII] Line Emission from Star-forming Galaxies at z~6 (2017)", ApJ 846 2, arXiv: [1708.04936](https://arxiv.org/abs/1708.04936)
  - 2017: Olsen, K. P., Greve, T. R., Brinch, C., Sommer-Larsen, J., Rasmussen, J., Toft, S., Zirm, A.: "SImulator of GAlaxy Millimeter/submillimeter Emission (SIGAME): CO emission from massive z=2 main sequence galaxies", arXiv: [1507.00012](http://arxiv.org/abs/1507.00012)
  - 2015: Olsen, K. P., Greve, T. R., Narayanan, D., Thompson, R., Toft, S. Brinch, C.: "Simulator of Galaxy Millimeter/Submillimeter Emission (SIGAME): The [CII]-SFR Relationship of Massive z=2 Main Sequence Galaxies", MNRAS 457 3, arXiv: [1507.00362](http://arxiv.org/abs/1507.00362)

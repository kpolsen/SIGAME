###========================================
#!/bin/bash

#PBS -q windfall
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb
#PBS -N sigame
#PBS -W group_list=kolsen
#PBS -l walltime=02:00:00
#PBS -j oe

cd <path_to_current_working_directory>

skirt simba_25Mpc_G$PBS_ARRAY_INDEX.ski


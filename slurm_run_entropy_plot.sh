#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=TTA_run   # sensible name for the job
#SBATCH --mem=128G                 # Default memory per CPU is 3GB.
#SBATCH --partition=orion         # Use hugemem if you need > 48 GB (check the orion documentation)
#SBATCH --constraint=avx2
#SBATCH --mail-user=anine.lome@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/entropy_plot-%A.out
#SBATCH --error=outputs/entropy_plot-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity


singularity exec --nv tensorflow_gpu_analysis.sif python -u $1 ${@:2}

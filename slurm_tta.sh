#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=uncertainty   # sensible name for the job
#SBATCH --mem=64G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --mail-user=anine.lome@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/tta-%A-%a.out
#SBATCH --error=outputs/tta-%A-%a.out
#SBATCH --input=inputs/all-no.in

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# If data files aren't copied, do so
#!/bin/bash
if [ $# -lt 2 ];
    then
    printf "Not enough arguments - %d\n" $#
    exit 0
    fi

echo "Finished seting up files."


# Run experiment
# export ITER_PER_EPOCH=200
# export NUM_CPUS=4
# export RAY_ROOT=$TMPDIR/ray
export MAX_SAVE_STEP_GB=0
# rm -rf $TMPDIR/ray/*
singularity exec --nv deoxys.sif python -u run_tta.py $1 $2 --iter $SLURM_ARRAY_TASK_ID

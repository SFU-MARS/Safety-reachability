#!/bin/bash
#SBATCH -J tt     # Name that will show up in squeue
#SBATCH --gres=gpu:1         # Request 4 GPU "generic resource"
#SBATCH --time=7-00:00       # Max job time is 3 hours
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=long     # long partition (allows up to 7 days runtime)
#SBATCH --nodelist=cs-venus-01   # if needed, set the node you want (similar to -w xyz)
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=6
#SBATCH --qos=overcap


# Your experiment setup logic here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wpnr
hostname
echo $CUDA_AVAILABLE_DEVICES
#export OMP_NUM_THREADS=1

# Note the actual command is run through srun
srun python -u /local-scratch/tara/project/WayPtNav-reachability/executables/rgb/resnet50/rgb_waypoint_trainer.py --job-dir=/local-scratch/tara/project/WayPtNav-reachability/log/train --params=/local-scratch/tara/project/WayPtNav-reachability/params/rgb_trainer/sbpd/projected_grid/resnet50/rgb_waypoint_trainer_finetune_params.py --device=0


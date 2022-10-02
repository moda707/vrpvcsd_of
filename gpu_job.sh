#!/bin/bash
#SBATCH --account=def-errico
#SBATCH --mail-user=moh.dastpak@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --gpus-per-node=v100:1              # Number of GPUs (per node)
#SBATCH --mem=3G               # memory (per node)
#SBATCH --time=7-00:00            # time (DD-HH:MM)
#SBATCH --job-name=mv_0_75
#SBATCH --output=reports/min/%x
module load python/3.6
source ~/modaenv/bin/activate
python -u main.py --density 0 --q 75 --dl 240.45 --model VRPSCD --operation train_min --trials 3000000 --preempt_action 1 --base_address Models/Min/VRPSCD/

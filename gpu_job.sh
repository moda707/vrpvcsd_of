#!/bin/bash
#SBATCH --account=def-errico
#SBATCH --mail-user=moh.dastpak@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mem=3G               # memory (per node)
#SBATCH --time=1-12:00            # time (DD-HH:MM)
#SBATCH --job-name=mv_0_75
#SBATCH --output=reports/Obs/%x
module load python/3.6
source ~/modaenv/bin/activate
python -u main.py --density 0 --q 25 --obs 0 --dl 240.45 --model VRPSCD --operation train_min --trials 3000000 --preempt_action 1 --base_address Models/Obs/VRPSCD/

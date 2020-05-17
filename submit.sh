#!/bin/bash
#SBATCH -p syyeung
#SBATCH --gres gpu:1
#SBATCH -o out_main
#SBATCH --mem 35gb
#SBATCH --time=2:00:00


#ml load cudnn/7.4.1.5
#ml load cuda/9.0.176
#ml load eigen/3.3.3

module load system
module load cairo/1.14.10

#conda activate simclr
 
export DETECTRON2_DATASETS=/scratch/users/zzweng/datasets
#export PYTHONPATH=/home/users/zzweng/unsupervised_segmentation/detectron2/:$PYTHONPATH

#python models/rpn.py

python run_lvis.py

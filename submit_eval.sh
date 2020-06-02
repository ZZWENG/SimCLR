#!/bin/bash
#SBATCH -p syyeung
#SBATCH --gres gpu:6
#SBATCH -o eval_lvis_101
#SBATCH --mem 20gb

#ml load cudnn/7.4.1.5
#ml load cuda/9.0.176
#ml load eigen/3.3.3
#ml load py-tensorflow/1.9.0_py27#
module load system
module load cairo/1.14.10
#module load cuda/10.1.
#module list

export DETECTRON2_DATASETS=/scratch/users/zzweng/datasets
#export PYTHONPATH=/home/users/zzweng/unsupervised_segmentation/detectron2/:$PYTHONPATH
#python train.py

python eval_script.py --num-gpus 6 --eval-only #--config-file /home/users/zzweng/unsupervised_segmentation/detectron2/configs/LVIS-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml MODEL.WEIGHTS "run/checkpoints/hyp=False_zdim=64_loss=nce/rpn_model_8000.pth"

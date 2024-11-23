#!/bin/bash

# python main_vif.py --mode training --dataset_path ~/AUTOAIF_DATA/loos_model/     --save_checkpoint_path ~/AUTOAIF_DATA/weights/spatial30%  --loss_weights 0.3 0.7 0 --epochs 200 --batch_size 1

# python main_vif.py --mode training --dataset_path ~/AUTOAIF_DATA/loos_model/     --save_checkpoint_path ~/AUTOAIF_DATA/weights/spatial50%  --loss_weights 0.5 0.5 0 --epochs 200 --batch_size 1

python main_vif.py --mode training --dataset_path /media/network_mriphysics/USC-PPG/AI_training/loos_model --save_checkpoint_path /media/network_mriphysics/USC-PPG/AI_training/weights/test_MAE2  --loss_weights 0 1 0 --epochs 200 --batch_size 1

#!/bin/bash

echo Run 1
python main_vif.py --mode training --dataset_path /media/network_mriphysics/USC-PPG/AI_training/loos_model \
    --save_checkpoint_path /media/network_mriphysics/USC-PPG/AI_training/weights/run1_fullVOL.h5 \
    --loss_weights 0 0 1


echo Run 2
python main_vif.py --mode training --dataset_path /media/network_mriphysics/USC-PPG/AI_training/loos_model \
    --save_checkpoint_path /media/network_mriphysics/USC-PPG/AI_training/weights/run2_fullMAE.h5 \
    --loss_weights 0 1 0

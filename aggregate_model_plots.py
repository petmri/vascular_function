# This bad boy will take multiple model paths and use each to plot a curve of a single image prediction


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorrt
import nibabel as nib
from tensorflow import keras
from tensorflow.keras import layers

from model_vif import *
from utils_vif import *

# List of model weight paths
model_paths = ['/home/mrispec/AUTOAIF_DATA/weights/run2_fullMAE/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/dataaug_preresampling_axial_214/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/normMAE/model_weight.h5', 
               '/home/mrispec/AUTOAIF_DATA/weights/origin_axial/model_weight.h5']

# strip the model weight paths to get the model names
model_names = [os.path.basename(path[:-16]) for path in model_paths]
print(model_names)

# Load models
# models = []
# for path in model_paths:
#     model = keras.models.load_model(path)
#     models.append(model)

# Path to image folder
image_folder = '/home/mrispec/AUTOAIF_DATA/loos_model/test/images'
output_folder = '/home/mrispec/AUTOAIF_DATA/results/aggregate'

def process_image(image_path):
    # Load image
    volume_img = nib.load(image_path)
    print(volume_img.shape)
    volume_data = volume_img.get_fdata()

    vol_pre = preprocessing(volume_data)


    vfs = []
    for model_weight in model_paths:
        model = unet3d(img_size = (X_DIM, Y_DIM, Z_DIM, T_DIM),\
                        learning_rate = 1e-3,\
                        learning_decay = 1e-9)
        model.trainable = False
        model.load_weights(model_weight)

        y_pred_mask, y_pred_vf, _ = model.predict(vol_pre)
        # y_pred_mask = y_pred_mask > 0.8
        y_pred_mask = y_pred_mask.astype(float)

        mask = resize_mask(y_pred_mask, volume_data)

        mask = mask.squeeze()
        mask_img = nib.Nifti1Image(mask, volume_img.affine)
        # masks.append(mask_img)
        vf = y_pred_vf
        vfs.append(vf)
    # Create subplot of prediction graphs
    x = np.arange(len(vfs[0][0]))
    plt.figure(figsize=(15,7), dpi=250)
    for i, vf in enumerate(vfs):
        # plt.subplot(1, len(vfs), i+1)
        # plt.imshow(mask.get_fdata()[:,:,0,0])
        # plt.subplot(1,2,1)
        plt.title('Vascular Function (VF): ' + image_path)
        # set axis titles
        plt.xlabel('t-slice', fontsize=19)
        plt.ylabel('Intensity:Baseline', fontsize=19)
        # x = np.arange(len(vf[0]))
        plt.yticks(fontsize=19)
        plt.xticks(fontsize=19)
        # plt.plot(x, vf[0] / vf[0][0], 'r', label='Auto', lw=3)
        plt.plot(x, vf[0] / vf[0][0], label=model_names[i], lw=3)
    
    # remove everything after test
    mask_dir = '/'.join(image_path.split('/')[:-2])
    mask_dir = mask_dir + '/masks'
    file = image_path.split('/')[-1].split('.')[0]
    path = image_path[:-len(image_path.split('/')[-1])-1]
    # plot manual mask if it exists
    if os.path.isfile(mask_dir + '/' + file + '.nii') or os.path.isfile(path + '/aif.nii'):
        if os.path.isfile(mask_dir + '/' + file + '.nii'):
            img = nib.load(mask_dir + '/' + file + '.nii')
            mask = np.array(img.dataobj)
            dce = nib.load(path + '/' + file + '.nii')
        elif os.path.isfile(path + '/aif.nii'):
            img = nib.load(path + '/aif.nii')
            mask = np.array(img.dataobj)
            mask = mask.squeeze()
            dce = nib.load(path + '/' + file + '.nii.gz')
        mask_crop = scipy.ndimage.zoom(mask, (X_DIM / mask.shape[0], Y_DIM / mask.shape[1], Z_DIM / mask.shape[2]), order=1)

        dce_data = np.array(dce.dataobj)
        dce_data = (dce_data - np.min(dce_data)) / ((np.max(dce_data) - np.min(dce_data)))

        dce_crop = scipy.ndimage.zoom(dce_data, (X_DIM / dce_data.shape[0], Y_DIM / dce_data.shape[1], Z_DIM / dce_data.shape[2], T_DIM / dce_data.shape[3]), order=1)
        mask_crop = mask_crop.reshape(X_DIM, Y_DIM, Z_DIM, 1)

        roi_ = mask_crop * dce_crop
        num = np.sum(roi_, axis = (0, 1, 2), keepdims=False)
        den = np.sum(dce_crop, axis = (0, 1, 2), keepdims=False)
        intensities = num/(den+1e-8)
        intensities = np.asarray(intensities)
        plt.plot(x, intensities / intensities[0], 'b', label='Manual', lw=3)
    
    plt.legend(loc="upper right", fontsize=16)
    plt.savefig(os.path.join(output_folder, file + '_curve.png'))
    # plt.savefig(os.path.join(output_folder, image_path + '_curve.svg'), bbox_inches="tight")
    plt.close()



for image in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image)
    process_image(image_path)

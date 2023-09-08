# This bad boy will take multiple model paths and use each to plot a curve of a single image prediction


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorrt
import nibabel as nib
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

from model_vif import *
from utils_vif import *

# List of model weight paths
model_paths = ['/home/mrispec/AUTOAIF_DATA/weights/run2_fullMAE/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/3MAEweights/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/gaggar_weights/model_weight.h5',
               '/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-1/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-8_cos/model_weight.h5']

# strip the model weight paths to get the model names
model_names = [os.path.basename(path[:-16]) for path in model_paths]
print(model_names)

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
    for i, model_weight in enumerate(model_paths):
        model = unet3d(img_size = (X_DIM, Y_DIM, Z_DIM, T_DIM),\
                        learning_rate = 1e-3,\
                        learning_decay = 1e-9)
        model.trainable = False
        model.load_weights(model_weight)

        y_pred_mask, y_pred_vf, _ = model.predict(vol_pre)
        y_pred_mask = y_pred_mask.astype(float)

        mask = resize_mask(y_pred_mask, volume_data)
        mask_thresholded = mask > 0.95

        mask = mask.squeeze()
        volume_data_thresholded = volume_data * mask_thresholded
        mask_thresholded = mask_thresholded.squeeze()
        mask_img = nib.Nifti1Image(mask_thresholded.astype(float), volume_img.affine)
        nib.save(mask_img, os.path.join(output_folder, image_path.split('/')[-1].split('.')[0] + '_' + model_names[i] + '_mask.nii'))

        # get new curve from masked volume
        vf = np.zeros((1, T_DIM))
        num = np.sum(volume_data_thresholded, axis=(1, 2, 3))
        den = np.sum(mask_thresholded, axis=(0, 1, 2))
        vf = num / (den + 1e-8)
        # vf = y_pred_vf
        vfs.append(vf)
    
    # Create subplot of prediction graphs
    x = np.arange(len(vfs[0][0]))
    plt.figure(figsize=(15,7), dpi=100)
    colors = ['r', 'g', 'c', 'm', 'y', 'k']
    for i, vf in enumerate(vfs):
        plt.title('Vascular Function (VF): ' + image_path)
        # set axis titles
        plt.xlabel('t-slice', fontsize=19)
        plt.ylabel('Intensity:Baseline', fontsize=19)
        plt.yticks(fontsize=19)
        plt.xticks(fontsize=19)
        plt.plot(x, vf[0] / vf[0][0], label=model_names[i], lw=2, color=colors[i])
    
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

        dce_data = np.array(dce.dataobj)
        dce_data = (dce_data - np.min(dce_data)) / ((np.max(dce_data) - np.min(dce_data)))

        # add 4th axis to mask
        mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1)

        roi_ = mask * dce_data
        num = np.sum(roi_, axis = (0, 1, 2), keepdims=False)
        den = np.sum(dce_data, axis = (0, 1, 2), keepdims=False)
        intensities = num/(den+1e-8)
        intensities = np.asarray(intensities)
        plt.plot(x, intensities / intensities[0], 'b', label='Manual', lw=3)
    
    plt.legend(loc="upper right", fontsize=16)
    plt.savefig(os.path.join(output_folder, file + '_curve.png'), bbox_inches="tight")
    plt.close()


for image in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image)
    process_image(image_path)

# Path to the folder containing the individual result images
results_folder = output_folder

# Output path for the final mosaic image
final_mosaic_path = results_folder + "/final_mosaic.png"

# Get a list of all result image file paths
result_image_paths = [os.path.join(results_folder, filename) for filename in os.listdir(results_folder) if filename.endswith(".png")]

# Determine the number of images per row in the mosaic
images_per_row = 5  # You can adjust this based on your preference

# Open all result images and calculate dimensions for the final mosaic
result_images = [Image.open(image_path) for image_path in result_image_paths]
image_width, image_height = result_images[0].size
mosaic_width = image_width * images_per_row
mosaic_height = image_height * ((len(result_images) - 1) // images_per_row + 1)

# Create a new blank mosaic image
final_mosaic = Image.new("RGB", (mosaic_width, mosaic_height), (255, 255, 255))

# Paste each result image onto the mosaic
for i, result_image in enumerate(result_images):
    row = i // images_per_row
    col = i % images_per_row
    x_offset = col * image_width
    y_offset = row * image_height
    final_mosaic.paste(result_image, (x_offset, y_offset))

# Save the final mosaic image
final_mosaic.save(final_mosaic_path)

print("Final mosaic image created:", final_mosaic_path)

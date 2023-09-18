# This bad boy will take multiple model paths and use each to plot a curve of a single image prediction


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorrt
import nibabel as nib
from matplotlib import colors as mcolors
from PIL import Image
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers

from model_vif import *
from utils_vif import *

# List of model weight paths
# model_paths = ['/home/mrispec/AUTOAIF_DATA/weights/run2_fullMAE/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/gaggar_weights/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-1/model_weight.h5',
#                '/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-8_cos/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-9_maedice/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-11_mse/model_weight.h5', 
#                '/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-11_huber/model_weight.h5']
model_paths = ['/home/mrispec/AUTOAIF_DATA/weights/run2_fullMAE/model_weight.h5', '/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-11_mse/model_weight.h5','/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-11_huber/model_weight.h5',
                '/home/mrispec/AUTOAIF_DATA/weights/gaggar_9-15/model_weight.h5']

# strip the model weight paths to get the model names
model_names = [os.path.basename(path[:-16]) for path in model_paths]
print(model_names)

# Path to image folder
image_folder = '/home/mrispec/AUTOAIF_DATA/loos_model/test/images'
output_folder = '/home/mrispec/AUTOAIF_DATA/results/gaggar_9-15'

def process_image(image_path):
    # Load image
    volume_img = nib.load(image_path)
    print(volume_img.shape)
    volume_data = volume_img.get_fdata()
    vol_pre = preprocessing(volume_data)

    vfs = []
    masks = []
    peak_index = []
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
        masks.append(mask_thresholded)
        peak_index.append(np.argmax(vf))

    peak_index = max(set(peak_index), key=peak_index.count)
    # plot prediction
    x = np.arange(len(vfs[0][0]))
    plt.figure(figsize=(15,7), dpi=100)
    colors = ['r', 'g', 'c', 'm', 'y', 'k', 'tab:orange']
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
    # plot manual curve if it exists
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
        den = np.sum(mask, axis = (0, 1, 2), keepdims=False)
        intensities = num/(den+1e-8)
        intensities = np.asarray(intensities)
        plt.plot(x, intensities / intensities[0], 'b', label='Manual', lw=3)
    
    plt.legend(loc="upper right", fontsize=16)
    plt.savefig(os.path.join(output_folder, file + '_curve.png'), bbox_inches="tight")
    plt.close()

    # overlay mask on image
    z_rois = []
    try:
        img = nib.load(path + '/' + file + '.nii')
    except:
        img = nib.load(path + '/' + file + '.nii.gz')
    img_data = img.get_fdata()
    img_data = img_data.squeeze()
    img_data = np.rot90(img_data, k=1, axes=(0,1))
    # load manual AIF if it exists
    if os.path.isfile(mask_dir + '/' + file + '.nii'):
        aif_img = nib.load(mask_dir + '/' + file + '.nii')
        aif_mask = np.array(aif_img.dataobj)
        # rotate mask 90 degrees counter-clockwise
        aif_mask = np.rot90(aif_mask, k=1, axes=(0,1))
        # add to first index of masks and model_names
        masks.insert(0, aif_mask)
        model_names.insert(0, 'Manual')

    for i, mask in enumerate(masks):

        # find z-slice with most voxels in mask
        z_roi = np.argmax(np.sum(mask, axis=(0,1)))
        if z_roi > img_data.shape[2]:
            z_roi = img_data.shape[2] - 1
        elif z_roi < 0:
            z_roi = 0
        z_rois.append(z_roi)

        # rotate mask 90 degrees counter-clockwise
        if i > 0:
            mask = np.rot90(mask, k=1, axes=(0,1))
        # plot mask from each model
        plt.figure(figsize=(15,5), dpi=125)
        plt.title(file + ' ' + model_names[i])
        # text top left saying % of voxels in slice
        plt.text(5, 15, str(np.round(np.sum(mask[:,:,z_roi]) / (np.round(np.sum(mask))) * 100, 1)) + '% of ' + str(int(np.round(np.sum(mask)))) + ' voxels in slice ' + str(z_roi), fontsize=14, color='white')
        plt.axis('off')
        # plot image
        if file.startswith('5'):
            plt.imshow(img_data[:,:,z_roi,peak_index], cmap='gray', vmax=200)
        else:
            plt.imshow(img_data[:,:,z_roi,peak_index], cmap='gray')
        
        # plot manual mask if it exists
        if os.path.isfile(mask_dir + '/' + file + '.nii'):
            manual_cmap = mcolors.LinearSegmentedColormap.from_list('custom cmap', [(0, 0, 0, 0), 'blue'])
            plt.imshow(aif_mask[:,:,z_roi], cmap=manual_cmap)
        
        if model_names[i] != 'Manual':
            cmap = mcolors.LinearSegmentedColormap.from_list('custom cmap', [(0, 0, 0, 0), 'green'])
            plt.imshow(mask[:,:,z_roi], cmap=cmap, alpha=0.5)
        plt.savefig(os.path.join(output_folder, file + '_' + model_names[i] + '_mask.png'), bbox_inches="tight")
        plt.close()
    print('Saved masked image at:', output_folder + '/' + file + '_mask.png')
    model_names.pop(0)


for image in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image)
    process_image(image_path)

# Path to the folder containing the individual result images
results_folder = output_folder

# Output path for the final mosaic image
curve_mosaic_path = results_folder + "/curve_mosaic.png"

# Get a list of all result image file paths
result_curve_paths = [os.path.join(results_folder, filename) for filename in os.listdir(results_folder) if filename.endswith("curve.png")]
# sort the list by subject
result_curve_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f)))) 

# Determine the number of images per row in the mosaic
images_per_row = 5  # You can adjust this based on your preference

# Open all result curve images and calculate dimensions for the final mosaic
result_images = [Image.open(image_path) for image_path in result_curve_paths]
image_width, image_height = result_images[0].size
mosaic_width = image_width * images_per_row
mosaic_height = image_height * ((len(result_images) - 1) // images_per_row + 1)

# Create a new blank mosaic image
curve_mosaic = Image.new("RGB", (mosaic_width, mosaic_height), (255, 255, 255))

# Paste each result image onto the mosaic
for i, result_image in enumerate(result_images):
    row = i // images_per_row
    col = i % images_per_row
    x_offset = col * image_width
    y_offset = row * image_height
    curve_mosaic.paste(result_image, (x_offset, y_offset))

# Save the final mosaic image
curve_mosaic.save(curve_mosaic_path)

print("Curve mosaic image created:", curve_mosaic_path)

# Now do the same for the masked images, but one subject at a time
# get subject names
subjects = [os.path.basename(path) for path in os.listdir(image_folder)]
subjects = list(set([subject[:-4] for subject in subjects]))

for subject in subjects:
    # Output path for the mask mosaics
    mosaic_path = results_folder + "/" + subject + "_mosaic.png"

    # Get a list of all result image file paths
    result_paths = [os.path.join(results_folder, filename) for filename in os.listdir(results_folder) if (filename.startswith(subject) and filename.endswith("mask.png"))]
    # sort the list by model name
    result_paths.sort(key=lambda f: f.split('_')[-2])

    # Determine the number of images per row in the mosaic
    images_per_row = len(model_names)+1  # You can adjust this based on your preference

    # Open all result images and calculate dimensions for the final mosaic
    result_images = [Image.open(image_path) for image_path in result_paths]
    image_width, image_height = result_images[0].size
    mosaic_width = image_width * images_per_row
    mosaic_height = image_height * ((len(result_images) - 1) // images_per_row + 1)

    # Create a new blank mosaic image
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height), (255, 255, 255))

    # Paste each result image onto the mosaic
    for i, result_image in enumerate(result_images):
        row = i // images_per_row
        col = i % images_per_row
        x_offset = col * image_width
        y_offset = row * image_height
        mosaic.paste(result_image, (x_offset, y_offset))

    # Save the final mosaic image
    mosaic.save(mosaic_path)

    print("Mosaic image created:", mosaic_path)

# I guess we can unify the mask mosaics
# Output path for the final mosaic image
mosaic_path = results_folder + "/mask_mosaic.png"

# Get a list of all result image file paths
result_paths = [os.path.join(results_folder, filename) for filename in os.listdir(results_folder) if not filename.startswith('curve') and filename.endswith("_mosaic.png")]

# Determine the number of images per row in the mosaic
images_per_row = 3  # You can adjust this based on your preference

# Open all result images and calculate dimensions for the final mosaic
result_images = [Image.open(image_path) for image_path in result_paths]
image_width, image_height = result_images[0].size
mosaic_width = image_width * images_per_row
mosaic_height = image_height * ((len(result_images) - 1) // images_per_row + 1)

# Create a new blank mosaic image
mosaic = Image.new("RGB", (mosaic_width, mosaic_height), (255, 255, 255))

# Paste each result image onto the mosaic
for i, result_image in enumerate(result_images):
    row = i // images_per_row
    col = i % images_per_row
    x_offset = col * image_width
    y_offset = row * image_height
    mosaic.paste(result_image, (x_offset, y_offset))

# Save the final mosaic image
mosaic.save(mosaic_path)

print("Mask giga mosaic image created:", mosaic_path)

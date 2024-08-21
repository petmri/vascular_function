#Vascular Input Function (VIF) Extraction deep learning model
import argparse
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import re
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# import tensorrt
from scipy import ndimage
from tensorboard.plugins.hparams import api as hp
from matplotlib import colors as mcolors
# from tensorflow.keras import mixed_precision
import psutil
import time

tf.keras.utils.set_random_seed(100)
# tf.debugging.set_log_device_placement(True)
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
# tf.config.optimizer.set_jit(True)
# mixed_precision.set_global_policy('mixed_float16')

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
AUTOTUNE = tf.data.AUTOTUNE

from model_vif import *
from utils_vif import *

X_DIM = 256
Y_DIM = 256
Z_DIM = 32
T_DIM = 32

def inference_mode(args, file):

    print('Loading data')
    volume_img = nib.load(args.input_path)
    print(volume_img.shape)
    volume_data = volume_img.get_fdata()

    print('Preprocessing')
    vol_pre = preprocessing(volume_data)

    print('Loading model')
    model = unet3d(img_size = (X_DIM, Y_DIM, Z_DIM, T_DIM),\
                     learning_rate = 1e-3,\
                     learning_decay = 1e-9)
    model.trainable = False
    model.load_weights(args.model_weight_path)

    print('Prediction')
    y_pred_mask, y_pred_vf, _ = model.predict(vol_pre)
    # y_pred_mask = y_pred_mask > 0.95
    y_pred_mask = y_pred_mask.astype(float)

    # re-apply thresholded mask to volume
    print('Resizing volume (padding)')
    mask = resize_mask(y_pred_mask, volume_data)

    mask = mask.squeeze()
    mask_img = nib.Nifti1Image(mask, volume_img.affine)

    # remove .nii from file
    if file.endswith(".nii"):
        file = file[:-4]
    elif file.endswith(".nii.gz"):
        file = file[:-7]

    # isolate filename if path is included, grab path while we're at it
    if '/' in file:
        file = file.split('/')[-1]
        path = args.input_path[:-len(file)-5]
    nib.save(mask_img, args.save_output_path+ '/' + file + '_float_mask.nii')
    top20 = np.argsort(mask, axis=None)[-20:]
    if top20.all() > 0.9:
        # save top 20 voxels as aif
        mask_top20 = np.zeros_like(mask)
        top20_indices = np.unravel_index(top20, mask.shape)
        mask_top20[top20_indices] = 1
        mask_top20 = mask_top20.astype(float)
        mask_top20_img = nib.Nifti1Image(mask_top20, volume_img.affine)
        nib.save(mask_top20_img, args.save_output_path+ '/' + file + '_mask.nii')
    else:
        # save top 5 voxels as aif
        mask_top5 = np.zeros_like(mask)
        top5 = np.argsort(mask, axis=None)[-5:]
        top5 = np.unravel_index(top5, mask.shape)
        mask_top5[top5] = 1
        mask_top5 = mask_top5.astype(float)
        mask_top5_img = nib.Nifti1Image(mask_top5, volume_img.affine)
        nib.save(mask_top5_img, args.save_output_path+ '/' + file + '_mask.nii')

    # remove rest of last file from input_path
    if args.input_path.endswith(".nii"):
        args.input_path = args.input_path[:-len(file)-5]
    elif args.input_path.endswith(".nii.gz"):
        args.input_path = args.input_path[:-len(file)-8]

    # remove last directory from input_path
    mask_dir = args.input_path[:-len(args.input_path.split('/')[-1])-1]
    mask_dir = mask_dir + '/masks'

    vf = y_pred_vf
    # plot vascular function
    if args.save_image == 1:
        plt.figure(figsize=(15,5), dpi=250)
        plt.subplot(1,2,1)
        plt.title('Vascular Function (VF): ' + file)
        # set axis titles
        plt.xlabel('t-slice', fontsize=19)
        plt.ylabel('Intensity:Baseline', fontsize=19)
        x = np.arange(len(vf[0]))
        plt.yticks(fontsize=19)
        plt.xticks(fontsize=19)
        plt.plot(x, vf[0] / vf[0][0], 'r', label='Auto', lw=3)
        # plot manual mask if it exists
        if os.path.isfile(mask_dir + '/' + file + '.nii') or os.path.isfile(path + '/aif.nii'):
            if os.path.isfile(mask_dir + '/' + file + '.nii'):
                img = nib.load(mask_dir + '/' + file + '.nii')
                mask = np.array(img.dataobj)
            elif os.path.isfile(path + '/aif.nii'):
                img = nib.load(path + '/aif.nii')
                mask = np.array(img.dataobj)
                mask = mask.squeeze()
            mask_crop = scipy.ndimage.zoom(mask, (X_DIM / mask.shape[0], Y_DIM / mask.shape[1], Z_DIM / mask.shape[2]), order=1)

            dce = nib.load(args.input_path + '/' + file + '.nii')
            dce_data = np.array(dce.dataobj)
            dce_data = (dce_data - np.min(dce_data)) / ((np.max(dce_data) - np.min(dce_data)))

            dce_crop = scipy.ndimage.zoom(dce_data, (X_DIM / dce_data.shape[0], Y_DIM / dce_data.shape[1], Z_DIM / dce_data.shape[2], T_DIM / dce_data.shape[3]), order=1)
            mask_crop = mask_crop.reshape(X_DIM, Y_DIM, Z_DIM, 1)

            roi_ = mask_crop * dce_crop
            num = np.sum(roi_, axis = (0, 1, 2), keepdims=False)
            den = np.sum(mask_crop, axis = (0, 1, 2), keepdims=False)
            intensities = num/(den+1e-8)
            intensities = np.asarray(intensities)
            plt.plot(x, intensities / intensities[0], 'b', label='Manual', lw=3)
        plt.legend(loc="upper right", fontsize=16)
        plt.savefig(os.path.join(args.save_output_path, file+'_curve.svg'), bbox_inches="tight")
        plt.close()
        # print('Saved image at:', args.save_output_path)
        print('Vascular Function (VF) of ' + file + ' saved as image in: ', args.save_output_path)
        # overlay mask on image
        try:
            img = nib.load(args.input_path + '/' + file + '.nii')
        except:
            img = nib.load(args.input_path + '/' + file + '.nii.gz')
        img_data = img.get_fdata()
        img_data = img_data.squeeze()
        plt.figure(figsize=(15,5), dpi=250)
        plt.subplot(1,2,1)
        plt.title('Mask: ' + file)
        # find center of mass of mask
        com = ndimage.center_of_mass(mask)
        # round to nearest integer
        z_roi = np.round(com[2]).astype(int)
        # rotate image
        img_data = np.rot90(img_data, axes=(0,1))
        mask = np.rot90(mask, axes=(0,1))
        # remove axes
        plt.axis('off')
        plt.imshow(img_data[:,:,z_roi,3], cmap='gray')
        # overlay mask, values below 0.5 are transparent
        cmap = mcolors.LinearSegmentedColormap.from_list('custom cmap', [(0, 0, 0, 0), 'blue', 'green', 'red'])
        plt.imshow(mask[:,:,z_roi], cmap=cmap, alpha=0.7)
        plt.savefig(os.path.join(args.save_output_path, file + '_mask.svg'), bbox_inches="tight")
        plt.close()
        print('Saved masked image at:', args.save_output_path)

    return y_pred_vf, mask, volume_img

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        # use this value as reference to calculate cumulative time taken
        self.timetaken = time.perf_counter()

    def on_epoch_begin(self, epoch, logs = {}):
        self.epoch_time_start = time.perf_counter()
        
    def on_epoch_end(self, epoch, logs = {}):
        with open(os.path.join(args.save_checkpoint_path,'log.txt'), 'a') as f:
            f.write("\nTime elapsed: " + str((time.perf_counter() - self.timetaken)))
            f.write("\nEpoch time: " + str((time.perf_counter() - self.epoch_time_start)))
            f.write('\n')
        print("\nTime taken:", (time.perf_counter() - self.timetaken))
        print("Percentage of RAM used:", psutil.virtual_memory().percent)
        
# log file callback
class logcallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        self.lowest_loss = 1000000
        self.log_file = log_file
        with open(self.log_file, 'a') as f:
            f.write("Loss weights: " + str(args.loss_weights))
            
    def on_epoch_end(self, epoch, logs = {}):
        with open(self.log_file, 'a') as f:
            f.write(str(logs))
            f.write('\n')
        if logs.get('val_loss') < self.lowest_loss:
            self.lowest_loss = logs.get('val_loss')
            with open(self.log_file, 'a') as f:
                f.write("New lowest loss: " + str(self.lowest_loss))
                f.write('\n')
    
    def on_train_end(self, logs = {}):
        with open(self.log_file, 'a') as f:
            f.write("Lowest loss: " + str(self.lowest_loss))
            f.write('\n')
        
def training_model(args, hparams=None):

    print("Tensorflow", tf.__version__)
    # print("Keras", keras.__version__)

    DATASET_DIR = args.dataset_path
    # get names of folders in path
    sites = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f)) and not f.startswith('test')]

    # randomly split each site into train, val, and test, 80/10/10 each
    train_set = []
    val_set = []
    test_set = []
    train_set1 = []
    val_set1 = []
    test_set1 = []
    
#     for site in sites:
#         imgs = [f for f in os.listdir(os.path.join(DATASET_DIR, site, 'images')) if f.endswith('.nii') or f.endswith('.nii.gz')]
#         imgs1 = [f for f in os.listdir(os.path.join(DATASET_DIR, site, 'masks')) if f.endswith('.nii') or f.endswith('.nii.gz')]
#         subjects = [f.split('_')[0] for f in os.listdir(os.path.join(DATASET_DIR, site, 'images')) if f.endswith('.nii') or f.endswith('.nii.gz')]
#         unique_subjects = sorted(list(set(subjects)))
#         np.random.shuffle(unique_subjects)

#         # Ensure all subjects' images are in the same split
#         train_set.extend([site + '/' + img for img in imgs if img.split('_')[0] in unique_subjects[:int(0.8*len(unique_subjects))]])
#         val_set.extend([site + '/' + img for img in imgs if img.split('_')[0] in unique_subjects[int(0.8*len(unique_subjects)):int(0.9*len(unique_subjects))]])
#         test_set.extend([site + '/' + img for img in imgs if img.split('_')[0] in unique_subjects[int(0.9*len(unique_subjects)):]])
        
#         train_set1.extend([site + '/' + img for img in imgs1 if img.split('_')[0] in unique_subjects[:int(0.8*len(unique_subjects))]])
#         val_set1.extend([site + '/' + img for img in imgs1 if img.split('_')[0] in unique_subjects[int(0.8*len(unique_subjects)):int(0.9*len(unique_subjects))]])
#         test_set1.extend([site + '/' + img for img in imgs1 if img.split('_')[0] in unique_subjects[int(0.9*len(unique_subjects)):]])
    
    TFRecord_path = os.path.join(DATASET_DIR, '../TFRecords')
    # make folder for saving checkpoints
    if not os.path.exists(args.save_checkpoint_path):
        os.makedirs(args.save_checkpoint_path)
    # save sets in txt file in checkpoint folder
    if not os.path.exists(TFRecord_path) or not os.listdir(TFRecord_path):
        with open(os.path.join(args.save_checkpoint_path,'train_set.txt'), 'w') as f:
            for item in train_set:
                f.write("%s\n" % item)
        with open(os.path.join(args.save_checkpoint_path,'val_set.txt'), 'w') as f:
            for item in val_set:
                f.write("%s\n" % item)
        with open(os.path.join(args.save_checkpoint_path,'test_set.txt'), 'w') as f:
            for item in test_set:
                f.write("%s\n" % item)

    # copy test imgs to test folder
    if not os.path.exists(os.path.join(DATASET_DIR, 'test')):
        os.makedirs(os.path.join(DATASET_DIR, 'test'))
        os.makedirs(os.path.join(DATASET_DIR, 'test', 'images'))
        os.makedirs(os.path.join(DATASET_DIR, 'test', 'masks'))
        for img in test_set:
            img_path = os.path.join(DATASET_DIR, img.replace('/', '/images/'))
            mask_path = os.path.join(DATASET_DIR, img.replace('/', '/masks/'))
            os.system(f'cp {img_path} {os.path.join(DATASET_DIR, "test", "images")}')
            os.system(f'cp {mask_path} {os.path.join(DATASET_DIR, "test", "masks")}')

  

    # log inputs
#     with open(os.path.join(args.save_checkpoint_path,'log.txt'), 'w') as f:
#         f.write("Train: " + str(len1) + '\n')
#         f.write("Val: " + str(len2) + '\n')
#         f.write("Test: " + str(len3) + '\n')
#         f.write("Batch size: " + str(args.batch_size) + '\n')
    

    if args.mode == "hp_tuning":
        model = unet3d( img_size        = (X_DIM, Y_DIM, Z_DIM, T_DIM),
                        learning_rate   = 1e-3,
                        learning_decay  = 1e-9,
                        kernel_size_ao  = eval(hparams[HP_KERNEL_SIZE_FIRST_LAST]),
                        kernel_size_body= eval(hparams[HP_KERNEL_SIZE_BODY]),
                        # weights         = hparams[HP_LOSS_WEIGHTS]
                        )
    else:
        model = unet3d( img_size        = (X_DIM, Y_DIM, Z_DIM, T_DIM),
                        learning_rate   = 1e-3,
                        learning_decay  = 1e-9,
                        weights         = args.loss_weights)
    
#     keras.utils.plot_model(model, "model.png", show_shapes=True)

    if args.mode == "hp_tuning":
        # batch_size = hparams[HP_BATCH_SIZE]
        batch_size = args.batch_size
    else:
        batch_size = args.batch_size
    
    # if TFRecords directory does not exist or is empty, write TFRecords
    if not os.path.exists(TFRecord_path) or not os.listdir(TFRecord_path):
        os.mkdir(TFRecord_path)
        imgs = [os.path.join(DATASET_DIR, img.replace('/', '/images/')) for img in train_set]
        masks = [os.path.join(DATASET_DIR, mask.replace('/', '/masks/')) for mask in train_set1]
        TFRecord_train_path = os.path.join(TFRecord_path, 'train')
        write_records(imgs, masks, 1, TFRecord_train_path)

        imgs = [os.path.join(DATASET_DIR, img.replace('/', '/images/')) for img in val_set]
        masks = [os.path.join(DATASET_DIR, mask.replace('/', '/masks/')) for mask in val_set1]
        TFRecord_val_path = os.path.join(TFRecord_path, 'val')
        write_records(imgs, masks, 1, TFRecord_val_path)
        
    train_records=[os.path.join(TFRecord_path, f) for f in os.listdir(TFRecord_path) if f.startswith('train') and f.endswith('.tfrecords')]
    val_records=[os.path.join(TFRecord_path, f) for f in os.listdir(TFRecord_path) if f.startswith('val') and f.endswith('.tfrecords')]
    
    
    print('Training')
    len1 = len(train_records)
    len2 = len(val_records)
#     len3 = len(test_set)
    
    print("Train:", len1)
    print("Val:", len2)
#     print("Test:", len3)
    
    train_data = get_batched_dataset(train_records, batch_size=batch_size, shuffle_size=50)
    val_data = get_batched_dataset(val_records, batch_size=batch_size, shuffle_size=1)

    model_path = os.path.join(args.save_checkpoint_path,'model_weight_exp-attn-200.h5')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_vf_quality_ultimate', factor=0.5, patience=40, min_lr=1e-15, mode='max')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_vf_quality_ultimate', patience=40, mode='max')
    save_model = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, monitor='val_vf_quality_ultimate', save_best_only=True, mode='max')
#     if args.mode == "hp_tuning":
#         log_dir = "logs/hp_tuning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     else:
#         log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch="70, 80")

    if args.mode == "hp_tuning":
        callbackscallbac  = [save_model, reduce_lr, early_stop, tensorboard_callback, hp.KerasCallback(log_dir, hparams), logcallback(os.path.join(args.save_checkpoint_path,'log.txt'))]
    else:
        callbackscallbac  = [save_model, early_stop, reduce_lr, timecallback()]

    print('Training')
    history = model.fit(
        train_data,
        validation_data=val_data,
        steps_per_epoch=len1//batch_size,
        epochs=args.epochs,
        validation_steps=len2//batch_size,
        callbacks = callbackscallbac,
    )

    try:
        np.save(os.path.join(args.save_checkpoint_path,'history.npy'), history.history)
        plot_history(os.path.join(args.save_checkpoint_path,'history.npy'), os.path.join(args.save_checkpoint_path,'history.png'))
    except:
        print("plot took a fat L")
    
    print("End")

def evaluate_model(args):

    model = unet3d(img_size = (X_DIM, Y_DIM, Z_DIM, T_DIM),\
                     learning_rate = 1e-3,\
                     learning_decay = 1e-9)

    model.load_weights(args.model_weight_path)

    data = load_data(os.path.join(args.input_folder,"images"))
    print('Number of images:', len(data))
    batch_size = 1
    table = []
    mae_array = []
    dist_array = []

    for i in range(len(data)):
        print('Image:', data[i])
        gen_ = train_generator(args.input_folder, [data[i]], batch_size, data_augmentation=False)
        batch_img, batch_label = next(gen_)# with cropping
        
        y_img = nib.load(os.path.join(args.input_folder,"images",data[i]))
        y = np.array(y_img.dataobj)#mask without cropping

        y_pred_mask, y_pred_vf, _ = model.predict(batch_img)
        y_pred_mask = y_pred_mask > 0.5
        y_pred_mask = y_pred_mask.astype(np.float16)

        mae = loss_mae(batch_label[1].astype(np.float16), y_pred_vf.astype(np.float16), False)
        d_distance = loss_computeCofDistance3D(batch_label[0], y_pred_mask.reshape(1, X_DIM, Y_DIM, Z_DIM, 1))
        table.append([data[i], d_distance.numpy(), (mae.numpy()),])

        mae_array.append(mae.numpy())
        dist_array.append(d_distance.numpy())

        if args.save_image == 1:
            plt.figure(figsize=(15,5), dpi=250)
            plt.subplot(1,2,1)
            plt.title('Vascular Function (VF):'+data[i])
            x = np.arange(len(y_pred_vf[0]))
            plt.yticks(fontsize=19)
            plt.xticks(fontsize=19)
            plt.plot(x, y_pred_vf[0] / y_pred_vf[0][0], 'r', label='Auto', lw=3)
            plt.plot(x, (batch_label[1])[0] / (batch_label[1])[0][0], 'b', label='Manual', lw=3)
            plt.legend(loc="upper right", fontsize=16)
            plt.savefig(os.path.join(args.save_output_path, data[i]+'.png'), bbox_inches="tight")
            plt.close()

    mae_array = np.asarray(mae_array)
    dist_array = np.asarray(dist_array)

    df = pd.DataFrame(table, columns =['Id','Distance (delta)','MAE'])
    df.to_csv(os.path.join(args.save_output_path, 'results.csv'))

if __name__== "__main__":

    parser = argparse.ArgumentParser(description="VIF model")
    parser.add_argument("--mode", type=str, default="inference", help="training mode (training) or inference mode (inference) or evaluate mode (eval) or hyperparameter tuning mode (hp_tuning)")
    parser.add_argument("--dataset_path", type=str, default=" ", help="path to dataset")
    parser.add_argument("--save_output_path", type=str, default=" ", help="path to save model results")
    parser.add_argument("--save_checkpoint_path", type=str, default=" ", help="path to save model's checkpoint")
    parser.add_argument("--model_weight_path", type=str, default=" ", help="file of the model's checkpoint")
    parser.add_argument("--input_path", type=str, default=" ", help="input image path")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--loss_weights", type=float, default=[0, 1, 0], nargs=3, help="loss weights for spatial information and temporal information")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--input_folder", type=str, default=" ", help="path of the folder to be evaluated")
    parser.add_argument("--save_image", type=int, default=0, help="save the vascular function as image")

    args = parser.parse_args()

    if args.mode == "inference":
        print('Mode:', args.mode)
        # If input_path is a folder, then it will process all the images in the folder
        # If input_path is a file, then it will process the image
        if os.path.isdir(args.input_path):
            files = os.listdir(args.input_path)
            for file in files:
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    print('Input:', file)
                    args.input_path = os.path.join(args.input_path, file)
                    print(args.input_path)
                    vf, mask, bozo = inference_mode(args, file)
        else:
            print('Input:', args.input_path)
            vf, mask, bozo = inference_mode(args, args.input_path)
    elif args.mode == "training":
        print('Mode:', args.mode)
        training_model(args)
    elif args.mode == "eval":
        print('Mode:', args.mode)
        evaluate_model(args)
    elif args.mode == "hp_tuning":
        print('Mode:', args.mode)
        # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1, 2, 4, 8, 16]))
        # HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
        HP_LR = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.0005, 0.001, 0.005, 0.01]))
        # HP_LOSS_WEIGHTS = hp.HParam('loss_weights', hp.Discrete([[0, 1, 0], [0, 0.7, 0.3], [0.3, 0.7, 0]]))
        # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
        HP_KERNEL_SIZE_FIRST_LAST = hp.HParam('kernel_size_ao', hp.Discrete(['(3, 3, 3)', '(5, 5, 5)', '(7, 7, 7)', '(9, 9, 9)', '(11, 11, 11)', '(3, 7, 7)', '(3, 9, 9)', '(3, 11, 11)', '(5, 7, 7)', '(5, 9, 9)', '(5, 11, 11)', '(7, 9, 9)', '(7, 11, 11)', '(9, 11, 11)']))
        HP_KERNEL_SIZE_BODY = hp.HParam('kernel_size_body', hp.Discrete(['(3, 3, 3)', '(5, 5, 5)', '(7, 7, 7)', '(9, 9, 9)', '(3, 7, 7)', '(3, 9, 9)', '(3, 11, 11)', '(5, 7, 7)', '(5, 9, 9)', '(5, 11, 11)', '(7, 9, 9)', '(7, 11, 11)']))
        # HP_VF_LOSS = hp.HParam('vf_loss', hp.Discrete(['mae', 'mse', 'mape', 'msle', 'huber_loss']))
        # METRIC_MAE = 'mean_absolute_error'

        session_num = 0
        # run hyperparameter tuning with dropout and optimizer
        # for loss_weights in HP_LOSS_WEIGHTS.domain.values:
        for kernel_size_body in (HP_KERNEL_SIZE_BODY.domain.values):
            for kernel_size_ao in (HP_KERNEL_SIZE_FIRST_LAST.domain.values):
        # for dropout_rate in (HP_DROPOUT.domain.values):
        #     for optimizer in (HP_OPTIMIZER.domain.values):
                hparams = {
                    # HP_DROPOUT: dropout_rate,
                    # HP_OPTIMIZER: optimizer,
                    HP_KERNEL_SIZE_FIRST_LAST : kernel_size_ao,
                    HP_KERNEL_SIZE_BODY : kernel_size_body
                    # HP_LOSS_WEIGHTS : loss_weights
                }
                run_name = f"run-{session_num}-{kernel_size_ao}-{kernel_size_body}"
                if os.path.exists(os.path.join(args.save_checkpoint_path, run_name)):
                    print('Skipping:', run_name)
                    session_num += 1
                    continue
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                args.save_checkpoint_path = os.path.join(args.save_checkpoint_path, run_name)
                training_model(args, hparams)
                # remove last directory from save_checkpoint_path
                args.save_checkpoint_path = args.save_checkpoint_path[:-len(run_name)-1]
                session_num += 1

    else:
        print('Error: mode not found!')
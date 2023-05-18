#Vascular Input Extraction (VIF) deep learning model
import argparse
import datetime
import os
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from model_vif import *
from utils_vif import *

X_DIM = 256
Y_DIM = 256
Z_DIM = 32
T_DIM = 32

def inference_mode(args):

    print('Loading data')
    volume_img = nib.load(args.input_path)
    print(volume_img.shape)
    # pad volume to 16 slices
    # if volume_img.shape[2] < Z_DIM:
    #     print('Padding volume to 16 slices')
    #     volume_data = np.zeros((volume_img.shape[0], volume_img.shape[1], 16, volume_img.shape[3]))
    #     volume_data[:, :, :volume_img.shape[2], :] = volume_img.get_fdata()
    #     print(volume_data.shape)

    # if z dim isn't divisible by 2^3, crop to nearest divisible by 2^3
    # elif volume_img.shape[2] % 2**3 != 0:
    #     volume_data = volume_img.get_fdata()[:, :, :volume_img.shape[2] - volume_img.shape[2] % 2**3, :]
    # else:
    #     volume_data = np.array(volume_img.dataobj)
    volume_data = volume_img.get_fdata()

    print('Preprocessing')
    vol_pre = preprocessing(volume_data)

    print('Loading model')
    model = unet3d(img_size = (X_DIM, Y_DIM, Z_DIM, T_DIM),\
                     learning_rate = 1e-3,\
                     learning_decay = 1e-9)

    model.load_weights(args.model_weight_path)

    print('Prediction')
    y_pred_mask, y_pred_vf, _ = model.predict(vol_pre)
    # y_pred_mask = y_pred_mask > 0.8
    # y_pred_mask = y_pred_mask * 2.0
    y_pred_mask = y_pred_mask.astype(np.float16)

    print('Resizing volume (padding)')
    y_pred_mask_rz = resize_mask(y_pred_mask, volume_data)#padding

    return  y_pred_vf, y_pred_mask_rz, volume_img

def training_model(args):

    print("Tensorflow", tf.__version__)
    print("Keras", keras.__version__)

    DATASET_DIR = args.dataset_path
    train_set = load_data(os.path.join(DATASET_DIR,"train/images"))
    val_set = load_data(os.path.join(DATASET_DIR,"val/images"))
    test_set = load_data(os.path.join(DATASET_DIR,"test/images"))

    print('Training')

    print("Train:", len(train_set))
    print("Val:", len(val_set))
    print("Test:", len(test_set))

    model = unet3d(img_size = (X_DIM, Y_DIM, Z_DIM, T_DIM),\
                     learning_rate = 1e-3,\
                     learning_decay = 1e-9, weights=args.loss_weights)
    keras.utils.plot_model(model, "fried.png", show_shapes=True)
    batch_size = args.batch_size
    train_gen = train_generator(os.path.join(DATASET_DIR,"train/"), train_set, batch_size)
    val_gen = train_generator(os.path.join(DATASET_DIR,"val/"), val_set, batch_size)

    model_path = os.path.join(args.save_checkpoint_path,'model_weight.h5')

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-15)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
    save_model = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=0, monitor='val_loss', save_best_only=True)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbackscallbac  = [reduce_lr, early_stop, save_model, tensorboard_callback]

    print('Training')
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_set)/batch_size,
        epochs=args.epochs,
        validation_data = val_gen,
        validation_steps=len(val_set)/batch_size,
        callbacks = callbackscallbac,
        use_multiprocessing=True,
        workers=8
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
    parser.add_argument("--mode", type=str, default="inference", help="training mode (training) or inference mode (inference) or evaluate mode (eval)")
    parser.add_argument("--dataset_path", type=str, default=" ", help="path to dataset")
    parser.add_argument("--save_output_path", type=str, default=" ", help="path to save model's checkpoint")
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
                    vf, mask, bozo = inference_mode(args)

                    mask = mask.squeeze()
                    mask_img = nib.Nifti1Image(mask, bozo.affine)
                    # remove .nii from file
                    file = file[:-4]
                    nib.save(mask_img, args.save_output_path+ '/' + file + '_mask.nii')
                    # np.save(args.save_output_path+'/aif.npy', vf)
                    # np.save(args.save_output_path+'/mask.npy', mask)
                    # scipy.io.savemat(args.save_output_path+'/mask.mat',{'mask_pred':mask})
                    # remove rest of last file from input_path
                    args.input_path = args.input_path[:-len(file)-5]
                    # plot vascular function
                    if args.save_image == 1:
                        plt.figure(figsize=(15,5), dpi=250)
                        plt.subplot(1,2,1)
                        plt.title('Vascular Function (VF):'+file)
                        x = np.arange(len(vf[0]))
                        plt.yticks(fontsize=19)
                        plt.xticks(fontsize=19)
                        plt.plot(x, vf[0] / vf[0][0], 'r', label='Auto', lw=3)
                        plt.legend(loc="upper right", fontsize=16)
                        plt.savefig(os.path.join(args.save_output_path, file+'.png'), bbox_inches="tight")
                        plt.close()
                        print('Saved image at:', args.save_output_path)
        else:
            vf, mask, bozo = inference_mode(args)
            mask = mask.squeeze()

            mask_img = nib.Nifti1Image(mask, bozo.affine)
            nib.save(mask_img, args.save_output_path + '/mask.nii')
            # np.save(args.save_output_path+'/aif.npy', vf)
            # np.save(args.save_output_path+'/mask.npy', mask)
            # scipy.io.savemat(args.save_output_path+'/mask.mat',{'mask_pred':mask})
            # plot vascular function
            if args.save_image == 1:
                plt.figure(figsize=(15,5), dpi=250)
                plt.subplot(1,2,1)
                plt.title('Vascular Function (VF): '+args.input_path)
                x = np.arange(len(vf[0]))
                plt.yticks(fontsize=19)
                plt.xticks(fontsize=19)
                plt.plot(x, vf[0] / vf[0][0], 'r', label='Auto', lw=3)
                plt.legend(loc="upper right", fontsize=16)
                plt.savefig(os.path.join(args.save_output_path+'/mask_vf.png'), bbox_inches="tight")
                plt.close()
                print('Vascular Function (VF) saved as image in: ', args.save_output_path+'/mask_vf.png')
    elif args.mode == "training":
        print('Mode:', args.mode)
        training_model(args)
    elif args.mode == "eval":
        print('Mode:', args.mode)
        evaluate_model(args)
    else:
        print('Error: mode not found!')

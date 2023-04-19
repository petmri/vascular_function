import os
import numpy as np
import nibabel as nib
import random
import matplotlib.pyplot as plt
import scipy

X_DIM = 256
Y_DIM = 256
Z_DIM = 32
T_DIM = 32

def preprocessing(vol):

    #original dimension: (256, 240, 120)
    #cropping dimension: (120, 120, 120)
    #cropping dimension: (120, 120, 16)
    # if vol.shape[3] > 7:
    #     vol = vol[:,:,:,(0, 4, 5, 6, 7, 8, -1)]
    
    # make image arrays of uniform size
    batch_images = np.empty((1, vol.shape[0], vol.shape[1], vol.shape[2], T_DIM))
    vol_crop = np.zeros([vol.shape[0], vol.shape[1], vol.shape[2], T_DIM])
    vol = (vol-np.min(vol))/((np.max(vol)-np.min(vol)))
    # vol_crop = vol[102:(256-34), 60:(240-60),:,:]
    # vol_crop = vol[57:281, 0:256,:,:]
    vol_crop = vol
    # plot the first slice of the first volume
    # plt.imshow(vol_crop[:,:,0,0])
    # plt.show()
    
    batch_images[0] = vol_crop

    return batch_images


def resize_mask(mask):

    # mask_rz = np.zeros([mask.shape[0], 256, 240, 120], dtype=float)
    # mask_rz[:,102:(256-34), 60:(240-60),:] = mask[:,:,:,:,0]
    # mask_rz = np.zeros([mask.shape[0], 320, 320, 16], dtype=float)
    # mask_rz[:,57:281, 0:296,:] = mask[:,:,:,:,0]
    mask_rz = np.zeros([mask.shape[0], X_DIM, Y_DIM, Z_DIM], dtype=float)
    mask_rz = mask[:,:,:,:,0]
    
    return mask_rz

def load_data(path):

    filesList = [f for f in os.listdir(path)]
    return np.asarray(filesList)

def shift_vol(vol, mask):
    new_vol = np.zeros(vol.shape)
    new_mask = np.zeros(mask.shape)

    shift_horizontal = np.random.randint(low=10, high=15, size=1)[0]
    direction = np.random.randint(2, size=1)[0]

    if direction:
        new_vol[:,0:vol.shape[1]-shift_horizontal, :, :] = vol[:,shift_horizontal:vol.shape[1], :, :]
        new_mask[:,0:vol.shape[1]-shift_horizontal, :] = mask[:,shift_horizontal:vol.shape[1], :]
    else:
        new_vol[:,shift_horizontal:vol.shape[1], :, :] = vol[:,0:vol.shape[1]-shift_horizontal, :, :]
        new_mask[:,shift_horizontal:vol.shape[1], :] = mask[:,0:vol.shape[1]-shift_horizontal, :]

    shift_vertical = np.random.randint(low=10, high=15, size=1)[0]
    new_vol[0:new_vol.shape[0]-shift_vertical, :, :, :] = new_vol[shift_vertical:vol.shape[0], :, :, :]
    new_mask[0:new_mask.shape[0]-shift_vertical, :, :] = new_mask[shift_vertical:vol.shape[0],: , :]

    return new_vol, new_mask

def train_generator(DATASET_DIR, data_set, batch_size = 1, temporal_res = T_DIM, data_augmentation = True, shuffle = True):

    batch_images = np.empty((batch_size, X_DIM, Y_DIM, Z_DIM, temporal_res))
    batch_masks = np.empty((batch_size, X_DIM, Y_DIM, Z_DIM, 1))
    batch_curve = np.empty((batch_size, temporal_res))
    batch_cof = np.empty((batch_size, 3))
    batch_vol = np.empty((batch_size, 1))

    while True:
        for i in range(batch_size):
            if shuffle == True:
                name_id = random.randint(0, len(data_set)-1)
            else:
                name_id = 0

            path_img = data_set[name_id]
            img = nib.load(DATASET_DIR + "images/" + path_img)
            vol = np.array(img.dataobj)
            vol_crop = np.zeros([X_DIM, Y_DIM, Z_DIM, temporal_res])
            #normalization
            vol = (vol - np.min(vol)) / ((np.max(vol) - np.min(vol)))
            img2 = nib.load(DATASET_DIR + "masks/" + path_img)
            mask = np.array(img2.dataobj)

            #data augmentation
            if data_augmentation:
                vol, mask = shift_vol(vol, mask)

            #cropping
            # vol_crop= vol[102:(256-34), 60:(240-60),:,:]
            # vol_crop= vol#[57:281, 0:296, :, :]
            # resample volume
            # print(path_img)
            # print(vol.shape)
            vol_crop = scipy.ndimage.zoom(vol, (X_DIM / vol.shape[0], Y_DIM / vol.shape[1], Z_DIM / vol.shape[2], T_DIM / vol.shape[3]), order=1)
            # print(vol_crop.shape)
            # plot vol
            # plt.imshow(vol_crop[:,:,0, 2])
            # plt.show()

            #cropping mask
            mask_crop = np.zeros([X_DIM, Y_DIM, Z_DIM])
            # mask_crop = mask[102:(256-34), 60:(240-60),:]
            mask_crop = scipy.ndimage.zoom(mask, (X_DIM / mask.shape[0], Y_DIM / mask.shape[1], Z_DIM / mask.shape[2]), order=1)

            #True VF
            mask_train_ = mask_crop.reshape(X_DIM, Y_DIM, Z_DIM, 1)
            roi_ = vol_crop * mask_train_
            num = np.sum(roi_, axis = (0, 1, 2), keepdims=False)
            den = np.sum(mask_train_, axis = (0, 1, 2), keepdims=False)
            intensities = num/(den+1e-8)
            intensities = np.asarray(intensities)

            #CoM
            ii, jj, kk = np.meshgrid(np.arange(X_DIM), np.arange(Y_DIM), np.arange(Z_DIM), indexing='ij')
            ii = ii.astype(np.float32)
            jj = jj.astype(np.float32)
            kk = kk.astype(np.float32)

            xx = ii*mask_crop
            yy = jj*mask_crop
            zz = kk*mask_crop

            xx = np.sum(xx).astype(np.float32)
            yy = np.sum(yy).astype(np.float32)
            zz = np.sum(zz).astype(np.float32)

            total = np.sum(mask_crop)
            total = total.astype(np.float32)
            #-----------------------------------------------------------------------

            batch_images[i] = vol_crop
            batch_masks[i] = mask_crop.reshape(X_DIM, Y_DIM, Z_DIM, 1)
            batch_curve[i] = intensities
            batch_cof[i] = np.array([float(xx/(total+1e-10)), float(yy/(total+1e-10)), float(zz/(total+1e-10))])
            batch_vol[i] = np.count_nonzero(mask_train_)

        yield batch_images, [batch_cof, batch_curve, batch_vol]

def plot_history(path, save_path):

    history = np.load(path, allow_pickle=True).item()
    for key in history.keys():
        print (key)

    plt.figure(figsize=(14, 5), dpi=350)
    plt.subplot(1, 3, 1)
    plt.grid('on')
    plt.title('Total loss')
    plt.plot(history['loss'], 'b', lw=2, alpha=0.7, label='Training')
    plt.plot(history['val_loss'], 'r', lw=2, alpha=0.7, label='Val')
    plt.legend(loc="upper right")

    plt.subplot(1, 3, 2)
    plt.title('MAE')
    plt.grid('on')
    plt.plot(history['lambda_vf_loss'], 'b', lw=2, alpha=0.7, label='Training')
    plt.plot(history['val_lambda_vf_loss'], 'r', lw=2, alpha=0.7, label='Val')
    plt.legend(loc="upper right")

    plt.subplot(1, 3, 3)
    plt.title('CoM')
    plt.grid('on')
    plt.title('Distance')
    plt.plot(history['lambda_normalization_loss'], 'b', lw=2, alpha=0.7, label='Training')
    plt.plot(history['val_lambda_normalization_loss'], 'r', lw=2, alpha=0.7, label='Val')
    plt.legend(loc="upper right")

    plt.subplot(1, 4, 4)
    plt.title('# of Voxels')
    plt.grid('on')
    plt.title('Volume')
    plt.plot(history['lambda_vol_loss'], 'b', lw=2, alpha=0.7, label='Training')
    plt.plot(history['val_lambda_vol_loss'], 'r', lw=2, alpha=0.7, label='Val')
    plt.legend(loc="upper right")

    plt.savefig(save_path, bbox_inches="tight")

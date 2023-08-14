import os
import numpy as np
import nibabel as nib
import random
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
tf.keras.utils.set_random_seed(100)

X_DIM = 256
Y_DIM = 256
Z_DIM = 32
T_DIM = 32

def preprocessing(vol):
    
    # make image arrays of uniform size
    batch_images = np.empty((1, X_DIM, Y_DIM, Z_DIM, T_DIM))
    vol_crop = np.zeros([X_DIM, Y_DIM, Z_DIM, T_DIM])
    # normalize to [0,1]
    vol = (vol-np.min(vol)) / ((np.max(vol)-np.min(vol)))

    # resample to 256x256x32 and 32 time points (interpolation order 1 = linear)
    vol_crop = scipy.ndimage.zoom(vol, (X_DIM / vol.shape[0], Y_DIM / vol.shape[1], Z_DIM / vol.shape[2], T_DIM / vol.shape[3]), order=1)

    # plot the first slice of the first volume
    # plt.imshow(vol_crop[:,:,0,0])
    # plt.show()
    
    batch_images[0] = vol_crop
    
    del vol_crop, vol

    return batch_images


def resize_mask(mask, vol):

    # mask_rz = np.zeros([mask.shape[0], X_DIM, Y_DIM, Z_DIM], dtype=float)
    # mask_rz = mask[:,:,:,:,0]

    # restore mask to original size
    mask_rz = scipy.ndimage.zoom(mask, (1, vol.shape[0] / X_DIM, vol.shape[1] / Y_DIM, vol.shape[2] / Z_DIM, 1), order=1)
    
    # mask_rz = np.round(mask_rz)
    # mask_rz = mask_rz.astype(int)
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
    

class train_generator(tf.keras.utils.Sequence):
    
    def __init__(self, paths, directory,
                 batch_size,
                 input_size=(X_DIM, Y_DIM, Z_DIM, T_DIM),
                 shuffle=True, data_augmentation=True):
        
        self.directory = directory
        self.paths = paths
        self.batch_size = 1
        self.input_size = input_size
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        self.indexes = np.arange(len(self.paths))
        
    def __len__(self):
        return int(np.floor(len(self.paths) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.paths[k] for k in indexes]

        # Generate data
        batch_images, (batch_cof, batch_curve, batch_vol) = self.__data_generation(list_IDs_temp)

        return batch_images, (batch_cof, batch_curve, batch_vol)
    
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        
        batch_images = np.empty((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2], self.input_size[3]))
        batch_masks = np.empty((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2], 1))
        batch_curve = np.empty((self.batch_size, self.input_size[3]))
        batch_cof = np.empty((self.batch_size, 3))
        batch_vol = np.empty((self.batch_size, 1))
        
        for i, ID in enumerate(list_IDs_temp):
            
            img = nib.load(self.directory + "images/" + ID)
            vol = np.array(img.dataobj)
            vol_crop = np.zeros([X_DIM, Y_DIM, Z_DIM, T_DIM])
            #normalization
            vol = (vol - np.min(vol)) / ((np.max(vol) - np.min(vol)))
            img2 = nib.load(self.directory + "masks/" + ID)
            mask = np.array(img2.dataobj)

            #data augmentation
            if self.data_augmentation:
                vol, mask = shift_vol(vol, mask)

            # resample volume
            vol_crop = scipy.ndimage.zoom(vol, (X_DIM / vol.shape[0], Y_DIM / vol.shape[1], Z_DIM / vol.shape[2], T_DIM / vol.shape[3]), order=1)

            # resample mask
            mask_crop = np.zeros([X_DIM, Y_DIM, Z_DIM])
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

            mask_crop = mask_crop.reshape(X_DIM, Y_DIM, Z_DIM, 1)

#             if self.data_augmentation:
#                 flip_decision = np.random.random()
#                 if flip_decision < 0.5:
#                     vol_crop = np.flip(vol_crop, axis=0)
#                     mask_crop = np.flip(mask_crop, axis=0)

#                 flip_decision = np.random.random()
#                 if flip_decision < 0.5:
#                     vol_crop = np.flip(vol_crop, axis=1)
#                     mask_crop = np.flip(mask_crop, axis=1)
                    
            batch_images[i] = vol_crop
            batch_masks[i] = mask_crop.reshape(X_DIM, Y_DIM, Z_DIM, 1)
            batch_curve[i] = intensities
            batch_cof[i] = np.array([float(xx/(total+1e-10)), float(yy/(total+1e-10)), float(zz/(total+1e-10))])
            batch_vol[i] = np.count_nonzero(mask_train_)

            del xx, yy, zz, total, mask_crop, intensities, roi_, num, den, mask_train_, vol_crop, vol, mask, img, img2
        
        return batch_images, (batch_cof, batch_curve, batch_vol)
    

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
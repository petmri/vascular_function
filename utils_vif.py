import os
import numpy as np
import random
import tensorflow as tf

tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)

tf.config.experimental.enable_op_determinism()

def seed_worker(worker_id):
    worker_seed = int(tf.random.uniform(shape=[], maxval=2**32, dtype=tf.int64))
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
# CODE ABOVE IS FOR REPRODUCIBILITY

import nibabel as nib
import random
import math
import matplotlib.pyplot as plt
import scipy
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
        
# make TFRecord
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _ndarray_feature(array):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))

def serialize_example(img: str, mask: str) -> tf.train.Example:
    # convert to numpy array
    # nii0 = np.asanyarray(nib.load(img).dataobj)
    # 
    # rescale to 0-1
    # if you want to use tf.image.per_image_standardization(), this would be the place to do so
    # instead of the rescaling
    # nii = (nii0 - nii0.min())/(nii0.max() - nii0.min()).astype(np.float32)
    # mask = np.asanyarray(nib.load(mask).dataobj)

    img = nib.load(img)
    vol = np.array(img.dataobj)
    vol_crop = np.zeros([X_DIM, Y_DIM, Z_DIM, T_DIM])
    #normalization
    vol = (vol - np.min(vol)) / ((np.max(vol) - np.min(vol))).astype(np.float32)
    img2 = nib.load(mask)
    mask = np.array(img2.dataobj)

    # there shouldn't be a t-dimension in the mask
    if len(mask.shape) == 4:
        mask = mask[:,:,:,0]

    # resample volume
    vol_crop = scipy.ndimage.zoom(vol, (X_DIM / vol.shape[0], Y_DIM / vol.shape[1], Z_DIM / vol.shape[2], T_DIM / vol.shape[3]), order=1)

    # resample mask
    mask_crop = np.zeros([X_DIM, Y_DIM, Z_DIM])
    mask_crop = scipy.ndimage.zoom(mask, (X_DIM / mask.shape[0], Y_DIM / mask.shape[1], Z_DIM / mask.shape[2]), order=1)

    #True VF
    mask_train_ = np.expand_dims(mask_crop, axis=3)
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

    feature = {
        # 'label': _float_feature(label),
        'image_raw': _bytes_feature(tf.io.serialize_tensor(vol_crop).numpy()),
        # 'mask': _bytes_feature(tf.io.serialize_tensor(mask).numpy())
        # 'curve': _ndarray_feature(intensities),
        'curve': _bytes_feature(tf.io.serialize_tensor(intensities).numpy()),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto

def write_records(niis, masks, n_per_record: int, outfile: str) -> None:
    """
    store list of niftis (and associated mask) into tfrecords for use as dataset
    """
    n_niis = len(niis)
    print(f"writing {n_niis} niis to {outfile}")
    n_records = math.ceil(len(niis) / n_per_record)
    print(f"writing {n_records} records")

    for i, shard in enumerate(range(0, n_niis, n_per_record)):
        print(f"writing record {i} of {n_records-1}")
        with tf.io.TFRecordWriter(
                f"{outfile}_{i:0>3}-of-{n_records-1:0>3}.tfrecords", 
            options= tf.io.TFRecordOptions(compression_type="GZIP")
        ) as writer:
            for nii, mask in zip(niis[shard:shard+n_per_record], masks[shard:shard+n_per_record]):
                example = serialize_example(img=nii, mask=mask)
                writer.write(example.SerializeToString())


def parse_1_example(example) -> tf.Tensor:
    X = tf.io.parse_tensor(example['image_raw'], out_type=tf.float32)
    Y = tf.io.parse_tensor(example['curve'], out_type=tf.float32)
    return X, Y


def decode_example(record_bytes)-> dict:
    example = tf.io.parse_example(
        record_bytes,
        features = {
          'image_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        #   "mask": tf.io.FixedLenFeature([], dtype=tf.string),
          'curve': tf.io.FixedLenFeature([], dtype=tf.string),
          }
    )
    return example

def get_batched_dataset(files, batch_size: int = 32, shuffle_size: int=1024) -> tf.data.Dataset:
    dataset = (
        tf.data.Dataset.list_files(files) # note shuffling is on by default
        .flat_map(lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE))
        .map(decode_example, num_parallel_calls=tf.data.AUTOTUNE)
        .map(parse_1_example, num_parallel_calls=tf.data.AUTOTUNE)
        # .cache()  # remove if all examples don't fit in memory (note interaction with shuffling of files, above)
        .shuffle(shuffle_size)
        .repeat()
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    # tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=8)
    return dataset

def get_baseline_from_curve(curve):
    peak_index = np.argmax(curve)
    return np.mean(curve[:peak_index-1][np.where(curve[:peak_index-1] < curve[0] * 1.75)])
    
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
    plt.plot(history['vf_loss'], 'b', lw=2, alpha=0.7, label='Training')
    plt.plot(history['val_vf_loss'], 'r', lw=2, alpha=0.7, label='Val')
    plt.legend(loc="upper right")

    # plt.subplot(1, 3, 3)
    # plt.title('CoM')
    # plt.grid('on')
    # plt.title('Distance')
    # plt.plot(history['cast_loss'], 'b', lw=2, alpha=0.7, label='Training')
    # plt.plot(history['val_cast_loss'], 'r', lw=2, alpha=0.7, label='Val')
    # plt.legend(loc="upper right")

    # plt.subplot(1, 4, 4)
    # plt.title('# of Voxels')
    # plt.grid('on')
    # plt.title('Volume')
    # plt.plot(history['vol_loss'], 'b', lw=2, alpha=0.7, label='Training')
    # plt.plot(history['val_vol_loss'], 'r', lw=2, alpha=0.7, label='Val')
    # plt.legend(loc="upper right")

    plt.savefig(save_path, bbox_inches="tight")
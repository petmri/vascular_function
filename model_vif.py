import numpy as np
import tensorflow as tf
import random
import os

tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)
os.environ['PYTHONHASHSEED'] = str(0)

tf.config.experimental.enable_op_determinism()

def seed_worker(worker_id):
    worker_seed = int(tf.random.uniform(shape=[], maxval=2**32, dtype=tf.int64))
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# CODE ABOVE FOR REPRODUCIBILITY

# tf.compat.v1.enable_eager_execution()
import tensorflow_addons as tfa
import aif_metric as aif
from tensorflow import keras
from tensorflow.keras.layers import (Conv3D, Dropout, Lambda, MaxPool3D, GlobalAveragePooling3D, Reshape, Dense, Activation, Add, Lambda,
                                     UpSampling3D, concatenate, Multiply, Permute, BatchNormalization)
from tensorflow.keras import regularizers
import sys


X_DIM = 256
Y_DIM = 256
Z_DIM = 32
T_DIM = 32

def quality_peak_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    quality_np = aif.quality_peak_new(y_pred_np)
    quality_tf = tf.convert_to_tensor(quality_np, dtype=tf.float32)
    return quality_tf

def quality_tail_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    quality_np = aif.quality_tail_new(y_pred_np)
    quality_tf = tf.convert_to_tensor(quality_np, dtype=tf.float32)
    return quality_tf

def quality_base_to_mean_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    quality_np = aif.quality_base_to_mean_new(y_pred_np)
    quality_tf = tf.convert_to_tensor(quality_np, dtype=tf.float32)
    return quality_tf

def quality_peak_time_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    quality_np = aif.quality_peak_time_new(y_pred_np)
    quality_tf = tf.convert_to_tensor(quality_np, dtype=tf.float32)
    return quality_tf

@keras.saving.register_keras_serializable(package='Custom', name='quality_ultimate_new')
def quality_ultimate_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    aifitness_np = aif.quality_ultimate_new(y_pred_np)
    aifitness_tf = tf.convert_to_tensor(aifitness_np, dtype=tf.float32)
    return aifitness_tf

def quality_peak_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    quality_np = aif.quality_peak_new(y_pred_np)
    quality_tf = tf.convert_to_tensor(quality_np, dtype=tf.float32)
    return quality_tf

def quality_tail_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    quality_np = aif.quality_tail_new(y_pred_np)
    quality_tf = tf.convert_to_tensor(quality_np, dtype=tf.float32)
    return quality_tf

def quality_base_to_mean_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    quality_np = aif.quality_base_to_mean_new(y_pred_np)
    quality_tf = tf.convert_to_tensor(quality_np, dtype=tf.float32)
    return quality_tf

def quality_peak_time_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    quality_np = aif.quality_peak_time_new(y_pred_np)
    quality_tf = tf.convert_to_tensor(quality_np, dtype=tf.float32)
    return quality_tf

@keras.saving.register_keras_serializable(package='Custom', name='quality_ultimate_new')
def quality_ultimate_new(y_true, y_pred):
    y_pred_np = tf.make_ndarray(tf.make_tensor_proto(y_pred))
    aifitness_np = aif.quality_ultimate_new(y_pred_np)
    aifitness_tf = tf.convert_to_tensor(aifitness_np, dtype=tf.float32)
    return aifitness_tf

@keras.saving.register_keras_serializable(package='Custom', name='loss_mae')
def loss_mae(y_true, y_pred):
    flatten = tf.keras.layers.Flatten()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Normalize data to emphasize intensity curve shape over magnitudes
    y_true_f = flatten(y_true / (y_true[:, 0]))
    y_pred_f = flatten(y_pred / (y_pred[:, 0]))
    # batch_size > 1 compatible
    # y_true_normalized = y_true / y_true[:, :1]
    # y_pred_normalized = y_pred / y_pred[:, :1]
    
    mae = tf.keras.losses.MeanAbsoluteError()
    huber = tf.keras.losses.Huber(delta=1.0, reduction='sum_over_batch_size', name='huber_loss')
    # Weights should be size [batch_size, T_DIM]
    # weight first 10 points 3:1 to last 22 points
    weights = np.concatenate((np.ones(10)*3, np.ones(22)))
    loss = huber(y_true_f, y_pred_f, weights)
    # loss = mae(y_true_normalized, y_pred_normalized)
    
    return 200*loss

def castTensor(tensor):

    tensor = tf.cast(tensor, tf.float32)

    return tensor

def ROIs(tensor):

    tensor1 = tensor[0]
    tensor2 = tensor[1]
    ans = tf.math.multiply(tensor1, tensor2)
    ans = tf.cast(ans, tf.float32)

    return ans


def computeCurve(tensor):

    mask = tensor[0]
    roi = tensor[1]

    num = tf.keras.backend.sum(roi, axis=(1, 2, 3), keepdims=False)
    den = tf.keras.backend.sum(mask, axis=(1, 2, 3), keepdims=False)
    curve = tf.math.divide(num,den + 1e-8)
    curve = tf.cast(curve, tf.float32)

    return curve

def computeQuality(tensor):
    mask = tensor[0]
    roi = tensor[1]
    # print(mask)
    # print(roi)

    num = tf.keras.backend.sum(roi, axis=(1, 2, 3), keepdims=False)
    den = tf.keras.backend.sum(mask, axis=(1, 2, 3), keepdims=False)
    curve = tf.math.divide(num,den + 1e-8)
    curve = tf.cast(curve, tf.float32)
    # print(curve)
    max = tf.reduce_max(curve)
    # print(max)
    max_base_ratio = max
    max_end_ratio = tf.divide(max, curve[-1])
    # print(max_base_ratio)
    # print(max_end_ratio)
    quality = 1/max_base_ratio + 1/max_end_ratio
    # print(quality)

    return curve
    

def normalizeOutput(tensor):

    tensor_norm = (tensor - tf.reduce_min(tensor)) / (tf.reduce_max(tensor) - tf.reduce_min(tensor) + 1e-10)
    return tensor_norm

def getVolume(tensor):
    mask = tensor
    threshold = tf.greater(mask, 0.9)
    mask_thresholded = mask * tf.cast(threshold, dtype=tf.float32)
    
    vol = tf.math.count_nonzero(mask_thresholded)
    return vol

def loss_computeCofDistance3D(y_true, y_pred):

    cof = y_true
    mask = y_pred
    cof = tf.cast(cof, tf.float32)
    mask = tf.cast(mask, tf.float32)

    ii, jj, zz, _ = tf.meshgrid(tf.range(X_DIM), tf.range(Y_DIM), tf.range(Z_DIM), tf.range(1), indexing='ij')
    ii = tf.cast(ii, tf.float32)
    jj = tf.cast(jj, tf.float32)
    zz = tf.cast(zz, tf.float32)

    dx = ((ii - cof[:, 0]) * .5469)**2
    dy = ((jj - cof[:, 1]) * .5469)**2
    dz = ((zz - cof[:, 2]) * 5.0)**2

    dtotal = (dx+dy+dz)
    dtotal = tf.math.sqrt(dtotal)
    dtotal = tf.math.multiply(dtotal,mask)
    dtotal = tf.reduce_sum(dtotal, axis=(1,2,3,4))

    return dtotal / (tf.reduce_sum(mask) + 1e-10)   # this division is made to avoid a trivial solution (mask all zeros)

def loss_volume(y_true, y_pred):
    true_mask = tf.cast(y_true, tf.float32)
    pred_mask = tf.cast(y_pred, tf.float32)
    loss = abs(pred_mask-true_mask)

    return loss

def loss_quality(y_true, y_pred):
    flatten = tf.keras.layers.Flatten()
    
    # normalize data to emphasize intensity curve shape over magnitudes
    # y_true_f = flatten(y_true / (y_true[:, 0]))
    y_pred_f = flatten(y_pred / (y_pred[:, 0]))
    
    # max_base_ratio_true = max(y_true_f)
    max_base_ratio_pred = max(y_pred_f)
    
    # max_end_ratio_true = max(y_true_f) / y_true_f[-1]
    max_end_ratio_pred = max(y_pred_f) / y_pred_f[-1]
    
    loss = 1/max_base_ratio_pred + 1/max_end_ratio_pred

    return loss


def self_attention_block(x, filters):
    theta = Conv3D(filters // 8, (1, 1, 1), padding='same')(x)
    theta = Reshape((-1, filters // 8))(theta)

    phi = Conv3D(filters // 8, (1, 1, 1), padding='same')(x)
    phi = Reshape((-1, filters // 8))(phi)
    phi = Permute((2, 1))(phi)

    attention = tf.matmul(theta, phi)
    attention = Activation('softmax')(attention)

    g = Conv3D(filters, (1, 1, 1), padding='same')(x)
    g = Reshape((-1, filters))(g)

    out = tf.matmul(attention, g)
    out = Reshape((x.shape[1], x.shape[2], x.shape[3], filters))(out)
    out = Conv3D(filters, (1, 1, 1), padding='same')(out)

    out = Add()([x, out])
    return out

    
def unet3d(img_size = (None, None, None), kernel_size_ao=(3, 11, 11), kernel_size_body=(3, 7, 7), drop_out = 0.35, nchannels = T_DIM):
    
    dropout = drop_out
    input_img = tf.keras.layers.Input((img_size[0], img_size[1], img_size[2], nchannels))
    
    # encoder
    conv1_1 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(input_img)
    conv1_2 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv1_1)
    conv1_2 = tfa.layers.InstanceNormalization()(conv1_2)
    pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1_2)

    conv2_1 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool1)
    conv2_2 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv2_1)
    pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2_2)

    conv3_1 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool2)
    conv3_2 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv3_1)
    conv3_2 = tfa.layers.InstanceNormalization()(conv3_2)
    pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3_2)

    # botleneck
    conv4_1 = Conv3D(320, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool3)
    conv4_2 = Conv3D(320, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv4_1)
#     conv4_2 = self_attention_block(conv4_2, 256)

    # decoder
    up1_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4_2), conv3_2],axis=-1)
    conv5_1 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up1_1)
    conv5_2 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv5_1)
    conv5_2 = tfa.layers.InstanceNormalization()(conv5_2)

    up2_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5_2), conv2_2],axis=-1)
    conv6_1 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up2_1)
    conv6_2 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv6_1)

    up3_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_2), conv1_2],axis=-1)
    conv7_1 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up3_1)
    conv7_2 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv7_1)
    conv7_2 = tfa.layers.InstanceNormalization()(conv7_2)

    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7_2)
    # normalization
    conv8 = Lambda(normalizeOutput, name='normalization')(conv8)

    # make binary mask (actually is float, values 0-1)
    binConv = Lambda(castTensor, name="cast")(conv8)
    # defining ROIs
    # pred AIF = original img SI * "binary mask"
    roiConv = Lambda(ROIs, name="roi")([input_img, binConv])
    # compute curve
    # sum of pred AIF / binary mask voxels
    curve = Lambda(computeCurve, name="vf")([binConv, roiConv])
    # count volume
    mask_vol = Lambda(getVolume, name="vol")(binConv)
    # quality
    quality = Lambda(computeQuality, name="lambda_quality")([binConv, roiConv])

    model = tf.keras.models.Model(inputs=input_img, outputs=(binConv, curve, mask_vol))

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss={"vf" : [loss_mae]}, metrics = {"vf" : [quality_ultimate_new]}, run_eagerly=True)

    return model
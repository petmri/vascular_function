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
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# CODE ABOVE IS FOR REPRODUCIBILITY

import aif_metric as aif
from tensorflow import keras
from tensorflow.keras.layers import (Conv3D, Lambda, MaxPool3D, GlobalAveragePooling3D, Reshape, Dense, Activation, Add, Lambda,
                                     UpSampling3D, concatenate, Multiply, Permute, BatchNormalization)
from tensorflow.keras import regularizers
import sys


X_DIM = 256
Y_DIM = 256
Z_DIM = 32
T_DIM = 32

def quality_peak(y_true, y_pred):
    peak_ratio = tf.reduce_max(y_pred) / tf.reduce_mean(y_pred)
    peak_ratio = tf.cast(peak_ratio, tf.float32)

    return (1 / (1 + tf.exp(-3.5 * peak_ratio + 7.5))) * (100 / 0.4499714351078607)

def quality_tail(y_true, y_pred):
    # end is mean of last 20% of curve
    end_ratio = tf.reduce_mean(y_pred[-int(float(int(len(y_pred)))*0.2):])
    end_ratio = tf.cast(end_ratio, tf.float32)
    quality = (1 - (end_ratio / (1.1 * tf.reduce_mean(y_pred))) ** 2)
    return quality * (100 / 0.33436023529043213)

def quality_base_to_mean(y_true, y_pred):
    baseline = tf.argmax(y_pred, axis=-1)
    mask = tf.less(y_pred[:baseline[0]-1], y_pred[0] * 1.75)
    mask.set_shape([None])  # Specify the mask dimensions
    masked_curve = tf.boolean_mask(y_pred[:baseline[0]-1], mask)
    baseline = tf.reduce_mean(masked_curve)
    quality = (1 - (baseline / tf.reduce_mean(y_pred)) ** 2) * (100 / 0.8831850876454762)
    return quality

def quality_peak_time(y_true, y_pred):
    peak_time = tf.argmax(y_pred)
    peak_time = tf.cast(peak_time, tf.float32)
    num_timeslices = tf.cast(len(y_pred), tf.float32)
    qpt = (num_timeslices - peak_time) / num_timeslices
    return qpt*(100/0.9081383928571428)

# @keras.saving.register_keras_serializable(package='Custom', name='quality_ultimate')
def quality_ultimate(y_true, y_pred):
    peak_ratio = quality_peak(y_true, y_pred)
    end_ratio = quality_tail(y_true, y_pred)
    base_to_mean = quality_base_to_mean(y_true, y_pred)
    peak_time = quality_peak_time(y_true, y_pred)

    # take weighted average
    return peak_ratio * 0.3 + end_ratio * 0.3 + base_to_mean * 0.3 + peak_time * 0.1

# @keras.saving.register_keras_serializable(package='Custom', name='loss_huber')
def loss_huber(y_true, y_pred):
    flatten = tf.keras.layers.Flatten()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Normalize data to emphasize intensity curve shape over magnitudes
    y_true_f = flatten(y_true / (y_true[:, 0]))
    y_pred_f = flatten(y_pred / (y_pred[:, 0]))
    
    huber = tf.keras.losses.Huber(delta=1.0, reduction='sum_over_batch_size', name='huber_loss')
    # Weights should be size [batch_size, T_DIM]
    # weight first 10 points 3:1 to last 22 points
    weights = np.concatenate((np.ones(10)*3, np.ones(22)))
    loss = huber(y_true_f, y_pred_f, weights)
    
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


def attention_block(x, filters):
    # Compute Query (Q), Key (K), and Value (V) using learned linear projections
    Q = Conv3D(filters, (1, 1, 1), padding='same')(x)  # Query
    Q = Reshape((-1, filters))(Q)

    K = Conv3D(filters, (1, 1, 1), padding='same')(x)  # Key
    K = Reshape((-1, filters))(K)
    K = Permute((2, 1))(K)

    V = Conv3D(filters, (1, 1, 1), padding='same')(x)  # Value
    V = Reshape((-1, filters))(V)

    attention_scores = tf.matmul(Q, K)  # Shape: (depth * height * width, depth * height * width)
    scaling_factor = tf.math.sqrt(tf.cast(filters, tf.float32))
    attention_scores = attention_scores / scaling_factor
    attention_weights = Activation('softmax')(attention_scores)

    out = tf.matmul(attention_weights, V)  # Shape: (depth * height * width, filters)

    # Reshape back to original spatial dimensions with updated filters
    out = Reshape((x.shape[1], x.shape[2], x.shape[3], filters))(out)

    return out

def modified_attention_block(x, filters):

    Q = Conv3D(filters // 8, (1, 1, 1), padding='same')(x)     
    Q = Reshape((-1, filters // 8))(Q)

    K = Conv3D(filters // 8, (1, 1, 1), padding='same')(x)
    K = Reshape((-1, filters // 8))(K)
    K = Permute((2, 1))(K)

    attention = tf.matmul(Q, K)
    attention = Activation('softmax')(attention)

    V = Conv3D(filters, (1, 1, 1), padding='same')(x)
    V = Reshape((-1, filters))(V)

    out = tf.matmul(attention, V)
    out = Reshape((x.shape[1], x.shape[2], x.shape[3], filters))(out)
    out = Conv3D(filters, (1, 1, 1), padding='same')(out)

    out = Add()([x, out])      # Added learned weights from previous layers
    return out


def unet3d_attention(img_size = (None, None, None), kernel_size_ao=(3, 11, 11), kernel_size_body=(3, 7, 7), nchannels = T_DIM):
    
    input_img = tf.keras.layers.Input((img_size[0], img_size[1], img_size[2], nchannels))
    
    # encoder
    conv1_1 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(input_img)
    conv1_2 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv1_1)
    conv1_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv1_2)
    pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1_2)

    conv2_1 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool1)
    conv2_2 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv2_1)
    pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2_2)

    conv3_1 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool2)
    conv3_2 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv3_1)
    conv3_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv3_2)
    pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3_2)

    # botleneck
    conv4_1 = Conv3D(256, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool3)
    conv4_2 = Conv3D(256, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv4_1)
    conv4_2 = attention_block(conv4_2, 256)

    # decoder
    up1_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4_2), conv3_2],axis=-1)
    conv5_1 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up1_1)
    conv5_2 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv5_1)
    conv5_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv5_2)

    up2_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5_2), conv2_2],axis=-1)
    conv6_1 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up2_1)
    conv6_2 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv6_1)

    up3_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_2), conv1_2],axis=-1)
    conv7_1 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up3_1)
    conv7_2 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv7_1)
    conv7_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv7_2)

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

    model.compile(optimizer=opt, loss={"vf" : [loss_huber]}, metrics = {"vf" : [quality_ultimate]})

    return model


def unet3d_modified_attention(img_size = (None, None, None), kernel_size_ao=(3, 11, 11), kernel_size_body=(3, 7, 7), nchannels = T_DIM):
    
    input_img = tf.keras.layers.Input((img_size[0], img_size[1], img_size[2], nchannels))
    
    # encoder
    conv1_1 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(input_img)
    conv1_2 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv1_1)
    conv1_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv1_2)
    pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1_2)

    conv2_1 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool1)
    conv2_2 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv2_1)
    pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2_2)

    conv3_1 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool2)
    conv3_2 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv3_1)
    conv3_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv3_2)
    pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3_2)

    # botleneck
    conv4_1 = Conv3D(256, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool3)
    conv4_2 = Conv3D(256, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv4_1)
    conv4_2 = modified_attention_block(conv4_2, 256)

    # decoder
    up1_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4_2), conv3_2],axis=-1)
    conv5_1 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up1_1)
    conv5_2 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv5_1)
    conv5_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv5_2)

    up2_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5_2), conv2_2],axis=-1)
    conv6_1 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up2_1)
    conv6_2 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv6_1)

    up3_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_2), conv1_2],axis=-1)
    conv7_1 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up3_1)
    conv7_2 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv7_1)
    conv7_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv7_2)

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

    model.compile(optimizer=opt, loss={"vf" : [loss_huber]}, metrics = {"vf" : [quality_ultimate]})

    return model

    
def unet3d_best(img_size = (None, None, None), kernel_size_ao=(3, 11, 11), kernel_size_body=(3, 7, 7), nchannels = T_DIM):
    
    input_img = tf.keras.layers.Input((img_size[0], img_size[1], img_size[2], nchannels))
    
    # encoder
    conv1_1 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(input_img)
    conv1_2 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv1_1)
    conv1_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv1_2)
    pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1_2)

    conv2_1 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool1)
    conv2_2 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv2_1)
    pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2_2)

    conv3_1 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool2)
    conv3_2 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv3_1)
    conv3_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv3_2)
    pool3 = MaxPool3D(pool_size=(2, 2, 2))(conv3_2)

    # botleneck
    conv4_1 = Conv3D(320, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool3)
    conv4_2 = Conv3D(320, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv4_1)

    # decoder
    up1_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4_2), conv3_2],axis=-1)
    conv5_1 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up1_1)
    conv5_2 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv5_1)
    conv5_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv5_2)

    up2_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5_2), conv2_2],axis=-1)
    conv6_1 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up2_1)
    conv6_2 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv6_1)

    up3_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_2), conv1_2],axis=-1)
    conv7_1 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up3_1)
    conv7_2 = Conv3D(32, kernel_size_ao, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv7_1)
    conv7_2 = tf.keras.layers.GroupNormalization(groups=-1)(conv7_2)

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

    model.compile(optimizer=opt, loss={"vf" : [loss_huber]}, metrics = {"vf" : [quality_ultimate]})

    return model
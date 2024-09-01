import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import (Conv3D, Dropout, Lambda, MaxPool3D, GlobalAveragePooling3D, Reshape, Dense, Activation, Add, Lambda,
                                     UpSampling3D, concatenate, Multiply, Permute, BatchNormalization, Concatenate)
from tensorflow.keras import regularizers
# tf.keras.utils.set_random_seed(100)



X_DIM = 256
Y_DIM = 256
Z_DIM = 32
T_DIM = 32

def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return 1 - (intersection+smooth)/(union+smooth)

def dice(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    
    return 750*(1 - (2. * intersection + smooth)/(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth))

def quality_peak(y_true, y_pred):
    peak_ratio = tf.reduce_max(y_pred) / y_pred[0]
    peak_ratio = tf.cast(peak_ratio, tf.float32)
    
    return peak_ratio*(100/7.546971123202499)

def quality_tail(y_true, y_pred):
    # end is mean of last 20% of curve
    end_ratio = tf.reduce_mean(y_pred[-int(float(int(len(y_pred)))*0.2):]) / y_pred[0]
    end_ratio = tf.cast(end_ratio, tf.float32)

    quality = (1 / (end_ratio + 1)) * (100/0.24035631585328981)
    # if quality > 200:
    #     quality = 200
    return quality

def quality_peak_to_end(y_true, y_pred):
    peak_ratio = quality_peak(y_true, y_pred)/(100/7.546971123202499)
    end_ratio = tf.reduce_mean(y_pred[-int(float(int(len(y_pred)))*0.2):]) / y_pred[0]
    end_ratio = tf.cast(end_ratio, tf.float32)
    
    return (peak_ratio / end_ratio)*(100/2.4085609761976534)

def quality_peak_time(y_true, y_pred):
    peak_time = tf.argmax(y_pred)
    peak_time = tf.cast(peak_time, tf.float32)
    num_timeslices = tf.cast(len(y_pred), tf.float32)
    qpt = (num_timeslices - peak_time) / num_timeslices
    return qpt*(100/0.9157142857142857)

@keras.saving.register_keras_serializable(package='Custom', name='quality_ultimate')
def quality_ultimate(y_true, y_pred):
    peak_ratio = quality_peak(y_true, y_pred)
    end_ratio = tf.cast(quality_tail(y_true, y_pred), tf.float32)
    peak_to_end = quality_peak_to_end(y_true, y_pred)
    peak_time = quality_peak_time(y_true, y_pred)

    # take weighted average
    return tf.multiply(peak_ratio, 0.3) + tf.multiply(end_ratio, 0.3) + tf.multiply(peak_to_end, 0.3) + tf.multiply(peak_time, 0.1)
    # return peak_ratio + 0.3*end_ratio + 0.3*peak_to_end + 0.1*peak_time

def quality_peak_new(y_true, y_pred):
    peak_ratio = tf.reduce_max(y_pred) / tf.reduce_mean(y_pred)
    peak_ratio = tf.cast(peak_ratio, tf.float32)

    return peak_ratio*(100/2.190064)

def quality_tail_new(y_true, y_pred):
    # end is mean of last 20% of curve
    end_ratio = tf.reduce_mean(y_pred[-int(float(int(len(y_pred)))*0.2):]) / y_pred[0]
    end_ratio = tf.cast(end_ratio, tf.float32)

    # quality = (tf.reduce_mean(tf.cast(y_pred, tf.float32)) / (end_ratio + tf.reduce_mean(tf.cast(y_pred,tf.float32))))
    quality = (1 - pow(tf.cast(end_ratio, tf.float32) / (1.1 * tf.reduce_mean(tf.cast(y_pred, tf.float32))), 2))
    # if quality > 200:
    #     quality = 200
    return pow(quality, 2)*(100/0.3368557913079319)

def quality_base_to_mean_new(y_true, y_pred):
    # peak_ratio = quality_peak(y_true, y_pred)
    # end_ratio = tf.reduce_mean(y_pred[-int(float(int(len(y_pred)))*0.2):]) / y_pred[0]
    # end_ratio = tf.cast(end_ratio, tf.float32)

    # return (peak_ratio / end_ratio)
    return tf.cast((1 - pow(1 / tf.reduce_mean(y_pred), 2)), tf.float32)*(100/0.886713712992177)

def quality_peak_time_new(y_true, y_pred):
    peak_time = tf.argmax(y_pred)
    peak_time = tf.cast(peak_time, tf.float32)
    num_timeslices = tf.cast(len(y_pred), tf.float32)
    qpt = (num_timeslices - peak_time) / num_timeslices
    return tf.cast(qpt, tf.float32)*(100/0.9107566964285715)

@keras.saving.register_keras_serializable(package='Custom', name='quality_ultimate_new')
def quality_ultimate_new(y_true, y_pred):
    peak_ratio = quality_peak_new(y_true, y_pred)
    end_ratio = tf.cast(quality_tail_new(y_true, y_pred), tf.float32)
    base_to_mean = quality_base_to_mean_new(y_true, y_pred)
    peak_time = quality_peak_time_new(y_true, y_pred)

    # take weighted average
    return tf.multiply(peak_ratio, 0.3) + tf.multiply(end_ratio, 0.3) + tf.multiply(base_to_mean, 0.3) + tf.multiply(peak_time, 0.1)
    # return peak_ratio + 0.3*end_ratio + 0.3*base_to_mean + 0.1*peak_time

@keras.saving.register_keras_serializable(package='Custom', name='loss_mae')
def loss_mae(y_true, y_pred, scale_loss = True):
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
    
#     if scale_loss:
#         return 200 * loss
#     else:
    return loss
        
def combined_loss(y_true, y_pred, scale_loss = True):
    flatten = tf.keras.layers.Flatten()
    
    # normalize data to emphasize intensity curve shape over magnitudes
    y_true_f = flatten(y_true / (y_true[:, 0]))
    y_pred_f = flatten(y_pred / (y_pred[:, 0]))
    
    mae = tf.keras.losses.MeanAbsoluteError()
    # weight 6:2 ratio of first 10 repetitions to last 22 repetitions
    weights = np.ones(32)*6
    loss1 = mae(y_true_f, y_pred_f, sample_weight=weights)
    
#     cosine_loss = tf.keras.losses.CosineSimilarity()
#     loss2 = cosine_loss(y_true_f, y_pred_f)
    
    return 200*loss1

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


# def unet3d(img_size = (None, None, None),learning_rate = 1e-8,\
#                  learning_decay = 1e-8, drop_out = 0.35, nchannels = T_DIM, weights = [0, 1, 0, 0]):

def unet3d(img_size = (None, None, None), kernel_size_ao=(3, 11, 11), kernel_size_body=(3, 7, 7), learning_rate = 1e-8,\
                 learning_decay = 0.9, drop_out = 0.35, nchannels = T_DIM, weights = [0, 1, 0], optimizer = 'adam'):
    dropout = drop_out
    input_img = tf.keras.layers.Input((img_size[0], img_size[1], img_size[2], nchannels))
    
    # encoder
    # conv1_1 = Conv3D(32, (3, 11, 11), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')
    # conv1_2 = Conv3D(32, (3, 11, 11), activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv1_1)
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
    conv4_1 = Conv3D(256, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(pool3)
    conv4_2 = Conv3D(256, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv4_1)
    conv4_2 = self_attention_block(conv4_2, 256)

    # decoder
    up1_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4_2), conv3_2],axis=-1)
    conv5_1 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up1_1)
    conv5_2 = Conv3D(128, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv5_1)
    conv5_2 = tfa.layers.InstanceNormalization()(conv5_2)

    up2_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5_2), conv2_2],axis=-1)
    conv6_1 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(up2_1)
    conv6_2 = Conv3D(64, kernel_size_body, activation=keras.layers.LeakyReLU(alpha=0.3), padding='same')(conv6_1)

    up3_1 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6_2), conv1_2],axis=-1)
#     up3_1 = Dropout(dropout)(up3_1)
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

    # opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, decay = learning_decay)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=10000,
        decay_rate=0.9)
    if optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    elif optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

    model.compile(optimizer=opt, loss={
        "cast" : [loss_computeCofDistance3D],
        "vf" : [loss_mae],
        # "vol" : [loss_volume],
        # "lambda_quality" : [quality_ultimate]
    },
    metrics = {"vf" : [quality_ultimate]},
    loss_weights = weights)

    return model
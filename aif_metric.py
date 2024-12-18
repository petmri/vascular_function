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

# get metric of AIF
def quality_peak(aif_curve):
    peak_ratio = max(aif_curve) / aif_curve[0]
    # peak_ratio = tf.cast(peak_ratio, tf.float32)
    return peak_ratio*(100/7.546971123202499)

def quality_tail(aif_curve):
    # end is mean of last 20% of curve
    end_ratio = np.mean(aif_curve[-int(float(int(len(aif_curve)))*0.2):]) / aif_curve[0]
    # end_ratio = tf.cast(end_ratio, tf.float32)

    quality = (1 / (end_ratio + 1)) * (100/0.24035631585328981)
    # if quality > 200:
    #     quality = 200
    return quality

def quality_peak_to_end(aif_curve):
    peak_ratio = quality_peak(aif_curve)/(100/7.546971123202499)
    end_ratio = np.mean(aif_curve[-int(float(int(len(aif_curve)))*0.2):]) / aif_curve[0]
    # end_ratio = tf.cast(end_ratio, tf.float32)
    
    return (peak_ratio / end_ratio)*(100/2.4085609761976534)

def quality_peak_time(aif_curve):
    peak_time = np.argmax(aif_curve)
    # peak_time = tf.cast(peak_time, tf.float32)
    num_timeslices = len(aif_curve)
    qpt = (num_timeslices - peak_time) / num_timeslices
    return qpt*(100/0.9157142857142857)

def quality_ultimate(aif_curve):
    peak_ratio = quality_peak(aif_curve)
    end_ratio = quality_tail(aif_curve)
    peak_to_end = quality_peak_to_end(aif_curve)
    peak_time = quality_peak_time(aif_curve)

    # take weighted average
    return peak_ratio*0.3 + end_ratio*0.3 + peak_to_end*0.3 + peak_time*0.1
    # return peak_ratio + 0.3*end_ratio + 0.3*peak_to_end + 0.1*peak_time

def quality_peak_new(aif_curve):
    peak_ratio = max(aif_curve) / np.mean(aif_curve)
    return (1 / (1 + np.exp(-3.5 * peak_ratio + 7.5))) * (100 / 0.4499714351078607)

def quality_tail_new(aif_curve):
    end_ratio = np.mean(aif_curve[-int(len(aif_curve) * 0.2):])
    quality = (1 - (end_ratio / (1.1 * np.mean(aif_curve))) ** 2)
    return quality * (100 / 0.33436023529043213)

def quality_base_to_mean_new(aif_curve):
    return (1 - (get_baseline_from_curve(aif_curve) / np.mean(aif_curve)) ** 2) * (100 / 0.8831850876454762)

def quality_peak_time_new(aif_curve):
    peak_time = np.argmax(aif_curve)
    num_timeslices = len(aif_curve)
    qpt = (num_timeslices - peak_time) / num_timeslices
    return qpt * (100 / 0.9081383928571428)

def quality_ultimate_new(aif_curve):
    peak_ratio = quality_peak_new(aif_curve)
    end_ratio = quality_tail_new(aif_curve)
    base_to_mean = quality_base_to_mean_new(aif_curve)
    peak_time = quality_peak_time_new(aif_curve)

    # take weighted average
    return peak_ratio * 0.3 + end_ratio * 0.3 + base_to_mean * 0.3 + peak_time * 0.1

def get_baseline_from_curve(curve):
    peak_index = np.argmax(curve)
    return np.mean(curve[:peak_index-1][np.where(curve[:peak_index-1] < curve[0] * 1.75)])
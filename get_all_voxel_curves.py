import nibabel as nib
import model_vif
import matplotlib.pyplot as plt
import cupy as cp
from numba import cuda, float64

import numpy as np
from numba import njit

import numpy as np
from numba import njit

@njit
def custom_max(array):
    max_val = array[0]
    for i in range(1, len(array)):
        if array[i] > max_val:
            max_val = array[i]
    return max_val

@njit
def custom_mean(array):
    sum_val = 0.0
    for i in range(len(array)):
        sum_val += array[i]
    return sum_val / len(array)

@njit
def custom_argmax(array):
    max_index = 0
    max_val = array[0]
    for i in range(1, len(array)):
        if array[i] > max_val:
            max_val = array[i]
            max_index = i
    return max_index

@njit
def quality_peak_new(y_pred):
    peak_ratio = custom_max(y_pred) / custom_mean(y_pred)
    # return peak_ratio * (100 / 2.190064)
    return (1 / (1 + np.e**(-3.5*peak_ratio+7.5)))*(100/1)

@njit
def quality_tail_new(y_pred):
    end_idx = int(len(y_pred) * 0.2)
    end_mean = custom_mean(y_pred[-end_idx:])
    end_ratio = end_mean / y_pred[0]
    # quality = (1 - np.e ** (end_ratio / (1.1 * custom_mean(y_pred))))
    quality = (1 - (end_ratio / (1.1 * custom_mean(y_pred))) ** 2)
    return quality * (100 / 0.7194740924786208)

@njit
def quality_base_to_mean_new(y_pred):
    return (1 - (y_pred[0] / custom_mean(y_pred)) ** 2) * (100 / 0.886713712992177)

@njit
def quality_peak_time_new(y_pred):
    peak_time = custom_argmax(y_pred)
    num_timeslices = len(y_pred)
    qpt = (num_timeslices - peak_time) / num_timeslices
    return qpt * (100 / 0.9107566964285715)

@njit
def quality_ultimate_new(y_pred):
    peak_ratio = quality_peak_new(y_pred)
    end_ratio = quality_tail_new(y_pred)
    base_to_mean = quality_base_to_mean_new(y_pred)
    peak_time = quality_peak_time_new(y_pred)

    return peak_ratio * 0.3 + end_ratio * 0.3 + base_to_mean * 0.3 + peak_time * 0.1


# Load image and move data to GPU
image_path = '/media/network_mriphysics/USC-PPG/AI_training/loos_model/test/images/500220_1st_timepoint.nii.gz'
volume_img = nib.load(image_path)
volume_data = cp.array(volume_img.get_fdata())
# sample 5 voxels for testing
# volume_data = volume_data[1:2, 1:2, 1:5, :]

# Assume a maximum time dimension (e.g., 100)
MAX_TIME_DIM = volume_data.shape[3]

# Prepare an array for storing voxel scores
vfs_scores = cp.zeros(volume_data.shape[:-1])

@cuda.jit
def calculate_voxel_scores(volume_data, vfs_scores):
    i, j, k = cuda.grid(3)
    if i < volume_data.shape[0] and j < volume_data.shape[1] and k < volume_data.shape[2]:
        # Allocate shared memory for curve_norm
        shared_curve_norm = cuda.shared.array(shape=(MAX_TIME_DIM,), dtype=float64)
        curve_0 = volume_data[i, j, k, 0]

        # Handle division by zero
        if curve_0 == 0:
            vfs_scores[i, j, k] = float64(0.0)  # or another default value
            return

        # Load data into shared memory
        # for t in range(volume_data.shape[3]):
        #     shared_curve_norm[t] = volume_data[i, j, k, t] / curve_0
        shared_curve_norm = volume_data[i, j, k, :]
        
        # Synchronize threads to ensure all data is loaded
        cuda.syncthreads()

        # Compute the score
        score = quality_ultimate_new(shared_curve_norm)
        vfs_scores[i, j, k] = score

# Define the block and grid sizes for CUDA
threads_per_block = (4, 4, 4)
blocks_per_grid_x = (volume_data.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (volume_data.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid_z = (volume_data.shape[2] + threads_per_block[2] - 1) // threads_per_block[2]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

# Launch the kernel on the GPU
calculate_voxel_scores[blocks_per_grid, threads_per_block](volume_data, vfs_scores)

# Copy the results back to the host for further processing
# print(vfs_scores)
vfs_scores_cpu = cp.asnumpy(vfs_scores)
# print(vfs_scores_cpu)

# Sorting and plotting
sorted_indices = cp.argsort(vfs_scores_cpu.flatten())[::-1]
print("Top 5 voxel scores: ", vfs_scores_cpu.flatten()[sorted_indices[:20]])
top_indices = cp.unravel_index(sorted_indices[:20], vfs_scores_cpu.shape)
for idx in zip(*top_indices):
    i, j, k = idx
    curve = volume_data[i, j, k, :].get()
    # expand figure size
    plt.figure(figsize=(10, 5))
    plt.plot(curve)
    peak = quality_peak_new(curve)
    tail = quality_tail_new(curve)
    btm = quality_base_to_mean_new(curve)
    at = quality_peak_time_new(curve)
    ult = quality_ultimate_new(curve)
    plt.title(f"voxel: {idx}, ult: {ult}, ptm: {peak}, tail: {tail}, btm: {btm}, at: {at}", fontsize=8)
    plt.savefig(f"ult_{np.floor(ult)}_voxel_{i}_{j}_{k}.png")
    plt.close()

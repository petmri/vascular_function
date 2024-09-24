# Import and compare different AIFs

import os
from statistics import correlation
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import nibabel as nib
from matplotlib import colors as mcolors
import re

print("Hello World")

# path to AIF values
manual_aif_values_dir = '/media/network_mriphysics/USC-PPG/bids_test/derivatives/dceprep-manualAIF'
auto_aif_values_dir = '/media/network_mriphysics/USC-PPG/bids_test/derivatives/dceprep-autoAIF_testset'

# path to subject IDs to compare
id_list_dir = '/media/network_mriphysics/USC-PPG/AI_training/weights/new_GRASP_masks/train_set.txt'
#training_images_dir = '/media/network_mriphysics/USC-PPG/AI_training/loos_model/'

# path to output directory
output_dir = '/media/network_mriphysics/USC-PPG/AI_training/results/test_score'


# read in the subject IDs from test list
with open(id_list_dir) as f:
    id_list = f.readlines()

aif_values = {}

# find all files from subject IDs in test list
for id in id_list:
    id = id.strip()

    # if id contains "LLU" or "Public", skip 
    # TODO: remove one we have this data
    if re.search(r'LLU', id) or re.search(r'Public', id):
        continue

    # use a regular expression to check id for a 6+ digit number and save it as a subject ID
    subject_id = re.search(r'\d+', id).group(0)
    subject_id = 'sub-' + subject_id
    # search id for "1st" or "ses-##" and save it as a session ID
    if re.search(r'1st', id) or re.search(r'ses-01', id):
        session_id = 'ses-01'
    elif re.search(r'2nd', id) or re.search(r'ses-02', id):
        session_id = 'ses-02'
    elif re.search(r'3rd', id) or re.search(r'ses-03', id):
        session_id = 'ses-03'
    elif re.search(r'4th', id) or re.search(r'ses-04', id):  
        session_id = 'ses-04'

    # Find, load AIF values
    manual_aif_file = os.path.join(manual_aif_values_dir, subject_id, session_id,'dce','B_dcefitted_R1info.log')
    auto_aif_file = os.path.join(auto_aif_values_dir, subject_id,session_id,'dce','B_dcefitted_R1info.log')

    # check if the files exist
    if not os.path.exists(manual_aif_file):
        #print(f"Manual AIF file for {subject_id} does not exist.")
        continue
    if not os.path.exists(auto_aif_file):
        print(f"Auto AIF file for {subject_id} does not exist.")
        continue

    # read in the AIF values
    with open(manual_aif_file) as f:
        manual_aif_text = f.readlines()
    with open(auto_aif_file) as f:
        auto_aif_text = f.readlines()
    manual_aif_section = re.search(r'AIF mmol:\s*(.*?)(?:\n\n|Finished B)', ''.join(manual_aif_text), re.DOTALL).group(1)
    auto_aif_section = re.search(r'AIF mmol:\s*(.*?)(?:\n\n|Finished B)', ''.join(auto_aif_text), re.DOTALL).group(1)

    # Find all floating-point numbers in the extracted section
    manual_aif_float = np.array([float(num) for num in re.findall(r'\d+\.\d+', manual_aif_section)])
    auto_aif_float = np.array([float(num) for num in re.findall(r'\d+\.\d+', auto_aif_section)])
    
    # save in a dictionary
    aif_values[subject_id] = {
        'session_id': session_id,
        'manual_aif_float': manual_aif_float,
        'auto_aif_float': auto_aif_float
    }

    # Find, load Ktrans values
    manual_ktrans_file = os.path.join(manual_aif_values_dir, subject_id, session_id,'dce',subject_id+'_'+session_id+'_Ktrans.nii')  
    auto_ktrans_file = os.path.join(auto_aif_values_dir, subject_id,session_id,'dce',subject_id+'_'+session_id+'_Ktrans.nii')

    # check if the files exist
    if not os.path.exists(manual_ktrans_file):
        print(f"Manual Ktrans file for {subject_id} does not exist.")
        continue
    if not os.path.exists(auto_ktrans_file):
        print(f"Auto Ktrans file for {subject_id} does not exist.")
        continue

    # Load image
    manual_volume_img = nib.load(manual_ktrans_file)
    manual_volume_data = manual_volume_img.get_fdata()
    auto_volume_img = nib.load(auto_ktrans_file)
    auto_volume_data = auto_volume_img.get_fdata()

    # append the Ktrans values to the dictionary
    aif_values[subject_id]['manual_ktrans'] = manual_volume_data
    aif_values[subject_id]['auto_ktrans'] = auto_volume_data

    
print(f"Number of subjects: {len(aif_values)}")

# Get all items in the dictionary
manual_mean_list = []
auto_mean_list = []
manual_max_list = []
auto_max_list = []
manual_ktrans_list = []
auto_ktrans_list = []

for key, value in aif_values.items():
    #print(f"Subject ID: {key}")
    #print(f"Session ID: {value['session_id']}")

    # Process AIF values
    if 'manual_aif_float' in value and 'auto_aif_float' in value:
        manual_mean = np.mean(value['manual_aif_float'])
        manual_mean_list.append(manual_mean)
        auto_mean = np.mean(value['auto_aif_float'])
        auto_mean_list.append(auto_mean)
        manual_max = value['manual_aif_float'].max()
        manual_max_list.append(manual_max)
        auto_max = value['auto_aif_float'].max()
        auto_max_list.append(auto_max)

    # Process Ktrans values
    if 'manual_ktrans' in value and 'auto_ktrans' in value:
        # Get mean value of manual_ktrans excluding zeros
        manual_ktrans = np.array(value['manual_ktrans'])
        valid_manual_ktrans = manual_ktrans[(manual_ktrans != 0)]
        if valid_manual_ktrans.size == 0:
            manual_ktrans_mean = 0
            print(f"Manual Ktrans for {key} is all zeros")
        else:
            manual_ktrans_mean = np.mean(valid_manual_ktrans)
            
        # Get mean value of auto_ktrans excluding zeros
        auto_ktrans = np.array(value['auto_ktrans'])
        valid_auto_ktrans = auto_ktrans[(auto_ktrans != 0)]
        if valid_auto_ktrans.size == 0:
            auto_ktrans_mean = 0
            print(f"Auto Ktrans for {key} is all zeros")
        else:
            auto_ktrans_mean = np.mean(valid_auto_ktrans)
        
        # exclude high outliers, they skew the r^2
        if manual_ktrans_mean<0.03 and auto_ktrans_mean<0.03:
            manual_ktrans_list.append(manual_ktrans_mean)    
            auto_ktrans_list.append(auto_ktrans_mean)

# Plot AIF values
plt.figure()
plt.scatter(manual_mean_list, auto_mean_list)
plt.xlabel('Manual Mean')
plt.ylabel('Auto Mean')
plt.title('Mean AIF Values')
max_axis = 2
plt.xlim(0, max_axis)
plt.ylim(0,max_axis)
# add a line of best fit
p = Polynomial.fit(manual_mean_list, auto_mean_list, 1)
x_vals = np.linspace(0, max_axis, 100)
plt.plot(x_vals, p(x_vals), color='gray')
# show the r^2 value on the plot, limit to 4 decimal places
correlation_matrix = np.corrcoef(manual_mean_list, auto_mean_list)
correlation_xy = correlation_matrix[0,1]
r_squared = round(correlation_matrix[0,1]**2,4)
plt.text(0.1*max_axis, 0.9*max_axis, f"$r^2$ = {r_squared}")


# Plot Ktrans values
plt.figure()
plt.scatter(manual_ktrans_list, auto_ktrans_list)
plt.xlabel('Manual Ktrans')
plt.ylabel('Auto Ktrans')
plt.title('Ktrans Values')
max_axis = 0.006
plt.xlim(0, max_axis)
plt.ylim(0,max_axis)
# add a line of best fit
p = Polynomial.fit(manual_ktrans_list, auto_ktrans_list, 1)
x_vals = np.linspace(0, max_axis, 100)
plt.plot(x_vals, p(x_vals), color='gray')
# show the r^2 value on the plot, limit to 4 decimal places
correlation_matrix = np.corrcoef(manual_ktrans_list, auto_ktrans_list)
correlation_xy = correlation_matrix[0,1]
r_squared = round(correlation_matrix[0,1]**2,4)
plt.text(0.1*max_axis, 0.9*max_axis, f"$r^2$ = {r_squared}")
plt.show()



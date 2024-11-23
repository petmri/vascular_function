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
import csv

import pandas as pd
from aif_metric import quality_ultimate_new
from scipy.stats import ttest_ind, ttest_rel
import pingouin as pg

ktrans_upper_limit = 0.06

print("Starting AIF comparison...")

# path to AIF values
manual_aif_values_dir = '/media/network_mriphysics/USC-PPG/bids_test/derivatives/dceprep-manualAIF'
auto_aif_values_dir = '/media/network_mriphysics/USC-PPG/bids_test/derivatives/dceprep-autoAIF_huber1'
#auto_aif_values_dir = '/media/network_mriphysics/USC-PPG/bids_test/derivatives/dceprep-autoAIF_rg_10-13'
#auto_aif_values_dir = '/media/network_mriphysics/USC-PPG/bids_test/derivatives/dceprep-smoothed'

# path to subject IDs to compare
id_list_dir = '/media/network_mriphysics/USC-PPG/AI_training/weights/rg_latest/test_set.txt'
#training_images_dir = '/media/network_mriphysics/USC-PPG/AI_training/loos_model/'

# path to output directory
output_dir = '/media/network_mriphysics/USC-PPG/AI_training/results/aif_comparison'


# read in the subject IDs from test list
with open(id_list_dir) as f:
    id_list = f.readlines()

print(f"Number of Test IDs Found: {len(id_list)}")

# find all files from subject IDs in test list
aif_values = {}
for id in id_list:
    id = id.strip()

    #print(f"Running {id} subject {n} of {len(id_list)}")

    # Get subject ID and session ID
    if re.search(r'CMR1OWO', id): #alt 'Public'
        #continue
        # "Public" data has a different subject ID format
        search_results = re.search(r'Pat\d+', id)
    else:
        # use a regular expression to check id for a 6+ digit number and save it as a subject ID
        search_results = re.search(r'\d{6,}', id)
    if search_results:
        subject_id = 'sub-' + search_results.group(0)
    else:
        print(f"Subject ID not found for {id}")
        continue
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
    manual_aif_scaled_file = os.path.join(manual_aif_values_dir, subject_id, session_id,'dce','AIF_values.txt')
    auto_aif_file = os.path.join(auto_aif_values_dir, subject_id,session_id,'dce','B_dcefitted_R1info.log')
    auto_aif_scaled_file = os.path.join(auto_aif_values_dir, subject_id,session_id,'dce','AIF_values.txt')

    # check if the files exist
    if not os.path.exists(manual_aif_file) or not os.path.exists(manual_aif_scaled_file):
        print(f"Manual AIF file for {subject_id} does not exist.")
        continue
    if not os.path.exists(auto_aif_file) or not os.path.exists(auto_aif_scaled_file):
        print(f"Auto AIF file for {subject_id} does not exist.")
        continue

    # read in the AIF values
    with open(manual_aif_file) as f:
        manual_aif_text = f.readlines()
    with open(auto_aif_file) as f:
        auto_aif_text = f.readlines()
    manual_aif_section_mmolar = re.search(r'AIF mmol:\s*(.*?)(?:\n\n|Finished B)', ''.join(manual_aif_text), re.DOTALL).group(1)
    auto_aif_section_mmolar = re.search(r'AIF mmol:\s*(.*?)(?:\n\n|Finished B)', ''.join(auto_aif_text), re.DOTALL).group(1)

    # read the AIF_text file
    with open(manual_aif_scaled_file) as f:
        manual_aif_scaled_text = f.readlines()
    with open(auto_aif_scaled_file) as f:
        auto_aif_scaled_text = f.readlines()

    # Find all floating-point numbers in the extracted section
    manual_aif_float_mmolar = np.array([float(num) for num in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', manual_aif_section_mmolar)])
    auto_aif_float_mmolar = np.array([float(num) for num in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', auto_aif_section_mmolar)])
    # convert auto_aif_scaled_text to a np.array of floats
    manual_aif_scaled_float = np.array([float(num) for num in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', ''.join(manual_aif_scaled_text))])
    auto_aif_scaled_float = np.array([float(num) for num in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', ''.join(auto_aif_scaled_text))])

    # save in a dictionary
    aif_values[subject_id+session_id] = {
        'manual_aif_float_mmolar': manual_aif_float_mmolar,
        'auto_aif_float_mmolar': auto_aif_float_mmolar,
        'manual_aif_scaled_float': manual_aif_scaled_float,
        'auto_aif_scaled_float': auto_aif_scaled_float
    }

    # Find, load Ktrans values
    manual_ktrans_file = os.path.join(manual_aif_values_dir, subject_id, session_id,'dce',subject_id+'_'+session_id+'_Ktrans.nii')  
    manual_ktrans_GM_file = os.path.join(manual_aif_values_dir, subject_id, session_id,'dce',subject_id+'_'+session_id+'_seg-GM_Ktrans.nii.gz')  
    manual_ktrans_WM_file = os.path.join(manual_aif_values_dir, subject_id, session_id,'dce',subject_id+'_'+session_id+'_seg-WM_Ktrans.nii.gz')  
    manual_ktrans_cerb_file = os.path.join(manual_aif_values_dir, subject_id, session_id,'dce',subject_id+'_'+session_id+'_seg-Cerebellum_Ktrans.nii.gz')
    manual_ktrans_muscle_file = os.path.join(manual_aif_values_dir, subject_id, session_id,'dce',subject_id+'_'+session_id+'_seg-Muscle_Ktrans.nii.gz')
    auto_ktrans_file = os.path.join(auto_aif_values_dir, subject_id,session_id,'dce',subject_id+'_'+session_id+'_Ktrans.nii')
    auto_ktrans_GM_file = os.path.join(auto_aif_values_dir, subject_id,session_id,'dce',subject_id+'_'+session_id+'_seg-GM_Ktrans.nii.gz')
    auto_ktrans_WM_file = os.path.join(auto_aif_values_dir, subject_id,session_id,'dce',subject_id+'_'+session_id+'_seg-WM_Ktrans.nii.gz')
    auto_ktrans_cerb_file = os.path.join(auto_aif_values_dir, subject_id,session_id,'dce',subject_id+'_'+session_id+'_seg-Cerebellum_Ktrans.nii.gz')
    auto_ktrans_muscle_file = os.path.join(auto_aif_values_dir, subject_id,session_id,'dce',subject_id+'_'+session_id+'_seg-Muscle_Ktrans.nii.gz')

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
    aif_values[subject_id+session_id]['manual_ktrans'] = manual_volume_data
    aif_values[subject_id+session_id]['auto_ktrans'] = auto_volume_data

    if os.path.exists(manual_ktrans_GM_file) and os.path.exists(auto_ktrans_GM_file):
        manual_GM_volume_img = nib.load(manual_ktrans_GM_file)
        manual_GM_volume_data = manual_GM_volume_img.get_fdata()
        auto_GM_volume_img = nib.load(auto_ktrans_GM_file)
        auto_GM_volume_data = auto_GM_volume_img.get_fdata()
        aif_values[subject_id+session_id]['manual_GM_ktrans'] = manual_GM_volume_data
        aif_values[subject_id+session_id]['auto_GM_ktrans'] = auto_GM_volume_data

    if os.path.exists(manual_ktrans_WM_file) and os.path.exists(auto_ktrans_WM_file):
        manual_WM_volume_img = nib.load(manual_ktrans_WM_file)
        manual_WM_volume_data = manual_WM_volume_img.get_fdata()
        auto_WM_volume_img = nib.load(auto_ktrans_WM_file)
        auto_WM_volume_data = auto_WM_volume_img.get_fdata()
        aif_values[subject_id+session_id]['manual_WM_ktrans'] = manual_WM_volume_data
        aif_values[subject_id+session_id]['auto_WM_ktrans'] = auto_WM_volume_data

    if os.path.exists(manual_ktrans_cerb_file) and os.path.exists(auto_ktrans_cerb_file):
        manual_cerb_volume_img = nib.load(manual_ktrans_cerb_file)
        manual_cerb_volume_mask = manual_cerb_volume_img.get_fdata()
        manual_cerb_volume_data = manual_volume_data[manual_cerb_volume_mask==1]
        auto_cerb_volume_img = nib.load(auto_ktrans_cerb_file)
        auto_cerb_volume_mask = auto_cerb_volume_img.get_fdata()
        auto_cerb_volume_data = auto_volume_data[auto_cerb_volume_mask==1]
        aif_values[subject_id+session_id]['manual_cerb_ktrans'] = manual_cerb_volume_data
        aif_values[subject_id+session_id]['auto_cerb_ktrans'] = auto_cerb_volume_data
    
    if os.path.exists(manual_ktrans_muscle_file) and os.path.exists(auto_ktrans_muscle_file):
        manual_muscle_volume_img = nib.load(manual_ktrans_muscle_file)
        manual_muscle_volume_mask = manual_muscle_volume_img.get_fdata()
        manual_muscle_volume_data = manual_volume_data[manual_muscle_volume_mask==1]
        auto_muscle_volume_img = nib.load(auto_ktrans_muscle_file)
        auto_muscle_volume_mask = auto_muscle_volume_img.get_fdata()
        auto_muscle_volume_data = auto_volume_data[auto_muscle_volume_mask==1]
        aif_values[subject_id+session_id]['manual_muscle_ktrans'] = manual_muscle_volume_data
        aif_values[subject_id+session_id]['auto_muscle_ktrans'] = auto_muscle_volume_data



print(f"Data found for subjects: {len(aif_values)}")

# Get all items in the dictionary
subject_id_list = list(aif_values.keys())
manual_mean_mmolar_list = []
auto_mean_mmolar_list = []
manual_aifitness_list = []
auto_aifitness_list = []
manual_max_list = []
auto_max_list = []
manual_ktrans_list = []
auto_ktrans_list = []
manual_ktrans_GM_list = []
auto_ktrans_GM_list = []
manual_ktrans_WM_list = []
auto_ktrans_WM_list = []
manual_ktrans_cerb_list = []
auto_ktrans_cerb_list = []
manual_ktrans_muscle_list = []
auto_ktrans_muscle_list = []
csv_list = []

for key, value in aif_values.items():
    #print(f"Subject ID: {key}")
    #print(f"Session ID: {value['session_id']}")

    # Initialize variables to ''
    manual_mmolar_mean = auto_mmolar_mean = manual_max = auto_max = manual_aifitness = \
        auto_aifitness = manual_ktrans_mean = \
        auto_ktrans_mean = manual_GM_ktrans_mean = auto_GM_ktrans_mean = \
        manual_WM_ktrans_mean = auto_WM_ktrans_mean = manual_cerb_ktrans_mean = \
        auto_cerb_ktrans_mean = manual_muscle_ktrans_mean = auto_muscle_ktrans_mean = ''
    
    # Process AIFitness values
    if 'manual_aif_scaled_float' in value and 'auto_aif_scaled_float' in value:
        manual_aifitness = quality_ultimate_new(value['manual_aif_scaled_float'])
        auto_aifitness = quality_ultimate_new(value['auto_aif_scaled_float'])
        if auto_aifitness < 54:
            print(f"Auto AIFitness for {key} is less than 54: {auto_aifitness}")
            continue
        manual_aifitness_list.append(manual_aifitness)
        auto_aifitness_list.append(auto_aifitness)

    # save subject 500256 aif scaled values
    if key == 'sub-500256ses-01':
        manual_aifitness_500256 = manual_aifitness
        auto_aifitness_500256 = auto_aifitness
        manual_aif_scaled_float_ideal = value['manual_aif_scaled_float']
        auto_aif_scaled_float_ideal = value['auto_aif_scaled_float']

    # Process AIF values
    if 'manual_aif_float_mmolar' in value and 'auto_aif_float_mmolar' in value:
        manual_mmolar_mean = np.mean(value['manual_aif_float_mmolar'])
        manual_mean_mmolar_list.append(manual_mmolar_mean)
        auto_mmolar_mean = np.mean(value['auto_aif_float_mmolar'])
        auto_mean_mmolar_list.append(auto_mmolar_mean)
        manual_max = value['manual_aif_float_mmolar'].max()
        manual_max_list.append(manual_max)
        auto_max = value['auto_aif_float_mmolar'].max()
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
            manual_ktrans_mean = np.median(valid_manual_ktrans)
            
        # Get mean value of auto_ktrans excluding zeros
        auto_ktrans = np.array(value['auto_ktrans'])
        valid_auto_ktrans = auto_ktrans[(auto_ktrans != 0)]
        if valid_auto_ktrans.size == 0:
            auto_ktrans_mean = 0
            print(f"Auto Ktrans for {key} is all zeros")
        else:
            auto_ktrans_mean = np.median(valid_auto_ktrans)
        
        # exclude high outliers, they skew the r^2
        if manual_ktrans_mean<ktrans_upper_limit and auto_ktrans_mean<ktrans_upper_limit:
            manual_ktrans_list.append(manual_ktrans_mean)    
            auto_ktrans_list.append(auto_ktrans_mean)
            #print(f"Ktrans values for {key} are {manual_ktrans_mean} and {auto_ktrans_mean}")
        else:
            print(f"Ktrans value for {key} is above limit: {ktrans_upper_limit}")
    
    #Process GM Ktrans values
    if 'manual_GM_ktrans' in value and 'auto_GM_ktrans' in value:
        # Get mean value of manual_ktrans excluding zeros
        manual_GM_ktrans = np.array(value['manual_GM_ktrans'])
        valid_manual_GM_ktrans = manual_GM_ktrans[(manual_GM_ktrans != 0)]
        if valid_manual_GM_ktrans.size == 0:
            manual_GM_ktrans_mean = 0
            print(f"Manual GM Ktrans for {key} is all zeros")
        else:
            manual_GM_ktrans_mean = np.median(valid_manual_GM_ktrans)
        # Get mean value of auto_ktrans excluding zeros
        auto_GM_ktrans = np.array(value['auto_GM_ktrans'])
        valid_auto_GM_ktrans = auto_GM_ktrans[(auto_GM_ktrans != 0)]
        if valid_auto_GM_ktrans.size == 0:
            auto_GM_ktrans_mean = 0
            print(f"Auto GM Ktrans for {key} is all zeros")
        else:
            auto_GM_ktrans_mean = np.median(valid_auto_GM_ktrans)
        # exclude high outliers, they skew the r^2
        if manual_GM_ktrans_mean<ktrans_upper_limit and auto_GM_ktrans_mean<ktrans_upper_limit:
            manual_ktrans_GM_list.append(manual_GM_ktrans_mean)    
            auto_ktrans_GM_list.append(auto_GM_ktrans_mean)
        else:
            print(f"Ktrans value for {key} is above limit: {ktrans_upper_limit}")
    
    #Process WM Ktrans values
    if 'manual_WM_ktrans' in value and 'auto_WM_ktrans' in value:
        # Get mean value of manual_ktrans excluding zeros
        manual_WM_ktrans = np.array(value['manual_WM_ktrans'])
        valid_manual_WM_ktrans = manual_WM_ktrans[(manual_WM_ktrans != 0)]
        if valid_manual_WM_ktrans.size == 0:
            manual_WM_ktrans_mean = 0
            print(f"Manual WM Ktrans for {key} is all zeros")
        else:
            manual_WM_ktrans_mean = np.median(valid_manual_WM_ktrans)
        # Get mean value of auto_ktrans excluding zeros
        auto_WM_ktrans = np.array(value['auto_WM_ktrans'])
        valid_auto_WM_ktrans = auto_WM_ktrans[(auto_WM_ktrans != 0)]
        if valid_auto_WM_ktrans.size == 0:
            auto_WM_ktrans_mean = 0
            print(f"Auto WM Ktrans for {key} is all zeros")
        else:
            auto_WM_ktrans_mean = np.median(valid_auto_WM_ktrans)
        # exclude high outliers, they skew the r^2
        if manual_WM_ktrans_mean<ktrans_upper_limit and auto_WM_ktrans_mean<ktrans_upper_limit:
            manual_ktrans_WM_list.append(manual_WM_ktrans_mean)    
            auto_ktrans_WM_list.append(auto_WM_ktrans_mean)
        else:
            print(f"Ktrans value for {key} is above limit: {ktrans_upper_limit}")
    
    #Process Cerebellum Ktrans values
    if 'manual_cerb_ktrans' in value and 'auto_cerb_ktrans' in value:
        # Get mean value of manual_ktrans excluding
        manual_cerb_ktrans = np.array(value['manual_cerb_ktrans'])
        valid_manual_cerb_ktrans = manual_cerb_ktrans[(manual_cerb_ktrans != 0)]
        if valid_manual_cerb_ktrans.size == 0:
            manual_cerb_ktrans_mean = 0
            print(f"Manual Cerebellum Ktrans for {key} is all zeros")
        else:
            manual_cerb_ktrans_mean = np.median(valid_manual_cerb_ktrans)
        # Get mean value of auto_ktrans excluding zeros
        auto_cerb_ktrans = np.array(value['auto_cerb_ktrans'])
        valid_auto_cerb_ktrans = auto_cerb_ktrans[(auto_cerb_ktrans != 0)]
        if valid_auto_cerb_ktrans.size == 0:
            auto_cerb_ktrans_mean = 0
            print(f"Auto Cerebellum Ktrans for {key} is all zeros")
        else:
            auto_cerb_ktrans_mean = np.median(valid_auto_cerb_ktrans)
        # exclude high outliers, they skew the r^2
        if manual_cerb_ktrans_mean<ktrans_upper_limit and auto_cerb_ktrans_mean<ktrans_upper_limit:
            manual_ktrans_cerb_list.append(manual_cerb_ktrans_mean)
            auto_ktrans_cerb_list.append(auto_cerb_ktrans_mean)
        else:
            print(f"Ktrans value for {key} is above limit: {ktrans_upper_limit}")
    
    #Process Muscle Ktrans values
    if 'manual_muscle_ktrans' in value and 'auto_muscle_ktrans' in value:
        # Get mean value of manual_ktrans excluding
        manual_muscle_ktrans = np.array(value['manual_muscle_ktrans'])
        valid_manual_muscle_ktrans = manual_muscle_ktrans[(manual_muscle_ktrans != 0)]
        if valid_manual_muscle_ktrans.size == 0:
            manual_muscle_ktrans_mean = 0
            print(f"Manual Muscle Ktrans for {key} is all zeros")
        else:
            manual_muscle_ktrans_mean = np.median(valid_manual_muscle_ktrans)
        # Get mean value of auto_ktrans excluding zeros
        auto_muscle_ktrans = np.array(value['auto_muscle_ktrans'])
        valid_auto_muscle_ktrans = auto_muscle_ktrans[(auto_muscle_ktrans != 0)]
        if valid_auto_muscle_ktrans.size == 0:
            auto_muscle_ktrans_mean = 0
            print(f"Auto Muscle Ktrans for {key} is all zeros")
        else:
            auto_muscle_ktrans_mean = np.median(valid_auto_muscle_ktrans)
        # exclude high outliers, they skew the r^2
        if manual_muscle_ktrans_mean<ktrans_upper_limit and auto_muscle_ktrans_mean<ktrans_upper_limit:
            manual_ktrans_muscle_list.append(manual_muscle_ktrans_mean)
            auto_ktrans_muscle_list.append(auto_muscle_ktrans_mean)
        else:
            print(f"Ktrans value for {key} is above limit: {ktrans_upper_limit}")

    # append the values to the csv list
    csv_list.append([key, manual_mmolar_mean, auto_mmolar_mean, manual_aifitness, auto_aifitness, manual_ktrans_mean, auto_ktrans_mean, manual_GM_ktrans_mean, auto_GM_ktrans_mean, manual_WM_ktrans_mean, auto_WM_ktrans_mean, manual_cerb_ktrans_mean, auto_cerb_ktrans_mean, manual_muscle_ktrans_mean, auto_muscle_ktrans_mean])


# export dictionary values to csv file
csv_filename = output_dir + '/aif_comparison_new.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    headers = ['subject_id+session', 'manual_aif_mmolar_mean', 'auto_aif_mmolar_mean', 'manual_aifitness', 'auto_aifitness', 
               'manual_ktrans_mean', 'auto_ktrans_mean', 
               'manual_GM_ktrans_mean', 'auto_GM_ktrans_mean', 'manual_WM_ktrans_mean', 'auto_WM_ktrans_mean', 
               'manual_cerb_ktrans_mean', 'auto_cerb_ktrans_mean', 'manual_muscle_ktrans_mean', 'auto_muscle_ktrans_mean']
    writer.writerow(headers)
    # Write the values from every list
    writer.writerows(csv_list)

# Plot ideal AIF curve
manual_aif_scaled_float_ideal = manual_aif_scaled_float_ideal * 53.33
auto_aif_scaled_float_ideal = auto_aif_scaled_float_ideal * 53.33
ideal_mean = np.mean(auto_aif_scaled_float_ideal)
#ideal_baseline = get_baseline_from_curve(auto_aif_scaled_float_ideal)
plt.figure()
plt.plot(auto_aif_scaled_float_ideal,color='black')
#plt.plot(auto_aif_scaled_float_ideal, linestyle='--', label='Auto AIF\nQuality Score: {:.1f}'.format(auto_aifitness_500256),color='black')
plt.axhline(y=ideal_mean, linestyle='--', color='gray', alpha=0.5)
plt.xlabel('Time Points')
plt.ylabel('AIF Signal (au)')
plt.title('Example AIF SI Curves')
plt.savefig(os.path.join(output_dir, 'manual_auto_aif_ideal_example.png'), dpi=300)


# Plot all manual and auto AIF float mmolar values
count = 0
max_length = max(len(value['manual_aif_float_mmolar']) for value in aif_values.values() if 'manual_aif_float_mmolar' in value)
num_aifs = sum(1 for value in aif_values.values() if 'manual_aif_float_mmolar' in value)
plot_manual_mean_mmolar = np.full((max_length, num_aifs), np.nan)
plot_auto_mean_mmolar = np.full((max_length, num_aifs), np.nan)
for key, value in aif_values.items():
    # shift the AIF values to align the peaks
    max_index_manual = np.argmax(value['manual_aif_float_mmolar'])
    max_index_auto = np.argmax(value['auto_aif_float_mmolar'])
    shift_manual = 5 - max_index_manual
    shift_auto = 5 - max_index_auto
    shifted_manual_aif_float_mmolar = np.full_like(value['manual_aif_float_mmolar'], np.nan)
    shifted_auto_aif_float_mmolar = np.full_like(value['auto_aif_float_mmolar'], np.nan)
    if shift_manual > 0:
        shifted_manual_aif_float_mmolar[shift_manual:] = value['manual_aif_float_mmolar'][:-shift_manual]
    else:
        shifted_manual_aif_float_mmolar[:shift_manual] = value['manual_aif_float_mmolar'][-shift_manual:]
    if shift_auto > 0:
        shifted_auto_aif_float_mmolar[shift_auto:] = value['auto_aif_float_mmolar'][:-shift_auto]
    else:
        shifted_auto_aif_float_mmolar[:shift_auto] = value['auto_aif_float_mmolar'][-shift_auto:]
    padded_manual = np.pad(shifted_manual_aif_float_mmolar, 
                           (0, max_length - len(shifted_manual_aif_float_mmolar)), 
                           constant_values=np.nan)
    padded_auto = np.pad(shifted_auto_aif_float_mmolar, 
                         (0, max_length - len(shifted_auto_aif_float_mmolar)), 
                         constant_values=np.nan)
    #concat values to a 2d array to average later
    plot_manual_mean_mmolar[:, count] = padded_manual
    plot_auto_mean_mmolar[:, count] = padded_auto
    count += 1
#average over second dimension ignoring nans
mean_manual_mmolar = np.nanmean(plot_manual_mean_mmolar, axis=1)
mean_auto_mmolar = np.nanmean(plot_auto_mean_mmolar, axis=1)
#get standard error of the mean
sem_manual_mmolar = np.nanstd(plot_manual_mean_mmolar, axis=1) / np.sqrt(np.sum(~np.isnan(plot_manual_mean_mmolar), axis=1))
sem_auto_mmolar = np.nanstd(plot_auto_mean_mmolar, axis=1) / np.sqrt(np.sum(~np.isnan(plot_auto_mean_mmolar), axis=1))
# Perform t-test for every point along axis=0 comparing manual and auto
t_values = np.full(max_length, np.nan)
p_values = np.full(max_length, np.nan)
for i in range(max_length):
    manual_values = []
    auto_values = []
    for j in range(num_aifs):
        if ~np.isnan(plot_manual_mean_mmolar[i, j]) and ~np.isnan(plot_auto_mean_mmolar[i, j]):
            manual_values.append(plot_manual_mean_mmolar[i, j])
            auto_values.append(plot_auto_mean_mmolar[i, j])
    if len(manual_values) > 1 and len(auto_values) > 1:
        t_values[i], p_values[i] = ttest_rel(manual_values, auto_values)
        # print warning if any p-values are less than 0.05
        if p_values[i] < 0.05:
            print(f"Significant difference at time point {i} with p-value {p_values[i]}")

# Plot p-values
if np.any(p_values < 0.05):
    #print("Some p-values are less than 0.05, indicating significant differences between manual and auto AIF values.")    
    plt.figure()
    plt.plot(p_values, label='p-values', color='black')
    plt.axhline(y=0.05, color='red', linestyle='--', label='Significance threshold (0.05)')
    plt.xlabel('Time Points')
    plt.ylabel('p-value')
    plt.title('p-values for t-test between Manual and Auto AIF values')
    plt.legend(loc='upper right')
    #plt.savefig(os.path.join(output_dir, 'p_values_ttest_manual_auto_aif.png'), dpi=300)
# Plot AIF curves
plt.figure()
plt.plot(mean_manual_mmolar, label='Manual',color='black')
plt.plot(mean_auto_mmolar, linestyle='--', label='Auto',color='black')
plt.fill_between(range(len(mean_manual_mmolar)), mean_manual_mmolar - sem_manual_mmolar, 
                 mean_manual_mmolar + sem_manual_mmolar, color='red', alpha=0.5)
plt.fill_between(range(len(mean_auto_mmolar)), mean_auto_mmolar - sem_auto_mmolar, 
                 mean_auto_mmolar + sem_auto_mmolar, color='blue', alpha=0.5)
plt.xlabel('Time Points')
plt.ylabel('AIF (mM)')
plt.title('AIF Mean and Standard Error of Mean for Test Cohort')
plt.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, 'manual_auto_aif_float_mmolar_all_subjects.png'), dpi=300)

# Plot mean of mMolar values for each participant and correlation (auto vs manual)
plt.figure()
plt.scatter(manual_mean_mmolar_list, auto_mean_mmolar_list)
plt.xlabel('Manual Mean')
plt.ylabel('Auto Mean')
plt.title('Mean AIF Values')
max_axis = 10
plt.xlim(0, max_axis)
plt.ylim(0,max_axis)
# add a line of best fit
p = Polynomial.fit(manual_mean_mmolar_list, auto_mean_mmolar_list, 1)
x_vals = np.linspace(0, max_axis, 100)
plt.plot(x_vals, p(x_vals), color='gray')
# show the r^2 value on the plot, limit to 4 decimal places
correlation_matrix = np.corrcoef(manual_mean_mmolar_list, auto_mean_mmolar_list)
r_squared = round(correlation_matrix[0,1]**2,4)
plt.text(0.1*max_axis, 0.9*max_axis, f"$r^2$ = {r_squared}")
plt.savefig(os.path.join(output_dir, 'aif_mean_si_comparison.png'), dpi=300)
# print warning if any values are above the max_axis
if any(np.array(manual_mean_mmolar_list)>max_axis) or any(np.array(auto_mean_mmolar_list)>max_axis):
    print(f"Warning: AIF value not displayed on plot value above {max_axis}")

# Plot Ktrans values
plt.figure()
plt.scatter(1000 * np.array(manual_ktrans_list), 1000 * np.array(auto_ktrans_list))
plt.xlabel('Manual Ktrans')
plt.ylabel('Auto Ktrans')
plt.title('Ktrans Values')
max_axis = 6
plt.xlim(0, max_axis)
plt.ylim(0,max_axis)
# add a line of best fit
p = Polynomial.fit(manual_ktrans_list, auto_ktrans_list, 1)
x_vals = np.linspace(0, max_axis, 100)
plt.plot(x_vals, p(x_vals), color='gray')
# show the r^2 value on the plot, limit to 4 decimal places
correlation_matrix = np.corrcoef(manual_ktrans_list, auto_ktrans_list)
r = round(correlation_matrix[0,1],4)
plt.text(0.1*max_axis, 0.9*max_axis, f"$r$ = {r}")
plt.savefig(os.path.join(output_dir, 'ktrans_all_comparison.png'), dpi=300)
# print warning if any values are above the max_axis
if any(np.array(manual_ktrans_list)*1000>max_axis) or any(np.array(auto_ktrans_list)*1000>max_axis):
    print(f"Warning: Ktrans value not displayed on plot value above {max_axis}")

# Plot GM Ktrans values
# plt.figure()
# plt.scatter(manual_ktrans_GM_list, auto_ktrans_GM_list)
# plt.xlabel('Manual GM Ktrans')
# plt.ylabel('Auto GM Ktrans')
# plt.title('GM Ktrans Values')
# plt.xlim(0, max_axis)
# plt.ylim(0,max_axis)
# # add a line of best fit
# p = Polynomial.fit(manual_ktrans_GM_list, auto_ktrans_GM_list, 1)
# x_vals = np.linspace(0, max_axis, 100)
# plt.plot(x_vals, p(x_vals), color='gray')
# # show the r^2 value on the plot, limit to 4 decimal places
# correlation_matrix = np.corrcoef(manual_ktrans_GM_list, auto_ktrans_GM_list)
# r_squared = round(correlation_matrix[0,1]**2,4)
# plt.text(0.1*max_axis, 0.9*max_axis, f"$r^2$ = {r_squared}")
# plt.savefig(os.path.join(output_dir, 'ktrans_gm_comparison.png'), dpi=300)
# # print warning if any values are above the max_axis
# if any(np.array(manual_ktrans_GM_list)>max_axis) or any(np.array(auto_ktrans_GM_list)>max_axis):
#     print(f"Warning: GM Ktrans value not displayed on plot value above {max_axis}")

# Plot WM Ktrans values
# plt.figure()
# plt.scatter(manual_ktrans_WM_list, auto_ktrans_WM_list)
# plt.xlabel('Manual WM Ktrans')
# plt.ylabel('Auto WM Ktrans')
# plt.title('WM Ktrans Values')
# plt.xlim(0, max_axis)
# plt.ylim(0,max_axis)
# # add a line of best fit
# p = Polynomial.fit(manual_ktrans_WM_list, auto_ktrans_WM_list, 1)
# x_vals = np.linspace(0, max_axis, 100)
# plt.plot(x_vals, p(x_vals), color='gray')
# # show the r^2 value on the plot, limit to 4 decimal places
# correlation_matrix = np.corrcoef(manual_ktrans_WM_list, auto_ktrans_WM_list)
# r_squared = round(correlation_matrix[0,1]**2,4)
# plt.text(0.1*max_axis, 0.9*max_axis, f"$r^2$ = {r_squared}")
# # save the plots
# plt.savefig(os.path.join(output_dir, 'ktrans_wm_comparison.png'), dpi=300)
# # print warning if any values are above the max_axis
# if any(np.array(manual_ktrans_WM_list)>max_axis) or any(np.array(auto_ktrans_WM_list)>max_axis):
#     print(f"Warning: WM Ktrans value not displayed on plot value above {max_axis}")

# Plot WM, GM, cerebellum Ktrans values
plt.figure()
plt.scatter(1000 * np.array(manual_ktrans_GM_list), 1000 * np.array(auto_ktrans_GM_list), label='GM', marker='x', color='black')
plt.scatter(1000 * np.array(manual_ktrans_WM_list), 1000 * np.array(auto_ktrans_WM_list), label='WM', marker='o', edgecolors='gray', facecolors='none')
plt.scatter(1000 * np.array(manual_ktrans_cerb_list), 1000 * np.array(auto_ktrans_cerb_list), label='Cerebellum', marker='s', edgecolors='darkgray', facecolors='none')
#plt.scatter(1000 * np.array(manual_ktrans_muscle_list), 1000 * np.array(auto_ktrans_muscle_list), label='Muscle', marker='^', edgecolors='green', facecolors='none')
plt.xlabel('Manual AIF Ktrans (/min * $10^{-3}$)')
plt.ylabel('Auto AIF Ktrans (/min * $10^{-3}$)')
plt.title('AIF Comparison')
plt.xlim(0, max_axis)
plt.ylim(0, max_axis)
plt.legend()
# add a line of best fit
manual_all = manual_ktrans_GM_list + manual_ktrans_WM_list + manual_ktrans_cerb_list# + manual_ktrans_muscle_list
auto_all = auto_ktrans_GM_list + auto_ktrans_WM_list + auto_ktrans_cerb_list# + auto_ktrans_muscle_list
p = Polynomial.fit(manual_all, auto_all,1)
x_vals = np.linspace(0, max_axis, 100)
plt.plot(x_vals, p(x_vals), color='gray')
# show the r^2 value on the plot, limit to 4 decimal places
correlation_matrix = np.corrcoef(manual_all, auto_all)
r_squared = round(correlation_matrix[0, 1]**2, 4)
r = round(correlation_matrix[0, 1], 4)
plt.text(0.1 * max_axis, 0.9 * max_axis, f"$r$ = {r}")
# save the plots
plt.savefig(os.path.join(output_dir, 'ktrans_roi_comparison.png'), dpi=300)
# print warning if any values are above the max_axis
if any(np.array(manual_all)*1000 > max_axis) or any(np.array(auto_all)*1000 > max_axis):
    print(f"Warning: GM, WM, Cerebellum Ktrans value not displayed on plot value above {max_axis}")

# Create Bland-Altman plot for manual_all and auto_all

# Convert lists to numpy arrays if they are not already
manual_array = np.array(manual_all)
auto_array = np.array(auto_all)

# Calculate mean and difference
mean_values = (manual_array + auto_array) / 2
diff_values = manual_array - auto_array

# Calculate mean difference and limits of agreement
mean_diff = np.mean(diff_values)
std_diff = np.std(diff_values)
loa_upper = mean_diff + 1.96 * std_diff
loa_lower = mean_diff - 1.96 * std_diff

# Plotting Bland-Altman plot
plt.figure()
plt.scatter(mean_values * 1000, diff_values * 1000, color='black', alpha=0.5)
plt.axhline(mean_diff * 1000, color='gray', linestyle='--', label='Mean difference')
plt.axhline(loa_upper * 1000, color='red', linestyle='--', label='Limits of agreement (±1.96 SD)')
plt.axhline(loa_lower * 1000, color='red', linestyle='--')
plt.xlabel('Mean Ktrans (/min × $10^{-3}$)')
plt.ylabel('Difference in Ktrans (/min × $10^{-3}$)')
plt.title('Bland-Altman Plot of Ktrans Values')
#plt.legend()
plt.savefig(os.path.join(output_dir, 'bland_altman_ktrans.png'), dpi=300)

plt.show()

# calculate a paired ttest of aifitness values
t_statistic = ttest_rel(manual_aifitness_list, auto_aifitness_list)
print(f"Paired t-test of AIFitness values: t-statistic = {t_statistic[0]:.3f}, p-value = {t_statistic[1]:.3f}")

# calculate a paired ttest of ktrans values
t_statistic = ttest_rel(manual_all, auto_all)
print(f"Paired t-test of Ktrans values: t-statistic = {t_statistic[0]:.3f}, p-value = {t_statistic[1]:.3f}")

# Calculate ICC of Ktrans values using pingouin
# Create dataframe for ICC calculation
icc_data = pd.DataFrame({
    'subjects': list(range(len(manual_all))) * 2,  # Subject IDs
    'raters': ['manual'] * len(manual_all) + ['auto'] * len(auto_all),  # Rater labels
    'scores': manual_all + auto_all  # Ktrans values
})

# Calculate ICC2 (two-way random effects, absolute agreement)
icc = pg.intraclass_corr(data=icc_data, targets='subjects', raters='raters', ratings='scores')

# Print ICC results
print("\nIntraclass Correlation Coefficient Results:")
print(f"ICC2: {icc.loc[icc['Type'] == 'ICC2', 'ICC'].values[0]:.3f}")
print(f"95% CI: [{icc.loc[icc['Type'] == 'ICC2', 'CI95%'].values[0][0]:.3f}, {icc.loc[icc['Type'] == 'ICC2', 'CI95%'].values[0][1]:.3f}]\n")

# get the median and standard deviation of the auto/manual ktrans values
manual_ktrans_array = np.array(manual_ktrans_list)
auto_ktrans_array = np.array(auto_ktrans_list)
manual_aifitness_array = np.array(manual_aifitness_list)
auto_aifitness_array = np.array(auto_aifitness_list)
manual_ktrans_GM_array = np.array(manual_ktrans_GM_list)
auto_ktrans_GM_array = np.array(auto_ktrans_GM_list)
manual_ktrans_WM_array = np.array(manual_ktrans_WM_list)
auto_ktrans_WM_array = np.array(auto_ktrans_WM_list)
manual_ktrans_cerb_array = np.array(manual_ktrans_cerb_list)
auto_ktrans_cerb_array = np.array(auto_ktrans_cerb_list)
manual_ktrans_muscle_array = np.array(manual_ktrans_muscle_list)
auto_ktrans_muscle_array = np.array(auto_ktrans_muscle_list)

# print the median and standard deviation of the ktrans values
print(f"Median and Standard Deviation of Ktrans values")
print(f"Manual Ktrans: Median = {np.median(manual_ktrans_array) * 1000:.2f}, Standard Dev = {np.std(manual_ktrans_array) * 1000:.2f}")
print(f"Auto Ktrans: Median = {np.median(auto_ktrans_array) * 1000:.2f}, Standard Dev = {np.std(auto_ktrans_array) * 1000:.2f}")
print(f"Manual AIFitness: Mean = {np.mean(manual_aifitness_array):.2f}, Standard Dev = {np.std(manual_aifitness_array):.2f}")
print(f"Auto AIFitness: Mean = {np.mean(auto_aifitness_array):.2f}, Standard Dev = {np.std(auto_aifitness_array):.2f}")
print(f"Manual AIFitness: 5th Percentile = {np.percentile(manual_aifitness_array, 5):.2f}")
print(f"Auto AIFitness: 5th Percentile = {np.percentile(auto_aifitness_array, 5):.2f}")
print(f"Manual GM Ktrans: Median = {np.median(manual_ktrans_GM_array) * 1000:.2f}, Standard Dev = {np.std(manual_ktrans_GM_array) * 1000:.2f}")
print(f"Auto GM Ktrans: Median = {np.median(auto_ktrans_GM_array) * 1000:.2f}, Standard Dev = {np.std(auto_ktrans_GM_array) * 1000:.2f}")
print(f"Manual WM Ktrans: Median = {np.median(manual_ktrans_WM_array) * 1000:.2f}, Standard Dev = {np.std(manual_ktrans_WM_array) * 1000:.2f}")
print(f"Auto WM Ktrans: Median = {np.median(auto_ktrans_WM_array) * 1000:.2f}, Standard Dev = {np.std(auto_ktrans_WM_array) * 1000:.2f}")
print(f"Manual Cerebellum Ktrans: Median = {np.median(manual_ktrans_cerb_array) * 1000:.2f}, Standard Dev = {np.std(manual_ktrans_cerb_array) * 1000:.2f}")
print(f"Auto Cerebellum Ktrans: Median = {np.median(auto_ktrans_cerb_array) * 1000:.2f}, Standard Dev = {np.std(auto_ktrans_cerb_array) * 1000:.2f}")
print(f"Manual Muscle Ktrans: Median = {np.median(manual_ktrans_muscle_array) * 1000:.2f}, Standard Dev = {np.std(manual_ktrans_muscle_array) * 1000:.2f}")
print(f"Auto Muscle Ktrans: Median = {np.median(auto_ktrans_muscle_array) * 1000:.2f}, Standard Dev = {np.std(auto_ktrans_muscle_array) * 1000:.2f}")
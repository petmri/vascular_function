import os
import numpy as np
import nibabel as nib
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import zip_longest
import re

auto_aif_dir = '/media/network_mriphysics/USC-PPG/bids_test/derivatives/dceprep-autoAIF_huber_real'
manual_aif_dir = '/media/network_mriphysics/USC-PPG/bids_test/derivatives/dceprep-manualAIF_refresh_testset'
replication_auto_aif_dir = '/home/mrispec/Desktop/network_mriphysics/USC-PPG/bids_test/derivatives/dceprep-autoAIF_huber1_naive'
id_list_dir = '/media/network_mriphysics/USC-PPG/AI_training/weights/rg_latest/test_set.txt'
output_dir = '/media/network_mriphysics/USC-PPG/analysis/autoAIF_paper/aif_comparison/boxplot'
overwrite_cache = True


def find_files(input_directory):
    files = []
    for root, dirs, filenames in os.walk(input_directory):
        for filename in filenames:
            if filename.endswith('seg-WM_Ktrans.nii.gz') or filename.endswith('seg-GM_Ktrans.nii.gz'):
                files.append(os.path.join(root, filename))
    return files

def get_mean_ktrans(files, ktrans_zero_threshold):
    ktrans_list = []
    for file in files:
        img = nib.load(file)
        data = img.get_fdata()

        # Get mean value of manual_ktrans excluding zeros
        valid_ktrans = data[(data != 0)]
        valid_ktrans = valid_ktrans[valid_ktrans > ktrans_zero_threshold]
        if valid_ktrans.size == 0:
            ktrans_mean = 0
            print(f"Ktrans for {file} is all zeros")
        else:
            ktrans_mean = np.median(valid_ktrans)*1000
        ktrans_list.append(ktrans_mean)
        
    return ktrans_list

def find_files_testset(input_directory):
    files = []

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
            continue
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

        # Get filenames
        gm_file = os.path.join(input_directory, subject_id, session_id,'dce',subject_id+'_'+session_id+'_seg-GM_Ktrans.nii.gz')
        wm_file = os.path.join(input_directory, subject_id, session_id,'dce',subject_id+'_'+session_id+'_seg-WM_Ktrans.nii.gz')

        # check if the files exist
        if not os.path.exists(gm_file) or not os.path.exists(wm_file):
            print(f"Files not found for {gm_file} or {wm_file}")
        else:
            files.append(gm_file)
            files.append(wm_file)
    return files



cache_file = os.path.join(output_dir, 'ktrans_data.csv')
if os.path.exists(cache_file) and not overwrite_cache:
    print(f"Cache file {cache_file} exists. Reading data from cache.")
    df = pd.read_csv(cache_file)
    ktrans_auto_aif = df['Auto AIF'].dropna().tolist()
    ktrans_manual_aif = df['Manual AIF'].dropna().tolist()
    ktrans_replication_auto_aif = df['Replication Auto AIF'].dropna().tolist()
else:
    print(f"Cache file {cache_file} does not exist. Processing data.")
    files = find_files_testset(auto_aif_dir)
    print(f"Number of files found in auto_aif_dir: {len(files)}")
    ktrans_auto_aif = get_mean_ktrans(files, 10e-6)

    files = find_files_testset(manual_aif_dir)
    print(f"Number of files found in manual_aif_dir: {len(files)}")
    ktrans_manual_aif = get_mean_ktrans(files, 10e-6)

    files = find_files(replication_auto_aif_dir)
    print(f"Number of files found in replication_auto_aif_dir: {len(files)}")
    ktrans_replication_auto_aif = get_mean_ktrans(files, 10e-6)

data = [ktrans_auto_aif, ktrans_manual_aif, ktrans_replication_auto_aif]
labels = ['Auto AIF: \nTest', 'Manual AIF: \nTest', 'Replication']

if not os.path.exists(cache_file) or overwrite_cache:
    # Cache data to csv file
    with open(cache_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Auto AIF', 'Manual AIF', 'Replication Auto AIF'])
        writer.writerows(zip_longest(*data, fillvalue=''))




plt.figure(figsize=(4, 4))
sns.violinplot(data=data, inner="quart")#, showfliers=False)
sns.stripplot(data=data, color="0", alpha=1, size=2)
plt.xticks(ticks=[0, 1, 2], labels=labels)
plt.ylabel('Ktrans (/min * $10^{-3}$)')
plt.title('AIF Comparison')
plt.savefig(os.path.join(output_dir, 'boxplot_ktrans.png'), dpi=300)
plt.show()
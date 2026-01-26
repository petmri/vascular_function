This is Keras/TensorFlow implementation for the Vascular Function Extraction Model (VFEM), forked from https://github.com/wallaceloos/vascular_function. A pretrained model is also provided.

Article: [Automatic detection of arterial input function for brain DCE-MRI in multi-site cohorts](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70020)
### Requirements

 - TensorFlow 2.12+
 - Keras 2.12+
 - Python 3.9+
 - Numpy
 - Scipy
 - Pandas

### Installation
`conda` and `venv` are recommended for managing your Python environments.
The easiest way to set up your environment is to use `venv`. For example:
```bash
sudo apt install python3.10-venv
python3 -m venv tf
pip install -r requirements.txt
```

### Preparing the data

<p align="justify">The data starts saved in gzipped NIFTI format following the radiological orientation. The data we used has a variety of dimensions, ranging from 208x256x40x50 to 320x320x14x64. Using the following methods, your dimensions should not matter unless they are significantly different, in which case you may modify the resample dimensions. At runtime, the image and mask are first normalized using the min-max normalization. Next, the image and mask are resampled to fit into the dimensions 256 x 256 x 32 x 32. The data is then converted into a TFRecord for efficiency with the TensorFlow pipeline.
</div>

### Inference

To use the model you can load the weights provided [here](https://github.com/petmri/vascular_function/releases/download/v2.0.0/model_weight_huber1.h5) and run:

    python main_vif.py --mode inference --input_path /path/to/data/input_data.nii.gz \
    --model_weight_path /path/to/model_weight/weight.h5  \
    --save_output_path /path/to/folder/output/ \
    --save_image 1

<p align="justify">The model will predict a vascular function and a 3D mask. It will automatically resample and apply the mask to the target image's original dimension, as well as provide figures on the predicted ROI and resulting VF curve. The pretrained model is exclusively using temporal loss (Huber). While spatial loss was available, we found that using only temporal loss gave the best results.

### Training
In order to train the model, please organize your input dataset by site. For example:
```
dataset
├── site1
│   ├──images
│        └── id_x.nii.gz
│   ├── masks
│        └── id_x.nii.gz
├── site2
│   ├──images
│        └── id_x.nii.gz
│   ├── masks
│        └── id_x.nii.gz
├── site3
│   ├──images
│        └── id_x.nii.gz
│   ├── masks
│        └── id_x.nii.gz
```
The data will be split 80:10:10 for each site by default. NIFTIs MUST BE 32-BIT.
The splits generated are recorded as text files in the checkpoint path. The model will read the splits files and use them if they are already there. TFRecords will not be written if they already exist.
The seed is currently fixed for reproducibility through the initial lines of code in all files.
To train the model you can run:

    python main_vif.py --mode training --dataset_path /path/to/dataset/ \
    --save_checkpoint_path  /path/to/save/save_weight/

To see what other options there are, simply run
```python
python main_vif.py -h
```

### Evaluating Model (DEPRECATED)

To evaluate the model you can run:  
 
    python main_vif.py --mode eval --input_folder /path/to/data/folder/ \
    --model_weight_path  /path/to/model/weight.h5 --save_output_path /path/to/folder/to/save/results/

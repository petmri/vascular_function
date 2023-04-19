import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# graph AIF image curves all on one plot
def plotAIFCurves(path):
    # get subject names from path
    files = os.listdir(path + '/images/')
    print(files)

    # loop through all files
    for file in files:
        # if file.startswith("500"):
        print(file)
        # load files
        dce_img = nib.load(path + '/images/' + file)
        mask_img = nib.load(path + '/masks/' + file)

        # get data from file
        mask = mask_img.get_fdata()
        dce = dce_img.get_fdata()

        # get curve from masked dce
        mask = mask.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 1)
        roi_ = dce * mask
        num = np.sum(roi_, axis = (0, 1, 2), keepdims=False)
        den = np.sum(mask, axis = (0, 1, 2), keepdims=False)

        # normalize to baseline
        intensities = num/(den+1e-8)
        intensities = np.asarray(intensities)
        intensities = intensities/intensities[0]
        if intensities[0] != 1:
            print("error")
        # if intensities[1] < 3 and intensities[2] < 3:
        #     print(file + " has a weak AIF curve with " + str(intensities[1]) + " and " + str(intensities[2]))
        # if intensities[2] > intensities[1] or intensities[3] > intensities[2]+.5:
        #     print(file + " has a delayed injection with " + str(intensities[1]) + " and " + str(intensities[2]) + " and " + str(intensities[3]))
        if any(intensities[10:30] < 2):
            print(file + " has an intensity < 2")

        plt.plot(intensities)
    
    # plt.show()

    # save plot
    plt.savefig(path + '/AIF_curves.png')

if __name__== "__main__":
    # path to set (test/train/val)
    path = sys.argv[1]
    print(path)
    # plot AIF curves
    plotAIFCurves(path)
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, measure, exposure, color
from scipy.ndimage import label
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
import initializeCyclops as init

__version__ = '0.1'
__date__ = '04-07-2024'
__authors__ = 'Richard Allen White III & Jose Luis Figueroa III'


def process_images(XG0, XGsnames, outDir, fsep, nLR_iter, px2nm, CORR, szMetric, SM_constraint, train_split):
    print("\nStep 4 begin:")

    if not os.path.exists(outDir + fsep + "Step-4_genMasks"):
        os.mkdir(outDir + fsep + "Step-4_genMasks")

    countCount = 1
    avgVec, intVec, szVec, mVec, bVec, objVec, omit, masks, XG_train0, XG_test0 = [], [], [], [], [], [], [], [], [], []
    
    for ii, XG in enumerate(XG0, 1):
        shortname = XGsnames[ii - 1]
        currOut = outDir + fsep + "Step-4_genMasks" + fsep + "genMask" + str(countCount) + "_" + shortname
        if not os.path.exists(currOut):
            os.mkdir(currOut)

        plt.imshow(XG, cmap='gray')
        plt.title("XG-" + shortname)
        plt.savefig(currOut + fsep + "XG-" + shortname + ".png", dpi=150)
        plt.close('all')

        print("Evaluating image " + str(countCount) + "/" + str(len(XG0)) + ": " + shortname)
        
        XG = restoration.richardson_lucy(XG, PSF, nLR_iter)
        XG_test0.append(XG)

        plt.figure("RL-deconvblind")
        plt.imshow(XG, cmap='gray')
        plt.title("XG-" + shortname + ": deconvoluted")
        plt.savefig(currOut + fsep + "XG-" + shortname + "_Deconvoluted.png", dpi=150)
        plt.close('all')

        threshold = threshold_otsu(XG)
        otsuMask = XG > threshold
        masks.append(otsuMask)

        labels, numObj = label(otsuMask), 0
        rprops = measure.regionprops(labels, intensity_image=XG)

        for prop in rprops:
            if szMetric == 1:
                currMajor = prop.axis_major_length
                currMajorSc = currMajor * px2nm * CORR
                majorVec.append(currMajor)
                eqD = prop.equivalent_diameter_area
                scEqD = eqD * px2nm * CORR
                dVec.append(scEqD)
            if szMetric == 2:
                currMajor = prop.axis_major_length
                currMajorSc = currMajor * px2nm * CORR
                majorVec.append(currMajor)
                currMinor = prop.axis_minor_length
                currDia = (currMajor + currMinor) / 2
                currDia = currDia * px2nm * CORR
                dVec.append(currDia)
            # More processing...

        XG_train0.append(otsuMask)

        countCount += 1

    XG_mu = np.mean(avgVec)
    print("Mean size over " + str(len(XGsnames)) + " images: " + str(round(XG_mu, 2)) + " nm")

    nSz1, nSz2, nSz3, nSz4, nSz5 = 0, 0, 0, 0, 0
    for jj in range(len(szVec)):
        currSizes = np.array(szVec[jj])
        for kk in range(len(currSizes)):
            # Size counting logic...

    print("Total # of objects identified: " + str(np.sum(objVec)))
    print("Size breakdown:")
    # Print size breakdown logic...

    nTrain = int(np.floor(train_split * len(XG0)))
    nTest = len(XG0) - nTrain
    XG_train, XG_test = XG_train0[:nTrain], XG_test0[nTrain:]

    print("Partitioning training/testing data...")

    # Log and end steps...


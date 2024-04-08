#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure
from skimage.filters import threshold_otsu

__version__ = '0.1'
__date__ = '04-07-2024'
__authors__ = 'Richard Allen White III & Jose Luis Figueroa III'

def getVP(outDir, snameVP, XBVP, pad):
    fsep = os.sep
    if not os.path.exists(outDir + fsep + "Step-1_VP"):
        os.mkdir(outDir + fsep + "Step-1_VP")
    
    plt.figure("XB-" + snameVP)
    plt.imshow(XBVP, cmap='gray')
    plt.title("XB-" + snameVP)
    plt.savefig(f"{outDir}{fsep}Step-1_VP{fsep}XB-{snameVP}", dpi=150)
    
    print('o-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-o\n')
    print("Step 1 begin:\nScanning calibration image for VP candidates...")
    
    start = time.time()
    threshold = threshold_otsu(XBVP)
    otsuMask = XBVP > threshold
    
    plt.figure("XB-" + snameVP + ": Otsu mask")
    plt.imshow(otsuMask, cmap='gray')
    plt.title("XB-" + snameVP + ": Otsu mask")
    plt.savefig(f"{outDir}{fsep}Step-1_VP{fsep}XB-{snameVP} - Otsu mask", dpi=150)
    
    otsuMask_eroded = ndimage.binary_erosion(otsuMask, iterations=1)
    
    plt.figure("XB-" + snameVP + ": Otsu mask eroded")
    plt.imshow(otsuMask_eroded, cmap='gray')
    plt.title("XB-" + snameVP + ": Otsu mask eroded")
    plt.savefig(f"{outDir}{fsep}Step-1_VP{fsep}XB-{snameVP} - Otsu mask eroded", dpi=150)
    
    labels = measure.label(otsuMask_eroded)
    rprops = measure.regionprops(labels, intensity_image=XBVP)
    
    cVec = [prop.centroid for prop in rprops]
    bBox = [prop.bbox for prop in rprops]
    
    cVec = np.array(cVec)
    dConstraint = 30
    dIdx = []
    proxVec = []
    
    for i, currCenter in enumerate(cVec):
        dVec = np.transpose(cdist([currCenter], cVec))
        for j, dist in enumerate(dVec):
            if 0 < dist <= dConstraint:
                dIdx.append(j)
                proxVec.append(float(dist))
    
    proxVec = proxVec[1::2]
    dIdx = np.array(dIdx)
    numVPs = len(dIdx) // 2
    
    print(f"Number of VP candidates identified: {numVPs}")
    if numVPs == 0:
        sys.exit(f"Cyclops ERROR: zero VPs identified within {dConstraint * 34.2} nm. Increase distance constraint or select new calibration image.")
    else:
        print(f"VPs saved to: {outDir}{fsep}Step-1_VP{fsep}")
    
    bBox = np.array(bBox)
    VPidx = np.arange(0, len(dIdx), 2)
    VPcoords = [np.concatenate((bBox[dIdx[k]], bBox[dIdx[k+1]])) for k in VPidx]
    
    VPs = []
    VPcent = [np.concatenate((cVec[n], cVec[n+1])) for n in range(0, len(cVec), 2)]
    
    for VPtemp in VPcoords:
        xVPmax = int(np.max(VPtemp[:, 3])) + pad
        xVPmin = int(np.min(VPtemp[:, 1])) - pad
        yVPmax = int(np.max(VPtemp[:, 2])) + pad
        yVPmin = int(np.min(VPtemp[:, 0])) - pad
        VP = XBVP[yVPmin:yVPmax, xVPmin:xVPmax]
        threshold = threshold_otsu(VP)
        otsuMask = VP > threshold
        otsuMask_eroded = ndimage.binary_erosion(otsuMask, iterations=1)
        labels = measure.label(otsuMask_eroded)
        rprops = measure.regionprops(labels, intensity_image=VP)
        VPs.append(VP)
    
    dMin = np.argmin(proxVec)
    VP0 = VPs[dMin]
    
    x1, y1 = VPcent[0][0]
    x2, y2 = VPcent[0][1]
    m = (y2 - y1) / (x2 - x1)
    bSlope = y1 - (m * x1)
    theta0 = np.arctan(m / 1) * (180 / np.pi)
    
    if -5 < theta0 < 5:
        optBox = VP0
    else:
        optBox = ndimage.rotate(VP0, theta0 if theta0 > 5 else -theta0, mode='constant')
    
    plt.figure("VP_Final: rotated")
    plt.imshow(optBox, cmap='gray')
    plt.title("VP_Final: rotated")
    plt.savefig(f"{outDir}{fsep}Step-1_VP{fsep}XB-{snameVP}_VP_Final.png", dpi=150)
    plt.close('all')
    
    print(f"Closest proximity at VP {dMin+1}: {round(proxVec[dMin] * 34.2, 2)} nm")
    print(f"VP candidate {dMin+1} selected.")
    
    stop = time.time() - start
    print(f"\nStep 1 end: {round(stop, 4)} seconds")
    print('\no-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-o')

# Example usage:
out_directory = "output"
sample_name = "sample"
XBVP_data = np.random.rand(100, 100)
padding = 5
getVP(out_directory, sample_name, XBVP_data, padding)


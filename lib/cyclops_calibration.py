#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, restoration, exposure
from skimage.color import label2rgb

__version__ = '0.1'
__date__ = '04-07-2024'
__authors__ = 'Richard Allen White III & Jose Luis Figueroa III'

print("\nStep 3 begin:")

start = time.time()

if not os.path.exists(outDir + fsep + "Step-3_Corr"):
    os.mkdir(outDir + fsep + "Step-3_Corr")

corrPath = "Cyclops_Data" + fsep + "Calibration" + fsep + "avgCORR.txt"

corrExists = os.path.exists(corrPath)

corrCount = 1
avgVec, intVec, szVec, mVec, bVec, objVec = [], [], [], [], [], []

for ii, XB in enumerate(XB0):
    shortname = XBsnames[ii]
    currOut = outDir + fsep + "Step-3_Corr" + fsep + "CORR" + str(corrCount) + "_" + shortname

    if not os.path.exists(currOut):
        os.mkdir(currOut)

    dname = outDir + fsep + "Step-3_Corr" + fsep + "CORR" + str(corrCount) + "_" + shortname

    print(f"Evaluating image {corrCount}/{nCorr}: {shortname}")

    plt.imshow(XB, cmap='gray')
    plt.title(f"XB-{shortname}")
    plt.savefig(dname + fsep + f"XB-{shortname}.png", dpi=150)
    plt.close('all')

    XB = restoration.richardson_lucy(XB, PSF, nLR_iter)

    plt.figure("RL-deconvblind")
    plt.imshow(XB, cmap='gray'), plt.title(f"XB-{shortname}: deconvoluted")
    plt.savefig(dname + fsep + f"XB-{shortname}_Deconvoluted.png", dpi=150)
    plt.close('all')

    threshold = measure.threshold_otsu(XB)
    otsuMask = XB > threshold
    labels = measure.label(otsuMask)
    rprops = measure.regionprops(labels, intensity_image=XB)

    dVec, iVec, eVec, oIdx, majorVec = [], [], [], [], []
    oCount = 0

    for prop in rprops:
        currMajor = prop.axis_major_length
        currMajorSc = currMajor * px2nm
        majorVec.append(currMajor)
        
        if szMetric == 1:
            currDia = prop.equivalent_diameter_area * px2nm
        elif szMetric == 2:
            currDia = ((prop.axis_major_length + prop.axis_minor_length) / 2) * px2nm

        dVec.append(currDia)
        iVec.append(prop.intensity_mean)
        eVec.append(prop.eccentricity)

        if currMajor == 0 or prop.eccentricity > 0.9999 or currMajorSc > SM_constraint:
            oIdx.append(oCount)

        oCount += 1

    oIdx = list(set(oIdx))
    oIdx.sort(reverse=True)

    for i in oIdx:
        del dVec[i]
        del iVec[i]
        del eVec[i]

    currAvg = round(np.mean(dVec), 2)
    avgVec.append(currAvg)
    szVec.append(dVec)
    intVec.append(iVec)
    numObj = len(dVec)
    objVec.append(numObj)

    print(f"Number of objects: {numObj}")
    print(f"{shortname} mean size: {currAvg} nm")

    plt.imshow(otsuMask, cmap='gray')
    plt.title(f"XB-{shortname}: Otsu mask")
    plt.savefig(dname + fsep + f"XB-{shortname}_OtsuMask.png", dpi=150)

    labelsColored = label2rgb(labels, colors=["blue"], bg_label=0)
    labelsColored = exposure.adjust_gamma(labelsColored, gamma=0.1, gain=1)

    plt.imshow(labelsColored)
    plt.title(f"XB-{shortname}: Labels")
    plt.savefig(dname + fsep + f"XB-{shortname}_Labels.png", dpi=150)
    plt.close('all')

    tempInt = intVec[ii]
    intensity = np.array(tempInt) * 1000
    tempSize = szVec[ii]
    size = np.array(tempSize)
    m, b = np.polyfit(tempSize, tempInt, 1)
    mVec.append(m)
    bVec.append(b)
    x = np.linspace(np.min(size), np.max(size), np.size(size))
    y = m * x + b

    plt.plot(x, y, 'k--', markersize=14)
    plt.axvline(currAvg, color='b', linestyle=":")
    plt.text(currAvg + (currAvg * 0.01), np.min(tempInt), f"µ = {currAvg} nm", size=12, fontweight='bold', color='blue')
    plt.scatter(tempSize, tempInt, c=tempInt, cmap='hot', edgecolors='black')
    plt.ylabel("Mean intensity (arb. unit)")
    plt.xlabel("Object diameter (nm)")
    plt.title(f"XB-{corrCount} Object mean intensity vs. diameter")
    plt.savefig(dname + fsep + f"XB-{corrCount}_IvsD.png", dpi=150)
    plt.close('all')

    plt.scatter(tempSize, eVec, c=eVec, cmap='hot', edgecolors='black')
    plt.ylabel("Linear eccentricity")
    plt.xlabel("Object diameter (nm)")
    plt.title(f"XB-{corrCount} Linear eccentricty vs. diameter")
    plt.savefig(dname + fsep + f"XB-{corrCount}_EvsD.png", dpi=150)
    plt.close('all')

    corrCount += 1

plt.scatter(bVec, mVec, c=mVec, cmap='turbo', edgecolors='black')
plt.xlabel("Slope intercept (b)")
plt.ylabel("Slope (m)")
plt.title("XB-all: Fit curve y-intercept vs. fit curve slope")
plt.savefig(outDir + fsep + "Step-3_Corr" + fsep + "XB-all_b-VS-m.png", dpi=150)
plt.close('all')

XB_mu = np.mean(avgVec)
CORR = sphereSize / XB_mu

print(f"XB correction factor: {round(CORR, 4)}")
logCorr(XBsnames, objVec, avgVec, CORR, XB_mu, nCorr, outDir, fsep)

stop = time.time() - start

print(f"Step 3 end: {round(stop, 4)} seconds")
print('o-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-o')

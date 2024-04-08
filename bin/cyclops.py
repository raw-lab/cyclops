#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.distance import cdist
from scipy import ndimage
from scipy.ndimage import label
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
from skimage import measure, io, restoration, img_as_float, exposure
from skimage.color import label2rgb
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage import img_as_ubyte
from scipy.spatial.distance import cdist
from scipy import ndimage
from pypher.pypher import psf2otf, zero_pad
from stardist.models import StarDist2D
import getVP

__version__ = '0.1'
__date__ = '04-07-2024'
__authors__ = 'Richard Allen White III & Jose Luis Figueroa III'

parser = argparse.ArgumentParser(add_help=False)
parser.set_defaults()
required = parser.add_argument_group('Input file(s) required')
required.add_argument('-c', '--config', help='Path to config file, command line takes priority', is_config_file=True)
required.add_argument('-i', '--input', action='append', default=[], help='Path to input image or folder containing images')
required.add_argument('--dapi', action='append', default=[], help='Path to DAPI images')
required.add_argument('--fitc', action='append', default=[], help='Path to FITC')
required.add_argument('--calibration', action='append', default=[], help='Path to calibration image')

optional = parser.add_argument_group('optional arguments')
optional.add_argument('--genFigs', type=bool, default=True, help='Toggle the generation of figures on (True) or off (False). [True]')
optional.add_argument('--sphereSize', type=float, default=175, help="Diameter of microspheres in nanometers.")
optional.add_argument('--scaleLength', type=float, default=585, help='Length of scale bar for imaging equipment in pixels (px). [585]')
optional.add_argument('--scaleMetric', type=float, default=20e3, help="Represented length of scale bar in nm.")
optional.add_argument('--pad', type=int, default=12, help="Set padding around VP centroids to expand bounding box. [12]")
optional.add_argument('--dConstraint', type=int, default=30, help="Set user-defined px distance constraint for VP candidates. [30]")
optional.add_argument('--fSize', type=int, default=31, help="Set dimensions of hybrid point-spread function, fSize-by-fSize. [31]")
optional.add_argument('--psfMethod', type=str, default='gam', choices=['gam', 'hyb', 'gau'], help="Set formula for point-spread function creation. The available options are: 'gam' (gamma sinc fn); 'hyb' (hybrid sinc fn); and 'gau' (gaussian). [gam]")
optional.add_argument('--nMLE_iter', type=int, default=10, help="Set maximum likelihood estimation (MLE) number of iterations. [10]")
optional.add_argument('--a', type=int, default=1, help="Parameter to control amplitude for gaussian component of GL PSF. [1]")
optional.add_argument('--b', type=int, default=1, help="Parameter to control amplitude for sinc component of hybrid PSF. [1]")
optional.add_argument('--sig', type=int, default=1, help="Parameter to control std of gaussian component of hybrid PSF. [1]")
optional.add_argument('--r', type=float, default=2.718281828, help="Parameter to control width for sinc component of hybrid PSF. [2.718281828]")
optional.add_argument('--tau', type=float, default=0.5, help="Parameter to control periodicity for sinc component of GL PSF. [0.5]")
optional.add_argument('--v', type=float, default=2.25, help="Parameter to control GL-PSFi (initial PSF) vertical stretch. [2.25]")
optional.add_argument('--s', type=float, default=0.0, help="Parameter to control GL-PSFi (initial PSF) vertical shift. [0.0]")
optional.add_argument('--tauVec', type=list, default=np.arange(1/(100*math.pi), 11/(100*math.pi), 1/(100*math.pi)), help="Tau parameters [1/(100*math.pi), 11/(100*math.pi), 1/(100*math.pi)]")
optional.add_argument('--vVec', type=list, default=np.arange(1/math.pi, 1.1*math.pi, 1/math.pi), help="Vector with v parameters [1/math.pi, 1.1*math.pi, 1/math.pi]")
#>----------------------------------| Size Correction |----------------------------------<#
optional.add_argument('--nLR_iter', type=int, default=80, help="Set number of iterations for Lucy-Richardson algorithm. [80]")
optional.add_argument('--szMetric', type=int, default=1, help="""Set metric to determine identified object sizes. The available
options are:
   1) Equivalent diameter area: the diameter of a circle having
      the same area as the identified object area (default).
   2) The average of the identified objects' semi-major and
      semi-minor axes. [1]""")
optional.add_argument('--SM_constraint', type=int, default=8000, help="Set threshold of semi-major axis to determine false positives. [8000]")
optional.add_argument('--train_split', type=float, default=0.8, help="""Set the desired training-testing split. The default is 0.8 which
results in an 80% training data and 20% testing data split. This number is rounded if not divisible. [0.8]""")
optional.add_argument('--outDir', type=str, default="Cyclops_Output", help="Set Cyclops output directory name. [Cyclops_Output]")
optional.add_argument('--cpus', type=int, help="Number of CPUs to use per task. System will try to detect available CPUs if not specified [Auto Detect]")
optional.add_argument('--version', '-v', action='version',
                    version=f'Cyclops: \n version: {__version__} {__date__}',
                    help='show the version number and exit')
optional.add_argument("-h", "--help", action="help", help="show this help message and exit")

args = parser.parse_args()

ct = datetime.datetime.now()
cTime = ct.strftime("%m-%d-%Y %H:%M:%S")
cTime2 = ct.strftime("%m-%d-%Y_%H-%M-%S")

if os.path.exists(args.outDir):
    shutil.rmtree(args.outDir)
os.mkdir(args.outDir)

eps = np.finfo(float).eps
corrCell = []

px2nm = args.scaleMetric / args.scaleLength

[XB0, XG0, XBVP, snameVP, nCorr, XBsnames, XGsnames, nF] = init.loadImages(args.dapi, args.fitc, args.calibration)
getVP.getVP(args.outDir, snameVP, XBVP, args.pad)

# Create Cyclops output directory if it doesn't exist
if os.path.exists(outDir):
    shutil.rmtree(outDir)
os.mkdir(outDir)

eps = np.finfo(float).eps  # Machine epsilon
corrCell = []  # List for correction factors
px2nm = scaleMetric / scaleLength  # Nanometers-per-pixel ratio

def load_images(path_DAPI, path_FITC, path_CALIB):
    fsep = os.sep
    print("Initializing Cyclops parameters and dirs...")
    fpath1 = f"Cyclops_Data{fsep}XB"
    fpath2 = f"Cyclops_Data{fsep}XG"
    fnamesXB = os.listdir(fpath1)
    fnamesXG = os.listdir(fpath2)
    nCorr = len(fnamesXB)
    nF = len(fnamesXG)
    print("Loading images from XB and XG dirs...")
    XBfnames = [f"{fpath1}{fsep}{name}" for name in fnamesXB]
    XBsnames = [name[:-5].replace(" ", "").replace("blue beads ", "BB-").replace(" + ", "+") for name in fnamesXB]
    XB0 = [img_as_float(io.imread(fname)) for fname in XBfnames]
    XGfnames = [f"{fpath2}{fsep}{name}" for name in fnamesXG]
    XGsnames = [name[:-5].replace(" ", "").replace("blue beads ", "BB-").replace(" + ", "+") for name in fnamesXG]
    XG0 = [img_as_float(io.imread(fname)) for fname in XGfnames]
    fpath3 = f"Cyclops_Data{fsep}Calibration"
    fnameVP = os.listdir(fpath3)[0]
    caliPath = f"{fpath3}{fsep}{fnameVP}"
    currVP = io.imread(caliPath)
    XBVP = img_as_float(currVP)
    print(f"Number of DAPI images: {nCorr}")
    print(f"Number of FITC images: {nF}")
    print("Cyclops initialization complete.\n")
    return XB0, XG0, XBVP, currVP[:-5].replace(" ", "").replace("blue beads ", "BB-").replace(" + ", "+"), nCorr, XBsnames, XGsnames, nF

def plot_distribution(testBox, path, fname, fsep):
    sz = np.shape(testBox)
    lenTestBox = sz[1]
    x = np.arange(lenTestBox)
    VPdist = np.sum(testBox, axis=0)
    pksIdx = find_peaks(VPdist)[0]
    if len(pksIdx) == 2:
        distMin = np.argmin(VPdist[pksIdx[0]:pksIdx[1]]) + pksIdx[0]
    else:
        distMin = np.floor(lenTestBox / 2)
    B = round(VPdist[int(distMin)], 2)
    plt.plot(x, VPdist, 'k-', linewidth=1.5)
    plt.xlabel("Column"), plt.ylabel("Intensity column-wise sum  (arb. unit)")
    plt.title("Virgilian pair distribution")
    plt.fill_between(x, VPdist, alpha=0.85)
    plt.axvline(x=distMin, color='white', linestyle=':', linewidth=4)
    txt = f"B = {B}"
    plt.text(distMin + 1, np.max(VPdist) / 6, txt, color='black', fontsize='x-large')
    plt.savefig(f"{path}{fsep}{fname}")
    plt.close('all')
    return x, VPdist, B

def create_PSF(fSize, a, b, sigma, r, tau, v, s, psfMethod):
    x = np.linspace(-fSize, fSize, fSize)
    y = np.linspace(-fSize, fSize, fSize)
    x, y = np.meshgrid(x, y)
    PSFi = np.zeros((fSize, fSize))
    R = np.sqrt(np.square(x) + np.square(y))
    if psfMethod == "gam":
        for i in range(fSize):
            for j in range(fSize):
                PSFi[i, j] = a * np.exp(v / ((math.gamma(1 + tau * R[i, j])) * (math.gamma(1 - tau * R[i, j])))) + s
    elif psfMethod == "gau":
        PSFi = gaussian_filter(np.exp(-(np.square(x) + np.square(y)) / (2. * (sigma ** 2))), sigma)
    elif psfMethod == "snc":
        for i in range(fSize):
            for j in range(fSize):
                PSFi[i, j] = (np.divide(np.sin(tau * R[i, j]), tau * R[i, j]))
    return PSFi, x, y

def get_MLE(optBox, fSize, PSFi, nMLE_iter):
    L, dim = optBox.shape
    J = np.zeros((L, dim))
    P = np.zeros((fSize * fSize, 2))
    fw = np.fft.fft2(np.ones((L, dim)))
    nMLE_iter = 10  # Set desired number of MLE iterations (default 10)
    for k in range(nMLE_iter):
        J_k, P_k = update_MLE(J, P, k)
        OTF = psf2otf(P_k, (L, dim))
        plusNoise = np.real(np.fft.ifft2(OTF * np.fft.fft2(J_k)))
        plusNoise[plusNoise <= 0] = np.finfo(plusNoise.dtype).eps
        currEst = J[0] / (plusNoise + np.finfo(plusNoise.dtype).eps)
        Psi_k = np.fft.fft2(currEst)
        J[2] = J[1]
        OTF = psf2otf(P[1], (L, dim))
        scale = np.real(np.fft.ifft2(np.conj(OTF) * fw)) + np.sqrt(np.finfo(scale.dtype).eps)
        OTFtemp = np.fft.ifft2(np.conj(OTF) * Psi_k)
        J[1] = np.maximum((J_k * np.real(OTFtemp) / scale), 0)
        J_k_flat = np.ndarray.flatten(J_k, order='F')
        J_flat = np.ndarray.flatten(J[1], order='F')
        J4 = J[3]
        Jtemp = J_flat - J_k_flat
        J[3] = np.vstack((Jtemp, J4[:, 0])).T
        P[2] = P[1]
        Jfreq = np.fft.fft2(J[2])
        OTF_k = np.conj(Jfreq) * fw
        scale = otf2psf(OTF_k, fSize) + np.sqrt(np.finfo(scale.dtype).eps)
        Jfreq_k = otf2psf(np.conj(Jfreq) * Psi_k, fSize)
        P[1] = np.maximum((P_k * Jfreq_k) / scale, 0)
        P[1] = P[1] / np.sum(P[1])
        P_flat = np.ndarray.flatten(P[1], order='F')
        P_k_flat = np.ndarray.flatten(P_k, order='F')
        P4 = P[3]
        Ptemp = P_flat - P_k_flat
        P[3] = np.vstack((Ptemp, P4[:, 0])).T
        if k == (nMLE_iter - 1):
            PSFr = P[1]
            Xdec = J[1]
    return PSFr, Xdec

def update_MLE(J, P, k):
    if k != 0:
        alpha_k_J = (J[3][:, 0].T @ J[3][:, 1]) / (J[3][:, 1].T @ J[3][:, 1] + np.finfo(alpha_k_J.dtype).eps)
        alpha_k_J = np.maximum(np.minimum(alpha_k_J, 0), 0)
        J_k = np.maximum(J[1] + alpha_k_J * (J[1] - J[2]), 0)
        alpha_k_P = (P[3][:, 0].T @ P[3][:, 1]) / (P[3][:, 1].T @ P[3][:, 1] + np.finfo(alpha_k_P.dtype).eps)
        alpha_k_P = np.maximum(np.minimum(alpha_k_P, 0), 0)
        P_k = np.maximum(P[1] + alpha_k_P * (P[1] - P[2]), 0)
        P_k = P_k / np.sum(P_k)
    else:
        J_k = np.maximum(J[1], 0)
        P_k = np.maximum(P[1], 0)
        P_k = P_k / (np.sum(P_k) + np.finfo(P_k.dtype).eps)
    return J_k, P_k

def otf2psf(OTF, fSize):
    if np.sum(OTF) == 0:
        PSF = np.zeros_like(OTF)
    else:
        PSF = np.fft.ifft2(OTF)
        if np.max(abs(PSF.imag)) / np.max(abs(PSF)) <= np.finfo(PSF.dtype).eps:
            PSF = PSF.real
        PSF = np.roll(PSF, int(np.floor(fSize / 2)), axis=0)
        PSF = np.roll(PSF, int(np.floor(fSize / 2)), axis=1)
        PSF = PSF[0:fSize, 0:fSize]
    return PSF

def opt_metrics(Xdec, entVec, energyVec):
    Xdec_uint = img_as_ubyte(Xdec)
    S = shannon_entropy(Xdec_uint, base=2)
    entVec.append(S)
    GLCM = graycomatrix(Xdec_uint, [2], [0], symmetric=True, normed=True)
    energy = graycoprops(GLCM, "energy")[0, 0]
    energyVec.append(energy)
    return entVec, energyVec

def plot_metrics(entVec, energyVec, path, fname, fsep):
    plt.figure()
    plt.plot(entVec, 'b-', linewidth=2)
    plt.xlabel("Iteration"), plt.ylabel("Shannon Entropy (bits)")
    plt.title("Shannon Entropy convergence")
    plt.savefig(f"{path}{fsep}{fname}_Entropy")
    plt.close('all')

    plt.figure()
    plt.plot(energyVec, 'r-', linewidth=2)
    plt.xlabel("Iteration"), plt.ylabel("GLCM Energy")
    plt.title("GLCM Energy convergence")
    plt.savefig(f"{path}{fsep}{fname}_Energy")
    plt.close('all')

def plot_3D(XB, XG, Z, path, fname, fsep):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XB, XG, Z, cmap='viridis')
    ax.set_xlabel('XB')
    ax.set_ylabel('XG')
    ax.set_zlabel('PSF')
    ax.set_title('3D Plot of XB, XG, and PSF')
    plt.savefig(f"{path}{fsep}{fname}_3D_Plot")
    plt.close('all')

def run_simulation(input_params):
    PSFi, optBox, nMLE_iter, fSize, path, fname, fsep = input_params
    PSFr, Xdec = get_MLE(optBox, fSize, PSFi, nMLE_iter)
    entVec, energyVec = [], []
    for i in range(nMLE_iter):
        entVec, energyVec = opt_metrics(Xdec, entVec, energyVec)
    plot_metrics(entVec, energyVec, path, fname, fsep)
    XB, XG, Z = plot_3D_PSFi(PSFi, path, fname, fsep)
    plot_3D(XB, XG, Z, path, fname, fsep)
    return PSFr, Xdec

def plot_3D_PSFi(PSFi, path, fname, fsep):
    XB, XG = np.meshgrid(np.arange(0, PSFi.shape[0]), np.arange(0, PSFi.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XB, XG, PSFi, cmap='viridis')
    ax.set_xlabel('XB')
    ax.set_ylabel('XG')
    ax.set_zlabel('PSFi')
    ax.set_title('3D Plot of XB, XG, and PSFi')
    plt.savefig(f"{path}{fsep}{fname}_3D_Plot_PSFi")
    plt.close('all')
    return XB, XG, PSFi
    
def main():
    path_DAPI = args.dapi[0]
    path_FITC = args.fitc[0]
    path_CALIB = args.calibration[0]
    XB0, XG0, XBVP, snameVP, nCorr, XBsnames, XGsnames, nF = load_images(path_DAPI, path_FITC, path_CALIB)
    PSFi, x, y = create_PSF(args.fSize, args.a, args.b, args.sig, args.r, args.tau, args.v, args.s, args.psfMethod)
    optBox = np.zeros((args.fSize, args.fSize))
    PSFr, Xdec = run_simulation((PSFi, optBox, args.nMLE_iter, args.fSize, args.outDir, f"Cyclops{fsep}{cTime2}", fsep))
    print(" ")
    print("Cyclops complete.")

if __name__ == "__main__":
    main()

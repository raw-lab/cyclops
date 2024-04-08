#!/usr/bin/env/ python
# -*- coding: utf-8 -*-

__version__ = '0.1'
__date__ = '04-07-2024'
__authors__ = 'Richard Allen White III & Jose Luis Figueroa III'

import os
import numpy as np
import time
import matplotlib.pyplot as plt

out_dir = outDir + os.sep + "Step-2_CyDecon"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print("\nStep 2 begin:")
print("Loading gamma sinc PSF...")

start_time = time.time()
L, dim = optBox.shape
f_max = min(L, dim)

if f_max % 2 == 0:
    f_max -= 1

filters = np.arange(3, f_max, 2)
ent_vec = []
energy_vec = []
count = 0
psf_params = []
b_vec = []
psf_vec = []
opt_count = 1
curr_out_dir = os.path.join(out_dir, "optimization")

if not os.path.exists(curr_out_dir):
    os.mkdir(curr_out_dir)

opt_start = time.time()

for f_size in filters:
    tau = round(tau, 10)
    v = round(v, 10)
    print(f"fSize: {f_size}; tau: {tau}; v: {v}")
    PSFi, X, Y = createPSFi(f_size, a, b, sig, r, tau, v, s, psfMethod)
    PSFr, Xdec = getMLE(optBox, f_size, PSFi, nMLE_iter)
    ent_vec, energy_vec = optMetrics(Xdec, ent_vec, energy_vec)
    count += 1
    psf_params.append([f_size, tau, v, s])

    for tau in tauVec:
        tau = round(tau, 10)
        v = round(v, 10)
        print(f"fSize: {f_size}; tau: {tau}; v: {v}")
        PSFi, X, Y = createPSFi(f_size, a, b, sig, r, tau, v, s, psfMethod)
        PSFr, Xdec = getMLE(optBox, f_size, PSFi, nMLE_iter)
        ent_vec, energy_vec = optMetrics(Xdec, ent_vec, energy_vec)
        count += 1
        psf_params.append([f_size, tau, v, s])

        for v in vVec:
            tau = round(tau, 10)
            v = round(v, 10)
            print(f"fSize: {f_size}; tau: {tau}; v: {v}")
            PSFi, X, Y = createPSFi(f_size, a, b, sig, r, tau, v, s, psfMethod)
            PSFr, Xdec = getMLE(optBox, f_size, PSFi, nMLE_iter)
            ent_vec, energy_vec = optMetrics(Xdec, ent_vec, energy_vec)
            count += 1
            psf_params.append([f_size, tau, v, s])

opt_stop = time.time() - opt_start

print("|.............................................|")
print(f"Number of sweeps: {count}")
print(f"Sweep rate: {round(count / opt_stop, 4)} sweeps/second")
print("Parameter sweeps complete.")

minSidx = np.argmin(ent_vec)
minS = ent_vec[minSidx]
maxEidx = np.argmax(energy_vec)
maxE = energy_vec[maxEidx]

print("Optimal PSF selected:")
print(f"  Minimum entropy: {minS} bits")
print(f"  GLCM energy: {maxE[0, 0]}")

opt_params = psf_params[minSidx]
f_size = opt_params[0]
tau = opt_params[1]
v = opt_params[2]

print("Final PSF parameters:")
print(f"  Filter size: {f_size}-by-{f_size}")
print(f"  tau: {tau}")
print(f"  v: {v}")

PSF, _, _ = createPSFi(f_size, a, b, sig, r, tau, v, s, psfMethod)
PSF, _ = getMLE(optBox, f_size, PSF, nMLE_iter)

print("Final PSF constructed.")

plt.scatter(np.arange(0, count), ent_vec, c=ent_vec, cmap='hot', edgecolors='black')
plt.ylabel("Shannon entropy (bits)")
plt.xlabel("Iteration")
plt.autoscale()
plt.xlim([0, count + 1])
plt.title("Shannon entropy vs. iteration")
plt.savefig(os.path.join(curr_out_dir, "EntVsIter.png"), dpi=150)
plt.close('all')

plt.scatter(np.arange(0, count), energy_vec, c=energy_vec, cmap='hot', edgecolors='black')
plt.ylabel("Energy")
plt.xlabel("Iteration")
plt.autoscale()
plt.xlim([0, count + 1])
plt.title("GLCM energy vs. iteration")
plt.savefig(os.path.join(curr_out_dir, "EnergyVsIter.png"), dpi=150)
plt.close('all')

getTextures(Xdec, curr_out_dir)
genPSFOTF(PSF, curr_out_dir)
getDist(optBox, curr_out_dir, "B-measure_initial.png", os.sep)
getDist(Xdec, curr_out_dir, "B-measure_final.png", os.sep)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(optBox, cmap="gray")
axes[0].set_xlabel("X (px)")
axes[0].set_ylabel("Y (px)")
axes[0].set_title("Initial optimization box")
axes[1].imshow(Xdec, cmap="gray")
axes[1].set_xlabel("X (px)")
axes[1].set_ylabel("Y (px)")
axes[1].set_title("Final optimization box")
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.01, top=0.99)
plt.savefig(outDir + os.sep + "Step-2_CyDecon" + os.sep + "InitialVsFinal_optBox.png", dpi=300)
plt.close('all')

logDecon(outDir, os.sep, count, opt_stop, minS, maxE, f_size, tau, v)
stop = time.time() - start_time
print(f"Step 2 end: {round(stop, 4)} seconds")
print('o-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-o')


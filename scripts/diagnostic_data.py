# diagnostic_data.py
import numpy as np
import torch
import pickle as pkl
import os

# Change from: './TransfomerDownscaling_MP/DownScale_Paper/param_stat_12_36/'
# To:
stat_dir = '../DownScale_Paper/param_stat_12_36/'

with open(os.path.join(stat_dir, 'ERA5_precip_stat.pkl'), 'rb') as f:
    era5_stat = pkl.load(f)

with open(os.path.join(stat_dir, 'CHIRPS_precip_obs_stat.pkl'), 'rb') as f:
    chirps_stat = pkl.load(f)

print("ERA5 Stats:")
print(f"  mean: {era5_stat.get('mean', 'N/A')}")
print(f"  var: {era5_stat.get('var', 'N/A')}")
print(f"  min: {era5_stat.get('min', 'N/A')}")
print(f"  max: {era5_stat.get('max', 'N/A')}")

print("\nCHIRPS Stats:")
print(f"  mean: {chirps_stat.get('mean', 'N/A')}")
print(f"  var: {chirps_stat.get('var', 'N/A')}")
print(f"  min: {chirps_stat.get('min', 'N/A')}")
print(f"  max: {chirps_stat.get('max', 'N/A')}")

# Check actual data - also fix these paths
lr = np.load('../DownScale_Paper/ERA5_precip_cut/20200601.npy')
hr = np.load('../DownScale_Paper/CHIRPS_precip_cut_obs/20200601.npy')

print("\nActual Data Stats:")
print(f"LR: min={lr.min():.6f}, max={lr.max():.6f}, mean={lr.mean():.6f}, std={lr.std():.6f}")
print(f"HR: min={hr.min():.6f}, max={hr.max():.6f}, mean={hr.mean():.6f}, std={hr.std():.6f}")
print(f"LR has NaN: {np.isnan(lr).any()}, has Inf: {np.isinf(lr).any()}")
print(f"HR has NaN: {np.isnan(hr).any()}, has Inf: {np.isinf(hr).any()}")

# Check if variance is near zero
if chirps_stat.get('var') is not None and chirps_stat['var'] < 1e-6:
    print("\n?? WARNING: CHIRPS variance is very small! This will cause NaN after normalization.")
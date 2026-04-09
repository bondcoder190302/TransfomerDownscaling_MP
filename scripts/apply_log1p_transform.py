import numpy as np
import glob
import os
import shutil

def apply_log1p_transform(input_dir, output_dir):
    """Apply log1p transform: log(1+x) to all .npy files"""
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    print(f"\nProcessing {len(files)} files from {input_dir}")
    print(f"Output to: {output_dir}\n")
    
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        data = np.load(filepath)
        data_log1p = np.log1p(data)  # Apply log1p transform
        
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, data_log1p)
        
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(files)}] {filename}")
    
    print(f"? Completed: {len(files)} files")

# Apply log1p transform
print("=" * 70)
print("APPLYING LOG1P TRANSFORM TO ALL DATASETS")
print("=" * 70)

apply_log1p_transform(
    '../DownScale_Paper/ERA5_precip_cut',
    '../DownScale_Paper/ERA5_precip_cut_log1p'
)

apply_log1p_transform(
    '../DownScale_Paper/CHIRPS_precip_cut_obs',
    '../DownScale_Paper/CHIRPS_precip_cut_obs_log1p'
)

print("\n" + "=" * 70)
print("DONE! Now update your dataset paths and recompute statistics.")
print("=" * 70)
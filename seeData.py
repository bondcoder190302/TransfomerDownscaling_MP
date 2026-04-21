import numpy as np
import os

folder_path = r'C:\Users\HP\Downloads\MTP_repo\TransfomerDownscaling_MP\DownScale_Paper\HGT_fix_cut_obs'

def inspect_npy_files(directory):
    found_nan = False

    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            data = np.load(file_path)

            print(f"--- File: {filename} ---")
            print(data) # Prints the actual values

            if np.isnan(data).any():
                print(f"⚠️ ALERT: NaN detected in {filename}")
                found_nan = True

            print("\n" + "="*30 + "\n")

    if not found_nan:
        print("Inspection complete: No NaN values found.")

inspect_npy_files(folder_path)
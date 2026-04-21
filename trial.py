import numpy as np
import os

# Define the path to your folder
folder_path = r'C:\Users\HP\Downloads\MTP_repo\TransfomerDownscaling_MP\DownScale_Paper\CHIRPS_precip_cut_obs_log1p'
def check_for_nans(directory):
    files_with_nan = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)

            # Load the numpy array
            data = np.load(file_path)

            # Check for any NaN values
            if np.isnan(data).any():
                files_with_nan.append(filename)

    return files_with_nan

# Run the check
nan_files = check_for_nans(folder_path)

if nan_files:
    print("Files containing NaN values:")
    for f in nan_files:
        print(f)
else:
    print("No NaN values found in any .npy files.")
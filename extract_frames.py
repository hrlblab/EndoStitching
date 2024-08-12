import os
import shutil


def extract_images(source_folder, destination_folder, interval=3):
    """
    Extracts images from a folder at a given interval.

    Parameters:
    source_folder (str): The path to the folder containing the images.
    destination_folder (str): The path to the folder where selected images will be saved.
    interval (int): The number of frames to skip. For 30 fps, every 30th image is one image per second.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_folder, exist_ok=True)

    # List all files in the source directory
    files = sorted(os.listdir(source_folder))

    # Filter to select every 'interval' image
    selected_files = files[::interval]

    # Copy selected images to the destination folder
    for file in selected_files:
        shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder, file))
    print(f"Copied {len(selected_files)} images to {destination_folder}")


# Define the source and destination folders
source_folder = '/Data2/stitching/Dataset/original_dataset/CV-010'
destination_folder = '/Data2/stitching/Dataset/extracted_dataset/CV-010_extracted_3'

# Call the function
extract_images(source_folder, destination_folder, interval=3)

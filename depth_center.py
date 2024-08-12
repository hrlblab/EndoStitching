from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torch.nn.functional as F
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
import open3d as o3d
import matplotlib.pyplot as plt
import os
from natsort import natsorted
from tqdm import tqdm
import json


encoder = 'vitl' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

def get_rgbd(image_list, output_dir='rgbd_images'):
    """
    Processes a list of image paths to generate RGBD images and save them as .npy files, with improved progress reporting.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Setup tqdm with custom description and format
    with tqdm(total=len(image_list), desc="Processing Images", unit="image", leave=True) as pbar:
        for idx, image_path in enumerate(image_list):
            try:
                ori_img = cv2.imread(image_path)
                if ori_img is None:
                    raise ValueError(f"Unable to load image: {image_path}")
                h, w = ori_img.shape[:2]
                image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB) / 255.0
                image = transform({'image': image})['image']
                image = torch.from_numpy(image).unsqueeze(0)

                with torch.no_grad():
                    depth_img = depth_anything(image)

                depth_img_resized = F.interpolate(depth_img[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
                depth_img_uint8 = depth_img_resized.cpu().numpy().astype(np.uint8)
                print(depth_img_uint8)
                if ori_img.shape[:2] != depth_img_uint8.shape[:2]:
                    raise ValueError("RGB and Depth images do not have the same dimensions")

                depth_img_expanded = np.expand_dims(depth_img_uint8, axis=-1)
                rgbd = np.concatenate((ori_img, depth_img_expanded), axis=-1)

                # Save the combined image
                np.save(os.path.join(output_dir, f'{idx}.npy'), rgbd)

                # Update progress bar
                pbar.set_description(f"Processed: {os.path.basename(image_path)}")
                pbar.update(1)

            except Exception as e:
                # Update progress bar with error message
                pbar.write(f"Error processing {image_path}: {str(e)}")
                pbar.update(1)  # Continue the progress even if one image fails


def get_image_names(folder_path):
    all_files = os.listdir(folder_path)
    image_names = [os.path.join(folder_path, file) for file in all_files if
                   file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    return natsorted(image_names)



def load_rgbd_image(file_path):
    # Assuming the file is saved in a format that combines RGB and depth data
    # Depth data is expected to be in the last channel
    rgbd_image = np.load(file_path)
    rgb_image = rgbd_image[:, :, :3]  # RGB Channels
    depth_image = rgbd_image[:, :, 3]  # Depth Channel
    return rgb_image, depth_image


def calculate_mean_depth_point(depth_image, margin = 500):
    # Find the minimum depth value excluding the sides

    min_value = np.min(depth_image[margin:-margin, margin:-margin])
    # Get the indices where the minimum depth value occurs
    min_locations = np.where(depth_image == min_value)
    # Calculate the mean coordinates of these locations
    mean_y = np.mean([y for y in min_locations[0] if margin < y < depth_image.shape[0] - margin])
    mean_x = np.mean([x for x in min_locations[1] if margin < x < depth_image.shape[1] - margin])
    return int(mean_x), int(mean_y)



def plot_mean_point_on_image(image, mean_point):
    """ Plot the mean depth point on the image using a red circle. """
    cv2.circle(image, (mean_point[0], mean_point[1]), 2, (0, 0, 255), -1)  # Red circle
    return image

def process_images_and_record_trajectory(folder_path, output_json='/Data2/stitching/Dataset/extracted_dataset/mean_depth_points_CV_004.json'):
    image_names = get_image_names(folder_path)
    depth_points = {}
    trajectory = []

    with tqdm(total=len(image_names), desc="Processing Images", unit="image", leave=True) as pbar:
        for idx, image_path in enumerate(image_names):
            try:
                ori_img = cv2.imread(image_path)
                if ori_img is None:
                    raise ValueError(f"Unable to load image: {image_path}")
                h, w = ori_img.shape[:2]
                image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB) / 255.0
                image = transform({'image': image})['image']
                image = torch.from_numpy(image).unsqueeze(0)

                with torch.no_grad():
                    depth_img = depth_anything(image)

                depth_img_resized = F.interpolate(depth_img[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
                depth_img_uint8 = depth_img_resized.cpu().numpy().astype(np.uint8)

                mean_point = calculate_mean_depth_point(depth_img_uint8)
                depth_points[image_path] = mean_point

                # Record the trajectory
                trajectory.append(mean_point)

                pbar.update(1)

            except Exception as e:
                pbar.write(f"Error processing {image_path}: {str(e)}")
                pbar.update(1)

    # Optionally save the mean depth points to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(depth_points, json_file, indent=4)

    print(f"Saved mean depth points to {output_json}")

    return trajectory, (w, h)  # Return trajectory and image dimensions for plotting

def plot_depth_center_trajectory(trajectory, image_dimensions):
    plt.figure(figsize=(10, 10))
    plt.plot([point[0] for point in trajectory], [point[1] for point in trajectory], marker='o', color='b', linestyle='-')
    plt.title('Depth Center Movement Trajectory')
    plt.xlabel('Image Width')
    plt.ylabel('Image Height')
    plt.xlim(0, image_dimensions[0])
    plt.ylim(image_dimensions[1], 0)  # Invert y-axis to match image coordinate system
    plt.grid(True)
    plt.show()

# Example usage
folder_path = r'/Data2/stitching/Dataset/extracted_dataset/CV-004_cut'
trajectory, image_dimensions = process_images_and_record_trajectory(folder_path)

# Plot the trajectory
plot_depth_center_trajectory(trajectory, image_dimensions)
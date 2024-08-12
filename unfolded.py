import json
import cv2
import numpy as np
import os

def polar_to_cartesian(u, v, center_x, center_y, r_inner, r_outer):
    radius = r_inner + v
    x = int(center_x + radius * np.cos(u))
    y = int(center_y + radius * np.sin(u))
    return x, y

def create_vcg(r_inner, r_outer, center_x, center_y):
    theta = np.linspace(0, 2 * np.pi, int(2 * np.pi * r_outer), endpoint=False)
    radii = np.linspace(0, r_outer - r_inner, r_outer - r_inner, endpoint=False)
    vcg = np.array([[polar_to_cartesian(t, r, center_x, center_y, r_inner, r_outer) for t in theta] for r in radii])
    return vcg

def unwrap_image(image, vcg):
    unwrapped_image = np.zeros((vcg.shape[0], vcg.shape[1], 3), dtype=np.uint8)
    for v in range(vcg.shape[0]):
        for u in range(vcg.shape[1]):
            x, y = vcg[v, u]
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                unwrapped_image[v, u] = image[y, x]
            else:
                unwrapped_image[v, u] = 0
    return unwrapped_image

def extract_annular_region(image, center_x, center_y, r_inner, r_outer):
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), r_outer, (255, 255, 255), -1)
    cv2.circle(mask, (center_x, center_y), r_inner, (0, 0, 0), -1)
    annular_region = cv2.bitwise_and(image, mask)
    return annular_region

def load_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def save_image(image, path):
    cv2.imwrite(path, image)

def create_output_directories(base_path, subfolders):
    paths = {}
    for folder in subfolders:
        path = os.path.join(base_path, folder)
        if not os.path.exists(path):
            os.makedirs(path)
        paths[folder] = path
    return paths

def process_and_unwrap_images(json_data, r_inner, r_outer, output_dir):
    dirs = create_output_directories(output_dir, ['unwrapped', 'annotated', 'annular'])

    for image_path, center in json_data.items():
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            center_x, center_y = center

            # Extract annular region
            annular_region = extract_annular_region(image, center_x, center_y, r_inner, r_outer)
            annular_image_path = os.path.join(dirs['annular'], os.path.basename(image_path))
            save_image(annular_region, annular_image_path)

            # Create VCG and unwrap the image
            vcg = create_vcg(r_inner, r_outer, center_x, center_y)
            unwrapped_image = unwrap_image(image, vcg)
            unwrapped_image_path = os.path.join(dirs['unwrapped'], os.path.basename(image_path))
            save_image(unwrapped_image, unwrapped_image_path)

            # Annotate original image
            annotated_image = image.copy()
            cv2.circle(annotated_image, (center_x, center_y), r_inner, (0, 255, 0), 2)
            cv2.circle(annotated_image, (center_x, center_y), r_outer, (0, 255, 0), 2)
            annotated_image_path = os.path.join(dirs['annotated'], os.path.basename(image_path))
            save_image(annotated_image, annotated_image_path)

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

# Example usage
def min_distance_to_edge(center_x, center_y, width, height):
    distances = [
        center_x,
        width - center_x,
        center_y,
        height - center_y
    ]
    return min(distances)

width, height = 1370, 1080

json_path = '/Data2/stitching/Dataset/extracted_dataset/mean_depth_points_CV_004.json'
json_data = load_json(json_path)

r_outer = float('inf')
for center in json_data.values():
    center_x, center_y = center
    r_outer = min(r_outer, min_distance_to_edge(center_x, center_y, width, height))

print("Determined r_outer:", r_outer)

r_inner = 10
output_dir = '/Data2/stitching/Dataset/extracted_dataset/CV-004_result'

process_and_unwrap_images(json_data, r_inner, r_outer, output_dir)

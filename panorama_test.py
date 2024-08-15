import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import kornia as K
from kornia.feature import LoFTR
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import seaborn as sns
import json
def ReadImage(ImageFolderPath):
    Images = []

    if os.path.isdir(ImageFolderPath):
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[int(os.path.splitext(os.path.basename(ImageName))[0]), ImageName] for ImageName in
                            ImageNames]
        ImageNames_Split = sorted(ImageNames_Split, key=lambda x: x[0])
        ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]

        for i in range(len(ImageNames_Sorted)):
            ImageName = ImageNames_Sorted[i]
            InputImage = cv2.imread(os.path.join(ImageFolderPath, ImageName))

            if InputImage is None:
                print("Not able to read image: {}".format(ImageName))
                exit(0)

            Images.append(InputImage)

    else:
        print("\nEnter valid Image Folder Path.\n")

    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        exit(1)

    return Images


def FindMatches(BaseImage, SecImage, horizontal_distance_threshold=5):
    loftr = LoFTR(pretrained="outdoor")

    original_size_img1 = BaseImage.shape[:2]
    original_size_img2 = SecImage.shape[:2]

    img0_raw = cv2.resize(BaseImage, (640, 480))
    img1_raw = cv2.resize(SecImage, (640, 480))

    img0 = torch.tensor(img0_raw).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img1 = torch.tensor(img1_raw).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    img0_gray = K.color.rgb_to_grayscale(img0)
    img1_gray = K.color.rgb_to_grayscale(img1)

    batch = {'image0': img0_gray, 'image1': img1_gray}

    with torch.no_grad():
        output = loftr(batch)

    keypoints0 = output["keypoints0"]
    keypoints1 = output["keypoints1"]
    confidence = output["confidence"]

    good_matches = []
    for i, conf in enumerate(confidence):
        if conf > 0.2:
            kp0_x, kp0_y = keypoints0[i].cpu().numpy()
            kp1_x, kp1_y = keypoints1[i].cpu().numpy()

            # Scale keypoints back to original image size
            kp0_x *= (original_size_img1[1] / 640)
            kp1_x *= (original_size_img2[1] / 640)

            # Check the horizontal distance between keypoints
            if abs(kp0_x - kp1_x) <= horizontal_distance_threshold:
                match = cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=conf.item())
                good_matches.append(match)

    scale_factor_img1 = (original_size_img1[1] / 640, original_size_img1[0] / 480)
    scale_factor_img2 = (original_size_img2[1] / 640, original_size_img2[0] / 480)

    kp1 = [cv2.KeyPoint(x=kp[0].item() * scale_factor_img1[0], y=kp[1].item() * scale_factor_img1[1], size=1) for kp in
           keypoints0]
    kp2 = [cv2.KeyPoint(x=kp[0].item() * scale_factor_img2[0], y=kp[1].item() * scale_factor_img2[1], size=1) for kp in
           keypoints1]

    return good_matches, kp1, kp2


def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    if len(Matches) < 4:
        print("\nNot enough matches found between the images.\n")
        exit(0)

    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match.queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match.trainIdx].pt)

    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    return HomographyMatrix, Status


def GetNewFrameSizeAndMatrix(HomographyMatrix, Sec_ImageShape, Base_ImageShape, horizontal_threshold):
    (Height, Width) = Sec_ImageShape

    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])

    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)

    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPoints = np.float32(np.array([x, y]).transpose())

    displacement = np.mean(NewFinalPoints[:, 0] - OldInitialPoints[:, 0])
    if abs(displacement) > horizontal_threshold:
        adjustment_factor = horizontal_threshold / abs(displacement)
        NewFinalPoints[:, 0] = OldInitialPoints[:, 0] + adjustment_factor * (
                NewFinalPoints[:, 0] - OldInitialPoints[:, 0])

    HomographyMatrix = cv2.getPerspectiveTransform(OldInitialPoints, NewFinalPoints)

    return [New_Height, New_Width], Correction, HomographyMatrix


def AdjustAlignment(HomographyMatrix, BaseImage, SecImage):
    # Project the keypoints from SecImage using the HomographyMatrix
    SecImage_projected = cv2.perspectiveTransform(np.float32([[[0, 0], [SecImage.shape[1] - 1, 0],
                                                               [SecImage.shape[1] - 1, SecImage.shape[0] - 1],
                                                               [0, SecImage.shape[0] - 1]]]),
                                                  HomographyMatrix)

    # Calculate overlap and edge distances
    left_overlap = SecImage_projected[0, :, 0].min() - BaseImage.shape[1]
    right_overlap = SecImage_projected[0, :, 0].max() - BaseImage.shape[1]

    # Adjust HomographyMatrix based on overlaps
    if left_overlap < 0:
        HomographyMatrix[0, 2] -= left_overlap  # Shift the new image to the right
    if right_overlap > BaseImage.shape[1]:
        HomographyMatrix[0, 2] -= (right_overlap - BaseImage.shape[1])  # Shift the new image to the left

    return HomographyMatrix


def PostStitchingAdjustment(StitchedImage, BaseImage, horizontal_threshold):
    """Adjust the stitched image if horizontal displacement exceeds the threshold."""
    height, width = BaseImage.shape[:2]

    # Calculate horizontal displacement based on image width
    stitched_width = StitchedImage.shape[1]
    base_width = BaseImage.shape[1]

    # If the stitched image width is larger than the allowed threshold, compress it horizontally
    if stitched_width > base_width + horizontal_threshold:
        compression_factor = (base_width + horizontal_threshold) / stitched_width
        StitchedImage = cv2.resize(StitchedImage, (int(stitched_width * compression_factor), height))

    return StitchedImage


def StitchImages(BaseImage, SecImage, Correction, HomographyMatrix, horizontal_threshold, max_width_ratio=1.5):
    SecImage_Cyl, mask_x, mask_y = ProjectOntoCylinder(SecImage)

    SecImage_Mask = np.zeros(SecImage_Cyl.shape, dtype=np.uint8)
    SecImage_Mask[mask_y, mask_x, :] = 255

    NewFrameSize, Correction, HomographyMatrix = GetNewFrameSizeAndMatrix(HomographyMatrix, SecImage_Cyl.shape[:2],
                                                                          BaseImage.shape[:2], horizontal_threshold)

    SecImage_Transformed = cv2.warpPerspective(SecImage_Cyl, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))
    SecImage_Transformed_Mask = cv2.warpPerspective(SecImage_Mask, HomographyMatrix, (NewFrameSize[1], NewFrameSize[0]))

    # Adjust alignment to ensure that the new image fits properly within the panorama
    HomographyMatrix = AdjustAlignment(HomographyMatrix, BaseImage, SecImage_Transformed)

    BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    BaseImage_Transformed[Correction[1]:Correction[1] + BaseImage.shape[0],
    Correction[0]:Correction[0] + BaseImage.shape[1]] = BaseImage

    StitchedImage = cv2.bitwise_or(SecImage_Transformed,
                                   cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))

    # Perform post-stitching adjustment for horizontal displacement by compression
    StitchedImage = PostStitchingAdjustment(StitchedImage, BaseImage, horizontal_threshold)

    return StitchedImage, Correction, HomographyMatrix


def visualize_matches(img1, img2, kp1, kp2, matches, output_dir, idx):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    output_path = os.path.join(output_dir, f"matches_{idx}.png")
    cv2.imwrite(output_path, img_matches)


def Convert_xy(x, y):
    global center, f

    xt = (f * np.tan((x - center[0]) / f)) + center[0]
    yt = ((y - center[1]) / np.cos((x - center[0]) / f)) + center[1]

    return xt, yt


def ProjectOntoCylinder(InitialImage):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    f = 1500

    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)

    AllCoordinates_of_ti = np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]

    ii_x, ii_y = Convert_xy(ti_x, ti_y)

    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w - 2)) * \
                  (ii_tl_y >= 0) * (ii_tl_y <= (h - 2))

    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]

    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]

    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]

    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx) * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx) * (dy)

    TransformedImage[ti_y, ti_x, :] = (weight_tl[:, None] * InitialImage[ii_tl_y, ii_tl_x, :]) + \
                                      (weight_tr[:, None] * InitialImage[ii_tl_y, ii_tl_x + 1, :]) + \
                                      (weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x, :]) + \
                                      (weight_br[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x + 1, :])

    min_x = min(ti_x)

    TransformedImage = TransformedImage[:, min_x: -min_x, :]

    return TransformedImage, ti_x - min_x, ti_y




def calculate_ssim(BaseImage, SecImage, overlap_region):
    """Calculate the SSIM between overlapping regions of two images."""
    # Ensure overlap region is within the bounds of the BaseImage
    start_idx = max(0, overlap_region[0])
    end_idx = min(BaseImage.shape[1], overlap_region[1])

    if start_idx >= end_idx:
        return 0.0  # No valid overlap region, return 0 for SSIM

    # Extract the overlapping regions
    BaseImage_overlap = BaseImage[:, start_idx:end_idx]
    SecImage_overlap = SecImage[:, :end_idx - start_idx]

    # Crop the overlapping regions to the minimum height to make them match
    min_height = min(BaseImage_overlap.shape[0], SecImage_overlap.shape[0])
    BaseImage_overlap = BaseImage_overlap[:min_height, :]
    SecImage_overlap = SecImage_overlap[:min_height, :]

    # Convert to grayscale if images are not already grayscale
    if BaseImage_overlap.ndim == 3:
        BaseImage_overlap = cv2.cvtColor(BaseImage_overlap, cv2.COLOR_BGR2GRAY)
    if SecImage_overlap.ndim == 3:
        SecImage_overlap = cv2.cvtColor(SecImage_overlap, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between overlapping regions
    ssim_value = ssim(BaseImage_overlap, SecImage_overlap)

    return ssim_value

def analyze_stitching_ssim(BaseImage, SecImage):
    # Define the overlap region
    overlap_region = (BaseImage.shape[1] - SecImage.shape[1], BaseImage.shape[1])

    # Calculate SSIM
    ssim_value = calculate_ssim(BaseImage, SecImage, overlap_region)

    return ssim_value
if __name__ == "__main__":
    data_path = "/Data2/stitching/Dataset/extracted_dataset/CV-004_result/unwrapped"
    Images = ReadImage(data_path)
    output_dir = "/Data2/stitching/Dataset/extracted_dataset/CV-004_result/test_1"
    os.makedirs(output_dir, exist_ok=True)

    match_counts = []  # To record the number of matches for each image pair
    rmse_values = []  # To record the RMSE for each image pair
    ssim_values = []  # To record the SSIM for each image pair

    BaseImage, _, _ = ProjectOntoCylinder(Images[0])
    Correction = [0, 0]

    HomographyMatrix = np.eye(3)
    horizontal_threshold = 1  # Set a threshold for horizontal displacement

    for i in tqdm(range(1, len(Images)), desc="Stitching Images"):
        Matches, BaseImage_kp, SecImage_kp = FindMatches(Images[i - 1], Images[i])
        visualize_matches(Images[i - 1], Images[i], BaseImage_kp, SecImage_kp, Matches, output_dir, i)

        match_counts.append(len(Matches))  # Record the number of matches for this pair

        HomographyMatrix, _ = FindHomography(Matches, BaseImage_kp, SecImage_kp)
        StitchedImage, Correction, HomographyMatrix = StitchImages(BaseImage, Images[i], Correction, HomographyMatrix,
                                                                   horizontal_threshold)
        BaseImage = StitchedImage.copy()

        output_path = os.path.join(output_dir, f"Stitched_Panorama_{i}.png")
        cv2.imwrite(output_path, BaseImage)



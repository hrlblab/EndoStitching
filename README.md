# EndoStitching
This is codes for "Automatic Image Unfolding and Stitching Framework for Esophageal Lining Video Based on Density-Weighted Feature Matching".

## Environment settings
1. Git clone Depth-anything
   ```
   git clone https://github.com/LiheYoung/Depth-Anything
   ```
   The file unfolded.py and depth_center.py file should be put in the Depth-anything folder to make sure its working properly.

2. create conda environment
   ```
   conda env create -f environment.yaml
   ```
3. activate conda environment
   ```
   conda activate endostitching


## Find depth center and unfolding the images
   run the depth_center.py. It will generate a json file to store all the coordinates of depth centers for each image.
   ```
   python depth_center.py
   ```
   Find the json file you generate and unfold the images
   ```
   python unfolded.py
   ```

## Stitch the unfolded images
   ```
   python ponarama_test.py
   ```



import cv2
import os
import numpy as np
import shutil

def is_blurry(image_path, threshold=100.0):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance < threshold

def is_pink(image_path, pink_threshold=50.0):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    (B, G, R) = cv2.split(image)
    avg_red = np.mean(R)
    avg_green = np.mean(G)
    avg_blue = np.mean(B)
    return avg_red > avg_blue and avg_red > avg_green and avg_blue > avg_green

def check_images_recursively(root_folder, error_folder='ERRORS', threshold=100.0, pink_threshold=50.0):
    os.makedirs(error_folder, exist_ok=True)  # Make sure ERRORS folder exists

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(dirpath, filename)
                try:
                    blurry = is_blurry(image_path, threshold)
                    pink = is_pink(image_path, pink_threshold)

                    if blurry or pink:
                        print(f"PROBLEMATIC: {image_path}")
                        # Ensure unique name in case of duplicates
                        dest_filename = os.path.relpath(image_path, root_folder).replace(os.sep, '__')
                        dest_path = os.path.join(error_folder, dest_filename)
                        shutil.copy2(image_path, dest_path)

                except Exception as e:
                    print(f"ERROR processing {image_path}: {e}")

# Set the root of your 47.04 folder here
folder_path = './47.04'
check_images_recursively(folder_path)

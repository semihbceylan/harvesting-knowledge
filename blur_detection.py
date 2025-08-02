import cv2
import os
import numpy as np

def is_blurry(image_path, threshold=100.0):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Compute the variance of the Laplacian
    variance = laplacian.var()
    
    # If the variance is less than the threshold, the image is considered blurry
    return variance < threshold

def is_pink(image_path, pink_threshold=50.0):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Split the image into its RGB channels
    (B, G, R) = cv2.split(image)
    
    # Calculate the average intensity of red, green, and blue channels
    avg_red = np.mean(R)
    avg_green = np.mean(G)
    avg_blue = np.mean(B)
    
    # Define a pink color condition (high red and blue, lower green)
    if avg_red > avg_blue and avg_red > avg_green and avg_blue > avg_green:
        return True
    return False

def check_images_in_folder(folder_path, threshold=100.0, pink_threshold=50.0):
    # Get all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out non-image files based on common image extensions
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Loop through all the image files and check if they are blurry or pink
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        # Check if the image is blurry
        blurry = is_blurry(image_path, threshold)
        if blurry:
            print(f"The image {image_file} is blurry.")
        else:
            print(f"The image {image_file} is sharp.")
        
        # Check if the image is pink
        pink = is_pink(image_path, pink_threshold)
        if pink:
            print(f"The image {image_file} is overall pink.")
        else:
            print(f"The image {image_file} is not pink.")

# Example usage:
folder_path = 'C:/Users/UYSM-9/Desktop/semihbc/ERRORS'
check_images_in_folder(folder_path)

import os
import glob
import random
import cv2
import numpy as np


# Dataset Directory
dataset_dir = "./Dataset"
categories = ["Capacitor", "Resistor"]

# Output Directory
output_dir = "./seg_images"
os.makedirs(output_dir, exist_ok=True)

# random 10 images
image_paths = []
for category in categories:
    image_files = glob.glob(os.path.join(dataset_dir, category, "*.jpeg")) + \
                  glob.glob(os.path.join(dataset_dir, category, "*.jpg")) + \
                  glob.glob(os.path.join(dataset_dir, category, "*.png"))
    
    if len(image_files) > 0:
        sampled_files = random.sample(image_files, min(len(image_files), 10))
        image_paths.extend(sampled_files)

print(f"Selected {len(image_paths)} images for processing.")

# HSV Color
COLOR_RANGES = {
    "Blue_Capacitor": ([90, 50, 50], [130, 255, 255]),    # blue capac
    "Brown_Resistor": ([10, 100, 20], [20, 255, 200]),    # braun resistor
    "Red_Resistor": ([0, 120, 70], [10, 255, 255]),       # red resistor
    "Green_Capacitor": ([35, 50, 50], [85, 255, 255]),    # green capacitor
    "Black_Component": ([0, 0, 0], [180, 255, 30]),       # black component
    "Yellow_Band": ([20, 100, 100], [30, 255, 255])       # yellow band
}

# Pipeline
for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Image file not found: {image_path}")
        continue

    file_name = os.path.basename(image_path)
    image_name, ext = os.path.splitext(file_name)
    image_folder = os.path.join(output_dir, image_name)
    os.makedirs(image_folder, exist_ok=True) 
    
    # Save original img
    cv2.imwrite(os.path.join(image_folder, f"Original_{file_name}"), image)

    # Convert GrayScale and Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Select Adaptive Thresholding vs Otsu
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold_result = adaptive_thresh if np.mean(adaptive_thresh) > np.mean(otsu_thresh) else otsu_thresh
    
    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # HSV Color Filter 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    filtered_images = {}
    color_segmented = image.copy()
    
    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # noise removal
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
        
        # filtering dark mask
        if np.mean(processed_mask) < 10:
            continue
        
        filtered = cv2.bitwise_and(image, image, mask=processed_mask)
        filtered_images[color_name] = filtered
        
        cv2.imwrite(os.path.join(image_folder, f"{color_name}_Mask_{file_name}"), processed_mask)
    
    # Contour and Bound Box
    all_contours = []
    
    for  color_name, filtered_img in filtered_images.items():
        gray_filtered = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        found_contours, _ = cv2.findContours(gray_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours += found_contours
    
    # Edge/Otsu 
    edge_contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    otsu_contours, _ = cv2.findContours(threshold_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_contours += edge_contours + otsu_contours
    
    # Leave only large contours
    unique_contours = []
    min_contour_area = (image.shape[0] * image.shape[1]) * 0.001
    
    for c in all_contours:
        if cv2.contourArea(c) < min_contour_area:
            x, y, w, h = cv2.boundingRect(c)
            if not any(abs(x - ux) < 10 and abs(y - uy) < 10 for ux, uy, _, _ in unique_contours):
                unique_contours.append((x, y, w, h))
    
    # Apply bounding box
    combined_segmented = image.copy()
    for x, y, w, h in unique_contours:
        cv2.rectangle(combined_segmented, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        # Extract ROI
        roi = image[y:y + h, x:x + w]
        if roi.shape[0] > 10 and roi.shape[1] > 10:
            cv2.imwrite(os.path.join(image_folder, f"ROI_{x}_{y}_{file_name}"), roi)
    
    cv2.imwrite(os.path.join(image_folder, f"Combined_Segmented_{file_name}"), combined_segmented)
    print(f"Saved segmented results for {file_name} in {image_folder}")

print("Image Segmentation Completed!")

import os
import cv2
import yaml
import numpy as np


def numeric_sort_key(filepath):
    """
    Extracts the numeric part of the filename for sorting.
    
    Parameters:
    filepath (str): The full path to the file (e.g., "path/to/image/1.jpg").
    
    Returns:
    int: The numeric part of the filename.
    """
    # Extract the filename from the full path
    filename = os.path.basename(filepath)
    
    # Extract the numeric part (before the extension)
    return int(os.path.splitext(filename)[0])


def save_images_to_memmap(images, memmap_file, crop_coords, channels):
    """
    Saves a list of cropped images to a memory-mapped file.
    
    Parameters:
    images (list of np.ndarray): List of images (as numpy arrays) to be saved.
    memmap_file (str): Path to the memmap file to save the images.
    crop_coords (tuple): Coordinates (top, left, bottom, right) for the rectangle to crop.
    channels (int): Number of channels in each image.
    """
    if len(images) == 0:
        raise ValueError("No images to save.")
    
    top, left, bottom, right = crop_coords
    cropped_height = bottom - top
    cropped_width = right - left

    # Verify that all images can be cropped to the specified shape
    for img in images:
        if img.shape[0] < bottom or img.shape[1] < right:
            raise ValueError("Image dimensions are smaller than crop dimensions.")

    # Determine the dtype from the first image
    img_dtype = images[0].dtype
    
    # Create a memory-mapped file to hold all cropped images
    num_images = len(images)
    memmap = np.memmap(memmap_file, dtype=img_dtype, mode='w+', shape=(num_images, cropped_height, cropped_width, channels))
    
    # Write each cropped image to the memory-mapped array
    for i, img in enumerate(images):
        cropped_img = img[top:bottom, left:right, :]
        memmap[i] = cropped_img
    
    # Flush changes to the disk
    memmap.flush()


if __name__ == "__main__":
    # Load the YAML configuration file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Save the images
    image_paths = [os.path.join(config["path_to_image_directory"], image) for image in os.listdir(config["path_to_image_directory"])]
    sorted_image_paths = sorted(image_paths, key=numeric_sort_key)
    
    images = [cv2.imread(image_path) for image_path in sorted_image_paths]

    crop_coords = (960, 395, config["height"] - 905, config["width"] - 395)  # (top, left, bottom, right)

    save_images_to_memmap(images=images,
                          memmap_file=config["path_to_image_memmap"],
                          crop_coords=crop_coords,
                          channels=3)

    # Show a cropped image
    num_images = len(os.listdir(config["path_to_image_directory"]))
    cropped_height = crop_coords[2] - crop_coords[0]
    cropped_width = crop_coords[3] - crop_coords[1]
    
    memmap = np.memmap(config["path_to_image_memmap"],
                       dtype=np.uint8,
                       mode='r',
                       shape=(num_images, cropped_height, cropped_width, 3)
                       )
    
    cv2.imshow(f"Cropped Image {101}", memmap[51])
    cv2.waitKey(0)

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


def save_images_to_memmap(images, memmap_file, height, width, channels):
    """
    Saves a list of images to a memory-mapped file.
    
    Parameters:
    images (list of np.ndarray): List of images (as numpy arrays) to be saved.
    memmap_file (str): Path to the memmap file to save the images.
    height (int): Height of each image.
    width (int): Width of each image.
    channels (int): Number of channels in each image.
    """
    if len(images) == 0:
        raise ValueError("No images to save.")
    
    # Verify that all images have the same shape
    for img in images:
        if img.shape != (height, width, channels):
            raise ValueError("All images must have the same shape.")
    
    # Determine the dtype from the first image
    img_dtype = images[0].dtype
    
    # Create a memory-mapped file to hold all images
    num_images = len(images)
    memmap = np.memmap(memmap_file, dtype=img_dtype, mode='w+', shape=(num_images, height, width, channels))
    
    # Write each image to the memory-mapped array
    for i, img in enumerate(images):
        memmap[i] = img
    
    # Flush changes to the disk
    memmap.flush()


if __name__ == "__main__":
    # Load the YAML configuration file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Save the images
    image_paths = [os.path.join(config["path_to_image_directory"], image) for image in os.listdir(config["path_to_image_directory"])]
    sorted_image_paths = sorted(image_paths,key=numeric_sort_key)
    
    images = [cv2.imread(image_path) for image_path in sorted_image_paths]

    save_images_to_memmap(images=images,
                          memmap_file=config["path_to_image_memmap"],
                          height=config["height"],
                          width=config["width"],
                          channels=3)

    # Show an image
    num_images = len(os.listdir(config["path_to_image_directory"]))
    memmap = np.memmap(config["path_to_image_memmap"],
                       dtype=np.uint8,
                       mode='r',
                       shape=(num_images, config["height"], config["width"], 3)
                       )
    
    cv2.imshow(f"Image {101}", memmap[51])
    cv2.waitKey(0)
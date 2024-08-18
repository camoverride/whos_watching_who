import os
import cv2

def resize_images(input_dir, output_dir, width, height):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read the image
        img = cv2.imread(input_path)
        
        if img is not None:
            # Resize the image
            resized_img = cv2.resize(img, (width, height))
            
            # Save the resized image to the output directory
            cv2.imwrite(output_path, resized_img)
            print(f"Resized and saved: {output_path}")
        else:
            print(f"Failed to load: {input_path}")

# Example usage
input_dir = "pics/jeff_1080-1920"
output_dir = "pics/jeff_1080-1920_resized"
width = 1920  # Desired width
height = 1080  # Desired height

resize_images(input_dir, output_dir, width, height)

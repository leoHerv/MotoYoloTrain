import cv2
import numpy as np
import os
import glob
from pathlib import Path

def convert_images_to_squares(input_dir: str, output_dir: str, target_size: int, container_classes: list[int]):
    """
    Converts a directory of non-square images to square images (images and labels) in the input directory.

    Args:
        input_dir (str): Root directory containing 'images' and 'labels' subdirectories.
        output_dir (str): Destination directory for the processed dataset.
        target_size (int): Target size for the images.
        container_classes (list[int]): List of class IDs for containers.
        min_box_size (int): Minimum size (width or height) for an object to be kept.
    """
    input_images_dir = os.path.join(input_dir, "images")
    input_labels_dir = os.path.join(input_dir, "labels")

    # Find all images
    # Supports common image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_images_dir, ext)))
        # Also check for uppercase extensions just in case
        image_files.extend(glob.glob(os.path.join(input_images_dir, ext.upper())))

    for image_path in image_files:
        # Construct corresponding label path
        # Assumes label has same basename as image but with .txt extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(input_labels_dir, base_name + ".txt")

        if os.path.exists(label_path):
            process_image_and_label(image_path, label_path, output_dir, target_size, container_classes)

def letterbox(img: np.ndarray, new_size: int, color: tuple=(0, 0, 0)) -> (np.ndarray, float, tuple):
    """
    Resizes an image to a target size using letterboxing (padding) to maintain aspect ratio.

    Args:
        img (np.ndarray): The input image.
        new_size (int): The target size (width and height).
        color (tuple): The color for the padding (B, G, R).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The resized and padded image.
            - float: The scale ratio used.
            - tuple: The padding (dw, dh) applied.
    """
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    ratio = min(new_size / shape[0], new_size / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio)) # (height, width)
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def process_image_and_label(image_path: str, label_path: str, output_dir: str, target_size: int, container_classes: list[int], min_box_size: int = 15):
    """
    Processes a single image and its corresponding label file.
    Resizes the image with letterboxing and transforms the OBB labels.

    Args:
        image_path (str): Path to the source image.
        label_path (str): Path to the source label file.
        output_dir (str): Path to the output directory.
        target_size (int): Target size for the square image.
        container_classes (list[int]): List of class IDs for containers.
        min_box_size (int): Minimum size (width or height) for an object to be kept.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return

    image_name = os.path.basename(image_path)

    # Create output directories if they don't exist
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    h0, w0 = img.shape[:2]

    # Process labels if file exists
    if not os.path.exists(label_path):
        return

    new_lines = []
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Removes the '\n'.
    lines = [line for line in lines if line.strip().split()]
    lines = [line for line in lines if int(line.strip().split()[0]) in container_classes]

    new_img = None

    if h0 == w0:
        new_img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        new_lines = lines
    else:
        # Resize image with letterbox
        new_img, ratio, (dw, dh) = letterbox(img, new_size=target_size)

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            cls = parts[0]
            coords = list(map(float, parts[1:]))
            
            # Reshape to (N, 2) points
            points = np.array(coords).reshape(-1, 2)
            
            # Denormalize
            points[:, 0] *= w0
            points[:, 1] *= h0
            
            # Calculate real dimensions
            width = np.linalg.norm(points[0] - points[1])
            height = np.linalg.norm(points[1] - points[2])
            
            if width < min_box_size or height < min_box_size:
                continue
            
            # Apply scaling
            points *= ratio
            
            # Apply padding
            points[:, 0] += dw
            points[:, 1] += dh
            
            # Renormalize
            points[:, 0] /= target_size
            points[:, 1] /= target_size
            
            # Clip to [0, 1] to be safe
            points = np.clip(points, 0, 1)
            
            # Flatten back to list
            new_coords = points.flatten().tolist()
            new_line = f"{cls} " + " ".join(f"{x:.6f}" for x in new_coords)
            new_lines.append(new_line)

    # Save new label
    label_name = os.path.basename(label_path)
    with open(os.path.join(output_labels_dir, label_name), 'w') as f:
        f.write('\n'.join(new_lines))

    # Save new image
    cv2.imwrite(os.path.join(output_images_dir, image_name), new_img)



def filter_and_remap_file(file_path: str, keep_classes: list[int]):
    """
    Filters and remaps class IDs in a label file.
    Keeps only the specified classes, removes all other classes and remaps the classes to be 0-ordered for training.
    
    Args:
        file_path (str): Path to the label file.
        keep_classes (list[int]): List of class IDs to keep. 
                                  IDs will be remapped to their index in this list.
    """
    if not os.path.exists(file_path):
        return

    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    
    # Create a mapping from old_id to new_id
    id_map = {old_id: new_id for new_id, old_id in enumerate(keep_classes)}

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        try:
            class_id = int(parts[0])
            if class_id in id_map:
                # Remap class ID
                new_class_id = id_map[class_id]
                # Reconstruct line with new class ID
                parts[0] = str(new_class_id)
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
            # If not in keep_classes, it is filtered out (skipped)
        except ValueError:
            continue

    # Write back to file
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

def process_labels_directory(directory_path: str, keep_classes: list[int]):
    """
    Processes all label files in a directory, filtering and remapping classes.
    
    Args:
        directory_path (str): Path to the directory containing label files.
        keep_classes (list[int]): List of class IDs to keep.
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    path = Path(directory_path)

    txt_files = glob.glob(os.path.join(path, "*.txt"))
    count = 0
    
    for txt_file in txt_files:
        filter_and_remap_file(txt_file, keep_classes)
        count += 1
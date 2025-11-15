from create_dataset import create_destination_directory

import cv2
import numpy as np
from pathlib import Path

# (B, G, R)
COLOR_BACKGROUND = (0, 0, 0)

def get_pixel_points(points_norm: list, img_w: int, img_h: int) -> list:
    """
    Converts normalized points to pixel coordinates.

    Args:
        points_norm (list): A list of 8 normalized coordinates [x1, y1, ..., x4, y4].
        img_w (int): The width of the image.
        img_h (int): The height of the image.

    Returns:
        list: A list of 8 pixel coordinates.
    """
    points_pix = []
    for i in range(8):
        points_pix.append(points_norm[i] * (img_w if i % 2 == 0 else img_h))

    return points_pix


def get_box_from_obb(points_norm: list, img_w: int, img_h: int) -> (int, int, int, int):
    """
    Finds the non-oriented bounding box (AABB) from 8 normalized coordinates.
    
    Args:
        points_norm (list): A list of 8 normalized coordinates [x1, y1, x2, y2, x3, y3, x4, y4].
        img_w (int): The width of the image.
        img_h (int): The height of the image.

    Returns:
        (int, int, int, int): A tuple (x_min, y_min, x_max, y_max) in pixels, clamped to the image dimensions.
    """
    points_pix = get_pixel_points(points_norm, img_w, img_h)

    x_coords = [points_pix[0], points_pix[2], points_pix[4], points_pix[6]]
    y_coords = [points_pix[1], points_pix[3], points_pix[5], points_pix[7]]

    x_min: int = max(0, int(np.min(x_coords)))
    y_min: int = max(0, int(np.min(y_coords)))
    x_max: int = min(img_w, int(np.max(x_coords)))
    y_max: int = min(img_h, int(np.max(y_coords)))

    return x_min, y_min, x_max, y_max


def get_area_from_image(img_path, label_path, output_img_dir, output_label_dir, container_classes: list, in_container_classes: list) -> int:
    """
    Processes a single image:
    1. Reads the image and its labels (8-point format).
    2. Finds all containers.
    3. For each container, creates a square crop (with smart padding).
    4. Finds the numbers whose center is within that container.
    5. Recalculates the 8 coordinates of the numbers for the new image.
    6. Saves the new image and the new label.
    
    Args:
        img_path: Path to the image file.
        label_path: Path to the label file.
        output_img_dir: Directory to save the output images.
        output_label_dir: Directory to save the output labels.
        container_classes (list): List of class IDs for containers.
        in_container_classes (list): List of class IDs for objects within containers.

    Returns:
        int: The number of crops created from the image.
    """
    try:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error: Cannot read the image : {img_path}")
            return

        img_h, img_w = image.shape[:2]

        with open(label_path, 'r') as f:
            lines = f.readlines()

        all_labels: list = []
        for line in lines:
            parts: list = [float(p) for p in line.strip().split()]
            # 1 ID + 8 coordinates = 9 parts
            if len(parts) == 9:
                all_labels.append(parts)

        # Separate containers and numbers
        containers_labels: list = [label for label in all_labels if int(label[0]) in container_classes]
        in_container_labels: list = [label for label in all_labels if int(label[0]) in in_container_classes]

        if not containers_labels:
            # No container, nothing to process.
            return

        img_name: str = img_path.stem  # Name of the file without extension.
        crop_counter: int = 0

        for container_label in containers_labels:
            container_class_id: int = int(container_label[0])
            container_points: list = container_label[1:]  # 8 points [x1, y1, ..., x4, y4]

            # Get the box from the container.
            container_x_min, container_y_min, container_x_max, container_y_max = get_box_from_obb(container_points, img_w, img_h)

            box_width: int = container_x_max - container_x_min
            box_height: int = container_y_max - container_y_min
            if box_width <= 0 or box_height <= 0:
                continue # Next container.

            box_size: int = max(box_width, box_height)

            # Box center.
            center_x: float = (container_x_min + container_x_max) / 2
            center_y: float = (container_y_min + container_y_max) / 2

            # Calculate the new coordinates of the square crop
            new_box_x_min: int = int(center_x - box_size / 2)
            new_box_y_min: int = int(center_y - box_size / 2)
            new_box_x_max: int = new_box_x_min + box_size
            new_box_y_max: int = new_box_y_min + box_size

            # Box canvas.
            box = np.full((box_size, box_size, 3), COLOR_BACKGROUND, dtype=np.uint8)

            # Container source.
            src_x_min = max(0, new_box_x_min)
            src_y_min = max(0, new_box_y_min)
            src_x_max = min(img_w, new_box_x_max)
            src_y_max = min(img_h, new_box_y_max)

            # Container destination in the box.
            dest_x_min = src_x_min - new_box_x_min
            dest_y_min = src_y_min - new_box_y_min
            dest_x_max = src_x_max - new_box_x_min
            dest_y_max = src_y_max - new_box_y_min

            # Extract the container and paste it in the box.
            if (src_x_max > src_x_min) and (src_y_max > src_y_min):
                image_container = image[src_y_min:src_y_max, src_x_min:src_x_max]
                box[dest_y_min:dest_y_max, dest_x_min:dest_x_max] = image_container

            # Find the object in the container and convert their coordinates.
            new_labels_for_box = []

            # Container label.
            container_pixel_points = get_pixel_points(container_points, img_w, img_h)
            container_label_line = f"{container_class_id}"
            i: int = 0
            for p in container_pixel_points:
                p = p - new_box_x_min if i % 2 == 0 else p - new_box_y_min
                p = p / box_size
                container_label_line = container_label_line + "".join(f" {p:.6f}")
                i += 1
            new_labels_for_box.append(container_label_line)

            for in_container_label in in_container_labels:
                in_container_class_id = int(in_container_label[0])
                in_container_points = in_container_label[1:]  # 8 points.

                # In container object coordinates on the original image.
                ic_x_min, ic_y_min, ic_x_max, ic_y_max = get_box_from_obb(in_container_points, img_w, img_h)
                ic_x_center = (ic_x_min + ic_x_max) / 2
                ic_y_center = (ic_y_min + ic_y_max) / 2

                # Verify if the center of the in container object is in the container.
                if (container_x_min <= ic_x_center <= container_x_max) and (container_y_min <= ic_y_center <= container_y_max):

                    # Points in pixel on the image.
                    ic_pixel_points = get_pixel_points(in_container_points, img_w, img_h)

                    new_points = []
                    for i in [0, 2, 4, 6]:
                        px, py = ic_pixel_points[i], ic_pixel_points[i + 1]

                        # Convert to coordinates relative to the new crop
                        # The origin is (new_box_x_min, new_box_y_min)
                        new_px = px - new_box_x_min
                        new_py = py - new_box_y_min

                        # Re-normalize by the square size
                        new_norm_x = new_px / box_size
                        new_norm_y = new_py / box_size

                        new_points.append(f"{new_norm_x:.6f}")
                        new_points.append(f"{new_norm_y:.6f}")

                    new_label_line = f"{in_container_class_id} " + " ".join(new_points)
                    new_labels_for_box.append(new_label_line)

            # New file name.
            new_file_name = f"{img_name}_c{crop_counter}"

            # Save the new image.
            new_img_path = output_img_dir / f"{new_file_name}.jpg"
            cv2.imwrite(str(new_img_path), box)

            # Save the new label file.
            new_label_path = output_label_dir / f"{new_file_name}.txt"
            with open(new_label_path, 'w') as f:
                f.write("\n".join(new_labels_for_box))

            crop_counter += 1
        return crop_counter # Return the number of crop done.

    except Exception as e:
        print(f"Failed to process {img_path}: {e}")


def get_areas_from_images(data_source: str, output_dir: str, container_classes: list, in_container_classes: list) -> (int, int):
    """
    Processes all images in a directory to extract areas based on container labels.

    Args:
        data_source (str): Path to the source data directory, containing 'images' and 'labels' subdirectories.
        output_dir (str): Path to the output directory where the new dataset will be saved.
        container_classes (list): List of class IDs for containers.
        in_container_classes (list): List of class IDs for objects within containers.

    Returns:
        (int, int): A tuple containing the number of processed images and the total number of crops created.
    """
    img_dir_path = Path(data_source) / "images"
    label_dir_path = Path(data_source) / "labels"
    output_dir_path = Path(output_dir)

    output_dir_path = create_destination_directory(output_dir_path)

    output_img_dir_path = output_dir_path / "images"
    output_label_dir_path = output_dir_path / "labels"

    # Create output directories.
    output_img_dir_path.mkdir(parents=True, exist_ok=True)
    output_label_dir_path.mkdir(parents=True, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp'] # For Yolo training.
    processed_count: int = 0
    crops_count: int = 0
    for img_path in img_dir_path.glob('*'):
        if img_path.suffix.lower() in image_extensions:

            # Find the label file.
            label_path = label_dir_path / f"{img_path.stem}.txt"

            # If no label file found, skip.
            if not label_path.exists():
                continue

            crops_count += get_area_from_image(img_path, label_path, output_img_dir_path, output_label_dir_path, container_classes, in_container_classes)
            processed_count += 1
    return processed_count, crops_count

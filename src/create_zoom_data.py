import cv2
import numpy as np
import os

def create_zoom_images(crop_size: int, path_source_raw: str, path_destination_raw: str, container_classes: list[int]) -> None:
    """
    Generates zoomed images (crops) based on labels.

    Args:
        crop_size (int): The size of the square crop (width and height).
        path_source_raw (str): Path to the source directory containing 'images' and 'labels' folders.
        path_destination_raw (str): Path to the destination directory where output 'images' and 'labels' folders will be created.
        container_classes (list[int]): List of class IDs to consider for generating crops.
    """
    # Creation of source directories.
    images_input_dir = os.path.join(path_source_raw, "images")
    labels_input_dir = os.path.join(path_source_raw, "labels")

    # Creation of output directories.
    images_output_dir = os.path.join(path_destination_raw, "images")
    labels_output_dir = os.path.join(path_destination_raw, "labels")

    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    if not os.path.exists(labels_output_dir):
        os.makedirs(labels_output_dir)

    # Retrieving images
    image_files = [file for file in os.listdir(images_input_dir) if os.path.isfile(os.path.join(images_input_dir, file))]
    image_files = [os.path.join(images_input_dir, file) for file in image_files if not file.endswith('.json')]

    for img_path in image_files:
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]

        # Corresponding label path
        label_path = os.path.join(labels_input_dir, name_no_ext + '.txt')

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Unable to read {img_path}")
            continue

        h_img, w_img = img.shape[:2]

        # Check image size vs crop
        if w_img < crop_size or h_img < crop_size:
            continue

        # Read labels
        labels = load_yolo_obb_labels(label_path, w_img, h_img, container_classes)
        if not labels:
            continue

        # For each label (Target Label), generate a crop
        for idx, target_label in enumerate(labels):
            t_points = target_label['points']

            # Check target annotation size relative to crop
            t_min_x = np.min(t_points[:, 0])
            t_max_x = np.max(t_points[:, 0])
            t_min_y = np.min(t_points[:, 1])
            t_max_y = np.max(t_points[:, 1])

            t_w = t_max_x - t_min_x
            t_h = t_max_y - t_min_y

            # If annotation is larger than crop, skip this crop
            if t_w > crop_size or t_h > crop_size:
                continue

            # 1. Calculate the center of the target label
            cx, cy = np.mean(t_points, axis=0)

            # 2. Define crop coordinates (centered on cx, cy)
            x_min = int(cx - crop_size / 2)
            y_min = int(cy - crop_size / 2)
            x_max = x_min + crop_size
            y_max = y_min + crop_size

            # 3. Border rule (Clamping / Shifting)
            # Shift the window so it fits in the image without changing its size
            if x_min < 0:
                x_min = 0
                x_max = crop_size
            if y_min < 0:
                y_min = 0
                y_max = crop_size

            if x_max > w_img:
                x_max = w_img
                x_min = w_img - crop_size
            if y_max > h_img:
                y_max = h_img
                y_min = h_img - crop_size

            # Final safety (in case image is exactly crop size or rounding issues)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w_img, x_min + crop_size)
            y_max = min(h_img, y_min + crop_size)

            # Extract cropped image
            crop_img = img[y_min:y_max, x_min:x_max]

            # Check that crop has expected size (otherwise skip)
            if crop_img.shape[0] != crop_size or crop_img.shape[1] != crop_size:
                # This should not happen with the shift logic above, unless image < crop
                continue

            # 4. Process labels for this crop
            valid_labels_in_crop = []

            for lbl in labels:
                points = lbl['points'].copy()

                # Quick intersection check (to avoid including labels at the other end of the image)
                if not is_label_visible(points, x_min, y_min, crop_size, crop_size):
                    continue

                # Point transformation (WITHOUT clamping)
                new_points = []

                for pt in points:
                    # Coordinates relative to crop
                    nx = pt[0] - x_min
                    ny = pt[1] - y_min

                    # Keep coordinate as is, even if < 0 or > crop_size
                    new_points.append([nx, ny])

                new_points = np.array(new_points)

                # Normalization (0-1) relative to CROP size
                # Note: Values can be < 0 or > 1
                norm_points = []
                for pt in new_points:
                    nx_norm = pt[0] / crop_size
                    ny_norm = pt[1] / crop_size
                    norm_points.extend([nx_norm, ny_norm])

                # Add to final list
                # Format: class x1 y1 x2 y2 x3 y3 x4 y4
                label_line = f"{lbl['class_id']} " + " ".join([f"{v:.6f}" for v in norm_points])
                valid_labels_in_crop.append(label_line)

            # 5. Save
            if valid_labels_in_crop:
                # Unique filename: Original_IndexLabel
                base_filename = f"{name_no_ext}_{idx}_{crop_size}"

                save_img_path = os.path.join(images_output_dir, base_filename + ".jpg")
                save_txt_path = os.path.join(labels_output_dir, base_filename + ".txt")

                cv2.imwrite(save_img_path, crop_img)

                with open(save_txt_path, 'w') as f_out:
                    f_out.write("\n".join(valid_labels_in_crop))

def load_yolo_obb_labels(label_path: str, img_w: int, img_h: int, container_classes: list[int]) -> list[dict]:
    """
    Reads a YOLO OBB label file and returns a list of dictionaries.
    Line format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)

    Args:
        label_path (str): Path to the label file.
        img_w (int): Width of the image.
        img_h (int): Height of the image.
        container_classes (list[int]): List of class IDs to include.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'class_id': int
            - 'points': np.array([[x,y]...]) in absolute pixels.
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 9:
            continue

        class_id = int(parts[0])
        coords = parts[1:]

        if class_id not in container_classes:
            continue

        # Normalized -> absolute conversion
        points = []
        for i in range(0, 8, 2):
            x = coords[i] * img_w
            y = coords[i + 1] * img_h
            points.append([x, y])

        labels.append({
            'class_id': class_id,
            'points': np.array(points, dtype=np.float32)
        })
    return labels

def is_label_visible(points: np.ndarray, crop_x: int, crop_y: int, crop_w: int, crop_h: int) -> bool:
    """
    Roughly checks if a label has a chance to be in the crop.
    Checks if the label's bounding box intersects the crop.

    Args:
        points (np.ndarray): Array of points of shape (N, 2).
        crop_x (int): X coordinate of the crop top-left corner.
        crop_y (int): Y coordinate of the crop top-left corner.
        crop_w (int): Width of the crop.
        crop_h (int): Height of the crop.

    Returns:
        bool: True if the label is visible (intersects), False otherwise.
    """
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    crop_x_max = crop_x + crop_w
    crop_y_max = crop_y + crop_h

    # Checks if the label is in the crop.
    return (max_x > crop_x) and (min_x < crop_x_max) and (max_y > crop_y) and (min_y < crop_y_max)
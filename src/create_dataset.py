import os
from pathlib import Path
import shutil

import random
import yaml

def convert_images_labels_to_dataset(path_source_raw: str, path_destination_raw: str, valid_perc: int, test_perc: int, labels):
    """
    Organizes raw image and label files into a YOLO dataset structure.

    This function orchestrates the creation of a YOLO-compatible dataset by splitting the
    source data into training, validation, and test sets, creating the necessary
    directory structure, and generating the 'dataset.yaml' configuration file.

    Args:
        path_source_raw (str): The path to the source directory containing 'images' and 'labels' folders.
        path_destination_raw (str): The path to the root directory where the new dataset will be created.
        valid_perc (int): The percentage of the dataset to be allocated for validation.
        test_perc (int): The percentage of the dataset to be allocated for testing.
        labels (list): A list of strings representing the class names for the dataset.
    """
    path_source: Path = Path(path_source_raw)
    path_destination: Path = Path(path_destination_raw)

    # Create a unique root directory for the new dataset.
    path_dst_with_number = create_destination_directory(path_destination)
    # Create subdirectories for train, valid, and test sets.
    path_train_images_destination, path_train_labels_destination = create_destination_subdirectory(path_dst_with_number / "train")
    path_valid_images_destination, path_valid_labels_destination = create_destination_subdirectory(path_dst_with_number / "valid")
    path_test_images_destination, path_test_labels_destination = create_destination_subdirectory(path_dst_with_number / "test")

    # Define source paths for images and labels.
    images_path: Path = path_source / "images"
    labels_path: Path = path_source / "labels"
    # Get lists of image names for each dataset split.
    names_images_train, names_images_valid, names_images_test = get_train_images_names(images_path, valid_perc, test_perc)

    # Copy the files to their final destinations.
    copy_images_and_labels_to(names_images_train, path_train_images_destination, path_train_labels_destination, images_path, labels_path)
    copy_images_and_labels_to(names_images_valid, path_valid_images_destination, path_valid_labels_destination, images_path, labels_path)
    copy_images_and_labels_to(names_images_test, path_test_images_destination, path_test_labels_destination, images_path, labels_path)

    # Create the 'dataset.yaml' file required for YOLO training.
    create_dataset_yolo_yaml(path_source=path_dst_with_number, labels=labels)

def create_destination_directory(path_destination: Path) -> Path:
    """
    Creates a new directory for the dataset, ensuring its name is unique.

    If the target directory already exists, it appends a sequential number
    (e.g., "dataset_01", "dataset_02") to make it unique.

    Args:
        path_destination (Path): The desired path for the destination directory.

    Returns:
        Path: The path to the successfully created directory.
    """
    num_path_dst = 0
    # Keep the original name to append numbers to it if the path exists
    original_name = path_destination.name
    while os.path.exists(path_destination):
        num_path_dst += 1
        path_destination = path_destination.parent / f"{original_name}_{num_path_dst:02}"
    os.makedirs(path_destination)
    return path_destination


def create_destination_subdirectory(path_directory: Path):
    """
    Creates the "images" and "labels" subdirectories inside a given directory.

    These subdirectories are used to store the images and their corresponding
    labels for a specific dataset split (e.g., train, valid, test).

    Args:
        path_directory (Path): The path to the directory where the subdirectories
                               will be created (e.g., ".../train").

    Returns:
        tuple[Path, Path]: A tuple containing the paths to the created "images"
                           and "labels" subdirectories, respectively.
    """
    dst_images_path = path_directory / "images"
    dst_labels_path = path_directory / "labels"
    os.makedirs(path_directory)
    os.makedirs(dst_images_path)
    os.makedirs(dst_labels_path)
    return dst_images_path, dst_labels_path


def get_train_images_names(images_path: Path, perc_valid, perc_test):
    """
    Partitions a list of image names into training, validation, and test sets.

    The function shuffles the images randomly before splitting them based on the
    provided percentages.

    Args:
        images_path (Path): The path to the directory containing the source images.
        perc_valid (int): The percentage of images to allocate to the validation set.
        perc_test (int): The percentage of images to allocate to the test set.

    Returns:
        tuple[list, list, list]: A tuple containing three lists of image names for
                                 the training, validation, and test sets.
    """
    name_images = [file for file in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, file))]
    images_count = len(name_images)
    random.shuffle(name_images)
    perc_valid_val = int(perc_valid * images_count / 100)
    perc_test_val = int(perc_test * images_count / 100)
    name_images_test = name_images[0:perc_test_val]
    names_images_valid = name_images[perc_test_val: perc_test_val + perc_valid_val]
    name_images_train = name_images[perc_test_val + perc_valid_val: images_count]
    return name_images_train, names_images_valid, name_images_test


def copy_images_and_labels_to(images_names, path_images_destination, path_labels_destination, path_images_source, path_labels_source):
    """
    Copies image files and their corresponding label files to a destination directory.

    For each image name provided, it looks for the image file and a corresponding
    label file with a '.txt' extension. If both files exist, they are copied.

    Args:
        images_names (list): A list of image file names to be copied.
        path_images_destination (Path): The destination directory for the image files.
        path_labels_destination (Path): The destination directory for the label files.
        path_images_source (Path): The source directory of the images.
        path_labels_source (Path): The source directory of the labels.
    """
    for image_name_str in images_names:
        # Ignore non-image files that might be in the directory
        if image_name_str.endswith('.json'):
            continue

        image_path = path_images_source / image_name_str
        # Assumes the label file has the same name as the image, but with a .txt extension
        label_path = path_labels_source / Path(image_name_str).with_suffix('.txt').name

        # If an image doesn't have a corresponding label file, skip it.
        if image_path.is_file() and label_path.is_file():
            shutil.copy(image_path, path_images_destination)
            shutil.copy(label_path, path_labels_destination)


def create_dataset_yolo_yaml(path_source, labels):
    """
    Creates a 'dataset.yaml' file required by YOLO for training.

    This file contains metadata about the dataset, including the relative paths to the
    train, validation, and test sets, the number of classes, and their names.

    Args:
        path_source (Path): The root path of the newly created dataset. The yaml file
                            will be created here.
        labels (list[str]): A list of strings containing the names of the classes.
    """
    # Create dataset.yaml in the root of the new dataset directory
    path_yaml = path_source / "dataset.yaml"
    data = {'path': path_source.resolve().as_posix(),
            'train': "train/images",
            'val': "valid/images",
            'test': "test/images",
            'nc': len(labels),
            'names': labels
            }
    with open(path_yaml, 'w') as file:
        yaml.dump(data, file, sort_keys=False)

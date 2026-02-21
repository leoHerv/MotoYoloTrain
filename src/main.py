from create_dataset import convert_images_labels_to_dataset
from create_data_cache import create_plate_data_cache, create_number_data_cache, delete_data_cache

import os
import sys

from dotenv import load_dotenv
from pathlib import Path

if __name__ == '__main__':

    load_dotenv()

    # Path to /images and /labels folders.
    path_data: str = os.getenv('PATH_DATA_IMAGES_LABELS')
    # Path to the dataset directory.
    path_dataset: str = os.getenv('PATH_DATASET_DIR')

    if not Path(path_data).is_dir() or not Path(path_dataset).is_dir():
        sys.exit()

    # Datasets paths.
    path_dataset_plates: str = path_dataset + "/dataset_plates"
    path_dataset_numbers: str = path_dataset + "/dataset_numbers"

    print("--- Start plate dataset creation ---")

    # Creates images for the plate dataset.
    list_path_cache_plates: list[str] = create_plate_data_cache(1024, path_data, [11, 12], [1024])

    # Create the yolo plate dataset.
    convert_images_labels_to_dataset(list_path_cache_plates,
                                     path_dataset_plates, 25, 5,
                                     [{0: "front"}, {1: "side"}])
    delete_data_cache()

    print("--- Start number dataset creation ---")

    # Creates images for the number dataset.
    path_cache_numbers: str = create_number_data_cache(256, path_data, [11, 12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Create the yolo plate dataset.
    convert_images_labels_to_dataset([path_cache_numbers],
                                     path_dataset_numbers, 25, 5,
                                     [{0: "0"}, {1: "1"}, {2: "2"}, {3: "3"}, {4: "4"}, {5: "5"}, {6: "6"}, {7: "7"}, {8: "8"}, {9: "9"}, {10: "M"}])

    delete_data_cache()
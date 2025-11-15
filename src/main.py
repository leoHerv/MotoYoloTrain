from create_dataset import convert_images_labels_to_dataset
from create_area_dataset import get_areas_from_images

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
    # Path to the plates data directory.
    path_process_data: str = os.getenv('PATH_PROCESS_DATA_DIR')

    if not Path(path_data).is_dir() or not Path(path_dataset).is_dir() or not Path(path_process_data).is_dir():
        sys.exit()

    # Create the yolo dataset.
    convert_images_labels_to_dataset(path_data, path_dataset + "/dataset", 25,5,
                                     [{0: "0"}, {1: "1"}, {2: "2"}, {3: "3"}, {4: "4"}, {5: "5"}, {6: "6"},
                                      {7: "7"}, {8: "8"}, {9: "9"}, {10: "front"}, {11: "side"}])

    process_count, crop_count = get_areas_from_images(path_data,
                          path_process_data + "/dataset_plates",
                          [10, 11], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    print(f"Images processed: {process_count}, crop count: {crop_count}")


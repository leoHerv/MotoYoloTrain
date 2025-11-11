from create_dataset import convert_images_labels_to_dataset

import os
import sys

from dotenv import load_dotenv
from pathlib import Path

if __name__ == '__main__':

    load_dotenv()

    # Path to /images and /labels folders.
    path_data: Path = Path(os.getenv('PATH_DATA_IMAGES_LABELS'))
    # Path to the dataset directory.
    path_dataset: str = os.getenv('PATH_DATASET')

    if not path_data.is_dir():
        sys.exit()

    # Create the yolo dataset.
    convert_images_labels_to_dataset(path_data.__str__(), path_dataset, 25,5,
                                     [{0: "0"}, {1: "1"}, {2: "2"}, {3: "3"}, {4: "4"}, {5: "5"}, {6: "6"},
                                      {7: "7"}, {8: "8"}, {9: "9"}, {10: "front"}, {11: "side"}])


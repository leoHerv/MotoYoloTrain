from create_training_images import convert_images_to_squares, process_labels_directory
from create_zoom_data import create_zoom_images
from create_area_data import get_areas_from_images

import os
import tempfile
import shutil

from pathlib import Path

CACHE_DIRECTORY: str = "MotoYoloTrain_cache"

def create_plate_data_cache(image_size: int, path_source_raw: str, container_classes: list[int], zooms: list[int]) -> list[str]:
    """

    :param image_size: (int)
    :param path_source_raw: (str)
    :param container_classes: (list[int])
    :param zooms:
    :return: (list[str])
    """

    path_cache: Path = Path(tempfile.gettempdir()) / CACHE_DIRECTORY
    os.makedirs(path_cache, exist_ok=True)

    # Converts all raw images in the right size and square them.
    path_full_cache: Path = path_cache / "f"
    convert_images_to_squares(path_source_raw, str(path_full_cache), image_size, container_classes)
    process_labels_directory(str(path_full_cache / "labels"), container_classes)

    list_data_cache: list[str] = [str(path_full_cache)]  # Result paths.

    for zoom in zooms:
        # Zoom in images based on container labels with a certain zoom level.
        path_zoom_cache: Path = path_cache / ("c_" + str(zoom))
        create_zoom_images(zoom, path_source_raw, str(path_zoom_cache), container_classes)
        process_labels_directory(str(path_zoom_cache / "labels"), container_classes)

        # If the zoom images are too large, we resize them.
        if zoom != image_size:
            path_zoom_resize_cache: str = str(path_cache / ("c_" + str(zoom) + "_" + str(image_size)))
            convert_images_to_squares(str(path_zoom_cache), path_zoom_resize_cache, image_size, container_classes)
            list_data_cache.append(path_zoom_resize_cache)
            # Deletes necessary images.
            shutil.rmtree(path_zoom_cache)
        else:
            list_data_cache.append(str(path_zoom_cache))

    return list_data_cache

def create_number_data_cache(image_size: int, path_source_raw: str, container_classes: list[int], in_container_classes: list[int]) -> str:
    """

    :param image_size: (int)
    :param path_source_raw: (str)
    :param container_classes: (list[int])
    :param in_container_classes:
    :return: (str)
    """

    path_cache: Path = Path(tempfile.gettempdir()) / CACHE_DIRECTORY
    os.makedirs(path_cache, exist_ok=True)

    # Converts all raw images in containers images and resize them.
    path_full_cache: Path = path_cache / "n"

    get_areas_from_images(path_source_raw, str(path_full_cache), container_classes, in_container_classes)

    convert_images_to_squares(str(path_full_cache), str(path_full_cache), image_size, in_container_classes)

    return str(path_full_cache)

def delete_data_cache() -> None:
    """
    Deletes the cache directory.
    """
    shutil.rmtree(Path(tempfile.gettempdir()) / CACHE_DIRECTORY)
import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    return list_files(basePath, valid_extensions=image_types, contains=contains)


def list_files(base_path, valid_extensions=None, contains=None):
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            file_extension = filename[filename.rfind("."):].lower()

            if valid_extensions is None or file_extension.endswith(valid_extensions):
                # construct the path to the image and yield it
                image_path = os.path.join(rootDir, filename)
                yield image_path

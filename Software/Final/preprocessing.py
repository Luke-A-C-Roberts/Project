from cv2 import blur, imread, imwrite, threshold, THRESH_TOZERO
from cv2.typing import MatLike

from datetime import timedelta
from functools import reduce, partial
from multiprocessing import Pool
from os import listdir, mkdir
from sys import argv
from typing import Callable
from time import time

from utils import compose


DOWNLOADS_PATH      : str   = "/home/luke/Downloads/"
ZENODO_IMAGES_FOLDER: str   = DOWNLOADS_PATH + "images_gz2/images/"
KAGGLE_IMAGES_FOLDER: str   = DOWNLOADS_PATH + "images_gz2/kaggle/"
EDITED_IMAGES_FOLDER: str   = DOWNLOADS_PATH + "images_gz2/preprocessed/"
THRESH_MIN          : float = 10.
THRESH_MAX          : float = 255.
CROP_WIDTH          : int   = 224
CROP_HEIGHT         : int   = 224
BLUR_KERNEL_DIMS    : tuple[int, int] = (5, 5)


# simple blur to reduce noise
def blur_image(image: MatLike | None) -> MatLike | None:
    if image is None: return None
    return blur(image, BLUR_KERNEL_DIMS)


# threshold removes noise from the image background
def threshold_image(image: MatLike | None) -> MatLike | None:
    if image is None: return None
    return threshold(image, THRESH_MIN, THRESH_MAX, THRESH_TOZERO, None)[1]


# gets the image into the correct shape
def crop_image(image: MatLike | None) -> MatLike | None:
    if image is None: return None
    
    dimentions: tuple[int, int, int] = image.shape
    (width, height, _) = dimentions

    x_offset: int = (width  - CROP_WIDTH)  // 2
    y_offset: int = (height - CROP_HEIGHT) // 2

    if width < CROP_WIDTH and height < CROP_HEIGHT:
        return None

    x_slice = slice(x_offset, width  - x_offset)
    y_slice = slice(y_offset, height - y_offset)
    return image[x_slice, y_slice]


def remove_3_pixel_edge(image: MatLike | None) -> MatLike | None:
    if image is None: return None
    
    dimentions: tuple[int, int, int] = image.shape
    (width, height, _) = dimentions
    
    if width < CROP_WIDTH and height < CROP_HEIGHT:
        return None

    x_slice = slice(3, width)
    y_slice = slice(3, width)

    return image[x_slice, y_slice]


def jpg_to_png(file_name: str) -> str:
    return file_name.replace(".jpg", ".png")


# reads and performs all preprocessing in call order
preprocessing_zenodo: Callable[[MatLike], None | MatLike] =\
compose(
    threshold_image,
    blur_image,
    crop_image
)


preprocessing_kaggle: Callable[[MatLike], None | MatLike] =\
compose(
    threshold_image,
    blur_image,
    remove_3_pixel_edge
)


# reads preprocesses and writes to new file from the images folder
def preprocess_image_zenodo(index: int, file_name: str) -> bool:
    if not index % 500: print(index)
    image: MatLike = preprocessing_zenodo(imread(ZENODO_IMAGES_FOLDER + file_name))
    if image is None: return False
    imwrite(EDITED_IMAGES_FOLDER + jpg_to_png(file_name), image)
    return True


# reads preprocesses and writes to new file from the kaggle folder
def preprocess_image_kaggle(index: int, file_name: str) -> bool:
    if not index % 500: print(index)
    image: MatLike = preprocessing_kaggle(imread(KAGGLE_IMAGES_FOLDER + file_name))
    if image is None: return False
    imwrite(EDITED_IMAGES_FOLDER + jpg_to_png(file_name), image)
    return True


def preprocess_all(preprocessing: Callable[[MatLike], None | MatLike], path: str) -> None:
    # if the folder for preprocessing doesn't exist, make it.
    try: mkdir(EDITED_IMAGES_FOLDER)
    except FileExistsError as e: pass 

    file_names: list[str] = listdir(path)
    num_files: int = len(file_names)

    results: list[bool] = []
    
    with Pool(10) as pool:
        results = pool.starmap(preprocessing, enumerate(file_names))

    results: int = sum(results)
    
    print(
          f"{results}/{num_files} ({round(results/num_files*100, ndigits=5)}%) "
        + f"successfully preprocessed"
    )
    

preprocess_all_zenodo = partial(
    preprocess_all,
    preprocessing=preprocess_image_zenodo,
    path=ZENODO_IMAGES_FOLDER
)

preprocess_all_kaggle = partial(
    preprocess_all,
    preprocessing=preprocess_image_kaggle,
    path=KAGGLE_IMAGES_FOLDER
)


def main() -> None:
    if "-k" in argv:
        preprocess_all_kaggle()
        return

    preprocess_all_zenodo()


main()

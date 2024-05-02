from cv2 import blur, imread, imwrite, threshold, THRESH_TOZERO
from cv2.typing import MatLike

from datetime import timedelta
from functools import reduce
from multiprocessing import Pool
from os import listdir, mkdir
from typing import Callable
from time import time


DOWNLOADS_PATH      : str   = "/home/luke/Downloads/"  # "/mnt/c/Users/Computing/Downloads/"
ZENODO_IMAGES_FOLDER: str   = DOWNLOADS_PATH + "images_gz2/images/"
EDITED_IMAGES_FOLDER: str   = DOWNLOADS_PATH + "images_gz2/preprocessed/"
THRESH_MIN          : float = 10.
THRESH_MAX          : float = 255.
CROP_WIDTH          : int   = 244
CROP_HEIGHT         : int   = 244
BLUR_KERNEL_DIMS    : tuple[int, int] = (5, 5)


def compose(*funcs: Callable) -> Callable:
    return lambda *args, **kwargs: reduce(lambda v, f: f(v), funcs[:-1], funcs[-1](*args, **kwargs))


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


def jpg_to_png(file_name: str) -> str:
    return file_name.replace(".jpg", ".png")


# reads and performs all preprocessing in call order
preprocessing: Callable[[MatLike], None | MatLike] = compose(
    threshold_image,
    blur_image,
    crop_image,
)


# reads preprocesses and writes to new file
def preprocess_image(index: int, file_name: str) -> bool:
    if not index % 500: print(index)
    image: MatLike = preprocessing(imread(ZENODO_IMAGES_FOLDER + file_name))
    if image is None: return False
    imwrite(EDITED_IMAGES_FOLDER + jpg_to_png(file_name), image)
    return True


def preprocess_all() -> None:    
    # if the folder for preprocessing doesn't exist, make it.
    try: mkdir(EDITED_IMAGES_FOLDER)
    except FileExistsError as e: pass 

    file_names: list[str] = listdir(ZENODO_IMAGES_FOLDER)
    num_files: int = len(file_names)

    results: list[bool] = []
    with Pool(10) as pool:
        results = pool.starmap(preprocess_image, enumerate(file_names))

    results: int = sum(results)
    
    print(
          f"{results}/{num_files} ({round(results/num_files*100, ndigits=5)}%) "
        + f"successfully preprocessed"
    )
        
preprocess_all()

from astropy.io.fits import open as fits_open, HDUList
from astropy.visualization import astropy_mpl_style
from os import listdir
from typing import Any


# https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python#44712152
def show_fits(path: str, ax: Any) -> None:
    with fits_open(path) as fits_file:
        arr: HDUList = fits_file[0].data
    ax.imshow(arr)


import matplotlib.pyplot as plt

f, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
plt.style.use(astropy_mpl_style)

for file, ax in zip(listdir("fits_images"), axs.flat):
    show_fits(f"fits_images/{file}", ax)

plt.show()

from astroquery.ipac.ned import Ned
from pandas import read_csv, Series
from os import path, makedirs
from astroquery.utils import commons

CSV_PATH: str = "./NED_list.csv"
IMAGE_DIR_PATH: str = "./fits_images"

ned = Ned()

galaxy_names: Series = read_csv(CSV_PATH)["Name"][
    14542:14554
].unique()  # remove [] for whole image collection
names_with_images: list[str] = []

# https://stackoverflow.com/questions/1274405/how-to-create-new-folder#1274465
if not path.exists(IMAGE_DIR_PATH):
    makedirs(IMAGE_DIR_PATH)

for name in galaxy_names:
    image_list = ned.get_image_list(name)
    if image_list == []:
        continue
    names_with_images.append(name)
    request = commons.FileContainer(
        image_list[0], encoding="binary", show_progress=False
    ).get_fits()
    path = f"{IMAGE_DIR_PATH}/{name}.fits"
    request.writeto(path, overwrite=True)

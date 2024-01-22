import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.visualization import astropy_mpl_style
from astroquery.ipac.ned import Ned, Conf, NedClass

ned = Ned()
# Conf.server = "ned.ipac.caltech.edu/cgi-bin/NEDspectra"
Conf.server = "ned.ipac.caltech.edu/cgi-bin/OBJatt"

table: pd.DataFrame = ned.get_image_table
print(table)
# ------------------------------
# Import the needed libraries
# ------------------------------

# Import the PAD python library
# The PAD python library file (PY_smoothing_on_sphere_library.py) as well as the PAD C++ shared library file (smoothing_on_sphere_Cxx_shared_library.so) 
# need to be in the same folder as the script using them
from PY_CSSS_library  import *

# Import netcdf library 
from netCDF4 import Dataset
import numpy as np

# assert(0)
# ------------------------------------------------------------------------------------------------------------------------
# convert data to lists of lats, lons, and field values for all grid points 
# ------------------------------------------------------------------------------------------------------------------------
resolution_degrees = 3

ln = np.arange(0,359.9,resolution_degrees)      # [0 4 8 ... 360]
if resolution_degrees == 1:
    lt = np.arange(-89.5,89.6,resolution_degrees)   # [-90 -86 ... 90]
elif resolution_degrees == 3:
    lt = np.arange(-88.5,88.6,resolution_degrees)

lon = np.tile(ln,lt.shape)
lat = np.repeat(lt,ln.shape)

# ------------------------------------------------------------------------------------------------------------------------
# Calculate area size data
# ------------------------------------------------------------------------------------------------------------------------
Earth_radius= 6371.0*1000.0
dlat = 3.0
area_size =  np.deg2rad(dlat)*Earth_radius*np.deg2rad(dlat)*Earth_radius*np.cos(np.deg2rad(lat))
area_size [area_size < 0] = 0  # fix small negative values that occur at the poles due to the float rounding error

# ------------------------------------------------------------------------------------------------------------------------
# convert all arrays to np.float64
# ------------------------------------------------------------------------------------------------------------------------
lon = np.ascontiguousarray(lon, dtype = np.float64)
lat = np.ascontiguousarray(lat, dtype = np.float64)
area_size = np.ascontiguousarray(area_size, dtype = np.float64)

# ------------------------------------------------------------------------------------------------------------------------
# Generate the smoothing data for smoothing kernel radiuses of 100 and 200 km and write it to the disk
# ------------------------------------------------------------------------------------------------------------------------

smoothing_kernel_radius_in_metres = [500*1000, 100*1000, 50*1000, 1000*1000, 5000*1000]

# set the output folder for the smoothing data files
smoothing_data_folder = bytes("smoothing_data/", encoding='utf8')

# create the folder if it does not exist
os.makedirs(smoothing_data_folder, exist_ok = True)

# Generate the smoothing data and write it to the disk
generate_smoothing_data_for_the_overlap_detection_based_approach_and_write_it_to_the_disk(lat, lon, smoothing_kernel_radius_in_metres, smoothing_data_folder)

np.save('area_size.npy', area_size)
print('Area size data saved to area_size.npy')
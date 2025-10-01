# ------------------------------
# Import the needed libraries
# ------------------------------

# Import the PAD python library
# The PAD python library file (PY_smoothing_on_sphere_library.py) as well as the PAD C++ shared library file (smoothing_on_sphere_Cxx_shared_library.so)
# need to be in the same folder as the script using them
from PY_CSSS_library import *

# Import netcdf library
from netCDF4 import Dataset

# ------------------------------------------------------------------------------------------------------------------------
# Read the sample precipiation field that is defined in two-dimenstions on a regular 0.25deg lat/long grid  -----------------------------------------------------------------------------------------------------------------------

nc_file_id = Dataset("../../PY_CSSS_example_field1.nc", "r")
lon_netcdf = nc_file_id.variables["lon"][:].data
lat_netcdf = nc_file_id.variables["lat"][:].data
f1_netcdf = nc_file_id.variables["precipitation"][:].data
f1_netcdf[f1_netcdf < 0] = 0
nc_file_id.close()

nc_file_id = Dataset("../../PY_CSSS_example_field2.nc", "r")
f2_netcdf = nc_file_id.variables["precipitation"][:].data
f2_netcdf[f2_netcdf < 0] = 0
nc_file_id.close()

# ------------------------------------------------------------------------------------------------------------------------
# convert data to lists of lats, lons, and field values for all grid points
# ------------------------------------------------------------------------------------------------------------------------

# Value 1 is the prediction
values1 = np.reshape(f1_netcdf, (-1))

# Perturb the batch outputs
values2_list = []
for k in range(5):
    values2_list.append(np.reshape(np.roll(f2_netcdf, k * 5, axis=1), (-1)))


lon = np.tile(lon_netcdf, f1_netcdf.shape[0])
lat = np.repeat(lat_netcdf, f1_netcdf.shape[1])

# ------------------------------------------------------------------------------------------------------------------------
# Calculate area size data
# ------------------------------------------------------------------------------------------------------------------------
Earth_radius = 6371.0 * 1000.0
dlat = 1.0  # resolution of lat/lon grid of the input field
area_size = (
    np.deg2rad(dlat)
    * Earth_radius
    * np.deg2rad(dlat)
    * Earth_radius
    * np.cos(np.deg2rad(lat))
)
area_size[area_size < 0] = (
    0  # fix small negative values that occur at the poles due to the float rounding error
)

# ------------------------------------------------------------------------------------------------------------------------
# convert all arrays to np.float64
# ------------------------------------------------------------------------------------------------------------------------
lon = np.ascontiguousarray(lon, dtype=np.float64)
lat = np.ascontiguousarray(lat, dtype=np.float64)
values1 = np.ascontiguousarray(values1, dtype=np.float64)

for k in range(len(values2_list)):
    values2_list[k] = np.ascontiguousarray(values2_list[k], dtype=np.float64)

area_size = np.ascontiguousarray(area_size, dtype=np.float64)

# ------------------------------------------------------------------------------------------------------------------------
# Generate the smoothing data for smoothing kernel radiuses of 500 km and write it to the disk
# ------------------------------------------------------------------------------------------------------------------------

smoothing_kernel_radius_in_metres = 500 * 1000

# set the output folder for the smoothing data files
smoothing_data_folder = bytes("smoothing_data_kdtree/", encoding="utf8")

# create the folder if it does not exist
os.makedirs(smoothing_data_folder, exist_ok=True)

# Generate the smoothing data and write it to the disk
generate_smoothing_data_for_the_kdtree_based_approach_and_write_it_to_the_disk(
    lat, lon, smoothing_kernel_radius_in_metres, smoothing_data_folder
)

# ------------------------------------------------------------------------------------------------------------------------
# Use the smoothing data to calculate the smoothed field for the 500 km smoothing kernel radius
# ------------------------------------------------------------------------------------------------------------------------

# read the smoothing data from the disk into the memory
smoothing_data_pointer = (
    read_smoothing_data_for_the_kdtree_based_approach_from_binary_file_ctypes(
        smoothing_data_folder, smoothing_kernel_radius_in_metres
    )
)

# calculate the smoothed field
values1_smoothed = smooth_field_using_smoothing_data_for_the_kdtree_approach(
    area_size, values1, smoothing_data_pointer
)

values2_smoothed_list = []
for k in range(len(values2_list)):
    values2 = values2_list[k]
    values2_smoothed = smooth_field_using_smoothing_data_for_the_kdtree_approach(
        area_size, values2, smoothing_data_pointer
    )
    values2_smoothed_list.append(values2_smoothed)

# calulate CSSS_p for p = 1.5
p = 2

# concatenate all values2 into a single array for batch processing
values2_batch = np.stack(values2_list, axis=0)
values1_batch = np.tile(values1.reshape(1, -1), (len(values2_list), 1))

assert values1_batch.shape == values2_batch.shape

[BATCH_CSSS_p_value, BATCH_CSSS_p_gradient] = (
    batch_calculate_CSSS2_value_with_gradient_kdtree(
        values1_batch, values2_batch, area_size, smoothing_data_pointer
    )
)

# Transform to [0, infty) loss and appropriate gradient
BATCH_CSSS_p_value = 1.0 - BATCH_CSSS_p_value
BATCH_CSSS_p_gradient = -BATCH_CSSS_p_gradient

print("BATCH_CSSS_p_value", BATCH_CSSS_p_value)
print("BATCH_CSSS_p_gradient", BATCH_CSSS_p_gradient.shape)

# free the smoothing data memory
free_smoothing_data_memory_kdtree(smoothing_data_pointer)
smoothing_data_pointer = None

# Import matplotlib library
import matplotlib
import matplotlib.pyplot
import matplotlib.ticker as mticker

# Import cartopy library
import cartopy

for k in range(len(values2_list)):

    central_latitude = 0
    central_longitude = 180

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=cartopy.crs.Orthographic(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            globe=None,
        ),
    )
    ax.set_global()
    matplotlib.pyplot.title("Original fields")
    cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rb_cmap", ["white", (0.7, 0.7, 1.0), "blue"], 512
    )
    cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rb_cmap", ["white", (1.0, 0.7, 0.7), "red"], 512
    )
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=100)
    fax = cmap_b(norm(f1_netcdf))
    fbx = cmap_r(norm(np.roll(f2_netcdf, k * 5, axis=1)))
    fx = fax * fbx
    img = matplotlib.pyplot.imshow(
        fx,
        transform=cartopy.crs.PlateCarree(),
        interpolation="nearest",
        origin="lower",
        extent=(0, 360, -90, 90),
    )
    ax.coastlines(resolution="110m", color="grey", linestyle="-", alpha=1)
    gl = ax.gridlines()
    gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 20)))
    # matplotlib.pyplot.show()
    matplotlib.pyplot.savefig(f"figs/input_data_B{k}.png")
    matplotlib.pyplot.close()

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=cartopy.crs.Orthographic(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            globe=None,
        ),
    )
    ax.set_global()
    matplotlib.pyplot.title("Smoothed fields")
    cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rb_cmap", ["white", (0.7, 0.7, 1.0), "blue"], 512
    )
    cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rb_cmap", ["white", (1.0, 0.7, 0.7), "red"], 512
    )
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=100)
    fax = cmap_b(norm(np.reshape(values1_smoothed, f1_netcdf.shape)))
    fbx = cmap_r(norm(np.reshape(values2_smoothed_list[k], f1_netcdf.shape)))
    fx = fax * fbx
    img = matplotlib.pyplot.imshow(
        fx,
        transform=cartopy.crs.PlateCarree(),
        interpolation="nearest",
        origin="lower",
        extent=(0, 360, -90, 90),
    )
    ax.coastlines(resolution="110m", color="grey", linestyle="-", alpha=1)
    gl = ax.gridlines()
    gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 20)))
    matplotlib.pyplot.savefig(
        f"figs/smoothed_data_{smoothing_kernel_radius_in_metres}_b{k}.png"
    )
    # matplotlib.pyplot.show()
    matplotlib.pyplot.close()

    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=cartopy.crs.Orthographic(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            globe=None,
        ),
    )
    ax.set_global()
    matplotlib.pyplot.title("CSSS_p gradient")
    img = matplotlib.pyplot.imshow(
        np.reshape(BATCH_CSSS_p_gradient[k, :], f1_netcdf.shape),
        transform=cartopy.crs.PlateCarree(),
        interpolation="nearest",
        origin="lower",
        extent=(0, 360, -90, 90),
        vmax=np.max(np.abs(BATCH_CSSS_p_gradient[k, :])),
        vmin=-np.max(np.abs(BATCH_CSSS_p_gradient[k, :])),
        cmap="bwr",
    )
    cb = fig.colorbar(img, extend="both", shrink=0.5)
    ax.coastlines(resolution="110m", color="grey", linestyle="-", alpha=1)
    gl = ax.gridlines()
    gl.xlocator = mticker.FixedLocator(list(np.arange(-180, 180, 20)))
    matplotlib.pyplot.savefig(f"figs/csss_grad_{smoothing_kernel_radius_in_metres}_b{k}.png")
    # matplotlib.pyplot.show()
    matplotlib.pyplot.close()

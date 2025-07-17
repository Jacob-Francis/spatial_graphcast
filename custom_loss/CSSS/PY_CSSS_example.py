# ------------------------------
# Import the needed libraries
# ------------------------------

# Import the PAD python library
# The PAD python library file (PY_smoothing_on_sphere_library.py) as well as the PAD C++ shared library file (smoothing_on_sphere_Cxx_shared_library.so) 
# need to be in the same folder as the script using them
from PY_CSSS_library  import *

# Import netcdf library 
from netCDF4 import Dataset

# ------------------------------------------------------------------------------------------------------------------------
# Read the sample precipiation field that is defined in two-dimenstions on a regular 0.25deg lat/long grid  -----------------------------------------------------------------------------------------------------------------------

nc_file_id = Dataset("../../PY_CSSS_example_field1.nc", 'r') 
lon_netcdf = nc_file_id.variables["lon"][:].data
lat_netcdf = nc_file_id.variables["lat"][:].data
f1_netcdf = nc_file_id.variables["precipitation"][:].data
f1_netcdf[f1_netcdf < 0] = 0
nc_file_id.close()

nc_file_id = Dataset("../../PY_CSSS_example_field2.nc", 'r') 
f2_netcdf = nc_file_id.variables["precipitation"][:].data
f2_netcdf[f2_netcdf < 0] = 0
nc_file_id.close()

# assert(0)
# ------------------------------------------------------------------------------------------------------------------------
# convert data to lists of lats, lons, and field values for all grid points 
# ------------------------------------------------------------------------------------------------------------------------

values1 =np.reshape(f1_netcdf,(-1))
values2 =np.reshape(f2_netcdf,(-1))
lon = np.tile(lon_netcdf,f1_netcdf.shape[0])
lat = np.repeat(lat_netcdf,f1_netcdf.shape[1])

# ------------------------------------------------------------------------------------------------------------------------
# Calculate area size data
# ------------------------------------------------------------------------------------------------------------------------
Earth_radius= 6371.0*1000.0
dlat = 1.0 # resolution of lat/lon grid of the input field
area_size =  np.deg2rad(dlat)*Earth_radius*np.deg2rad(dlat)*Earth_radius*np.cos(np.deg2rad(lat))
area_size [area_size < 0] = 0  # fix small negative values that occur at the poles due to the float rounding error

# ------------------------------------------------------------------------------------------------------------------------
# convert all arrays to np.float64
# ------------------------------------------------------------------------------------------------------------------------
lon = np.ascontiguousarray(lon, dtype = np.float64)
lat = np.ascontiguousarray(lat, dtype = np.float64)
values1 = np.ascontiguousarray(values1, dtype = np.float64)
values2 = np.ascontiguousarray(values2, dtype = np.float64)
area_size = np.ascontiguousarray(area_size, dtype = np.float64)


# ------------------------------------------------------------------------------------------------------------------------
# Generate the smoothing data for smoothing kernel radiuses of 100 and 200 km and write it to the disk
# ------------------------------------------------------------------------------------------------------------------------

#smoothing_kernel_radius_in_metres = [100*1000, 200*1000, 500*1000, 1000*1000, 2000*1000]
#smoothing_kernel_radius_in_metres = [1*1000, 100*1000, 200*1000, 500*1000, 1000*1000, 2000*1000]
smoothing_kernel_radius_in_metres = [100*1000, 500*1000]

# set the output folder for the smoothing data files
smoothing_data_folder = bytes("smoothing_data/", encoding='utf8')

# create the folder if it does not exist
os.makedirs(smoothing_data_folder, exist_ok = True)

# Generate the smoothing data and write it to the disk
# generate_smoothing_data_for_the_overlap_detection_based_approach_and_write_it_to_the_disk(lat, lon, smoothing_kernel_radius_in_metres, smoothing_data_folder)
# assert 0
# ------------------------------------------------------------------------------------------------------------------------
# Use the smoothing data to calculate the smoothed field for the 500 km smoothing kernel radius
# ------------------------------------------------------------------------------------------------------------------------

# read the smoothing data from the disk into the memory
smoothing_data_pointer = read_smoothing_data_from_binary_file(smoothing_data_folder, 500*1000, len(lat))

# calculate the smoothed field
values1_smoothed = smooth_field_using_overlap_detection(area_size, values1, smoothing_data_pointer)
values2_smoothed = smooth_field_using_overlap_detection(area_size, values2, smoothing_data_pointer)

# -----------------------------------------------------------
#            Single
# -----------------------------------------------------------

# [CSSS_value, CSSS_gradient] = calculate_CSSS2_value_with_gradient(values1, values2, area_size, smoothing_data_pointer)

# # free the smoothing data memory
# free_smoothing_data_memory(smoothing_data_pointer, len(lat))

# print(CSSS_value)

# # Import matplotlib library 
# import matplotlib
# matplotlib.use("Qt5Agg")  # if you have a display server running
# # import matplotlib.pyplot as plt
# import matplotlib.pyplot
# import matplotlib.ticker as mticker
# # Import cartopy library 
# import cartopy

# central_latitude = 0
# central_longitude = 180

# fig = matplotlib.pyplot.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude, globe=None))
# ax.set_global()
# matplotlib.pyplot.title("Original fields")
# cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(0.7, 0.7, 1.0), "blue"],512)
# cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(1.0, 0.7, 0.7), "red"],512)
# norm = matplotlib.colors.LogNorm(vmin=1, vmax=100)
# fax = cmap_b(norm(f1_netcdf))
# fbx = cmap_r(norm(f2_netcdf))
# fx = fax*fbx
# img = matplotlib.pyplot.imshow(fx, transform=cartopy.crs.PlateCarree(), interpolation='nearest', origin='lower', extent=(0, 360, -90, 90))
# ax.coastlines(resolution='110m', color='grey', linestyle='-', alpha=1)
# gl = ax.gridlines()
# gl.xlocator = mticker.FixedLocator(list(np.arange(-180,180,20)))
# matplotlib.pyplot.savefig("aaa1.png", dpi=300, bbox_inches='tight')
# matplotlib.pyplot.show()
# matplotlib.pyplot.close()


# fig = matplotlib.pyplot.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude, globe=None))
# ax.set_global()
# matplotlib.pyplot.title("Smoothed fields")
# cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(0.7, 0.7, 1.0), "blue"],512)
# cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(1.0, 0.7, 0.7), "red"],512)
# norm = matplotlib.colors.LogNorm(vmin=1, vmax=100)
# fax = cmap_b(norm(np.reshape(values1_smoothed,f1_netcdf.shape)))
# fbx = cmap_r(norm(np.reshape(values2_smoothed,f1_netcdf.shape)))
# fx = fax*fbx
# img = matplotlib.pyplot.imshow(fx, transform=cartopy.crs.PlateCarree(), interpolation='nearest', origin='lower', extent=(0, 360, -90, 90))
# ax.coastlines(resolution='110m', color='grey', linestyle='-', alpha=1)
# gl = ax.gridlines()
# gl.xlocator = mticker.FixedLocator(list(np.arange(-180,180,20)))
# matplotlib.pyplot.savefig("aaa2.png", dpi=300, bbox_inches='tight')
# matplotlib.pyplot.show()
# matplotlib.pyplot.close()


# fig = matplotlib.pyplot.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude, globe=None))
# ax.set_global()
# matplotlib.pyplot.title("CSSS gradient")
# img = matplotlib.pyplot.imshow(np.reshape(CSSS_gradient,f1_netcdf.shape), transform=cartopy.crs.PlateCarree(), interpolation='nearest', origin='lower', extent=(0, 360, -90, 90),  vmax= np.max(np.abs(CSSS_gradient)), vmin=-np.max(np.abs(CSSS_gradient)), cmap ="bwr")
# #cb = fig.colorbar(img, extend='both', shrink=0.5)
# ax.coastlines(resolution='110m', color='grey', linestyle='-', alpha=1)
# gl = ax.gridlines()
# gl.xlocator = mticker.FixedLocator(list(np.arange(-180,180,20)))
# matplotlib.pyplot.savefig("aaa3.png", dpi=300, bbox_inches='tight')
# matplotlib.pyplot.show()
# matplotlib.pyplot.close()
# assert 0

# -----------------------------------------------------------
#            Batch
# -----------------------------------------------------------
print(values1.shape)

bvalues1 = np.tile(values1[np.newaxis, :], (5, 1))
bvalues2 = np.tile(values2[np.newaxis, :], (5, 1))
barea_size = np.tile(area_size[np.newaxis, :], (5, 1))

print('Shape', bvalues1.shape, bvalues2.shape, area_size.shape)

[bCSSS_value, bCSSS_gradient] = batch_calculate_CSSS2_value_with_gradient(bvalues1, bvalues2, area_size, smoothing_data_pointer)

print(bCSSS_value.shape, bCSSS_gradient.shape)
print('Batched CSSS values', bCSSS_value)

assert 0 

# free the smoothing data memory
free_smoothing_data_memory(smoothing_data_pointer, len(lat))

# Import matplotlib library 
import matplotlib
matplotlib.use("Qt5Agg")  # if you have a display server running
# import matplotlib.pyplot as plt
import matplotlib.pyplot
import matplotlib.ticker as mticker
# Import cartopy library 
import cartopy

central_latitude = 0
central_longitude = 180

# fig = matplotlib.pyplot.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude, globe=None))
# ax.set_global()
# matplotlib.pyplot.title("Original fields")
# cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(0.7, 0.7, 1.0), "blue"],512)
# cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(1.0, 0.7, 0.7), "red"],512)
# norm = matplotlib.colors.LogNorm(vmin=1, vmax=100)
# fax = cmap_b(norm(f1_netcdf))
# fbx = cmap_r(norm(f2_netcdf))
# fx = fax*fbx
# img = matplotlib.pyplot.imshow(fx, transform=cartopy.crs.PlateCarree(), interpolation='nearest', origin='lower', extent=(0, 360, -90, 90))
# ax.coastlines(resolution='110m', color='grey', linestyle='-', alpha=1)
# gl = ax.gridlines()
# gl.xlocator = mticker.FixedLocator(list(np.arange(-180,180,20)))
# matplotlib.pyplot.savefig("baaa1.png", dpi=300, bbox_inches='tight')
# matplotlib.pyplot.show()
# matplotlib.pyplot.close()


# fig = matplotlib.pyplot.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude, globe=None))
# ax.set_global()
# matplotlib.pyplot.title("Smoothed fields")
# cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(0.7, 0.7, 1.0), "blue"],512)
# cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(1.0, 0.7, 0.7), "red"],512)
# norm = matplotlib.colors.LogNorm(vmin=1, vmax=100)
# fax = cmap_b(norm(np.reshape(values1_smoothed,f1_netcdf.shape)))
# fbx = cmap_r(norm(np.reshape(values2_smoothed,f1_netcdf.shape)))
# fx = fax*fbx
# img = matplotlib.pyplot.imshow(fx, transform=cartopy.crs.PlateCarree(), interpolation='nearest', origin='lower', extent=(0, 360, -90, 90))
# ax.coastlines(resolution='110m', color='grey', linestyle='-', alpha=1)
# gl = ax.gridlines()
# gl.xlocator = mticker.FixedLocator(list(np.arange(-180,180,20)))
# matplotlib.pyplot.savefig("aaa2.png", dpi=300, bbox_inches='tight')
# matplotlib.pyplot.show()
# matplotlib.pyplot.close()

for b in range(len(bCSSS_value)):
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude, globe=None))
    ax.set_global()
    matplotlib.pyplot.title("CSSS gradient")
    img = matplotlib.pyplot.imshow(np.reshape(bCSSS_gradient[:, b],f1_netcdf.shape), transform=cartopy.crs.PlateCarree(), interpolation='nearest', origin='lower', extent=(0, 360, -90, 90),  vmax= np.max(np.abs(bCSSS_gradient[:, b])), vmin=-np.max(np.abs(bCSSS_gradient[:, b])), cmap ="bwr")
    #cb = fig.colorbar(img, extend='both', shrink=0.5)
    ax.coastlines(resolution='110m', color='grey', linestyle='-', alpha=1)
    gl = ax.gridlines()
    gl.xlocator = mticker.FixedLocator(list(np.arange(-180,180,20)))
    matplotlib.pyplot.savefig(f"baaa{b}.png", dpi=300, bbox_inches='tight')
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

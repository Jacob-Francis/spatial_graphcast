# ------------------------------
# Import the needed libraries
# ------------------------------

# Import the PAD python library
# The python library file (PY_MSE_python_lib.py) as well as the PAD C++ shared library file (CC_MSE_python_lib.so) 
# need to be in the same folder as the script using them
from PY_MSE_python_lib import *

# Import matplotlib 
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from jax import numpy as jnp

# ------------------------------
# Construct sample fields
# ------------------------------

# Set domain size
dimx=200
dimy=100

# draw two gaussians, one in each field
sigma = 20
x1=80
y1=50
x2=120
y2=y1

grid=np.mgrid[range(dimy), range(dimx)]
f1= (1/(2*np.pi*sigma**2) * np.exp(-((x1-grid[1])**2/(2*sigma**2) + (y1-grid[0])**2/(2*sigma**2))))
f2= (1/(2*np.pi*sigma**2) * np.exp(-((x2-grid[1])**2/(2*sigma**2) + (y2-grid[0])**2/(2*sigma**2))))

# visualize fields
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
plt.title("Visualization of fields")
cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(0.3,0.3,1.0)],512)
cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(1.0,0.3,0.3)],512)
norm_b = matplotlib.colors.Normalize()
norm_r = matplotlib.colors.Normalize()
# convert to rgb values using colormaps
fax = cmap_b(norm_b(f1))
fbx = cmap_r(norm_r(f2))
# get the combined colors for the image using the multiply effect
fx = fax*fbx
img = plt.imshow(fx, interpolation='nearest')
plt.show()
plt.close()

# ------------------------------
# Calculate MSE value and gradient
# ------------------------------

# assume the area size of all points is equal to one
area_size = np.full(f1.shape, 1)

# prepare 1D vectors that are needed for the interaction with C++ code
f1_1D = np.ascontiguousarray(f1.flatten(order='C'), dtype=np.float64)
f2_1D = np.ascontiguousarray(f2.flatten(order='C'), dtype=np.float64)
area_size_1D = np.ascontiguousarray(area_size.flatten(order='C'), dtype=np.float64)

# calculate MSE value and gradient - the partial derivates are done with respec to the second field which should be the forecast !
print('shapes:', f1_1D.shape, f2_1D.shape, area_size_1D.shape)


[MSE_value, MSE_gradient_1D] = calculate_MSE_with_gradient(f1_1D.astype(jnp.bfloat16)
, f2_1D.astype(jnp.bfloat16)
, area_size_1D.astype(jnp.bfloat16)
)
print("MSE value: " + str(MSE_value))
MSE_gradient =  np.reshape(MSE_gradient_1D,f1.shape)

# Visualize gradient field
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
plt.title("MSE gradient")
img = plt.imshow(MSE_gradient, vmax= np.max(np.abs(MSE_gradient)), vmin=-np.max(np.abs(MSE_gradient)), interpolation='nearest', cmap ="bwr")
cbar = fig.colorbar(img, orientation='horizontal')
plt.show()
# plt.close()


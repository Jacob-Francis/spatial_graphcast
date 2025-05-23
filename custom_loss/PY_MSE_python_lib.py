
import numpy as np
import os
import ctypes
from ctypes import *
from jax import numpy as jnp

# search for the PAD C++ shared library file (PAD_on_sphere_Cxx_shared_library.so) in the same folder
libc = ctypes.CDLL("/home/jacob/spatial_ml_losses/cplusplus_code/CC_MSE_python_lib.so") 

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

ND_POINTER_1D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C")
ND_POINTER_2D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

libc.calculate_MSE_with_gradient_ctypes.argtypes = [ND_POINTER_1D, ND_POINTER_1D, ND_POINTER_1D, c_size_t, POINTER(c_float), ND_POINTER_1D ]
libc.calculate_MSE_with_gradient_ctypes.restype = None


def calculate_MSE_with_gradient(values1, values2, area_size):

	# Convert from jnp.bfloat16 to np.flaot32 which are closer than np.float64
	values1, values2, area_size = np.asarray(values1, dtype=np.float32), np.asarray(values2, dtype=np.float32), np.asarray(area_size, dtype=np.float32)
	
	# Dealing with Batch dimension: (Batch, Time slices, Lat, Lon)
	# The area_size changes with Lat only so has shape (Lat,)
	# I'm sure there's a better way to do this on the c++ side..
	B, T, L1, L2 = values1.shape
	assert(L1==area_size.shape[0])

	values1 = np.reshape(values1, (B*T*L1*L2))
	values2 = np.reshape(values2, (B*T*L1*L2))
	area_size = np.tile(area_size.squeeze(), (B*T*L2))

	MSE_gradient = np.ascontiguousarray(np.zeros_like(values1))
	c_MSE_value = c_float()

	libc.calculate_MSE_with_gradient_ctypes(values1,values2,area_size,len(values1),byref(c_MSE_value),MSE_gradient)
	
	MSE_value = c_MSE_value.value
	
	return ([jnp.bfloat16(MSE_value),jnp.bfloat16(MSE_gradient)])


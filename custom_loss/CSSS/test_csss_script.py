from custom_loss import CustomMSE, CustomCSSS
import torch
from torch.autograd import Function
import numpy as np
import os
import ctypes
from ctypes import *
from pathlib import Path

import torch
import numpy as np

# from .PY_CSSS_library  import *

from torch.autograd import Function
import torch
import numpy as np

batch_size = 5
levels = 1
lat_shape = 1
lon_shape = 65160

from netCDF4 import Dataset

nc_file_id = Dataset("/home/jacob/spatial_ml_losses/PY_CSSS_example_field1.nc", 'r') 

lon_netcdf = nc_file_id.variables["lon"][:].data
lat_netcdf = nc_file_id.variables["lat"][:].data
f1_netcdf = nc_file_id.variables["precipitation"][:].data
f1_netcdf[f1_netcdf < 0] = 0
nc_file_id.close()

nc_file_id = Dataset("/home/jacob/spatial_ml_losses/PY_CSSS_example_field2.nc", 'r') 
f2_netcdf = nc_file_id.variables["precipitation"][:].data
f2_netcdf[f2_netcdf < 0] = 0
nc_file_id.close()


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

lon = np.ascontiguousarray(lon, dtype = np.float64)
lat = np.ascontiguousarray(lat, dtype = np.float64)
y = np.ascontiguousarray(values1, dtype = np.float64)
target = np.ascontiguousarray(values2, dtype = np.float64)
area_size = np.ascontiguousarray(area_size, dtype = np.float64)

x = torch.tensor(y, requires_grad=True).view(1, 1, lat_shape, lon_shape).repeat(batch_size, levels, 1, 1)
w =  torch.rand((levels, lat_shape, lon_shape), requires_grad=True,  dtype=torch.float64)


target = torch.Tensor(target)
area_size = torch.Tensor(area_size)

# Reshape it to broadcast over batch, levels, lat, lon
area_size_expanded = area_size.view(1, 1, lat_shape, lon_shape).repeat(batch_size, levels, 1, 1)

# Forward computation (elementwise multiply inputs)
y_c = x * w

target_c = target.view(1, 1, lat_shape, lon_shape).repeat(batch_size, levels, 1, 1)

# loss_func  = CustomCSSS
binary_file = CustomCSSS.load_binary_file('/home/jacob/spatial_ml_losses/custom_loss/CSSS/smoothing_data')
assert 0

# compute loss using custom function
# loss = loss_func.apply(y_c.view, target_c, area_size_expanded, binary_file)
loss = CustomCSSS.apply(y_c, target_c, area_size_expanded, binary_file)
print("Loss", loss)

# backward pass
loss.backward()

# print("x.grad:", x.grad)
print("w.grad:", w.grad.shape)
print("Loss:", loss.grad())


### Required shapes for Uroc's model;
# torch.Size([5, 84, 60, 120]) torch.Size([5, 84, 60, 120])
# torch.float64 torch.float64
# tensor(0.1004, device='cuda:0', grad_fn=<MulBackwar



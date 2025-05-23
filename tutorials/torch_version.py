import ctypes
from ctypes import *
from pathlib import Path

import torch
import numpy as np

# search for the PAD C++ shared library file (PAD_on_sphere_Cyhatyhat_shared_library.so) in the same folder
libc = ctypes.CDLL("/home/jacob/spatial_ml_losses/cplusplus_code/CC_MSE_LOWERING.so")

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

ND_POINTER_1D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C")
ND_POINTER_2D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

libc.calculate_MSE_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    c_size_t,
    POINTER(c_float),
]
libc.calculate_MSE_ctypes.restype = None

libc.calculate_MSE_gradient_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    c_size_t,
    ND_POINTER_1D,
]
libc.calculate_MSE_gradient_ctypes.restype = None


def calculate_MSE(values1, values2, area_size):

    # # Convert from jnp.bfloat16 to np.flaot32 which are closer than np.float64
    # values1, values2, area_size = (
    #     np.asarray(jax.device_get(values1), dtype=np.float32),
    #     np.asarray(jax.device_get(values2), dtype=np.float32),
    #     np.asarray(jax.device_get(area_size), dtype=np.float32),
    # )

    # Need to detach from gradient tree, then reactach gradients at the final step.

    # Dealing with Batch dimension: (Batch, Time slices, Lat, Lon)
    # The area_size changes with Lat only so has shape (Lat,)
    # I'm sure there's a better way to do this on the c++ side..
    # B, T, L1, L2 = values1.shape
    # assert L1 == area_size.shape[0]

    # Changing so that we assume everything comes in flat; will process outside this
    # values1 = np.reshape(values1, (B * T * L1 * L2))
    # values2 = np.reshape(values2, (B * T * L1 * L2))
    # area_size = np.tile(area_size.squeeze(), (B * T * L2))
    # print('area_size', area_size)
    # MSE_gradient = np.ascontiguousarray(np.zeros_like(values1))
    c_MSE_value = c_float()

    libc.calculate_MSE_ctypes(
        values1, values2, area_size, len(values1), byref(c_MSE_value)
    )

    MSE_value = c_MSE_value.value

    return [MSE_value]  # , jnp.bfloat16(MSE_gradient)]


def calculate_MSE_gradient(values1, values2, area_size):

    # Convert from jnp.bfloat16 to np.flaot32 which are closer than np.float64
    # values1, values2, area_size = (
    #     np.asarray(jax.device_get(values1), dtype=np.float32),
    #     np.asarray(jax.device_get(values2), dtype=np.float32),
    #     np.asarray(jax.device_get(area_size), dtype=np.float32),
    # )

    # Need to detach from gradient tree, then reactach gradients at the final step.

    # Dealing with Batch dimension: (Batch, Time slices, Lat, Lon)
    # The area_size changes with Lat only so has shape (Lat,)
    # I'm sure there's a better way to do this on the c++ side..
    # B, T, L1, L2 = values1.shape
    # assert L1 == area_size.shape[0]

    # values1 = np.reshape(values1, (B * T * L1 * L2))
    # values2 = np.reshape(values2, (B * T * L1 * L2))
    # area_size = np.tile(area_size.squeeze(), (B * T * L2))
    # print('area_size', area_size)
    MSE_gradient = np.ascontiguousarray(np.zeros_like(values1))

    libc.calculate_MSE_gradient_ctypes(
        values1, values2, area_size, len(values1), MSE_gradient
    )

    return MSE_gradient

if __name__ == "__main__":
    # Test the C++ code
    values1 = np.random.rand(10).astype(np.float32)
    values2 = np.random.rand(10).astype(np.float32)
    area_size = np.random.rand(10).astype(np.float32)

    MSE_value = calculate_MSE(values1, values2, area_size)
    print("MSE value:", MSE_value)

    MSE_gradient = calculate_MSE_gradient(values1, values2, area_size)
    print("MSE gradient:", MSE_gradient)

    values1 = np.ones(10).astype(np.float32)
    values2 = np.ones(10).astype(np.float32)
    area_size = np.ones(10).astype(np.float32)/10

    MSE_value = calculate_MSE(values1, values2, area_size)
    print("MSE value:", MSE_value)

    MSE_gradient = calculate_MSE_gradient(values1, values2, area_size)
    print("MSE gradient:", MSE_gradient)

    values1 = np.ones(10).astype(np.float32)
    values2 = np.zeros(10).astype(np.float32)
    area_size = np.ones(10).astype(np.float32)/10

    MSE_value = calculate_MSE(values1, values2, area_size)
    print("MSE value:", MSE_value)

    MSE_gradient = calculate_MSE_gradient(values1, values2, area_size)
    print("MSE gradient:", MSE_gradient)


import torch

# Simple input tensor, requires gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Simple weight tensor, also with gradients
w = torch.tensor([0.5, -1.0], requires_grad=True)

area_size = torch.tensor([0.5, 0.5], requires_grad=False)

# Target tensor
target = torch.tensor([1.0, -2.0])

# Simple forward computation (linear operation)
y = x * w

# # Built-in MSE loss (you can replace this with your C++-wrapped one)
loss = torch.mean((y - target) ** 2)

loss.backward()
# # Inspect gradients
print("true x.grad:", x.grad)
print("true w.grad:", w.grad)
print("Loss:", loss.item())


import torch
from torch.autograd import Function
import numpy as np

class CustomMSE(Function):
    @staticmethod
    def forward(ctx, y, target, area_size):
        # Convert inputs to numpy float32 arrays
        y_np = y.detach().cpu().numpy().astype(np.float32)
        target_np = target.detach().cpu().numpy().astype(np.float32)
        area_np = area_size.detach().cpu().numpy().astype(np.float32)

        # Call your C++ MSE function
        loss_val = calculate_MSE(y_np, target_np, area_np)[0]

        # Save tensors for backward
        ctx.save_for_backward(y, target, area_size)

        # Return loss as torch tensor
        return torch.tensor(loss_val, device=y.device, dtype=y.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        y, target, area_size = ctx.saved_tensors

        # Call your C++ gradient function
        grad_np = calculate_MSE_gradient(
            y.detach().cpu().numpy().astype(np.float32),
            target.detach().cpu().numpy().astype(np.float32),
            area_size.detach().cpu().numpy().astype(np.float32),
        )

        grad_y = torch.from_numpy(grad_np).to(y.device).to(y.dtype)

        # grad_output is gradient from upstream (scalar)
        grad_y = grad_output * grad_y

        # Return gradients wrt inputs in order of forward inputs:
        # y, target, area_size. Usually target and area_size are constants.
        return grad_y, None, None

# Simple input tensor, requires gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Simple weight tensor, also with gradients
w = torch.tensor([0.5, -1.0], requires_grad=True)

area_size = torch.tensor([0.5, 0.5], requires_grad=False)

# Target tensor
target = torch.tensor([1.0, -2.0])

# Simple forward computation (linear operation)
y = x * w

# compute loss using custom function
loss = CustomMSE.apply(y, target, area_size)


# backward pass
loss.backward()

print("x.grad:", x.grad)
print("w.grad:", w.grad)
print("Loss:", loss.item())


### Required shapes for Uroc's model;
# torch.Size([5, 84, 60, 120]) torch.Size([5, 84, 60, 120])
# torch.float32 torch.float32
# tensor(0.1004, device='cuda:0', grad_fn=<MulBackwar
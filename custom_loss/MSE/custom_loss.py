
import torch
from torch.autograd import Function
import numpy as np
import os
import ctypes
from ctypes import *
from pathlib import Path

import torch
import numpy as np

# Change path to the directory where the shared library is located
os.chdir(Path(__file__).parent)


# search for the PAD C++ shared library file (PAD_on_sphere_Cyhatyhat_shared_library.so) in the same folder
libc = ctypes.CDLL("./CC_MSE_TORCH.so")

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

ND_POINTER_1D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
ND_POINTER_2D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C")

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

libc.calculate_MSE_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    c_size_t,
    POINTER(c_double),
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


# import numpy as np
# from ctypes import c_double, c_size_t, POINTER

def calculate_MSE(values1, values2, area_size):
    B, N = values1.shape

    # values1 = np.ascontiguousarray(values1, dtype=np.float64)
    # values2 = np.ascontiguousarray(values2, dtype=np.float64)
    # area_size = np.ascontiguousarray(area_size, dtype=np.float64)
    c_MSE_value = c_double()
    MSE_value = np.zeros(B, dtype=np.float64)
    for b in range(B):
        
        libc.calculate_MSE_ctypes(
            values1[b, :],
            values2[b, :],
            area_size[b, :],
            c_size_t(N),
            byref(c_MSE_value)
        )
        MSE_value[b] = c_MSE_value.value

    return MSE_value 



def calculate_MSE_gradient(values1, values2, area_size):


    # Need to detach from gradient tree, then reactach gradients at the final step.

    MSE_gradient = np.ascontiguousarray(np.zeros_like(values1))

    # ToDo:
    # Convert to numpy float64 arrays
    for b in range(values1.shape[0]):
        libc.calculate_MSE_gradient_ctypes(
        values1[b, :], values2[b, :], area_size[b, :], len(values1[b, :]), MSE_gradient[b, :]
        )

    return -MSE_gradient

class CustomMSE(Function):
    @staticmethod
    def forward(ctx, y, target, area_size=None):
        B = y.shape[0]

        if area_size is None:
            area_size = torch.ones_like(y)
        assert y.shape == target.shape
        assert y.shape == area_size.shape

        # Convert inputs to numpy float64 arrays
        y_np = y.detach().view(B, -1, ).cpu().numpy().astype(np.float64)
        target_np = target.detach().view(B, -1, ).cpu().numpy().astype(np.float64)
        area_np = area_size.detach().view(B, -1, ).cpu().numpy().astype(np.float64)

        # Call your C++ MSE function
        loss_val = calculate_MSE(y_np, target_np, area_np)

        # Save tensors for backward
        ctx.save_for_backward(y, target, area_size)

        # Return loss as torch tensor
        return torch.tensor(loss_val, device=y.device, dtype=y.dtype).mean()

    @staticmethod
    def backward(ctx, grad_output):
        y, target, area_size = ctx.saved_tensors
        B = y.shape[0]

        if area_size is None:
            area_size = torch.ones_like(y)

        # Call your C++ gradient function
        grad_np = calculate_MSE_gradient(
            y.detach().view(B, -1, ).cpu().numpy().astype(np.float64),
            target.detach().view(B, -1, ).cpu().numpy().astype(np.float64),
            area_size.detach().view(B, -1, ).cpu().numpy().astype(np.float64),
        )

        grad_y = torch.from_numpy(grad_np).to(y.device).to(y.dtype)

        # grad_output is gradient from upstream (scalar)
        grad_y = grad_output * grad_y
    
        # Return gradients wrt inputs in order of forward inputs:
        # y, target, area_size. Usually target and area_size are constants.
        return grad_y.view(y.shape), None, None

# ToDo: I don't think the area elements are actually doing anything....

if __name__ == "__main__":
    # ------------------------ TEST CASE -----------------------

    # Shape parameters
    # torch.Size([5, 84, 60, 120]) torch.Size([5, 84, 60, 120])

    batch_size = 5
    levels = 84
    lat = 60
    lon = 120

    # Simple input tensor, requires gradients
    x = torch.rand((batch_size, levels, lat, lon), requires_grad=True, dtype=torch.float64)
    x_c = x.clone()

    # Simple weight tensor, also with gradients
    w = torch.rand((batch_size, levels, lat, lon), requires_grad=True,  dtype=torch.float64)
    w_c = w.clone()

    # Target tensor
    target = torch.rand((batch_size, levels, lat, lon),  dtype=torch.float64)
    target_c = target.clone()

    # Area size tensor — let's assume it varies by levels (like geospatial area weighting)
    area_size = torch.rand(levels,  dtype=torch.float64)
    # Reshape it to broadcast over batch, levels, lat, lon
    area_size_expanded = area_size.view(1, levels, 1, 1).repeat(batch_size, 1, lat, lon)

    # Forward computation (elementwise multiply inputs)
    y = x * w

    # MSE Loss with area weighting
    mse = ((y - target) ** 2) * area_size_expanded  # apply area weighting
    loss = mse.sum()/area_size_expanded.sum()  # mean over all dims

    # Backward pass
    loss.backward()

    # Inspect gradients
    print("x.grad shape:", x.grad.mean())
    print("w.grad shape:", w.grad.mean())
    print("loss value:", loss.item())

    y_c = x_c * w_c

    # compute loss using custom function
    loss = CustomMSE.apply(y_c, target_c, area_size_expanded)

    # backward pass
    loss.backward()

    print("x.grad:", x.grad.mean())
    print("w.grad:", w.grad.mean())
    print("Loss:", loss.item())


    ### Required shapes for Uroc's model;
    # torch.Size([5, 84, 60, 120]) torch.Size([5, 84, 60, 120])
    # torch.float64 torch.float64
    # tensor(0.1004, device='cuda:0', grad_fn=<MulBackwar



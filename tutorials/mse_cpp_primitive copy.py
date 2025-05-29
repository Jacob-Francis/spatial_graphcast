import numpy as np
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
from jax import core, lax
from jax.extend import core as ext_core
from jax.interpreters import ad, mlir, batching
from jax._src.lib.mlir.dialects import hlo

import numpy as np
import os
import ctypes
from ctypes import *
from jax import numpy as jnp
from pathlib import Path
import jax

from jax.lib import xla_client

# search for the PAD C++ shared library file (PAD_on_sphere_Cyhatyhat_shared_library.so) in the same folder
libc = ctypes.CDLL("/home/jacob/spatial_ml_losses/cplusplus_code/CC_MSE_LOWERING.so")

jax.ffi.register_ffi_target("Mse", jax.ffi.pycapsule(libc.Mse), platform="cpu")

jax.ffi.register_ffi_target("MseGrad", jax.ffi.pycapsule(libc.MseGrad), platform="cpu")

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

ND_POINTER_1D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C")
ND_POINTER_2D = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C")

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

libc.calculate_MSE_ctypes.argtypes = [
    ND_POINTER_2D,
    ND_POINTER_2D,
    ND_POINTER_2D,
    c_size_t,
    ND_POINTER_1D   #POINTER(c_float),
]
libc.calculate_MSE_ctypes.restype = None

libc.calculate_MSE_gradient_ctypes.argtypes = [
    ND_POINTER_2D,
    ND_POINTER_2D,
    ND_POINTER_2D,
    c_size_t,
    ND_POINTER_2D,
]
libc.calculate_MSE_gradient_ctypes.restype = None


def calculate_MSE(values1, values2, area_size):

    # Convert from jnp.bfloat16 to np.flaot32 which are closer than np.float64
    values1, values2, area_size = (
        np.asarray(jax.device_get(values1), dtype=np.float32),
        np.asarray(jax.device_get(values2), dtype=np.float32),
        np.asarray(jax.device_get(area_size), dtype=np.float32),
    )

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
    MSE_value = np.zeros(values1.shape[0], dtype=np.float32)


    libc.calculate_MSE_ctypes(
        values1, values2, area_size, len(values1), MSE_value
    )

    return [jnp.bfloat16(MSE_value)]  # , jnp.bfloat16(MSE_gradient)]


def calculate_MSE_gradient(values1, values2, area_size):

    # Convert from jnp.bfloat16 to np.flaot32 which are closer than np.float64
    values1, values2, area_size = (
        np.asarray(jax.device_get(values1), dtype=np.float32),
        np.asarray(jax.device_get(values2), dtype=np.float32),
        np.asarray(jax.device_get(area_size), dtype=np.float32),
    )

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

    libc.calculate_MSE_gradient_ctypes(values1, values2, area_size, len(values1), MSE_gradient)

    return [jnp.bfloat16(MSE_gradient)]


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


# Step 1: Define primitive
cpp_mse = ext_core.Primitive("weighted_mse")


# Step 2: Function interface
def weighted_mse(yhat, y, a):
    return cpp_mse.bind(yhat, y, a)


# Step 3: Primal (concrete) implementation
def weighted_mse_impl(yhat, y, a):
    return calculate_MSE(yhat, y, a)


cpp_mse.def_impl(weighted_mse_impl)

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

# test
print(
    weighted_mse(
        jnp.ones((3, 6)), 0.5*jnp.ones((3, 6)), jnp.array([[0.4, 0.3, 0.4, 0.3, 0.4, 0.3], [0.4, 0.3, 0.4, 0.3, 0.4, 0.3],[0.4, 0.3, 0.4, 0.3, 0.4, 0.3]])
    )
)

# Something is still wrong with the batches
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


# Step 4: Abstract evaluation
def weighted_mse_abstract(yhats, ys, zs):
    assert yhats.shape == ys.shape
    assert yhats.shape == zs.shape
    return core.ShapedArray((), yhats.dtype)


cpp_mse.def_abstract_eval(weighted_mse_abstract)


from jax.interpreters import mlir
from jax.extend import ffi as jax_ffi
import jaxlib.mlir.ir as ir


# Step 5: XLA lowering (CPU only shown here)
mlir.register_lowering(
    cpp_mse,
    jax.ffi.ffi_lowering(
        "Mse",
        operand_layouts=[(0,), (0,), (0,)],
        result_layouts=[()],
    ),
    platform="cpu",
)


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
from jax._src import api

print(
    api.jit(lambda x, y, a: weighted_mse(x, y, a))(
        jnp.ones((2, 6)), 0.5 * jnp.ones((2, 6)), jnp.array([[0.4, 0.3, 0.4, 0.3, 0.4, 0.3], [0.4, 0.3, 0.4, 0.3, 0.4, 0.3]])
    )
)

assert 0
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


def Mse_fwd(yhat, y, a):
    out = jax.ffi.ffi_call(
        "Mse",
        (
            jax.ShapeDtypeStruct(yhat.shape, yhat.dtype),
            jax.ShapeDtypeStruct(yhat.shape, yhat.dtype),
            jax.ShapeDtypeStruct(yhat.shape, yhat.dtype),
        ),
        vmap_method="broadcast_all",
    )(yhat, y, a)
    return out


def Mse_bwd(input, ct):
    yhat, y, a = input
    out = jax.ffi.ffi_call(
        "MseGrad",
        (
            jax.ShapeDtypeStruct(yhat.shape, yhat.dtype),
            jax.ShapeDtypeStruct(yhat.shape, yhat.dtype),
            jax.ShapeDtypeStruct(yhat.shape, yhat.dtype),
        ),
        vmap_method="broadcast_all",
    )(yhat, y, a)
    return out.dot(ct)


weighted_mse = jax.custom_vjp(weighted_mse)
weighted_mse.defvjp(Mse_fwd, Mse_bwd)

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


print(
    (jnp.ones((6)), 0.5 * jnp.ones((6)), jnp.array([0.4, 0.3, 0.4, 0.3, 0.4, 0.3])),
    (jnp.ones((6))),
)

api.jvp(
    weighted_mse,
    (jnp.ones((6)), 0.5 * jnp.ones((6)), jnp.array([0.4, 0.3, 0.4, 0.3, 0.4, 0.3])),
    (jnp.ones((6)), jnp.zeros((6)), jnp.zeros((6))),
)

# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


assert 0


# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


# Step 6: Forward-mode autodiff (JVP)
def weighted_mse_jvp(arg_values, arg_tangents):
    yhat, y, a = arg_values
    yhatt, yt, zt = arg_tangents

    def make_zero(tan):
        return lax.zeros_like_array(x) if type(tan) is ad.Zero else tan

    # primal = weighted_mse(yhat, y, a)
    # tangent = weighted_mse(zero_if_needed(yhatt, yhat), y,
    #                        weighted_mse(yhat, zero_if_needed(yt, y), zero_if_needed(zt, a)))
    primal = weighted_mse(yhat, y, a)
    tangent = jax.ffi.ffi_call(
        "MseGrad",
        (
            jax.ShapeDtypeStruct(yhat.shape, x.dtype),
            jax.ShapeDtypeStruct(yhat.shape, x.dtype),
            jax.ShapeDtypeStruct(yhat.shape, x.dtype),
        ),
        vmap_method="broadcast_all",
    )(yhat, y, a)

    return primal, tangent


ad.primitive_jvps[cpp_mse] = weighted_mse_jvp


# Step 7: Reverse-mode autodiff (transpose)
def weighted_mse_transpose(ct, yhat, y, a):
    if not ad.is_undefined_primal(yhat):
        ct_y = (
            weighted_mse(yhat, ct, lax.zeros_like_array(yhat))
            if not isinstance(ct, ad.Zero)
            else ad.Zero(y.aval)
        )
        return None, ct_y, ct
    else:
        ct_yhat = (
            weighted_mse(ct, y, lax.zeros_like_array(y))
            if not isinstance(ct, ad.Zero)
            else ad.Zero(yhat.aval)
        )
        return ct_yhat, None, ct


ad.primitive_transposes[cpp_mse] = weighted_mse_transpose


# Step 8: Batching
def weighted_mse_batch(vector_args, batch_ayhates):
    assert batch_ayhates[0] == batch_ayhates[1] == batch_ayhates[2]
    return weighted_mse(*vector_args), batch_ayhates[0]


batching.primitive_batchers[cpp_mse] = weighted_mse_batch

# --- Optional: usage and testing ---
if __name__ == "__main__":
    pass
    # import jax.numpy as jnp

    # def square_add(a, b):
    #     return weighted_mse(a, a, b)

    # print("Primal:", square_add(2., 10.))  # 2*2 + 10 = 14
    # print("Grad:", jax.grad(square_add)(2., 10.))  # d(2*2 + 10)/da = 4
    # print("JIT:", jax.jit(square_add)(2., 10.))
    # print("Vmap:", jax.vmap(square_add)(jnp.array([2., 3.]), jnp.array([10., 20.])))

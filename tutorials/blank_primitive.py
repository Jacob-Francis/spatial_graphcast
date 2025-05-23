import numpy as np
import jax
from jax import core, lax
from jax.extend import core as ext_core
from jax.interpreters import ad, mlir, batching
from jax._src.lib.mlir.dialects import hlo

# Step 1: Define primitive
multiply_add_p = ext_core.Primitive("multiply_add")

# Step 2: Function interface
def multiply_add(x, y, z):
    return multiply_add_p.bind(x, y, z)

# Step 3: Primal (concrete) implementation
def multiply_add_impl(x, y, z):
    return np.add(np.multiply(x, y), z)

multiply_add_p.def_impl(multiply_add_impl)

# Step 4: Abstract evaluation
def multiply_add_abstract(xs, ys, zs):
    assert xs.shape == ys.shape == zs.shape
    return core.ShapedArray(xs.shape, xs.dtype)

multiply_add_p.def_abstract_eval(multiply_add_abstract)

# Step 5: XLA lowering (CPU only shown here)
def multiply_add_lowering(ctx, x, y, z):
    return [hlo.AddOp(hlo.MulOp(x, y), z).result]

mlir.register_lowering(multiply_add_p, multiply_add_lowering, platform="cpu")

# Step 6: Forward-mode autodiff (JVP)
def multiply_add_jvp(arg_values, arg_tangents):
    x, y, z = arg_values
    xt, yt, zt = arg_tangents

    def zero_if_needed(t, ref):
        return lax.zeros_like_array(ref) if isinstance(t, ad.Zero) else t

    primal = multiply_add(x, y, z)
    tangent = multiply_add(zero_if_needed(xt, x), y,
                           multiply_add(x, zero_if_needed(yt, y), zero_if_needed(zt, z)))
    return primal, tangent

ad.primitive_jvps[multiply_add_p] = multiply_add_jvp

# Step 7: Reverse-mode autodiff (transpose)
def multiply_add_transpose(ct, x, y, z):
    if not ad.is_undefined_primal(x):
        ct_y = multiply_add(x, ct, lax.zeros_like_array(x)) if not isinstance(ct, ad.Zero) else ad.Zero(y.aval)
        return None, ct_y, ct
    else:
        ct_x = multiply_add(ct, y, lax.zeros_like_array(y)) if not isinstance(ct, ad.Zero) else ad.Zero(x.aval)
        return ct_x, None, ct

ad.primitive_transposes[multiply_add_p] = multiply_add_transpose

# Step 8: Batching
def multiply_add_batch(vector_args, batch_axes):
    assert batch_axes[0] == batch_axes[1] == batch_axes[2]
    return multiply_add(*vector_args), batch_axes[0]

batching.primitive_batchers[multiply_add_p] = multiply_add_batch

# --- Optional: usage and testing ---
if __name__ == "__main__":
    import jax.numpy as jnp

    def square_add(a, b):
        return multiply_add(a, a, b)

    print("Primal:", square_add(2., 10.))  # 2*2 + 10 = 14
    print("Grad:", jax.grad(square_add)(2., 10.))  # d(2*2 + 10)/da = 4
    print("JIT:", jax.jit(square_add)(2., 10.))
    print("Vmap:", jax.vmap(square_add)(jnp.array([2., 3.]), jnp.array([10., 20.])))

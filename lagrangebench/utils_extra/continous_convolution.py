from .mapping_function import batched_mapping, ball_to_cube
import jax.numpy as jnp
from jax.ops import segment_sum
import jax
from .interpolation import bilinear_interpolate_features
from functools import partial

def ball_to_cube_2d(position):
    new_pos = ball_to_cube(position)
    return 0.5 * jnp.array([new_pos[0], new_pos[1]]) + jnp.array([0.5, 0.5])

def window_poly6(r):
    return jnp.clip((1 - r*r)**3, 0, 1)

mapping_function_batched = batched_mapping(ball_to_cube_2d)
window_function_batched = batched_mapping(window_poly6)

@partial(jax.jit, static_argnames=["aggregate_points"])
def continous_conv_operation(kernel, receivers, relative_positions, features, a, aggregate_points, window_function=window_poly6, normalize=False):
    mapped_positions = mapping_function_batched(relative_positions)
    R, _, ChIn, ChOut = kernel.shape
    norm_coeff = jax.lax.cond(normalize, lambda: jnp.sum(a), lambda: jnp.array(1.0, dtype=a.dtype))
    patch_matrix = bilinear_interpolate_features(R, mapped_positions, features, receivers, a, norm_coeff, aggregate_points)
    kernel_flat = kernel.reshape((R * R * ChIn, ChOut))
    return patch_matrix @ kernel_flat
        
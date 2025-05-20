import jax.numpy as jnp
import jax
from functools import partial

@partial(jax.jit,
         static_argnames=("R","num_receivers"))
def bilinear_interpolate_features(R, uv, features, receivers, a, normalization_coeff, num_receivers):
    _, ChIn = features.shape
    
    K2     = R * R
    coords = uv * (R - 1)
    x = coords[..., 0]
    y = coords[..., 1]

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, R - 1)
    y1 = jnp.clip(y0 + 1, 0, R - 1)

    idx_a = x0 * R + y0
    idx_b = x0 * R + y1
    idx_c = x1 * R + y0
    idx_d = x1 * R + y1

    dx = (x - x0)
    dy = (y - y0)

    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy

    M3 = jnp.zeros((num_receivers, K2, ChIn), dtype=features.dtype)
        
    def scatter_corner(M, idx, w):
        weight =  (a * w) / normalization_coeff
        vals = weight[:, None] * features
        return M.at[receivers, idx, :].add(vals)
    
    M3 = scatter_corner(M3, idx_a, wa)
    M3 = scatter_corner(M3, idx_b, wb)
    M3 = scatter_corner(M3, idx_c, wc)
    M3 = scatter_corner(M3, idx_d, wd)

    M_flat = M3.reshape((num_receivers, K2 * ChIn))
    return M_flat

def bilinear_interpolate(
    grid: jnp.ndarray,  
    uv:   jnp.ndarray   
) -> jnp.ndarray:
    """
    Do bilinear interpolation on `grid` at the fractional positions `uv`. kernel size
    """
    R = grid.shape[0]
    coords = uv * (R - 1)
    x = coords[..., 0]
    y = coords[..., 1]

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, R - 1)
    y1 = jnp.clip(y0 + 1, 0, R - 1)

    dx = (x - x0)[..., None, None]  
    dy = (y - y0)[..., None, None]

    Ia = grid[x0, y0]  
    Ib = grid[x0, y1]
    Ic = grid[x1, y0]
    Id = grid[x1, y1]

    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy

    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def calculate_new_features(features, weights, a, normalization_coeff):
    #jax.debug.print("x={x} y={y} z={z}", x=jnp.ones((features.shape[0],  weights.shape[1])).shape, y=weights.shape, z=jnp.matmul(features, weights).shape)
    return 1/normalization_coeff * a * jnp.ones((weights.shape[1],), dtype=features.dtype)#jnp.matmul(features, weights)
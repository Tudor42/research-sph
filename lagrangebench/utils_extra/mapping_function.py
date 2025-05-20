import jax
import jax.numpy as jnp


def ball_to_cylinder(pos):
    if len(pos) == 3:
        u, v, w = pos
    else:
        u, v = pos
        w = 0
    u_2 = u*u
    v_2 = v*v
    w_2 = w*w
    t =  jnp.sqrt(u_2 + v_2 + w_2)
    def map_pos():
        s, z = jax.lax.cond(5.0 / 4 * w_2  > u_2 + v_2, 
                            lambda: jnp.array((jnp.sqrt(3*t/(t+jnp.abs(w))), 1.0*jnp.sign(w)*t), dtype=t.dtype),
                            lambda: jnp.array((t/jnp.sqrt(u_2+v_2), 3.0 * w / 2), dtype=t.dtype))

        return jnp.array([s*u, s*v, z], dtype=t.dtype)
    return jax.lax.cond(t == 0, lambda: jnp.array([0.0, 0.0, 0.0], dtype=t.dtype), map_pos)

def cylinder_to_cube(pos):
    if len(pos) == 3:
        cx, cy, cz = pos
    else:
        cx, cy = pos
        cz = 0
    cx_2 = cx * cx
    cy_2 = cy * cy
    r_xy = jnp.sqrt(cx_2 + cy_2)
    sy = jnp.sign(cy)
    sx = jnp.sign(cx)

    def map_disc():
        return jax.lax.cond(cy_2 < cx_2,
                            lambda: jnp.array([
                                sx * r_xy,
                                (4.0 / jnp.pi) * sx * r_xy * jnp.arctan(cy / cx),
                                cz
                            ]), 
                            lambda: jnp.array([
                                (4.0 / jnp.pi) * sy * r_xy * jnp.arctan(cx / cy),
                                sy * r_xy,
                                cz
                            ]))

    return jax.lax.cond(jnp.logical_and(cx == 0, cy == 0), lambda: jnp.array([0.0, 0.0, cz]), map_disc)

def ball_to_cube(pos):
    return cylinder_to_cube(ball_to_cylinder(pos))


def batched_mapping(mapping_func):
    return jax.vmap(mapping_func, in_axes=0, out_axes=0)


if __name__ == "__main__":
    print(ball_to_cube(jnp.array([0.5, 0.5])))

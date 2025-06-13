import argparse
from jax import config
from application.client.remote_state_manager import RemoteStateManager
from application.gui.window import Window
import jax.numpy as jnp
import taichi as ti
import os
import jax

from application.utils.tkinter_component import get_connection
from jax_sph.utils import Tag

def update(window: Window):
    tags = window.state_manager.get_tags()
    positions = window.state_manager.get_positions()
    vels = window.state_manager.get_velocities()
    valid = tags != Tag.PAD_VALUE
    tags = tags[valid]
    pts  = positions[valid]

    speeds = jnp.linalg.norm(vels, axis=1)
    eps = 1e-8
    s_min = 0
    s_max = 1
    norm = ( speeds - s_min) / (s_max - s_min + eps)

    if True:
        per_vertex_color = jnp.stack([
            norm,
            jnp.full_like(norm, 0.2),
            1.0 - norm,
            jnp.ones_like(norm),
        ], axis=1)
    else:
        per_vertex_color = jnp.array((0.2, 0.5, 0.9, 1.0))
    fluid_mask = tags == Tag.FLUID
    per_vertex_color = jnp.where(fluid_mask[:, None], per_vertex_color, jnp.array((0.7, 0.7, 0.7, 1.0)))

    gui.draw_circles(
        pts,
        radius=gui.radius,
        size=pts.shape[0],
        per_vertex_color=per_vertex_color,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", type=bool, default=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50007)
    parser.add_argument("--password", type=str, help="Password clients must provide in init", default="")

    args = parser.parse_args()

    config.update("jax_enable_x64", True)
 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # pick GPU #0
    jax.config.update('jax_platform_name', 'gpu')   # force GPU

    if args.password == "":
        sm = get_connection()
    else:
        sm  = RemoteStateManager(args.host, args.port, args.password)
    gui = Window(arch=ti.cuda, state_manager=sm)
    gui.run(update_fn=update)

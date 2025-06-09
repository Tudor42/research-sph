from jax import config
from application.gui.window import Window
import numpy as np
import jax.numpy as jnp
import taichi as ti
import os
import jax

from jax_sph.utils import Tag

def update(window: Window):
    tags = window.state_manager.get_tags()
    positions = window.state_manager.get_positions()

    valid = tags != Tag.PAD_VALUE
    tags = tags[valid]
    pts  = positions[valid]

    fluid_mask = tags == Tag.FLUID
    per_vertex_color = jnp.where(fluid_mask[:, None], jnp.array((0.2, 0.5, 0.9, 1.0)), jnp.array((0.7, 0.7, 0.7, 1.0)))

    gui.draw_circles(
        pts,
        radius=gui.radius,
        size=pts.shape[0],
        per_vertex_color=per_vertex_color,
    )

if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp

    # 1) List all available devices:
    print("All devices:", jax.devices())
    # → [CpuDevice(id=0)] or [GpuDevice(id=0), CpuDevice(id=0)], etc.

    # 2) List just local (attached) devices:
    print("Local devices:", jax.local_devices())

    # 3) See the default backend/platform name:
    print("Default backend:", jax.default_backend())  # e.g. 'gpu' or 'cpu'
    x = jnp.zeros((4,))

    # In JAX ≥0.4, you can do:
    print("x is on:", x.device)

    # In older versions you might need:
    print("x is on:", x.device_buffer.device())
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # pick GPU #0
    jax.config.update('jax_platform_name', 'gpu')   # force GPU
    gui = Window(arch=ti.cuda)
    gui.run(update_fn=update)

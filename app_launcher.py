from application.gui.window import Window
import numpy as np
import jax.numpy as jnp
import taichi as ti

gui = Window(arch=ti.cuda)
N = 2000

buffer_a = np.random.rand(N, 2).astype(np.float32)
buffer_b = np.random.rand(N, 3).astype(np.float32)

front_buffer = buffer_a
pts = jnp.array(front_buffer)

def update(gui: Window):
    gui.draw_circles(pts, size=N, radius=gui.radius, per_vertex_color=buffer_b)

gui.run(update_fn=update)

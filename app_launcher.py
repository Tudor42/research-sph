from application.gui import GUI
import time
import numpy as np
import jax
import jax.numpy as jnp
import taichi as ti

gui = GUI(arch=ti.cuda)
N = 200000

buffer_a = np.random.rand(N, 2).astype(np.float32)

front_buffer = buffer_a
pts = jnp.array(front_buffer)

def update(gui: GUI):
    gui.draw_circles(pts, radius=gui.radius, color=(0.5, 0.5, 0.5))

gui.run(update_fn=update)

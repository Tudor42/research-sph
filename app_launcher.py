from jax import config
from application.gui.window import Window
import numpy as np
import jax.numpy as jnp
import taichi as ti

if __name__ == "__main__":
    config.update("jax_enable_x64", True)
    gui = Window(arch=ti.cuda)
    gui.run(update_fn=gui.case_manager.update)

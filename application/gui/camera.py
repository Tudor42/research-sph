import taichi as ti
import jax
import jax.numpy as jnp

from application.utils.event_handler import EventHandler

@jax.jit
def camera_ndc(pts, center, scale, ratio):
    rel = pts - center
    norm = rel / scale
    norm = norm.at[:, 0].set(ratio * norm[:, 0])
    return norm + 0.5

class Camera2D(EventHandler):
    def __init__(self,  
                 center=(0.0, 0.0), 
                 scale=1.0):
        self.center = jnp.array(center)
        self.scale = jnp.array(scale)
        self._last_mouse = None
        self._ctrl_pressed = False

    def world_to_ndc(self, pts_world, ratio=1):
        return camera_ndc(pts_world, self.center, self.scale, ratio)
    
    def handle_press(self, key, x, y):
        if key == 'q':
            self.scale *= 0.9
        elif key == 'e':
            self.scale *= 1.1

        pan = 0.1 * float(self.scale)
        if key in ('a', ti.ui.LEFT):
            self.center += jnp.array((-pan, 0.0))
        elif key in ('d', ti.ui.RIGHT):
            self.center += jnp.array((+pan, 0.0))
        elif key in ('w', ti.ui.UP):
            self.center += jnp.array((0.0, +pan))
        elif key in ('s', ti.ui.DOWN):
            self.center += jnp.array((0.0, -pan))

        # Begin mouse‐drag pan on left‐button press
        if key == ti.ui.LMB:
            self._last_mouse = (x, y)
        if key == ti.ui.CTRL:
            self._ctrl_pressed = True

    def handle_motion(self, x, y):
        if self._last_mouse is not None and self._ctrl_pressed:
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            world_dx = -dx * float(self.scale)
            world_dy = -dy * float(self.scale)
            self.center += jnp.array((world_dx, world_dy))
            self._last_mouse = (x, y)

    def handle_release(self, key):
        if key == ti.ui.LMB:
            self._last_mouse = None
        if key == ti.ui.CTRL:
            self._ctrl_pressed = False

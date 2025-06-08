import taichi as ti
import numpy as np
import jax
import jax.numpy as jnp

@jax.jit
def camera_ndc(pts, center, scale, ratio):
    rel = pts - center
    norm = rel / scale
    norm = norm.at[:, 0].set(ratio * norm[:, 0])
    return norm + 0.5

class Camera2D:
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


class GUI:
    def __init__(self, 
                 resolution=(800, 600), 
                 title="SPH Simulator",
                 background_color=(0.0, 0.0, 0.0),
                 fps_limit=60,
                 arch=ti.cpu):
        ti.init(arch=arch)

        self.gui = ti.ui.Window(title,
                                res=resolution,
                                fps_limit=fps_limit)
        self.camera = Camera2D(center=(0.5,0.5), scale=1.0)
        self.canvas = self.gui.get_canvas()
        self.canvas.set_background_color(background_color)
        self.circles_taichi_field = ti.Vector.field(2, ti.f32, shape=1)
        self.radius = 5

    def set_camera(self, center, scale):
        self.camera.center = np.array(center, dtype=np.float32)
        self.camera.scale = float(scale)

    def alloc_circles_buffer(self, N):
        if self.circles_taichi_field.shape[0] != N:
            self.circles_taichi_field = ti.Vector.field(2, ti.f32, shape=N)

    def draw_circles(self, pts_world, radius, color, palette=None, palette_indices=None):
        win_size = self.gui.get_window_shape()
        pts_ndc = self.camera.world_to_ndc(pts_world, win_size[1] / win_size[0])
        self.alloc_circles_buffer(pts_ndc.shape[0])
        self.circles_taichi_field.from_numpy(np.array(pts_ndc))
        self.canvas.circles(self.circles_taichi_field, radius=self.camera.scale * (radius / min(win_size[0], win_size[1])), color=color)
   
    def _process_event(self):
        x, y = self.gui.get_cursor_pos()
        for e in self.gui.get_events(ti.ui.PRESS):
            self.camera.handle_press(e.key, x, y)

        for e in self.gui.get_events(ti.ui.RELEASE):
            self.camera.handle_release(e.key)

        self.camera.handle_motion(x, y)  

    def show(self):
        self.gui.show()

    def running(self):
        return self.gui.running

    def run(self, update_fn):
        while self.running():
            gui = self.gui.get_gui()
            gui.begin("Controls", 0.05, 0.05, 0.3, 0.2)
            self.radius = gui.slider_float("Particles Radius", self.radius, 1, 5)
            gui.end()

            self._process_event()
            update_fn(self)
            self.show()

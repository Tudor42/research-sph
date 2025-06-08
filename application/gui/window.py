import taichi as ti
import numpy as np
from .camera import Camera2D

class Window:
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

        self.handlers = {
            'camera_move': self.camera,
        }
        self.mode = 'camera_move'

    def set_camera(self, center, scale):
        self.camera.center = np.array(center, dtype=np.float32)
        self.camera.scale = float(scale)

    def alloc_circles_buffer(self, N):
        if self.circles_taichi_field.shape[0] != N:
            self.circles_taichi_field = ti.Vector.field(2, ti.f32, shape=N)
            self.color_field = ti.Vector.field(3, dtype=ti.f32, shape=N)

    def draw_circles(self, pts_world, radius, color=(0.0, 0.0, 0.0), size=0, per_vertex_color=None):
        if size == 0:
            return
        win_size = self.gui.get_window_shape()
        pts_ndc = self.camera.world_to_ndc(pts_world[:size], win_size[1] / win_size[0])
        self.alloc_circles_buffer(pts_ndc.shape[0])
        self.circles_taichi_field.from_numpy(np.array(pts_ndc))
        if per_vertex_color is not None:
            self.color_field.from_numpy(np.array(per_vertex_color[:size]))
            self.canvas.circles(self.circles_taichi_field, radius= radius / (min(win_size[0], win_size[1]) * self.camera.scale), color=color, per_vertex_color=self.color_field)
            return
        self.canvas.circles(self.circles_taichi_field, radius= radius / (min(win_size[0], win_size[1]) * self.camera.scale), color=color)

    def _process_event(self):
        event_list = self.gui.get_events(ti.ui.PRESS)
        for e in event_list:
            if e.key == "c":
                self.mode = 'camera_move'
            elif e.key == "b":
                self.mode = 'builder'
        h = self.handlers.get(self.mode)
        if h is None:
            return
        x, y = self.gui.get_cursor_pos()
        for e in event_list:
            if e.key == "c":
                self.mode = 'camera_move'
                return
            elif e.key == "b":
                self.mode = 'builder'
                return

            h.handle_press(e.key, x, y)

        for e in self.gui.get_events(ti.ui.RELEASE):
            h.handle_release(e.key)

        h.handle_motion(x, y)  

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

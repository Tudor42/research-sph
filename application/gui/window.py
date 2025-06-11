import taichi as ti
import numpy as np


from application.server.managers.state_manager import StateManager
from application.utils.tkinter_component import open_case_file
from .camera import Camera2D

class Window:
    def __init__(self, 
                 resolution=(800, 600), 
                 title="SPH Simulator",
                 background_color=(0.0, 0.0, 0.0),
                 fps_limit=60,
                 arch=ti.cpu, 
                 state_manager=None):
        ti.init(arch=arch)
        self.state_manager = state_manager or StateManager()

        self.gui = ti.ui.Window(title,
                                res=resolution,
                                fps_limit=fps_limit)
        self.camera = Camera2D(center=(0.5,0.5), scale=1.0)
        self.canvas = self.gui.get_canvas()
        self.canvas.set_background_color(background_color)
        self.radius = 5
        self.circles_taichi_field = None
        self.handlers = {
            'normal': self.camera,
        }
        self.mode = 'normal'
        self.run_sim = False
        self.steps_before_draw = 1

    def set_camera(self, center, scale):
        self.camera.center = np.array(center, dtype=np.float32)
        self.camera.scale = float(scale)

    def alloc_circles_buffer(self, N):
        if self.circles_taichi_field is None or self.circles_taichi_field.shape[0] != N:
            self.circles_taichi_field = ti.Vector.field(2, ti.f32, shape=N)
            self.color_field = ti.Vector.field(4, dtype=ti.f32, shape=N)

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
        event_list_release = self.gui.get_events(ti.ui.RELEASE)
        for e in event_list:
            if e.key == "n":
                self.mode = 'normal'
            elif e.key == "l" and self.mode == "normal" and not self.run_sim:
                self.mode = 'select case'
            elif e.key == "m" and self.mode == "normal" and not self.run_sim:
                self.mode = 'select model'
            elif e.key == "c" and self.mode == "select model":
                self.state_manager.select_solver("cconv")
                self.mode = "normal"
            elif e.key == "g" and self.mode == "select model":
                self.state_manager.select_solver("gns")
                self.mode = "normal"
            elif e.key == "w" and self.mode == "select model":
                self.state_manager.select_solver("wcsph")
                self.mode = "normal"
            elif e.key == 'd' and self.mode == 'select case':
                self.state_manager.select_case("db")
                self.mode = 'normal'
            elif e.key == 'f' and self.mode == 'select case':
                self.state_manager.select_case("ft2d")
                self.mode = 'normal'
            elif e.key == 'e' and self.mode == 'select case':
                self.state_manager.select_case("empty")
                self.mode = 'normal'
            elif e.key == ti.ui.SPACE and (self.mode == "normal"):
                self.run_sim = not self.run_sim
            elif e.key == 'r' and (self.mode == "normal"):
                self.state_manager.reset_scene()
            elif e.key == 'o' and self.mode == "normal":
                print(open_case_file())
        
        h = self.handlers.get(self.mode)
        if h is None:
            return
        x, y = self.gui.get_cursor_pos()
        for e in event_list:
            h.handle_press(e.key, x, y)

        for e in event_list_release:
            h.handle_release(e.key)

        h.handle_motion(x, y)  
 
    def run(self, update_fn):
        while self.gui.running:
            gui = self.gui.get_gui()
            gui.begin("Mode: " + self.mode, 0.05, 0.05, 0.3, 0.2)
            gui.text("TIME: " + str(self.state_manager.step * self.state_manager.dt))
            self.radius = gui.slider_float("Particles Radius", self.radius, 1, 10)
            self.steps_before_draw = gui.slider_int("Simulation steps before drawing", self.steps_before_draw, 1, 10)
            show_str = [f"l{name[0]}: {name}" for i, name in enumerate(self.state_manager.cases_names())]
            gui.text("Choose test case pres:\n\t" + "\n\t".join(show_str))
            show_str = [f"l{name[0]}: {name}" for name in self.state_manager.solvers_names()]
            gui.text("Choose solver:\n" + "\n\t".join([""]))
            gui.end()

            self._process_event()
            if self.run_sim:
                for _ in range(self.steps_before_draw):
                    self.state_manager.advance()
            update_fn(self)
            self.gui.show()

from omegaconf import OmegaConf
from jax_sph.defaults import defaults
from jax_sph.case_setup import load_case
from jax_sph.utils import Tag
import os
import jax.numpy as jnp

from main_jax_sph import load_embedded_configs

class CaseManager:
    def __init__(self):
        self.cases = {
            "db": load_case("cases/", "db.py"),
            "ft2d": load_case("cases/", "ft2d.py")
        }
    
    def list_names(self):
        return list(self.cases.keys())
    
    def select(self, identifier):
        case = self.cases[identifier]
        args = OmegaConf.create(dict(config=os.path.join("cases", identifier + ".yaml")))
        cfg = load_embedded_configs(args)
        self._current = case(cfg)
        (
            self.cfg,
            self.box_size,
            self.state,
            self.g_ext_fn,
            self.bc_fn,
            self.nw_fn,
            self.eos,
            self.key,
            self.displacement_fn,
            self.shift_fn,
        ) = self._current.initialize()
        self.step = 0.0
        self.state0 = self.state

    def update(self, gui):
        if self._current is None:
            return
        tags = self.state['tag']
        positions = self.state['r']
        # Filter out padding values
        valid = tags != Tag.PAD_VALUE
        tags = tags[valid]
        pts  = positions[valid]
        # split masks
        fluid_mask = tags == Tag.FLUID
        per_vertex_color = jnp.where(fluid_mask[:, None], jnp.array((0.2, 0.5, 0.9)), jnp.array((0.7, 0.7, 0.7)))

        gui.draw_circles(
            pts,
            radius=gui.radius,
            size=pts.shape[0],
            per_vertex_color=per_vertex_color,
        )

    def reset(self):
        self.step = 0.0
        self.state = self.state0
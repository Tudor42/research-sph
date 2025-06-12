from omegaconf import OmegaConf
from application.server.managers.solver_manager import create_wcsph
from jax_sph.case_setup import load_case, set_relaxation
import os
import copy

from jax_sph.io_state import io_setup, write_state
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
        self.curr_case_name = identifier
        case = self.cases[identifier]
        args = OmegaConf.create(dict(config=os.path.join("cases", identifier + ".yaml")))
        cfg = load_embedded_configs(args)
        _state = self.do_relaxation(case, cfg)
        cfg.state0_path=str(os.path.join("sim_data", "relaxed", identifier + "_2_0.02_2.h5"))
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

    def do_relaxation(self, case, cfg):
        cfg = copy.deepcopy(cfg)
        cfg.case.mode = "rlx"
        cfg.case.r0_noise_factor = 0.25
        cfg.solver.tvf = 1.0
        cfg.io.data_path=str(os.path.join("sim_data", "relaxed"))
        cfg.seed = 2
        case = set_relaxation(case, cfg)
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
        ) = case.initialize()
        cfg = self.cfg
        advance, neighbor_fn, neighbors, num_particles = create_wcsph(self)
        state = self.state
        _state, _neighbors = advance(0.0, state, neighbors)
        _state["v"].block_until_ready()
        dir = io_setup(cfg)

        for step in range(cfg.solver.sequence_length + 2):        
            write_state(step - 1, state, dir, cfg)
            state_, neighbors_ = advance(cfg.solver.dt, state, neighbors, step * cfg.solver.dt)

            if neighbors_.did_buffer_overflow:
                edges_ = neighbors.idx.shape
                print(f"Reallocate neighbors list {edges_} at step {step}")
                neighbors = neighbor_fn.allocate(state["r"], num_particles=num_particles)
                print(f"To list {neighbors.idx.shape}")

                # To run the loop N times even if sometimes did_buffer_overflow > 0
                # we directly rerun the advance step here
                state, neighbors = advance(cfg.solver.dt, state, neighbors)
            else:
                state, neighbors = state_, neighbors_

import jax
import numpy as np
from jax_sph import partition
from jax_sph.integrator import si_euler
from jax_sph.jax_md.partition import Sparse
from jax_sph.solver import WCSPH
from jax_sph.utils import Tag


class SolverManager:
    def __init__(self, case_manager):
        # map of name -> solver factory
        self.solvers = {
            "wcsph": WCSPH,
            # other solvers can be added here
        }
        self._current_name = None
        self._current_solver = None
        self.case_manager = case_manager
        self.advance = None
        self.dt = 0.0

    def list_names(self):
        return list(self.solvers.keys())

    def select(self, identifier):
        if isinstance(identifier, int):
            key = list(self.solvers.keys())[identifier]
        else:
            key = identifier
        self.curr_solver_key = key
        self.step = 0
    
    def get_next_state(self):
        state_, neighbors_ = self.advance(self.case_manager.cfg.solver.dt, self.case_manager.state, self.neighbors, self.case_manager.step * self.case_manager.cfg.solver.dt)

        # Check whether the edge list is too small and if so, create longer one
        if neighbors_.did_buffer_overflow:
            edges_ = self.neighbors.idx.shape
            print(f"Reallocate neighbors list {edges_} at step {self.case_manager.step}")
            self.neighbors = self.neighbor_fn.allocate(self.case_manager.state["r"], num_particles=self.num_particles)
            print(f"To list {self.neighbors.idx.shape}")

            # To run the loop N times even if sometimes did_buffer_overflow > 0
            # we directly rerun the advance step here
            self.case_manager.state, self.neighbors = self.advance(self.case_manager.cfg.solver.dt, self.case_manager.state, self.neighbors)
        else:
            self.case_manager.state, self.neighbors = state_, neighbors_
        self.case_manager.step += 1

    def init_solver(self):
        if self.curr_solver_key == "wcsph":
            solver = WCSPH(
                self.case_manager.displacement_fn,
                self.case_manager.eos,
                self.case_manager.g_ext_fn,
                self.case_manager.cfg.case.dx,
                self.case_manager.cfg.case.dim,
                self.case_manager.cfg.solver.dt,
                self.case_manager.cfg.case.c_ref,
                self.case_manager.cfg.solver.eta_limiter,
                self.case_manager.cfg.solver.diff_delta,
                self.case_manager.cfg.solver.diff_alpha,
                self.case_manager.cfg.solver.name,
                self.case_manager.cfg.kernel.name,
                self.case_manager.cfg.kernel.h_factor,
                self.case_manager.cfg.solver.is_bc_trick,
                self.case_manager.cfg.solver.density_evolution,
                self.case_manager.cfg.solver.artificial_alpha,
                self.case_manager.cfg.solver.free_slip,
                self.case_manager.cfg.solver.density_renormalize,
                self.case_manager.cfg.solver.heat_conduction,
            )
            forward = solver.forward_wrapper()
            self.neighbor_fn = partition.neighbor_list(
                self.case_manager.displacement_fn,
                self.case_manager.box_size,
                r_cutoff=solver._kernel_fn.cutoff,
                backend=self.case_manager.cfg.nl.backend,
                capacity_multiplier=1.25,
                mask_self=False,
                format=Sparse,
                num_particles_max=self.case_manager.state["r"].shape[0],
                num_partitions=self.case_manager.cfg.nl.num_partitions,
                pbc=np.array(self.case_manager.cfg.case.pbc),
            )
            self.num_particles = (self.case_manager.state["tag"] != Tag.PAD_VALUE).sum()
            self.neighbors = self.neighbor_fn.allocate(self.case_manager.state["r"], num_particles=self.num_particles)
            self.advance = jax.jit(si_euler(self.case_manager.cfg.solver.tvf, forward, self.case_manager.shift_fn, self.case_manager.bc_fn, self.case_manager.nw_fn))
            self.dt = self.case_manager.cfg.solver.dt

import jax
import jax.numpy as jnp
import jmp
import numpy as np
from omegaconf import OmegaConf
from jax_sph import partition
from jax_sph.integrator import si_euler
from jax_sph.jax_md.partition import Sparse
from jax_sph.solver import WCSPH
from jax_sph.utils import Tag
import os
from jax import config
import haiku as hk
from lagrangebench import models
from lagrangebench.utils import load_haiku
from main import load_embedded_configs

class SolverManager:
    def __init__(self, case_manager):
        self.solvers = {
            "wcsph": WCSPH,
            "cconv": None,
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

        if neighbors_.did_buffer_overflow:
            edges_ = self.neighbors.idx.shape
            print(f"Reallocate neighbors list {edges_} at step {self.case_manager.step}")
            self.neighbors = self.neighbor_fn.allocate(self.case_manager.state["r"], num_particles=self.num_particles)
            print(f"To list {self.neighbors.idx.shape}")

            self.case_manager.state, self.neighbors = self.advance(self.case_manager.cfg.solver.dt, self.case_manager.state, self.neighbors, self.case_manager.step * self.case_manager.cfg.solver.dt)
        else:
            self.case_manager.state, self.neighbors = state_, neighbors_
        self.case_manager.step += 1

    def init_solver(self):
        if self.curr_solver_key == "wcsph":
            self.create_wcsph()
        elif self.curr_solver_key == "cconv":
            config_path = os.path.join("ckp/cconv_FT2D_every250_20250607-200713", "config.yaml")

            cli_args = OmegaConf.create(dict(gpu=0, load_ckp="ckp/cconv_FT2D_every250_20250607-200713"))
            cli_args.xla_mem_fraction = 0.75
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cli_args.xla_mem_fraction)


            cfg = load_embedded_configs(config_path, cli_args)
            if cfg.dtype == "float64":
                config.update("jax_enable_x64", True)

            self.num_particles = (self.case_manager.state["tag"] != Tag.PAD_VALUE).sum()
            self.generate_input_seq(cfg.model.input_seq_length)

            load_ckp = cfg.load_ckp

            default_connectivity_radius = 1.45 * self.case_manager.cfg.case.dx
            
            def model_fn(x):
                return models.MyParticleNetwork(radius=default_connectivity_radius, num_particles=self.num_particles)(x, isTraining=False)
            MODEL = models.MyParticleNetwork
            model = hk.without_apply_rng(hk.transform_with_state(model_fn))

            policy = jmp.get_policy("params=float32,compute=float32,output=float32")
            hk.mixed_precision.set_policy(MODEL, policy)
            self.model_apply = jax.jit(model.apply)
            params, _, _, _ = load_haiku(load_ckp)
            self.model_params = params["model"]
            
            self.neighbor_fn = partition.neighbor_list(
                self.case_manager.displacement_fn,
                self.case_manager.box_size,
                r_cutoff=default_connectivity_radius,
                backend=self.case_manager.cfg.nl.backend,
                capacity_multiplier=1.25,
                mask_self=cfg.model.mask_self,
                format=Sparse,
                num_particles_max=self.case_manager.state["r"].shape[0],
                num_partitions=self.case_manager.cfg.nl.num_partitions,
                pbc=np.array(self.case_manager.cfg.case.pbc),
            )
            self.neighbors = self.neighbor_fn.allocate(self.case_manager.state["r"], num_particles=self.num_particles)

    def generate_input_seq(self, input_seq_length):
        self.create_wcsph()
        seq = []
        state, neighbors = self.case_manager.state, self.neighbors
        for _ in range(input_seq_length):
            seq.append(state["r"])
            self.case_manager.state = state
            self.neighbors = neighbors
            state_, neighbors_ = self.advance(self.case_manager.cfg.solver.dt, state, neighbors, self.case_manager.step * self.case_manager.cfg.solver.dt)
            if neighbors_.did_buffer_overflow:
                edges_ = neighbors.idx.shape
                print(f"Reallocate neighbors list {edges_} at step {self.case_manager.step}")
                neighbors = self.neighbor_fn.allocate(self.case_manager.state["r"], num_particles=self.num_particles)
                print(f"To list {self.neighbors.idx.shape}")

                state, neighbors = self.advance(self.case_manager.cfg.solver.dt, state, neighbors, self.case_manager.step * self.case_manager.cfg.solver.dt)
            else:
                state, neighbors = state_, neighbors_
        self.input_seq0 = jnp.stack(seq, axis=0)
        self.step = input_seq_length - 1

    def create_wcsph(self):
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
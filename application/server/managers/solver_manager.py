from functools import partial
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from omegaconf import OmegaConf
from jax_sph import partition
from jax_sph.integrator import si_euler
from jax_sph.jax_md import space
from jax_sph.jax_md.partition import Sparse
from jax_sph.solver import WCSPH
from jax_sph.utils import Tag
import os
from jax import config
import haiku as hk
from lagrangebench import models
from lagrangebench.utils import get_kinematic_mask, load_haiku
from main import load_embedded_configs

class SolverManager:
    def __init__(self):
        self.solvers = {"wcsph", "cconv", "gns"}
        self.curr_solver_name = None
        self.model_cfg = None
        self.is_solver_initialized = False

    def list_names(self):
        return list(self.solvers)

    def select(self, identifier):
        if isinstance(identifier, int):
            identifier = list(self.solvers)[identifier]
        self.curr_solver_name = identifier
        if self.curr_solver_name == "cconv":
            self.model_cfg = get_model_cfg("ckp/cconv_FT2D_every250_20250607-200713/best")
        elif self.curr_solver_name == "gns":
            self.model_cfg = None
        else:
            self.model_cfg = None
        self.is_solver_initialized = False

    def init_solver(self, case_manager):
        self.displacement_fn_vmap = jax.vmap(case_manager.displacement_fn, in_axes=(0, 0))
        self.seq0 = None
        if self.curr_solver_name == "wcsph":
            self.advance, self.neighbor_fn, self.neighbors, self.num_particles = create_wcsph(case_manager)
        elif self.curr_solver_name == "cconv":
            self.input_seq_length = self.model_cfg.model.input_seq_length
            self.select("wcsph")
            self.init_solver(case_manager)
            state = case_manager.state
            seq = [state['r']]
            for step in range(self.input_seq_length - 1):
                state = self.next(case_manager, step, state)
                seq.append(state['r'])
            self.seq0 = jnp.stack(seq, axis=1)
            self.select("cconv")
            self.model_apply, self.model_params, self.model_state, self.neighbor_fn, self.neighbors, self.input_seq_length, self.num_particles = create_cconv(case_manager, self.model_cfg)
        elif self.curr_solver_name == "gns":
            self.input_seq_length = self.model_cfg.model.input_seq_length
            self.select("wcsph")
            self.init_solver(case_manager)
            state = case_manager.state
            seq = [state['r']]
            for step in range(self.input_seq_length - 1):
                state = self.next(case_manager, step, state)
                seq.append(state['r'])
            self.seq0 = jnp.stack(seq, axis=1)
        self.neighbors0 = self.neighbors

    def next(self, case_manager, step, state):
        if not self.is_solver_initialized:
            self.init_solver(case_manager)
            self.is_solver_initialized = True
        if step == 0:
            self.neighbors = self.neighbors0
            self.seq = self.seq0

        if self.curr_solver_name == "wcsph":
            state_, neighbors_ = self.advance(case_manager.cfg.solver.dt, state, self.neighbors, step * case_manager.cfg.solver.dt)
            if neighbors_.did_buffer_overflow:
                edges_ = self.neighbors.idx.shape
                print(f"Reallocate neighbors list {edges_} at step {step}")
                self.neighbors = self.neighbor_fn.allocate(state["r"], num_particles=self.num_particles)
                print(f"To list {self.neighbors.idx.shape}")

                state, self.neighbors = self.advance(case_manager.cfg.solver.dt, state, self.neighbors, step * case_manager.cfg.solver.dt)
            else:
                state, self.neighbors = state_, neighbors_
            return state
        elif self.curr_solver_name == "cconv":
            return self.advance_nn_model(case_manager, step, state)


    def advance_nn_model(self, case_manager, step, state):
        dt = case_manager.cfg.solver.dt * 500
        if self.seq.shape[1] > step:
            state["r"] = self.seq[:, step]
            return state
        features, neighbors_ = self.create_features(self.seq, state, self.neighbors, case_manager.g_ext_fn, case_manager.bc_fn, case_manager.shift_fn, case_manager.cfg, step)
        non_kinematic_mask = jnp.logical_not(get_kinematic_mask(state["tag"]))[:, None]

        if neighbors_.did_buffer_overflow:
            edges_ = self.neighbors.idx.shape
            print(f"Reallocate neighbors list {edges_} at step {step}")
            self.neighbors = self.neighbor_fn.allocate(features["abs_pos"][:, 0], num_particles=self.num_particles)
            print(f"To list {self.neighbors.idx.shape}")
            features,  self.neighbors = self.create_features(self.seq, state, self.neighbors, case_manager.g_ext_fn, case_manager.bc_fn, case_manager.shift_fn, case_manager.cfg, step)
        else:
            self.neighbors = neighbors_
        pred, self.model_state = self.model_apply(self.model_params, self.model_state, (features, state["tag"]))

        state["u"] = jnp.where(non_kinematic_mask, self.displacement_fn_vmap(pred["pos"], state["r"]) / dt, state["u"])
        state["r"] = jnp.where(non_kinematic_mask, pred["pos"], state["r"])
        
        tail = self.seq[:, 1:, :]
        self.seq = jnp.concatenate([tail, state["r"][:, None, :]], axis=1)

        return state

    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7))
    def create_features(self, position_seq, state, neighbors, g_ext_fn, bc_fn, shift_fn, cfg, step):
        dt = cfg.solver.dt * 100
        tvf = cfg.solver.tvf
        forces = g_ext_fn(position_seq[:, -1], cfg.solver.dt * step)
        non_kinematic_mask = jnp.logical_not(get_kinematic_mask(state["tag"]))[:, None]

        most_recent_positions = position_seq[:, -1]
        vel1 = self.displacement_fn_vmap(position_seq[:, -1], position_seq[:, -2]) / dt

        vel2_candidate = vel1 + cfg.solver.dt * forces
        pos2_candidate = shift_fn(most_recent_positions, cfg.solver.dt * (vel2_candidate + vel1) / 2.0)
        
        state = bc_fn(state, cfg.solver.dt * step)
        state["u"] += 1.0 * dt * state["dudt"]
        state["v"] = state["u"] + tvf * 0.5 * dt * state["dvdt"]

        wall_pos = shift_fn(state["r"], 1.0 * dt * state["v"])
        
        most_recent_position = jnp.where(non_kinematic_mask, pos2_candidate, wall_pos)

        neighbors = neighbors.update(
            most_recent_position, num_particles=self.num_particles
        )
       
        features = {}
        features["abs_pos"] = most_recent_position[:, None]
        features["vel2_candidates"] = jnp.where(non_kinematic_mask, vel2_candidate, self.displacement_fn_vmap(most_recent_position, self.seq[:, -1]) / dt)        
        receivers, senders = neighbors.idx
        features["senders"] = senders
        features["receivers"] = receivers
        displacement = self.displacement_fn_vmap(
            most_recent_position[receivers], most_recent_position[senders]
        )
        normalized_relative_displacements = displacement / (1.45 * cfg.case.dx)
        features["rel_disp"] = normalized_relative_displacements
        normalized_relative_distances = space.distance(
            normalized_relative_displacements
        )
        features["rel_dist"] = normalized_relative_distances[:, None]
        displacement = self.displacement_fn_vmap(
                position_seq[senders, -1], position_seq[receivers, -1]
        )
        normalized_relative_displacements = displacement / (1.45 * cfg.case.dx)
        features["rel_disp_from_prev_time"] = normalized_relative_displacements
        return features, neighbors

def get_model_cfg(ckp_directory):
    config_path = os.path.join(ckp_directory, "config.yaml")

    cli_args = OmegaConf.create(dict(gpu=0, load_ckp=ckp_directory))
    cli_args.xla_mem_fraction = 0.75
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cli_args.xla_mem_fraction)

    return load_embedded_configs(config_path, cli_args)

def create_cconv(case_manager, cfg):
    num_particles = (case_manager.state["tag"] != Tag.PAD_VALUE).sum()
    # self.generate_input_seq(cfg.model.input_seq_length)

    load_ckp = cfg.load_ckp
    default_connectivity_radius = 1.45 * case_manager.cfg.case.dx
    
    def model_fn(x):
        return models.MyParticleNetwork(radius=default_connectivity_radius, num_particles=num_particles)(x, isTraining=False)
    MODEL = models.MyParticleNetwork
    model = hk.without_apply_rng(hk.transform_with_state(model_fn))

    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)
    model_apply = jax.jit(model.apply)
    params, model_state, _, _ = load_haiku(load_ckp)
    model_params = params["model"]
    
    neighbor_fn = partition.neighbor_list(
        case_manager.displacement_fn,
        case_manager.box_size,
        r_cutoff=default_connectivity_radius,
        backend=case_manager.cfg.nl.backend,
        capacity_multiplier=1.25,
        mask_self=cfg.model.mask_self,
        format=Sparse,
        num_particles_max=case_manager.state["r"].shape[0],
        num_partitions=case_manager.cfg.nl.num_partitions,
        pbc=np.array(case_manager.cfg.case.pbc),
    )
    neighbors = neighbor_fn.allocate(case_manager.state["r"], num_particles=num_particles)
    return model_apply, model_params, model_state, neighbor_fn, neighbors, cfg.model.input_seq_length, num_particles

def create_wcsph(case_manager):
    solver = WCSPH(
        case_manager.displacement_fn,
        case_manager.eos,
        case_manager.g_ext_fn,
        case_manager.cfg.case.dx,
        case_manager.cfg.case.dim,
        case_manager.cfg.solver.dt,
        case_manager.cfg.case.c_ref,
        case_manager.cfg.solver.eta_limiter,
        case_manager.cfg.solver.diff_delta,
        case_manager.cfg.solver.diff_alpha,
        case_manager.cfg.solver.name,
        case_manager.cfg.kernel.name,
        case_manager.cfg.kernel.h_factor,
        case_manager.cfg.solver.is_bc_trick,
        case_manager.cfg.solver.density_evolution,
        case_manager.cfg.solver.artificial_alpha,
        case_manager.cfg.solver.free_slip,
        case_manager.cfg.solver.density_renormalize,
        case_manager.cfg.solver.heat_conduction,
    )
    forward = solver.forward_wrapper()
    neighbor_fn = partition.neighbor_list(
        case_manager.displacement_fn,
        case_manager.box_size,
        r_cutoff=solver._kernel_fn.cutoff,
        backend=case_manager.cfg.nl.backend,
        capacity_multiplier=1.25,
        mask_self=False,
        format=Sparse,
        num_particles_max=case_manager.state["r"].shape[0],
        num_partitions=case_manager.cfg.nl.num_partitions,
        pbc=np.array(case_manager.cfg.case.pbc),
    )
    num_particles = (case_manager.state["tag"] != Tag.PAD_VALUE).sum()
    neighbors = neighbor_fn.allocate(case_manager.state["r"], num_particles=num_particles)
    advance = jax.jit(si_euler(case_manager.cfg.solver.tvf, forward, case_manager.shift_fn, case_manager.bc_fn, case_manager.nw_fn))
    # dt = case_manager.cfg.solver.dt
    return advance, neighbor_fn, neighbors, num_particles

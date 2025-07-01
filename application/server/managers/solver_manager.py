from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from application.utils.create_solvers import create_model, create_wcsph
from jax_sph.jax_md import space
import os
from lagrangebench.data.utils import get_dataset_stats
from lagrangebench.utils import get_kinematic_mask
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
            self.model_cfg = get_model_cfg("ckp/cconv_dam2d_20250630-232100/best")
        elif self.curr_solver_name == "gns":
            self.model_cfg = get_model_cfg("ckp/gns_dam2d_20250607-003352/best")
        else:
            self.model_cfg = None
        self.is_solver_initialized = False

    def init_solver(self, case_manager):
        if self.curr_solver_name == "wcsph":
            self.advance, self.neighbor_fn, self.neighbors, self.num_particles = create_wcsph(case_manager)
        else:
            self._warmup(case_manager)
            self._init_nn(case_manager)
        self.is_solver_initialized = True

    def _warmup(self, case_manager):
        T = int(self.model_cfg.model.input_seq_length)
        prev = self.curr_solver_name
        self.select("wcsph")
        self.init_solver(case_manager)
        state = jax.tree_util.tree_map(lambda x: jnp.array(x), case_manager.state)
        seq = []
        for step in range(1, T * 100 + 1):
            state_, neighbors_ = self.advance(case_manager.cfg.solver.dt, state, self.neighbors, step * case_manager.cfg.solver.dt)
            if neighbors_.did_buffer_overflow:
                edges_ = self.neighbors.idx.shape
                print(f"Reallocate neighbors list {edges_} at step {step}")
                self.neighbors = self.neighbor_fn.allocate(state["r"], num_particles=self.num_particles)
                print(f"To list {self.neighbors.idx.shape}")

                state, self.neighbors = self.advance(case_manager.cfg.solver.dt, state, self.neighbors, step * case_manager.cfg.solver.dt)
            else:
                state, self.neighbors = state_, neighbors_
            if step % 100 == 0:
                seq.append(state["r"])
        # stack into (N, T, dim)
        self.seq0 = jnp.stack(seq, axis=1)
        # restore original solver selection
        self.select(prev)
        self.mask = None

    def _init_nn(self, case_manager):
        num_particles = None
        if case_manager.curr_case_name == "db":
            L, H = 5.366, 2.0
            r = self.seq0[:, 0]
            tag = case_manager.state["tag"]
            mask_bottom = np.where(r[:, 1] < 2 * case_manager.cfg.case.dx, False, True)
            mask_lid = np.where(r[:, 1] > H + 4 * case_manager.cfg.case.dx, False, True)
            mask_left = np.where(
                ((r[:, 0] < 2 * case_manager.cfg.case.dx) * (tag == 1)), False, True
            )
            mask_right = np.where(
                (r[:, 0] > L + 4 * case_manager.cfg.case.dx) * (tag == 1), False, True
            )
            mask = mask_bottom * mask_lid * mask_left * mask_right
            
            self.seq0 = self.seq0[mask]
            self.mask = mask
            num_particles =  mask.sum()
        else: 
            self.mask = None
        self.model_apply, self.model_params, self.model_state, self.neighbor_fn, self.neighbors, self.input_seq_length, self.num_particles = create_model(case_manager, self.model_cfg, self.curr_solver_name, num_particles)
        self.integrate_fn = get_integrate_func(case_manager.displacement_fn, case_manager.shift_fn)

    def next(self, case_manager, step, state):
        if not self.is_solver_initialized:
            self.init_solver(case_manager)
            self.is_solver_initialized = True
        if step == 0 and self.curr_solver_name != "wcsph":
            if  self.mask is not None:
                def _maybe_mask(x):
                    if isinstance(x, jnp.ndarray) and x.ndim > 0 and x.shape[0] == self.mask.shape[0]:
                        return x[self.mask]
                    else:
                        return x

                state = jax.tree_util.tree_map(_maybe_mask, state)
            self.seq = jnp.array(self.seq0)

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
        else:
            return self.advance_nn_model(case_manager, step, state)


    def advance_nn_model(self, case_manager, step, state):
        dt = case_manager.cfg.solver.dt * 100

        features, neighbors_ = self.create_features(case_manager.g_ext_fn, case_manager.shift_fn, case_manager.displacement_fn, case_manager.cfg, self.curr_solver_name, self.neighbors, step, self.seq, state['tag'])
        
        non_kinematic_mask = jnp.logical_not(get_kinematic_mask(state["tag"]))[:, None]

        if neighbors_.did_buffer_overflow:
            edges_ = self.neighbors.idx.shape
            print(f"Reallocate neighbors list {edges_} at step {step}")
            self.neighbors = self.neighbor_fn.allocate(features["abs_pos"][:, -1], num_particles=self.num_particles)
            print(f"To list {self.neighbors.idx.shape}")
            features, neighbors_ = self.create_features(case_manager.g_ext_fn, case_manager.shift_fn, case_manager.displacement_fn, case_manager.cfg, self.curr_solver_name, self.neighbors, step, self.seq, state['tag'])
        else:
            self.neighbors = neighbors_
        pred, self.model_state = self.model_apply(self.model_params, self.model_state, (features, state["tag"]))
        pos = self.integrate_fn(pred, self.seq)
        state["u"] = jnp.where(non_kinematic_mask, jax.vmap(case_manager.displacement_fn, in_axes=(0, 0))(pos, self.seq[:, -1]) / dt, jax.vmap(case_manager.displacement_fn, in_axes=(0, 0))(features["abs_pos"][:, 0], self.seq[:, -1]) / dt)
        state["r"] = jnp.where(non_kinematic_mask, pos, features["abs_pos"][:, -1])
        r_copy = jnp.array(state["r"])
        tail = self.seq[:, 1:, :]
        self.seq = jnp.concatenate([tail, r_copy[:, None, :]], axis=1)

        return state

    @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
    def create_features(self, g_ext_fn, shift_fn, displacement_fn, cfg, model_name, neighbors, step, position_seq, tags,):
        dt = cfg.solver.dt * 100
        forces = g_ext_fn(position_seq[:, -1], dt * step)
        non_kinematic_mask = jnp.logical_not(get_kinematic_mask(tags))[:, None]
        default_connectivity_radius = 0.029
        normalization_stats = {
            "vel_mean": jnp.array([
                0.0040724738501012325,
                -0.0007674989756196737
            ]),
            "vel_std": jnp.array([
                0.0125808697193861,
                0.005147312767803669
            ]),
            "acc_mean": jnp.array([
                -1.4104562978900503e-05,
                1.7534126754981116e-06
            ]),
            "acc_std": jnp.array([
                0.0006388923502527177,
                0.0007237975369207561
            ])
        }
        normalization_stats = get_dataset_stats(normalization_stats, False, 0.001)
        displacement_fn_vmap = jax.vmap(displacement_fn, in_axes=(0, 0))    
        displacement_fn_dvmap = jax.vmap(displacement_fn_vmap, in_axes=(0, 0))
        if model_name == "cconv":
            most_recent_positions = position_seq[:, -1]
            vel1 = displacement_fn_vmap(position_seq[:, -1], position_seq[:, -2]) / dt
            vel2_candidate = vel1 + dt * forces
            pos2_candidate = shift_fn(most_recent_positions, dt * (vel2_candidate + vel1) / 2.0)

            most_recent_position = jnp.where(non_kinematic_mask, pos2_candidate, most_recent_positions)
            neighbors = neighbors.update(
                most_recent_position, num_particles=self.num_particles
            )  

            features = {}
            features["abs_pos"] = most_recent_position[:, None]
            features["vel2_candidates"] = jnp.where(non_kinematic_mask, vel2_candidate, vel1)        
            receivers, senders = neighbors.idx
            features["senders"] = senders
            features["receivers"] = receivers
            displacement = displacement_fn_vmap(
                most_recent_position[senders], most_recent_position[receivers]
            )
            normalized_relative_displacements = displacement / default_connectivity_radius
            features["rel_disp"] = normalized_relative_displacements
            normalized_relative_distances = space.distance(
                normalized_relative_displacements
            )
            features["rel_dist"] = normalized_relative_distances[:, None]
            displacement = displacement_fn_vmap(
                    position_seq[senders, -1], position_seq[receivers, -1]
            )
            normalized_relative_displacements = displacement / default_connectivity_radius
            features["rel_disp_from_prev_time"] = normalized_relative_displacements
        else:
            features = {}
            n_total_points = position_seq.shape[0]
            most_recent_position = position_seq[:, -1] 
            velocity_sequence = displacement_fn_dvmap(position_seq[:, 1:], position_seq[:, :-1])
            velocity_stats = normalization_stats["velocity"]
            normalized_velocity_sequence = (
                velocity_sequence - velocity_stats["mean"]
            ) / velocity_stats["std"]
            flat_velocity_sequence = normalized_velocity_sequence.reshape(
                n_total_points, -1
            )
            features["abs_pos"] = position_seq
            features["vel_hist"] = flat_velocity_sequence
            
            neighbors = neighbors.update(
                position_seq[:, -1], num_particles=self.num_particles
            )
            receivers, senders = neighbors.idx
            features["senders"] = senders
            features["receivers"] = receivers
            displacement = displacement_fn_vmap(
                most_recent_position[receivers], most_recent_position[senders]
            )
            normalized_relative_displacements = displacement / default_connectivity_radius
            features["rel_disp"] = normalized_relative_displacements

            normalized_relative_distances = space.distance(
                normalized_relative_displacements
            )
            features["rel_dist"] = normalized_relative_distances[:, None]
            features["force"] = forces
        features = jax.tree_util.tree_map(lambda f: f, features)
        return features, neighbors

def get_model_cfg(ckp_directory):
    config_path = os.path.join(ckp_directory, "config.yaml")

    cli_args = OmegaConf.create(dict(gpu=0, load_ckp=ckp_directory))
    cli_args.xla_mem_fraction = 0.75
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cli_args.xla_mem_fraction)

    return load_embedded_configs(config_path, cli_args)

def get_integrate_func(displacement_fn, shift_fn):
    displacement_fn_set = jax.vmap(displacement_fn, in_axes=(0, 0))

    @jax.jit
    def integrate_fn(normalized_in, position_sequence):
        """Euler integrator to get position shift."""
        assert any([key in normalized_in for key in ["pos", "vel", "acc"]])

        normalization_stats = {
            "vel_mean": jnp.array([
                0.0040724738501012325,
                -0.0007674989756196737
            ]),
            "vel_std": jnp.array([
                0.0125808697193861,
                0.005147312767803669
            ]),
            "acc_mean": jnp.array([
                -1.4104562978900503e-05,
                1.7534126754981116e-06
            ]),
            "acc_std": jnp.array([
                0.0006388923502527177,
                0.0007237975369207561
            ])
        }
        normalization_stats = get_dataset_stats(normalization_stats, False, 0.001)
        if "pos" in normalized_in:
            # Zeroth euler step
            return normalized_in["pos"]
        else:
            most_recent_position = position_sequence[:, -1]
            if "vel" in normalized_in:
                # invert normalization
                velocity_stats = normalization_stats["velocity"]
                new_velocity = velocity_stats["mean"] + (
                    normalized_in["vel"] * velocity_stats["std"]
                )
            elif "acc" in normalized_in:
                # invert normalization.
                acceleration_stats = normalization_stats["acceleration"]
                acceleration = acceleration_stats["mean"] + (
                    normalized_in["acc"] * acceleration_stats["std"]
                )
                # Second Euler step
                most_recent_velocity = displacement_fn_set(
                    most_recent_position, position_sequence[:, -2]
                )
                new_velocity = most_recent_velocity + acceleration  # * dt = 1

            # First Euler step
            return shift_fn(most_recent_position, new_velocity)
    return integrate_fn
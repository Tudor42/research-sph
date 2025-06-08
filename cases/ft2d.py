import jax
import numpy as np
from omegaconf import DictConfig
from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag, pos_box_2d, pos_box_3d, pos_init_cartesian_2d, pos_init_cartesian_3d
import jax.numpy as jnp
from jax.experimental import io_callback

class FT2D(SimulationSetup):
    """FuelTank"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        if cfg.case.dim != 2:
            raise Exception("Only 2D dimension")
        self.time_tilts = jnp.array(self.special.time_tilts)
        self.angles = jnp.append(jnp.array(self.special.angles) / (self.time_tilts[1:] - self.time_tilts[:-1]), jnp.array([0]))
        self.initial_acceleration = self.special.initial_plane_acceleration
        self.next_tilt = 0
        self.frame = jnp.int32(0)
        if self.case.mode == "rlx":
            self._set_default_rlx()

        if self.case.r0_type == "relaxed":
            self._load_only_fluid = True
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0

    def _box_size2D(self, n_walls):
        dx2n = self.case.dx * n_walls * 2
        sp = self.special
        self.space_size = sp.L_wall + sp.H_wall + dx2n
        return np.array([self.space_size, self.space_size])

    def _box_size3D(self, n_walls):
        dx2n = self.case.dx * n_walls * 2
        sp = self.special
        return np.array([sp.L, sp.H + dx2n, 0.4])

    def _init_walls_2d(self, dx, n_walls):
        sp = self.special
        rw = pos_box_2d(np.array([sp.L_wall, sp.H_wall]), dx, n_walls)
        return rw

    def _init_walls_3d(self, dx, n_walls):
        rw = pos_box_3d(np.array([10, 10, 10]), dx, n_walls)
        return rw

    def _init_pos2D(self, box_size, dx, n_walls):
        sp = self.special
        dx2n = self.case.dx * n_walls * 2
        offset = jnp.array([(self.space_size - sp.L_wall - dx2n) / 2, (self.space_size - sp.H_wall - dx2n) / 2])
        # initialize walls
        r_w = self._init_walls_2d(dx, n_walls) + offset
        r_f = offset + n_walls * dx + pos_init_cartesian_2d(np.array([sp.L_wall, sp.H]), dx)
        
        self.rotation_center = r_w.mean(axis=0)

        tag_f = jnp.full(len(r_f), Tag.FLUID, dtype=int)
        tag_w = jnp.full(len(r_w), Tag.MOVING_WALL, dtype=int)

        r = jax.numpy.asarray(np.concatenate([r_w, r_f]))
        tag = jax.numpy.asarray(np.concatenate([tag_w, tag_f]))
        return r, tag

    def _init_pos3D(self, box_size, dx, n_walls):
        sp = self.special

        # initialize fluid phase
        r_f = np.array([0.0, 1.0, 0.0]) * n_walls * dx + pos_init_cartesian_3d(
            np.array([sp.L, sp.H, 0.4]), dx
        )

        # initialize walls
        r_w = self._init_walls_3d(dx, n_walls)

        # set tags
        tag_f = jnp.full(len(r_f), Tag.FLUID, dtype=int)
        tag_w = jnp.full(len(r_w), Tag.SOLID_WALL, dtype=int)

        r = np.concatenate([r_w, r_f])
        tag = np.concatenate([tag_w, tag_f])

        # set velocity wall tag
        box_size = self._box_size3D(n_walls)
        mask_lid = r[:, 1] > (box_size[1] - n_walls * self.case.dx)
        tag = jnp.where(mask_lid, Tag.MOVING_WALL, tag)
        return r, tag
    
    def _offset_vec(self):
        dim = self.cfg.case.dim
        if dim == 2:
            res = np.ones(dim) * self.cfg.solver.n_walls * self.cfg.case.dx
        elif dim == 3:
            res = np.array([1.0, 1.0, 0.0]) * self.cfg.solver.n_walls * self.cfg.case.dx
        return res

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r, time: float = 0.0):
        res = jnp.zeros_like(r)
        res = res.at[:, 1].set(-self.case.g_ext_magnitude)
        
        res = jax.lax.cond(time < self.time_tilts[0], lambda: res.at[:, 0].set(-self.initial_acceleration), lambda: res)
        return res


    def _boundary_conditions_fn(self, state, time: float = 0.0):
        def scan_body(carry, next_time):
            curr_tilt = jax.lax.cond(time >= next_time[0], lambda: jnp.array([next_time[0], next_time[1]]), lambda: carry)
            return curr_tilt, 1

        curr_tilt, _ = jax.lax.scan(scan_body, jnp.array([0.0, 0.0]), jnp.stack([self.time_tilts, self.angles], axis=1))
        wall_angular_velocity = curr_tilt[1]

        mask1 = state["tag"][:, None] == Tag.MOVING_WALL
        #temp = rotate_around_point(state["r"], self.rotation_center, wall_angular_velocity) - state["r"]
        temp = rigid_body_velocity(state["r"], self.rotation_center, wall_angular_velocity)
        
        state["u"] = jnp.where(mask1, temp, state["u"])
        state["v"] = jnp.where(mask1, temp, state["v"])


        state["dudt"] = jnp.where(mask1, 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask1, 0.0, state["dvdt"])
    
        return state

def rigid_body_velocity(points, center, omega):
    rel_pos = points - center  
    return omega * jnp.stack([-rel_pos[:, 1], rel_pos[:, 0]], axis=1)
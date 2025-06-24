from jax_sph.utils import Tag
from lagrangebench import models
from lagrangebench.utils import NodeType
import jax
import jmp
import numpy as np
from jax_sph import partition
from jax_sph.integrator import si_euler
from jax_sph.jax_md.partition import Sparse
from jax_sph.solver import WCSPH
from jax_sph.utils import Tag
import haiku as hk
from lagrangebench import models
from lagrangebench.utils import NodeType, load_haiku

def create_model(case_manager, cfg, model_name, num_particles):
    count_array = (case_manager.state["tag"] != Tag.PAD_VALUE).sum()

    if hasattr(count_array, "block_until_ready"):
        count_array = count_array.block_until_ready()
    if num_particles is None:
        num_particles = int(count_array.item())

    load_ckp = cfg.load_ckp
    default_connectivity_radius = 0.029
        
    if model_name == "gns":
        def model_fn(x, isTraining=False):
            return models.GNS(
                particle_dimension=2,
                latent_size=cfg.model.latent_dim,
                blocks_per_step=cfg.model.num_mlp_layers,
                num_mp_steps=cfg.model.num_mp_steps,
                num_particle_types=NodeType.SIZE,
                particle_type_embedding_size=16,
            )(x, isTraining)

        MODEL = models.GNS
    elif model_name == "cconv":
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
        num_particles_max=num_particles,
        num_partitions=case_manager.cfg.nl.num_partitions,
        pbc=np.array(case_manager.cfg.case.pbc),
    )
    neighbors = neighbor_fn.allocate(case_manager.state["r"][:num_particles], num_particles=num_particles)
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

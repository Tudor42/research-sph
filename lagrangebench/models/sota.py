from .cconv import CConvLayer, ASCC
import jax.numpy as jnp
import haiku as hk
import jax
from .base import BaseModel
from typing import Dict, Tuple
from jax_sph.utils import Tag
from ..utils_extra.continous_convolution import window_function_batched

class AFF(hk.Module):
    """
    Attention-based Feature Fusion (AFF) module using CConv or ASCC.
    """
    def __init__(
        self,
        channels: int = 32,
        inter_channels: int = 64,
        conv_type: str = 'cconv',
        num_particles: int = 0,
        name=None,
    ):
        super().__init__(name=name)

        if conv_type == 'cconv':
            self.ConvLayer = CConvLayer
        else:
            self.ConvLayer = ASCC
        self.cconv1 = self.ConvLayer(
            in_ch=channels*2,
            out_ch=inter_channels,
            kernel_size=(4,4),
            aggregation_points=num_particles,
        )
        self.bn1 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
        self.cconv2 = self.ConvLayer(
            in_ch=inter_channels,
            out_ch=channels,
            kernel_size=(4,4),
            aggregation_points=num_particles,
        )
        self.bn2 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)

    def __call__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        senders,
        receivers,
        rel_pos: jnp.ndarray,
        window_support,
        a
    ) -> jnp.ndarray:
        xa = jnp.concatenate([x, y], axis=-1)
        
        xl = self.cconv1(xa[senders], receivers, rel_pos, window_support, a)
        xl = self.bn1(xl, is_training=True)
        xl = jax.nn.relu(xl)

        xl = self.cconv2(xl[senders], receivers, rel_pos, window_support, a)

        xl = self.bn2(xl, is_training=True)
        wei = jax.nn.sigmoid(xl)
        
        return 2.0 * x * wei + 2.0 * y * (1.0 - wei)


class IAFF(hk.Module):
    """
    Iterative Attention-based Feature Fusion (IAFF) module with two AFF stages.
    """
    def __init__(
        self,
        channels: int = 32,
        inter_channels: int = 64,
        conv_type: str = 'cconv',
        num_particles: int = 0,
        name=None,
    ):
        super().__init__(name=name)
        if conv_type == 'cconv':
            self.ConvLayer = CConvLayer
        else:
            self.ConvLayer = ASCC
        # first AFF block
        self.cconv1 = self.ConvLayer(2*channels, inter_channels, kernel_size=(4,4), aggregation_points=num_particles)
        self.bn1 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
        self.cconv2 = self.ConvLayer(inter_channels, channels, kernel_size=(4,4), aggregation_points=num_particles)
        self.bn2 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
        # second AFF block
        self.cconv3 = self.ConvLayer(channels, inter_channels, kernel_size=(4,4), aggregation_points=num_particles)
        self.bn3 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)
        self.cconv4 = self.ConvLayer(inter_channels, channels, kernel_size=(4,4), aggregation_points=num_particles)
        self.bn4 = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)

    def __call__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        senders,
        receivers,
        rel_pos: jnp.ndarray,
        window_support,
        a
    ) -> jnp.ndarray:
        xa = jnp.concatenate([x, y], axis=-1)
        # first AFF
        xl = self.cconv1(xa[senders], receivers, rel_pos, window_support, a)
        xl = self.bn1(xl, is_training=True)
        xl = jax.nn.relu(xl)

        xl = self.cconv2(xl[senders], receivers, rel_pos, window_support, a)
        xl = self.bn2(xl, is_training=True)
        wei1 = jax.nn.sigmoid(xl)
        xo = 2.0 * x * wei1 + 2.0 * y * (1.0 - wei1)
        # second AFF
        xl2 = self.cconv3(xo[senders], receivers, rel_pos, window_support, a)
        xl2 = self.bn3(xl2, is_training=True)
        xl2 = jax.nn.relu(xl2)
        xl2 = self.cconv4(xl2[senders], receivers, rel_pos, window_support, a)
        xl2 = self.bn4(xl2, is_training=True)
        wei2 = jax.nn.sigmoid(xl2)
        return 2.0 * x * wei2 + 2.0 * y * (1.0 - wei2)
    

class MyParticleNetwork(BaseModel):
    """
    Haiku translation of the PyTorch MyParticleNetwork for fluid simulation.

    Combines CConv/ASCC, AFF, IAFF modules with dense layers and integration logic.
    """
    def __init__(
        self,
        displ_fn,
        shift_fn,
        kernel_size=(4,4),
        radius: float = 0.025,
        timestep: float = 1/50,
        num_particles: int = 0,
        name=None,
    ):
        super().__init__(name=name)
        self.displ_fn = displ_fn
        self.shift_fn = shift_fn
        self.layer_channels = [32, 64, 128, 64, 2]
        self.kernel_size = tuple(kernel_size)
        self.radius = radius
        self.timestep = timestep
        self.num_particles = num_particles
        def ConvLayer(in_ch, out_ch, conv_type='cconv'):
            return (CConvLayer if conv_type=='cconv' else ASCC)(
                in_ch, out_ch, kernel_size=kernel_size, aggregation_points=num_particles
            )
        # AFF and IAFF
        self.aff_cconv = IAFF(32, 64, num_particles=num_particles)
        # initial fluid/obstacle conv and dense
        self.conv0_fluid = ConvLayer(3, self.layer_channels[0])
        self.dense0_fluid = hk.Linear(output_size=self.layer_channels[0])
        self.conv0_obstacle = ConvLayer(4, self.layer_channels[0])
        
        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i-1] if i>1 else 64
            out_ch = self.layer_channels[i]
            dense = hk.Linear(output_size=out_ch)
            conv = ConvLayer(in_ch, out_ch)
            self.denses.append(dense)
            self.convs.append(conv)
        
        self.aff_ascc = IAFF(channels=32, inter_channels=64, num_particles=num_particles, conv_type='ascc')
        self.conv0_fluid_ascc = ConvLayer(3, self.layer_channels[0], conv_type="ascc")
        self.dense0_fluid_ascc = hk.Linear(output_size=self.layer_channels[0])
        self.conv0_obstacle_ascc = ConvLayer(4, self.layer_channels[0], conv_type="ascc")

        self.convs_ascc = []
        self.denses_ascc = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i-1] if i>1 else 64
            out_ch = self.layer_channels[i]
            dense = hk.Linear(output_size=out_ch)
            conv = ConvLayer(in_ch, out_ch, conv_type="ascc")
            self.denses_ascc.append(dense)
            self.convs_ascc.append(conv)
     
        self.affs = []
        self.aff0 = AFF(channels=self.layer_channels[0]*2, inter_channels=self.layer_channels[0]*2, num_particles=self.num_particles, conv_type='cconv')
        for i in range(1, len(self.layer_channels)):
            ch = self.layer_channels[i]
            aff = AFF(channels=ch, inter_channels=ch, num_particles=self.num_particles, conv_type='cconv')
            self.affs.append(aff)
        self.resAff = AFF(channels=64, inter_channels=64, num_particles=self.num_particles, conv_type='cconv')

    def integrate_pos_vel(self, pos1, vel1, mask, force):
        dt   = self.timestep
        # compute candidate update everywhere
        vel2_candidate = vel1 + dt * force           # shape (N, dim)
        # select updated velocities where mask is True, else keep old
        vel2 = jnp.where(mask[:, None],
                        vel2_candidate,
                        vel1)                       # shape (N, dim)

        # now do the position update
        pos2_candidate = self.shift_fn(pos1, dt * (vel2 + vel1) / 2.0)   # shape (N, dim)
        
        return pos2_candidate, vel2

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        features, particle_types = sample

        pos2, vel2 = self.integrate_pos_vel(features["abs_pos"][:,-1,:], features["vel_hist"][:, -2:], particle_types == Tag.FLUID, features["force"])
        
        fluid_feats = jnp.concatenate([jnp.ones((pos2.shape[0],1)), vel2], axis=-1)
        box_feat = jnp.concatenate([pos2, vel2], axis=-1)
        
        senders, receivers = features["senders"], features["receivers"]
        
        rel_pos = self.displ_fn(pos2[senders], pos2[receivers]) / self.radius
        
        fw_mask = ((particle_types[senders] == Tag.MOVING_WALL) | (particle_types[senders] == Tag.SOLID_WALL) | (particle_types[senders] == Tag.DIRICHLET_WALL)) & (particle_types[receivers] == Tag.FLUID)
        ff_mask = (particle_types[senders] == Tag.FLUID) & (particle_types[receivers] == Tag.FLUID)

        a_fw = jnp.where(fw_mask, window_function_batched(jnp.linalg.norm(rel_pos, ord=2, axis=1) / self.radius), jnp.array(0.0, dtype=rel_pos.dtype))
        a_ff = jnp.where(ff_mask, window_function_batched(jnp.linalg.norm(rel_pos, ord=2, axis=1) / self.radius), jnp.array(0.0, dtype=rel_pos.dtype))
        # first conv0
        ans_f = self.conv0_fluid(fluid_feats[senders], receivers, rel_pos, self.radius, a=a_ff)
        ans_d = self.dense0_fluid(fluid_feats)
        obs_f = self.conv0_obstacle(box_feat[senders], receivers, rel_pos, self.radius, a=a_fw)
        hybrid = self.aff_cconv(ans_f, obs_f, senders, receivers, rel_pos, self.radius, a=a_ff)
        feats = jnp.concatenate([hybrid, ans_d], axis=-1)
        
        # ascc
        ans_f_ascc = self.conv0_fluid_ascc(fluid_feats[senders], receivers, rel_pos, self.radius, a=a_ff)
        ans_d_ascc = self.dense0_fluid_ascc(fluid_feats)
        obs_f_ascc = self.conv0_obstacle_ascc(box_feat[senders], receivers, rel_pos, self.radius, a=a_fw)
        hybrid_ascc = self.aff_ascc(ans_f_ascc, obs_f_ascc, senders, receivers, rel_pos, self.radius, a=a_ff)
        feats_ascc = jnp.concatenate([hybrid_ascc, ans_d_ascc], axis=-1)

        feats_select = self.aff0(feats, feats_ascc, senders, receivers, rel_pos, self.radius, a=a_ff)

        ans_convs = [feats_select]

        for conv_cconv, dense_cconv, conv_ascc, dense_ascc, aff in zip(self.convs, self.denses, self.convs_ascc, self.denses_ascc, self.affs):
            inp_feats = jax.nn.relu(ans_convs[-1])
            #cconv
            ans_conv_cconv = conv_cconv(inp_feats[senders], receivers, rel_pos, self.radius, a=a_ff)
            ans_dense_cconv = dense_cconv(inp_feats)
            ans_cconv = ans_conv_cconv + ans_dense_cconv
            #ascc
            ans_conv_ascc = conv_ascc(inp_feats[senders], receivers, rel_pos, self.radius, a=a_ff)
            ans_dense_ascc = dense_ascc(inp_feats)
            ans_ascc = ans_conv_ascc + ans_dense_ascc
            #aff
            ans_select = aff(ans_cconv, ans_ascc, senders, receivers, rel_pos, self.radius, a=a_ff)
            #ResAFF
            if len(ans_convs) == 3 and ans_dense_cconv.shape[-1] == ans_convs[-2].shape[-1]:
                ans_select = self.resAff(ans_select, ans_convs[-2], senders, receivers, rel_pos, self.radius, a=a_ff)
            ans_convs.append(ans_select)
        # pos2 +=  
        return {"acc": ans_convs[-1] / 128.0}
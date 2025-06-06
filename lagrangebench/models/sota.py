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
        self.conv_type = conv_type
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
        a,
        isTraining=False
    ) -> jnp.ndarray:
        xa = jnp.concatenate([x, y], axis=-1)
        if self.conv_type == "cconv":
            inp_feat = xa[senders]
        else:
            inp_feat = xa[senders] + xa[receivers]
        xl = self.cconv1(inp_feat, receivers, rel_pos, a)
        xl = self.bn1(xl, is_training=isTraining)
        xl = jax.nn.relu(xl)
        if self.conv_type == "cconv":
            inp_feat = xl[senders]
        else:
            inp_feat = xl[senders] + xl[receivers]
        xl = self.cconv2(inp_feat, receivers, rel_pos, a)
        xl = self.bn2(xl, is_training=isTraining)
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
        self.conv_type = conv_type
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
        a,
        isTraining=False
    ) -> jnp.ndarray:
        xa = jnp.concatenate([x, y], axis=-1)
        # first AFF

        if self.conv_type == "cconv":
            inp_feat = xa[senders]
        else:
            inp_feat = xa[senders] + xa[receivers]
        
        xl = self.cconv1(inp_feat, receivers, rel_pos, a)
        xl = self.bn1(xl, is_training=isTraining)
        xl = jax.nn.relu(xl)
        
        if self.conv_type == "cconv":
            inp_feat = xl[senders]
        else:
            inp_feat = xl[senders] + xl[receivers]
        xl = self.cconv2(inp_feat, receivers, rel_pos, a)
        xl = self.bn2(xl, is_training=isTraining)
        wei1 = jax.nn.sigmoid(xl)
        xo = 2.0 * x * wei1 + 2.0 * y * (1.0 - wei1)
        # second AFF
        if self.conv_type == "cconv":
            inp_feat = xo[senders]
        else:
            inp_feat = xo[senders] + xo[receivers]
        xl2 = self.cconv3(inp_feat, receivers, rel_pos, a)
        xl2 = self.bn3(xl2, is_training=isTraining)
        xl2 = jax.nn.relu(xl2)

        if self.conv_type == "cconv":
            inp_feat = xl2[senders]
        else:
            inp_feat = xl2[senders] + xl2[receivers]
        xl2 = self.cconv4(inp_feat, receivers, rel_pos, a)
        xl2 = self.bn4(xl2, is_training=isTraining)
        wei2 = jax.nn.sigmoid(xl2)
        return 2.0 * x * wei2 + 2.0 * y * (1.0 - wei2)
    

class MyParticleNetwork(BaseModel):
    """
    Haiku translation of the PyTorch MyParticleNetwork for fluid simulation.

    Combines CConv/ASCC, AFF, IAFF modules with dense layers and integration logic.
    """
    def __init__(
        self,
        kernel_size=(4,4),
        radius: float = 0.025,
        num_particles: int = 0,
        name=None,
    ):
        super().__init__(name=name)
        self.layer_channels = [32, 64, 128, 64, 2]
        self.kernel_size = tuple(kernel_size)
        self.radius = radius
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
        self.conv0_obstacle = ConvLayer(5, self.layer_channels[0])
        
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
        self.conv0_obstacle_ascc = ConvLayer(5, self.layer_channels[0], conv_type="ascc")

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
        
    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray], isTraining=True,
    ) -> Dict[str, jnp.ndarray]:
        features, particle_types = sample

        pos2, vel2 = features["abs_pos"][:, 0], features["vel2_candidates"]
        fluid_mask = particle_types == Tag.FLUID
        fluid_mask = fluid_mask[..., None]  
        
        fluid_feats = jnp.concatenate([jnp.ones((pos2.shape[0],1)), vel2], axis=-1)
        box_feat = jnp.concatenate([2*jnp.ones((pos2.shape[0],1)), vel2], axis=-1)
        #fm = fluid_mask[:, None]  # shape: (num_particles, 1)
        
        fluid_feats = jnp.where(fluid_mask, fluid_feats, 0.0)
        box_feat   = jnp.where(fluid_mask, 0.0,   box_feat)
        senders, receivers = features["senders"], features["receivers"]
        
        rel_pos = features["rel_disp"]
        box_sender_feats = jnp.concatenate([features["rel_disp_from_prev_time"], box_feat[senders]], axis=-1)

        
        fw_mask = (particle_types[senders] != Tag.FLUID) & (particle_types[receivers] == Tag.FLUID) & (receivers != senders)
        ff_mask = (particle_types[senders] == Tag.FLUID) & (particle_types[receivers] == Tag.FLUID) & (receivers != senders)
        w = window_function_batched(features["rel_dist"][:, 0])
        a_fw = jnp.where(fw_mask, w, jnp.array(0.0, dtype=rel_pos.dtype))
        a_ff = jnp.where(ff_mask, w, jnp.array(0.0, dtype=rel_pos.dtype))
        # first conv0
        ans_f = self.conv0_fluid(fluid_feats[senders], receivers, rel_pos, a=a_ff)
        ans_d = self.dense0_fluid(fluid_feats)
        ans_d = jnp.where(fluid_mask, ans_d, 0.0)

        obs_f = self.conv0_obstacle(box_sender_feats, receivers, rel_pos, a=a_fw)
        hybrid = self.aff_cconv(ans_f, obs_f, senders, receivers, rel_pos, a=a_ff, isTraining=isTraining)
        feats = jnp.concatenate([hybrid, ans_d], axis=-1)
        
        # ascc
        ans_f_ascc = self.conv0_fluid_ascc(fluid_feats[senders] + fluid_feats[receivers], receivers, rel_pos, a=a_ff)
        ans_d_ascc = self.dense0_fluid_ascc(fluid_feats)
        ans_d_ascc = jnp.where(fluid_mask, ans_d_ascc, 0.0)
        obs_f_ascc = self.conv0_obstacle_ascc(box_sender_feats, receivers, rel_pos, a=a_fw)
        hybrid_ascc = self.aff_ascc(ans_f_ascc, obs_f_ascc, senders, receivers, rel_pos, a=a_ff, isTraining=isTraining)
        feats_ascc = jnp.concatenate([hybrid_ascc, ans_d_ascc], axis=-1)

        feats_select = self.aff0(feats, feats_ascc, senders, receivers, rel_pos, a=a_ff, isTraining=isTraining)

        ans_convs = [feats_select]
        
        for conv_cconv, dense_cconv, conv_ascc, dense_ascc, aff in zip(self.convs, self.denses, self.convs_ascc, self.denses_ascc, self.affs):
            inp_feats = jnp.where(fluid_mask, jax.nn.relu(ans_convs[-1]), 0.0)
        
            #cconv
            ans_conv_cconv = conv_cconv(inp_feats[senders], receivers, rel_pos, a=a_ff)
            ans_dense_cconv = dense_cconv(inp_feats)
            ans_dense_cconv = jnp.where(fluid_mask, ans_dense_cconv, 0.0)
            
            ans_cconv = ans_conv_cconv + ans_dense_cconv
            
            #ascc
            ans_conv_ascc = conv_ascc(inp_feats[senders] + inp_feats[receivers], receivers, rel_pos, a=a_ff)
            ans_dense_ascc = dense_ascc(inp_feats)
            ans_dense_ascc = jnp.where(fluid_mask, ans_dense_ascc, 0.0)

            ans_ascc = ans_conv_ascc + ans_dense_ascc
            
            #aff
            ans_select = aff(ans_cconv, ans_ascc, senders, receivers, rel_pos, a=a_ff, isTraining=isTraining)

            #ResAFF
            if len(ans_convs) == 3 and ans_dense_cconv.shape[-1] == ans_convs[-2].shape[-1]:
                ans_select = self.resAff(ans_select, ans_convs[-2], senders, receivers, rel_pos, a=a_ff, isTraining=isTraining)

            ans_convs.append(ans_select)
        #jax.debug.print("corrections mean magnitude {}", jnp.sum(jnp.linalg.norm(ans_convs[-1], ord=2, axis=1)) / jnp.sum(fluid_mask[:, 0]))
        # s = hk.get_parameter(
        #     "out_scale",             
        #     shape=(),                 
        #     init=hk.initializers.Constant(1/109.37)
        # )
        #jax.debug.print("s={}", s)
        return {"pos": pos2 + 1/110*ans_convs[-1]}

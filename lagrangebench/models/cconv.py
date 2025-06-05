import haiku as hk
import jax.numpy as jnp
from ..utils_extra.continous_convolution import continous_conv_operation
import jax
from functools import partial

class CConvLayer(hk.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(4, 4), aggregation_points=4, name=None):
        super().__init__(name=name)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv_operation = partial(continous_conv_operation, aggregate_points=aggregation_points)

    def __call__(self, features, receivers: jnp.ndarray, relative_positions: jnp.ndarray, window_support, a) -> jnp.ndarray:
        kh, kw = self.kernel_size
        init = hk.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        kernel = hk.get_parameter(
            "kernel",
            shape=(kh, kw, self.in_ch, self.out_ch),
            init=init
        )
        bias = hk.get_parameter("bias", shape=(self.out_ch,), init=jnp.zeros)
        return self.conv_operation(kernel, receivers, relative_positions, window_support, features, a) + bias

class ASCC(hk.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size=(4, 4), aggregation_points=4, name=None):
        super().__init__(name=name)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = (kernel_size[0], kernel_size[1] // 2)

        self.conv_operation = partial(continous_conv_operation, aggregate_points=aggregation_points)

    def __call__(self, features, receivers: jnp.ndarray, relative_positions: jnp.ndarray, window_support, a) -> jnp.ndarray:
        kh, kw = self.kernel_size
        init = hk.initializers.RandomUniform(minval=-0.05, maxval=0.05)
        kernel = hk.get_parameter(
            "kernel",
            shape=(kh, kw, self.in_ch, self.out_ch),
            init=init
        )
        bias = hk.get_parameter("bias", shape=(self.out_ch,), init=jnp.zeros)
        kernel_flipped = -jnp.flip(kernel, axis=(0,1))
        return self.conv_operation(jnp.concatenate([kernel, kernel_flipped], axis=1), receivers, relative_positions, window_support, features, a) + bias

def test_cconv_layer():
    # Define dummy data
    N, C_in, E, C_out = 5, 2, 8, 3
    features = jax.random.normal(jax.random.PRNGKey(0), (E, C_in))
    receivers = jax.random.randint(jax.random.PRNGKey(1), (E,), 0, N)
    rel_pos = jax.random.uniform(jax.random.PRNGKey(2), (E, 2), minval=-1, maxval=1)
    radius = 0.5

    # Build model
    def forward(f, rcv, rp, rad):
        layer = ASCC(in_ch=C_in, out_ch=C_out, kernel_size=(2, 2), aggregation_points=N)
        return layer(f, rcv, rp, rad)

    model = hk.transform(forward)
    rng = jax.random.PRNGKey(42)

    # Initialize and apply
    params = model.init(rng, features, receivers, rel_pos, radius)
    output = model.apply(params, rng, features, receivers, rel_pos, radius)

    # Assertions
    assert output.shape == (N, C_out), f"Expected shape {(N,C_out)}, got {output.shape}"
    print("Test passed! Output shape:", output.shape)


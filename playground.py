from lagrangebench.utils_extra.continous_convolution import window_function_batched, continous_conv_operation
import jax
import jax.numpy as jnp
from lagrangebench.utils_extra.interpolation import bilinear_interpolate_features
# ----------------------------------------------
# 2) Test Case Parameters (R=2, ChIn=1, 4 receivers)
# ----------------------------------------------
R = 2
ChIn = 1
num_receivers = 4

# 2.1) Define four "particles" with uv ∈ [0,1]^2
#     We will place them so that three land at distinct corners, 
#     and one (with a=0) also lies somewhere but contributes zero.
uv = jnp.array([
    [0.0, 0.0],  # Receiver 0 at top-left corner exactly (x0=0,y0=0)
    [1.0, 0.0],  # Receiver 1 at top-right corner exactly (x0=1,y0=0)
    [0.0, 1.0],  # Receiver 2 at bottom-left corner exactly (x0=0,y0=1)
    [0.5, 0.5],  # Receiver 3 sits in the middle of the 2x2 grid
], dtype=jnp.float32)  # shape (4,2)

# 2.2) Each receiver has a single‐channel feature (ChIn=1). For simplicity, let:
features = jnp.array([
    [1.0],  # feature for receiver 0
    [2.0],  # feature for receiver 1
    [3.0],  # feature for receiver 2
    [4.0],  # feature for receiver 3 (but we'll zero it out via a[3]=0)
], dtype=jnp.float32)  # shape (4,1)

# 2.3) The 'receivers' array says "receiver i writes into row i" of the output:
receivers = jnp.arange(num_receivers, dtype=jnp.int32)  # [0,1,2,3]

# 2.4) The coefficient vector 'a' (length 4):
#      Set the last entry to zero so that receiver 3 contributes NOTHING.
a = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float32)

# 2.5) Use normalization_coeff = 1.0 for simplicity.
normalization_coeff = 1.0


# ----------------------------------------------
# 3) Run the test
# ----------------------------------------------
M_flat = bilinear_interpolate_features(
    R=R,
    uv=uv,
    features=features,
    receivers=receivers,
    a=a,
    normalization_coeff=normalization_coeff,
    num_receivers=num_receivers,
)

print("Output shape:", M_flat.shape)
print("M_flat:\n", M_flat)


# ------------------------------------------------------
# 4) Build the test case for 4 particles, 3 active interactions
# ------------------------------------------------------
R = 2
ChIn = 1
ChOut = 2
num_receivers = 4

# 4.1) Define relative_positions ∈ [0,1]^2 for each of 4 points:
#     – Receivers 0,1,2 lie exactly on corners of 2×2 grid.
#     – Receiver 3 lies at the center (0.5,0.5), but we will zero it out (a[3]=0).
relative_positions = jnp.array([
    [0.0, 0.0],  # Receiver 0 → top-left
    [1.0, 0.0],  # Receiver 1 → top-right
    [0.0, 1.0],  # Receiver 2 → bottom-left
    [0.5, 0.5],  # Receiver 3 → center (will be masked out)
    [0.0, 0.0],
    [1.0, 0.0]
], dtype=jnp.float32)  # shape (4,2)

# 4.2) Per-receiver features, each with ChIn=1
features = jnp.array([
    [1.0],  # feature for receiver 0
    [2.0],  # feature for receiver 1
    [3.0],  # feature for receiver 2
    [4.0],  # feature for receiver 3
], dtype=jnp.float32)  # shape (4, 1)

# 4.3) “receivers” array maps receiver i → output row i
#receivers = jnp.arange(num_receivers, dtype=jnp.int32)  # [0, 1, 2, 3]
senders = jnp.array([2, 1, 0, 2, 4, 4])
receivers = jnp.array([2, 3, 3,1, 4, 4])
particle_types = jnp.array([0, 0, 0, 1])
@jax.jit
def get_a():
    return jnp.where((particle_types[senders] == 0) & (particle_types[receivers] == 1), 1.0, 0.0)

a = get_a()   
jax.debug.print("{}", a)
# 4.5) Define a simple 2×2×1×2 kernel. For reproducibility, choose known values:
#     Let kernel[..., 0→ChOut=0] = [[1, 2], [3, 4]]  (flattened → [1,2,3,4]), 
#     and kernel[..., 1→ChOut=1] = [[5, 6], [7, 8]]  (flattened → [5,6,7,8]).
#     
#     In other words, kernel[r, c, 0, 0] = r * R + c + 1 (1..4),
#                      kernel[r, c, 0, 1] = 4 + (r*R + c + 1) (5..8).
kernel = jnp.stack([
    # ChOut = 0:
    jnp.array([[1.0, 5.0],   # (r=0,c=0) → (1,5)
               [2.0, 6.0]]), # (r=0,c=1) → (2,6)
    jnp.array([[3.0, 7.0],   # (r=1,c=0) → (3,7)
               [4.0, 8.0]])  # (r=1,c=1) → (4,8)
], axis=0)  # shape (2, 2, 1, 2)
# Explanation of shape: 
#   - Outer stack over r=0..1 → shape (2, 2, 2). 
#   - We need to reshape into (2, 2, 1, 2). So insert ChIn=1 dimension:
kernel = kernel.reshape((2, 2, 1, 2))


# 4.7) aggregate_points = num_receivers = 4
aggregate_points = num_receivers

# 4.8) Whether to normalize or not: choose False for simplicity
normalize = False

# ------------------------------------------------------
# 5) Run the test and print outputs
# ------------------------------------------------------
output = continous_conv_operation(
    kernel,
    receivers,
    relative_positions,
    features[senders],
    a,
    aggregate_points,
    normalize=normalize
)  # → shape (4, ChOut=2)

print("Output shape:", output.shape)
print("Output (each row is [out_ch0, out_ch1]):\n", output)
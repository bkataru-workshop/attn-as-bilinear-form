"""Shared test helpers, strategies, and utilities for attn-tensors tests."""

import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# =============================================================================
# Numerical Tolerances (float32 relaxed)
# =============================================================================

RTOL = 1e-4
ATOL = 1e-5

# =============================================================================
# Helper Functions
# =============================================================================


def assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=""):
    """Assert arrays are close with standard tolerances."""
    np.testing.assert_allclose(
        np.array(actual),
        np.array(expected),
        rtol=rtol,
        atol=atol,
        err_msg=err_msg,
    )


def assert_shape(tensor, expected_shape, name="tensor"):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, (
        f"{name}: Expected shape {expected_shape}, got {tensor.shape}"
    )


def assert_finite(tensor, name="tensor"):
    """Assert all values in tensor are finite (no NaN or Inf)."""
    assert jnp.all(jnp.isfinite(tensor)), f"{name} contains non-finite values"


def assert_nonnegative(tensor, name="tensor"):
    """Assert all values are >= 0."""
    assert jnp.all(tensor >= 0), f"{name} contains negative values"


def assert_probability_distribution(probs, axis=-1, name="probs"):
    """Assert tensor is a valid probability distribution along axis."""
    assert_nonnegative(probs, name)
    sums = jnp.sum(probs, axis=axis)
    assert_allclose(sums, jnp.ones_like(sums), err_msg=f"{name} doesn't sum to 1")


def assert_symmetric(matrix, name="matrix"):
    """Assert matrix is symmetric."""
    assert_allclose(matrix, matrix.T, err_msg=f"{name} is not symmetric")


def assert_positive_definite(matrix, name="matrix"):
    """Assert matrix is positive definite (all eigenvalues > 0)."""
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    assert jnp.all(eigenvalues > -ATOL), (
        f"{name} is not positive definite: eigenvalues = {eigenvalues}"
    )


def assert_positive_semidefinite(matrix, name="matrix"):
    """Assert matrix is positive semi-definite (all eigenvalues >= 0)."""
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    assert jnp.all(eigenvalues >= -ATOL), (
        f"{name} is not positive semi-definite: eigenvalues = {eigenvalues}"
    )


# =============================================================================
# Hypothesis Strategies
# =============================================================================

# Dimension strategies
tiny_dims = st.integers(min_value=1, max_value=4)
small_dims = st.integers(min_value=1, max_value=16)
medium_dims = st.integers(min_value=2, max_value=32)

# Safe float strategy (no NaN, no Inf, bounded)
safe_floats = st.floats(
    min_value=-10.0,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)

# Smaller floats for better numerical stability in some tests
small_floats = st.floats(
    min_value=-3.0,
    max_value=3.0,
    allow_nan=False,
    allow_infinity=False,
)

# Positive floats for things like temperature
positive_floats = st.floats(
    min_value=0.01,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def random_vector(draw, d=None):
    """Generate a random vector of dimension d (or random small dimension)."""
    if d is None:
        d = draw(small_dims)
    arr = draw(arrays(np.float32, (d,), elements=safe_floats))
    return jnp.array(arr)


@st.composite
def random_matrix(draw, m=None, n=None):
    """Generate a random m x n matrix."""
    if m is None:
        m = draw(small_dims)
    if n is None:
        n = draw(small_dims)
    arr = draw(arrays(np.float32, (m, n), elements=safe_floats))
    return jnp.array(arr)


@st.composite
def valid_metric(draw, d=None):
    """Generate a valid (positive definite) metric tensor of dimension d."""
    if d is None:
        d = draw(small_dims)
    # Use smaller floats for numerical stability
    W = draw(arrays(np.float32, (d, d), elements=small_floats))
    # W^T W is positive semi-definite; add small diagonal for positive definite
    metric = W.T @ W + np.eye(d, dtype=np.float32) * 0.1
    return jnp.array(metric)


@st.composite
def qkv_tensors(draw, n_q=None, n_k=None, d=None):
    """Generate Q, K, V tensors for attention tests."""
    if n_q is None:
        n_q = draw(st.integers(min_value=1, max_value=8))
    if n_k is None:
        n_k = draw(st.integers(min_value=1, max_value=8))
    if d is None:
        d = draw(st.integers(min_value=2, max_value=16))

    Q = draw(arrays(np.float32, (n_q, d), elements=small_floats))
    K = draw(arrays(np.float32, (n_k, d), elements=small_floats))
    V = draw(arrays(np.float32, (n_k, d), elements=small_floats))

    return {
        "Q": jnp.array(Q),
        "K": jnp.array(K),
        "V": jnp.array(V),
        "n_q": n_q,
        "n_k": n_k,
        "d": d,
    }


@st.composite
def softmax_input(draw, n=None):
    """Generate input vector for softmax tests."""
    if n is None:
        n = draw(st.integers(min_value=2, max_value=32))
    arr = draw(arrays(np.float32, (n,), elements=safe_floats))
    return jnp.array(arr)


@st.composite
def attention_scores_2d(draw, n_q=None, n_k=None):
    """Generate 2D attention score matrix."""
    if n_q is None:
        n_q = draw(st.integers(min_value=1, max_value=8))
    if n_k is None:
        n_k = draw(st.integers(min_value=1, max_value=8))
    arr = draw(arrays(np.float32, (n_q, n_k), elements=safe_floats))
    return jnp.array(arr)

"""
Tests for the einsum module.

Tests verify that einsum implementations match standard NumPy/JAX operations.
"""

import jax.numpy as jnp
from jax import random

from attn_tensors.einsum import (
    EINSUM_EXAMPLES,
    attention_output_einsum,
    attention_scores_einsum,
    batch_bilinear_form,
    batch_matrix_multiply,
    batched_attention_output_einsum,
    batched_attention_scores_einsum,
    batched_multihead_project_einsum,
    batched_multihead_scores_einsum,
    bilinear_form_einsum,
    dot_product,
    explain_einsum,
    frobenius_norm_squared,
    hadamard_product,
    matrix_multiply,
    matrix_transpose,
    multihead_combine_einsum,
    multihead_output_einsum,
    multihead_project_einsum,
    multihead_scores_einsum,
    outer_product,
    parse_einsum,
    trace,
)


class TestBasicOperations:
    """Test basic einsum operations."""

    def test_dot_product(self):
        """Test dot product matches numpy."""
        u = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([4.0, 5.0, 6.0])

        result = dot_product(u, v)
        expected = jnp.dot(u, v)

        assert jnp.allclose(result, expected)

    def test_outer_product(self):
        """Test outer product matches numpy."""
        u = jnp.array([1.0, 2.0])
        v = jnp.array([3.0, 4.0, 5.0])

        result = outer_product(u, v)
        expected = jnp.outer(u, v)

        assert jnp.allclose(result, expected)

    def test_matrix_transpose(self):
        """Test transpose matches numpy."""
        A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = matrix_transpose(A)
        expected = A.T

        assert jnp.allclose(result, expected)

    def test_trace(self):
        """Test trace matches numpy."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = trace(A)
        expected = jnp.trace(A)

        assert jnp.allclose(result, expected)

    def test_matrix_multiply(self):
        """Test matrix multiplication matches numpy."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = matrix_multiply(A, B)
        expected = A @ B

        assert jnp.allclose(result, expected)

    def test_batch_matrix_multiply(self):
        """Test batched matrix multiplication."""
        key = random.PRNGKey(42)
        A = random.normal(key, (4, 3, 5))
        B = random.normal(random.split(key)[0], (4, 5, 2))

        result = batch_matrix_multiply(A, B)

        # Check against loop
        for i in range(4):
            expected_i = A[i] @ B[i]
            assert jnp.allclose(result[i], expected_i)

    def test_frobenius_norm_squared(self):
        """Test Frobenius norm squared."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = frobenius_norm_squared(A)
        expected = jnp.sum(A**2)

        assert jnp.allclose(result, expected)

    def test_hadamard_product(self):
        """Test element-wise product."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = hadamard_product(A, B)
        expected = A * B

        assert jnp.allclose(result, expected)


class TestBilinearForms:
    """Test bilinear form operations."""

    def test_bilinear_form_identity(self):
        """Test bilinear form with identity is dot product."""
        u = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([4.0, 5.0, 6.0])
        M = jnp.eye(3)

        result = bilinear_form_einsum(u, M, v)
        expected = jnp.dot(u, v)

        assert jnp.allclose(result, expected)

    def test_bilinear_form_general(self):
        """Test bilinear form with general matrix."""
        u = jnp.array([1.0, 2.0])
        v = jnp.array([3.0, 4.0])
        M = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = bilinear_form_einsum(u, M, v)
        expected = u @ M @ v

        assert jnp.allclose(result, expected)

    def test_batch_bilinear_form(self):
        """Test batch bilinear form for attention scores."""
        n_q, n_k, d = 4, 6, 8
        key = random.PRNGKey(42)
        keys = random.split(key, 3)

        Q = random.normal(keys[0], (n_q, d))
        K = random.normal(keys[1], (n_k, d))
        g = jnp.eye(d)

        result = batch_bilinear_form(Q, g, K)

        # With identity metric, should be Q @ K.T
        expected = Q @ K.T

        assert result.shape == (n_q, n_k)
        assert jnp.allclose(result, expected)


class TestAttentionOperations:
    """Test attention einsum operations."""

    def test_attention_scores_shape(self):
        """Test attention scores have correct shape."""
        n_q, n_k, d_k = 5, 7, 32
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        Q = random.normal(keys[0], (n_q, d_k))
        K = random.normal(keys[1], (n_k, d_k))

        S = attention_scores_einsum(Q, K)

        assert S.shape == (n_q, n_k)

    def test_attention_scores_scaling(self):
        """Test attention scores are properly scaled."""
        d_k = 64
        Q = jnp.ones((1, d_k))
        K = jnp.ones((1, d_k))

        S_scaled = attention_scores_einsum(Q, K, scale=True)
        S_unscaled = attention_scores_einsum(Q, K, scale=False)

        assert jnp.allclose(S_scaled, S_unscaled / jnp.sqrt(d_k))

    def test_attention_output_shape(self):
        """Test attention output has correct shape."""
        n_q, n_k, d_v = 5, 7, 16
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        A = jnp.abs(random.normal(keys[0], (n_q, n_k)))
        A = A / A.sum(axis=-1, keepdims=True)  # Normalize
        V = random.normal(keys[1], (n_k, d_v))

        O = attention_output_einsum(A, V)

        assert O.shape == (n_q, d_v)

    def test_batched_attention_scores(self):
        """Test batched attention scores."""
        batch, n_q, n_k, d_k = 3, 5, 7, 32
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        Q = random.normal(keys[0], (batch, n_q, d_k))
        K = random.normal(keys[1], (batch, n_k, d_k))

        S = batched_attention_scores_einsum(Q, K)

        assert S.shape == (batch, n_q, n_k)

    def test_batched_attention_output(self):
        """Test batched attention output."""
        batch, n_q, n_k, d_v = 3, 5, 7, 16
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        A = jnp.abs(random.normal(keys[0], (batch, n_q, n_k)))
        A = A / A.sum(axis=-1, keepdims=True)
        V = random.normal(keys[1], (batch, n_k, d_v))

        O = batched_attention_output_einsum(A, V)

        assert O.shape == (batch, n_q, d_v)


class TestMultiHeadOperations:
    """Test multi-head attention einsum operations."""

    def test_multihead_project(self):
        """Test projection to per-head space."""
        seq_len, d_model, n_heads, d_k = 10, 64, 8, 8
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        X = random.normal(keys[0], (seq_len, d_model))
        W = random.normal(keys[1], (n_heads, d_model, d_k))

        X_h = multihead_project_einsum(X, W)

        assert X_h.shape == (n_heads, seq_len, d_k)

    def test_multihead_scores(self):
        """Test per-head attention scores."""
        n_heads, n_q, n_k, d_k = 8, 10, 12, 8
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        Q_h = random.normal(keys[0], (n_heads, n_q, d_k))
        K_h = random.normal(keys[1], (n_heads, n_k, d_k))

        S = multihead_scores_einsum(Q_h, K_h)

        assert S.shape == (n_heads, n_q, n_k)

    def test_multihead_output(self):
        """Test per-head weighted values."""
        n_heads, n_q, n_k, d_v = 8, 10, 12, 8
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        A = jnp.abs(random.normal(keys[0], (n_heads, n_q, n_k)))
        A = A / A.sum(axis=-1, keepdims=True)
        V_h = random.normal(keys[1], (n_heads, n_k, d_v))

        O = multihead_output_einsum(A, V_h)

        assert O.shape == (n_heads, n_q, d_v)

    def test_multihead_combine(self):
        """Test combining heads to output."""
        n_heads, n_q, d_v, d_model = 8, 10, 8, 64
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        O_h = random.normal(keys[0], (n_heads, n_q, d_v))
        W_O = random.normal(keys[1], (n_heads, d_v, d_model))

        Y = multihead_combine_einsum(O_h, W_O)

        assert Y.shape == (n_q, d_model)

    def test_batched_multihead_project(self):
        """Test batched projection."""
        batch, seq_len, d_model, n_heads, d_k = 4, 10, 64, 8, 8
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        X = random.normal(keys[0], (batch, seq_len, d_model))
        W = random.normal(keys[1], (n_heads, d_model, d_k))

        X_h = batched_multihead_project_einsum(X, W)

        assert X_h.shape == (batch, n_heads, seq_len, d_k)

    def test_batched_multihead_scores(self):
        """Test batched per-head scores."""
        batch, n_heads, n_q, n_k, d_k = 4, 8, 10, 12, 8
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        Q_h = random.normal(keys[0], (batch, n_heads, n_q, d_k))
        K_h = random.normal(keys[1], (batch, n_heads, n_k, d_k))

        S = batched_multihead_scores_einsum(Q_h, K_h)

        assert S.shape == (batch, n_heads, n_q, n_k)


class TestParseEinsum:
    """Test einsum parsing utilities."""

    def test_parse_matmul(self):
        """Test parsing matrix multiplication."""
        parsed = parse_einsum("ij,jk->ik")

        assert parsed["inputs"] == ["ij", "jk"]
        assert parsed["output"] == "ik"
        assert set(parsed["summation_indices"]) == {"j"}
        assert set(parsed["free_indices"]) == {"i", "k"}

    def test_parse_trace(self):
        """Test parsing trace."""
        parsed = parse_einsum("ii->")

        assert parsed["inputs"] == ["ii"]
        assert parsed["output"] == ""
        assert set(parsed["summation_indices"]) == {"i"}
        assert len(parsed["free_indices"]) == 0

    def test_parse_attention_scores(self):
        """Test parsing attention scores."""
        parsed = parse_einsum("ia,ja->ij")

        assert parsed["inputs"] == ["ia", "ja"]
        assert parsed["output"] == "ij"
        assert set(parsed["summation_indices"]) == {"a"}
        assert set(parsed["free_indices"]) == {"i", "j"}

    def test_parse_multihead(self):
        """Test parsing multi-head projection."""
        parsed = parse_einsum("id,hda->hia")

        assert parsed["inputs"] == ["id", "hda"]
        assert parsed["output"] == "hia"
        assert set(parsed["summation_indices"]) == {"d"}
        assert set(parsed["free_indices"]) == {"h", "i", "a"}


class TestExplainEinsum:
    """Test einsum explanation utility."""

    def test_explain_matmul(self):
        """Test explanation for matrix multiplication."""
        explanation = explain_einsum("ij,jk->ik")

        assert "ij,jk->ik" in explanation
        assert "j" in explanation  # Summation index
        assert "Tensor 1" in explanation
        assert "Tensor 2" in explanation

    def test_explain_attention_scores(self):
        """Test explanation for attention scores."""
        explanation = explain_einsum("ia,ja->ij")

        assert "ia,ja->ij" in explanation
        assert "Sum over" in explanation


class TestEinsumExamples:
    """Test that EINSUM_EXAMPLES dictionary is valid."""

    def test_all_examples_are_valid_einsum(self):
        """Test all examples can be parsed."""
        for name, (subscripts, description) in EINSUM_EXAMPLES.items():
            parsed = parse_einsum(subscripts)
            assert len(parsed["inputs"]) >= 1, f"Failed for {name}"

    def test_examples_cover_attention(self):
        """Test that attention operations are included."""
        assert "attention_scores" in EINSUM_EXAMPLES
        assert "attention_output" in EINSUM_EXAMPLES
        assert "mh_project" in EINSUM_EXAMPLES
        assert "mh_combine" in EINSUM_EXAMPLES


class TestConsistencyWithAttentionModule:
    """Test that einsum implementations match main attention module."""

    def test_scores_match_attention_module(self):
        """Test einsum scores match attention module."""
        from attn_tensors.attention import attention_scores

        n_q, n_k, d_k = 5, 7, 32
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        Q = random.normal(keys[0], (n_q, d_k))
        K = random.normal(keys[1], (n_k, d_k))

        result_einsum = attention_scores_einsum(Q, K, scale=True)
        result_module = attention_scores(Q, K, scale=True)

        assert jnp.allclose(result_einsum, result_module)

    def test_output_match_attention_module(self):
        """Test einsum output matches attention module."""
        from attn_tensors.attention import attention_output

        n_q, n_k, d_v = 5, 7, 16
        key = random.PRNGKey(42)
        keys = random.split(key, 2)

        A = jnp.abs(random.normal(keys[0], (n_q, n_k)))
        A = A / A.sum(axis=-1, keepdims=True)
        V = random.normal(keys[1], (n_k, d_v))

        result_einsum = attention_output_einsum(A, V)
        result_module = attention_output(A, V)

        assert jnp.allclose(result_einsum, result_module)

"""
Einstein summation utilities and examples.

This module provides educational examples of einsum notation
and utility functions for common tensor operations in attention.

Einsum is a compact notation for expressing tensor operations where
repeated indices are implicitly summed. This maps directly to index
notation from tensor calculus.

References:
    - Shape Rotation 101: https://sankalp.bearblog.dev/einsum-new/ (Sankalp)
    - Basic guide to einsum: https://ajcr.net/Basic-guide-to-einsum/
    - Einstein summation in NumPy: https://obilaniu6266h16.wordpress.com
    - xjdr's JAX transformer: https://github.com/xjdr-alt/simple_transformer

Index conventions for attention:
    b: batch size
    l, i: query sequence length
    m, j: key/memory sequence length
    d: model dimension
    h: number of attention heads
    k: per-head key/query dimension
    c: per-head value dimension
    a: general feature dimension
"""

import jax.numpy as jnp
from jax import Array

# =============================================================================
# Basic Operations (Educational)
# =============================================================================


def dot_product(u: Array, v: Array) -> Array:
    """
    Dot product of two vectors.

    Math: s = u^a v_a = sum_a u_a v_a
    Einsum: 'a,a->'

    Args:
        u: Vector of shape (d,)
        v: Vector of shape (d,)

    Returns:
        Scalar dot product
    """
    return jnp.einsum("a,a->", u, v)


def outer_product(u: Array, v: Array) -> Array:
    """
    Outer product of two vectors.

    Math: M_{ab} = u_a v_b
    Einsum: 'a,b->ab'

    Args:
        u: Vector of shape (m,)
        v: Vector of shape (n,)

    Returns:
        Matrix of shape (m, n)
    """
    return jnp.einsum("a,b->ab", u, v)


def matrix_transpose(A: Array) -> Array:
    """
    Matrix transpose.

    Math: B_{ji} = A_{ij}
    Einsum: 'ij->ji'

    Args:
        A: Matrix of shape (m, n)

    Returns:
        Transposed matrix of shape (n, m)
    """
    return jnp.einsum("ij->ji", A)


def trace(A: Array) -> Array:
    """
    Matrix trace (sum of diagonal).

    Math: tr(A) = A_{ii} = sum_i A_{ii}
    Einsum: 'ii->'

    Args:
        A: Square matrix of shape (n, n)

    Returns:
        Scalar trace
    """
    return jnp.einsum("ii->", A)


def matrix_multiply(A: Array, B: Array) -> Array:
    """
    Matrix multiplication.

    Math: C_{ik} = A_{ij} B_{jk} = sum_j A_{ij} B_{jk}
    Einsum: 'ij,jk->ik'

    The index j is the "summation index" (appears in inputs but not output).

    Args:
        A: Matrix of shape (m, n)
        B: Matrix of shape (n, p)

    Returns:
        Matrix of shape (m, p)
    """
    return jnp.einsum("ij,jk->ik", A, B)


def batch_matrix_multiply(A: Array, B: Array) -> Array:
    """
    Batched matrix multiplication.

    Math: C_{bij} = sum_k A_{bik} B_{bkj}
    Einsum: 'bik,bkj->bij'

    The batch index b is preserved (appears in output).

    Args:
        A: Tensor of shape (batch, m, n)
        B: Tensor of shape (batch, n, p)

    Returns:
        Tensor of shape (batch, m, p)
    """
    return jnp.einsum("bik,bkj->bij", A, B)


def frobenius_norm_squared(A: Array) -> Array:
    """
    Squared Frobenius norm of a matrix.

    Math: ||A||_F^2 = sum_{ij} A_{ij}^2 = A_{ij} A_{ij}
    Einsum: 'ij,ij->'

    Args:
        A: Matrix of any shape

    Returns:
        Scalar squared norm
    """
    return jnp.einsum("ij,ij->", A, A)


def hadamard_product(A: Array, B: Array) -> Array:
    """
    Element-wise (Hadamard) product.

    Math: C_{ij} = A_{ij} B_{ij}
    Einsum: 'ij,ij->ij'

    Args:
        A: Matrix of shape (m, n)
        B: Matrix of shape (m, n)

    Returns:
        Matrix of shape (m, n)
    """
    return jnp.einsum("ij,ij->ij", A, B)


# =============================================================================
# Bilinear Forms
# =============================================================================


def bilinear_form_einsum(u: Array, M: Array, v: Array) -> Array:
    """
    Bilinear form u^T M v.

    Math: B(u,v) = u^a M_{ab} v^b = sum_{a,b} u_a M_{ab} v_b
    Einsum: 'a,ab,b->'

    This is the core operation in attention scores.

    Args:
        u: Vector of shape (m,)
        M: Matrix of shape (m, n)
        v: Vector of shape (n,)

    Returns:
        Scalar
    """
    return jnp.einsum("a,ab,b->", u, M, v)


def batch_bilinear_form(Q: Array, g: Array, K: Array) -> Array:
    """
    Batched bilinear forms for attention scores.

    Math: S_{ij} = Q^{ia} g_{ab} K^{jb}
    Einsum: 'ia,ab,jb->ij'

    Computes score between each query i and key j.

    Args:
        Q: Queries of shape (n_q, d)
        g: Metric tensor of shape (d, d)
        K: Keys of shape (n_k, d)

    Returns:
        Score matrix of shape (n_q, n_k)
    """
    return jnp.einsum("ia,ab,jb->ij", Q, g, K)


# =============================================================================
# Attention Operations
# =============================================================================


def attention_scores_einsum(Q: Array, K: Array, scale: bool = True) -> Array:
    """
    Compute attention scores using einsum.

    Math: S_{ij} = Q^{ia} K^{ja} / sqrt(d_k)
    Einsum: 'ia,ja->ij'

    The feature index a is contracted (summed over).

    Args:
        Q: Queries of shape (n_q, d_k)
        K: Keys of shape (n_k, d_k)
        scale: Whether to apply 1/sqrt(d_k) scaling

    Returns:
        Score matrix of shape (n_q, n_k)
    """
    d_k = Q.shape[-1]
    S = jnp.einsum("ia,ja->ij", Q, K)
    if scale:
        S = S / jnp.sqrt(d_k)
    return S


def attention_output_einsum(A: Array, V: Array) -> Array:
    """
    Compute attention output using einsum.

    Math: O_{ib} = A^{ij} V^{jb}
    Einsum: 'ij,jb->ib'

    Weighted sum of values.

    Args:
        A: Attention weights of shape (n_q, n_k)
        V: Values of shape (n_k, d_v)

    Returns:
        Output of shape (n_q, d_v)
    """
    return jnp.einsum("ij,jb->ib", A, V)


def batched_attention_scores_einsum(Q: Array, K: Array) -> Array:
    """
    Batched attention scores.

    Math: S_{bij} = Q^{bia} K^{bja} / sqrt(d_k)
    Einsum: 'bia,bja->bij'

    Args:
        Q: Queries of shape (batch, n_q, d_k)
        K: Keys of shape (batch, n_k, d_k)

    Returns:
        Scores of shape (batch, n_q, n_k)
    """
    d_k = Q.shape[-1]
    return jnp.einsum("bia,bja->bij", Q, K) / jnp.sqrt(d_k)


def batched_attention_output_einsum(A: Array, V: Array) -> Array:
    """
    Batched attention output.

    Math: O_{bic} = A^{bij} V^{bjc}
    Einsum: 'bij,bjc->bic'

    Args:
        A: Attention weights of shape (batch, n_q, n_k)
        V: Values of shape (batch, n_k, d_v)

    Returns:
        Output of shape (batch, n_q, d_v)
    """
    return jnp.einsum("bij,bjc->bic", A, V)


# =============================================================================
# Multi-Head Attention Operations
# =============================================================================


def multihead_project_einsum(X: Array, W: Array) -> Array:
    """
    Project input to per-head representations.

    Math: X^{hia} = X^{id} W^{hda}
    Einsum: 'id,hda->hia'

    Args:
        X: Input of shape (seq_len, d_model)
        W: Projection weights of shape (n_heads, d_model, d_k)

    Returns:
        Per-head projections of shape (n_heads, seq_len, d_k)
    """
    return jnp.einsum("id,hda->hia", X, W)


def multihead_scores_einsum(Q_h: Array, K_h: Array) -> Array:
    """
    Per-head attention scores.

    Math: S^{hij} = Q^{hia} K^{hja} / sqrt(d_k)
    Einsum: 'hia,hja->hij'

    Args:
        Q_h: Per-head queries of shape (n_heads, n_q, d_k)
        K_h: Per-head keys of shape (n_heads, n_k, d_k)

    Returns:
        Per-head scores of shape (n_heads, n_q, n_k)
    """
    d_k = Q_h.shape[-1]
    return jnp.einsum("hia,hja->hij", Q_h, K_h) / jnp.sqrt(d_k)


def multihead_output_einsum(A: Array, V_h: Array) -> Array:
    """
    Per-head attention output.

    Math: O^{hic} = A^{hij} V^{hjc}
    Einsum: 'hij,hjc->hic'

    Args:
        A: Per-head attention weights of shape (n_heads, n_q, n_k)
        V_h: Per-head values of shape (n_heads, n_k, d_v)

    Returns:
        Per-head outputs of shape (n_heads, n_q, d_v)
    """
    return jnp.einsum("hij,hjc->hic", A, V_h)


def multihead_combine_einsum(O_h: Array, W_O: Array) -> Array:
    """
    Combine per-head outputs.

    Math: Y_{id} = O^{hic} W_O^{hcd}
    Einsum: 'hic,hcd->id'

    Both h (heads) and c (per-head dimension) are summed over.

    Args:
        O_h: Per-head outputs of shape (n_heads, n_q, d_v)
        W_O: Output projection of shape (n_heads, d_v, d_model)

    Returns:
        Combined output of shape (n_q, d_model)
    """
    return jnp.einsum("hic,hcd->id", O_h, W_O)


def batched_multihead_project_einsum(X: Array, W: Array) -> Array:
    """
    Batched projection to per-head representations.

    Math: X^{bhia} = X^{bid} W^{hda}
    Einsum: 'bid,hda->bhia'

    Args:
        X: Input of shape (batch, seq_len, d_model)
        W: Projection weights of shape (n_heads, d_model, d_k)

    Returns:
        Per-head projections of shape (batch, n_heads, seq_len, d_k)
    """
    return jnp.einsum("bid,hda->bhia", X, W)


def batched_multihead_scores_einsum(Q_h: Array, K_h: Array) -> Array:
    """
    Batched per-head attention scores.

    Math: S^{bhij} = Q^{bhia} K^{bhja} / sqrt(d_k)
    Einsum: 'bhia,bhja->bhij'

    Args:
        Q_h: Per-head queries of shape (batch, n_heads, n_q, d_k)
        K_h: Per-head keys of shape (batch, n_heads, n_k, d_k)

    Returns:
        Per-head scores of shape (batch, n_heads, n_q, n_k)
    """
    d_k = Q_h.shape[-1]
    return jnp.einsum("bhia,bhja->bhij", Q_h, K_h) / jnp.sqrt(d_k)


# =============================================================================
# Utility: Parse Einsum String
# =============================================================================


def parse_einsum(subscripts: str) -> dict:
    """
    Parse einsum subscript string to understand the operation.

    Args:
        subscripts: Einsum string like 'ij,jk->ik'

    Returns:
        Dictionary with:
            - inputs: List of input index strings
            - output: Output index string
            - summation_indices: Indices that are summed over
            - free_indices: Indices that appear in output
    """
    if "->" in subscripts:
        inputs_str, output = subscripts.split("->")
    else:
        # Implicit output: all indices that appear exactly once
        inputs_str = subscripts
        from collections import Counter

        counts = Counter(c for c in inputs_str if c.isalpha())
        output = "".join(sorted(i for i, count in counts.items() if count == 1))

    inputs = [s.strip() for s in inputs_str.split(",")]

    # Find all input indices
    input_indices = set()
    for inp in inputs:
        input_indices.update(c for c in inp if c.isalpha())

    # Output indices
    output_indices = set(c for c in output if c.isalpha())

    # Summation indices: in inputs but not in output
    summation_indices = input_indices - output_indices

    return {
        "inputs": inputs,
        "output": output,
        "input_indices": sorted(input_indices),
        "free_indices": sorted(output_indices),
        "summation_indices": sorted(summation_indices),
    }


def explain_einsum(subscripts: str) -> str:
    """
    Generate a human-readable explanation of an einsum operation.

    Args:
        subscripts: Einsum string like 'ij,jk->ik'

    Returns:
        Multi-line explanation string
    """
    parsed = parse_einsum(subscripts)

    lines = [
        f"Einsum: '{subscripts}'",
        "",
        f"Input tensors: {len(parsed['inputs'])}",
    ]

    for i, inp in enumerate(parsed["inputs"]):
        dims = len([c for c in inp if c.isalpha()])
        lines.append(f"  Tensor {i + 1}: indices '{inp}' ({dims}D)")

    lines.extend(
        [
            "",
            f"Output indices: '{parsed['output']}'",
            f"Free indices (in output): {parsed['free_indices']}",
            f"Summation indices (contracted): {parsed['summation_indices']}",
        ]
    )

    if parsed["summation_indices"]:
        lines.append("")
        lines.append("Operations:")
        lines.append("  - Multiply along shared indices")
        lines.append(f"  - Sum over: {', '.join(parsed['summation_indices'])}")

    return "\n".join(lines)


# =============================================================================
# Examples Dictionary
# =============================================================================

EINSUM_EXAMPLES = {
    # Basic operations
    "dot_product": ("a,a->", "Dot product of two vectors"),
    "outer_product": ("a,b->ab", "Outer product of two vectors"),
    "transpose": ("ij->ji", "Matrix transpose"),
    "trace": ("ii->", "Matrix trace (sum of diagonal)"),
    "sum_all": ("ij->", "Sum of all elements"),
    "sum_rows": ("ij->i", "Sum along rows (axis 1)"),
    "sum_cols": ("ij->j", "Sum along columns (axis 0)"),
    # Matrix operations
    "matmul": ("ij,jk->ik", "Matrix multiplication"),
    "matmul_transpose": ("ij,jk->ki", "Matrix multiplication with transpose"),
    "hadamard": ("ij,ij->ij", "Element-wise (Hadamard) product"),
    "frobenius_sq": ("ij,ij->", "Squared Frobenius norm"),
    "trace_product": ("ij,ji->", "Trace of matrix product"),
    # Batch operations
    "batch_matmul": ("bij,bjk->bik", "Batched matrix multiplication"),
    "batch_trace": ("bii->b", "Trace per batch element"),
    # Bilinear forms
    "bilinear": ("a,ab,b->", "Bilinear form u^T M v"),
    "batch_bilinear": ("ia,ab,jb->ij", "Batch bilinear form (attention scores)"),
    # Attention operations
    "attention_scores": ("ia,ja->ij", "Attention scores Q @ K^T"),
    "attention_output": ("ij,jb->ib", "Attention output A @ V"),
    "batched_scores": ("bia,bja->bij", "Batched attention scores"),
    "batched_output": ("bij,bjc->bic", "Batched attention output"),
    # Multi-head attention
    "mh_project": ("id,hda->hia", "Project to per-head"),
    "mh_scores": ("hia,hja->hij", "Per-head scores"),
    "mh_output": ("hij,hjc->hic", "Per-head weighted values"),
    "mh_combine": ("hic,hcd->id", "Combine heads"),
    # Batched multi-head
    "bmh_project": ("bid,hda->bhia", "Batched project to per-head"),
    "bmh_scores": ("bhia,bhja->bhij", "Batched per-head scores"),
}

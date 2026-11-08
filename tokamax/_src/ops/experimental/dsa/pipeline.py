"""Paged DSA forward: Lightning Indexer, SparseCore TopK/gather, and MLA."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from tokamax._src.ops.lightning_indexer import api as indexer_api
from tokamax._src.ops.lightning_indexer import pallas_mosaic_tpu as indexer_tpu
from tokamax._src.ops.lightning_indexer.kernel import config as indexer_config
from tokamax._src.ops.sparse_mla import api as mla_api


class Inputs(NamedTuple):
  """Already projected queries and populated caches for a single-device call.

  Both caches address the same logical tokens, but may have different physical
  page tables. Indexer keys use HEAD_ALONG_SUBLANE FP8 records. Attention uses
  the existing DSV4 448-dimensional NoPE + 64-dimensional RoPE cache format.
  No KV compression, context parallelism, SWA branch, or attention sink is used.
  """

  q: jax.Array  # [tokens, attention_heads, 512], bf16
  indexer_q: jax.Array  # [tokens, indexer_heads, 128], bf16
  indexer_weights: jax.Array  # [tokens, indexer_heads], f32
  indexer_cache: jax.Array  # [pages, page_size // 4, 4, 256], uint8
  cache_nope: jax.Array  # [pages, page_size, 128], int32
  cache_rope: jax.Array  # [pages, page_size // 4, 128], int32
  seq_lens: jax.Array  # [sequences], int32, includes current query tokens
  indexer_page_indices: jax.Array  # [sequences * pages_per_sequence], int32
  attention_page_indices: jax.Array  # [sequences * pages_per_sequence], int32
  cu_q_lens: jax.Array  # [sequences + 1], int32
  distribution: jax.Array  # [3], int32


def select(
    inputs: Inputs,
    *,
    k: int,
    implementation: str = "mosaic",
    indexer_chunk_tokens: int | None = None,
) -> jax.Array:
  """Returns causal logical KV indices; padded positions are -1."""
  if implementation not in ("mosaic", "xla"):
    raise ValueError(f"Unknown implementation: {implementation}")
  indexer = implementation
  if implementation == "mosaic" and indexer_chunk_tokens is not None:
    indexer = indexer_tpu.PallasTpuLightningIndexer(
        config=indexer_tpu.Config(chunk_tokens=indexer_chunk_tokens)
    )
  return indexer_api.lightning_indexer(
      inputs.indexer_q,
      inputs.indexer_weights,
      inputs.indexer_cache,
      inputs.seq_lens,
      inputs.indexer_page_indices,
      inputs.cu_q_lens,
      inputs.distribution,
      k=k,
      compression_ratio=1,
      kv_layout=indexer_config.KVLayout.HEAD_ALONG_SUBLANE,
      implementation=indexer,
  )


def attend(
    inputs: Inputs,
    indices: jax.Array,
    *,
    implementation: str = "mosaic",
) -> jax.Array:
  """Gathers selected KV and evaluates attention without SWA or sink mass."""
  tokens, heads, dim = inputs.q.shape
  return mla_api.sparse_mla(
      inputs.q,
      inputs.cache_nope,
      inputs.cache_rope,
      indices,
      inputs.attention_page_indices,
      inputs.cu_q_lens,
      inputs.distribution,
      jnp.full((heads,), -jnp.inf, dtype=jnp.float32),
      jnp.zeros_like(inputs.q),
      jnp.zeros((tokens, heads), dtype=jnp.float32),
      jnp.zeros((tokens, heads), dtype=jnp.float32),
      sm_scale=dim**-0.5,
      implementation="mosaic_tpu" if implementation == "mosaic" else implementation,
  )


def dsa_forward(
    inputs: Inputs,
    *,
    k: int,
    implementation: str = "mosaic",
    indexer_chunk_tokens: int | None = None,
    return_indices: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Composes the entire selection-to-attention forward in one JIT-able call.

  The result is a composition of existing kernels, not a persistent mega kernel
  or a full model layer. Projection, RoPE, cache writes and output projection
  are outside this function; there is no training backward/teacher/KL path.
  """
  if k <= 0 or k % 128:
    raise ValueError("k must be a positive multiple of 128 for SparseMLA.")
  if inputs.q.shape[0] != inputs.indexer_q.shape[0]:
    raise ValueError("Attention and indexer queries must have equal token counts.")
  if inputs.q.shape[-1] != 512:
    raise ValueError("The existing SparseMLA cache format requires head_dim=512.")
  indices = select(
      inputs,
      k=k,
      implementation=implementation,
      indexer_chunk_tokens=indexer_chunk_tokens,
  )
  output = attend(inputs, indices, implementation=implementation)
  return (output, indices) if return_indices else output

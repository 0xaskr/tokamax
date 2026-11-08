"""Deterministic, separately paged index and attention caches for DSA tests."""

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.dsv4_compressor import csa_cache_layout
from tokamax._src.ops.experimental.dsa import pipeline


def make_inputs(
    *,
    q_lens: tuple[int, ...],
    seq_lens: tuple[int, ...],
    num_heads: int = 64,
    indexer_heads: int = 64,
    page_size: int = 128,
    seed: int = 42,
) -> pipeline.Inputs:
  """Builds inputs; packing and host/device transfers are outside timing."""
  if len(q_lens) != len(seq_lens) or not q_lens:
    raise ValueError("q_lens and seq_lens must have the same nonzero length.")
  if any(q <= 0 or q > s for q, s in zip(q_lens, seq_lens, strict=True)):
    raise ValueError("Require 0 < query length <= KV length for every sequence.")
  if page_size % 128:
    raise ValueError("The indexer requires a page size divisible by 128.")
  rng = np.random.default_rng(seed)
  pages_per_seq = (max(seq_lens) + page_size - 1) // page_size
  pages = len(q_lens) * pages_per_seq
  tokens = sum(q_lens)
  index_pages = rng.permutation(pages).astype(np.int32)
  attn_pages = rng.permutation(pages).astype(np.int32)

  # Independent permutations ensure logical indices, rather than physical cache
  # offsets, are passed between the two operators.
  index_keys = rng.standard_normal((pages, page_size, 128), dtype=np.float32)
  index_keys = jnp.asarray(index_keys[np.argsort(index_pages)])
  scale = jnp.exp2(jnp.ceil(jnp.log2(
      jnp.maximum(jnp.max(jnp.abs(index_keys), axis=-1, keepdims=True), 1e-4)
      / 448.0
  )))
  key_bytes = jax.lax.bitcast_convert_type(
      (index_keys / scale).astype(jnp.float8_e4m3fn), jnp.uint8
  )
  scale_bytes = jax.lax.bitcast_convert_type(
      scale.astype(jnp.float8_e8m0fnu), jnp.uint8
  )
  records = jnp.pad(
      jnp.concatenate((key_bytes, scale_bytes), axis=-1),
      ((0, 0), (0, 0), (0, 127)),
  )
  index_cache = records.reshape(pages, page_size // 4, 4, 256)

  kv = rng.standard_normal((pages, page_size, 512), dtype=np.float32)
  kv = jnp.asarray(kv[np.argsort(attn_pages)], dtype=jnp.bfloat16)
  blocks = kv[..., :448].reshape(pages, page_size, 7, 64)
  scales = jnp.exp2(jnp.ceil(jnp.log2(
      jnp.maximum(jnp.max(jnp.abs(blocks), axis=-2, keepdims=True), 1e-4)
      / 448.0
  )))
  quant = (blocks / scales).astype(jnp.float8_e4m3fn)
  nope_bytes = jax.lax.bitcast_convert_type(
      quant.reshape(pages, page_size, 448), jnp.uint8
  )
  nope_scales = jax.lax.bitcast_convert_type(
      jnp.squeeze(scales, -2).astype(jnp.float8_e8m0fnu), jnp.uint8
  )
  cache_nope = csa_cache_layout.slabs_to_words(
      jnp.concatenate((nope_bytes, nope_scales), axis=-1).reshape(
          pages, page_size, 4, 128
      )
  )
  cache_rope = csa_cache_layout.encode_rope(kv[..., 448:].reshape(-1, 64))
  cache_rope = cache_rope.reshape(pages, page_size // 4, 128)
  num_decode = 0
  while num_decode < len(q_lens) and q_lens[num_decode] == 1:
    num_decode += 1
  return pipeline.Inputs(
      q=jnp.asarray(rng.standard_normal((tokens, num_heads, 512)), jnp.bfloat16),
      indexer_q=jnp.asarray(
          rng.standard_normal((tokens, indexer_heads, 128)), jnp.bfloat16
      ),
      indexer_weights=jnp.asarray(
          rng.uniform(0.25, 1.75, (tokens, indexer_heads)), jnp.float32
      ),
      indexer_cache=index_cache,
      cache_nope=cache_nope,
      cache_rope=cache_rope,
      seq_lens=jnp.asarray(seq_lens, jnp.int32),
      indexer_page_indices=jnp.asarray(index_pages),
      attention_page_indices=jnp.asarray(attn_pages),
      cu_q_lens=jnp.asarray([0, *np.cumsum(q_lens)], jnp.int32),
      distribution=jnp.asarray([num_decode, num_decode, len(q_lens)], jnp.int32),
  )


def assert_valid_indices(inputs: pipeline.Inputs, indices: jax.Array) -> None:
  """Checks causal limits, no duplicate selections, and suffix-only padding."""
  indices = np.asarray(indices)
  cu = np.asarray(inputs.cu_q_lens)
  lengths = np.asarray(inputs.seq_lens)
  for seq in range(len(lengths)):
    for token in range(cu[seq], cu[seq + 1]):
      causal_end = lengths[seq] - (cu[seq + 1] - cu[seq]) + token - cu[seq] + 1
      row = indices[token]
      valid = row[row >= 0]
      assert len(valid) == min(causal_end, row.size)
      assert np.all(valid < causal_end)
      assert np.unique(valid).size == valid.size
      assert np.all(row[len(valid):] == -1)

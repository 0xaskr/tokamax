"""Checks causal DSA composition against a float64 attention oracle."""

import functools

import jax
import numpy as np
import pytest
from tokamax._src.ops.experimental.dsa import pipeline
from tokamax._src.ops.experimental.dsa import testing
from tokamax._src.ops.sparse_mla import reference as mla_reference


@pytest.mark.parametrize("q_lens,seq_lens", [((1, 1), (64, 129)), ((1, 5), (96, 257))])
def test_reference_composition(q_lens, seq_lens):
  inputs = testing.make_inputs(
      q_lens=q_lens, seq_lens=seq_lens, num_heads=4, indexer_heads=4
  )
  run = jax.jit(functools.partial(
      pipeline.dsa_forward, k=128, implementation="xla", return_indices=True
  ))
  actual, indices = run(inputs)
  testing.assert_valid_indices(inputs, indices)
  kv = np.asarray(mla_reference.dequantize_kv_cache(
      inputs.cache_nope, inputs.cache_rope
  ), dtype=np.float64)
  page_table = np.asarray(inputs.attention_page_indices).reshape(len(q_lens), -1)
  expected = np.empty(actual.shape, dtype=np.float64)
  for seq in range(len(q_lens)):
    for token in range(sum(q_lens[:seq]), sum(q_lens[:seq + 1])):
      selected = np.asarray(indices[token])
      selected = selected[selected >= 0]
      values = kv[page_table[seq, selected // 128], selected % 128]
      scores = np.asarray(inputs.q[token], np.float64) @ values.T / np.sqrt(512)
      weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
      expected[token] = (weights / weights.sum(axis=-1, keepdims=True)) @ values
  np.testing.assert_allclose(np.asarray(actual, np.float32), expected, atol=0.003, rtol=0.02)

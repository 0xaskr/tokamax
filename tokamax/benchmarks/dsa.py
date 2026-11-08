"""Validate and time composed DSA on TPU v7 using hardware XProf timing."""

import argparse
import functools
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import tempfile
import time

import jax
import numpy as np
from tokamax._src.ops.experimental.dsa import pipeline
from tokamax._src.ops.experimental.dsa import testing
from tokamax.benchmarks.dsa_experiment import verify_timing


CASES = {
    "decode_4x4k": ((1, 1, 1, 1), (4096,) * 4, 1024),
    "prefill_128x8k": ((128,), (8192,), 1024),
    "prefill_128x32k": ((128,), (32768,), 2048),
}


def _error(actual, expected):
  actual = np.asarray(actual, np.float32)
  expected = np.asarray(expected, np.float32)
  return {
      "max_abs": float(np.max(np.abs(actual - expected))),
      "relative_l2": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
  }


def run_case(name, *, iterations, method, include_chunked):
  q_lens, seq_lens, k = CASES[name]
  inputs = testing.make_inputs(q_lens=q_lens, seq_lens=seq_lens)
  jax.block_until_ready(inputs)
  ref_fn = functools.partial(
      pipeline.dsa_forward, k=k, implementation="xla", return_indices=True
  )
  print(f"VALIDATING {name}: reference", flush=True)
  with jax.default_matmul_precision("highest"):
    reference, reference_indices = jax.jit(ref_fn)(inputs)
  jax.block_until_ready(reference)
  results = []
  for chunk_tokens in ([None, 32] if include_chunked and sum(q_lens) >= 64 else [None]):
    print(f"VALIDATING {name}: mosaic chunk={chunk_tokens}", flush=True)
    fn = functools.partial(
        pipeline.dsa_forward, k=k, indexer_chunk_tokens=chunk_tokens
    )
    output, indices = jax.jit(functools.partial(fn, return_indices=True))(inputs)
    jax.block_until_ready(output)
    testing.assert_valid_indices(inputs, indices)
    # Separate attention error from a possible TopK boundary difference.
    with jax.default_matmul_precision("highest"):
      same_indices_reference = jax.jit(functools.partial(
          pipeline.attend, implementation="xla"
      ))(inputs, indices)
    np.testing.assert_allclose(
        np.asarray(output, np.float32), np.asarray(same_indices_reference, np.float32),
        atol=0.02, rtol=0.05,
    )
    np.testing.assert_allclose(
        np.asarray(output, np.float32), np.asarray(reference, np.float32),
        atol=0.02, rtol=0.05,
    )
    overlap = np.mean([
        np.intersect1d(a, b).size / k
        for a, b in zip(np.asarray(indices), np.asarray(reference_indices), strict=True)
    ])
    # Repeated execution exercises the same compiled graph with populated buffers.
    repeat = jax.jit(fn)
    for _ in range(3):
      np.testing.assert_array_equal(np.asarray(repeat(inputs)), np.asarray(output))
    select = functools.partial(pipeline.select, k=k, indexer_chunk_tokens=chunk_tokens)
    timings = {}
    measurements = (
        ("end_to_end", fn, inputs),
        ("indexer_topk", select, inputs),
        ("gather_attention", lambda args: pipeline.attend(*args), (inputs, indices)),
    )
    if chunk_tokens is None:
      measurements += ((
          "xla_end_to_end", functools.partial(
              pipeline.dsa_forward, k=k, implementation="xla"
          ), inputs,
      ),)
    for stage, stage_fn, args in measurements:
      print(f"BENCHMARK {name} chunk={chunk_tokens} stage={stage}", flush=True)
      artifact_root = os.environ.get("ARTIFACT_LOCAL_DIR") or os.environ.get(
          "WORKLOAD_ARTIFACTS_DIR", "/tmp"
      )
      pathlib.Path(artifact_root).mkdir(parents=True, exist_ok=True)
      trace_dir = pathlib.Path(tempfile.mkdtemp(
          prefix=f"dsa_{name}_{chunk_tokens}_{stage}_", dir=artifact_root
      ))
      # The override belongs to the f32 JAX reference only. Passing it into
      # Mosaic changes its mixed bf16/FP8 matmul to an unsupported fp32 mode.
      if stage == "xla_end_to_end":
        with jax.default_matmul_precision("highest"):
          data = verify_timing.measure(
              stage_fn, args, trace_dir, iterations=iterations
          )
      else:
        data = verify_timing.measure(
            stage_fn, args, trace_dir, iterations=iterations
        )
      timings[stage] = data
      timings[stage]["median_ms"] = data["module_median_ms"]
      print("TIMING " + json.dumps({"case": name, "chunk_tokens": chunk_tokens,
          "stage": stage, "median_ms": data["module_median_ms"]}), flush=True)
    row = {
        "case": name, "q_lens": q_lens, "seq_lens": seq_lens, "topk": k,
        "indexer_heads": 64, "indexer_dim": 128, "attention_heads": 64,
        "attention_dim": 512, "q_dtype": "bfloat16", "page_size": 128,
        "indexer_chunk_tokens": chunk_tokens, "topk_set_overlap": float(overlap),
        "end_to_end_error": _error(output, reference),
        "attention_error_same_indices": _error(output, same_indices_reference),
        "timing_method": method, "timings": timings,
        "input_bytes": sum(x.size * x.dtype.itemsize for x in inputs),
    }
    results.append(row)
    print("RESULT " + json.dumps(row), flush=True)
  return results


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
  parser.add_argument("--iterations", type=int, default=5)
  parser.add_argument("--include-chunked", action="store_true")
  parser.add_argument("--output", default="/tmp/tpu_logs/dsa_results.json")
  args = parser.parse_args()
  if jax.default_backend() != "tpu" or "7" not in jax.devices()[0].device_kind:
    raise RuntimeError(f"Requires TPU v7, got {jax.devices()}")
  metadata = {
      "python": platform.python_version(), "jax": jax.__version__,
      "jaxlib": importlib.metadata.version("jaxlib"),
      "libtpu": importlib.metadata.version("libtpu"),
      "device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind,
      "visible_devices": len(jax.devices()), "devices_used": 1,
      "xla_flags": os.environ.get("XLA_FLAGS", ""),
      "base_commit": os.environ.get("TOKAMAX_BASE_COMMIT") or subprocess.check_output(
          ["git", "rev-parse", "HEAD"], text=True
      ).strip(),
      "source_overlay_sha256": os.environ.get("DSA_SOURCE_SHA256"),
      "reference_matmul_precision": "highest",
      "mosaic_matmul_precision": "default",
      "timing_metric": "XProf XLA Modules interval union; includes device execution gaps, excludes Python and input preparation",
      "scope": "Projected queries + prepacked caches -> selected-KV attention output; forward only",
  }
  output_path = pathlib.Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  report = {"metadata": metadata, "results": []}
  print("ENVIRONMENT " + json.dumps(metadata), flush=True)
  started = time.monotonic()
  with jax.default_device(jax.devices()[0]):
    for case in args.cases:
      report["results"].extend(run_case(case, iterations=args.iterations,
          method="xprof_xla_modules", include_chunked=args.include_chunked))
      report["wall_seconds"] = time.monotonic() - started
      output_path.write_text(json.dumps(report, indent=2) + "\n")
  print("DSA_BENCHMARK_OK", flush=True)


if __name__ == "__main__":
  main()

"""Cross-check DSA XLA-op timings against whole-module traces and wall time."""

import argparse
import functools
import json
import os
import pathlib
import time

import jax
import numpy as np
from tokamax._src.ops.experimental.dsa import pipeline
from tokamax._src.ops.experimental.dsa import testing
from xprof.cli.tools import get_kernel_stats_tool


def measure(fn, args, directory, *, iterations=5):
  """Measures whole XLA Modules; active-op traces can omit Pallas execution."""
  compiled = jax.jit(fn).lower(args).compile()
  for _ in range(3):
    jax.block_until_ready(compiled(args))
  wall_times = []
  for _ in range(10):
    start = time.perf_counter()
    jax.block_until_ready(compiled(args))
    wall_times.append((time.perf_counter() - start) * 1000)
  summaries = []
  for repeat in range(iterations):
    trace_path = directory / str(repeat)
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    options.enable_hlo_proto = False
    options.advanced_configuration = {
        "tpu_trace_mode": "TRACE_ONLY_XLA", "tpu_perf_counters": False
    }
    jax.profiler.start_trace(trace_path, profiler_options=options)
    jax.block_until_ready(compiled(args))
    jax.profiler.stop_trace()
    path, = trace_path.glob("**/*.xplane.pb")
    profile = jax.profiler.ProfileData.from_serialized_xspace(path.read_bytes())
    summary = json.loads(get_kernel_stats_tool.compute_kernel_stats(
        profile, include_summary=True, limit=30
    ))
    summaries.append(summary)
  module_ms = [s["by_line_duration_ns"]["XLA Modules"] / 1e6 for s in summaries]
  return {
      "module_times_ms": module_ms,
      "module_median_ms": float(np.median(module_ms)),
      "active_ops_times_ms": [s.get("total_device_duration_ms") for s in summaries],
      "blocked_wall_times_ms": wall_times,
      "blocked_wall_median_ms": float(np.median(wall_times)),
      "summary": summaries[-1],
  }


def main():
  from tokamax.benchmarks.dsa import CASES  # pylint: disable=g-import-not-at-top

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case", choices=CASES, default="prefill_128x32k")
  args = parser.parse_args()
  root = pathlib.Path(os.environ["ARTIFACT_LOCAL_DIR"]) / "timing_crosscheck" / args.case
  root.mkdir(parents=True, exist_ok=True)
  q_lens, seq_lens, k = CASES[args.case]
  rows = []
  with jax.default_device(jax.devices()[0]):
    inputs = testing.make_inputs(q_lens=q_lens, seq_lens=seq_lens)
    indices = jax.jit(functools.partial(pipeline.select, k=k))(inputs)
    stages = [
        ("end_to_end", functools.partial(pipeline.dsa_forward, k=k), inputs),
        ("indexer_topk", functools.partial(pipeline.select, k=k), inputs),
        ("gather_attention", lambda x: pipeline.attend(*x), (inputs, indices)),
    ]
    if sum(q_lens) >= 64:
      stages.append(("end_to_end_chunk32", functools.partial(
          pipeline.dsa_forward, k=k, indexer_chunk_tokens=32
      ), inputs))
    for name, fn, values in stages:
      print(f"CROSSCHECK_START {args.case} {name}", flush=True)
      row = {"case": args.case, "stage": name, **measure(fn, values, root / name)}
      rows.append(row)
      print("CROSSCHECK " + json.dumps(row), flush=True)
    with jax.default_matmul_precision("highest"):
      row = {"case": args.case, "stage": "xla_end_to_end", **measure(
          functools.partial(pipeline.dsa_forward, k=k, implementation="xla"),
          inputs, root / "xla_end_to_end"
      )}
    rows.append(row)
    print("CROSSCHECK " + json.dumps(row), flush=True)
  (root / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
  print("DSA_TIMING_CROSSCHECK_OK", flush=True)


if __name__ == "__main__":
  main()

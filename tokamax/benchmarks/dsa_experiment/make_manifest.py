"""Package the local DSA experiment into a finite Falcon TPU v7 workload."""

import argparse
import base64
import gzip
import hashlib
import json
import pathlib
import subprocess
import textwrap


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--name", default="tokamax-composed-dsa-v7-20261106")
  parser.add_argument("--output", default="/tmp/tokamax-dsa-v7.json")
  parser.add_argument("--crosscheck-only", action="store_true")
  args = parser.parse_args()
  root = pathlib.Path(__file__).resolve().parents[3]
  commit = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=root, text=True
  ).strip()
  sources = sorted((root / "tokamax/_src/ops/experimental/dsa").glob("*.py"))
  sources.append(root / "tokamax/benchmarks/dsa.py")
  sources.append(root / "tokamax/benchmarks/dsa_experiment/verify_timing.py")
  source_json = json.dumps({str(p.relative_to(root)): p.read_text() for p in sources})
  digest = hashlib.sha256(source_json.encode()).hexdigest()
  payload = base64.b64encode(gzip.compress(source_json.encode(), mtime=0)).decode()
  bootstrap = textwrap.dedent(f"""\
      import base64, gzip, hashlib, io, json, pathlib, tarfile, urllib.request
      root = pathlib.Path('/tmp/tokamax-dsa')
      url = 'https://codeload.github.com/openxla/tokamax/tar.gz/{commit}'
      with urllib.request.urlopen(url, timeout=120) as response:
          archive = response.read()
      with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as source:
          for member in source.getmembers():
              parts = pathlib.PurePosixPath(member.name).parts[1:]
              if not parts:
                  continue
              member.name = str(pathlib.PurePosixPath(*parts))
              source.extract(member, root, filter='data')
      raw = gzip.decompress(base64.b64decode('{payload}'))
      assert hashlib.sha256(raw).hexdigest() == '{digest}'
      for name, contents in json.loads(raw).items():
          path = root / name
          path.parent.mkdir(parents=True, exist_ok=True)
          path.write_text(contents)
      print('DSA_SOURCE_SHA256={digest}', flush=True)
      """)
  command = "\n".join([
      "set -eu", "ulimit -c 0",
      "mkdir -p /tmp/tpu_logs/tmp /tmp/tokamax-dsa",
      "export TMPDIR=/tmp/tpu_logs/tmp",
      "python3 - <<'BOOTSTRAP'", bootstrap, "BOOTSTRAP",
      "python3 -m venv /tmp/tpu_logs/dsa-venv",
      "/tmp/tpu_logs/dsa-venv/bin/pip install --no-cache-dir uv",
      "/tmp/tpu_logs/dsa-venv/bin/uv pip install --python /tmp/tpu_logs/dsa-venv/bin/python --no-cache 'jax[tpu]==0.11.2' 'libtpu==0.0.48.1' 'xprof==2.24.2' -e /tmp/tokamax-dsa pytest",
      "export PATH=/tmp/tpu_logs/dsa-venv/bin:$PATH",
      "cd /tmp/tokamax-dsa",
      "python -m pip freeze > /tmp/tpu_logs/dsa_packages.txt",
      "JAX_PLATFORMS=cpu python -m pytest -q tokamax/_src/ops/experimental/dsa/pipeline_test.py",
      "for dsa_case in decode_4x4k prefill_128x8k prefill_128x32k; do",
      "  timeout 600 python -m tokamax.benchmarks.dsa --cases \"$dsa_case\" --include-chunked --iterations 5 --output \"/tmp/tpu_logs/dsa_${dsa_case}.json\"",
      "done",
      "python - <<'COLLECT'",
      "import json, pathlib",
      "paths = [pathlib.Path('/tmp/tpu_logs') / ('dsa_' + name + '.json') for name in ('decode_4x4k', 'prefill_128x8k', 'prefill_128x32k')]",
      "reports = [json.loads(path.read_text()) for path in paths]",
      "report = {'metadata': reports[0]['metadata'], 'results': [row for part in reports for row in part['results']]}",
      "pathlib.Path('/tmp/tpu_logs/dsa_results.json').write_text(json.dumps(report, indent=2) + '\\n')",
      "print('DSA_ALL_CASES_OK', flush=True)",
      "COLLECT",
  ])
  if args.crosscheck_only:
    command = command[:command.index("for dsa_case in")]
    command += "\nfor dsa_case in prefill_128x32k decode_4x4k prefill_128x8k; do\n"
    command += "  timeout 600 python -m tokamax.benchmarks.dsa_experiment.verify_timing --case \"$dsa_case\"\n"
    command += "done\n"
  else:
    command += '\nmkdir -p "$ARTIFACT_LOCAL_DIR"\n'
    command += 'cp /tmp/tpu_logs/dsa_*.json /tmp/tpu_logs/dsa_packages.txt "$ARTIFACT_LOCAL_DIR/"\n'
  manifest = {
      "name": args.name, "exp_type": "TRAINING", "artifact_type": "GCS",
      "cluster_name": "gke-tpu-train-us-central1-2-prod", "priority": 0,
      "config": json.dumps({"purpose": "composed DSA correctness and latency",
          "base_commit": commit, "source_overlay_sha256": digest,
          "jax": "0.11.2", "libtpu": "0.0.48.1", "benchmark_devices": 1}),
      "role_to_task_spec": {"worker": {
          "command": command, "replica": 1,
          "image": "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1",
          "device_count": 8, "device_type": "v7x", "device_topo": "2x2x1",
          "envs": {"PYTHONUNBUFFERED": "1", "TOKAMAX_BASE_COMMIT": commit,
              "DSA_SOURCE_SHA256": digest, "WORKLOAD_ARTIFACTS_DIR": "/tmp/tpu_logs",
              "TOKAMAX_DUMP_XPROF": "true"},
      }},
  }
  pathlib.Path(args.output).write_text(json.dumps(manifest, indent=2) + "\n")
  print(json.dumps({"manifest": args.output, "source_overlay_sha256": digest}))


if __name__ == "__main__":
  main()

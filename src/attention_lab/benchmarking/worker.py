"""Isolated benchmark worker: reads one BenchConfig as JSON on stdin, prints one JSON record."""
import json
import sys
import warnings

warnings.filterwarnings("ignore", message=".*Profiler clears events.*")


def main() -> int:
    from attention_lab.benchmarking.runner import BenchConfig, run_one

    cfg = BenchConfig.from_json(json.loads(sys.stdin.read()))
    rec = run_one(cfg, isolated=True)
    sys.stdout.write(json.dumps(rec, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

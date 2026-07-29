"""Create and optionally submit a Slurm job while recording its metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from _common import results_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--time", default="2:00:00")
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--partition", default="cpu")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required after the options")
    root = results_dir("slurm")
    sub, out, err = root / "sub", root / "out", root / "err"
    for path in (sub, out, err):
        path.mkdir(parents=True, exist_ok=True)
    script_path = sub / f"{args.job_name}.sh"
    command = shlex.join(args.command)
    script = "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={args.job_name}",
            f"#SBATCH --output={out / (args.job_name + '.out')}",
            f"#SBATCH --error={err / (args.job_name + '.err')}",
            f"#SBATCH --time={args.time}",
            f"#SBATCH --mem={args.mem}",
            f"#SBATCH --cpus-per-task={args.cpus}",
            f"#SBATCH --partition={args.partition}",
            "set -euo pipefail",
            command,
            "",
        ]
    )
    script_path.write_text(script)
    metadata = {
        "job_name": args.job_name,
        "script": str(script_path),
        "command": command,
        "submitted": args.submit,
    }
    if args.submit:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["sbatch_output"] = result.stdout.strip()
    (root / f"{args.job_name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {script_path}")
    if args.submit:
        print(metadata["sbatch_output"])


if __name__ == "__main__":
    main()

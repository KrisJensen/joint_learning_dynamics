"""Helpers for writing and submitting simple Slurm jobs."""

from __future__ import annotations

from pathlib import Path
import subprocess

SLURM_ROOT = Path("/ceph/behrens/kris/research/learning_codex/slurm")
SUB_DIR = SLURM_ROOT / "sub"
OUT_DIR = SLURM_ROOT / "out"
ERR_DIR = SLURM_ROOT / "err"


def _ensure_slurm_dirs() -> None:
    for path in (SUB_DIR, OUT_DIR, ERR_DIR):
        path.mkdir(parents=True, exist_ok=True)


def slurm_submission_script(command, jobname, time = "2:00:00", mem = "16G", nodes = 1, cpus_per_task = 1, partition = "cpu", extra_commands = ""):

    script = f"""#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={OUT_DIR}/{jobname}.out
#SBATCH --error={ERR_DIR}/{jobname}.err
#SBATCH --nodes={nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --ntasks-per-node=1
{extra_commands}

source ~/.bashrc
conda activate learning

conda info

{command}
"""
    
    return script
    

def submit_slurm(command, jobname, **kwargs):

    _ensure_slurm_dirs()
    slurm_fname = SUB_DIR / f"{jobname}.sh"

    with slurm_fname.open("w") as f:
        f.write(slurm_submission_script(command, jobname, **kwargs))

    return subprocess.run(["sbatch", slurm_fname], check=True)

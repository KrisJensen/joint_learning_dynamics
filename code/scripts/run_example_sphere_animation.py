"""Generate an example trajectory and animate it on the unit sphere."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from _common import REPO_ROOT, parse_schedule, with_metadata, write_records
from learning_dynamics import Params, State, run_analytical


# Keep Matplotlib's runtime files out of the user's home directory.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PLOT_ROOT = Path(__file__).resolve().parents[1] / "plot"
if str(PLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOT_ROOT))

from _sphere import animate_sphere  # noqa: E402


DEFAULT_SCHEDULE = "online:5,offline:3,online:5,offline:3,online:10"


def main() -> None:
    """Write the example CSV under results/examples and GIF under figures/examples."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule",
        type=parse_schedule,
        default=DEFAULT_SCHEDULE,
        metavar="MODE:DURATION,...",
        help=f"learning phases (default: {DEFAULT_SCHEDULE})",
    )
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument(
        "--reward-rate-bar",
        action="store_true",
        help="show the instantaneous reward rate beside the sphere",
    )
    args = parser.parse_args()
    if args.schedule.total_duration <= 0.0 or args.step <= 0.0:
        parser.error("schedule duration and step must be positive")
    if args.frames < 2:
        parser.error("frames must be at least two")
    if args.fps <= 0 or args.dpi <= 0:
        parser.error("fps and dpi must be positive")

    # Starting from three orthogonal directions makes their subsequent
    # convergence especially easy to see on the sphere.
    initial = State(rho_se=0.0, rho_me=0.0, rho_sm=0.0)
    params = Params(
        T=5,
        M=3,
        K=3,
        beta=2.0,
        eta_2f=0.0,
        eta_3f=0.5,
        eta_wm=0.5,
        eta_e=0.0,
        gh_order=24,
        p_flip = 0.2,
    )
    records = with_metadata(
        run_analytical(
            initial,
            params,
            schedule=args.schedule,
            step=args.step,
        ),
        params,
        source="analytical",
        condition="example trajectory",
    )

    result_path = REPO_ROOT / "results" / "examples" / "sphere_trajectory.csv"
    figure_path = REPO_ROOT / "figures" / "examples" / "sphere_trajectory.gif"
    write_records(result_path, records)
    print(f"Wrote {result_path}")
    animate_sphere(
        records,
        figure_path,
        frame_count=args.frames,
        fps=args.fps,
        dpi=args.dpi,
        show_reward_rate=args.reward_rate_bar,
    )
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()

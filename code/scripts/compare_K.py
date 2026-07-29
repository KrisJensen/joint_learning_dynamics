"""Compare fixed and instantaneously optimized planner sample counts K."""

import argparse

from _common import (
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    destination,
    initial_state,
    make_params,
    parse_list,
    with_metadata,
    write_records,
)
from learning_dynamics import run_analytical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_analytical_args(parser)
    parser.add_argument("--K-values", default="1,4,10,50")
    parser.add_argument("--objectives", default="drift")
    args = parser.parse_args()
    rows = []
    for K in parse_list(args.K_values, int):
        print("running K", K)
        params = make_params(args, K=K, K_mode="constant")
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"K={K}",
            compared_K=K,
        )
    for objective in parse_list(args.objectives, str):
        if objective not in {"reward", "drift"}:
            parser.error("objectives must be reward and/or drift")
        params = make_params(args, K_mode=objective)
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"optimal K ({objective})",
            compared_K=f"optimal_{objective}",
        )
    path = destination(args, "K_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

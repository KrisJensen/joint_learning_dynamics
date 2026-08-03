"""Compute fixed-M, optimal-M, and eta_3f-only prefix-length trajectories."""

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
    # The report absorbs the legacy zero-field score factor of 1/2 into the
    # learning-rate definitions, so legacy rates/caps of 2.5/20 become
    # 1.25/10. The remaining defaults reproduce the legacy fixed-WM analysis.
    parser.set_defaults(
        schedule="online:25",
        rho_me0=1.0,
        rho_sm0=0.0,
        T=8,
        K=5,
        K_mode="constant",
        beta=3.0,
        temperature_mode="optimal",
        beta_max=20.0,
        eta_2f=1.25,
        eta_3f=1.25,
        eta_off=1.25,
        eta_wm=0.0,
        eta_max=10.0,
        learning_rate_mode="constant",
        fix_eta_wm_zero=True,
        eta_e=0.0,
        delta_on=0.1,
        delta_off=0.1,
        gh_order=25,
    )
    parser.add_argument("--M-values", default=None, help="defaults to 1..T")
    parser.add_argument("--objectives", default="drift")
    args = parser.parse_args()

    values = (
        list(range(1, args.T + 1))
        if args.M_values is None
        else parse_list(args.M_values, int)
    )
    initial = initial_state(
        args,
        rho_me=1.0,
        rho_sm=args.rho_se0,
    )
    rows = []
    for M in values:
        print(f"running eta_2f-only M = {M}")
        params = make_params(
            args,
            M=M,
            M_mode="constant",
            eta_3f=0.0,
            fix_eta_2f_zero=False,
            fix_eta_3f_zero=True,
        )
        rows += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"eta_2f only, M={M}",
            comparison_group="eta_2f_fixed_M",
            compared_M=M,
            nonzero_rate="eta_2f",
            nonzero_rate_mode=args.learning_rate_mode,
        )
    for objective in parse_list(args.objectives, str):
        if objective not in {"reward", "drift"}:
            parser.error("objectives must be reward and/or drift")
        print(f"running eta_2f-only optimal M ({objective})")
        params = make_params(
            args,
            M_mode=objective,
            eta_3f=0.0,
            fix_eta_2f_zero=False,
            fix_eta_3f_zero=True,
        )
        rows += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"eta_2f only, optimal M ({objective})",
            comparison_group="eta_2f_optimal_M",
            compared_M=f"optimal_{objective}",
            nonzero_rate="eta_2f",
            nonzero_rate_mode=args.learning_rate_mode,
        )
    print(f"running eta_3f-only baseline with M = {args.T}")
    params = make_params(
        args,
        M=args.T,
        M_mode="constant",
        eta_2f=0.0,
        fix_eta_2f_zero=True,
        fix_eta_3f_zero=False,
    )
    rows += with_metadata(
        run_analytical(initial, params, **analytical_kwargs(args)),
        params,
        source="analytical",
        condition=f"eta_3f only, M={args.T}",
        comparison_group="eta_3f_only",
        compared_M=args.T,
        nonzero_rate="eta_3f",
        nonzero_rate_mode=args.learning_rate_mode,
    )
    path = destination(args, "prefix_length_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

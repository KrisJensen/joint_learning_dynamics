"""Compare fixed and optimal prefix lengths for online student learning."""

import argparse

from _common import (
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    make_params,
    output_path,
    with_metadata,
    write_records,
)
from joint_dynamics import State, run_analytical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(
        schedule="online:25",
        duration=25.0,
        rho_me0=1.0,
        rho_sm0=0.0,
        eta_e=0.0,
        eta_wm=0.0,
        T=8,
        temperature_mode="optimal",
        fix_eta_wm_zero=True,
        learning_rate_mode="constant",
    )
    add_analytical_args(parser)
    parser.add_argument(
        "--prefix-length-values",
        default=None,
        help="comma-separated fixed prefix lengths; defaults to 1..T",
    )
    parser.add_argument(
        "--include-constant-prefixes",
        action="store_true",
        help="include eta_on1-only runs with different constant M values",
    )
    parser.add_argument(
        "--include-optimal-prefix",
        action="store_true",
        help="include the eta_on1-only run with optimal M",
    )
    parser.add_argument(
        "--include-eta-on2",
        action="store_true",
        help="include the eta_on2-only baseline analysis",
    )
    parser.add_argument(
        "--exclude-constant-prefixes",
        action="store_true",
        help="exclude eta_on1-only runs with different constant M values",
    )
    parser.add_argument(
        "--exclude-optimal-prefix",
        action="store_true",
        help="exclude the eta_on1-only run with optimal M",
    )
    parser.add_argument(
        "--exclude-eta-on2",
        action="store_true",
        help="exclude the eta_on2-only baseline analysis",
    )
    args = parser.parse_args()

    include_flags = (
        args.include_constant_prefixes,
        args.include_optimal_prefix,
        args.include_eta_on2,
    )
    if any(include_flags):
        run_constant_prefixes = args.include_constant_prefixes
        run_optimal_prefix = args.include_optimal_prefix
        run_eta_on2 = args.include_eta_on2
    else:
        run_constant_prefixes = True
        run_optimal_prefix = True
        run_eta_on2 = True
    run_constant_prefixes = run_constant_prefixes and not args.exclude_constant_prefixes
    run_optimal_prefix = run_optimal_prefix and not args.exclude_optimal_prefix
    run_eta_on2 = run_eta_on2 and not args.exclude_eta_on2
    if not (run_constant_prefixes or run_optimal_prefix or run_eta_on2):
        parser.error("at least one prefix-length analysis must be included")

    if args.prefix_length_values is None:
        prefix_lengths = list(range(1, args.T + 1))
    else:
        prefix_lengths = [
            int(value.strip())
            for value in args.prefix_length_values.split(",")
            if value.strip()
        ]

    initial = State(args.rho_se0, 1.0, args.rho_se0)
    rows = []
    if run_constant_prefixes:
        for prefix_length in prefix_lengths:
            print("Running eta_on1-only fixed prefix length =", prefix_length)
            params = make_params(
                args,
                prefix_length=prefix_length,
                prefix_length_mode="constant",
                learning_rate_mode=args.learning_rate_mode,
                fix_eta_on1_zero=False,
                fix_eta_on2_zero=True,
                fix_eta_wm_zero=True,
                eta_on2=0.0,
                eta_wm=0.0,
                eta_e=0.0,
                eta_on1 = 2.5,
            )
            rows += with_metadata(
                run_analytical(initial, params, **analytical_kwargs(args)),
                params,
                source="analytical",
                condition=f"eta_on1 only, prefix_length={prefix_length}",
                comparison_group="eta_on1_fixed_prefix",
                compared_prefix_length=prefix_length,
                nonzero_rate_mode=args.learning_rate_mode,
            )

    if run_optimal_prefix:
        print("Running eta_on1-only optimal prefix length")
        optimal_prefix = make_params(
            args,
            prefix_length=None,
            prefix_length_mode="optimal",
            learning_rate_mode=args.learning_rate_mode,
            fix_eta_on1_zero=False,
            fix_eta_on2_zero=True,
            fix_eta_wm_zero=True,
            eta_on2=0.0,
            eta_wm=0.0,
            eta_e=0.0,
            eta_on1 = 2.5,
        )
        rows += with_metadata(
            run_analytical(initial, optimal_prefix, **analytical_kwargs(args)),
            optimal_prefix,
            source="analytical",
            condition="eta_on1 only, optimal prefix_length",
            comparison_group="eta_on1_optimal_prefix",
            compared_prefix_length="optimal",
            nonzero_rate_mode=args.learning_rate_mode,
        )

    if run_eta_on2:
        eta_on2_prefix_length = args.T
        print("Running eta_on2-only baseline")
        eta_on2_baseline = make_params(
            args,
            prefix_length=eta_on2_prefix_length,
            prefix_length_mode="constant",
            learning_rate_mode=args.learning_rate_mode,
            fix_eta_on1_zero=True,
            fix_eta_on2_zero=False,
            fix_eta_wm_zero=True,
            eta_on1=0.0,
            eta_wm=0.0,
            eta_e=0.0,
            eta_on2 = 2.5,
        )
        rows += with_metadata(
            run_analytical(initial, eta_on2_baseline, **analytical_kwargs(args)),
            eta_on2_baseline,
            source="analytical",
            condition=f"eta_on2 only, prefix_length={eta_on2_prefix_length}",
            comparison_group="eta_on2_only",
            compared_prefix_length=eta_on2_prefix_length,
            nonzero_rate_mode=args.learning_rate_mode,
        )

    destination = args.output or output_path("prefix_length_comparison", args.suffix, args.output_subdir)
    write_records(destination, rows)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()

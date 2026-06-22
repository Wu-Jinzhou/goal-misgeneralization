#!/usr/bin/env python
from __future__ import annotations

import argparse

from gm_nim.plotting import (
    plot_accuracy_curves,
    plot_causal_trace,
    plot_logit_lens,
    plot_probe_heatmap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reproduction diagnostics.")
    sub = parser.add_subparsers(required=True)

    curves = sub.add_parser("curves")
    curves.add_argument("--metrics-file", required=True)
    curves.add_argument("--output-file", required=True)
    curves.add_argument("--x", default="step")
    curves.add_argument("--y", default="exact")
    curves.add_argument("--hue", default="condition")
    curves.add_argument("--chance-column")
    curves.set_defaults(
        func=lambda a: plot_accuracy_curves(
            a.metrics_file,
            a.output_file,
            x=a.x,
            y=a.y,
            hue=a.hue,
            chance_column=a.chance_column,
        )
    )

    probe = sub.add_parser("probe")
    probe.add_argument("--csv-file", required=True)
    probe.add_argument("--output-file", required=True)
    probe.set_defaults(func=lambda a: plot_probe_heatmap(a.csv_file, a.output_file))

    trace = sub.add_parser("trace")
    trace.add_argument("--csv-file", required=True)
    trace.add_argument("--output-file", required=True)
    trace.set_defaults(func=lambda a: plot_causal_trace(a.csv_file, a.output_file))

    lens = sub.add_parser("lens")
    lens.add_argument("--csv-file", required=True)
    lens.add_argument("--output-file", required=True)
    lens.set_defaults(func=lambda a: plot_logit_lens(a.csv_file, a.output_file))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


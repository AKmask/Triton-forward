"""Run the 3-D VTI CUDA/Triton solver and export records and snapshots.

Each selected double-couple source runs with a newly initialized solver.
"""

from __future__ import annotations

import argparse

from vti10.runtime_paths import configure_runtime_cache

# Configure caches before third-party libraries initialize their runtimes.
CACHE_ROOT = configure_runtime_cache()

from vti10.fdwave3d_io import load_fdwave3d_case
from vti10.output import save_result
from vti10.solver import RegionFusedSplitPMLSolver
from vti10.wavefield_plot import PLOT_COMPONENTS, save_wavefield_figures


def main():
    """Load the case and run each requested source with fresh solver state."""
    parser = argparse.ArgumentParser(description="Version 10: five-coefficient VTI input with version 09 region-fused split-PML modelling")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--source", default="dc1", choices=["dc1", "dc2", "both"])
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--dtype", choices=["fp32", "fp64"], default="fp64")
    parser.add_argument("--record-stride", type=int, default=1)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument(
        "--snapshot-step",
        type=int,
        default=-1,
        help="Time step for three-plane snapshots; -1 selects the last active time step",
    )
    parser.add_argument("--output-prefix", default="vti", help="Output filename prefix")
    parser.add_argument("--output-dir", default="outputs_10_region_fused_splitpml_vti")
    parser.add_argument(
        "--plot-component",
        choices=PLOT_COMPONENTS,
        default="vx",
        help="Field shown in automatic wavefield plots; magnitude=sqrt(vx^2+vy^2+vz^2)",
    )
    parser.add_argument("--plot-cmap", default="seismic", help="Matplotlib colormap name")
    parser.add_argument("--plot-dpi", type=int, default=300)
    parser.add_argument("--plot-alpha", type=float, default=0.95, help="Peak opacity of the 3-D wavefield slices, in (0,1]")
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=0,
        help="Maximum grid points per axis in the 3-D plot; 0 keeps the full grid (default)",
    )
    parser.add_argument("--plot-elevation", type=float, default=20.0, help="3-D viewing elevation in degrees")
    parser.add_argument("--plot-azimuth", type=float, default=-60.0, help="3-D viewing azimuth in degrees")
    parser.add_argument("--plot-colorbar", action="store_true", help="Add a colorbar to the clean 3-D slice view")
    parser.add_argument("--no-wavefield-plots", action="store_true", help="Disable automatic wavefield plots after modelling")
    parser.add_argument("--sync-debug", action="store_true")
    args = parser.parse_args()

    case = load_fdwave3d_case(args.case_dir)
    kinds = ["dc1", "dc2"] if args.source == "both" else [args.source]
    print("material_model =", case.material_model)
    print("VTI validation normalized_margin_min =", case.vti_validation["normalized_stability_margin_min"])
    for kind in kinds:
        solver = RegionFusedSplitPMLSolver(
            case,
            args.order,
            args.dtype,
            block=args.block,
            sync_debug=args.sync_debug,
        )
        print("region_stats =", solver.stats)
        result = solver.run(kind, args.record_stride, args.snapshot_step)
        stem = f"{args.output_prefix}_{kind}_10_region_fused_splitpml_vti"
        result_file = save_result(
            result,
            case,
            args.output_dir,
            stem,
            args.order,
            "10-region-fused-splitpml-vti",
            args.dtype,
        )
        print(f"{kind}: runtime={result['runtime_s']:.6f}s")
        print(f"saved: {result_file}")
        if not args.no_wavefield_plots:
            figure_paths = save_wavefield_figures(
                result,
                case,
                args.output_dir,
                stem,
                component=args.plot_component,
                cmap_name=args.plot_cmap,
                dpi=args.plot_dpi,
                alpha=args.plot_alpha,
                maximum_points=args.plot_max_points,
                elevation=args.plot_elevation,
                azimuth=args.plot_azimuth,
                show_colorbar=args.plot_colorbar,
            )
            for name, path in figure_paths.items():
                print(f"saved wavefield {name.upper()}: {path}")


if __name__ == "__main__":
    main()

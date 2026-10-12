"""Run 2-D Hess VTI forward modelling and save records, metadata, and plots.

The CPU backend uses NumPy; the GPU backend is imported only when selected.
"""

from __future__ import annotations

import argparse

from vti2d.runtime_paths import configure_runtime_cache

# Configure caches before third-party libraries initialize their runtimes.
CACHE_ROOT = configure_runtime_cache()

from vti2d.case_io import load_vti2d_case
from vti2d.cpu_solver import CPUReferenceSolver
from vti2d.output import save_result
from vti2d.plotting import save_receiver_figure, save_wavefield_figure
from vti2d.run_info import format_run_completion, format_run_configuration
from vti2d.separate_plots import save_separate_component_figures
from vti2d.source import SOURCE_KINDS


def main() -> None:
    """Load a case, run the selected backend, and save results and optional figures."""
    parser = argparse.ArgumentParser(description="Accelerated 2-D Hess VTI qP-qSV forward modelling")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--source", choices=["case", *SOURCE_KINDS], default="case")
    parser.add_argument("--order", type=int, choices=[2, 4, 6, 8, 10], default=10)
    parser.add_argument("--dtype", choices=["fp32", "fp64"], default="fp64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--block", type=int, choices=[64, 128, 256, 512, 1024], default=256)
    parser.add_argument("--record-stride", type=int, default=1)
    parser.add_argument("--snapshot-step", type=int, default=-1)
    parser.add_argument("--output-dir", default="outputs_12_2d_hess_vti")
    parser.add_argument("--output-prefix", default="hess_vti2d")
    parser.add_argument("--plot-cmap", default="seismic")
    parser.add_argument("--plot-dpi", type=int, default=180)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--sync-debug", action="store_true")
    args = parser.parse_args()

    if args.record_stride < 1:
        parser.error("--record-stride must be positive")
    if args.plot_dpi < 1:
        parser.error("--plot-dpi must be positive")

    case = load_vti2d_case(args.case_dir, order=args.order)
    source_kind = case.default_source_kind if args.source == "case" else args.source
    # Keep CPU-only runs independent of the CUDA/Triton imports.
    if args.backend == "gpu":
        from vti2d.gpu_solver import TritonRegionFusedSolver

        solver = TritonRegionFusedSolver(
            case, args.order, args.dtype, device=args.device, block=args.block, sync_debug=args.sync_debug
        )
    else:
        solver = CPUReferenceSolver(case, args.order, args.dtype)
    print(format_run_configuration(case, args, source_kind, solver.stats), flush=True)
    result = solver.run(source_kind, args.record_stride, args.snapshot_step)
    print(format_run_completion(result, case), flush=True)
    stem = f"{args.output_prefix}_{source_kind}_{args.backend}_vti2d"
    paths = save_result(
        result, case, args.output_dir, stem, args.order, args.backend,
        args.dtype, source_kind, args.record_stride,
    )
    for name, path in paths.items():
        print(f"saved {name}: {path}")
    if not args.no_plots:
        wavefield_path = save_wavefield_figure(
            result, case, args.output_dir, stem, cmap=args.plot_cmap, dpi=args.plot_dpi
        )
        receiver_path = save_receiver_figure(
            result, case, args.output_dir, stem, cmap=args.plot_cmap, dpi=args.plot_dpi
        )
        print("saved wavefield figure:", wavefield_path)
        print("saved receiver figure:", receiver_path)
        separate_paths = save_separate_component_figures(
            result, case, args.output_dir, stem, cmap=args.plot_cmap, dpi=args.plot_dpi
        )
        for name, path in separate_paths.items():
            print(f"saved separate {name}: {path}")


if __name__ == "__main__":
    main()

# Triton 3-D VTI Forward Modelling

A CUDA/Triton staggered-grid velocity-stress solver for 3-D elastic propagation
in vertically transversely isotropic (VTI) media. It reads density and five
independent stiffness fields (`C11`, `C13`, `C33`, `C44`, `C66`) and uses separate
interior updates and fused split-PML kernels. Two predefined double-couple sources
are available as `dc1` and `dc2`.

## Requirements

Use a Python environment with the packages in `requirements.txt`, a CUDA GPU,
and compatible CUDA-enabled PyTorch and Triton installations. This package has
no CPU propagation backend. Case generation and model plotting use NumPy,
SciPy, and Matplotlib.

```bash
python -m pip install -r requirements.txt
```

## Quick start

Edit `layered_vti_config.json` to set the grid, layer properties, source,
receivers, PML, and plotting options. The supplied model has three layers on a
256 × 256 × 256 grid. Run from this directory:

```bash
python build_layered_vti_model.py --config layered_vti_config.json --output-dir generated_cases/layered_demo
python main.py --case-dir generated_cases/layered_demo --source dc1 --order 6 --dtype fp32 --output-dir outputs_layered
```

Use `--source both` to run the two sources separately. The builder also writes
`run_forward.ps1`; update `forward.python_executable` for your environment before
generating this PowerShell launcher. The JSON `forward` section supplies launcher
options; direct `main.py` calls use their command-line options and defaults.

Rebuild after changing the JSON. Add `--overwrite` to the builder command when
intentionally replacing generated files. English `__comment_*` fields provide
configuration guidance and do not change the numerical parameters.

## Conventions and output

- JSON geometry uses zero-based `[x, y, z]` indices. Layer ranges are half-open
  `[z_start, z_end)` and must cover the full depth without gaps or overlaps.
  Spacing is in metres, time in seconds, density in kg/m³, and stiffness in Pa.
- Generated MAT arrays use `[z, x, y]`; the loader converts them to internal
  `[z, y, x]` arrays and converts MATLAB geometry to zero-based indices.
  Input cases must use MATLAB v7 MAT files rather than v7.3.
- Solver orders are 4, 6, 8, and 10. Receiver sampling currently requires
  `--record-stride 1`. Use a stable time step: the builder's Courant indicator
  is a diagnostic, not an enforced anisotropic CFL limit.
- `--snapshot-step -1` selects step `N-3`. Snapshots are XY/XZ/YZ planes through
  the source; use `--plot-component` to select a component or velocity magnitude,
  or `--no-wavefield-plots` to disable forward figures.
- Runs write `.mat`, `.npz`, and `_meta.json` files with `Sx`/`Sy`/`Sz` receiver
  records, near-source traces, plane snapshots, and run metadata. PNG outputs
  include the three slices and a combined 3-D view.

`build_layered_vti_model.py` prepares cases; `main.py` runs them; `vti10/` contains
the loader, solver, kernels, output, and plotting utilities. `vti10.compare.compare`
compares receiver records against FDwave3D results; its optional MATLAB v7.3
result reader uses `h5py`.

## Runtime caches

The entry points configure process-local caches in `../.cache/`, under the
shared upload directory. This includes Matplotlib, Triton, CUDA, Python bytecode,
TorchInductor, and temporary files; no system-wide environment is changed.

## Orthogonal wavefield view

The saved `*_wavefield_3d.png` retains three intersecting source-plane slices,
orthographic projection, physical distance axes, and downward depth. Its styling
follows : a white background, a shared symmetric
color scale, thin gray axes, and nearest-neighbour cells centred on grid nodes.
The individual XY/XZ/YZ figures use the same scale and pixel-centre convention.
Each slice starts with a complete four-edge rectangular perimeter. Outline
segments hidden behind another slice are clipped according to the camera view,
including reversed depth and unequal axis lengths. Only visible segments are
drawn in faint gray; slice-intersection lines and a surrounding box are omitted.
The three coordinate axes remain separate and slightly darker. Outline occlusion
uses the finite slice geometry, independently of wavefield transparency.

The default field remains signed `vx` with the seismic red/white/blue colormap.
Use `--plot-component vy`, `vz`, or `magnitude` to select another field, and
`--plot-cmap gray` or `RdBu_r` for other colormaps used by the comparison script.
Magnitude uses the positive half of the symmetric scale. Every original grid
sample is rendered by default, without averaging neighbouring amplitudes.
An optional `--plot-max-points 161` reduces rendering cost for large grids;
`--plot-max-points 0` retains the full grid.

The default view is `--plot-elevation 20 --plot-azimuth -60` at 300 dpi.
`--plot-alpha` controls peak opacity. In the 3-D view only, weak amplitudes fade
to transparency to keep intersecting planes visible; this affects appearance,
not saved data. The combined view omits the title, source marker, grid, and
colorbar; add `--plot-colorbar` for a shared scale in scientific notation.
The three individual figures and numerical MAT/NPZ outputs are also saved.
Plot styling does not change the simulation parameters or stored wavefields.

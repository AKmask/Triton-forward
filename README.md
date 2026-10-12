# Triton 2-D and 3-D VTI Forward Modelling

Staggered-grid velocity-stress solvers for vertically transversely isotropic
(VTI) media. Both GPU implementations use one-dimensional Triton blocks,
separate interior/PML regions, and fused split-PML updates.

| Package | Propagation and backend | Sources | Spatial orders |
| --- | --- | --- | --- |
| `Triton-2D forward` | x-z qP-qSV; CUDA/Triton or NumPy CPU reference | Explosive or x-z double-couple | 4, 6, 8, 10 |
| `Triton-3D forward` | 3-D elastic; CUDA/Triton only | `dc1`, `dc2`, or `both` | 4, 6, 8, 10 |

## Setup

GPU runs require a CUDA GPU and compatible CUDA-enabled PyTorch and Triton.
From the selected package directory, install its dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Start each example from the directory containing this README.

**2-D:** Hess MAT inputs are not included. Edit `hess_vti2d_config.json` to set
`hess_model.directory`, file/variable mappings, and input spacing. Stiffness
input requires density, `C11`, `C13`, `C33`, and `C44`; optional `C66` defaults
to `C44`. Thomsen parameter input is also supported.

```bash
cd "Triton-2D forward"
python build_hess_vti2d_case.py --config hess_vti2d_config.json --output-dir generated_cases/hess_demo
python main.py --case-dir generated_cases/hess_demo --backend gpu --source case --order 6 --dtype fp32 --output-dir outputs_hess
```

Use `--backend cpu` for the 2-D reference. Only `topABC` split-PML is supported.
Move the supplied receiver `[30, 1]` inside the active stencil region before
recording: its depth index is inactive at order 6.

**3-D:** Edit `layered_vti_config.json` for grid, layers, source, receivers, and
PML. The supplied case has three layers on a 256 x 256 x 256 grid and uses
density plus `C11`, `C13`, `C33`, `C44`, and `C66`.

```bash
cd "Triton-3D forward"
python build_layered_vti_model.py --config layered_vti_config.json --output-dir generated_cases/layered_demo
python main.py --case-dir generated_cases/layered_demo --source dc1 --order 6 --dtype fp32 --output-dir outputs_layered
```

## Configuration and conventions

- Rebuild after editing the JSON; use builder option `--overwrite` to replace
  existing cases. English `__comment_*` fields are explanatory only.
- Set `forward.python_executable` before generating PowerShell launchers.
  The JSON `forward` section configures those launchers; direct `main.py` runs
  use CLI options and defaults. Run `python main.py --help` for all options.
- Use MATLAB v7 input files. JSON geometry uses zero-based `[x, z]` or
  `[x, y, z]` indices. 2-D arrays use `[z, x]`; 3-D MAT arrays use `[z, x, y]`
  and are converted internally to `[z, y, x]`.
- Units are metres, seconds, kg/m^3, and Pa. Choose a stable time step: 2-D
  checks a conservative CFL limit; the 3-D Courant indicator is diagnostic only.
- Keep sources and receivers inside the active stencil region. Receiver stride
  is configurable in 2-D; 3-D currently requires `--record-stride 1`.

## Results and figures

Runs save `.mat`, `.npz`, `_meta.json`, and PNG figures, including receiver
records (`Sx`/`Sz` in 2-D; `Sx`/`Sy`/`Sz` in 3-D), snapshots, and metadata.
`--snapshot-step -1` selects the last active step, `N-3`.

3-D figures include XY/XZ/YZ source-plane slices and a combined orthogonal view.
They use a shared symmetric color scale; the combined view clips hidden rectangle
edges and omits intersection lines and a surrounding box. The default field is
`vx`; select `vy`, `vz`, or `magnitude` with `--plot-component`, or add
`--plot-colorbar`. Full-grid rendering is the default; `--plot-max-points 161`
reduces rendering cost. Disable figures with `--no-plots` (2-D) or
`--no-wavefield-plots` (3-D).
![The 3D wavefield map under the current configuration](https://github.com/AKmask/Triton-forward/blob/main/Triton-3D%20forward/outputs/vti_dc1_10_region_fused_splitpml_vti_wavefield_3d.png)  

## Caches

Entry points keep runtime caches and temporary files in the shared `.cache/`
directory beside both packages, without changing system-wide settings.
Some Windows Triton builds cannot decode non-ASCII cache paths; use ASCII-only
runtime paths in affected environments.

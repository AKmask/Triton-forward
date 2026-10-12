# Triton 2-D VTI Forward Modelling

A staggered-grid velocity-stress solver for 2-D x-z qP-qSV propagation in
Hess VTI models. It provides a NumPy CPU reference and a CUDA/Triton backend
with separate interior and fused split-PML updates. Explosive and x-z
double-couple sources are supported, including models with an acoustic water layer.

## Requirements

Use a Python environment with the packages in `requirements.txt`. GPU runs
require a CUDA GPU and compatible CUDA-enabled PyTorch and Triton installations.
The CPU backend uses NumPy, SciPy, and Matplotlib without importing the GPU backend.

```bash
python -m pip install -r requirements.txt
```

Hess input MAT files are **not included**. In `hess_vti2d_config.json`, update
`hess_model.directory`, the filename/variable mappings, and the input grid spacing.
The default stiffness mode reads density and `C11`, `C13`, `C33`, `C44`; optional
`C66` defaults to `C44`. The loader also supports a Thomsen parameterization.
Use MATLAB v7 MAT inputs rather than v7.3 files.

## Quick start

Run these commands from this directory after configuring the Hess inputs:

```bash
python build_hess_vti2d_case.py --config hess_vti2d_config.json --output-dir generated_cases/hess_demo
python main.py --case-dir generated_cases/hess_demo --backend gpu --source case --order 6 --dtype fp32 --output-dir outputs_hess
```

For the CPU reference, replace `--backend gpu` with `--backend cpu`. The builder
also writes `run_gpu.ps1` and `run_cpu.ps1`; set `forward.python_executable` to
your interpreter before generating these PowerShell launchers.

Rebuild the case after editing the JSON. Add `--overwrite` to the builder command
when intentionally replacing generated case files. The JSON `forward` section
configures the generated launchers; direct `main.py` calls use their command-line
options and defaults.

## Conventions and output

- Material arrays use `[z, x]`; JSON geometry uses zero-based `[x, z]` grid indices.
  Spacing is in metres, time in seconds, density in kg/m³, and stiffness in Pa.
- Spatial orders are 2, 4, 6, 8, and 10. Only `topABC` split-PML is supported.
  The builder and loader check a conservative CFL time-step limit.
- Keep sources and receivers inside the active stencil region. The supplied
  receiver `[30, 1]` is in the inactive halo at order 6 and will not record
  propagated motion there; adjust its position for your experiment.
- `--record-stride` controls receiver sampling. `--snapshot-step -1` selects the
  last active step, `N-3`. Use `--no-plots` to disable forward figures.
- Each run writes `.mat`, `.npz`, and `_meta.json` files containing `Sx`/`Sz`
  records, snapshots, geometry, and run information, plus PNG figures by default.

`build_hess_vti2d_case.py` prepares cases; `main.py` runs them; `vti2d/` contains
loaders, solvers, plotting utilities, and Triton kernels. English `__comment_*`
fields document the JSON settings and do not change the numerical parameters.

## Runtime caches

The entry points configure process-local caches in `../.cache/`, under the
shared upload directory. This includes Matplotlib, Triton, CUDA, Python bytecode,
TorchInductor, and temporary files; no system-wide environment is changed.

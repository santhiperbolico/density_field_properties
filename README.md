# density_field_properties

Tools for computing and analyzing properties derived from the **dark matter density field** in cosmological simulations.

This repository provides methods and scripts for:
- Building the **density field** from dark matter particles using grid assignment schemes (CIC, TSC, etc.).
- Computing the **tidal tensor** and its eigenvalues/eigenvectors.
- Deriving related quantities such as **tidal anisotropy** and **linear bias**.
- Performing auxiliary analyses on density fields, including smoothing, Fourier transforms, and memory-efficient batching.

---

## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/<your-username>/density_field_properties.git
cd density_field_properties
conda env create -f environment.yml
conda activate density_field
````

Alternatively, install it as a local editable package:

```bash
pip install -e .
```
---
## Pre-commit installation

To ensure consistent code style and quality, this repository uses [**pre-commit**](https://pre-commit.com/) hooks configured with:

* **[Black](https://black.readthedocs.io/en/stable/)** — automatic code formatter
* **[isort](https://pycqa.github.io/isort/)** — import sorting
* **[flake8](https://flake8.pycqa.org/en/latest/)** — code linting

Install `pre-commit`

```bash
pip install pre-commit
```

Set up hooks in your local repository

```bash
pre-commit install
```

Run checks manually (optional)

```bash
pre-commit run --all-files
```

The hooks will automatically run on every `git commit`, ensuring that code is formatted, imports are sorted, and style issues are caught early.

---

## Main dependencies

* `numpy`
* `scipy`
* `h5py`
* `matplotlib`
* `tqdm`
* `numba` *(optional, for JIT acceleration)*

---

## Scientific context

This codebase is designed for studies involving:

* **Assembly bias** and extended HOD models (e.g. [`hod_madrid_py`](https://github.com/computationalAstroUAM/hod_madrid_py)).
* **Tidal tensor** and **anisotropy** in large-scale structure ([Haloscope project](https://arxiv.org/pdf/2410.07361)).
* Applications to **UNIT** ([UNIT simulations](https://ui.adsabs.harvard.edu/abs/2024A%26A...689A..69G/abstract)), **FastPM** ([FastPM GitHub](https://github.com/fastpm/fastpm)), and **CAMELS** ([CAMELS project](https://camels.readthedocs.io/en/latest/)).
* Integration within cosmological survey frameworks such as **Euclid** ([ESA Euclid mission](https://www.euclid-ec.org)) and **DESI** ([DESI collaboration](https://www.desi.lbl.gov/)).

---

## FastPM halo catalogs (Rockstar `.list`)

Rockstar text catalogs produced after FastPM runs (for example `out_0.list` under
`rockstar_out_pm` or `rockstar_out_nbody`) are read with the same
`RockstarCatalogReader` as standalone Rockstar outputs. Do not use
`FastPMCatalogReader` for those `.list` files; that reader is only for native
halos stored in FastPM BigFile snapshots.

**Design:** reuse `RockstarCatalogReader` only—no duplicate parser in
`fastpm.py` and no separate `fastpm_rockstar` catalog alias unless a future
requirement justifies it.

**CLI example:**

```bash
python scripts/your_pipeline.py \
  --halo_catalog_name rockstar \
  --halo_file /path/to/fastpm_tfm/rockstar_out_pm/out_0.list
```

Typical folder layout on shared storage is documented in
[`config/fastpm_folders.md`](config/fastpm_folders.md) (`rockstar_out_pm` vs
`rockstar_out_nbody`).
